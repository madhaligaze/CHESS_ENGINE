//! Human-likeness benchmark: replays held-out games of a player and measures how often the
//! engine plays the move the human actually played. Games are numbered exactly as in
//! `datagen`, so `--holdout 8` selects the same games the training scripts kept out.

use crate::board::Board;
use crate::datagen::name_matches;
use crate::engine::{think, Game, Nets, Options};
use crate::pgn::PgnReader;
use crate::san::parse_san;
use crate::search::{Limits, SearchPool};
use std::fs::File;
use std::io::BufReader;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

pub struct HumanEvalConfig {
    pub input: String,
    pub player: Option<String>,
    pub holdout: u32,
    pub sample: f64,
    pub max_positions: usize,
    pub min_ply: usize,
    pub limits: Limits,
    pub opts: Options,
    pub seed: u64,
}

#[derive(Default)]
struct Stats {
    n: usize,
    net_top1: usize,
    net_top3: usize,
    engine: usize,
    chosen: usize,
    cp_loss_sum: i64,
    cp_loss_n: usize,
    chosen_differs_from_engine: usize,
}

fn next_f64(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

pub fn run(cfg: &HumanEvalConfig) -> std::io::Result<()> {
    let nets = Nets::load(cfg.opts.net_dir.as_deref());
    let net = nets.for_style(cfg.opts.style);
    println!(
        "style {:?} (eval {:?}), network {}, humanity {}, variety {}, max loss {}",
        cfg.opts.style,
        cfg.opts.eval_style.unwrap_or(cfg.opts.style),
        net.as_ref().map_or("none", |n| n.name.as_str()),
        cfg.opts.humanity,
        cfg.opts.variety,
        cfg.opts.max_loss
    );
    let mut searcher = SearchPool::new(cfg.opts.hash_mb, 1, Arc::new(AtomicBool::new(false)));
    let reader = PgnReader::new(BufReader::new(File::open(&cfg.input)?));
    let mut rng = cfg.seed.max(1);
    let mut st = Stats::default();
    let t0 = Instant::now();
    let mut game_id = 0u32;
    'games: for g in reader {
        game_id += 1;
        if cfg.holdout > 1 && game_id % cfg.holdout != 0 {
            continue;
        }
        if !matches!(g.result.as_str(), "1-0" | "0-1" | "1/2-1/2") || g.tag("FEN").is_some() {
            continue;
        }
        let colors = match &cfg.player {
            Some(p) => [name_matches(g.tag("White").unwrap_or(""), p), name_matches(g.tag("Black").unwrap_or(""), p)],
            None => [true, true],
        };
        if !colors[0] && !colors[1] {
            continue;
        }
        let mut game = Game::new(Board::startpos());
        searcher.clear();
        for (ply, san) in g.moves.iter().enumerate() {
            let Some(m) = parse_san(&game.board, san) else { continue 'games };
            if colors[game.board.stm] && ply >= cfg.min_ply && next_f64(&mut rng) < cfg.sample {
                let th = think(&mut searcher, &game, &cfg.limits, &cfg.opts, net.as_deref(), rng, &[]);
                let (chosen, result) = (th.mv, th.result);
                st.n += 1;
                let best = result.root_moves.first().map_or(result.best, |r| r.mv);
                st.engine += (best == m) as usize;
                st.chosen += (chosen == m) as usize;
                if chosen != best {
                    st.chosen_differs_from_engine += 1;
                }
                if let (Some(b), Some(c)) =
                    (result.root_moves.first(), result.root_moves.iter().find(|r| r.mv == chosen && r.exact))
                {
                    st.cp_loss_sum += (b.score - c.score).clamp(0, 1000) as i64;
                    st.cp_loss_n += 1;
                }
                if let Some(n) = &net {
                    let (last, prev) = game.last_moves();
                    let e = n.evaluate(&game.board, last, prev);
                    st.net_top1 += (e.policy.first().map(|x| x.0) == Some(m)) as usize;
                    st.net_top3 += e.policy.iter().take(3).any(|x| x.0 == m) as usize;
                }
                if st.n % 100 == 0 {
                    eprintln!(
                        "  {} positions, chosen {:.1}%, engine {:.1}%, net {:.1}% ({:.0}s)",
                        st.n,
                        pct(st.chosen, st.n),
                        pct(st.engine, st.n),
                        pct(st.net_top1, st.n),
                        t0.elapsed().as_secs_f64()
                    );
                }
                if st.n >= cfg.max_positions {
                    game.play(m);
                    break 'games;
                }
            }
            game.play(m);
        }
    }
    let n = st.n.max(1);
    println!("positions: {}", st.n);
    println!("network top-1 matches human: {:.1}%  (top-3: {:.1}%)", pct(st.net_top1, n), pct(st.net_top3, n));
    println!("pure engine best matches human: {:.1}%", pct(st.engine, n));
    println!("engine in this mode matches human: {:.1}%", pct(st.chosen, n));
    println!(
        "chosen differs from engine best in {:.1}% of moves; average cost {:.1} cp",
        pct(st.chosen_differs_from_engine, n),
        st.cp_loss_sum as f64 / st.cp_loss_n.max(1) as f64
    );
    Ok(())
}

fn pct(a: usize, n: usize) -> f64 {
    100.0 * a as f64 / n.max(1) as f64
}
