//! Turns a search into a move the way a strong human would choose it.
//!
//! 1. The network predicts which moves a strong human (or Tal, or Carlsen) would consider.
//! 2. The alpha-beta search scores those candidates exactly, alongside its own best move.
//! 3. The move is chosen by KL-regularised selection (the "piKL" rule from human-AI
//!    research): maximise  expected_score(eval) + beta * ln(human probability).
//!    Among moves that are about equally good the human one wins; a clearly better engine
//!    move still wins; anything losing more than `max_loss` centipawns is never played.
//! 4. Thinking time follows the position like a human's does: obvious moves (a forced
//!    recapture, a move every strong player would play) are made quickly, complex
//!    positions get more time.

use crate::board::Board;
use crate::eval::Style;
use crate::nn::{NetEval, Network};
use crate::search::{uci_score, Limits, RootMove, SearchParams, SearchPool, SearchResult, MATE_BOUND};
use crate::types::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StyleKind {
    /// Strong human player in general (base network).
    Human,
    Tal,
    Carlsen,
    /// Pure engine: no network, maximum strength, no human-like choices.
    Engine,
}

impl StyleKind {
    pub fn parse(s: &str) -> Option<StyleKind> {
        match s.to_ascii_lowercase().as_str() {
            "human" => Some(StyleKind::Human),
            "tal" => Some(StyleKind::Tal),
            "carlsen" => Some(StyleKind::Carlsen),
            "engine" => Some(StyleKind::Engine),
            _ => None,
        }
    }

    pub fn eval_style(self) -> Style {
        match self {
            StyleKind::Human => Style::HUMAN,
            StyleKind::Tal => Style::TAL,
            StyleKind::Carlsen => Style::CARLSEN,
            StyleKind::Engine => Style::NEUTRAL,
        }
    }

    pub fn net_file(self) -> &'static str {
        match self {
            StyleKind::Tal => "tal.nn",
            StyleKind::Carlsen => "carlsen.nn",
            _ => "base.nn",
        }
    }

    /// (humanity, max_loss) multipliers in percent: Tal trusts his intuition more and
    /// accepts objectively dubious moves (sacrifices); Carlsen trusts it but stays precise.
    fn choice_scale(self) -> (u32, i32) {
        match self {
            StyleKind::Tal => (130, 150),
            StyleKind::Carlsen => (110, 100),
            _ => (100, 100),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub hash_mb: usize,
    pub threads: usize,
    pub style: StyleKind,
    /// 0..100: how strongly the human prior pulls the choice away from the engine's best.
    pub humanity: u32,
    /// 0..100: randomness among the acceptable moves (0 = deterministic).
    pub variety: u32,
    /// Never play a move whose score is more than this many centipawns below the best.
    pub max_loss: i32,
    pub human_timing: bool,
    pub move_overhead: u64,
    pub net_dir: Option<PathBuf>,
    /// No UCI info output (batch evaluation).
    pub quiet: bool,
    /// Evaluation personality to use instead of the style's own (experiments).
    pub eval_style: Option<StyleKind>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            hash_mb: 64,
            threads: 1,
            style: StyleKind::Human,
            humanity: 50,
            variety: 20,
            max_loss: 60,
            human_timing: true,
            move_overhead: 30,
            net_dir: None,
            quiet: false,
            eval_style: None,
        }
    }
}

/// The game so far: current position, the hashes of every position (for repetitions) and
/// the moves played (the network looks at the last two).
#[derive(Clone)]
pub struct Game {
    pub board: Board,
    pub hashes: Vec<u64>,
    pub moves: Vec<Move>,
}

impl Game {
    pub fn new(board: Board) -> Game {
        Game { hashes: vec![board.hash], board, moves: Vec::new() }
    }

    pub fn play(&mut self, m: Move) {
        self.board = self.board.make_move(m);
        self.hashes.push(self.board.hash);
        self.moves.push(m);
    }

    /// (opponent's last move, our previous move)
    pub fn last_moves(&self) -> (Move, Move) {
        let n = self.moves.len();
        let last = if n >= 1 { self.moves[n - 1] } else { Move::NONE };
        let prev = if n >= 2 { self.moves[n - 2] } else { Move::NONE };
        (last, prev)
    }
}

/// Networks found on disk, by file name.
#[derive(Default, Clone)]
pub struct Nets {
    pub dir: Option<PathBuf>,
    pub base: Option<Arc<Network>>,
    pub tal: Option<Arc<Network>>,
    pub carlsen: Option<Arc<Network>>,
}

impl Nets {
    /// Looks for nets/ next to the executable, two levels up (cargo target dir) and in the
    /// working directory, unless a directory is given.
    pub fn load(dir: Option<&Path>) -> Nets {
        let candidates: Vec<PathBuf> = match dir {
            Some(d) => vec![d.to_path_buf()],
            None => {
                let mut v = Vec::new();
                if let Ok(exe) = std::env::current_exe() {
                    if let Some(d) = exe.parent() {
                        v.push(d.join("nets"));
                        v.push(d.join("../../nets"));
                    }
                }
                v.push(PathBuf::from("nets"));
                v
            }
        };
        let Some(dir) = candidates.into_iter().find(|d| d.join("base.nn").exists()) else {
            return Nets::default();
        };
        let load = |name: &str| match Network::load(&dir.join(name)) {
            Ok(n) => Some(Arc::new(n)),
            Err(e) => {
                if dir.join(name).exists() {
                    println!("info string cannot load {name}: {e}");
                }
                None
            }
        };
        Nets { base: load("base.nn"), tal: load("tal.nn"), carlsen: load("carlsen.nn"), dir: Some(dir) }
    }

    pub fn for_style(&self, style: StyleKind) -> Option<Arc<Network>> {
        match style {
            StyleKind::Engine => None,
            StyleKind::Tal => self.tal.clone().or_else(|| self.base.clone()),
            StyleKind::Carlsen => self.carlsen.clone().or_else(|| self.base.clone()),
            StyleKind::Human => self.base.clone(),
        }
    }
}

/// Soft (stop starting iterations) and hard (abort) time limits in milliseconds.
pub fn time_budget(limits: &Limits, stm: usize, overhead: u64) -> (Option<u64>, Option<u64>) {
    if let Some(mt) = limits.movetime {
        let t = mt.saturating_sub(overhead).max(1);
        return (Some(t), Some(t));
    }
    if limits.infinite {
        return (None, None);
    }
    if let Some(time) = limits.time[stm] {
        let inc = limits.inc[stm].unwrap_or(0);
        let mtg = limits.movestogo.map_or(28, |m| m.clamp(1, 50)) as u64;
        let avail = time.saturating_sub(overhead).max(1);
        let soft = (avail / mtg + inc * 3 / 4).min(avail * 2 / 5).max(1);
        let hard = (soft * 3).min(avail * 3 / 5).max(soft);
        return (Some(soft), Some(hard));
    }
    (None, None)
}

/// How long a human would think here, relative to an average move.
fn complexity_factor(net: &NetEval, game: &Game) -> f64 {
    let Some(&(top, p)) = net.policy.first() else { return 1.0 };
    let mut f: f64 = match p {
        p if p >= 0.85 => 0.35,
        p if p >= 0.65 => 0.6,
        p if p >= 0.40 => 0.9,
        p if p >= 0.25 => 1.15,
        _ => 1.4,
    };
    // An expected recapture is played almost at once.
    let (last, _) = game.last_moves();
    if last.is_some() && last.is_capture() && top.is_capture() && top.to() == last.to() && p >= 0.5 {
        f = f.min(0.4);
    }
    // Familiar opening moves come from memory.
    if game.moves.len() < 16 && p >= 0.3 {
        f *= 0.5;
    }
    f
}

/// Human probabilities -> the set of moves worth scoring exactly.
pub fn candidate_moves(net: &NetEval) -> Vec<Move> {
    let mut out = Vec::new();
    let mut cum = 0.0;
    for &(m, p) in &net.policy {
        if out.len() >= 6 || (cum >= 0.92 && !out.is_empty()) || (p < 0.02 && !out.is_empty()) {
            break;
        }
        out.push(m);
        cum += p;
    }
    out
}

#[inline]
fn expected_score(cp: i32) -> f64 {
    1.0 / (1.0 + (-(cp as f64) / 170.0).exp())
}

fn next_rand(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

pub struct Decision {
    pub mv: Move,
    pub explanation: String,
}

/// KL-regularised choice among the root moves of a finished search.
pub fn choose(result: &SearchResult, opts: &Options, rng: &mut u64) -> Decision {
    let Some(&best) = result.root_moves.first() else {
        return Decision { mv: result.best, explanation: String::new() };
    };
    if result.depth == 0 || best.score.abs() >= MATE_BOUND || opts.humanity == 0 {
        return Decision { mv: best.mv, explanation: "engine move (forced, mate or humanity 0)".into() };
    }
    let beta = opts.humanity as f64 / 100.0 * 0.05;
    let floor = best.score - opts.max_loss;
    let utility = |r: &RootMove| expected_score(r.score) + beta * (r.prior.max(0.002) as f64).ln();
    let options: Vec<(RootMove, f64)> = result
        .root_moves
        .iter()
        .filter(|r| r.mv == best.mv || (r.candidate && r.exact && r.score >= floor))
        .map(|r| (*r, utility(r)))
        .collect();
    let max_u = options.iter().map(|o| o.1).fold(f64::MIN, f64::max);
    let chosen = if opts.variety == 0 {
        options.iter().find(|o| o.1 == max_u).unwrap().0
    } else {
        let tau = opts.variety as f64 / 100.0 * 0.02;
        let weights: Vec<f64> = options.iter().map(|o| ((o.1 - max_u) / tau).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut x = next_rand(rng) * total;
        let mut pick = options[0].0;
        for (o, w) in options.iter().zip(&weights) {
            if x < *w {
                pick = o.0;
                break;
            }
            x -= w;
        }
        pick
    };
    let describe = |r: &RootMove| format!("{} (human {:.0}%, {})", r.mv, r.prior * 100.0, uci_score(r.score));
    let alternatives: Vec<String> = options.iter().filter(|o| o.0.mv != chosen.mv).map(|o| describe(&o.0)).collect();
    let explanation = if chosen.mv == best.mv {
        format!("played engine best {}; human alternatives considered: [{}]", describe(&chosen), alternatives.join(", "))
    } else {
        format!("played human move {} instead of engine best {}", describe(&chosen), describe(&best))
    };
    Decision { mv: chosen.mv, explanation }
}

/// Full move decision for `game`: network, search, human choice. Prints UCI info lines.
pub fn think(pool: &mut SearchPool, game: &Game, limits: &Limits, opts: &Options, net: Option<&Network>, seed: u64) -> (Move, SearchResult) {
    let style = opts.style;
    pool.main.eval_params.style = opts.eval_style.unwrap_or(style).eval_style();
    let (h_pct, l_pct) = style.choice_scale();
    let opts = &Options {
        humanity: (opts.humanity * h_pct / 100).min(100),
        max_loss: opts.max_loss * l_pct / 100,
        ..opts.clone()
    };
    let (last, prev) = game.last_moves();
    let net_eval = net.map(|n| n.evaluate(&game.board, last, prev));

    let (mut soft, hard) = time_budget(limits, game.board.stm, opts.move_overhead);
    let mut params = SearchParams {
        depth: limits.depth,
        nodes: limits.nodes,
        soft_ms: soft,
        hard_ms: hard,
        candidate_margin: (opts.max_loss + 60).max(100),
        verbose: !opts.quiet,
        ..Default::default()
    };
    if let Some(ne) = &net_eval {
        params.priors = ne.policy.clone();
        params.candidates = candidate_moves(ne);
        if opts.human_timing && limits.movetime.is_none() {
            if let Some(s) = soft {
                let f = complexity_factor(ne, game);
                soft = Some(((s as f64 * f) as u64).max(1));
                params.soft_ms = soft;
            }
        }
        let top: Vec<String> = ne.policy.iter().take(5).map(|(m, p)| format!("{m} {:.0}%", p * 100.0)).collect();
        if !opts.quiet {
            println!(
            "info string {} net: {} | human W/D/L {:.0}/{:.0}/{:.0}",
            net.map_or("", |n| n.name.as_str()),
            top.join(", "),
            ne.wdl[0] * 100.0,
            ne.wdl[1] * 100.0,
            ne.wdl[2] * 100.0
            );
        }
    }
    // A single legal move needs no thought.
    let legal = crate::movegen::legal_moves(&game.board);
    if legal.len() == 1 && (limits.time[game.board.stm].is_some() || limits.movetime.is_some()) {
        params.depth = Some(1);
    }

    let result = pool.search(&game.board, &game.hashes, &params);
    let mv = if net_eval.is_some() && style != StyleKind::Engine {
        let mut rng = seed | 1;
        let d = choose(&result, opts, &mut rng);
        if !d.explanation.is_empty() && !opts.quiet {
            println!("info string {}", d.explanation);
        }
        d.mv
    } else {
        result.best
    };
    (mv, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rm(mv: &str, score: i32, prior: f32) -> RootMove {
        let b = Board::startpos();
        let m = crate::san::parse_uci(&b, mv).unwrap();
        RootMove { mv: m, score, exact: true, depth: 10, prior, candidate: true }
    }

    fn result(moves: Vec<RootMove>) -> SearchResult {
        SearchResult { best: moves[0].mv, score: moves[0].score, depth: 10, seldepth: 10, nodes: 0, time_ms: 0, pv: vec![], root_moves: moves }
    }

    #[test]
    fn prefers_human_move_when_nearly_equal() {
        let r = result(vec![rm("g1h3", 30, 0.01), rm("e2e4", 20, 0.5), rm("d2d4", 18, 0.3)]);
        let o = Options { variety: 0, ..Default::default() };
        assert_eq!(choose(&r, &o, &mut 1).mv.to_uci(), "e2e4");
    }

    #[test]
    fn keeps_clearly_better_engine_move() {
        let r = result(vec![rm("g1h3", 120, 0.01), rm("e2e4", 10, 0.5)]);
        let o = Options { variety: 0, ..Default::default() };
        assert_eq!(choose(&r, &o, &mut 1).mv.to_uci(), "g1h3");
    }

    #[test]
    fn never_exceeds_max_loss() {
        let r = result(vec![rm("g1h3", 100, 0.001), rm("e2e4", 30, 0.99)]);
        let o = Options { variety: 0, humanity: 100, max_loss: 60, ..Default::default() };
        assert_eq!(choose(&r, &o, &mut 1).mv.to_uci(), "g1h3");
    }
}
