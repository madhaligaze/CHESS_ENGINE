//! Post-game analysis of saved games (`magician analyze`): every position is searched more
//! deeply than during play, each move of both sides is graded the way Lichess does it
//! (inaccuracy / mistake / blunder by the drop in winning chances), and the findings feed
//! the engine's learning:
//! - annotated PGN and a short report next to the games,
//! - the engine's own mistakes go to learn.txt, so it does not repeat them,
//! - training records for the human-move network: the better move in positions where the
//!   engine erred, and the moves of human opponents (data/learned/*.bin).

use crate::board::{Board, START_FEN};
use crate::engine::Game;
use crate::features::{encode_record, RecordMeta};
use crate::learn::Learn;
use crate::movegen::legal_moves;
use crate::pgn::{PgnGame, PgnReader};
use crate::san::{parse_san, to_san};
use crate::search::{SearchParams, SearchPool, MATE};
use crate::types::*;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct AnalyzeConfig {
    pub inputs: Vec<PathBuf>,
    pub movetime_ms: u64,
    pub threads: usize,
    /// Where learn.txt and data/learned/ live.
    pub data_dir: PathBuf,
    pub learn: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Judgement {
    Inaccuracy,
    Mistake,
    Blunder,
}

impl Judgement {
    pub fn nag(self) -> &'static str {
        match self {
            Judgement::Inaccuracy => "$6",
            Judgement::Mistake => "$2",
            Judgement::Blunder => "$4",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Judgement::Inaccuracy => "Неточность",
            Judgement::Mistake => "Ошибка",
            Judgement::Blunder => "Зевок",
        }
    }
}

/// Lichess "winning chances" in [-1, 1] for a centipawn score.
pub fn winning_chances(cp: i32) -> f64 {
    let cp = cp.clamp(-1500, 1500) as f64;
    2.0 / (1.0 + (-0.003_682_08 * cp).exp()) - 1.0
}

/// Lichess move accuracy (0..100) from win percentages before and after the move.
fn move_accuracy(before_wc: f64, after_wc: f64) -> f64 {
    let drop = (before_wc - after_wc).max(0.0) * 50.0;
    (103.1668 * (-0.04354 * drop).exp() - 3.1669).clamp(0.0, 100.0)
}

pub struct MoveReport {
    pub ply: usize,
    pub side: usize,
    pub san: String,
    pub best_san: String,
    /// Mover's point of view, centipawns (mates mapped to large values).
    pub before: i32,
    pub after: i32,
    pub judgement: Option<Judgement>,
    pub accuracy: f64,
}

pub struct GameReport {
    pub white: String,
    pub black: String,
    pub result: String,
    pub magician: [bool; 2],
    pub opponent_type: String,
    pub moves: Vec<MoveReport>,
    pub annotated_pgn: String,
}

impl GameReport {
    fn side_summary(&self, side: usize) -> (usize, usize, usize, f64) {
        let mine: Vec<&MoveReport> = self.moves.iter().filter(|m| m.side == side).collect();
        let count = |j: Judgement| mine.iter().filter(|m| m.judgement == Some(j)).count();
        let acc = if mine.is_empty() { 0.0 } else { mine.iter().map(|m| m.accuracy).sum::<f64>() / mine.len() as f64 };
        (count(Judgement::Inaccuracy), count(Judgement::Mistake), count(Judgement::Blunder), acc)
    }

    pub fn text(&self) -> String {
        let mut s = format!("{} - {}  {}\n", self.white, self.black, self.result);
        for side in [WHITE, BLACK] {
            let (i, m, b, acc) = self.side_summary(side);
            let name = if side == WHITE { &self.white } else { &self.black };
            let _ = writeln!(s, "  {name}: точность {acc:.0}%, неточностей {i}, ошибок {m}, зевков {b}");
        }
        let errors: Vec<&MoveReport> = self.moves.iter().filter(|m| m.judgement.is_some()).collect();
        if !errors.is_empty() {
            s.push_str("  Ошибки:\n");
        }
        for m in errors {
            let mover = if m.side == WHITE { &self.white } else { &self.black };
            let _ = writeln!(
                s,
                "    {}{} {} ({mover}): {}; оценка {} -> {}, лучше было {}",
                m.ply / 2 + 1,
                if m.side == WHITE { "." } else { "..." },
                m.san,
                m.judgement.unwrap().name().to_lowercase(),
                fmt_cp(m.before),
                fmt_cp(m.after),
                m.best_san
            );
        }
        s
    }
}

fn fmt_cp(cp: i32) -> String {
    if cp.abs() >= 1500 {
        if cp > 0 { "мат".into() } else { "получает мат".into() }
    } else {
        format!("{:+.2}", cp as f64 / 100.0)
    }
}

/// Score of the position for the side to move, by search (or by the rules at the end).
fn evaluate_position(pool: &mut SearchPool, game: &Game, movetime_ms: u64) -> (i32, Move) {
    let legal = legal_moves(&game.board);
    if legal.is_empty() {
        return (if game.board.in_check() { -1500 } else { 0 }, Move::NONE);
    }
    let params = SearchParams { soft_ms: Some(movetime_ms), hard_ms: Some(movetime_ms), ..Default::default() };
    let r = pool.search(&game.board, &game.hashes, &params);
    // Mates count as decisive but bounded scores, as on Lichess.
    let score = if r.score.abs() >= MATE - 1000 { r.score.signum() * 1500 } else { r.score.clamp(-1500, 1500) };
    (score, r.best)
}

fn game_sides(g: &PgnGame) -> [bool; 2] {
    match g.tag("MagicianSide") {
        Some("white") => [true, false],
        Some("black") => [false, true],
        Some("both") => [true, true],
        Some(_) => [false, false],
        None => {
            let is = |k: &str| g.tag(k).is_some_and(|n| n.contains("Magician"));
            [is("White"), is("Black")]
        }
    }
}

pub fn analyze_game(pool: &mut SearchPool, g: &PgnGame, movetime_ms: u64) -> Option<(GameReport, Vec<(Board, Move, Move, Move, Move)>)> {
    let start = match g.tag("FEN") {
        Some(f) => Board::from_fen(f).ok()?,
        None => Board::from_fen(START_FEN).ok()?,
    };
    let mut game = Game::new(start.clone());
    let mut moves = Vec::new();
    for san in &g.moves {
        let m = parse_san(&game.board, san)?;
        moves.push(m);
        game.play(m);
    }
    pool.clear();
    // Search every position from the start.
    let mut replay = Game::new(start.clone());
    let mut evals = Vec::with_capacity(moves.len() + 1);
    evals.push(evaluate_position(pool, &replay, movetime_ms));
    for &m in &moves {
        replay.play(m);
        evals.push(evaluate_position(pool, &replay, movetime_ms));
    }

    let mut report = GameReport {
        white: g.tag("White").unwrap_or("?").into(),
        black: g.tag("Black").unwrap_or("?").into(),
        result: g.result.clone(),
        magician: game_sides(g),
        opponent_type: g.tag("OpponentType").unwrap_or("unknown").into(),
        moves: Vec::new(),
        annotated_pgn: String::new(),
    };
    // (position, last, prev, played, best) for each move, for training data.
    let mut positions = Vec::new();
    let mut b = start.clone();
    let (mut last, mut prev) = (Move::NONE, Move::NONE);
    let mut movetext = Vec::new();
    for (i, &m) in moves.iter().enumerate() {
        let side = b.stm;
        let (before, best) = evals[i];
        let after = -evals[i + 1].0;
        let (wc_before, wc_after) = (winning_chances(before), winning_chances(after));
        let drop = wc_before - wc_after;
        let judgement = if m == best {
            None
        } else if drop >= 0.3 {
            Some(Judgement::Blunder)
        } else if drop >= 0.2 {
            Some(Judgement::Mistake)
        } else if drop >= 0.1 {
            Some(Judgement::Inaccuracy)
        } else {
            None
        };
        let san = to_san(&b, m);
        let best_san = if best.is_some() { to_san(&b, best) } else { "-".into() };
        let number = if side == WHITE { format!("{}. ", b.fullmove) } else { format!("{}... ", b.fullmove) };
        let mut token = format!("{number}{san}");
        if let Some(j) = judgement {
            let _ = write!(token, " {} {{ {}. Лучше {} ({}) }}", j.nag(), j.name(), best_san, fmt_cp(before));
        }
        let white_view = if side == WHITE { after } else { -after };
        let _ = write!(token, " {{ [%eval {:.2}] }}", white_view as f64 / 100.0);
        movetext.push(token);
        report.moves.push(MoveReport {
            ply: i,
            side,
            san,
            best_san,
            before,
            after,
            judgement,
            accuracy: move_accuracy(wc_before, wc_after),
        });
        positions.push((b.clone(), last, prev, m, best));
        b = b.make_move(m);
        prev = last;
        last = m;
    }
    let mut pgn = String::new();
    for (k, v) in &g.tags {
        let _ = writeln!(pgn, "[{k} \"{v}\"]");
    }
    let _ = writeln!(pgn, "[Annotator \"Magician analyze, {movetime_ms} ms per position\"]\n");
    pgn.push_str(&movetext.join(" "));
    let _ = writeln!(pgn, " {}\n", g.result);
    report.annotated_pgn = pgn;
    Some((report, positions))
}

fn result_for(result: &str, side: usize) -> u8 {
    match (result, side) {
        ("1-0", WHITE) | ("0-1", BLACK) => 2,
        ("1-0", _) | ("0-1", _) => 0,
        _ => 1,
    }
}

pub fn run(cfg: &AnalyzeConfig) -> io::Result<()> {
    let mut pool = SearchPool::new(64, cfg.threads, Arc::new(AtomicBool::new(false)));
    pool.main.eval_params.style = crate::eval::Style::NEUTRAL;
    let mut learn = Learn::load(&cfg.data_dir.join("learn.txt"));
    let learned_dir = cfg.data_dir.join("data").join("learned");
    let mut new_lessons = 0;
    let mut records_errors = 0;
    let mut records_human = 0;
    for path in &cfg.inputs {
        let file = fs::File::open(path)?;
        let out_dir = path.parent().unwrap_or(Path::new(".")).join("analysis");
        fs::create_dir_all(&out_dir)?;
        for (gi, g) in PgnReader::new(BufReader::new(file)).enumerate() {
            let Some((report, positions)) = analyze_game(&mut pool, &g, cfg.movetime_ms) else {
                eprintln!("{}: game {} could not be replayed, skipped", path.display(), gi + 1);
                continue;
            };
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("game");
            let name = if gi == 0 { stem.to_string() } else { format!("{stem}_{}", gi + 1) };
            fs::write(out_dir.join(format!("{name}.pgn")), &report.annotated_pgn)?;
            let text = report.text();
            fs::write(out_dir.join(format!("{name}.txt")), &text)?;
            print!("{text}");

            if !cfg.learn {
                continue;
            }
            fs::create_dir_all(&learned_dir)?;
            let mut errors_out = fs::OpenOptions::new().create(true).append(true).open(learned_dir.join("engine_errors.bin"))?;
            let mut human_out = fs::OpenOptions::new().create(true).append(true).open(learned_dir.join("opponent_moves.bin"))?;
            for (mr, (board, last, prev, played, best)) in report.moves.iter().zip(&positions) {
                let meta = RecordMeta {
                    result: result_for(&report.result, mr.side),
                    mover_elo: 0,
                    opp_elo: 0,
                    style: 3,
                    time_class: 0,
                    ply: mr.ply as u16,
                    game_id: 0,
                    clock: None,
                };
                if report.magician[mr.side] {
                    let serious = matches!(mr.judgement, Some(Judgement::Mistake | Judgement::Blunder));
                    if serious && best.is_some() {
                        let note = format!("{} {}: {} -> {}", report.white, report.black, fmt_cp(mr.before), fmt_cp(mr.after));
                        if learn.add(board, *played, &note)? {
                            new_lessons += 1;
                        }
                        errors_out.write_all(&encode_record(board, *last, *prev, *best, &meta))?;
                        records_errors += 1;
                    }
                } else if report.opponent_type == "human" {
                    human_out.write_all(&encode_record(board, *last, *prev, *played, &meta))?;
                    records_human += 1;
                }
            }
        }
    }
    if cfg.learn {
        println!(
            "\nпамять ошибок: +{new_lessons} (всего {}), {}; обучающие записи: ошибки движка +{records_errors}, ходы людей +{records_human} в {}",
            learn.len(),
            cfg.data_dir.join("learn.txt").display(),
            learned_dir.display()
        );
    }
    Ok(())
}
