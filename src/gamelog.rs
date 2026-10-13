//! Game records. Every game the engine plays is saved as PGN (readable by any chess program)
//! with the thinking behind each engine move: evaluation, depth, time, the engine's own best
//! move and how likely a human was to play the move. `magician analyze` later finds the
//! errors of both sides in these files.

use crate::board::{Board, START_FEN};
use crate::movegen::legal_moves;
use crate::san::to_san;
use crate::search::{MATE, MATE_BOUND};
use crate::types::*;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// What an engine thought about the move it played.
#[derive(Clone, Debug, Default)]
pub struct EngineNote {
    /// Score of the move played, centipawns from the mover's point of view.
    pub score: Option<i32>,
    pub depth: i32,
    pub nodes: u64,
    pub time_ms: u64,
    /// The engine's own best move (differs from the move played when a human move was chosen).
    pub best: Move,
    pub best_score: Option<i32>,
    /// Probability that a strong human plays the move played, from the network.
    pub human_prob: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct LoggedMove {
    pub mv: Move,
    pub san: String,
    /// Mover's clock after the move, in milliseconds.
    pub clock_ms: Option<u64>,
    pub note: Option<EngineNote>,
}

#[derive(Clone)]
pub struct GameLog {
    pub start: Board,
    pub board: Board,
    hashes: Vec<u64>,
    pub moves: Vec<LoggedMove>,
    pub white: String,
    pub black: String,
    pub event: String,
    /// PGN TimeControl, e.g. "60+0"; "-" if unknown.
    pub time_control: String,
    pub result: String,
    pub termination: String,
    /// Sides played by Magician.
    pub magician: [bool; 2],
    /// "human", "engine" or "unknown".
    pub opponent_type: String,
    date: String,
    time: String,
    /// File the game is written to (chosen on the first save, rewritten after every move).
    pub file: Option<PathBuf>,
}

pub fn result_for(winner: Option<usize>) -> &'static str {
    match winner {
        Some(WHITE) => "1-0",
        Some(_) => "0-1",
        None => "1/2-1/2",
    }
}

impl GameLog {
    pub fn new(start: Board, white: &str, black: &str, event: &str) -> GameLog {
        let (date, time) = local_now();
        GameLog {
            hashes: vec![start.hash],
            board: start.clone(),
            start,
            moves: Vec::new(),
            white: white.into(),
            black: black.into(),
            event: event.into(),
            time_control: "-".into(),
            result: "*".into(),
            termination: String::new(),
            magician: [false; 2],
            opponent_type: "unknown".into(),
            date,
            time,
            file: None,
        }
    }

    pub fn push(&mut self, mv: Move, clock_ms: Option<u64>, note: Option<EngineNote>) {
        let san = to_san(&self.board, mv);
        self.board = self.board.make_move(mv);
        self.hashes.push(self.board.hash);
        self.moves.push(LoggedMove { mv, san, clock_ms, note });
    }

    pub fn move_list(&self) -> Vec<Move> {
        self.moves.iter().map(|m| m.mv).collect()
    }

    pub fn finished(&self) -> bool {
        self.result != "*"
    }

    pub fn set_result(&mut self, result: &str, termination: &str) {
        self.result = result.into();
        self.termination = termination.into();
    }

    /// Sets the result when the current position ends the game by the rules.
    pub fn check_end(&mut self) -> bool {
        let b = &self.board;
        if legal_moves(b).is_empty() {
            if b.in_check() {
                self.set_result(result_for(Some(b.stm ^ 1)), "checkmate");
            } else {
                self.set_result(result_for(None), "stalemate");
            }
        } else if b.is_insufficient_material() {
            self.set_result(result_for(None), "insufficient material");
        } else if b.halfmove >= 100 {
            self.set_result(result_for(None), "fifty-move rule");
        } else {
            let window = (b.halfmove as usize + 1).min(self.hashes.len());
            let repeats = self.hashes[self.hashes.len() - window..].iter().filter(|&&h| h == b.hash).count();
            if repeats >= 3 {
                self.set_result(result_for(None), "threefold repetition");
            }
        }
        self.finished()
    }

    pub fn to_pgn(&self) -> String {
        let mut s = String::new();
        let side = match self.magician {
            [true, true] => "both",
            [true, false] => "white",
            [false, true] => "black",
            _ => "none",
        };
        let mut tag = |k: &str, v: &str| {
            let _ = writeln!(s, "[{k} \"{}\"]", v.replace('\\', "\\\\").replace('"', "\\\""));
        };
        tag("Event", &self.event);
        tag("Site", "Magician");
        tag("Date", &self.date);
        tag("Round", "-");
        tag("White", &self.white);
        tag("Black", &self.black);
        tag("Result", &self.result);
        tag("Time", &self.time);
        tag("TimeControl", &self.time_control);
        if !self.termination.is_empty() {
            tag("Termination", &self.termination);
        }
        tag("MagicianSide", side);
        tag("OpponentType", &self.opponent_type);
        let start_fen = self.start.to_fen();
        if start_fen != START_FEN {
            tag("SetUp", "1");
            tag("FEN", &start_fen);
        }
        s.push('\n');

        let mut tokens: Vec<String> = Vec::new();
        let mut b = self.start.clone();
        let mut after_comment = true;
        for m in &self.moves {
            let number = b.fullmove;
            if b.stm == WHITE {
                tokens.push(format!("{number}. {}", m.san));
            } else if after_comment {
                tokens.push(format!("{number}... {}", m.san));
            } else {
                tokens.push(m.san.clone());
            }
            let comment = comment_for(m, &b);
            after_comment = !comment.is_empty();
            if after_comment {
                tokens.push(format!("{{ {comment} }}"));
            }
            b = b.make_move(m.mv);
        }
        tokens.push(self.result.clone());
        let mut line = String::new();
        for t in tokens {
            if !line.is_empty() && line.len() + t.len() + 1 > 90 {
                s.push_str(&line);
                s.push('\n');
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&t);
        }
        s.push_str(&line);
        s.push_str("\n\n");
        s
    }

    /// Writes the game to its PGN file in `dir` (a new file on the first call, the same file
    /// afterwards, so an unfinished game is never lost) and returns the path.
    pub fn save(&mut self, dir: &Path) -> io::Result<PathBuf> {
        let path = match &self.file {
            Some(p) => p.clone(),
            None => {
                fs::create_dir_all(dir)?;
                let clean = |n: &str| -> String {
                    let c: String = n.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
                    c.trim_matches('_').chars().take(30).collect()
                };
                let stamp = format!("{}_{}", self.date.replace('.', "-"), self.time.replace(':', "-"));
                let base = format!("{stamp}_{}_vs_{}", clean(&self.white), clean(&self.black));
                let mut path = dir.join(format!("{base}.pgn"));
                let mut n = 2;
                while path.exists() {
                    path = dir.join(format!("{base}_{n}.pgn"));
                    n += 1;
                }
                self.file = Some(path.clone());
                path
            }
        };
        fs::write(&path, self.to_pgn())?;
        Ok(path)
    }
}

/// "[%clk 0:00:58.4] [%eval 0.31] ..." for one move played in position `before`.
fn comment_for(m: &LoggedMove, before: &Board) -> String {
    let mut parts = Vec::new();
    if let Some(ms) = m.clock_ms {
        parts.push(format!("[%clk {}]", clock_text(ms)));
    }
    if let Some(n) = &m.note {
        if let Some(sc) = n.score {
            parts.push(format!("[%eval {}]", eval_text(if before.stm == WHITE { sc } else { -sc })));
        }
        let mut text = format!("глубина {}, {:.2} с", n.depth, n.time_ms as f64 / 1000.0);
        if n.best.is_some() && n.best != m.mv && legal_moves(before).contains(n.best) {
            let white_view = |s: i32| if before.stm == WHITE { s } else { -s };
            let best = n.best_score.map_or(String::new(), |s| format!(" ({})", eval_text(white_view(s))));
            let _ = write!(text, "; движок предлагал {}{best}", to_san(before, n.best));
        }
        if let Some(p) = n.human_prob {
            let _ = write!(text, "; у людей {:.0}%", p * 100.0);
        }
        parts.push(text);
    }
    parts.join(" ")
}

pub fn clock_text(ms: u64) -> String {
    let t = ms / 100;
    format!("{}:{:02}:{:02}.{}", t / 36000, t / 600 % 60, t / 10 % 60, t % 10)
}

/// Pawns with two decimals, or "#N" for a mate in N moves (negative when being mated).
pub fn eval_text(cp: i32) -> String {
    if cp >= MATE_BOUND {
        format!("#{}", (MATE - cp + 1) / 2)
    } else if cp <= -MATE_BOUND {
        format!("#-{}", (MATE + cp) / 2)
    } else {
        format!("{:.2}", cp as f64 / 100.0)
    }
}

/// Local date ("2026.10.13") and time ("04:15:33").
fn local_now() -> (String, String) {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct SystemTime {
            year: u16,
            month: u16,
            day_of_week: u16,
            day: u16,
            hour: u16,
            minute: u16,
            second: u16,
            ms: u16,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetLocalTime(t: *mut SystemTime);
        }
        let mut t = SystemTime::default();
        unsafe { GetLocalTime(&mut t) };
        (format!("{:04}.{:02}.{:02}", t.year, t.month, t.day), format!("{:02}:{:02}:{:02}", t.hour, t.minute, t.second))
    }
    #[cfg(not(windows))]
    {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
        let (y, m, d) = civil_from_days(secs.div_euclid(86400));
        let sod = secs.rem_euclid(86400);
        (format!("{y:04}.{m:02}.{d:02}"), format!("{:02}:{:02}:{:02}", sod / 3600, sod / 60 % 60, sod % 60))
    }
}

#[cfg(not(windows))]
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// Records the games played through UCI. GUIs only send positions, so a game is followed
/// from one `position` command to the next while the move list keeps growing; anything else
/// starts a new game and the previous one is written to disk.
pub struct Recorder {
    pub enabled: bool,
    pub dir: PathBuf,
    pub opponent_name: Option<String>,
    pub opponent_type: Option<String>,
    log: Option<GameLog>,
}

impl Recorder {
    pub fn new(dir: PathBuf) -> Recorder {
        Recorder { enabled: true, dir, opponent_name: None, opponent_type: None, log: None }
    }

    /// A `position` command: the game from its start position to now.
    pub fn sync(&mut self, start: &Board, moves: &[Move]) {
        if !self.enabled {
            return;
        }
        let continues = self.log.as_ref().is_some_and(|l| {
            l.start.hash == start.hash
                && l.moves.len() <= moves.len()
                && l.moves.iter().zip(moves).all(|(a, &b)| a.mv == b)
        });
        if !continues {
            self.flush();
            self.log = Some(GameLog::new(start.clone(), "?", "?", "Magician UCI game"));
        }
        let log = self.log.as_mut().unwrap();
        for &m in &moves[log.moves.len()..] {
            log.push(m, None, None);
        }
    }

    /// A `go` command: remember who plays which side and the clocks, if known.
    pub fn go(&mut self, engine_name: &str, clocks: [Option<u64>; 2], incs: [Option<u64>; 2]) {
        let Some(log) = self.log.as_mut() else { return };
        let us = log.board.stm;
        log.magician[us] = true;
        let opponent = self.opponent_name.clone().unwrap_or_else(|| "Opponent".into());
        if log.magician[us ^ 1] {
            log.white = engine_name.into();
            log.black = engine_name.into();
        } else if us == WHITE {
            log.white = engine_name.into();
            log.black = opponent;
        } else {
            log.white = opponent;
            log.black = engine_name.into();
        }
        if let Some(t) = &self.opponent_type {
            log.opponent_type = t.clone();
        }
        if log.time_control == "-" {
            if let Some(t) = clocks[us] {
                log.time_control = format!("{}+{}", t.div_ceil(1000), incs[us].unwrap_or(0) / 1000);
            }
        }
        // The opponent's clock after its last move is only known now.
        if let (Some(last), Some(t)) = (log.moves.last_mut(), clocks[us ^ 1]) {
            if last.note.is_none() && last.clock_ms.is_none() {
                last.clock_ms = Some(t);
            }
        }
    }

    /// The engine played `mv` in the current position. The game file is updated right away.
    pub fn engine_move(&mut self, mv: Move, clock_ms: Option<u64>, note: EngineNote) {
        if let Some(log) = self.log.as_mut() {
            log.push(mv, clock_ms, Some(note));
            log.check_end();
            let _ = log.save(&self.dir.join("games"));
        }
    }

    /// Writes the current game (if the engine played in it) and forgets it.
    pub fn flush(&mut self) -> Option<PathBuf> {
        let mut log = self.log.take()?;
        if !log.moves.iter().any(|m| m.note.is_some()) {
            return None;
        }
        log.check_end();
        log.save(&self.dir.join("games")).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::san::parse_san;

    #[test]
    fn writes_readable_pgn_with_clocks_and_evals() {
        let mut log = GameLog::new(Board::startpos(), "Magician (Human)", "Opponent", "test");
        log.magician[WHITE] = true;
        for (i, san) in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"].iter().enumerate() {
            let m = parse_san(&log.board, san).unwrap();
            let note = (i % 2 == 0).then(|| EngineNote { score: Some(35), depth: 12, time_ms: 420, best: m, ..Default::default() });
            log.push(m, Some(59_000 - i as u64 * 1000), note);
        }
        assert!(log.check_end());
        assert_eq!(log.result, "1-0");
        let pgn = log.to_pgn();
        assert!(pgn.contains("[Termination \"checkmate\"]"));
        assert!(pgn.contains("1. e4 { [%clk 0:00:59.0] [%eval 0.35] глубина 12"));
        assert!(pgn.contains("4. Qxf7#"));
        // Our own PGN reader must read it back.
        let games: Vec<_> = crate::pgn::PgnReader::new(pgn.as_bytes()).collect();
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].moves.len(), 7);
        assert_eq!(games[0].result, "1-0");
        assert_eq!(games[0].clocks[0], Some(59));
    }
}
