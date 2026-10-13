//! UCI protocol front end. The search runs on its own thread so `stop` and `isready` are
//! answered while thinking.

use crate::board::Board;
use crate::engine::{data_dir, think, Game, Nets, Options, StyleKind};
use crate::eval::{evaluate_full, Breakdown, EvalParams};
use crate::gamelog::{EngineNote, Recorder};
use crate::learn::Learn;
use crate::search::{Limits, SearchPool, SEARCH_STACK};
use crate::types::*;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

pub const NAME: &str = "Magician 0.1";
pub const AUTHOR: &str = "Magician authors";

fn max_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).max(1)
}

pub fn style_name(s: StyleKind) -> &'static str {
    match s {
        StyleKind::Human => "Human",
        StyleKind::Tal => "Tal",
        StyleKind::Carlsen => "Carlsen",
        StyleKind::Engine => "Engine",
    }
}

pub struct Uci {
    game: Game,
    opts: Options,
    nets: Nets,
    searcher: Arc<Mutex<SearchPool>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    recorder: Arc<Mutex<Recorder>>,
    learn: Arc<Learn>,
    data_dir: PathBuf,
}

impl Default for Uci {
    fn default() -> Self {
        Self::new()
    }
}

impl Uci {
    pub fn new() -> Uci {
        let opts = Options::default();
        let stop = Arc::new(AtomicBool::new(false));
        let dir = data_dir();
        Uci {
            game: Game::new(Board::startpos()),
            nets: Nets::load(None),
            searcher: Arc::new(Mutex::new(SearchPool::new(opts.hash_mb, opts.threads, stop.clone()))),
            opts,
            stop,
            worker: None,
            recorder: Arc::new(Mutex::new(Recorder::new(dir.clone()))),
            learn: Arc::new(Learn::load(&dir.join("learn.txt"))),
            data_dir: dir,
        }
    }

    fn wait(&mut self) {
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }

    pub fn run(&mut self) {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if !self.command(line.trim_start_matches('\u{feff}').trim()) {
                break;
            }
        }
        self.stop.store(true, Ordering::Relaxed);
        self.wait();
        self.recorder.lock().unwrap().flush();
    }

    /// Handles one command; returns false on `quit`.
    pub fn command(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else { return true };
        match cmd {
            "uci" => self.uci_id(),
            "isready" => println!("readyok"),
            "ucinewgame" => {
                self.wait();
                self.searcher.lock().unwrap().clear();
                self.game = Game::new(Board::startpos());
                self.recorder.lock().unwrap().flush();
                self.learn = Arc::new(Learn::load(&self.data_dir.join("learn.txt")));
            }
            "setoption" => {
                self.wait();
                self.set_option(&tokens[1..]);
            }
            "position" => {
                self.wait();
                if let Err(e) = self.position(&tokens[1..]) {
                    println!("info string {e}");
                }
            }
            "go" => {
                self.wait();
                self.go(&tokens[1..]);
            }
            "stop" => self.stop.store(true, Ordering::Relaxed),
            "ponderhit" => {}
            "quit" => return false,
            "d" => print!("{}", self.game.board.pretty()),
            "eval" => self.print_eval(),
            "wait" => self.wait(),
            _ => println!("info string unknown command '{cmd}'"),
        }
        true
    }

    fn uci_id(&self) {
        println!("id name {NAME}");
        println!("id author {AUTHOR}");
        println!("option name Hash type spin default 64 min 1 max 8192");
        println!("option name Threads type spin default 1 min 1 max {}", max_threads());
        println!("option name Style type combo default Human var Human var Tal var Carlsen var Engine");
        println!("option name Humanity type spin default 25 min 0 max 100");
        println!("option name Variety type spin default 20 min 0 max 100");
        println!("option name MaxLoss type spin default 60 min 0 max 1000");
        println!("option name HumanTiming type check default true");
        println!("option name Move Overhead type spin default 30 min 0 max 5000");
        println!("option name NetDir type string default <auto>");
        println!("option name GameLog type check default true");
        println!("option name UCI_Opponent type string default <empty>");
        println!("option name Clear Hash type button");
        match &self.nets.dir {
            Some(d) => println!(
                "info string networks from {}: base {}, tal {}, carlsen {}",
                d.display(),
                self.nets.base.is_some(),
                self.nets.tal.is_some(),
                self.nets.carlsen.is_some()
            ),
            None => println!("info string no networks found: human styles fall back to the pure engine"),
        }
        println!("uciok");
    }

    fn set_option(&mut self, t: &[&str]) {
        // setoption name <name words> [value <value words>]
        let vi = t.iter().position(|&w| w == "value");
        let name = t[1.min(t.len())..vi.unwrap_or(t.len())].join(" ").to_ascii_lowercase();
        let value = vi.map(|i| t[i + 1..].join(" ")).unwrap_or_default();
        let num = |v: &str| v.trim().parse::<i64>().ok();
        match name.as_str() {
            "hash" => {
                if let Some(mb) = num(&value) {
                    self.opts.hash_mb = mb.clamp(1, 8192) as usize;
                    self.searcher.lock().unwrap().resize_tt(self.opts.hash_mb);
                }
            }
            "threads" => {
                self.opts.threads = num(&value).unwrap_or(1).clamp(1, max_threads() as i64) as usize;
                self.searcher.lock().unwrap().set_threads(self.opts.threads);
            }
            "style" => match StyleKind::parse(&value) {
                Some(s) => self.opts.style = s,
                None => println!("info string unknown style '{value}'"),
            },
            "humanity" => self.opts.humanity = num(&value).unwrap_or(25).clamp(0, 100) as u32,
            "variety" => self.opts.variety = num(&value).unwrap_or(20).clamp(0, 100) as u32,
            "maxloss" => self.opts.max_loss = num(&value).unwrap_or(60).clamp(0, 1000) as i32,
            "humantiming" => self.opts.human_timing = value.eq_ignore_ascii_case("true"),
            "move overhead" => self.opts.move_overhead = num(&value).unwrap_or(30).clamp(0, 5000) as u64,
            "netdir" => {
                let dir = if value.is_empty() || value == "<auto>" { None } else { Some(PathBuf::from(&value)) };
                self.nets = Nets::load(dir.as_deref());
                self.opts.net_dir = dir;
                println!("info string networks loaded: base {}", self.nets.base.is_some());
            }
            "clear hash" => self.searcher.lock().unwrap().clear(),
            "gamelog" => self.recorder.lock().unwrap().enabled = value.eq_ignore_ascii_case("true"),
            "uci_opponent" => {
                // "<title> <rating> <computer|human> <name>"
                let parts: Vec<&str> = value.split_whitespace().collect();
                let mut rec = self.recorder.lock().unwrap();
                if parts.len() >= 4 {
                    let title = if parts[0].eq_ignore_ascii_case("none") { String::new() } else { format!("{} ", parts[0]) };
                    rec.opponent_name = Some(format!("{title}{} ({})", parts[3..].join(" "), parts[1]));
                    rec.opponent_type = Some(if parts[2].eq_ignore_ascii_case("human") { "human" } else { "engine" }.into());
                }
            }
            _ => println!("info string unknown option '{name}'"),
        }
    }

    fn position(&mut self, t: &[&str]) -> Result<(), String> {
        let mi = t.iter().position(|&w| w == "moves");
        let board = match t.first() {
            Some(&"startpos") => Board::startpos(),
            Some(&"fen") => Board::from_fen(&t[1..mi.unwrap_or(t.len())].join(" "))?,
            _ => return Err("position: expected startpos or fen".into()),
        };
        let mut game = Game::new(board.clone());
        if let Some(i) = mi {
            for s in &t[i + 1..] {
                let m = crate::san::parse_uci(&game.board, s).ok_or(format!("illegal move {s}"))?;
                game.play(m);
            }
        }
        self.recorder.lock().unwrap().sync(&board, &game.moves);
        self.game = game;
        Ok(())
    }

    fn go(&mut self, t: &[&str]) {
        let mut limits = Limits::default();
        let mut i = 0;
        let num = |i: usize| t.get(i + 1).and_then(|v| v.parse::<i64>().ok()).map(|v| v.max(0) as u64);
        while i < t.len() {
            match t[i] {
                "wtime" => limits.time[WHITE] = num(i),
                "btime" => limits.time[BLACK] = num(i),
                "winc" => limits.inc[WHITE] = num(i),
                "binc" => limits.inc[BLACK] = num(i),
                "movestogo" => limits.movestogo = num(i).map(|v| v as u32),
                "depth" => limits.depth = num(i).map(|v| v.max(1) as i32),
                "nodes" => limits.nodes = num(i),
                "movetime" => limits.movetime = num(i),
                "infinite" => limits.infinite = true,
                _ => {}
            }
            i += 1;
        }
        self.stop.store(false, Ordering::Relaxed);
        let searcher = self.searcher.clone();
        let game = self.game.clone();
        let opts = self.opts.clone();
        let net = self.nets.for_style(opts.style);
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
        let engine_name = format!("Magician ({})", style_name(opts.style));
        self.recorder.lock().unwrap().go(&engine_name, limits.time, limits.inc);
        let recorder = self.recorder.clone();
        let avoid = self.learn.avoid(&game.board);
        let handle = std::thread::Builder::new()
            .stack_size(SEARCH_STACK)
            .spawn(move || {
                let mut s = searcher.lock().unwrap();
                let th = think(&mut s, &game, &limits, &opts, net.as_deref(), seed, &avoid);
                println!("bestmove {}", th.mv.to_uci());
                let stm = game.board.stm;
                let r = &th.result;
                let note = EngineNote {
                    score: Some(th.played_score()),
                    depth: r.depth,
                    nodes: r.nodes,
                    time_ms: r.time_ms,
                    best: r.root_moves.first().map_or(th.mv, |m| m.mv),
                    best_score: r.root_moves.first().map(|m| m.score),
                    human_prob: (!th.policy.is_empty()).then(|| th.played_prior()),
                };
                let clock = limits.time[stm].map(|t| t.saturating_sub(r.time_ms) + limits.inc[stm].unwrap_or(0));
                recorder.lock().unwrap().engine_move(th.mv, clock, note);
            })
            .expect("spawn search thread");
        self.worker = Some(handle);
    }

    fn print_eval(&self) {
        let b = &self.game.board;
        let params = EvalParams { style: self.opts.style.eval_style(), engine_color: b.stm };
        let mut bd = Breakdown::default();
        let v = evaluate_full(b, &params, Some(&mut bd));
        println!("{bd:?}");
        println!("static eval (side to move): {v} cp");
        if let Some(net) = self.nets.for_style(self.opts.style) {
            let (last, prev) = self.game.last_moves();
            let e = net.evaluate(b, last, prev);
            println!("network '{}': W/D/L {:.1}/{:.1}/{:.1}", net.name, e.wdl[0] * 100.0, e.wdl[1] * 100.0, e.wdl[2] * 100.0);
            for (m, p) in e.policy.iter().take(8) {
                println!("  {:7} {:5.1}%", crate::san::to_san(b, *m), p * 100.0);
            }
        }
    }
}
