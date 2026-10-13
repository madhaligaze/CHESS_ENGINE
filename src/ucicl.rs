//! Minimal UCI client: lets the play app put another engine (e.g. Stockfish) at the board.

use crate::types::Move;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::Duration;

pub struct UciEngine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    pub name: String,
}

pub struct EngineReply {
    pub bestmove: String,
    /// Last reported score, centipawns from the engine's point of view (mates as +-30000).
    pub score: Option<i32>,
    pub depth: i32,
    pub nodes: u64,
}

impl UciEngine {
    pub fn start(path: &Path, options: &[(&str, String)]) -> io::Result<UciEngine> {
        let mut cmd = Command::new(path);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| io::Error::other("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| io::Error::other("no stdout"))?;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut e = UciEngine { child, stdin, lines: rx, name: path.file_stem().map_or("engine".into(), |s| s.to_string_lossy().into()) };
        e.send("uci")?;
        loop {
            let l = e.read(Duration::from_secs(10))?;
            if let Some(n) = l.strip_prefix("id name ") {
                e.name = n.trim().to_string();
            }
            if l.trim() == "uciok" {
                break;
            }
        }
        for (k, v) in options {
            e.send(&format!("setoption name {k} value {v}"))?;
        }
        e.ready()?;
        Ok(e)
    }

    fn send(&mut self, s: &str) -> io::Result<()> {
        writeln!(self.stdin, "{s}")?;
        self.stdin.flush()
    }

    fn read(&self, timeout: Duration) -> io::Result<String> {
        match self.lines.recv_timeout(timeout) {
            Ok(l) => Ok(l),
            Err(RecvTimeoutError::Timeout) => Err(io::Error::new(io::ErrorKind::TimedOut, "engine timed out")),
            Err(RecvTimeoutError::Disconnected) => Err(io::Error::other("engine exited")),
        }
    }

    fn ready(&mut self) -> io::Result<()> {
        self.send("isready")?;
        while self.read(Duration::from_secs(10))?.trim() != "readyok" {}
        Ok(())
    }

    pub fn new_game(&mut self) -> io::Result<()> {
        self.send("ucinewgame")?;
        self.ready()
    }

    /// Asks for a move with the given clocks (milliseconds).
    pub fn go(&mut self, start_fen: &str, moves: &[Move], clocks: [u64; 2], inc: u64) -> io::Result<EngineReply> {
        let list: Vec<String> = moves.iter().map(|m| m.to_uci()).collect();
        let pos = if list.is_empty() {
            format!("position fen {start_fen}")
        } else {
            format!("position fen {start_fen} moves {}", list.join(" "))
        };
        self.send(&pos)?;
        self.send(&format!("go wtime {} btime {} winc {inc} binc {inc}", clocks[0], clocks[1]))?;
        let mut reply = EngineReply { bestmove: String::new(), score: None, depth: 0, nodes: 0 };
        let deadline = Duration::from_millis(clocks[0].max(clocks[1]) + 10_000);
        loop {
            let l = self.read(deadline)?;
            let t: Vec<&str> = l.split_whitespace().collect();
            match t.first() {
                Some(&"bestmove") => {
                    reply.bestmove = t.get(1).unwrap_or(&"").to_string();
                    return Ok(reply);
                }
                Some(&"info") => {
                    for i in 0..t.len() {
                        match t[i] {
                            "depth" => reply.depth = t.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(reply.depth),
                            "nodes" => reply.nodes = t.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(reply.nodes),
                            "cp" => reply.score = t.get(i + 1).and_then(|v| v.parse().ok()),
                            "mate" => {
                                reply.score = t.get(i + 1).and_then(|v| v.parse::<i32>().ok()).map(|n| {
                                    if n > 0 {
                                        crate::search::MATE - (2 * n - 1)
                                    } else {
                                        -crate::search::MATE - 2 * n
                                    }
                                })
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn stop(&mut self) {
        let _ = self.send("stop");
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        std::thread::sleep(Duration::from_millis(50));
        let _ = self.child.kill();
    }
}

/// Stockfish shipped next to the program or in tools/stockfish of the project.
pub fn find_stockfish(data_dir: &Path) -> Option<PathBuf> {
    let mut dirs = vec![data_dir.to_path_buf(), data_dir.join("stockfish"), data_dir.join("tools").join("stockfish")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.to_path_buf());
        }
    }
    let mut stack = dirs;
    let mut depth = 0;
    while depth < 3 && !stack.is_empty() {
        let mut next = Vec::new();
        for d in stack {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
                if p.is_file() && name.starts_with("stockfish") && name.ends_with(".exe") {
                    return Some(p);
                }
                if p.is_dir() && name.starts_with("stockfish") {
                    next.push(p);
                }
            }
        }
        stack = next;
        depth += 1;
    }
    None
}
