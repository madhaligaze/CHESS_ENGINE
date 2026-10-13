//! Memory of mistakes: positions where `magician analyze` found that a move of the engine was
//! an error. The engine does not play such a move again in the same position (unless it is
//! the only legal move). Stored as text, one entry per line:
//! `<position hash, 16 hex digits> <move in UCI> <FEN and note>`.

use crate::board::Board;
use crate::movegen::legal_moves;
use crate::types::Move;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[derive(Default, Clone)]
pub struct Learn {
    path: PathBuf,
    map: HashMap<u64, Vec<String>>,
}

impl Learn {
    pub fn load(path: &Path) -> Learn {
        let mut l = Learn { path: path.to_path_buf(), map: HashMap::new() };
        if let Ok(text) = fs::read_to_string(path) {
            for line in text.lines() {
                let mut it = line.split_whitespace();
                if let (Some(h), Some(m)) = (it.next(), it.next()) {
                    if let Ok(hash) = u64::from_str_radix(h, 16) {
                        l.map.entry(hash).or_default().push(m.to_string());
                    }
                }
            }
        }
        l
    }

    pub fn len(&self) -> usize {
        self.map.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Legal moves known to be mistakes in this position.
    pub fn avoid(&self, b: &Board) -> Vec<Move> {
        let Some(bad) = self.map.get(&b.hash) else { return Vec::new() };
        legal_moves(b).as_slice().iter().copied().filter(|m| bad.contains(&m.to_uci())).collect()
    }

    /// Remembers `mv` as a mistake in `b` and appends it to the file. False if already known.
    pub fn add(&mut self, b: &Board, mv: Move, note: &str) -> io::Result<bool> {
        let uci = mv.to_uci();
        let entry = self.map.entry(b.hash).or_default();
        if entry.contains(&uci) {
            return Ok(false);
        }
        entry.push(uci.clone());
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut f = fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{:016x} {uci} {} | {note}", b.hash, b.to_fen())?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembers_and_reloads_mistakes() {
        let dir = std::env::temp_dir().join(format!("magician-learn-{}", std::process::id()));
        let path = dir.join("learn.txt");
        let _ = fs::remove_file(&path);
        let b = Board::startpos();
        let m = crate::san::parse_uci(&b, "f2f3").unwrap();
        let mut l = Learn::load(&path);
        assert!(l.add(&b, m, "test").unwrap());
        assert!(!l.add(&b, m, "test").unwrap());
        let again = Learn::load(&path);
        assert_eq!(again.avoid(&b), vec![m]);
        assert!(again.avoid(&b.make_move(m)).is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
