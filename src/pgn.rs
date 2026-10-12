//! Streaming PGN reader. Keeps tags and the main line; skips comments (except `%clk`
//! clock annotations), variations and NAGs. Tolerates Latin-1 bytes and wrapped lines.

use std::io::BufRead;

pub struct PgnGame {
    pub tags: Vec<(String, String)>,
    pub moves: Vec<String>,
    /// Clock of the player who just moved, in seconds, when the PGN carries `[%clk]`.
    pub clocks: Vec<Option<u32>>,
    pub result: String,
}

impl PgnGame {
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

pub struct PgnReader<R: BufRead> {
    reader: R,
    raw: Vec<u8>,
    pending: Option<String>,
}

impl<R: BufRead> PgnReader<R> {
    pub fn new(reader: R) -> Self {
        PgnReader { reader, raw: Vec::new(), pending: None }
    }

    fn next_line(&mut self) -> Option<String> {
        if let Some(l) = self.pending.take() {
            return Some(l);
        }
        self.raw.clear();
        match self.reader.read_until(b'\n', &mut self.raw) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(String::from_utf8_lossy(&self.raw).trim_end().to_string()),
        }
    }
}

fn parse_tag(line: &str) -> Option<(String, String)> {
    let inner = line.trim().strip_prefix('[')?.strip_suffix(']')?;
    let (key, rest) = inner.split_once(' ')?;
    let value = rest.trim().strip_prefix('"')?.strip_suffix('"')?;
    Some((key.to_string(), value.replace("\\\"", "\"")))
}

fn is_tag_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('[') && t[1..].starts_with(|c: char| c.is_ascii_alphabetic()) && t.trim_end().ends_with(']')
}

impl<R: BufRead> Iterator for PgnReader<R> {
    type Item = PgnGame;

    fn next(&mut self) -> Option<PgnGame> {
        let mut tags = Vec::new();
        let mut movetext = String::new();
        let mut in_comment = false;
        let mut seen_any = false;
        while let Some(line) = self.next_line() {
            if !in_comment && is_tag_line(&line) {
                if !movetext.trim().is_empty() {
                    self.pending = Some(line);
                    break;
                }
                if let Some(t) = parse_tag(&line) {
                    tags.push(t);
                    seen_any = true;
                }
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
            for c in line.chars() {
                match c {
                    '{' => in_comment = true,
                    '}' => in_comment = false,
                    _ => {}
                }
            }
            movetext.push_str(&line);
            movetext.push('\n');
            seen_any = true;
        }
        if !seen_any {
            return None;
        }
        let (moves, clocks, text_result) = parse_movetext(&movetext);
        let result = tags
            .iter()
            .find(|(k, _)| k == "Result")
            .map(|(_, v)| v.clone())
            .unwrap_or(text_result);
        Some(PgnGame { tags, moves, clocks, result })
    }
}

fn parse_clock(comment: &str) -> Option<u32> {
    let i = comment.find("%clk")?;
    let t = comment[i + 4..].trim_start();
    let t = t.split(|c: char| c == ']' || c.is_whitespace()).next()?;
    let mut secs = 0u32;
    for part in t.split(':') {
        let whole = part.split('.').next()?;
        secs = secs * 60 + whole.parse::<u32>().ok()?;
    }
    Some(secs)
}

fn parse_movetext(text: &str) -> (Vec<String>, Vec<Option<u32>>, String) {
    let b = text.as_bytes();
    let mut moves = Vec::new();
    let mut clocks: Vec<Option<u32>> = Vec::new();
    let mut result = "*".to_string();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'{' => {
                let end = text[i..].find('}').map_or(b.len(), |j| i + j);
                if let Some(c) = parse_clock(&text[i + 1..end]) {
                    if let Some(last) = clocks.last_mut() {
                        *last = Some(c);
                    }
                }
                i = end + 1;
            }
            b'(' => {
                let mut depth = 0;
                while i < b.len() {
                    match b[i] {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        b'{' => {
                            while i < b.len() && b[i] != b'}' {
                                i += 1;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                i += 1;
            }
            b';' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'$' => {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            c if c.is_ascii_whitespace() || c == b')' || c == b'}' => i += 1,
            _ => {
                let start = i;
                while i < b.len() && !b[i].is_ascii_whitespace() && !b"{}();".contains(&b[i]) {
                    i += 1;
                }
                let token = &text[start..i];
                if matches!(token, "1-0" | "0-1" | "1/2-1/2" | "*") {
                    result = token.to_string();
                    continue;
                }
                // Strip a leading move number ("12." / "12..." / "1.e4"), but keep "0-0".
                let digits = token.trim_start_matches(|c: char| c.is_ascii_digit());
                let san = if digits.len() != token.len() && digits.starts_with('.') {
                    digits.trim_start_matches('.')
                } else {
                    token
                };
                if !san.is_empty() {
                    moves.push(san.to_string());
                    clocks.push(None);
                }
            }
        }
    }
    (moves, clocks, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_games_with_comments_variations_and_clocks() {
        let pgn = r#"[Event "A"]
[White "Tal, Mihail"]
[Black "X"]
[Result "1-0"]

1.e4 e5 2.Nf3 {good} (2.f4 exf4 (2...d5)) 2...Nc6 $1 3.Bb5 a6 1-0

[Event "B"]
[White "Y"]
[Black "Z"]
[Result "0-1"]

1. d4 { [%clk 0:03:00] } 1... Nf6 { [%clk 0:02:58.5] } 2. O-O-O 0-1
"#;
        let games: Vec<PgnGame> = PgnReader::new(pgn.as_bytes()).collect();
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].moves, vec!["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
        assert_eq!(games[0].tag("White"), Some("Tal, Mihail"));
        assert_eq!(games[0].result, "1-0");
        assert_eq!(games[1].moves, vec!["d4", "Nf6", "O-O-O"]);
        assert_eq!(games[1].clocks, vec![Some(180), Some(178), None]);
        assert_eq!(games[1].result, "0-1");
    }
}
