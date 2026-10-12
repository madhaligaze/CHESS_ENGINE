//! Converts PGN games into fixed-size training records for the human-move model.

use crate::board::Board;
use crate::features::{encode_record, RecordMeta};
use crate::pgn::{PgnGame, PgnReader};
use crate::san::parse_san;
use crate::types::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

pub struct DatagenConfig {
    pub inputs: Vec<String>,
    pub output: String,
    /// Written into each record; lets one file mix sources.
    pub style: u8,
    /// Both players must be rated at least this (ignored when `player` is set).
    pub min_elo: u16,
    /// Only sample the moves of this player (matched on the surname / first name token).
    pub player: Option<String>,
    /// Probability of keeping a position (after the opening).
    pub sample: f64,
    /// Probability of keeping a position in the first 12 plies, which repeat a lot.
    pub opening_sample: f64,
    /// Skip games faster than this class (1 bullet, 2 blitz, 3 rapid, 4 classical).
    pub min_time_class: u8,
    /// Skip moves made with less than this many seconds on the clock (when known).
    pub min_clock: u32,
    pub max_games: usize,
    pub seed: u64,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        DatagenConfig {
            inputs: Vec::new(),
            output: String::new(),
            style: 0,
            min_elo: 0,
            player: None,
            sample: 1.0,
            opening_sample: 1.0,
            min_time_class: 0,
            min_clock: 0,
            max_games: 0,
            seed: 1,
        }
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub games_read: usize,
    pub games_used: usize,
    pub games_filtered: usize,
    pub parse_errors: usize,
    pub positions: usize,
}

/// 0 unknown, 1 bullet, 2 blitz, 3 rapid, 4 classical (Lichess thresholds on base + 40 * inc).
pub fn time_class(tc: Option<&str>) -> u8 {
    let Some(tc) = tc else { return 0 };
    let mut parts = tc.split('+');
    let Some(base) = parts.next().and_then(|s| s.parse::<f64>().ok()) else { return 0 };
    let inc = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let est = base + 40.0 * inc;
    if est < 180.0 {
        1
    } else if est < 480.0 {
        2
    } else if est < 1500.0 {
        3
    } else {
        4
    }
}

pub fn name_matches(name: &str, player: &str) -> bool {
    let first = name.split(|c: char| c == ',' || c == '.' || c.is_whitespace()).next().unwrap_or("");
    first.eq_ignore_ascii_case(player)
}

fn next_f64(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn elo(g: &PgnGame, key: &str) -> u16 {
    g.tag(key).and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

pub fn run(cfg: &DatagenConfig) -> std::io::Result<Stats> {
    let mut out = BufWriter::with_capacity(1 << 20, File::create(&cfg.output)?);
    let mut rng = cfg.seed.max(1);
    let mut stats = Stats::default();
    let mut game_id = 0u32;
    'files: for path in &cfg.inputs {
        let reader = PgnReader::new(BufReader::with_capacity(1 << 20, File::open(path)?));
        for game in reader {
            if cfg.max_games > 0 && stats.games_used >= cfg.max_games {
                break 'files;
            }
            stats.games_read += 1;
            game_id += 1;
            match game_records(&game, cfg, game_id, &mut rng) {
                Ok(Some(records)) => {
                    stats.games_used += 1;
                    stats.positions += records.len();
                    for r in &records {
                        out.write_all(r)?;
                    }
                }
                Ok(None) => stats.games_filtered += 1,
                Err(()) => stats.parse_errors += 1,
            }
            if stats.games_read % 100_000 == 0 {
                eprintln!("  {} games read, {} positions", stats.games_read, stats.positions);
            }
        }
    }
    out.flush()?;
    Ok(stats)
}

type Record = [u8; crate::features::RECORD_SIZE];

fn game_records(g: &PgnGame, cfg: &DatagenConfig, game_id: u32, rng: &mut u64) -> Result<Option<Vec<Record>>, ()> {
    let result: [u8; 2] = match g.result.as_str() {
        "1-0" => [2, 0],
        "0-1" => [0, 2],
        "1/2-1/2" => [1, 1],
        _ => return Ok(None),
    };
    if g.tag("FEN").is_some() || g.tag("Variant").is_some_and(|v| !v.eq_ignore_ascii_case("standard")) {
        return Ok(None);
    }
    let tc = time_class(g.tag("TimeControl"));
    if tc != 0 && tc < cfg.min_time_class {
        return Ok(None);
    }
    let elos = [elo(g, "WhiteElo"), elo(g, "BlackElo")];
    let sample_color = match &cfg.player {
        Some(p) => [
            name_matches(g.tag("White").unwrap_or(""), p),
            name_matches(g.tag("Black").unwrap_or(""), p),
        ],
        None => {
            let ok = elos[0] >= cfg.min_elo && elos[1] >= cfg.min_elo;
            [ok, ok]
        }
    };
    if !sample_color[0] && !sample_color[1] {
        return Ok(None);
    }
    let mut b = Board::startpos();
    let (mut last, mut prev) = (Move::NONE, Move::NONE);
    let mut records = Vec::new();
    for (ply, san) in g.moves.iter().enumerate() {
        let m = parse_san(&b, san).ok_or(())?;
        let color = b.stm;
        if sample_color[color] {
            let clock = if ply >= 2 { g.clocks.get(ply - 2).copied().flatten() } else { None };
            let p = if ply < 12 { cfg.opening_sample } else { cfg.sample };
            let clock_ok = clock.map_or(true, |c| c >= cfg.min_clock);
            if clock_ok && next_f64(rng) < p {
                let meta = RecordMeta {
                    result: result[color],
                    mover_elo: elos[color],
                    opp_elo: elos[color ^ 1],
                    style: cfg.style,
                    time_class: tc,
                    ply: ply as u16,
                    game_id,
                    clock,
                };
                records.push(encode_record(&b, last, prev, m, &meta));
            }
        }
        b = b.make_move(m);
        prev = last;
        last = m;
    }
    Ok(Some(records))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_classes() {
        assert_eq!(time_class(Some("60+0")), 1);
        assert_eq!(time_class(Some("180+2")), 2);
        assert_eq!(time_class(Some("600+0")), 3);
        assert_eq!(time_class(Some("1800+20")), 4);
        assert_eq!(time_class(Some("-")), 0);
        assert_eq!(time_class(None), 0);
    }

    #[test]
    fn player_names() {
        assert!(name_matches("Tal, Mihail", "tal"));
        assert!(name_matches("Tal,M", "Tal"));
        assert!(name_matches("Carlsen,Magnus", "carlsen"));
        assert!(!name_matches("Talukdar, A", "tal"));
        assert!(!name_matches("Vitali, X", "tal"));
    }
}
