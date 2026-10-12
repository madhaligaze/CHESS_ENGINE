//! Standard algebraic notation: parsing (for PGN input) and formatting (for output).

use crate::board::Board;
use crate::movegen::legal_moves;
use crate::types::*;

pub fn to_san(b: &Board, m: Move) -> String {
    let mut s = String::new();
    if m.is_castle() {
        s.push_str(if m.flag() == KING_CASTLE { "O-O" } else { "O-O-O" });
    } else {
        let pt = b.piece_on(m.from());
        if pt == PAWN {
            if m.is_capture() {
                s.push((b'a' + file_of(m.from()) as u8) as char);
                s.push('x');
            }
            s.push_str(&sq_name(m.to()));
            if m.is_promo() {
                s.push('=');
                s.push(PIECE_CHARS[m.promo_piece()].to_ascii_uppercase());
            }
        } else {
            s.push(PIECE_CHARS[pt].to_ascii_uppercase());
            let legal = legal_moves(b);
            let rivals: Vec<usize> = legal
                .as_slice()
                .iter()
                .filter(|o| o.to() == m.to() && o.from() != m.from() && b.piece_on(o.from()) == pt)
                .map(|o| o.from())
                .collect();
            if !rivals.is_empty() {
                let file = (b'a' + file_of(m.from()) as u8) as char;
                let rank = (b'1' + rank_of(m.from()) as u8) as char;
                if !rivals.iter().any(|&f| file_of(f) == file_of(m.from())) {
                    s.push(file);
                } else if !rivals.iter().any(|&f| rank_of(f) == rank_of(m.from())) {
                    s.push(rank);
                } else {
                    s.push(file);
                    s.push(rank);
                }
            }
            if m.is_capture() {
                s.push('x');
            }
            s.push_str(&sq_name(m.to()));
        }
    }
    let after = b.make_move(m);
    if after.in_check() {
        s.push(if legal_moves(&after).is_empty() { '#' } else { '+' });
    }
    s
}

/// Finds the legal move written as `san`. Returns None for illegal or ambiguous input.
pub fn parse_san(b: &Board, san: &str) -> Option<Move> {
    let s = san.trim_end_matches(['+', '#', '!', '?']);
    let legal = legal_moves(b);
    let castle = |flag: u16| legal.as_slice().iter().copied().find(|m| m.flag() == flag);
    match s {
        "O-O" | "0-0" => return castle(KING_CASTLE),
        "O-O-O" | "0-0-0" => return castle(QUEEN_CASTLE),
        _ => {}
    }
    let mut body = s;
    let mut promo = None;
    if let Some(i) = body.find('=') {
        promo = Some(piece_from_char(body[i + 1..].chars().next()?)?);
        body = &body[..i];
    } else if body.len() >= 3 && body.ends_with(['N', 'B', 'R', 'Q']) {
        promo = piece_from_char(body.chars().last()?);
        body = &body[..body.len() - 1];
    }
    let (pt, rest) = match body.chars().next()? {
        c @ ('N' | 'B' | 'R' | 'Q' | 'K') => (piece_from_char(c)?, &body[1..]),
        _ => (PAWN, body),
    };
    let rest: String = rest.chars().filter(|&c| c != 'x' && c != '-' && c != ':').collect();
    if rest.len() < 2 || !rest.is_ascii() {
        return None;
    }
    let to = parse_sq(&rest[rest.len() - 2..])?;
    let (mut dfile, mut drank) = (None, None);
    for c in rest[..rest.len() - 2].chars() {
        match c {
            'a'..='h' => dfile = Some(c as usize - 'a' as usize),
            '1'..='8' => drank = Some(c as usize - '1' as usize),
            _ => return None,
        }
    }
    let mut found = None;
    for &m in legal.as_slice() {
        if m.to() != to || m.is_castle() || b.piece_on(m.from()) != pt {
            continue;
        }
        if dfile.is_some_and(|f| file_of(m.from()) != f) || drank.is_some_and(|r| rank_of(m.from()) != r) {
            continue;
        }
        let promo_ok = match promo {
            Some(p) => m.is_promo() && m.promo_piece() == p,
            None => !m.is_promo() || m.promo_piece() == QUEEN,
        };
        if !promo_ok {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(m);
    }
    found
}

/// Parses a move in UCI long algebraic form ("e2e4", "e7e8q").
pub fn parse_uci(b: &Board, s: &str) -> Option<Move> {
    let legal = legal_moves(b);
    legal.as_slice().iter().copied().find(|m| m.to_uci() == s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn san_round_trip_on_random_games() {
        let mut rng = 0x9E37_79B9u64;
        for _ in 0..40 {
            let mut b = Board::startpos();
            for _ in 0..120 {
                let legal = legal_moves(&b);
                if legal.is_empty() {
                    break;
                }
                for &m in legal.as_slice() {
                    let san = to_san(&b, m);
                    assert_eq!(parse_san(&b, &san), Some(m), "{san} in {}", b.to_fen());
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                b = b.make_move(legal.as_slice()[(rng % legal.len() as u64) as usize]);
            }
        }
    }

    #[test]
    fn san_variants() {
        let b = Board::from_fen("r3k2r/1P6/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        assert_eq!(parse_san(&b, "O-O").unwrap().to_uci(), "e1g1");
        assert_eq!(parse_san(&b, "0-0-0").unwrap().to_uci(), "e1c1");
        assert_eq!(parse_san(&b, "bxa8=Q+").unwrap().to_uci(), "b7a8q");
        assert_eq!(parse_san(&b, "bxa8N").unwrap().to_uci(), "b7a8n");
        assert_eq!(parse_san(&b, "b8=R").unwrap().to_uci(), "b7b8r");
        assert_eq!(parse_san(&b, "Rxa8").unwrap().to_uci(), "a1a8");
        assert!(parse_san(&b, "Nf3").is_none());
    }
}
