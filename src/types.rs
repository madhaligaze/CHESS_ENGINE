//! Basic chess types: colors, piece types, squares and the 16-bit move encoding.
//!
//! Squares are numbered a1 = 0, b1 = 1, ..., h8 = 63 (little-endian rank-file).

pub type Bitboard = u64;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;
pub const NO_PIECE: usize = 6;

pub const NO_SQ: usize = 64;

pub const PIECE_CHARS: [char; 6] = ['p', 'n', 'b', 'r', 'q', 'k'];

#[inline]
pub const fn file_of(sq: usize) -> usize {
    sq & 7
}

#[inline]
pub const fn rank_of(sq: usize) -> usize {
    sq >> 3
}

#[inline]
pub const fn make_sq(file: usize, rank: usize) -> usize {
    rank * 8 + file
}

/// Mirror a square vertically (a1 <-> a8). Used to view the board from Black's side.
#[inline]
pub const fn flip_sq(sq: usize) -> usize {
    sq ^ 56
}

pub fn sq_name(sq: usize) -> String {
    if sq >= 64 {
        return "-".to_string();
    }
    let f = (b'a' + file_of(sq) as u8) as char;
    let r = (b'1' + rank_of(sq) as u8) as char;
    format!("{f}{r}")
}

pub fn parse_sq(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let f = b[0].wrapping_sub(b'a');
    let r = b[1].wrapping_sub(b'1');
    if f < 8 && r < 8 {
        Some(make_sq(f as usize, r as usize))
    } else {
        None
    }
}

pub fn piece_from_char(c: char) -> Option<usize> {
    match c.to_ascii_lowercase() {
        'p' => Some(PAWN),
        'n' => Some(KNIGHT),
        'b' => Some(BISHOP),
        'r' => Some(ROOK),
        'q' => Some(QUEEN),
        'k' => Some(KING),
        _ => None,
    }
}

// Move flags (upper 4 bits of the move).
pub const QUIET: u16 = 0;
pub const DOUBLE_PUSH: u16 = 1;
pub const KING_CASTLE: u16 = 2;
pub const QUEEN_CASTLE: u16 = 3;
pub const CAPTURE: u16 = 4;
pub const EP_CAPTURE: u16 = 5;
/// Promotion to (KNIGHT + n) is PROMO + n; with capture it is PROMO_CAPTURE + n.
pub const PROMO: u16 = 8;
pub const PROMO_CAPTURE: u16 = 12;

/// A move packed into 16 bits: from (6) | to (6) | flag (4).
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);

    #[inline]
    pub const fn new(from: usize, to: usize, flag: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (flag << 12))
    }

    #[inline]
    pub const fn from(self) -> usize {
        (self.0 & 63) as usize
    }

    #[inline]
    pub const fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }

    #[inline]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn is_capture(self) -> bool {
        self.flag() & 4 != 0
    }

    #[inline]
    pub const fn is_promo(self) -> bool {
        self.flag() & 8 != 0
    }

    /// Piece type a pawn promotes to. Only meaningful when `is_promo()`.
    #[inline]
    pub const fn promo_piece(self) -> usize {
        (self.flag() & 3) as usize + KNIGHT
    }

    #[inline]
    pub const fn is_castle(self) -> bool {
        matches!(self.flag(), KING_CASTLE | QUEEN_CASTLE)
    }

    #[inline]
    pub const fn is_ep(self) -> bool {
        self.flag() == EP_CAPTURE
    }

    /// Captures and queen promotions: the moves quiescence search looks at.
    #[inline]
    pub const fn is_noisy(self) -> bool {
        self.is_capture() || (self.is_promo() && self.promo_piece() == QUEEN)
    }

    pub fn to_uci(self) -> String {
        if self.is_none() {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", sq_name(self.from()), sq_name(self.to()));
        if self.is_promo() {
            s.push(PIECE_CHARS[self.promo_piece()]);
        }
        s
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}
