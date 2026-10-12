//! Board representation: bitboards plus a mailbox, FEN I/O, copy-make move execution,
//! attack queries and static exchange evaluation.

use crate::bitboard::*;
use crate::types::*;
use crate::zobrist::ZOBRIST;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

const fn castle_masks() -> [u8; 64] {
    let mut m = [15u8; 64];
    m[0] = 15 & !CASTLE_WQ;
    m[4] = 15 & !(CASTLE_WK | CASTLE_WQ);
    m[7] = 15 & !CASTLE_WK;
    m[56] = 15 & !CASTLE_BQ;
    m[60] = 15 & !(CASTLE_BK | CASTLE_BQ);
    m[63] = 15 & !CASTLE_BK;
    m
}

/// Castling rights that survive a move touching a square.
static CASTLE_MASK: [u8; 64] = castle_masks();

/// Piece values used by static exchange evaluation.
pub const SEE_VALUES: [i32; 7] = [100, 300, 300, 500, 900, 0, 0];

#[derive(Clone)]
pub struct Board {
    pub pieces: [u64; 6],
    pub colors: [u64; 2],
    pub mailbox: [u8; 64],
    pub stm: usize,
    pub castling: u8,
    /// En passant target square, set only when a capture there is pseudo-legal; else NO_SQ.
    pub ep: usize,
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
    /// Enemy pieces giving check to the side to move.
    pub checkers: u64,
    /// Pieces of the side to move pinned to their own king.
    pub pinned: u64,
}

impl Board {
    fn empty() -> Board {
        Board {
            pieces: [0; 6],
            colors: [0; 2],
            mailbox: [NO_PIECE as u8; 64],
            stm: WHITE,
            castling: 0,
            ep: NO_SQ,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            checkers: 0,
            pinned: 0,
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen(START_FEN).expect("start FEN is valid")
    }

    #[inline]
    pub fn occupied(&self) -> u64 {
        self.colors[WHITE] | self.colors[BLACK]
    }

    #[inline]
    pub fn piece_bb(&self, color: usize, pt: usize) -> u64 {
        self.pieces[pt] & self.colors[color]
    }

    #[inline]
    pub fn piece_on(&self, sq: usize) -> usize {
        self.mailbox[sq] as usize
    }

    #[inline]
    pub fn color_on(&self, sq: usize) -> usize {
        if self.colors[WHITE] & bb(sq) != 0 {
            WHITE
        } else {
            BLACK
        }
    }

    #[inline]
    pub fn king_sq(&self, color: usize) -> usize {
        lsb(self.pieces[KING] & self.colors[color])
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }

    #[inline]
    pub fn has_non_pawn_material(&self, color: usize) -> bool {
        self.colors[color] & !(self.pieces[PAWN] | self.pieces[KING]) != 0
    }

    #[inline]
    fn add_piece(&mut self, color: usize, pt: usize, sq: usize) {
        self.pieces[pt] |= bb(sq);
        self.colors[color] |= bb(sq);
        self.mailbox[sq] = pt as u8;
        self.hash ^= ZOBRIST.pieces[color][pt][sq];
    }

    #[inline]
    fn remove_piece(&mut self, color: usize, pt: usize, sq: usize) {
        self.pieces[pt] &= !bb(sq);
        self.colors[color] &= !bb(sq);
        self.mailbox[sq] = NO_PIECE as u8;
        self.hash ^= ZOBRIST.pieces[color][pt][sq];
    }

    #[inline]
    fn move_piece(&mut self, color: usize, pt: usize, from: usize, to: usize) {
        let m = bb(from) | bb(to);
        self.pieces[pt] ^= m;
        self.colors[color] ^= m;
        self.mailbox[from] = NO_PIECE as u8;
        self.mailbox[to] = pt as u8;
        self.hash ^= ZOBRIST.pieces[color][pt][from] ^ ZOBRIST.pieces[color][pt][to];
    }

    /// All pieces (both colors) attacking `sq` given occupancy `occ`.
    #[inline]
    pub fn attackers_to(&self, sq: usize, occ: u64) -> u64 {
        (pawn_attacks(WHITE, sq) & self.piece_bb(BLACK, PAWN))
            | (pawn_attacks(BLACK, sq) & self.piece_bb(WHITE, PAWN))
            | (knight_attacks(sq) & self.pieces[KNIGHT])
            | (king_attacks(sq) & self.pieces[KING])
            | (bishop_attacks(sq, occ) & (self.pieces[BISHOP] | self.pieces[QUEEN]))
            | (rook_attacks(sq, occ) & (self.pieces[ROOK] | self.pieces[QUEEN]))
    }

    #[inline]
    pub fn is_attacked_by(&self, sq: usize, by: usize, occ: u64) -> bool {
        let them = self.colors[by];
        (pawn_attacks(by ^ 1, sq) & self.pieces[PAWN] & them) != 0
            || (knight_attacks(sq) & self.pieces[KNIGHT] & them) != 0
            || (king_attacks(sq) & self.pieces[KING] & them) != 0
            || (bishop_attacks(sq, occ) & (self.pieces[BISHOP] | self.pieces[QUEEN]) & them) != 0
            || (rook_attacks(sq, occ) & (self.pieces[ROOK] | self.pieces[QUEEN]) & them) != 0
    }

    fn update_check_info(&mut self) {
        let us = self.stm;
        let them = us ^ 1;
        let ksq = self.king_sq(us);
        let occ = self.occupied();
        self.checkers = self.attackers_to(ksq, occ) & self.colors[them];
        let snipers = ((rook_attacks(ksq, 0) & (self.pieces[ROOK] | self.pieces[QUEEN]))
            | (bishop_attacks(ksq, 0) & (self.pieces[BISHOP] | self.pieces[QUEEN])))
            & self.colors[them];
        let mut pinned = 0;
        for s in bits(snipers) {
            let blockers = between(ksq, s) & occ;
            if blockers != 0 && !more_than_one(blockers) && blockers & self.colors[us] != 0 {
                pinned |= blockers;
            }
        }
        self.pinned = pinned;
    }

    /// Returns the position after the (legal) move `m`.
    pub fn make_move(&self, m: Move) -> Board {
        let mut b = self.clone();
        let us = self.stm;
        let them = us ^ 1;
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let pt = self.piece_on(from);
        debug_assert!(pt != NO_PIECE, "no piece on {} for move {m} in {}", sq_name(from), self.to_fen());

        if b.ep != NO_SQ {
            b.hash ^= ZOBRIST.ep_file[file_of(b.ep)];
            b.ep = NO_SQ;
        }
        b.halfmove += 1;
        if us == BLACK {
            b.fullmove += 1;
        }

        if flag == EP_CAPTURE {
            let cap = if us == WHITE { to - 8 } else { to + 8 };
            b.remove_piece(them, PAWN, cap);
        } else if m.is_capture() {
            b.remove_piece(them, self.piece_on(to), to);
        }
        if m.is_capture() || pt == PAWN {
            b.halfmove = 0;
        }

        b.move_piece(us, pt, from, to);

        match flag {
            DOUBLE_PUSH => {
                let ep = (from + to) / 2;
                if pawn_attacks(us, ep) & b.piece_bb(them, PAWN) != 0 {
                    b.ep = ep;
                    b.hash ^= ZOBRIST.ep_file[file_of(ep)];
                }
            }
            KING_CASTLE => {
                let (rf, rt) = if us == WHITE { (7, 5) } else { (63, 61) };
                b.move_piece(us, ROOK, rf, rt);
            }
            QUEEN_CASTLE => {
                let (rf, rt) = if us == WHITE { (0, 3) } else { (56, 59) };
                b.move_piece(us, ROOK, rf, rt);
            }
            _ => {}
        }

        if m.is_promo() {
            b.remove_piece(us, PAWN, to);
            b.add_piece(us, m.promo_piece(), to);
        }

        let nc = b.castling & CASTLE_MASK[from] & CASTLE_MASK[to];
        if nc != b.castling {
            b.hash ^= ZOBRIST.castling[b.castling as usize] ^ ZOBRIST.castling[nc as usize];
            b.castling = nc;
        }

        b.stm = them;
        b.hash ^= ZOBRIST.side;
        b.update_check_info();
        b
    }

    /// Pass the move to the opponent (null-move pruning). Must not be called in check.
    pub fn make_null(&self) -> Board {
        let mut b = self.clone();
        if b.ep != NO_SQ {
            b.hash ^= ZOBRIST.ep_file[file_of(b.ep)];
            b.ep = NO_SQ;
        }
        b.halfmove += 1;
        b.stm ^= 1;
        b.hash ^= ZOBRIST.side;
        b.update_check_info();
        b
    }

    pub fn compute_hash(&self) -> u64 {
        let mut h = 0;
        for c in 0..2 {
            for pt in 0..6 {
                for sq in bits(self.piece_bb(c, pt)) {
                    h ^= ZOBRIST.pieces[c][pt][sq];
                }
            }
        }
        h ^= ZOBRIST.castling[self.castling as usize];
        if self.ep != NO_SQ {
            h ^= ZOBRIST.ep_file[file_of(self.ep)];
        }
        if self.stm == BLACK {
            h ^= ZOBRIST.side;
        }
        h
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("FEN has too few fields: '{fen}'"));
        }
        let mut b = Board::empty();
        let (mut rank, mut file) = (7i32, 0i32);
        for c in parts[0].chars() {
            match c {
                '/' => {
                    rank -= 1;
                    file = 0;
                    if rank < 0 {
                        return Err(format!("FEN has too many ranks: '{fen}'"));
                    }
                }
                '1'..='8' => file += c as i32 - '0' as i32,
                _ => {
                    let pt = piece_from_char(c).ok_or_else(|| format!("bad FEN piece '{c}'"))?;
                    if !(0..8).contains(&file) {
                        return Err(format!("FEN rank overflow: '{fen}'"));
                    }
                    let color = if c.is_ascii_uppercase() { WHITE } else { BLACK };
                    b.add_piece(color, pt, make_sq(file as usize, rank as usize));
                    file += 1;
                }
            }
        }
        b.stm = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            s => return Err(format!("bad side to move '{s}'")),
        };
        if let Some(c) = parts.get(2) {
            for ch in c.chars() {
                b.castling |= match ch {
                    'K' => CASTLE_WK,
                    'Q' => CASTLE_WQ,
                    'k' => CASTLE_BK,
                    'q' => CASTLE_BQ,
                    _ => 0,
                };
            }
        }
        // Drop rights that the piece placement cannot support.
        let has = |b: &Board, c: usize, pt: usize, sq: usize| b.piece_bb(c, pt) & bb(sq) != 0;
        if !has(&b, WHITE, KING, 4) {
            b.castling &= !(CASTLE_WK | CASTLE_WQ);
        }
        if !has(&b, WHITE, ROOK, 7) {
            b.castling &= !CASTLE_WK;
        }
        if !has(&b, WHITE, ROOK, 0) {
            b.castling &= !CASTLE_WQ;
        }
        if !has(&b, BLACK, KING, 60) {
            b.castling &= !(CASTLE_BK | CASTLE_BQ);
        }
        if !has(&b, BLACK, ROOK, 63) {
            b.castling &= !CASTLE_BK;
        }
        if !has(&b, BLACK, ROOK, 56) {
            b.castling &= !CASTLE_BQ;
        }
        if let Some(e) = parts.get(3) {
            if *e != "-" {
                let sq = parse_sq(e).ok_or_else(|| format!("bad en passant square '{e}'"))?;
                if pawn_attacks(b.stm ^ 1, sq) & b.piece_bb(b.stm, PAWN) != 0 {
                    b.ep = sq;
                }
            }
        }
        b.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        b.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);

        if popcnt(b.piece_bb(WHITE, KING)) != 1 || popcnt(b.piece_bb(BLACK, KING)) != 1 {
            return Err(format!("FEN must have exactly one king per side: '{fen}'"));
        }
        b.hash = b.compute_hash();
        b.update_check_info();
        if b.is_attacked_by(b.king_sq(b.stm ^ 1), b.stm, b.occupied()) {
            return Err(format!("side not to move is in check: '{fen}'"));
        }
        Ok(b)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = make_sq(file, rank);
                let pt = self.piece_on(sq);
                if pt == NO_PIECE {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    s.push_str(&empty.to_string());
                    empty = 0;
                }
                let c = PIECE_CHARS[pt];
                s.push(if self.color_on(sq) == WHITE { c.to_ascii_uppercase() } else { c });
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push_str(if self.stm == WHITE { " w " } else { " b " });
        if self.castling == 0 {
            s.push('-');
        } else {
            for (flag, ch) in [(CASTLE_WK, 'K'), (CASTLE_WQ, 'Q'), (CASTLE_BK, 'k'), (CASTLE_BQ, 'q')] {
                if self.castling & flag != 0 {
                    s.push(ch);
                }
            }
        }
        s.push(' ');
        s.push_str(&sq_name(self.ep));
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    /// Positions where neither side can possibly mate.
    pub fn is_insufficient_material(&self) -> bool {
        if self.pieces[PAWN] | self.pieces[ROOK] | self.pieces[QUEEN] != 0 {
            return false;
        }
        let minors = self.pieces[KNIGHT] | self.pieces[BISHOP];
        if popcnt(minors) <= 1 {
            return true;
        }
        if self.pieces[KNIGHT] == 0 {
            let b = self.pieces[BISHOP];
            return b & LIGHT_SQUARES == 0 || b & !LIGHT_SQUARES == 0;
        }
        false
    }

    /// Static exchange evaluation: does `m` win at least `threshold` centipawns
    /// once all captures on the destination square are played out?
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        if m.is_castle() || m.is_ep() || m.is_promo() {
            return 0 >= threshold;
        }
        let (from, to) = (m.from(), m.to());
        let mut swap = SEE_VALUES[self.piece_on(to)] - threshold;
        if swap < 0 {
            return false;
        }
        swap = SEE_VALUES[self.piece_on(from)] - swap;
        if swap <= 0 {
            return true;
        }
        let mut occ = self.occupied() ^ bb(from) ^ bb(to);
        let mut stm = self.stm;
        let mut attackers = self.attackers_to(to, occ);
        let diag = self.pieces[BISHOP] | self.pieces[QUEEN];
        let orth = self.pieces[ROOK] | self.pieces[QUEEN];
        let mut res = 1;
        loop {
            stm ^= 1;
            attackers &= occ;
            let stm_att = attackers & self.colors[stm];
            if stm_att == 0 {
                break;
            }
            res ^= 1;
            let pick = |pt: usize| stm_att & self.pieces[pt];
            let attacker = (PAWN..=KING).find(|&pt| pick(pt) != 0).unwrap();
            if attacker == KING {
                return if attackers & !self.colors[stm] != 0 { res ^ 1 != 0 } else { res != 0 };
            }
            swap = SEE_VALUES[attacker] - swap;
            if swap < res {
                break;
            }
            let b = pick(attacker);
            occ ^= b & b.wrapping_neg();
            if matches!(attacker, PAWN | BISHOP | QUEEN) {
                attackers |= bishop_attacks(to, occ) & diag;
            }
            if matches!(attacker, ROOK | QUEEN) {
                attackers |= rook_attacks(to, occ) & orth;
            }
        }
        res != 0
    }

    pub fn pretty(&self) -> String {
        let mut s = String::from("\n +---+---+---+---+---+---+---+---+\n");
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = make_sq(file, rank);
                let pt = self.piece_on(sq);
                let c = if pt == NO_PIECE {
                    ' '
                } else if self.color_on(sq) == WHITE {
                    PIECE_CHARS[pt].to_ascii_uppercase()
                } else {
                    PIECE_CHARS[pt]
                };
                s.push_str(&format!(" | {c}"));
            }
            s.push_str(&format!(" | {}\n +---+---+---+---+---+---+---+---+\n", rank + 1));
        }
        s.push_str("   a   b   c   d   e   f   g   h\n\n");
        s.push_str(&format!("Fen: {}\nKey: {:016X}\n", self.to_fen(), self.hash));
        s
    }
}
