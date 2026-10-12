//! Fully legal move generation using check masks and pin rays.

use crate::bitboard::*;
use crate::board::{Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::types::*;

pub const MAX_MOVES: usize = 256;

pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    pub fn new() -> MoveList {
        MoveList { moves: [Move::NONE; MAX_MOVES], len: 0 }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        &mut self.moves[..self.len]
    }

    pub fn contains(&self, m: Move) -> bool {
        self.as_slice().contains(&m)
    }
}

/// All legal moves.
pub fn generate_legal(b: &Board, list: &mut MoveList) {
    generate::<true, true>(b, list);
}

/// Legal captures and queen promotions.
pub fn generate_noisy(b: &Board, list: &mut MoveList) {
    generate::<true, false>(b, list);
}

/// Legal quiet moves, castling and under-promotions.
pub fn generate_quiet(b: &Board, list: &mut MoveList) {
    generate::<false, true>(b, list);
}

pub fn legal_moves(b: &Board) -> MoveList {
    let mut list = MoveList::new();
    generate_legal(b, &mut list);
    list
}

#[inline]
fn shift(x: u64, n: i32) -> u64 {
    if n > 0 {
        x << n
    } else {
        x >> -n
    }
}

#[inline]
fn pin_ok(b: &Board, ksq: usize, from: usize, to: usize) -> bool {
    b.pinned & bb(from) == 0 || line(ksq, from) & bb(to) != 0
}

fn generate<const NOISY: bool, const QUIETS: bool>(b: &Board, list: &mut MoveList) {
    let us = b.stm;
    let them = us ^ 1;
    let occ = b.occupied();
    let our = b.colors[us];
    let their = b.colors[them];
    let ksq = b.king_sq(us);

    // King moves: the destination must not be attacked once the king has left its square.
    let mut ktargets = king_attacks(ksq) & !our;
    if !NOISY {
        ktargets &= !their;
    }
    if !QUIETS {
        ktargets &= their;
    }
    let occ_no_king = occ ^ bb(ksq);
    for to in bits(ktargets) {
        if !b.is_attacked_by(to, them, occ_no_king) {
            let flag = if their & bb(to) != 0 { CAPTURE } else { QUIET };
            list.push(Move::new(ksq, to, flag));
        }
    }

    if more_than_one(b.checkers) {
        return;
    }
    let check_mask = if b.checkers != 0 { between(ksq, lsb(b.checkers)) | b.checkers } else { !0 };

    let mut piece_targets = 0;
    if NOISY {
        piece_targets |= their & check_mask;
    }
    if QUIETS {
        piece_targets |= !occ & check_mask;
    }

    let add = |from: usize, targets: u64, list: &mut MoveList| {
        for to in bits(targets) {
            let flag = if their & bb(to) != 0 { CAPTURE } else { QUIET };
            list.push(Move::new(from, to, flag));
        }
    };

    for from in bits(b.pieces[KNIGHT] & our & !b.pinned) {
        add(from, knight_attacks(from) & piece_targets, list);
    }
    for from in bits((b.pieces[BISHOP] | b.pieces[QUEEN]) & our) {
        let mut t = bishop_attacks(from, occ) & piece_targets;
        if b.pinned & bb(from) != 0 {
            t &= line(ksq, from);
        }
        add(from, t, list);
    }
    for from in bits((b.pieces[ROOK] | b.pieces[QUEEN]) & our) {
        let mut t = rook_attacks(from, occ) & piece_targets;
        if b.pinned & bb(from) != 0 {
            t &= line(ksq, from);
        }
        add(from, t, list);
    }

    // Pawns.
    let pawns = b.pieces[PAWN] & our;
    let (up, left, right, rank7, rank3) = if us == WHITE {
        (8, 7, 9, RANK_7, RANK_3)
    } else {
        (-8, -9, -7, RANK_2, RANK_6)
    };
    let empty = !occ;
    let normal = pawns & !rank7;
    let promoting = pawns & rank7;

    if QUIETS {
        let single = shift(normal, up) & empty;
        let double = shift(single & rank3, up) & empty;
        for to in bits(single & check_mask) {
            let from = (to as i32 - up) as usize;
            if pin_ok(b, ksq, from, to) {
                list.push(Move::new(from, to, QUIET));
            }
        }
        for to in bits(double & check_mask) {
            let from = (to as i32 - 2 * up) as usize;
            if pin_ok(b, ksq, from, to) {
                list.push(Move::new(from, to, DOUBLE_PUSH));
            }
        }
    }

    if NOISY {
        let cap_l = shift(normal & !FILE_A, left) & their & check_mask;
        let cap_r = shift(normal & !FILE_H, right) & their & check_mask;
        for to in bits(cap_l) {
            let from = (to as i32 - left) as usize;
            if pin_ok(b, ksq, from, to) {
                list.push(Move::new(from, to, CAPTURE));
            }
        }
        for to in bits(cap_r) {
            let from = (to as i32 - right) as usize;
            if pin_ok(b, ksq, from, to) {
                list.push(Move::new(from, to, CAPTURE));
            }
        }
        if b.ep != NO_SQ {
            for from in bits(pawn_attacks(them, b.ep) & pawns) {
                if ep_is_legal(b, from, b.ep) {
                    list.push(Move::new(from, b.ep, EP_CAPTURE));
                }
            }
        }
    }

    if promoting != 0 {
        let pushes = shift(promoting, up) & empty & check_mask;
        let cap_l = shift(promoting & !FILE_A, left) & their & check_mask;
        let cap_r = shift(promoting & !FILE_H, right) & their & check_mask;
        let promos = |targets: u64, delta: i32, capture: bool, list: &mut MoveList| {
            let base = if capture { PROMO_CAPTURE } else { PROMO };
            for to in bits(targets) {
                let from = (to as i32 - delta) as usize;
                if !pin_ok(b, ksq, from, to) {
                    continue;
                }
                if NOISY {
                    list.push(Move::new(from, to, base + 3));
                }
                if QUIETS {
                    for n in 0..3 {
                        list.push(Move::new(from, to, base + n));
                    }
                }
            }
        };
        promos(pushes, up, false, list);
        promos(cap_l, left, true, list);
        promos(cap_r, right, true, list);
    }

    if QUIETS && b.checkers == 0 {
        let attacked = |sq: usize| b.is_attacked_by(sq, them, occ);
        if us == WHITE {
            if b.castling & CASTLE_WK != 0 && occ & (bb(5) | bb(6)) == 0 && !attacked(5) && !attacked(6) {
                list.push(Move::new(4, 6, KING_CASTLE));
            }
            if b.castling & CASTLE_WQ != 0
                && occ & (bb(1) | bb(2) | bb(3)) == 0
                && !attacked(3)
                && !attacked(2)
            {
                list.push(Move::new(4, 2, QUEEN_CASTLE));
            }
        } else {
            if b.castling & CASTLE_BK != 0 && occ & (bb(61) | bb(62)) == 0 && !attacked(61) && !attacked(62) {
                list.push(Move::new(60, 62, KING_CASTLE));
            }
            if b.castling & CASTLE_BQ != 0
                && occ & (bb(57) | bb(58) | bb(59)) == 0
                && !attacked(59)
                && !attacked(58)
            {
                list.push(Move::new(60, 58, QUEEN_CASTLE));
            }
        }
    }
}

/// En passant can expose the king along a rank (both pawns leave it), so test it fully.
fn ep_is_legal(b: &Board, from: usize, to: usize) -> bool {
    let us = b.stm;
    let cap = if us == WHITE { to - 8 } else { to + 8 };
    let occ = (b.occupied() ^ bb(from) ^ bb(cap)) | bb(to);
    let ksq = b.king_sq(us);
    b.attackers_to(ksq, occ) & b.colors[us ^ 1] & !bb(cap) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noisy_and_quiet_partition_legal() {
        let fens = [
            crate::board::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];
        for fen in fens {
            let b = Board::from_fen(fen).unwrap();
            let all = legal_moves(&b);
            let mut noisy = MoveList::new();
            let mut quiet = MoveList::new();
            generate_noisy(&b, &mut noisy);
            generate_quiet(&b, &mut quiet);
            assert_eq!(noisy.len() + quiet.len(), all.len(), "{fen}");
            for &m in all.as_slice() {
                assert!(noisy.contains(m) != quiet.contains(m), "{fen} {m}");
            }
            for &m in noisy.as_slice() {
                assert!(m.is_noisy());
            }
        }
    }
}
