//! Neural-network input encoding, shared by training-data generation and inference so the
//! two can never disagree.
//!
//! Everything is seen from the side to move: when Black is to move the board is mirrored
//! vertically, so "own" pawns always advance up the board and a move is indexed as
//! `relative_from * 64 + relative_to`.
//!
//! The training record layout (256 bytes, little-endian) is documented in
//! training/dataset.py, which is its only reader.

use crate::bitboard::*;
use crate::board::{Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::movegen::legal_moves;
use crate::types::*;

pub const NUM_PLANES: usize = 26;
pub const RECORD_SIZE: usize = 256;
pub const NONE_SQ: u8 = 255;

/// Ratings are fed to the network as (elo - ELO_CENTER) / ELO_SCALE.
pub const ELO_CENTER: f32 = 2500.0;
pub const ELO_SCALE: f32 = 400.0;

#[inline]
pub fn rel_sq(sq: usize, white: bool) -> usize {
    if white {
        sq
    } else {
        sq ^ 56
    }
}

#[inline]
pub fn rel_bb(b: u64, white: bool) -> u64 {
    if white {
        b
    } else {
        b.swap_bytes()
    }
}

/// Policy index of a move from the mover's point of view.
#[inline]
pub fn policy_index(m: Move, white: bool) -> usize {
    rel_sq(m.from(), white) * 64 + rel_sq(m.to(), white)
}

/// The position as the network sees it.
#[derive(Clone, Debug)]
pub struct NetInput {
    /// Own pawn..king, then opponent pawn..king, in relative orientation.
    pub pieces: [u64; 12],
    pub last_from: u8,
    pub last_to: u8,
    pub prev_from: u8,
    pub prev_to: u8,
    /// bit0 own O-O, bit1 own O-O-O, bit2 opponent O-O, bit3 opponent O-O-O.
    pub castling: u8,
    pub ep: u8,
    pub halfmove: u8,
    pub stm_white: bool,
}

fn rel_move_squares(m: Move, white: bool) -> (u8, u8) {
    if m.is_none() {
        (NONE_SQ, NONE_SQ)
    } else {
        (rel_sq(m.from(), white) as u8, rel_sq(m.to(), white) as u8)
    }
}

impl NetInput {
    /// `last` is the opponent's previous move, `prev` our own move before that.
    pub fn new(b: &Board, last: Move, prev: Move) -> NetInput {
        let white = b.stm == WHITE;
        let (us, them) = (b.stm, b.stm ^ 1);
        let mut pieces = [0u64; 12];
        for pt in 0..6 {
            pieces[pt] = rel_bb(b.piece_bb(us, pt), white);
            pieces[6 + pt] = rel_bb(b.piece_bb(them, pt), white);
        }
        let (last_from, last_to) = rel_move_squares(last, white);
        let (prev_from, prev_to) = rel_move_squares(prev, white);
        let (ok, oq, tk, tq) = if white {
            (CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ)
        } else {
            (CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ)
        };
        let mut castling = 0;
        for (i, flag) in [ok, oq, tk, tq].into_iter().enumerate() {
            if b.castling & flag != 0 {
                castling |= 1 << i;
            }
        }
        NetInput {
            pieces,
            last_from,
            last_to,
            prev_from,
            prev_to,
            castling,
            ep: if b.ep == NO_SQ { NONE_SQ } else { rel_sq(b.ep, white) as u8 },
            halfmove: b.halfmove.min(255) as u8,
            stm_white: white,
        }
    }

    /// Decodes a training record; also returns the mover's and opponent's rating as fed to
    /// the network (unknown ratings become ELO_CENTER, as in training/dataset.py).
    pub fn from_record(r: &[u8]) -> (NetInput, f32, f32) {
        let mut pieces = [0u64; 12];
        for (i, p) in pieces.iter_mut().enumerate() {
            *p = u64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().unwrap());
        }
        let inp = NetInput {
            pieces,
            last_from: r[224],
            last_to: r[225],
            prev_from: r[226],
            prev_to: r[227],
            castling: r[228],
            ep: r[229],
            halfmove: r[230],
            stm_white: r[231] & 1 != 0,
        };
        let elo = |lo: usize| {
            let e = u16::from_le_bytes([r[lo], r[lo + 1]]) as f32;
            if e > 0.0 {
                e
            } else {
                ELO_CENTER
            }
        };
        (inp, elo(236), elo(238))
    }

    /// Legal (relative from, relative to) pairs stored in a training record.
    pub fn record_legal_moves(r: &[u8]) -> Vec<(usize, usize)> {
        let own = (0..6).fold(0u64, |acc, i| acc | u64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().unwrap()));
        let mut out = Vec::new();
        for (i, from) in bits(own).enumerate().take(16) {
            let tos = u64::from_le_bytes(r[96 + i * 8..104 + i * 8].try_into().unwrap());
            out.extend(bits(tos).map(|to| (from, to)));
        }
        out
    }

    /// Fills `out` (NUM_PLANES * 64 floats, plane-major, square = rank * 8 + file).
    pub fn planes(&self, mover_elo: f32, opp_elo: f32, out: &mut [f32]) {
        assert_eq!(out.len(), NUM_PLANES * 64);
        out.fill(0.0);
        let mut set_plane = |p: usize, v: f32| out[p * 64..(p + 1) * 64].fill(v);
        if self.castling & 1 != 0 {
            set_plane(16, 1.0);
        }
        if self.castling & 2 != 0 {
            set_plane(17, 1.0);
        }
        if self.castling & 4 != 0 {
            set_plane(18, 1.0);
        }
        if self.castling & 8 != 0 {
            set_plane(19, 1.0);
        }
        if self.stm_white {
            set_plane(21, 1.0);
        }
        set_plane(22, self.halfmove as f32 / 100.0);
        set_plane(23, 1.0);
        set_plane(24, (mover_elo - ELO_CENTER) / ELO_SCALE);
        set_plane(25, (opp_elo - ELO_CENTER) / ELO_SCALE);
        for (p, &b) in self.pieces.iter().enumerate() {
            for sq in bits(b) {
                out[p * 64 + sq] = 1.0;
            }
        }
        for (p, sq) in [(12, self.last_from), (13, self.last_to), (14, self.prev_from), (15, self.prev_to), (20, self.ep)] {
            if sq != NONE_SQ {
                out[p * 64 + sq as usize] = 1.0;
            }
        }
    }
}

/// Destination bitboards of the legal moves, one slot per own piece in ascending relative
/// square order (the order of set bits in the union of own piece bitboards).
pub fn legal_slots(b: &Board) -> [u64; 16] {
    let white = b.stm == WHITE;
    let mut tos = [0u64; 64];
    for &m in legal_moves(b).as_slice() {
        tos[rel_sq(m.from(), white)] |= bb(rel_sq(m.to(), white));
    }
    let mut slots = [0u64; 16];
    for (i, sq) in bits(rel_bb(b.colors[b.stm], white)).enumerate().take(16) {
        slots[i] = tos[sq];
    }
    slots
}

pub struct RecordMeta {
    /// 0 = loss, 1 = draw, 2 = win, from the mover's point of view.
    pub result: u8,
    pub mover_elo: u16,
    pub opp_elo: u16,
    pub style: u8,
    pub time_class: u8,
    pub ply: u16,
    pub game_id: u32,
    pub clock: Option<u32>,
}

pub fn encode_record(b: &Board, last: Move, prev: Move, played: Move, meta: &RecordMeta) -> [u8; RECORD_SIZE] {
    let inp = NetInput::new(b, last, prev);
    let white = inp.stm_white;
    let mut r = [0u8; RECORD_SIZE];
    for (i, p) in inp.pieces.iter().enumerate() {
        r[i * 8..i * 8 + 8].copy_from_slice(&p.to_le_bytes());
    }
    for (i, s) in legal_slots(b).iter().enumerate() {
        r[96 + i * 8..104 + i * 8].copy_from_slice(&s.to_le_bytes());
    }
    r[224] = inp.last_from;
    r[225] = inp.last_to;
    r[226] = inp.prev_from;
    r[227] = inp.prev_to;
    r[228] = inp.castling;
    r[229] = inp.ep;
    r[230] = inp.halfmove;
    r[231] = white as u8;
    r[232] = rel_sq(played.from(), white) as u8;
    r[233] = rel_sq(played.to(), white) as u8;
    r[234] = if played.is_promo() { played.promo_piece() as u8 } else { 0 };
    r[235] = meta.result;
    r[236..238].copy_from_slice(&meta.mover_elo.to_le_bytes());
    r[238..240].copy_from_slice(&meta.opp_elo.to_le_bytes());
    r[240] = meta.style;
    r[241] = meta.time_class;
    r[242..244].copy_from_slice(&meta.ply.to_le_bytes());
    r[244..248].copy_from_slice(&meta.game_id.to_le_bytes());
    let clock = meta.clock.map_or(u16::MAX, |c| c.min(u16::MAX as u32 - 1) as u16);
    r[248..250].copy_from_slice(&clock.to_le_bytes());
    r
}
