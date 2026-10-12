//! Hand-crafted evaluation: PeSTO material and piece-square tables (tapered between middle
//! game and endgame) plus mobility, king safety, pawn structure and a few classic terms.
//!
//! A `Style` skews the terms that define a playing personality: how much the engine values
//! attacking the enemy king, how careful it is with its own, and how much it avoids draws.

use crate::bitboard::*;
use crate::board::Board;
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// Scale (%) of king-attack pressure the engine exerts on the opponent's king.
    pub king_attack: i32,
    /// Scale (%) of the danger the engine sees around its own king.
    pub king_defense: i32,
    /// Scale (%) of piece mobility for the engine's side.
    pub mobility: i32,
    /// Scale (%) of passed pawn bonuses for the engine's side.
    pub passed: i32,
    /// Value of a draw for the engine, in centipawns below zero (positive = avoid draws).
    pub contempt: i32,
}

impl Style {
    pub const NEUTRAL: Style = Style { king_attack: 100, king_defense: 100, mobility: 100, passed: 100, contempt: 0 };
    /// Initiative and attack first; the king's own safety is a secondary concern.
    pub const TAL: Style = Style { king_attack: 145, king_defense: 85, mobility: 120, passed: 100, contempt: 35 };
    /// Universal, solid, squeezes small edges in long endgames and does not settle for draws.
    pub const CARLSEN: Style = Style { king_attack: 100, king_defense: 110, mobility: 105, passed: 120, contempt: 25 };
    pub const HUMAN: Style = Style { king_attack: 110, king_defense: 100, mobility: 100, passed: 100, contempt: 15 };
}

#[derive(Clone, Copy, Debug)]
pub struct EvalParams {
    pub style: Style,
    /// The side the engine is playing: style scaling is asymmetric.
    pub engine_color: usize,
}

impl Default for EvalParams {
    fn default() -> Self {
        EvalParams { style: Style::NEUTRAL, engine_color: WHITE }
    }
}

const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];
const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];

// PeSTO tables (Ronald Friederich), from White's point of view with a8 = index 0.
#[rustfmt::skip]
const MG_PST: [[i32; 64]; 6] = [
    [0, 0, 0, 0, 0, 0, 0, 0,
     98, 134, 61, 95, 68, 126, 34, -11,
     -6, 7, 26, 31, 65, 56, 25, -20,
     -14, 13, 6, 21, 23, 12, 17, -23,
     -27, -2, -5, 12, 17, 6, 10, -25,
     -26, -4, -4, -10, 3, 3, 33, -12,
     -35, -1, -20, -23, -15, 24, 38, -22,
     0, 0, 0, 0, 0, 0, 0, 0],
    [-167, -89, -34, -49, 61, -97, -15, -107,
     -73, -41, 72, 36, 23, 62, 7, -17,
     -47, 60, 37, 65, 84, 129, 73, 44,
     -9, 17, 19, 53, 37, 69, 18, 22,
     -13, 4, 16, 13, 28, 19, 21, -8,
     -23, -9, 12, 10, 19, 17, 25, -16,
     -29, -53, -12, -3, -1, 18, -14, -19,
     -105, -21, -58, -33, -17, -28, -19, -23],
    [-29, 4, -82, -37, -25, -42, 7, -8,
     -26, 16, -18, -13, 30, 59, 18, -47,
     -16, 37, 43, 40, 35, 50, 37, -2,
     -4, 5, 19, 50, 37, 37, 7, -2,
     -6, 13, 13, 26, 34, 12, 10, 4,
     0, 15, 15, 15, 14, 27, 18, 10,
     4, 15, 16, 0, 7, 21, 33, 1,
     -33, -3, -14, -21, -13, -12, -39, -21],
    [32, 42, 32, 51, 63, 9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
     -24, -11, 7, 26, 24, 35, -8, -20,
     -36, -26, -12, -1, 9, -7, 6, -23,
     -45, -25, -16, -17, 3, 0, -5, -33,
     -44, -16, -20, -9, -1, 11, -6, -71,
     -19, -13, 1, 17, 16, 7, -37, -26],
    [-28, 0, 29, 12, 59, 44, 43, 45,
     -24, -39, -5, 1, -16, 57, 28, 54,
     -13, -17, 7, 8, 29, 56, 47, 57,
     -27, -27, -16, -16, -1, 17, -2, 1,
     -9, -26, -9, -10, -2, -4, 3, -3,
     -14, 2, -11, -2, -5, 2, 14, 5,
     -35, -8, 11, 2, 8, 15, -3, 1,
     -1, -18, -9, 10, -15, -25, -31, -50],
    [-65, 23, 16, -15, -56, -34, 2, 13,
     29, -1, -20, -7, -8, -4, -38, -29,
     -9, 24, 2, -16, -20, 6, 22, -22,
     -17, -20, -12, -27, -30, -25, -14, -36,
     -49, -1, -27, -39, -46, -44, -33, -51,
     -14, -14, -22, -46, -44, -30, -15, -27,
     1, 7, -8, -64, -43, -16, 9, 8,
     -15, 36, 12, -54, 8, -28, 24, 14],
];

#[rustfmt::skip]
const EG_PST: [[i32; 64]; 6] = [
    [0, 0, 0, 0, 0, 0, 0, 0,
     178, 173, 158, 134, 147, 132, 165, 187,
     94, 100, 85, 67, 56, 53, 82, 84,
     32, 24, 13, 5, -2, 4, 17, 17,
     13, 9, -3, -7, -7, -8, 3, -1,
     4, 7, -6, 1, 0, -5, -1, -8,
     13, 8, 8, 10, 13, 0, 2, -7,
     0, 0, 0, 0, 0, 0, 0, 0],
    [-58, -38, -13, -28, -31, -27, -63, -99,
     -25, -8, -25, -2, -9, -25, -24, -52,
     -24, -20, 10, 9, -1, -9, -19, -41,
     -17, 3, 22, 22, 22, 11, 8, -18,
     -18, -6, 16, 25, 16, 17, 4, -18,
     -23, -3, -1, 15, 10, -3, -20, -22,
     -42, -20, -10, -5, -2, -20, -23, -44,
     -29, -51, -23, -15, -22, -18, -50, -64],
    [-14, -21, -11, -8, -7, -9, -17, -24,
     -8, -4, 7, -12, -3, -13, -4, -14,
     2, -8, 0, -1, -2, 6, 0, 4,
     -3, 9, 12, 9, 14, 10, 3, 2,
     -6, 3, 13, 19, 7, 10, -3, -9,
     -12, -3, 8, 10, 13, 3, -7, -15,
     -14, -18, -7, -1, 4, -9, -15, -27,
     -23, -9, -23, -5, -9, -16, -5, -17],
    [13, 10, 18, 15, 12, 12, 8, 5,
     11, 13, 13, 11, -3, 3, 8, 3,
     7, 7, 7, 5, 4, -3, -5, -3,
     4, 3, 13, 1, 2, 1, -1, 2,
     3, 5, 8, 4, -5, -6, -8, -11,
     -4, 0, -5, -1, -7, -12, -8, -16,
     -6, -6, 0, 2, -9, -9, -11, -3,
     -9, 2, 3, -1, -5, -13, 4, -20],
    [-9, 22, 22, 27, 27, 19, 10, 20,
     -17, 20, 32, 41, 58, 25, 30, 0,
     -20, 6, 9, 49, 47, 35, 19, 9,
     3, 22, 24, 45, 57, 40, 57, 36,
     -18, 28, 19, 47, 31, 34, 39, 23,
     -16, -27, 15, 6, 9, 17, 10, 5,
     -22, -23, -30, -16, -16, -23, -36, -32,
     -33, -28, -22, -43, -5, -32, -20, -41],
    [-74, -35, -18, -18, -11, 15, 4, -17,
     -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26, 3,
     -18, -4, 21, 24, 27, 23, 9, -11,
     -19, -3, 11, 21, 23, 16, 7, -9,
     -27, -11, 4, 13, 14, 4, -5, -17,
     -53, -34, -21, -11, -28, -14, -24, -43],
];

// Mobility per reachable square (centred on a typical count so material stays calibrated).
const MOB_MG: [i32; 6] = [0, 4, 5, 2, 1, 0];
const MOB_EG: [i32; 6] = [0, 4, 5, 4, 2, 0];
const MOB_CENTER: [i32; 6] = [0, 4, 6, 6, 12, 0];

// King attack units per attacked square of the king zone.
const ATTACK_UNITS: [i32; 6] = [0, 2, 2, 3, 5, 0];

const PASSED_MG: [i32; 8] = [0, 0, 0, 5, 10, 20, 30, 0];
const PASSED_EG: [i32; 8] = [0, 5, 10, 20, 35, 55, 80, 0];
const DOUBLED: (i32, i32) = (-10, -20);
const ISOLATED: (i32, i32) = (-10, -12);
const BISHOP_PAIR: (i32, i32) = (25, 50);
const ROOK_OPEN: (i32, i32) = (25, 10);
const ROOK_SEMI_OPEN: (i32, i32) = (12, 6);
const SHIELD: i32 = 10;
const TEMPO: i32 = 12;

#[inline]
fn adjacent_files(sq: usize) -> u64 {
    let f = file_bb(sq);
    ((f << 1) & !FILE_A) | ((f >> 1) & !FILE_H)
}

/// Squares in front of `sq` (from `color`'s view) on its file and the adjacent files.
fn front_span(color: usize, sq: usize) -> u64 {
    let files = file_bb(sq) | adjacent_files(sq);
    let r = rank_of(sq);
    let ahead = if color == WHITE {
        if r == 7 {
            0
        } else {
            !0u64 << (8 * (r + 1))
        }
    } else if r == 0 {
        0
    } else {
        !0u64 >> (8 * (8 - r))
    };
    files & ahead
}

#[inline]
fn relative_rank(color: usize, sq: usize) -> usize {
    if color == WHITE {
        rank_of(sq)
    } else {
        7 - rank_of(sq)
    }
}

/// Smooth quadratic ramp of king danger, capped (about the shape of the classic CPW table).
#[inline]
fn king_danger(units: i32) -> i32 {
    (units * units / 8 + units).min(500)
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Breakdown {
    pub material_pst: (i32, i32),
    pub pawns: (i32, i32),
    pub mobility: (i32, i32),
    pub king: (i32, i32),
    pub pieces: (i32, i32),
    pub phase: i32,
    pub scale: i32,
}

/// Static evaluation in centipawns from the side to move's point of view.
pub fn evaluate(b: &Board, p: &EvalParams) -> i32 {
    evaluate_full(b, p, None)
}

pub fn evaluate_full(b: &Board, p: &EvalParams, mut breakdown: Option<&mut Breakdown>) -> i32 {
    let style = &p.style;
    // Scale for terms that favour `color`: style applies to the engine's side only.
    let own = |color: usize, pct: i32, v: i32| if color == p.engine_color { v * pct / 100 } else { v };

    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut phase = 0;
    for c in 0..2 {
        for pt in 0..6 {
            for sq in bits(b.piece_bb(c, pt)) {
                let idx = if c == WHITE { sq ^ 56 } else { sq };
                mg[c] += MG_VALUE[pt] + MG_PST[pt][idx];
                eg[c] += EG_VALUE[pt] + EG_PST[pt][idx];
                phase += PHASE_INC[pt];
            }
        }
    }
    let mat = (mg[0] - mg[1], eg[0] - eg[1]);

    // Pawn structure.
    let pawns = [b.piece_bb(WHITE, PAWN), b.piece_bb(BLACK, PAWN)];
    let mut pawn_term = [(0i32, 0i32); 2];
    for c in 0..2 {
        let (ours, theirs) = (pawns[c], pawns[c ^ 1]);
        for sq in bits(ours) {
            if ours & adjacent_files(sq) == 0 {
                pawn_term[c].0 += ISOLATED.0;
                pawn_term[c].1 += ISOLATED.1;
            }
            if front_span(c, sq) & theirs == 0 && front_span(c, sq) & file_bb(sq) & ours == 0 {
                let r = relative_rank(c, sq);
                pawn_term[c].0 += own(c, style.passed, PASSED_MG[r]);
                pawn_term[c].1 += own(c, style.passed, PASSED_EG[r]);
            }
        }
        for f in 0..8 {
            if popcnt(ours & (FILE_A << f)) > 1 {
                pawn_term[c].0 += DOUBLED.0;
                pawn_term[c].1 += DOUBLED.1;
            }
        }
    }

    // Mobility, king attacks, piece terms.
    let occ = b.occupied();
    let pawn_attacks_of = |c: usize| {
        let p = pawns[c];
        if c == WHITE {
            ((p << 7) & !FILE_H) | ((p << 9) & !FILE_A)
        } else {
            ((p >> 9) & !FILE_H) | ((p >> 7) & !FILE_A)
        }
    };
    let pawn_att = [pawn_attacks_of(WHITE), pawn_attacks_of(BLACK)];
    let mut mob = [(0i32, 0i32); 2];
    let mut king = [(0i32, 0i32); 2];
    let mut pieces = [(0i32, 0i32); 2];
    for c in 0..2 {
        let them = c ^ 1;
        let area = !b.colors[c] & !pawn_att[them];
        let eksq = b.king_sq(them);
        let zone = king_attacks(eksq) | bb(eksq);
        let mut units = 0;
        let mut attackers = 0;
        for pt in KNIGHT..=QUEEN {
            for sq in bits(b.piece_bb(c, pt)) {
                let att = piece_attacks(pt, sq, occ);
                let n = popcnt(att & area) as i32 - MOB_CENTER[pt];
                mob[c].0 += own(c, style.mobility, MOB_MG[pt] * n);
                mob[c].1 += own(c, style.mobility, MOB_EG[pt] * n);
                let z = att & zone;
                if z != 0 {
                    attackers += 1;
                    units += ATTACK_UNITS[pt] * popcnt(z) as i32;
                }
                if pt == ROOK {
                    let file = file_bb(sq);
                    if file & (pawns[0] | pawns[1]) == 0 {
                        pieces[c].0 += ROOK_OPEN.0;
                        pieces[c].1 += ROOK_OPEN.1;
                    } else if file & pawns[c] == 0 {
                        pieces[c].0 += ROOK_SEMI_OPEN.0;
                        pieces[c].1 += ROOK_SEMI_OPEN.1;
                    }
                }
            }
        }
        // Attacking the enemy king only counts with a queen or several attackers.
        if attackers >= 2 && (b.piece_bb(c, QUEEN) != 0 || attackers >= 3) {
            let danger = king_danger(units);
            // Attack on `them`'s king: our aggression if we are the engine, else their caution.
            let pct = if c == p.engine_color { style.king_attack } else { style.king_defense };
            king[c].0 += danger * pct / 100;
        }
        if popcnt(b.piece_bb(c, BISHOP)) >= 2 {
            pieces[c].0 += BISHOP_PAIR.0;
            pieces[c].1 += BISHOP_PAIR.1;
        }
        // Pawn shield in front of a castled king.
        let ksq = b.king_sq(c);
        if relative_rank(c, ksq) <= 1 && !matches!(file_of(ksq), 3 | 4) {
            let r = rank_of(ksq);
            let ranks = if c == WHITE {
                (RANK_1 << (8 * (r + 1))) | (RANK_1 << (8 * (r + 2)))
            } else {
                (RANK_1 << (8 * (r - 1))) | (RANK_1 << (8 * (r - 2)))
            };
            let zone = (file_bb(ksq) | adjacent_files(ksq)) & ranks;
            let shield = popcnt(zone & pawns[c]).min(3) as i32;
            king[c].0 += own(c, style.king_defense, SHIELD * shield);
        }
    }

    let sum = |t: [(i32, i32); 2]| (t[0].0 - t[1].0, t[0].1 - t[1].1);
    let (pw, mb, kg, pc) = (sum(pawn_term), sum(mob), sum(king), sum(pieces));
    let mg_total = mat.0 + pw.0 + mb.0 + kg.0 + pc.0;
    let eg_total = mat.1 + pw.1 + mb.1 + kg.1 + pc.1;

    let scale = endgame_scale(b, eg_total);
    let ph = phase.min(24);
    let score = (mg_total * ph + eg_total * (24 - ph)) / 24 * scale / 128;

    if let Some(bd) = breakdown.as_deref_mut() {
        *bd = Breakdown { material_pst: mat, pawns: pw, mobility: mb, king: kg, pieces: pc, phase: ph, scale };
    }
    let stm_score = if b.stm == WHITE { score } else { -score };
    stm_score + TEMPO
}

/// Out of 128: shrink scores of endgames that cannot realistically be converted.
fn endgame_scale(b: &Board, eg: i32) -> i32 {
    let strong = if eg > 0 { WHITE } else { BLACK };
    let weak = strong ^ 1;
    let npm = |c: usize| {
        300 * popcnt(b.piece_bb(c, KNIGHT) | b.piece_bb(c, BISHOP)) as i32
            + 500 * popcnt(b.piece_bb(c, ROOK)) as i32
            + 900 * popcnt(b.piece_bb(c, QUEEN)) as i32
    };
    if b.piece_bb(strong, PAWN) == 0 && npm(strong) - npm(weak) <= 300 {
        return if npm(strong) < 500 { 0 } else { 16 };
    }
    // Opposite-coloured bishops with nothing else but pawns.
    let only_bishops = |c: usize| npm(c) == 300 && b.piece_bb(c, BISHOP) != 0;
    if only_bishops(WHITE) && only_bishops(BLACK) {
        let wb = b.piece_bb(WHITE, BISHOP);
        let bb_ = b.piece_bb(BLACK, BISHOP);
        if (wb & LIGHT_SQUARES != 0) != (bb_ & LIGHT_SQUARES != 0) {
            return 64;
        }
    }
    128
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mirror_fen(fen: &str) -> String {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let ranks: Vec<String> = parts[0]
            .split('/')
            .rev()
            .map(|r| r.chars().map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() }).collect())
            .collect();
        let stm = if parts[1] == "w" { "b" } else { "w" };
        let castling: String = if parts[2] == "-" {
            "-".into()
        } else {
            let mut c: Vec<char> = parts[2]
                .chars()
                .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
                .collect();
            c.sort_by_key(|ch| "KQkq".find(*ch).unwrap());
            c.into_iter().collect()
        };
        format!("{} {stm} {castling} - 0 1", ranks.join("/"))
    }

    #[test]
    fn evaluation_is_color_symmetric() {
        let fens = [
            crate::board::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "6k1/5ppp/8/8/8/8/1B3PPP/6K1 w - - 0 1",
        ];
        let p = EvalParams::default();
        for fen in fens {
            let a = Board::from_fen(fen).unwrap();
            let m = Board::from_fen(&mirror_fen(fen)).unwrap();
            let mp = EvalParams { engine_color: BLACK, ..p };
            assert_eq!(evaluate(&a, &p), evaluate(&m, &mp), "{fen}");
        }
    }

    #[test]
    fn sane_values() {
        let p = EvalParams::default();
        let start = evaluate(&Board::startpos(), &p);
        assert!(start.abs() < 40, "start {start}");
        let up_a_queen = Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(evaluate(&up_a_queen, &p) > 800);
        let kbk = Board::from_fen("8/8/4k3/8/8/2B5/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&kbk, &p).abs() <= TEMPO + 5);
    }
}
