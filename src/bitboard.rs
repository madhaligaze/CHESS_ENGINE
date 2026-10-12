//! Bitboard helpers and precomputed attack tables (magic bitboards for sliders).

use crate::types::*;
use std::sync::LazyLock;

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_B: u64 = FILE_A << 1;
pub const FILE_G: u64 = FILE_A << 6;
pub const FILE_H: u64 = FILE_A << 7;
pub const RANK_1: u64 = 0xFF;
pub const RANK_2: u64 = RANK_1 << 8;
pub const RANK_3: u64 = RANK_1 << 16;
pub const RANK_4: u64 = RANK_1 << 24;
pub const RANK_5: u64 = RANK_1 << 32;
pub const RANK_6: u64 = RANK_1 << 40;
pub const RANK_7: u64 = RANK_1 << 48;
pub const RANK_8: u64 = RANK_1 << 56;

pub const LIGHT_SQUARES: u64 = 0x55AA_55AA_55AA_55AA;

#[inline]
pub const fn bb(sq: usize) -> u64 {
    1u64 << sq
}

#[inline]
pub fn lsb(b: u64) -> usize {
    b.trailing_zeros() as usize
}

#[inline]
pub fn msb(b: u64) -> usize {
    63 - b.leading_zeros() as usize
}

#[inline]
pub fn pop_lsb(b: &mut u64) -> usize {
    let s = lsb(*b);
    *b &= *b - 1;
    s
}

#[inline]
pub fn popcnt(b: u64) -> u32 {
    b.count_ones()
}

#[inline]
pub fn more_than_one(b: u64) -> bool {
    b & b.wrapping_sub(1) != 0
}

#[inline]
pub const fn file_bb(sq: usize) -> u64 {
    FILE_A << (sq & 7)
}

#[inline]
pub const fn rank_bb(sq: usize) -> u64 {
    RANK_1 << (sq & 56)
}

/// Iterator over the set squares of a bitboard, lowest first.
pub struct Bits(pub u64);

impl Iterator for Bits {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(pop_lsb(&mut self.0))
        }
    }
}

#[inline]
pub fn bits(b: u64) -> Bits {
    Bits(b)
}

/// Shift a bitboard "up" from the point of view of `color` (towards rank 8 for White).
#[inline]
pub const fn shift_up(b: u64, color: usize) -> u64 {
    if color == WHITE {
        b << 8
    } else {
        b >> 8
    }
}

#[derive(Clone, Copy, Default)]
struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: u32,
}

pub struct Tables {
    knight: [u64; 64],
    king: [u64; 64],
    pawn: [[u64; 64]; 2],
    rook: [Magic; 64],
    bishop: [Magic; 64],
    slider: Vec<u64>,
    between: Vec<u64>,
    line: Vec<u64>,
}

static TABLES: LazyLock<Tables> = LazyLock::new(Tables::init);

/// Force table construction (otherwise it happens on first use).
pub fn init() {
    LazyLock::force(&TABLES);
}

#[inline]
pub fn knight_attacks(sq: usize) -> u64 {
    TABLES.knight[sq]
}

#[inline]
pub fn king_attacks(sq: usize) -> u64 {
    TABLES.king[sq]
}

/// Squares attacked by a pawn of `color` standing on `sq`.
#[inline]
pub fn pawn_attacks(color: usize, sq: usize) -> u64 {
    TABLES.pawn[color][sq]
}

#[inline]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    let t = &*TABLES;
    let m = &t.bishop[sq];
    let idx = m.offset as usize + ((occ & m.mask).wrapping_mul(m.magic) >> m.shift) as usize;
    unsafe { *t.slider.get_unchecked(idx) }
}

#[inline]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    let t = &*TABLES;
    let m = &t.rook[sq];
    let idx = m.offset as usize + ((occ & m.mask).wrapping_mul(m.magic) >> m.shift) as usize;
    unsafe { *t.slider.get_unchecked(idx) }
}

#[inline]
pub fn queen_attacks(sq: usize, occ: u64) -> u64 {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

/// Attacks of a non-pawn piece type.
#[inline]
pub fn piece_attacks(pt: usize, sq: usize, occ: u64) -> u64 {
    match pt {
        KNIGHT => knight_attacks(sq),
        BISHOP => bishop_attacks(sq, occ),
        ROOK => rook_attacks(sq, occ),
        QUEEN => queen_attacks(sq, occ),
        KING => king_attacks(sq),
        _ => 0,
    }
}

/// Squares strictly between `a` and `b` if they share a rank, file or diagonal; else 0.
#[inline]
pub fn between(a: usize, b: usize) -> u64 {
    TABLES.between[a * 64 + b]
}

/// The whole line (edge to edge) through `a` and `b` if aligned; else 0.
#[inline]
pub fn line(a: usize, b: usize) -> u64 {
    TABLES.line[a * 64 + b]
}

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn on_board(f: i32, r: i32) -> bool {
    (0..8).contains(&f) && (0..8).contains(&r)
}

fn step_attacks(sq: usize, offsets: &[(i32, i32)]) -> u64 {
    let (f0, r0) = (file_of(sq) as i32, rank_of(sq) as i32);
    let mut att = 0;
    for &(df, dr) in offsets {
        let (f, r) = (f0 + df, r0 + dr);
        if on_board(f, r) {
            att |= bb((r * 8 + f) as usize);
        }
    }
    att
}

fn slide(sq: usize, occ: u64, dirs: &[(i32, i32)]) -> u64 {
    let (f0, r0) = (file_of(sq) as i32, rank_of(sq) as i32);
    let mut att = 0;
    for &(df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        while on_board(f, r) {
            let s = (r * 8 + f) as usize;
            att |= bb(s);
            if occ & bb(s) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    att
}

/// Occupancy bits that matter for a slider: the rays without their last (edge) square.
fn relevant_mask(sq: usize, dirs: &[(i32, i32)]) -> u64 {
    let (f0, r0) = (file_of(sq) as i32, rank_of(sq) as i32);
    let mut m = 0;
    for &(df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        while on_board(f + df, r + dr) {
            m |= bb((r * 8 + f) as usize);
            f += df;
            r += dr;
        }
    }
    m
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    s.wrapping_mul(2_685_821_657_736_338_717)
}

fn find_magic(sq: usize, dirs: &[(i32, i32)], rng: &mut u64, slider: &mut Vec<u64>) -> Magic {
    let mask = relevant_mask(sq, dirs);
    let nbits = popcnt(mask);
    let n = 1usize << nbits;
    let mut occs = Vec::with_capacity(n);
    let mut atts = Vec::with_capacity(n);
    let mut sub = 0u64;
    loop {
        occs.push(sub);
        atts.push(slide(sq, sub, dirs));
        sub = sub.wrapping_sub(mask) & mask;
        if sub == 0 {
            break;
        }
    }
    let shift = 64 - nbits;
    let mut table = vec![0u64; n];
    let mut epoch = vec![0u32; n];
    let mut attempt = 0u32;
    loop {
        let magic = xorshift(rng) & xorshift(rng) & xorshift(rng);
        if popcnt(mask.wrapping_mul(magic) & 0xFF00_0000_0000_0000) < 6 {
            continue;
        }
        attempt += 1;
        let mut ok = true;
        for i in 0..n {
            let idx = (occs[i].wrapping_mul(magic) >> shift) as usize;
            if epoch[idx] < attempt {
                epoch[idx] = attempt;
                table[idx] = atts[i];
            } else if table[idx] != atts[i] {
                ok = false;
                break;
            }
        }
        if ok {
            let offset = slider.len() as u32;
            slider.extend_from_slice(&table);
            return Magic { mask, magic, shift, offset };
        }
    }
}

impl Tables {
    fn init() -> Tables {
        let mut t = Tables {
            knight: [0; 64],
            king: [0; 64],
            pawn: [[0; 64]; 2],
            rook: [Magic::default(); 64],
            bishop: [Magic::default(); 64],
            slider: Vec::with_capacity(107_648),
            between: vec![0; 64 * 64],
            line: vec![0; 64 * 64],
        };
        const KNIGHT_OFFS: [(i32, i32); 8] =
            [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
        const KING_OFFS: [(i32, i32); 8] =
            [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
        for sq in 0..64 {
            t.knight[sq] = step_attacks(sq, &KNIGHT_OFFS);
            t.king[sq] = step_attacks(sq, &KING_OFFS);
            t.pawn[WHITE][sq] = step_attacks(sq, &[(-1, 1), (1, 1)]);
            t.pawn[BLACK][sq] = step_attacks(sq, &[(-1, -1), (1, -1)]);
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for sq in 0..64 {
            t.rook[sq] = find_magic(sq, &ROOK_DIRS, &mut rng, &mut t.slider);
        }
        for sq in 0..64 {
            t.bishop[sq] = find_magic(sq, &BISHOP_DIRS, &mut rng, &mut t.slider);
        }
        for a in 0..64 {
            for b in 0..64 {
                if a == b {
                    continue;
                }
                for dirs in [&ROOK_DIRS, &BISHOP_DIRS] {
                    if slide(a, 0, dirs) & bb(b) != 0 {
                        t.between[a * 64 + b] = slide(a, bb(b), dirs) & slide(b, bb(a), dirs);
                        t.line[a * 64 + b] = (slide(a, 0, dirs) & slide(b, 0, dirs)) | bb(a) | bb(b);
                    }
                }
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_lookups_match_ray_walk() {
        let mut rng = 12345u64;
        for _ in 0..20_000 {
            let occ = xorshift(&mut rng) & xorshift(&mut rng);
            let sq = (xorshift(&mut rng) % 64) as usize;
            assert_eq!(rook_attacks(sq, occ), slide(sq, occ, &ROOK_DIRS));
            assert_eq!(bishop_attacks(sq, occ), slide(sq, occ, &BISHOP_DIRS));
        }
    }

    #[test]
    fn between_and_line() {
        // a1..h8 diagonal
        assert_eq!(between(0, 63), bb(9) | bb(18) | bb(27) | bb(36) | bb(45) | bb(54));
        assert_eq!(between(0, 1), 0);
        assert_eq!(between(0, 10), 0); // a1-c2: not aligned
        assert_eq!(line(0, 7), RANK_1);
        assert_eq!(line(8, 16), FILE_A);
    }
}
