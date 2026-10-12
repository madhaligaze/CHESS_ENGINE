//! Transposition table: one 16-byte slot per index, lock-free. Each slot stores
//! `key ^ data` next to `data`, so a torn write from another thread is detected on read.

use crate::types::Move;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct TtEntry {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: u8,
}

struct Slot {
    check: AtomicU64,
    data: AtomicU64,
}

pub struct TranspositionTable {
    slots: Vec<Slot>,
    age: AtomicU8,
}

fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: u8, age: u8) -> u64 {
    (mv.0 as u64)
        | ((score as i16 as u16 as u64) << 16)
        | ((eval as i16 as u16 as u64) << 32)
        | ((depth.clamp(0, 255) as u64) << 48)
        | ((bound as u64 & 3) << 56)
        | ((age as u64 & 63) << 58)
}

fn unpack(d: u64) -> (TtEntry, u8) {
    let e = TtEntry {
        mv: Move(d as u16),
        score: (d >> 16) as u16 as i16 as i32,
        eval: (d >> 32) as u16 as i16 as i32,
        depth: ((d >> 48) & 255) as i32,
        bound: ((d >> 56) & 3) as u8,
    };
    (e, ((d >> 58) & 63) as u8)
}

impl TranspositionTable {
    pub fn new(mb: usize) -> TranspositionTable {
        let n = (mb.max(1) * 1024 * 1024 / std::mem::size_of::<Slot>()).max(1024);
        let slots = (0..n).map(|_| Slot { check: AtomicU64::new(0), data: AtomicU64::new(0) }).collect();
        TranspositionTable { slots, age: AtomicU8::new(0) }
    }

    pub fn clear(&self) {
        for s in &self.slots {
            s.check.store(0, Relaxed);
            s.data.store(0, Relaxed);
        }
        self.age.store(0, Relaxed);
    }

    pub fn new_search(&self) {
        self.age.store((self.age.load(Relaxed) + 1) & 63, Relaxed);
    }

    #[inline]
    fn age(&self) -> u8 {
        self.age.load(Relaxed)
    }

    #[inline]
    fn index(&self, key: u64) -> usize {
        ((key as u128 * self.slots.len() as u128) >> 64) as usize
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let s = &self.slots[self.index(key)];
        let data = s.data.load(Relaxed);
        if data != 0 && s.check.load(Relaxed) ^ data == key {
            Some(unpack(data).0)
        } else {
            None
        }
    }

    #[inline]
    pub fn store(&self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8) {
        let s = &self.slots[self.index(key)];
        let old = s.data.load(Relaxed);
        let same = s.check.load(Relaxed) ^ old == key;
        let cur_age = self.age();
        if old != 0 {
            let (e, age) = unpack(old);
            // Keep deeper entries of the current search for other positions.
            if !same && age == cur_age && e.depth > depth + 2 && bound != BOUND_EXACT {
                return;
            }
            // Same position: do not lose the best move when the new result has none.
            if same && mv.is_none() && e.mv.is_some() {
                let data = pack(e.mv, score, eval, depth, bound, cur_age);
                s.data.store(data, Relaxed);
                s.check.store(key ^ data, Relaxed);
                return;
            }
        }
        let data = pack(mv, score, eval, depth, bound, cur_age);
        s.data.store(data, Relaxed);
        s.check.store(key ^ data, Relaxed);
    }

    /// Permille of slots written during the current search (sampled).
    pub fn hashfull(&self) -> usize {
        let n = self.slots.len().min(1000);
        self.slots[..n]
            .iter()
            .filter(|s| {
                let d = s.data.load(Relaxed);
                d != 0 && unpack(d).1 == self.age()
            })
            .count()
            * 1000
            / n
    }
}
