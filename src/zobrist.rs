//! Zobrist hashing keys, generated at compile time from a fixed seed.

pub struct Zobrist {
    pub pieces: [[[u64; 64]; 6]; 2],
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
    pub side: u64,
}

const fn splitmix(state: u64) -> (u64, u64) {
    let s = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (s, z ^ (z >> 31))
}

const fn build() -> Zobrist {
    let mut z = Zobrist { pieces: [[[0; 64]; 6]; 2], castling: [0; 16], ep_file: [0; 8], side: 0 };
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut q = 0;
            while q < 64 {
                let (ns, v) = splitmix(s);
                s = ns;
                z.pieces[c][p][q] = v;
                q += 1;
            }
            p += 1;
        }
        c += 1;
    }
    let mut i = 0;
    while i < 16 {
        let (ns, v) = splitmix(s);
        s = ns;
        z.castling[i] = if i == 0 { 0 } else { v };
        i += 1;
    }
    let mut f = 0;
    while f < 8 {
        let (ns, v) = splitmix(s);
        s = ns;
        z.ep_file[f] = v;
        f += 1;
    }
    let (_, v) = splitmix(s);
    z.side = v;
    z
}

pub static ZOBRIST: Zobrist = build();
