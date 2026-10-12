//! Inference for the human move model trained in training/ (see training/model.py).
//!
//! File format (written by training/export.py): the line "MGNN1", then `key=value` header
//! lines ending with an empty line, then tensors as (u32 element count, f32 LE values) in a
//! fixed order. Batch norm is already folded into the convolutions.

use crate::board::Board;
use crate::features::{rel_sq, NetInput, NUM_PLANES};
use crate::movegen::legal_moves;
use crate::types::*;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

struct Block {
    c1_w: Vec<f32>,
    c1_b: Vec<f32>,
    c2_w: Vec<f32>,
    c2_b: Vec<f32>,
    se1_w: Vec<f32>,
    se1_b: Vec<f32>,
    se2_w: Vec<f32>,
    se2_b: Vec<f32>,
}

pub struct Network {
    pub name: String,
    /// Ratings fed to the network (the conditioning it was trained or fine-tuned with).
    pub mover_elo: f32,
    pub opp_elo: f32,
    c: usize,
    se: usize,
    pd: usize,
    vc: usize,
    vh: usize,
    stem_w: Vec<f32>,
    stem_b: Vec<f32>,
    blocks: Vec<Block>,
    pq_w: Vec<f32>,
    pq_b: Vec<f32>,
    pk_w: Vec<f32>,
    pk_b: Vec<f32>,
    pbias: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    v1_w: Vec<f32>,
    v1_b: Vec<f32>,
    v2_w: Vec<f32>,
    v2_b: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct NetEval {
    /// Legal moves with the probability that a (strong) human plays them, sorted descending.
    pub policy: Vec<(Move, f32)>,
    /// Win / draw / loss probabilities for the side to move, as humans would score it.
    pub wdl: [f32; 3],
}

impl NetEval {
    pub fn prob(&self, m: Move) -> f32 {
        self.policy.iter().find(|(pm, _)| *pm == m).map_or(0.0, |(_, p)| *p)
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn tensor(&mut self, expected: usize, what: &str) -> Result<Vec<f32>, String> {
        let header = self.data.get(self.pos..self.pos + 4).ok_or(format!("truncated before {what}"))?;
        let n = u32::from_le_bytes(header.try_into().unwrap()) as usize;
        if n != expected {
            return Err(format!("{what}: expected {expected} values, file has {n}"));
        }
        self.pos += 4;
        let bytes = self.data.get(self.pos..self.pos + 4 * n).ok_or(format!("truncated in {what}"))?;
        self.pos += 4 * n;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// 3x3 convolution with padding 1 over an 8x8 board. `input` is square-major (64 x cin),
/// `w` is laid out (ky, kx, cin, cout), output is square-major (64 x cout).
fn conv3x3(input: &[f32], cin: usize, w: &[f32], bias: &[f32], out: &mut [f32], relu: bool) {
    let cout = bias.len();
    for r in 0..8usize {
        for f in 0..8usize {
            let acc = &mut out[(r * 8 + f) * cout..(r * 8 + f + 1) * cout];
            acc.copy_from_slice(bias);
            for ky in 0..3usize {
                let rr = r + ky;
                if !(1..=8).contains(&rr) {
                    continue;
                }
                for kx in 0..3usize {
                    let ff = f + kx;
                    if !(1..=8).contains(&ff) {
                        continue;
                    }
                    let sq = (rr - 1) * 8 + (ff - 1);
                    let x = &input[sq * cin..(sq + 1) * cin];
                    let wk = &w[(ky * 3 + kx) * cin * cout..(ky * 3 + kx + 1) * cin * cout];
                    for (ci, &xv) in x.iter().enumerate() {
                        if xv == 0.0 {
                            continue;
                        }
                        let wrow = &wk[ci * cout..(ci + 1) * cout];
                        for (a, &wv) in acc.iter_mut().zip(wrow) {
                            *a += xv * wv;
                        }
                    }
                }
            }
            if relu {
                for a in acc.iter_mut() {
                    *a = a.max(0.0);
                }
            }
        }
    }
}

/// y = W x + b for W stored row-major (out x in).
fn linear(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let n_in = x.len();
    for (o, (row, &bias)) in out.iter_mut().zip(w.chunks_exact(n_in).zip(b)) {
        *o = bias + row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
    }
}

impl Network {
    pub fn load(path: &Path) -> Result<Network, String> {
        let mut data = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut data))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if !data.starts_with(b"MGNN1\n") {
            return Err(format!("{}: not a Magician network file", path.display()));
        }
        let end = data
            .windows(2)
            .position(|w| w == b"\n\n")
            .ok_or("network header is not terminated")?;
        let header_text = String::from_utf8_lossy(&data[6..end]).to_string();
        let mut header = HashMap::new();
        for line in header_text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                header.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        let get = |k: &str| -> Result<usize, String> {
            header.get(k).and_then(|v| v.parse().ok()).ok_or(format!("header field '{k}' missing"))
        };
        let (c, nblocks, se, pd, vc, vh) = (
            get("channels")?,
            get("blocks")?,
            get("se")?,
            get("policy_dim")?,
            get("value_channels")?,
            get("value_hidden")?,
        );
        if get("planes")? != NUM_PLANES {
            return Err("network expects a different input encoding".into());
        }
        let mut r = Reader { data: &data, pos: end + 2 };
        let stem_w = r.tensor(9 * NUM_PLANES * c, "stem weights")?;
        let stem_b = r.tensor(c, "stem bias")?;
        let mut blocks = Vec::with_capacity(nblocks);
        for i in 0..nblocks {
            let name = format!("block {i}");
            blocks.push(Block {
                c1_w: r.tensor(9 * c * c, &name)?,
                c1_b: r.tensor(c, &name)?,
                c2_w: r.tensor(9 * c * c, &name)?,
                c2_b: r.tensor(c, &name)?,
                se1_w: r.tensor(se * c, &name)?,
                se1_b: r.tensor(se, &name)?,
                se2_w: r.tensor(2 * c * se, &name)?,
                se2_b: r.tensor(2 * c, &name)?,
            });
        }
        let net = Network {
            name: header.get("name").cloned().unwrap_or_else(|| "unnamed".into()),
            mover_elo: header.get("mover_elo").and_then(|v| v.parse().ok()).unwrap_or(2700.0),
            opp_elo: header.get("opp_elo").and_then(|v| v.parse().ok()).unwrap_or(2700.0),
            c,
            se,
            pd,
            vc,
            vh,
            stem_w,
            stem_b,
            blocks,
            pq_w: r.tensor(pd * c, "policy query")?,
            pq_b: r.tensor(pd, "policy query bias")?,
            pk_w: r.tensor(pd * c, "policy key")?,
            pk_b: r.tensor(pd, "policy key bias")?,
            pbias: r.tensor(4096, "policy bias")?,
            v_w: r.tensor(vc * c, "value conv")?,
            v_b: r.tensor(vc, "value conv bias")?,
            v1_w: r.tensor(vh * vc * 64, "value fc1")?,
            v1_b: r.tensor(vh, "value fc1 bias")?,
            v2_w: r.tensor(3 * vh, "value fc2")?,
            v2_b: r.tensor(3, "value fc2 bias")?,
        };
        if r.pos != data.len() {
            return Err(format!("{} trailing bytes in network file", data.len() - r.pos));
        }
        Ok(net)
    }

    /// Runs the trunk. Returns per-square policy queries and keys (64 x pd each) and WDL logits.
    pub fn forward(&self, planes: &[f32]) -> (Vec<f32>, Vec<f32>, [f32; 3]) {
        let c = self.c;
        // Plane-major input -> square-major.
        let mut x0 = vec![0f32; 64 * NUM_PLANES];
        for p in 0..NUM_PLANES {
            for sq in 0..64 {
                x0[sq * NUM_PLANES + p] = planes[p * 64 + sq];
            }
        }
        let mut x = vec![0f32; 64 * c];
        conv3x3(&x0, NUM_PLANES, &self.stem_w, &self.stem_b, &mut x, true);
        let mut y1 = vec![0f32; 64 * c];
        let mut y2 = vec![0f32; 64 * c];
        let mut pooled = vec![0f32; c];
        let mut hidden = vec![0f32; self.se];
        let mut gates = vec![0f32; 2 * c];
        for blk in &self.blocks {
            conv3x3(&x, c, &blk.c1_w, &blk.c1_b, &mut y1, true);
            conv3x3(&y1, c, &blk.c2_w, &blk.c2_b, &mut y2, false);
            pooled.fill(0.0);
            for sq in 0..64 {
                for (p, v) in pooled.iter_mut().zip(&y2[sq * c..(sq + 1) * c]) {
                    *p += v;
                }
            }
            for p in pooled.iter_mut() {
                *p /= 64.0;
            }
            linear(&blk.se1_w, &blk.se1_b, &pooled, &mut hidden);
            for h in hidden.iter_mut() {
                *h = h.max(0.0);
            }
            linear(&blk.se2_w, &blk.se2_b, &hidden, &mut gates);
            let (scale, shift) = gates.split_at(c);
            for sq in 0..64 {
                let xs = &mut x[sq * c..(sq + 1) * c];
                let ys = &y2[sq * c..(sq + 1) * c];
                for ch in 0..c {
                    let g = 1.0 / (1.0 + (-scale[ch]).exp());
                    xs[ch] = (xs[ch] + ys[ch] * g + shift[ch]).max(0.0);
                }
            }
        }
        let pd = self.pd;
        let mut q = vec![0f32; 64 * pd];
        let mut k = vec![0f32; 64 * pd];
        for sq in 0..64 {
            let t = &x[sq * c..(sq + 1) * c];
            linear(&self.pq_w, &self.pq_b, t, &mut q[sq * pd..(sq + 1) * pd]);
            linear(&self.pk_w, &self.pk_b, t, &mut k[sq * pd..(sq + 1) * pd]);
        }
        // Value head; flatten in PyTorch order (channel-major).
        let mut vflat = vec![0f32; self.vc * 64];
        let mut tmp = vec![0f32; self.vc];
        for sq in 0..64 {
            linear(&self.v_w, &self.v_b, &x[sq * c..(sq + 1) * c], &mut tmp);
            for j in 0..self.vc {
                vflat[j * 64 + sq] = tmp[j].max(0.0);
            }
        }
        let mut vh = vec![0f32; self.vh];
        linear(&self.v1_w, &self.v1_b, &vflat, &mut vh);
        for v in vh.iter_mut() {
            *v = v.max(0.0);
        }
        let mut wdl = [0f32; 3];
        linear(&self.v2_w, &self.v2_b, &vh, &mut wdl);
        (q, k, wdl)
    }

    /// Policy logit of the (relative) from -> to move.
    #[inline]
    pub fn logit(&self, q: &[f32], k: &[f32], from: usize, to: usize) -> f32 {
        let pd = self.pd;
        let dot: f32 = q[from * pd..(from + 1) * pd].iter().zip(&k[to * pd..(to + 1) * pd]).map(|(a, b)| a * b).sum();
        dot / (pd as f32).sqrt() + self.pbias[from * 64 + to]
    }

    pub fn evaluate(&self, b: &Board, last: Move, prev: Move) -> NetEval {
        self.evaluate_with_elo(b, last, prev, self.mover_elo, self.opp_elo)
    }

    pub fn evaluate_with_elo(&self, b: &Board, last: Move, prev: Move, mover_elo: f32, opp_elo: f32) -> NetEval {
        let input = NetInput::new(b, last, prev);
        let mut planes = vec![0f32; NUM_PLANES * 64];
        input.planes(mover_elo, opp_elo, &mut planes);
        let (q, k, wdl_logits) = self.forward(&planes);
        let white = b.stm == WHITE;
        let legal = legal_moves(b);
        let mut policy: Vec<(Move, f32)> = legal
            .as_slice()
            .iter()
            .map(|&m| {
                let mut l = self.logit(&q, &k, rel_sq(m.from(), white), rel_sq(m.to(), white));
                // The policy has one output per (from, to); under-promotions share it.
                if m.is_promo() && m.promo_piece() != QUEEN {
                    l -= 6.0;
                }
                (m, l)
            })
            .collect();
        softmax_in_place(&mut policy);
        policy.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut wdl = wdl_logits;
        let mx = wdl.iter().cloned().fold(f32::MIN, f32::max);
        let sum: f32 = wdl.iter_mut().map(|v| {
            *v = (*v - mx).exp();
            *v
        }).sum();
        for v in wdl.iter_mut() {
            *v /= sum;
        }
        NetEval { policy, wdl: [wdl[2], wdl[1], wdl[0]] }
    }
}

fn softmax_in_place(v: &mut [(Move, f32)]) {
    let mx = v.iter().map(|x| x.1).fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        x.1 = (x.1 - mx).exp();
        sum += x.1;
    }
    for x in v.iter_mut() {
        x.1 /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::NetInput;

    /// training/export.py --testvec writes positions with PyTorch outputs; compare.
    #[test]
    fn matches_pytorch_reference() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("nets");
        let (net_path, vec_path) = (dir.join("test.nn"), dir.join("test.vec"));
        if !net_path.exists() || !vec_path.exists() {
            eprintln!("skipping: run training/export.py --testvec first");
            return;
        }
        let net = Network::load(&net_path).unwrap();
        let data = std::fs::read(&vec_path).unwrap();
        let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let item = 256 + 4096 * 4 + 3 * 4;
        let mut worst = 0f32;
        for i in 0..n {
            let rec = &data[4 + i * item..4 + i * item + 256];
            let floats: Vec<f32> = data[4 + i * item + 256..4 + (i + 1) * item]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let (inp, me, oe) = NetInput::from_record(rec);
            let mut planes = vec![0f32; NUM_PLANES * 64];
            inp.planes(me, oe, &mut planes);
            let (q, k, wdl) = net.forward(&planes);
            for (from, to) in NetInput::record_legal_moves(rec) {
                let ours = net.logit(&q, &k, from, to);
                worst = worst.max((ours - floats[from * 64 + to]).abs());
            }
            for j in 0..3 {
                worst = worst.max((wdl[j] - floats[4096 + j]).abs());
            }
        }
        assert!(worst < 2e-3, "max abs difference vs PyTorch: {worst}");
        eprintln!("{n} positions, max abs difference vs PyTorch {worst:e}");
    }
}
