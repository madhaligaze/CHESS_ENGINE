"""Reader for the 256-byte training records written by `magician datagen` (src/features.rs).

Everything is from the mover's point of view (board mirrored when Black is to move).

Record layout (little-endian):
  off  size
  0    96   12 x u64 piece bitboards: own P N B R Q K, then opponent P N B R Q K
  96   128  16 x u64 legal destination bitboards, one per own piece in ascending square order
  224  1    last_from  opponent's previous move (255 = none)
  225  1    last_to
  226  1    prev_from  mover's own previous move
  227  1    prev_to
  228  1    castling   bit0 own O-O, bit1 own O-O-O, bit2 opp O-O, bit3 opp O-O-O
  229  1    en passant square (255 = none)
  230  1    halfmove clock
  231  1    flags      bit0 = White to move
  232  1    played move: from
  233  1    played move: to
  234  1    promotion piece (0 none, 1 N, 2 B, 3 R, 4 Q)
  235  1    game result for the mover (0 loss, 1 draw, 2 win)
  236  2    mover Elo (0 = unknown)
  238  2    opponent Elo
  240  1    style id   241 1 time class   242 2 ply
  244  4    game id    248 2 mover clock in seconds (65535 = unknown)
"""

import queue
import threading

import numpy as np
import torch

RECORD = 256
NUM_PLANES = 26
ELO_CENTER = 2500.0
ELO_SCALE = 400.0


class RecordFile:
    def __init__(self, path, val_every=64):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, RECORD)
        game_id = np.ascontiguousarray(self.data[:, 244:248]).view("<u4").ravel()
        is_val = (game_id % val_every) == 0
        self.train_idx = np.nonzero(~is_val)[0]
        self.val_idx = np.nonzero(is_val)[0]

    def __len__(self):
        return self.data.shape[0]

    def rows(self, idx):
        idx = np.sort(idx)
        return np.asarray(self.data[idx])


def decode(rows, elo_override=None):
    """(B, 256) uint8 -> planes (B,26,8,8) f32, legal mask (B,4096) bool, target (B,) i64, wdl (B,) i64."""
    b = rows.shape[0]
    ar = np.arange(b)
    pieces = np.unpackbits(rows[:, :96], axis=1, bitorder="little").reshape(b, 12, 64)
    planes = np.zeros((b, NUM_PLANES, 64), dtype=np.float32)
    planes[:, :12] = pieces
    for plane, col in ((12, 224), (13, 225), (14, 226), (15, 227), (20, 229)):
        sq = rows[:, col].astype(np.int64)
        ok = sq < 64
        planes[ar[ok], plane, sq[ok]] = 1.0
    castling = rows[:, 228]
    for i in range(4):
        planes[:, 16 + i] = ((castling >> i) & 1)[:, None]
    planes[:, 21] = (rows[:, 231] & 1)[:, None]
    planes[:, 22] = (rows[:, 230].astype(np.float32) / np.float32(100.0))[:, None]
    planes[:, 23] = 1.0
    if elo_override is not None:
        me = np.full(b, elo_override[0], dtype=np.float32)
        oe = np.full(b, elo_override[1], dtype=np.float32)
    else:
        me = np.ascontiguousarray(rows[:, 236:238]).view("<u2").ravel().astype(np.float32)
        oe = np.ascontiguousarray(rows[:, 238:240]).view("<u2").ravel().astype(np.float32)
        me = np.where(me > 0, me, np.float32(ELO_CENTER))
        oe = np.where(oe > 0, oe, np.float32(ELO_CENTER))
    planes[:, 24] = ((me - np.float32(ELO_CENTER)) / np.float32(ELO_SCALE))[:, None]
    planes[:, 25] = ((oe - np.float32(ELO_CENTER)) / np.float32(ELO_SCALE))[:, None]

    slots = np.unpackbits(rows[:, 96:224], axis=1, bitorder="little").reshape(b, 16, 64)
    own = pieces[:, :6].max(axis=1)
    slot_of_sq = np.clip(np.cumsum(own, axis=1, dtype=np.int64) - 1, 0, 15)
    mask = np.take_along_axis(slots, slot_of_sq[:, :, None], axis=1) * own[:, :, None]

    target = rows[:, 232].astype(np.int64) * 64 + rows[:, 233].astype(np.int64)
    wdl = rows[:, 235].astype(np.int64)
    return (
        torch.from_numpy(planes).view(b, NUM_PLANES, 8, 8),
        torch.from_numpy(mask.reshape(b, 4096).astype(bool)),
        torch.from_numpy(target),
        torch.from_numpy(wdl),
    )


class Sampler:
    """Draws training batches from several record files with mixing weights, one pass
    (shuffled) per file before reshuffling. Decoding happens on a background thread."""

    def __init__(self, sources, batch, seed=1, elo_override=None, prefetch=4):
        self.sources = sources  # list of (RecordFile, weight)
        self.batch = batch
        self.elo_override = elo_override
        self.rng = np.random.default_rng(seed)
        w = np.array([wt for _, wt in sources], dtype=np.float64)
        self.probs = w / w.sum()
        self.perms = [self.rng.permutation(f.train_idx) for f, _ in sources]
        self.pos = [0] * len(sources)
        self.q = queue.Queue(maxsize=prefetch)
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _take(self, i, n):
        out = []
        while n > 0:
            perm = self.perms[i]
            k = min(n, len(perm) - self.pos[i])
            out.append(perm[self.pos[i] : self.pos[i] + k])
            self.pos[i] += k
            n -= k
            if self.pos[i] >= len(perm):
                self.perms[i] = self.rng.permutation(self.sources[i][0].train_idx)
                self.pos[i] = 0
        return np.concatenate(out)

    def _run(self):
        while not self.stop:
            counts = self.rng.multinomial(self.batch, self.probs)
            rows = [f.rows(self._take(i, c)) for i, ((f, _), c) in enumerate(zip(self.sources, counts)) if c > 0]
            self.q.put(decode(np.concatenate(rows), self.elo_override))

    def next(self):
        return self.q.get()


def val_batches(f, batch, n_batches, elo_override=None, seed=123):
    rng = np.random.default_rng(seed)
    idx = f.val_idx
    if len(idx) > batch * n_batches:
        idx = rng.choice(idx, batch * n_batches, replace=False)
    idx = np.sort(idx)
    return [decode(f.rows(idx[i : i + batch]), elo_override) for i in range(0, len(idx), batch)]
