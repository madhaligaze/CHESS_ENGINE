"""Export a training checkpoint to the engine's network format (see src/nn.rs).

  python export.py ../nets/base.pt ../nets/base.nn --name base
  python export.py ../nets/base.pt ../nets/test.nn --testvec ../data/bin/elite.bin
      (also writes ../nets/test.vec: records + PyTorch outputs for the Rust parity test)
"""

import argparse
import os

import numpy as np
import torch

from dataset import NUM_PLANES, RecordFile, decode
from model import HumanNet


def fold(conv, bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * scale[:, None, None, None], bn.bias - bn.running_mean * scale


def conv3(w):
    return w.permute(2, 3, 1, 0)  # (out, in, ky, kx) -> (ky, kx, in, out)


def tensors(model):
    out = []
    w, b = fold(model.stem, model.stem_bn)
    out += [conv3(w), b]
    for blk in model.blocks:
        w, b = fold(blk.conv1, blk.bn1)
        out += [conv3(w), b]
        w, b = fold(blk.conv2, blk.bn2)
        out += [conv3(w), b]
        out += [blk.se.fc1.weight, blk.se.fc1.bias, blk.se.fc2.weight, blk.se.fc2.bias]
    out += [model.pq.weight, model.pq.bias, model.pk.weight, model.pk.bias, model.pbias]
    w, b = fold(model.vconv, model.vbn)
    out += [w.reshape(w.shape[0], -1), b]
    out += [model.vfc1.weight, model.vfc1.bias, model.vfc2.weight, model.vfc2.bias]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("output")
    p.add_argument("--name", default=None)
    p.add_argument("--elo", default=None, help="mover,opp ratings the engine should feed (default: training override or 2700,2700)")
    p.add_argument("--testvec", default=None, help="record file to draw parity-test positions from")
    p.add_argument("--n-test", type=int, default=64)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HumanNet(**ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.elo:
        elo = [float(x) for x in args.elo.split(",")]
    elif ckpt.get("elo"):
        elo = list(ckpt["elo"])
    else:
        elo = [2700.0, 2700.0]
    cfg = ckpt["cfg"]
    header = {
        "name": args.name or os.path.splitext(os.path.basename(args.output))[0],
        "planes": NUM_PLANES,
        "channels": cfg["channels"],
        "blocks": cfg["blocks"],
        "se": cfg["se"],
        "policy_dim": cfg["policy_dim"],
        "value_channels": cfg.get("value_channels", 8),
        "value_hidden": cfg.get("value_hidden", 128),
        "mover_elo": elo[0],
        "opp_elo": elo[1],
        "train_step": ckpt.get("step", 0),
    }
    with torch.no_grad(), open(args.output, "wb") as f:
        f.write(b"MGNN1\n")
        f.write(("".join(f"{k}={v}\n" for k, v in header.items()) + "\n").encode())
        for t in tensors(model):
            a = t.detach().float().contiguous().numpy().astype("<f4").ravel()
            f.write(np.uint32(a.size).astype("<u4").tobytes())
            f.write(a.tobytes())
    print(f"wrote {args.output}: {header}")

    if args.testvec:
        rec = RecordFile(args.testvec)
        idx = np.random.default_rng(5).choice(len(rec), args.n_test, replace=False)
        rows = rec.rows(idx)
        planes, mask, target, wdl = decode(rows)
        with torch.no_grad():
            logits, v = model(planes)
        vec_path = os.path.splitext(args.output)[0] + ".vec"
        with open(vec_path, "wb") as f:
            f.write(np.uint32(len(rows)).astype("<u4").tobytes())
            for i in range(len(rows)):
                f.write(rows[i].tobytes())
                f.write(logits[i].numpy().astype("<f4").tobytes())
                f.write(v[i].numpy().astype("<f4").tobytes())
        print(f"wrote {vec_path}: {len(rows)} positions")


if __name__ == "__main__":
    main()
