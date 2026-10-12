"""Train (or fine-tune) the human move model on records from `magician datagen`.

Base model:
  python train.py --data ../data/bin/elite.bin --out ../nets/base.pt --steps 20000
Style fine-tune (Tal, Carlsen), starting from the base model:
  python train.py --data ../data/bin/tal.bin --init ../nets/base.pt --out ../nets/tal.pt \
      --steps 1500 --lr 3e-4 --elo-override 2700,2600
"""

import argparse
import math
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset import RecordFile, Sampler, val_batches
from model import HumanNet, masked_policy_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="record files, optionally path:weight")
    p.add_argument("--val-data", nargs="*", default=[], help="extra record files to report accuracy on")
    p.add_argument("--out", required=True, help="checkpoint to write (.pt)")
    p.add_argument("--init", help="checkpoint to start from (fine-tuning)")
    p.add_argument("--resume", action="store_true", help="continue --out, including optimizer state")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--se", type=int, default=16)
    p.add_argument("--policy-dim", type=int, default=32)
    p.add_argument("--elo-override", help="mover,opponent Elo fed to the net instead of the record's")
    p.add_argument("--holdout", type=int, default=64, help="every N-th game of --data goes to validation")
    p.add_argument("--val-holdout", type=int, default=8, help="same for --val-data files")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bench", type=int, default=0, help="only time this many steps and exit")
    p.add_argument("--keep-best", action="store_true", help="save only when top-1 on the first --data file's validation improves")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto = CUDA GPU when available")
    p.add_argument("--gpu", choices=["dml"], help="also train on a DirectML GPU (needs torch-directml): each batch is split between CPU and GPU")
    p.add_argument("--gpu-frac", type=float, default=0.5, help="initial share of each batch for the GPU (adapts to keep both busy)")
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    floor = args.lr * args.min_lr_frac
    return floor + (args.lr - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model, batches, value_weight):
    model.eval()
    dev = next(model.parameters()).device
    n = correct = correct3 = 0
    ploss = vloss = 0.0
    for batch in batches:
        planes, mask, target, wdl = (t.to(dev) for t in batch)
        logits, v = model(planes)
        pl, logits = masked_policy_loss(logits, mask, target)
        ploss += pl.item() * len(target)
        vloss += F.cross_entropy(v, wdl).item() * len(target)
        top3 = logits.topk(3, dim=1).indices
        correct += (top3[:, 0] == target).sum().item()
        correct3 += (top3 == target[:, None]).any(dim=1).sum().item()
        n += len(target)
    model.train()
    return dict(policy=ploss / n, value=vloss / n, top1=correct / n, top3=correct3 / n, n=n)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    elo = tuple(float(x) for x in args.elo_override.split(",")) if args.elo_override else None

    sources = []
    for spec in args.data:
        path, _, w = spec.partition(":")
        sources.append((RecordFile(path, args.holdout), float(w) if w else 1.0))
    for f, w in sources:
        print(f"{f.path}: {len(f.train_idx)} train / {len(f.val_idx)} val positions, weight {w}")

    ckpt = None
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    elif args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"] if ckpt else dict(
        channels=args.channels, blocks=args.blocks, se=args.se, policy_dim=args.policy_dim
    )
    model = HumanNet(**cfg)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: no CUDA GPU (install the CUDA build of torch)")
    use_cuda = not args.gpu and args.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"model {model.cfg}, {params / 1e6:.2f}M parameters, device {device}")

    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr
    )
    step = 0
    if args.resume and ckpt:
        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception as e:  # e.g. a checkpoint from another torch version
            print(f"optimizer state not restored ({type(e).__name__}: {e}); continuing with a fresh one")
        step = ckpt["step"]
    hybrid = HybridGpu(model, args.gpu_frac) if args.gpu else None

    val_sets = [(os.path.basename(f.path), val_batches(f, args.batch, args.val_batches, elo)) for f, _ in sources]
    for path in args.val_data:
        f = RecordFile(path, args.val_holdout)
        val_sets.append((os.path.basename(path), val_batches(f, args.batch, args.val_batches, elo)))

    def report(tag):
        first = None
        for name, batches in val_sets:
            r = evaluate(model, batches, args.value_weight)
            first = first if first is not None else r["top1"]
            print(
                f"  [{tag}] {name:14s} top1 {r['top1']:.4f} top3 {r['top3']:.4f} "
                f"policy {r['policy']:.4f} value {r['value']:.4f} (n={r['n']})",
                flush=True,
            )
        return first

    best_top1 = -1.0
    if not args.bench:
        best_top1 = report(f"step {step}")
        if args.keep_best:  # the starting point counts: there is always an output file
            save(args.out, model, opt, step, args, elo)
    sampler = Sampler(sources, args.batch, seed=args.seed + step, elo_override=elo)
    model.train()
    t0 = time.time()
    seen = 0
    run_p = run_v = None
    total = args.bench if args.bench else args.steps
    step_times = []
    while step < total:
        t_step = time.time()
        planes, mask, target, wdl = sampler.next()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        opt.zero_grad(set_to_none=True)
        if hybrid:
            pl, vl = hybrid.backward(planes, mask, target, wdl, args.value_weight)
        else:
            planes, mask, target, wdl = (t.to(device, non_blocking=True) for t in (planes, mask, target, wdl))
            logits, v = model(planes)
            pl, _ = masked_policy_loss(logits, mask, target)
            vl = F.cross_entropy(v, wdl)
            (pl + args.value_weight * vl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if hybrid:
            hybrid.sync_weights()
        step += 1
        seen += len(target)
        step_times.append(time.time() - t_step)
        run_p = pl.item() if run_p is None else 0.98 * run_p + 0.02 * pl.item()
        run_v = vl.item() if run_v is None else 0.98 * run_v + 0.02 * vl.item()
        if step % 50 == 0 or args.bench:
            el = time.time() - t0
            gpu_info = f", gpu {hybrid.ng}/{len(target)}" if hybrid else ""
            print(
                f"step {step}/{total} lr {lr_at(step, args):.2e} policy {run_p:.4f} value {run_v:.4f} "
                f"| {seen / el:.0f} pos/s, {el / 60:.1f} min{gpu_info}",
                flush=True,
            )
        if not args.bench and (step % args.val_every == 0 or step == total):
            top1 = report(f"step {step}")
            if not args.keep_best:
                save(args.out, model, opt, step, args, elo)
            elif top1 > best_top1:
                best_top1 = top1
                save(args.out, model, opt, step, args, elo)
                print(f"  saved best ({top1:.4f}) at step {step}", flush=True)
    sampler.stop = True
    if args.bench:
        tail = sorted(step_times[len(step_times) // 2 :])
        print(f"steady state: {args.batch / tail[len(tail) // 2]:.0f} positions/s (median of the last {len(tail)} steps)")


class HybridGpu:
    """Data parallelism across the CPU and a DirectML GPU in one process. The CPU model holds
    the master weights and the optimizer; a replica on the GPU computes gradients for its share
    of every batch on a second thread. Gradients and batch-norm statistics are averaged
    (weighted by sub-batch size) and the weights are copied back to the GPU after each step."""

    def __init__(self, model, frac):
        import torch_directml

        self.dev = torch_directml.device()
        self.model = model
        self.replica = HumanNet(**model.cfg).to(self.dev)
        self.replica.train()
        self.frac = frac
        # DirectML compiles kernels per input shape, so the GPU's share of the batch only
        # changes in steps of 32 positions and only every `adapt_every` steps.
        self.ng = None
        self.adapt_every = 50
        self.timings = []
        self.sync_weights()
        print(f"hybrid training on CPU + {torch_directml.device_name(0).strip(chr(0))}, initial GPU share {frac}")

    def backward(self, planes, mask, target, wdl, value_weight):
        n = len(target)
        if self.ng is None:
            self.ng = min(n - 32, max(32, int(round(n * self.frac / 32)) * 32))
        ng = self.ng
        result = {}

        def gpu_part():
            t = time.perf_counter()
            logits, v = self.replica(planes[:ng].to(self.dev))
            pl, _ = masked_policy_loss(logits, mask[:ng].to(self.dev), target[:ng].to(self.dev))
            vl = F.cross_entropy(v, wdl[:ng].to(self.dev))
            for p in self.replica.parameters():
                p.grad = None
            (pl + value_weight * vl).backward()
            result["grads"] = [p.grad.to("cpu") for p in self.replica.parameters()]
            result["loss"] = (pl.item(), vl.item())
            result["time"] = time.perf_counter() - t

        worker = threading.Thread(target=gpu_part)
        worker.start()
        t = time.perf_counter()
        logits, v = self.model(planes[ng:])
        pl, _ = masked_policy_loss(logits, mask[ng:], target[ng:])
        vl = F.cross_entropy(v, wdl[ng:])
        (pl + value_weight * vl).backward()
        cpu_time = time.perf_counter() - t
        worker.join()
        if "grads" not in result:
            raise RuntimeError("GPU part of the step failed")

        wc, wg = (n - ng) / n, ng / n
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), result["grads"]):
                p.grad.mul_(wc).add_(g, alpha=wg)
            # Batch-norm running statistics: weighted average of both replicas.
            for (name, bc), bg in zip(self.model.named_buffers(), self.replica.buffers()):
                if bc.dtype.is_floating_point:
                    bc.mul_(wc).add_(bg.to("cpu"), alpha=wg)
        # Every `adapt_every` steps, give each device a share proportional to its speed
        # (median over the window; the first steps after a shape change include compilation).
        self.timings.append(((n - ng) / max(cpu_time, 1e-6), ng / max(result["time"], 1e-6)))
        if len(self.timings) >= self.adapt_every:
            rates = sorted(self.timings[self.adapt_every // 5 :], key=lambda r: r[1] / (r[0] + r[1]))
            cpu_rate, gpu_rate = rates[len(rates) // 2]
            self.frac = gpu_rate / (cpu_rate + gpu_rate)
            new_ng = min(n - 32, max(32, int(round(n * self.frac / 32)) * 32))
            if abs(new_ng - self.ng) >= 32:
                self.ng = new_ng
            self.timings = []
        gpl, gvl = result["loss"]
        return (
            torch.tensor(wc * pl.item() + wg * gpl),
            torch.tensor(wc * vl.item() + wg * gvl),
        )

    def sync_weights(self):
        with torch.no_grad():
            for pg, pc in zip(self.replica.parameters(), self.model.parameters()):
                pg.copy_(pc.to(self.dev))
            for bg, bc in zip(self.replica.buffers(), self.model.buffers()):
                bg.copy_(bc.to(self.dev))


def save(path, model, opt, step, args, elo):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=model.cfg, elo=elo, args=vars(args)),
        tmp,
    )
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
