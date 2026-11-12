"""Does a short SO(3) fine-tune make a released in-domain model orientation-robust?

PriOr-RAFT (ICCV 2025, released FlowScape weights) loses about 4x its error under a Haar
rotation of the camera (UNIVERSALITY_TABLE 16.45). This probe fine-tunes the released weights
for a few hundred steps with the ERP re-render rotation of run_erp_raft_train.py and scores
the same held-out test pairs upright and rotated, before and after.

Arms (same steps, same pairs, same lr; only the rotation probability changes):
  so3   each training pair is rotated with probability --so3-prob (Haar)
  ctrl  no rotation (control for "more steps on the benchmark")

Metric: geodesic endpoint error in degrees, cos-latitude weighted (sphere mean), on pixels
whose GT endpoint stays inside the raster; zero flow is scored the same way. ERP-px EPE is
reported next to it.

Departure from the released recipe: only the primitive-view sequence loss is used (the
orthogonal-view loss needs a helper that is not vendored), batch 1, constant lr.

Pre-registered reading, on the rotated error after the so3 arm:
  falls by >= 50 % and upright rises by <= 25 %   the remedy works on a released model
  falls by < 20 %                                   a short fine-tune is not enough
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import common as C
from run_erp_raft_train import rotate_erp_pair
from spherical_flow.geometry import equirectangular_pixels_to_unit_vectors
from spherical_flow.prior_adapter import _InputPadder, load_prior_checkpoint
from spherical_flow.so3_augment import sample_rotation


def to_chw(rec):
    f1 = torch.from_numpy(rec["frame1"]).permute(2, 0, 1)
    f2 = torch.from_numpy(rec["frame2"]).permute(2, 0, 1)
    flow = torch.from_numpy(np.ascontiguousarray(rec["flow"])).permute(2, 0, 1).float()
    return f1, f2, flow, torch.ones(flow.shape[1:], dtype=torch.bool)


def geodesic_deg(flow_a: torch.Tensor, flow_b: torch.Tensor) -> torch.Tensor:
    """Angle between the endpoints of two ERP flows [2, H, W], in degrees, [H, W]."""
    _, h, w = flow_a.shape
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    a = equirectangular_pixels_to_unit_vectors((u + flow_a[0]).reshape(-1), (v + flow_a[1]).reshape(-1), h, w)
    b = equirectangular_pixels_to_unit_vectors((u + flow_b[0]).reshape(-1), (v + flow_b[1]).reshape(-1), h, w)
    ang = torch.atan2(torch.linalg.cross(a, b).norm(dim=-1), (a * b).sum(-1))
    return torch.rad2deg(ang).reshape(h, w)


def cos_weights(h: int, w: int) -> torch.Tensor:
    lat = math.pi / 2 - (torch.arange(h, dtype=torch.float32) + 0.5) / h * math.pi
    return torch.cos(lat)[:, None].expand(h, w)


@torch.no_grad()
def predict(model, f1, f2, device, iters):
    padder = _InputPadder(f1.shape)
    a, b = padder.pad(f1[None].float().to(device), f2[None].float().to(device))
    return padder.unpad(model(a, b, iters=iters, test_mode=True))[0].float().cpu()


@torch.no_grad()
def evaluate(model, samples, device, iters):
    """samples: list of (f1, f2, flow, valid). Returns sphere-mean degrees and px EPE."""
    model.eval()
    tot = {"deg": 0.0, "zero_deg": 0.0, "px": 0.0, "wsum": 0.0, "npx": 0}
    for f1, f2, flow, valid in samples:
        pred = predict(model, f1, f2, device, iters)
        wgt = cos_weights(*flow.shape[1:]) * valid
        tot["deg"] += (geodesic_deg(pred, flow) * wgt).sum().item()
        tot["zero_deg"] += (geodesic_deg(torch.zeros_like(flow), flow) * wgt).sum().item()
        tot["wsum"] += wgt.sum().item()
        d = pred - flow
        w = flow.shape[2]
        d[0] = torch.remainder(d[0] + w / 2, w) - w / 2
        tot["px"] += d.norm(dim=0)[valid].sum().item()
        tot["npx"] += int(valid.sum())
    return {"deg": tot["deg"] / tot["wsum"], "zero_deg": tot["zero_deg"] / tot["wsum"], "px": tot["px"] / tot["npx"]}


def sequence_loss(preds, flow, valid, gamma=0.8, max_flow=400.0):
    """PriOr-Flow's uniform loss on the primitive view: cos-lat weighted L1 over the iterations."""
    h, w = flow.shape[-2:]
    mask = cos_weights(h, w).to(flow.device)
    mask = mask / mask.sum()
    ok = valid & (flow.norm(dim=1) < max_flow)
    loss = 0.0
    for i, p in enumerate(preds):
        loss = loss + gamma ** (len(preds) - i - 1) * (ok * mask * (p - flow).abs().sum(1)).sum()
    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--raw-root", default=C.RAW_ROOT)
    p.add_argument("--shards", default="", help="sfprep shard root (box); empty = raw FlowScape")
    p.add_argument("--sources", default="flowscape:test")
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--train-pool", type=int, default=800, help="train pairs kept in memory (shards only)")
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--weathers", default="sunny,cloud,fog,rain")
    p.add_argument("--arm", default="so3", choices=["so3", "ctrl"])
    p.add_argument("--so3-prob", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-at", default="", help="comma list of steps to evaluate at (0 and the last are always in)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train-iters", type=int, default=12)
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--eval-pairs", type=int, default=32)
    p.add_argument("--eval-stride", type=int, default=43)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="mps")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    model, meta = load_prior_checkpoint(args.checkpoint, device, args.eval_iters)
    print(f"PriOr-RAFT loaded: {meta['params']} params, device {device}, arm {args.arm}", flush=True)

    # held-out test pairs, each scored upright and under one fixed Haar rotation
    eye = torch.eye(3)
    egen = torch.Generator().manual_seed(1234)
    up, rot = [], []
    targs = argparse.Namespace(shards=args.shards, sources=args.sources, raw_root=args.raw_root, split="test",
                               weathers=args.weathers, stride=args.eval_stride, max_pairs=args.eval_pairs)
    for rec in C.iter_pairs(targs):
        chw = to_chw(rec)
        up.append(rotate_erp_pair(*chw, eye))
        rot.append(rotate_erp_pair(*chw, sample_rotation(egen, uniform_so3=True)))
    print(f"eval pairs: {len(up)} (upright and Haar-rotated)", flush=True)

    if args.shards:  # the shards have no random access: keep a pool of train pairs in memory
        pargs = argparse.Namespace(shards=args.shards, sources=args.train_sources, stride=args.train_stride,
                                   max_pairs=args.train_pool)
        pool = [to_chw(rec) for rec in C.iter_pairs(pargs)]
        loaders = [lambda _, s=s: s for s in pool]
    else:
        loaders = [lambda _, load=load: to_chw(load(None))
                   for load in C._raw_pairs(Path(args.raw_root), "train", args.weathers.split(","))]
    print(f"train pairs available: {len(loaders)}", flush=True)

    eval_at = sorted({0, args.steps} | {int(s) for s in args.eval_at.split(",") if s})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
    history, start = [], time.time()

    def run_eval(step):
        row = {"step": step, "upright": evaluate(model, up, device, args.eval_iters),
               "rotated": evaluate(model, rot, device, args.eval_iters)}
        history.append(row)
        u, r = row["upright"], row["rotated"]
        print(f"[eval step {step}] upright {u['deg']:.4f} deg ({u['px']:.2f} px, zero {u['zero_deg']:.3f}) | "
              f"rotated {r['deg']:.4f} deg ({r['px']:.2f} px, zero {r['zero_deg']:.3f}) ({time.time() - start:.0f}s)",
              flush=True)

    run_eval(0)
    for step in range(1, args.steps + 1):
        model.train()
        model.freeze_bn()
        f1, f2, flow, valid = rng.choice(loaders)(None)
        rotated = args.arm == "so3" and rng.random() < args.so3_prob
        rotation = sample_rotation(gen, uniform_so3=True) if rotated else eye
        f1, f2, flow, valid = rotate_erp_pair(f1, f2, flow, valid, rotation)
        preds, _ = model(f1[None].float().to(device), f2[None].float().to(device), iters=args.train_iters)
        loss = sequence_loss(preds, flow[None].to(device), valid[None].to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.4f} rot {int(rotated)} ({time.time() - start:.0f}s)", flush=True)
        if step in eval_at:
            run_eval(step)

    first, last = history[0], history[-1]
    d_rot = 100 * (last["rotated"]["deg"] / first["rotated"]["deg"] - 1)
    d_up = 100 * (last["upright"]["deg"] / first["upright"]["deg"] - 1)
    print(f"\nP11 {args.arm}: rotated {first['rotated']['deg']:.4f} -> {last['rotated']['deg']:.4f} deg ({d_rot:+.1f} %), "
          f"upright {first['upright']['deg']:.4f} -> {last['upright']['deg']:.4f} deg ({d_up:+.1f} %)")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out / f"prior_raft_{args.arm}.pth")
    (out / f"p11_{args.arm}.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2))
    print(f"saved={out / f'p11_{args.arm}.json'}")


if __name__ == "__main__":
    main()
