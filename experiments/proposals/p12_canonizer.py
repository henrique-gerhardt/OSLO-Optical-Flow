"""Can the camera orientation be estimated from one frame well enough to canonize it?

Oracle canonization keeps the released models' accuracy under any rotation (PriOr-RAFT
0.2412 -> 0.2440 deg, ORIENTATION_ROBUSTNESS 4d), and the tolerance to a tilt error is
+18-27 % at 2 deg, +69-86 % at 5 deg (4e). This probe trains a small regressor of the
canonical up and forward directions on rotated training frames (the labels are free: the
rotation is ours) and reports its angular errors on rotated test frames.

Pre-registered reading, tilt (up-vector) error on flowscape:test under Haar rotations:
  median <= 2 deg and p90 <= 5 deg   canonization is viable; run the end-to-end views
  median >= 5 deg                    not viable with this estimator
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from run_raft_shard_baseline import erp_direction_grid, rotate_erp
from spherical_flow.canonizer import INPUT_HW, Canonizer, labels_from_rotation, prepare, rotation_from_vectors
from spherical_flow.so3_augment import sample_rotation


def small(frame_u8: np.ndarray) -> torch.Tensor:
    """uint8 [H, W, 3] -> float [160, 320, 3] in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frame_u8)).float().div(255.0).permute(2, 0, 1)[None]
    return F.interpolate(x, size=INPUT_HW, mode="area")[0].permute(1, 2, 0).contiguous()


def load_frames(args, sources, split, stride, max_pairs):
    a = argparse.Namespace(shards=args.shards, sources=sources, raw_root=args.raw_root, split=split,
                           weathers=args.weathers, stride=stride, max_pairs=max_pairs)
    return [small(rec["frame1"]) for rec in C.iter_pairs(a)]


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards", default="")
    p.add_argument("--raw-root", default=C.RAW_ROOT)
    p.add_argument("--weathers", default="sunny,cloud,fog,rain")
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--sources", default="flowscape:test")
    p.add_argument("--train-frames", type=int, default=2500)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--eval-frames", type=int, default=400)
    p.add_argument("--eval-stride", type=int, default=3)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    grid = erp_direction_grid(*INPUT_HW)

    train = load_frames(args, args.train_sources, "train", args.train_stride, args.train_frames)
    test = load_frames(args, args.sources, "test", args.eval_stride, args.eval_frames)
    print(f"frames: train {len(train)}, test {len(test)}", flush=True)

    egen = torch.Generator().manual_seed(1234)
    eval_set = []
    for frame in test:
        rot = sample_rotation(egen, uniform_so3=True)
        eval_set.append((prepare(rotate_erp(frame, rot, grid)), *labels_from_rotation(rot), rot))

    model = Canonizer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    @torch.no_grad()
    def evaluate():
        model.eval()
        up_err, fwd_err, rot_err = [], [], []
        for i in range(0, len(eval_set), 64):
            chunk = eval_set[i:i + 64]
            up, fwd = model(torch.stack([c[0] for c in chunk]).to(device))
            up, fwd = up.float().cpu(), fwd.float().cpu()
            up_err.append(angle_deg(up, torch.stack([c[1] for c in chunk])))
            fwd_err.append(angle_deg(fwd, torch.stack([c[2] for c in chunk])))
            for j, c in enumerate(chunk):  # residual rotation after full canonization
                resid = rotation_from_vectors(up[j], fwd[j]).T @ c[3]
                rot_err.append(torch.rad2deg(torch.acos(((resid.trace() - 1) / 2).clamp(-1, 1))))
        out = {}
        for name, v in (("tilt", torch.cat(up_err)), ("heading", torch.cat(fwd_err)), ("rotation", torch.stack(rot_err))):
            out[name] = {"median": float(v.median()), "mean": float(v.mean()), "p90": float(v.quantile(0.9)),
                         "frac_le_2": float((v <= 2).float().mean()), "frac_le_5": float((v <= 5).float().mean())}
        return out

    history, start = [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        xs, ups, fwds = [], [], []
        for _ in range(args.batch_size):
            rot = sample_rotation(gen, uniform_so3=True)
            frame = rng.choice(train)
            if rng.random() < 0.5:  # photometric jitter, so the estimator does not key on exact colours
                frame = (frame * (0.7 + 0.6 * rng.random()) + 0.1 * (rng.random() - 0.5)).clamp(0, 1)
            xs.append(prepare(rotate_erp(frame, rot, grid)))
            u, f = labels_from_rotation(rot)
            ups.append(u)
            fwds.append(f)
        up, fwd = model(torch.stack(xs).to(device))
        loss = (1 - (up * torch.stack(ups).to(device)).sum(1)).mean() + (1 - (fwd * torch.stack(fwds).to(device)).sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} ({time.time() - start:.0f}s)", flush=True)
        if step % max(1, args.steps // 4) == 0 or step == args.steps:
            row = {"step": step, **evaluate()}
            history.append(row)
            print(f"[eval step {step}] " + " | ".join(
                f"{k} median {row[k]['median']:.2f} mean {row[k]['mean']:.2f} p90 {row[k]['p90']:.2f} deg "
                f"(<=2: {100 * row[k]['frac_le_2']:.0f} %, <=5: {100 * row[k]['frac_le_5']:.0f} %)"
                for k in ("tilt", "heading", "rotation")), flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out / "canonizer.pth")
    (out / "p12_canonizer.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2))
    print(f"saved={out / 'canonizer.pth'}")


if __name__ == "__main__":
    main()
