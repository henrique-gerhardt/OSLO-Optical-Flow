"""Planar RAFT trained from scratch on the ERP frames of the sfprep shards.

The matched-backbone baseline of RAFT_PARITY.md section 9: TorchVision RAFT (small or
large, no pretrained weights) trained on the same shards, for the same number of
optimizer steps and pairs per step as an OSLO-RAFT-R run, with RAFT's own recipe
adapted to ERP:

- random crops (RAFT trains on crops); the crop must fit every source's frame size;
- photometric jitter (RAFT's ColorJitter ranges, asymmetric with probability 0.2);
- horizontal ERP roll by a random integer shift, which is an exact yaw rotation of the
  sphere (the flow values do not change);
- horizontal flip with probability 0.5 (a mirror of the sphere: du negated);
- no scaling, no vertical flip (both change the ERP geometry or gravity).

Loss: RAFT's sequence loss (L1, gamma 0.8, pixels with |flow| < max_flow and valid).
Optimizer: AdamW + OneCycle (RAFT: 5% warm-up, linear). Frames enter as x/127.5 - 1,
which is what ``run_raft_shard_baseline.py --tv-checkpoint`` feeds at evaluation; that
script scores the checkpoint with the same spherical metrics as ``run_oslo_raft.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from spherical_flow.shard_dataset import _import_sfprep


def parse_sources(text: str) -> List[Tuple[str, str]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            dataset, split = item.split(":")
            out.append((dataset, split))
    return out


class ERPPairStream(IterableDataset):
    """Endless, shuffled, augmented ERP crops from the shards."""

    def __init__(self, shards_dir: str, sources: List[Tuple[str, str]], crop: Tuple[int, int],
                 seed: int, shuffle_buffer: int = 64, photo_prob: float = 1.0,
                 roll: bool = True, hflip_prob: float = 0.5) -> None:
        self._iter_shard, list_shards = _import_sfprep()
        self.shards = [s for d, sp in sources for s in list_shards(Path(shards_dir), d, sp)]
        if not self.shards:
            raise RuntimeError(f"no shards for {sources} under {shards_dir}")
        self.crop, self.seed, self.buffer = crop, seed, shuffle_buffer
        self.photo_prob, self.roll, self.hflip_prob = photo_prob, roll, hflip_prob

    def _records(self, rng: random.Random) -> Iterator[dict]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        epoch = 0
        while True:
            shards = list(self.shards)
            random.Random(self.seed + epoch).shuffle(shards)
            for shard in shards[wid::nw]:
                yield from self._iter_shard(shard)
            epoch += 1

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = random.Random(self.seed * 1009 + wid)
        gen = torch.Generator().manual_seed(self.seed * 1009 + wid)
        jitter = _color_jitter()
        buffer: List[dict] = []
        for record in self._records(rng):
            buffer.append(record)
            if len(buffer) < self.buffer:
                continue
            yield self._augment(buffer.pop(rng.randrange(len(buffer))), rng, gen, jitter)

    def _augment(self, record: dict, rng: random.Random, gen: torch.Generator, jitter) -> dict:
        f1 = torch.from_numpy(np.ascontiguousarray(record["frame1"])).permute(2, 0, 1)
        f2 = torch.from_numpy(np.ascontiguousarray(record["frame2"])).permute(2, 0, 1)
        flow = torch.from_numpy(np.ascontiguousarray(record["flow"])).float().permute(2, 0, 1)
        valid = torch.from_numpy(np.ascontiguousarray(record["valid"])).bool()
        height, width = valid.shape
        ch, cw = self.crop
        if ch > height or cw > width:
            raise ValueError(f"crop {self.crop} larger than frame {height}x{width}")

        if rng.random() < self.photo_prob:
            if rng.random() < 0.2:  # asymmetric: each frame its own jitter
                torch.manual_seed(rng.randrange(2**31))
                f1, f2 = jitter(f1), jitter(f2)
            else:
                torch.manual_seed(rng.randrange(2**31))
                both = jitter(torch.cat([f1, f2], dim=2))
                f1, f2 = both[:, :, :width], both[:, :, width:]
        if self.roll:
            shift = rng.randrange(width)
            f1, f2 = f1.roll(shift, dims=2), f2.roll(shift, dims=2)
            flow, valid = flow.roll(shift, dims=2), valid.roll(shift, dims=1)
        if rng.random() < self.hflip_prob:
            f1, f2, flow, valid = f1.flip(2), f2.flip(2), flow.flip(2), valid.flip(1)
            flow = torch.stack([-flow[0], flow[1]])
        y0 = rng.randrange(height - ch + 1)
        x0 = rng.randrange(width - cw + 1)
        sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
        return {
            "frame1": f1[:, sl[0], sl[1]].contiguous(),
            "frame2": f2[:, sl[0], sl[1]].contiguous(),
            "flow": flow[:, sl[0], sl[1]].contiguous(),
            "valid": valid[sl[0], sl[1]].contiguous(),
        }


def _color_jitter():
    from torchvision.transforms import ColorJitter

    return ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / math.pi)


def sequence_loss(predictions, flow, valid, gamma: float, max_flow: float):
    mag = flow.pow(2).sum(dim=1).sqrt()
    mask = (valid & (mag < max_flow)).float().unsqueeze(1)
    n = len(predictions)
    loss = flow.new_zeros(())
    for i, pred in enumerate(predictions):
        weight = gamma ** (n - i - 1)
        loss = loss + weight * ((pred - flow).abs() * mask).sum() / mask.sum().clamp_min(1.0) / 2
    epe = ((predictions[-1] - flow).pow(2).sum(dim=1).sqrt() * mask[:, 0]).sum()
    return loss, (epe / mask.sum().clamp_min(1.0)).item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards", default="/data/shards")
    p.add_argument("--train-sources", default="replica360:train,flowscape:train")
    p.add_argument("--model", default="raft_small", choices=["raft_small", "raft_large"])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pct-start", type=float, default=0.05)
    p.add_argument("--iters", type=int, default=12, help="GRU updates at training.")
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--max-flow", type=float, default=400.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--crop", default="384x768", help="HxW, must fit every source frame.")
    p.add_argument("--photo-prob", type=float, default=1.0)
    p.add_argument("--no-roll", action="store_true", help="Disable the random ERP yaw roll.")
    p.add_argument("--hflip-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--smoke-test", action="store_true", help="3 steps, then save and exit.")
    p.add_argument("--output-dir", default="/outputs/erp_raft")
    args = p.parse_args()

    from torchvision.models.optical_flow import raft_large, raft_small

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    crop = tuple(int(x) for x in args.crop.lower().split("x"))
    if crop[0] % 8 or crop[1] % 8:
        raise SystemExit("--crop must be divisible by 8")

    model = (raft_small if args.model == "raft_small" else raft_large)(weights=None).to(device)
    model.train()
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    steps = 3 if args.smoke_test else args.steps
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=steps + 100, pct_start=args.pct_start,
        cycle_momentum=False, anneal_strategy="linear")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    ds = ERPPairStream(args.shards, parse_sources(args.train_sources), crop, args.seed,
                       photo_prob=args.photo_prob, roll=not args.no_roll,
                       hflip_prob=args.hflip_prob)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    print(f"erp raft model={args.model} params={n_params:,} crop={crop} "
          f"batch={args.batch_size}x{args.grad_accum} steps={steps} lr={args.lr} "
          f"sources={args.train_sources} device={device}", flush=True)

    batches = iter(loader)
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_sum, epe_sum = 0.0, 0.0
        for _ in range(args.grad_accum):
            batch = next(batches)
            f1 = batch["frame1"].to(device).float().div(127.5).sub(1.0)
            f2 = batch["frame2"].to(device).float().div(127.5).sub(1.0)
            flow, valid = batch["flow"].to(device), batch["valid"].to(device)
            with torch.autocast(device.type, enabled=args.amp and device.type == "cuda"):
                preds = model(f1, f2, num_flow_updates=args.iters)
            loss, epe = sequence_loss([q.float() for q in preds], flow, valid,
                                      args.gamma, args.max_flow)
            scaler.scale(loss / args.grad_accum).backward()
            loss_sum += loss.item() / args.grad_accum
            epe_sum += epe / args.grad_accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step == 1 or step % args.log_every == 0 or step == steps:
            print(f"step={step:06d} loss={loss_sum:.4f} train_epe_px={epe_sum:.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} elapsed_s={time.time() - t0:.0f}", flush=True)

    ckpt = out / "raft_erp.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "params": n_params}, ckpt)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved_checkpoint={ckpt} elapsed_s={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
