"""Overshoot experiment: an ERP polar-band head on top of a frozen OSLO checkpoint.

About 3 % of FlowScape pixels (rows next to the nadir, some at the zenith) have a GT
endpoint outside the raster. There the official EPE compares raw ERP (du, dv), and |du| has
a median of ~130 px: near a pole, ERP longitude moves a lot for little angular motion. A
node model reads its flow out through sphere endpoints, so it cannot write these values:
the r6 oracle itself errs ~112 px there, and OSLO 103-123 px against RAFT-small 27-35 px.
That region is the whole official-EPE gap of the matched 30k pair (in-raster OSLO 5.1-5.5
vs RAFT-small 6.1-7.4 px).

This script keeps OSLO frozen and trains a small planar CNN on the top and bottom K = H/16
ERP rows only. Its inputs are both frames, the OSLO readout and longitude / pole-distance
channels. It predicts a residual (du, dv) for every band pixel, supervised with the raw GT,
overshoot included. The top band is flipped so that both bands put the pole on the last row.

Rows on flowscape:test:
  oslo         the checkpoint's readout (= its benchmark row)
  oslo_band    the same with the band head applied
  gt_clamped   GT with endpoints clamped into the raster: the floor of any readout
               that only writes in-raster endpoints (context, not a method)

Pre-registered reading: success if oslo_band brings overshoot EPE <= 40 px while in-raster
EPE moves by <= 0.05 px. Then the next step is to train the head end to end with OSLO.
Overshoot still > 80 px means the GT there is not learnable from these inputs.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.erp_readout import bilinear_node_weights, erp_pixel_directions, nodes_to_erp_flow
from spherical_flow.shard_dataset import _to_chw_free_float, sample_pair_to_nodes

FLOW_NORM = 32.0


def build_oslo_runner(args):
    """Frozen OSLO checkpoint -> function(record) -> ERP flow [H, W, 2] (same path as p3)."""
    sys.argv = ["x", "--predictor", "oslo", "--init-checkpoint", args.init_checkpoint,
                "--pyramid-cache", args.pyramid_cache, "--eval-iters", str(args.eval_iters),
                "--device", args.device]
    eargs = C.ee.parse_args()
    C.ee.adopt_checkpoint_grids(eargs)
    device = torch.device(args.device)
    model, pyramid = C.ee.build_oslo(eargs, device)
    model.eval()
    sup = healpix_unit_vectors(eargs.resolution)
    sbe, sbn = tangent_basis(sup)
    ret = pyramid.retina_level.points.cpu()
    rbe, rbn = tangent_basis(ret)
    cache = {}

    def run(rec):
        flow = rec["flow"]
        h, w = flow.shape[:2]
        if (h, w) not in cache:
            cache[(h, w)] = (bilinear_node_weights(eargs.resolution, h, w), erp_pixel_directions(h, w))
        wts, dirs = cache[(h, w)]
        gt = torch.from_numpy(np.ascontiguousarray(flow)).float()
        nodes = sample_pair_to_nodes(_to_chw_free_float(rec["frame1"]), _to_chw_free_float(rec["frame2"]), gt,
                                     torch.from_numpy(np.ascontiguousarray(rec["valid"])), ret, rbe, rbn,
                                     target_points=sup)
        with torch.no_grad():
            node = model(nodes["frame1"][None].to(device), nodes["frame2"][None].to(device), pyramid,
                         iters=eargs.eval_iters)[-1][0].float().cpu()
        return nodes_to_erp_flow(node, sup, sbe, sbn, h, w, wts, pixel_dirs=dirs)

    return run


def wrap_du(flow: np.ndarray, w: int) -> np.ndarray:
    out = flow.copy()
    out[..., 0] = np.remainder(out[..., 0] + w / 2, w) - w / 2
    return out


def bands(rec, pred: np.ndarray, k: int):
    """Two band samples (bottom, top flipped), each dict of [K, W, ...] arrays + flag."""
    f1, f2 = np.asarray(rec["frame1"]), np.asarray(rec["frame2"])
    h, w = pred.shape[:2]
    gt = wrap_du(np.asarray(rec["flow"], dtype=np.float32), w)
    out = []
    for flag, sl in ((1.0, slice(h - k, h)), (-1.0, slice(k - 1, None, -1))):
        p, g = pred[sl].astype(np.float32).copy(), gt[sl].copy()
        if flag < 0:  # flipped top band: rows grow toward the pole, so dv changes sign
            p[..., 1] *= -1
            g[..., 1] *= -1
        out.append({"f1": f1[sl].copy(), "f2": f2[sl].copy(), "pred": p.astype(np.float16), "gt": g, "flag": flag})
    return out


def unband(pred: np.ndarray, outs, k: int) -> np.ndarray:
    """Write the two band outputs ([K, W, 2], band frame) back into a copy of pred."""
    full = pred.copy()
    bot, top = outs
    h = pred.shape[0]
    full[h - k:] = bot
    t = top[::-1].copy()
    t[..., 1] *= -1
    full[:k] = t
    return full


class CConv(nn.Module):
    """3x3 conv, circular padding in longitude, replicate in latitude, GELU."""

    def __init__(self, cin, cout, stride=(1, 1), dil=(1, 1), act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, dilation=dil)
        self.dil = dil
        self.act = act

    def forward(self, x):
        dy, dx = self.dil
        x = F.pad(x, (dx, dx, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, dy, dy), mode="replicate")
        x = self.conv(x)
        return F.gelu(x) if self.act else x


class BandHead(nn.Module):
    """Small U-net over a [K, W] band; coarsest level is K/8 x W/32 with dilated row convs."""

    def __init__(self, cin=12, c=48):
        super().__init__()
        self.e0 = nn.Sequential(CConv(cin, c), CConv(c, c))
        self.e1 = nn.Sequential(CConv(c, c, (2, 2)), CConv(c, c))
        self.e2 = nn.Sequential(CConv(c, 2 * c, (2, 2)), CConv(2 * c, 2 * c))
        self.e3 = nn.Sequential(CConv(2 * c, 2 * c, (2, 2)), CConv(2 * c, 2 * c))
        self.e4 = nn.Sequential(CConv(2 * c, 4 * c, (1, 2)), CConv(4 * c, 4 * c, (1, 2)),
                                CConv(4 * c, 4 * c, dil=(1, 2)), CConv(4 * c, 4 * c, dil=(1, 4)),
                                CConv(4 * c, 4 * c, dil=(1, 8)))
        self.d3 = CConv(6 * c, 2 * c)
        self.d2 = CConv(4 * c, 2 * c)
        self.d1 = CConv(3 * c, c)
        self.d0 = CConv(2 * c, c)
        self.out = nn.Conv2d(c, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x0 = self.e0(x)
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)

        def up(a, b):
            return torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)

        y = self.d3(up(x4, x3))
        y = self.d2(up(y, x2))
        y = self.d1(up(y, x1))
        y = self.d0(up(y, x0))
        return self.out(y) * FLOW_NORM


def to_input(batch, device):
    """List of band dicts -> (x [B, 12, K, W], base pred [B, 2, K, W])."""
    f1 = torch.from_numpy(np.stack([b["f1"] for b in batch])).float().div(255).sub(0.5)
    f2 = torch.from_numpy(np.stack([b["f2"] for b in batch])).float().div(255).sub(0.5)
    p = torch.from_numpy(np.stack([b["pred"] for b in batch]).astype(np.float32))
    bsz, k, w = p.shape[:3]
    lon = (torch.arange(w, dtype=torch.float32) + 0.5) / w * 2 * math.pi
    pole = (k - 0.5 - torch.arange(k, dtype=torch.float32)) / k
    flag = torch.tensor([b["flag"] for b in batch], dtype=torch.float32).view(bsz, 1, 1, 1).expand(bsz, 1, k, w)
    coords = torch.stack([torch.sin(lon).expand(k, w), torch.cos(lon).expand(k, w), pole.view(k, 1).expand(k, w)])
    x = torch.cat([f1.permute(0, 3, 1, 2), f2.permute(0, 3, 1, 2), p.permute(0, 3, 1, 2) / FLOW_NORM,
                   coords.expand(bsz, 3, k, w), flag], 1)
    return x.to(device), p.permute(0, 3, 1, 2).to(device)


def predict(head, samples, device):
    with torch.no_grad():
        x, base = to_input(samples, device)
        return (base + head(x)).permute(0, 2, 3, 1).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--train-split", default="train", help="raw-FlowScape split (local runs)")
    p.add_argument("--train-pairs", type=int, default=2000)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    oslo = build_oslo_runner(args)

    targs = argparse.Namespace(**{**vars(args), "split": args.train_split, "sources": args.train_sources,
                                  "stride": args.train_stride, "max_pairs": args.train_pairs})
    train, k, start = [], None, time.time()
    for i, rec in enumerate(C.iter_pairs(targs)):
        pred = oslo(rec).numpy()
        k = pred.shape[0] // 16
        train.extend(bands(rec, pred, k))
        if (i + 1) % 200 == 0:
            print(f"cached {i + 1} train pairs ({time.time() - start:.0f}s)", flush=True)
    print(f"train bands {len(train)} (K={k} rows)", flush=True)

    head = BandHead(c=args.width).to(device)
    n_params = sum(t.numel() for t in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    start = time.time()
    for step in range(args.steps):
        batch = [train[j] for j in rng.integers(0, len(train), args.batch)]
        x, base = to_input(batch, device)
        gt = torch.from_numpy(np.stack([b["gt"] for b in batch])).permute(0, 3, 1, 2).to(device)
        loss = (base + head(x) - gt).norm(dim=1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            base_l = (base - gt).norm(dim=1).mean().item()
            print(f"step {step} band EPE {loss.item():.2f} (oslo {base_l:.2f}) ({time.time() - start:.0f}s)",
                  flush=True)
    del train
    head.eval()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "args": vars(args), "k": k}, out_dir / "band_head.pt")

    scorer = C.Scorer()
    start = time.time()
    for i, rec in enumerate(C.iter_pairs(args)):
        pred = oslo(rec).numpy()
        h, w = pred.shape[:2]
        band = np.zeros((h, w), dtype=bool)
        band[:k] = band[h - k:] = True
        band_t = torch.from_numpy(band)
        over = C.ee.overshoot_masks(torch.from_numpy(np.asarray(rec["flow"], dtype=np.float32)))["overshoot"]
        extra = {"band": band_t, "band_inraster": band_t & ~over}
        flow = rec["flow"]
        scorer.add("oslo", torch.from_numpy(pred), flow, extra)
        full = unband(pred, predict(head, bands(rec, pred, k), device), k)
        scorer.add("oslo_band", torch.from_numpy(full), flow, extra)
        rows = np.arange(h, dtype=np.float32)[:, None]
        clamped = wrap_du(np.asarray(flow, dtype=np.float32), w)
        clamped[..., 1] = np.clip(rows + clamped[..., 1], -0.5, h - 0.5) - rows
        scorer.add("gt_clamped", torch.from_numpy(clamped), flow, extra)
        if (i + 1) % 50 == 0:
            print(f"test pair {i + 1}: oslo {scorer.per_pair['oslo'][-1]:.2f} "
                  f"band {scorer.per_pair['oslo_band'][-1]:.2f} ({time.time() - start:.0f}s)", flush=True)
    summary = scorer.summary(base_row="oslo")
    print(f"\nP6 polar-band head (K={k} rows, {n_params} params, {args.steps} steps) on {args.init_checkpoint}. "
          + C.REFERENCE)
    scorer.print_table(summary, extra_cols=[("band", "official_band_epe"),
                                            ("band_in", "official_band_inraster_epe")])
    C.save(args.output_dir, {"args": vars(args), "k": k, "params": n_params, "rows": summary}, "p6_polar_band_head.json")


if __name__ == "__main__":
    main()
