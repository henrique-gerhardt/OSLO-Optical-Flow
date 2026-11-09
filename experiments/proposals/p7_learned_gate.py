"""Proposal 3, learned: how much of the oracle static-gate ceiling does a trained gate recover?

The P3 box probe (flowscape:test) measured the ceiling of zeroing the prediction where the
GT is static (< 0.25 deg). The oracle cuts EPE by 18.7 % on scale_fs30k s7 and by 8.0 % on
oslo_pretrain_ft s7. A training-free proxy (zero where the predicted motion is < 0.5 deg)
recovers only 3.3 % / 2.4 %.

This script keeps OSLO frozen and trains a small ERP CNN that outputs a per-pixel static
probability g. The gated flow is pred * (1 - g). Inputs at H/4: both frames, |frame1 - frame2|,
the OSLO readout, log(1 + predicted motion in deg) and latitude / longitude channels. The
logit is upsampled to the output grid. Loss = in-raster EPE of the gated flow (GT du wrapped)
+ 0.1 * BCE against the static label. Overshoot pixels are left out of the loss, since the
polar-band head (p6) handles them, but they are still scored.

Rows on flowscape:test:
  base          the checkpoint's readout (= its benchmark row)
  gate_soft     pred * (1 - g)
  gate_hard     pred zeroed where g > 0.5
  oracle_gate   pred zeroed where GT < --static-deg (the ceiling, as in p3)
  mag_gate_0.5  pred zeroed where the predicted motion < 0.5 deg (the proxy, as in p3)

Pre-registered reading, on the scale_fs30k s7 checkpoint:
  success    gate_soft recovers >= half of the oracle gain (EPE change <= -9 %) and moving-pixel
             EPE rises by <= 2 %. Next step: train the gate jointly with OSLO.
  promising  between the proxy and half of the oracle gain.
  kill       no better than mag_gate_0.5.
The other checkpoints say whether the gain depends on how badly a checkpoint does on static pixels.

RAFT control (--predictor raft --tv-checkpoint ...): the same gate on a frozen trained RAFT-small
(raft_erp.pt, full-res ERP, identity transform, as in its benchmark row). The gate is a head that
any predictor can take, so an OSLO-vs-RAFT comparison is only fair with both gated. Reading: if the
gated OSLO-RAFT gap stays >= the ungated gap, the gate does not change the comparison.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
from p3_gate_probe import auroc
from p6_polar_band_head import FLOW_NORM, CConv, build_oslo_runner, wrap_du
from spherical_flow.raft_adapter import load_raft_model, predict_raft_flow

DOWN = 4  # gate-net input resolution = output / DOWN
LOSS_DOWN = 2  # training loss is computed on a point-sampled grid at output / LOSS_DOWN
IN_CH = 14


def motion_deg(flow: torch.Tensor) -> torch.Tensor:
    """Angular motion of an ERP flow field [H, W, 2] in degrees (official protocol geometry)."""
    return torch.rad2deg(C.ee.official_maps(torch.zeros_like(flow), flow)[1])


def gate_inputs(rec, pred: torch.Tensor) -> torch.Tensor:
    """[IN_CH, H/DOWN, W/DOWN] float16 input of the gate net."""
    f1 = torch.from_numpy(np.ascontiguousarray(rec["frame1"])).float().permute(2, 0, 1) / 255 - 0.5
    f2 = torch.from_numpy(np.ascontiguousarray(rec["frame2"])).float().permute(2, 0, 1) / 255 - 0.5
    p = pred.permute(2, 0, 1) / FLOW_NORM
    mag = torch.log1p(motion_deg(pred))[None]
    diff = (f1 - f2).abs().mean(0, keepdim=True)
    x = F.avg_pool2d(torch.cat([f1, f2, diff, p, mag], 0)[None], DOWN)[0]
    h, w = x.shape[-2:]
    lat = (0.5 - (torch.arange(h, dtype=torch.float32) + 0.5) / h) * math.pi
    lon = (torch.arange(w, dtype=torch.float32) + 0.5) / w * 2 * math.pi
    coords = torch.stack([torch.sin(lat).view(h, 1).expand(h, w), torch.cos(lat).view(h, 1).expand(h, w),
                          torch.sin(lon).expand(h, w), torch.cos(lon).expand(h, w)])
    return torch.cat([x, coords], 0).half()


class GateNet(nn.Module):
    """Small U-net on the H/4 ERP grid -> one static logit per cell."""

    def __init__(self, cin=IN_CH, c=24):
        super().__init__()
        self.e0 = nn.Sequential(CConv(cin, c), CConv(c, c))
        self.e1 = nn.Sequential(CConv(c, 2 * c, (2, 2)), CConv(2 * c, 2 * c))
        self.e2 = nn.Sequential(CConv(2 * c, 2 * c, (2, 2)), CConv(2 * c, 2 * c))
        self.e3 = nn.Sequential(CConv(2 * c, 4 * c, (2, 2)), CConv(4 * c, 4 * c, dil=(1, 2)),
                                CConv(4 * c, 4 * c, dil=(2, 4)), CConv(4 * c, 4 * c, dil=(1, 8)))
        self.d2 = CConv(6 * c, 2 * c)
        self.d1 = CConv(4 * c, 2 * c)
        self.d0 = CConv(3 * c, c)
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -4.0)  # starts as the identity (g ~ 0.02)

    def forward(self, x):
        x0 = self.e0(x)
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)

        def up(a, b):
            return torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)

        y = self.d2(up(x3, x2))
        y = self.d1(up(y, x1))
        y = self.d0(up(y, x0))
        return self.out(y)


def logit_at(net, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Gate logits [B, 1, h, w] (bilinear upsampling of the H/4 output)."""
    return F.interpolate(net(x), size=(h, w), mode="bilinear", align_corners=False)


def train_sample(rec, pred: torch.Tensor, static_deg: float):
    gt = torch.from_numpy(wrap_du(np.asarray(rec["flow"], dtype=np.float32), pred.shape[1]))
    raw = torch.from_numpy(np.ascontiguousarray(rec["flow"])).float()
    inraster = ~C.ee.overshoot_masks(raw)["overshoot"]
    static = motion_deg(raw) < static_deg
    o = LOSS_DOWN // 2
    sl = (slice(o, None, LOSS_DOWN), slice(o, None, LOSS_DOWN))
    return {"x": gate_inputs(rec, pred), "pred": pred[sl].half(), "gt": gt[sl].half(),
            "mask": (inraster[sl].to(torch.uint8) | (static[sl].to(torch.uint8) << 1))}


def collate(batch, device):
    x = torch.stack([b["x"] for b in batch]).float().to(device)
    pred = torch.stack([b["pred"] for b in batch]).float().permute(0, 3, 1, 2).to(device)
    gt = torch.stack([b["gt"] for b in batch]).float().permute(0, 3, 1, 2).to(device)
    m = torch.stack([b["mask"] for b in batch]).to(device)
    return x, pred, gt, (m & 1).bool()[:, None], (m & 2).bool()[:, None]


def build_raft_runner(args):
    """Frozen trained TorchVision RAFT (raft_erp.pt) -> function(record) -> ERP flow [H, W, 2]."""
    device = torch.device(args.device)
    raft, transforms, _, _ = load_raft_model(args.raft_model, "none", device)
    raft.load_state_dict(torch.load(args.tv_checkpoint, map_location=device)["model"])
    raft.eval()

    def run(rec):
        f1 = torch.from_numpy(np.ascontiguousarray(rec["frame1"])).permute(2, 0, 1).contiguous()
        f2 = torch.from_numpy(np.ascontiguousarray(rec["frame2"])).permute(2, 0, 1).contiguous()
        with torch.no_grad():
            flow = predict_raft_flow(raft, transforms, f1[None], f2[None], device, "identity")
        return flow[0].permute(1, 2, 0).float().cpu().contiguous()

    return run


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--predictor", default="oslo", choices=["oslo", "raft"])
    p.add_argument("--init-checkpoint", default="", help="OSLO checkpoint (--predictor oslo)")
    p.add_argument("--tv-checkpoint", default="", help="trained RAFT raft_erp.pt (--predictor raft)")
    p.add_argument("--raft-model", default="raft_small", choices=["raft_small", "raft_large"])
    p.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--static-deg", type=float, default=0.25)
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--train-split", default="train", help="raw-FlowScape split (local runs)")
    p.add_argument("--train-pairs", type=int, default=1500)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--bce-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if args.predictor == "raft":
        if not args.tv_checkpoint:
            raise SystemExit("--predictor raft needs --tv-checkpoint")
        oslo = build_raft_runner(args)
    else:
        if not args.init_checkpoint:
            raise SystemExit("--predictor oslo needs --init-checkpoint")
        oslo = build_oslo_runner(args)
    ckpt_name = args.tv_checkpoint if args.predictor == "raft" else args.init_checkpoint

    targs = argparse.Namespace(**{**vars(args), "split": args.train_split, "sources": args.train_sources,
                                  "stride": args.train_stride, "max_pairs": args.train_pairs})
    train, start = [], time.time()
    for i, rec in enumerate(C.iter_pairs(targs)):
        train.append(train_sample(rec, oslo(rec), args.static_deg))
        if (i + 1) % 200 == 0:
            print(f"cached {i + 1} train pairs ({time.time() - start:.0f}s)", flush=True)
    static_train = float(np.mean([((b["mask"] & 2) > 0).float().mean().item() for b in train]))
    print(f"train pairs {len(train)}, static fraction {static_train:.3f}", flush=True)

    net = GateNet(c=args.width).to(device)
    n_params = sum(t.numel() for t in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    start = time.time()
    for step in range(args.steps):
        x, pred, gt, inr, static = collate([train[j] for j in rng.integers(0, len(train), args.batch)], device)
        logit = logit_at(net, x, *pred.shape[-2:])
        g = torch.sigmoid(logit)
        epe = (pred * (1 - g) - gt).norm(dim=1, keepdim=True)
        loss_epe = epe[inr].mean()
        bce = F.binary_cross_entropy_with_logits(logit, static.float())
        loss = loss_epe + args.bce_weight * bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            base_l = (pred - gt).norm(dim=1, keepdim=True)[inr].mean().item()
            print(f"step {step} in-raster EPE {loss_epe.item():.3f} (oslo {base_l:.3f}) bce {bce.item():.3f} "
                  f"gate mean {g.mean().item():.3f} ({time.time() - start:.0f}s)", flush=True)
    del train
    net.eval()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "args": vars(args)}, out_dir / "gate_net.pt")

    scorer = C.Scorer()
    auc_learned, auc_mag, start = [], [], time.time()
    for i, rec in enumerate(C.iter_pairs(args)):
        pred = oslo(rec)
        flow = rec["flow"]
        h, w = pred.shape[:2]
        raw = torch.from_numpy(np.ascontiguousarray(flow)).float()
        static = motion_deg(raw) < args.static_deg
        pr_deg = motion_deg(pred)
        with torch.no_grad():
            logit = logit_at(net, gate_inputs(rec, pred)[None].float().to(device), h, w)[0, 0].cpu()
        g = torch.sigmoid(logit)
        extra = {"static": static, "moving": ~static}
        scorer.add("base", pred, flow, extra)
        scorer.add("gate_soft", pred * (1 - g)[..., None], flow, extra)
        scorer.add("gate_hard", pred.masked_fill((g > 0.5)[..., None], 0.0), flow, extra)
        scorer.add("oracle_gate", pred.masked_fill(static[..., None], 0.0), flow, extra)
        scorer.add("mag_gate_0.5", pred.masked_fill((pr_deg < 0.5)[..., None], 0.0), flow, extra)
        auc_learned.append(auroc(logit.reshape(-1), static.reshape(-1)))
        auc_mag.append(auroc(-pr_deg.reshape(-1), static.reshape(-1)))
        if (i + 1) % 50 == 0:
            print(f"test pair {i + 1}: base {scorer.per_pair['base'][-1]:.3f} "
                  f"soft {scorer.per_pair['gate_soft'][-1]:.3f} oracle {scorer.per_pair['oracle_gate'][-1]:.3f} "
                  f"({time.time() - start:.0f}s)", flush=True)
    summary = scorer.summary(base_row="base")
    base = summary["base"]["official_all_epe"]
    print(f"\nP7 learned static gate ({n_params} params, {args.steps} steps; static = GT < {args.static_deg} deg; "
          f"AUROC learned {np.nanmean(auc_learned):.3f} vs -|pred| {np.nanmean(auc_mag):.3f}) on "
          f"{args.predictor} {ckpt_name}. " + C.REFERENCE)
    scorer.print_table(summary, extra_cols=[("static", "official_static_epe"), ("moving", "official_moving_epe")])
    mov = summary["base"]["official_moving_epe"]
    for row, m in summary.items():
        print(f"  {row:<14} EPE change vs base {100 * (m['official_all_epe'] - base) / base:+.1f}%  "
              f"moving {100 * (m['official_moving_epe'] - mov) / mov:+.1f}%")
    C.save(args.output_dir, {"args": vars(args), "params": n_params, "rows": summary,
                             "auroc_learned": float(np.nanmean(auc_learned)),
                             "auroc_mag": float(np.nanmean(auc_mag)), "static_fraction_train": static_train},
           "p7_learned_gate.json")


if __name__ == "__main__":
    main()
