"""Does a pretrained planar flow network fine-tune better on HEALPix faces than on the ERP raster?

P4 ran a frozen RAFT on the 12 padded HEALPix base faces (the DLWP-HPX representation): in-raster
it tied the ERP run and it was ahead at the equator, with no training. The measured causes of
OSLO-RAFT's distance to the published models are the output grid, the missing pretraining and the
cost of the sphere operator; faces remove all three. This probe trains the missing arm.

Two arms, one script, so that only the representation differs:
  --repr faces   the pairs are rendered on the 12 faces (r8: 256 px + margin, equal-area pixels);
                 a training batch is --face-batch random faces
  --repr erp     the usual full ERP frames, --erp-batch per step
Same pretrained TorchVision RAFT-small, same optimizer and schedule, same pixels per step (7 faces
of 384^2 = 1.03 M, 2 ERP frames = 1.05 M), and the same loss: RAFT's sequence loss on the ANGLE
between the predicted and the true endpoint on the sphere, as a sphere mean (plain mean on the
equal-area faces, cos-latitude weights on the ERP), in units of --loss-unit-deg.

Evaluation, the metric of scripts/orientation_suite.sh: sphere-mean geodesic error in degrees over
valid r6 nodes. Both arms are read at the r6 node centres (bilinear in their own raster).
Views: upright, and haar (one Haar rotation per pair, the same for both arms; there is no rotation
in training, so this asks whether the face representation depends less on the orientation).

Pre-registered reading on flowscape:test, seeds 7 and 11, upright:
  faces ahead   faces error lower than ERP by >= 5 % on both seeds
  ERP ahead     ERP error lower than faces by >= 5 % on both seeds
  tie           anything else
Secondary: the haar / upright ratio of each arm (lower = less orientation-dependent).
Prior, stated before the run: a tie upright is the likely outcome.

Limits: RAFT-small only; flows longer than the face margin cannot be matched on a face; the margin
continues the chart geodesically instead of copying the neighbour face, as in P4.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

import common as C
from p13_nested_residual import sample_erp
from p4_faces_probe import spread_bits, validate_chart
from run_erp_raft_train import rotate_erp_pair
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.geometry import equirectangular_pixels_to_unit_vectors, expmap
from spherical_flow.shard_dataset import sample_pair_to_nodes
from spherical_flow.so3_augment import sample_rotation

JRLL = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
JPLL = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7]


def chart(jr0: torch.Tensor, jp: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """HEALPix face coordinates (x, y in [0, 1]) -> unit vectors (xyf2loc). jr0 / jp: JRLL / JPLL of each element."""
    jr = jr0 - x - y
    nr = torch.where(jr < 1, jr, torch.where(jr > 3, 4 - jr, torch.ones_like(jr)))
    z = torch.where(jr < 1, 1 - nr * nr / 3, torch.where(jr > 3, nr * nr / 3 - 1, (2 - jr) * 2 / 3))
    tmp = jp * nr + x - y
    tmp = torch.where(tmp < 0, tmp + 8, torch.where(tmp >= 8, tmp - 8, tmp))
    phi = torch.where(nr < 1e-15, torch.zeros_like(tmp), (math.pi / 4) * tmp / nr.clamp_min(1e-15))
    s = torch.sqrt((1 - z * z).clamp_min(0))
    return torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], -1)


def chart_ext(jr0, jp, x, y, eps=1e-6, h=1e-4):
    """The chart continued outside [0, 1]^2 by a geodesic step along the edge Jacobian (as in P4)."""
    xc, yc = x.clamp(eps, 1 - eps), y.clamp(eps, 1 - eps)
    d0 = chart(jr0, jp, xc, yc)
    sx = torch.where(xc > 0.5, -h, h)
    sy = torch.where(yc > 0.5, -h, h)
    jx = (chart(jr0, jp, xc + sx, yc) - d0) / sx[..., None]
    jy = (chart(jr0, jp, xc, yc + sy) - d0) / sy[..., None]
    v = jx * (x - xc)[..., None] + jy * (y - yc)[..., None]
    v = v - (v * d0).sum(-1, keepdim=True) * d0
    ext = expmap(d0.reshape(-1, 3), v.reshape(1, -1, 3))[0].reshape(d0.shape)
    out = (x != xc) | (y != yc)
    return torch.where(out[..., None], ext, d0)


class Faces:
    """Geometry of the 12 padded faces at one HEALPix order, on a device (fp64)."""

    def __init__(self, order: int, pad: int, device):
        self.nside, self.pad = 1 << order, pad
        self.size = self.nside + 2 * pad
        if self.size % 8:
            raise SystemExit("nside + 2 * pad must be divisible by 8")
        r, c = torch.meshgrid(torch.arange(self.size), torch.arange(self.size), indexing="ij")
        self.cols = c.double().to(device)
        self.rows = r.double().to(device)
        self.jr = torch.tensor(JRLL, dtype=torch.float64, device=device).view(12, 1, 1)
        self.jp = torch.tensor(JPLL, dtype=torch.float64, device=device).view(12, 1, 1)
        self.dirs = self.to_dirs(torch.arange(12, device=device), torch.zeros(12, 2, self.size, self.size, device=device))
        ix, iy = c[pad:pad + self.nside, pad:pad + self.nside] - pad, r[pad:pad + self.nside, pad:pad + self.nside] - pad
        self.nest = (torch.arange(12).view(12, 1, 1) * self.nside ** 2 + spread_bits(ix) + 2 * spread_bits(iy)).to(device)

    def to_dirs(self, face: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Faces [B] and face-pixel flow [B, 2, S, S] -> endpoint directions [B, S, S, 3] (fp64, differentiable)."""
        x = (self.cols + flow[:, 0].double() - self.pad + 0.5) / self.nside
        y = (self.rows + flow[:, 1].double() - self.pad + 0.5) / self.nside
        return chart_ext(self.jr[face], self.jp[face], x, y)


def erp_grid(h: int, w: int, device):
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                          torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
    return u, v


def erp_endpoints(flow: torch.Tensor) -> torch.Tensor:
    """ERP pixel flow [B, 2, H, W] -> endpoint directions [B, H, W, 3] (differentiable in the flow)."""
    h, w = flow.shape[-2:]
    u, v = erp_grid(h, w, flow.device)
    return equirectangular_pixels_to_unit_vectors(u + flow[:, 0], v + flow[:, 1], h, w)


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    chord = torch.sqrt(((a - b) ** 2).sum(-1) + 1e-12)
    return torch.rad2deg(2.0 * torch.asin((0.5 * chord).clamp(max=1.0 - 1e-7)))


def gt_fields(flow: torch.Tensor, valid: torch.Tensor):
    """GT ERP flow [B, 2, H, W] -> ambient displacement [B, H, W, 3], validity [B, H, W, 1] (endpoint inside the raster)."""
    h, w = flow.shape[-2:]
    u, v = erp_grid(h, w, flow.device)
    base = equirectangular_pixels_to_unit_vectors(u, v, h, w)
    inside = (v + flow[:, 1] >= 0) & (v + flow[:, 1] <= h - 1) & valid
    return erp_endpoints(flow) - base, inside.float().unsqueeze(-1)


def normalize(img_u8: torch.Tensor) -> torch.Tensor:
    return img_u8.float() / 127.5 - 1.0


def render_faces(geo: Faces, img: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
    """ERP fields [B, H, W, C] -> the chosen face of each element, [B, S, S, C]."""
    s = geo.size
    return sample_erp(img, geo.dirs[face].reshape(len(face), s * s, 3).float()).reshape(len(face), s, s, -1)


CENTRAL = [3, 6, 9, 12]  # NESTED (Z-order) indices of the 4 r8 pixels around the centre of an r6 pixel


def to_r6(end_r8: torch.Tensor, pts8: torch.Tensor, pts6: torch.Tensor) -> torch.Tensor:
    """Endpoints at the r8 node centres [N8, 3] -> endpoints at the r6 node centres.

    An r6 centre is the common corner of its 4 central r8 children, so their mean is the bilinear
    value of the face field there: a point read, like the ERP arm's bilinear read at the same centre.
    """
    disp = (end_r8 - pts8).reshape(pts6.shape[0], 16, 3)[:, CENTRAL].mean(1)
    out = pts6 + disp
    return out / out.norm(dim=-1, keepdim=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--repr", default="faces", choices=["faces", "erp"])
    p.add_argument("--face-order", type=int, default=8)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--face-batch", type=int, default=7)
    p.add_argument("--erp-batch", type=int, default=2)
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--train-pool", type=int, default=1200, help="train pairs kept in memory (shards only)")
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1.25e-4)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--loss-unit-deg", type=float, default=0.3, help="the loss is the endpoint angle in this unit (about one pixel)")
    p.add_argument("--eval-views", nargs="+", default=["upright", "haar"])
    p.add_argument("--max-angle-deg", type=float, default=60.0, help="pixels with a longer true motion are left out of the loss (RAFT's max_flow)")
    p.add_argument("--eval-at", default="", help="comma list of extra steps to evaluate at")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    err = validate_chart(4)
    if err > 1e-4:
        raise SystemExit("face chart does not match HEALPix NESTED pixel centres")
    geo = Faces(args.face_order, args.pad, device) if args.repr == "faces" else None
    pts8 = healpix_unit_vectors(8)
    pts6 = healpix_unit_vectors(6)
    be6, bn6 = tangent_basis(pts6)
    if geo is not None and geo.nside != 256:
        raise SystemExit("the evaluation reads the faces at the r8 nodes: --face-order must be 8")

    model = torchvision.models.optical_flow.raft_small(weights="DEFAULT").to(device)
    params = sum(q.numel() for q in model.parameters())
    print(f"raft_small pretrained, {params} params, repr {args.repr}"
          + (f", faces r{args.face_order} pad {args.pad} ({geo.size}^2 x {args.face_batch})" if geo else f", erp x {args.erp_batch}"),
          flush=True)

    def load(rec):
        return (torch.from_numpy(np.ascontiguousarray(rec["frame1"])), torch.from_numpy(np.ascontiguousarray(rec["frame2"])),
                torch.from_numpy(np.ascontiguousarray(rec["flow"])).float(), torch.from_numpy(np.ascontiguousarray(rec["valid"])))

    if args.shards:
        pargs = argparse.Namespace(shards=args.shards, sources=args.train_sources, stride=args.train_stride,
                                   max_pairs=args.train_pool)
        pool = [load(rec) for rec in C.iter_pairs(pargs)]
        loaders = [lambda s=s: s for s in pool]
    else:
        loaders = [lambda ld=ld: load(ld(None)) for ld in C._raw_pairs(Path(args.raw_root), "train", args.weathers.split(","))]
    print(f"train pairs available: {len(loaders)}", flush=True)

    def batch_of(n):
        items = [rng.choice(loaders)() for _ in range(n)]
        f1, f2 = (torch.stack([it[k] for it in items]).to(device) for k in (0, 1))                 # [B, H, W, 3] uint8
        flow = torch.stack([it[2] for it in items]).to(device).permute(0, 3, 1, 2)                 # [B, 2, H, W]
        valid = torch.stack([it[3] for it in items]).to(device).bool()
        return f1, f2, flow, valid

    def predict(f1, f2, iters):
        """uint8 [B, H, W, 3] ERP frames -> list of (endpoint directions, GT-free) per iteration, and helpers."""
        return model(normalize(f1).permute(0, 3, 1, 2), normalize(f2).permute(0, 3, 1, 2), num_flow_updates=iters)

    def loss_fn(f1, f2, flow, valid):
        disp, ok = gt_fields(flow, valid)
        if geo is None:
            preds = predict(f1, f2, args.iters)
            h, w = flow.shape[-2:]
            u, v = erp_grid(h, w, device)
            target = equirectangular_pixels_to_unit_vectors(u, v, h, w) + disp
            target = target / target.norm(dim=-1, keepdim=True)
            lat = (0.5 - (v + 0.5) / h) * math.pi
            weight = torch.cos(lat)[None] * ok[..., 0]
            ends = [erp_endpoints(q) for q in preds]
        else:
            face = torch.randint(0, 12, (f1.shape[0],), device=device)
            a, b = render_faces(geo, f1.float(), face), render_faces(geo, f2.float(), face)
            preds = predict(a, b, args.iters)
            fields = render_faces(geo, torch.cat([disp, ok], dim=-1), face)
            target = geo.dirs[face] + fields[..., :3].double()
            target = target / target.norm(dim=-1, keepdim=True)
            weight = (fields[..., 3] > 0.999).double()
            ends = [geo.to_dirs(face, q) for q in preds]
        base = geo.dirs[face] if geo is not None else equirectangular_pixels_to_unit_vectors(u, v, h, w)
        weight = weight * (angle_deg(base, target) <= args.max_angle_deg)
        total = 0.0
        for i, end in enumerate(ends):
            ang = angle_deg(end, target) / args.loss_unit_deg
            total = total + args.gamma ** (len(ends) - i - 1) * (ang * weight).sum() / weight.sum().clamp_min(1)
        return total.float()

    @torch.no_grad()
    def node_endpoints(f1, f2):
        """One ERP pair (uint8 [H, W, 3] on device) -> predicted endpoints at r6 nodes [N6, 3] (CPU, fp32)."""
        if geo is None:
            flow = predict(f1[None], f2[None], args.iters)[-1]
            h, w = flow.shape[-2:]
            u, v = erp_grid(h, w, device)
            disp = erp_endpoints(flow) - equirectangular_pixels_to_unit_vectors(u, v, h, w)
            d6 = pts6.to(device)
            end = d6 + sample_erp(disp, d6[None])[0]
            return (end / end.norm(dim=-1, keepdim=True)).float().cpu()
        else:
            face = torch.arange(12, device=device)
            a = render_faces(geo, f1[None].float().expand(12, -1, -1, -1), face)
            b = render_faces(geo, f2[None].float().expand(12, -1, -1, -1), face)
            flow = predict(a, b, args.iters)[-1]
            inner = slice(geo.pad, geo.pad + geo.nside)
            ends = geo.to_dirs(face, flow)[:, inner, inner]
            end = torch.zeros(pts8.shape[0], 3, dtype=torch.float64, device=device)
            end[geo.nest.reshape(-1)] = ends.reshape(-1, 3)
        return to_r6(end.float().cpu(), pts8, pts6)

    eargs = argparse.Namespace(**vars(args))
    egen = torch.Generator().manual_seed(1234)
    eye = torch.eye(3)
    eval_set = []
    for rec in C.iter_pairs(eargs):
        chw = (torch.from_numpy(rec["frame1"]).permute(2, 0, 1), torch.from_numpy(rec["frame2"]).permute(2, 0, 1),
               torch.from_numpy(np.ascontiguousarray(rec["flow"])).permute(2, 0, 1).float(),
               torch.from_numpy(np.ascontiguousarray(rec["valid"])).bool())
        eval_set.append((chw, sample_rotation(egen, uniform_so3=True)))
    print(f"eval pairs: {len(eval_set)}, views {args.eval_views}", flush=True)

    def evaluate(tag):
        model.eval()
        out = {}
        for view in args.eval_views:
            tot = {"err": 0.0, "zero": 0.0, "n": 0}
            for chw, rot in eval_set:
                f1, f2, flow, valid = rotate_erp_pair(*chw, rot if view == "haar" else eye)
                n = sample_pair_to_nodes(f1.permute(1, 2, 0).float() / 255, f2.permute(1, 2, 0).float() / 255,
                                         flow.permute(1, 2, 0), valid, pts6, be6, bn6)
                ok = n["valid"].bool().reshape(-1)
                gt = n["endpoint"].reshape(-1, 3)
                pred = node_endpoints(f1.permute(1, 2, 0).to(device), f2.permute(1, 2, 0).to(device))
                tot["err"] += angle_deg(pred, gt)[ok].sum().item()
                tot["zero"] += angle_deg(pts6, gt)[ok].sum().item()
                tot["n"] += int(ok.sum())
            out[view] = {"deg": tot["err"] / tot["n"], "zero_deg": tot["zero"] / tot["n"]}
        print(f"[eval {tag}] " + " | ".join(f"{v} {m['deg']:.4f} deg (zero {m['zero_deg']:.3f})" for v, m in out.items())
              + f" ({time.time() - start:.0f}s)", flush=True)
        return out

    start = time.time()
    history = [{"step": 0, **evaluate("pretrained")}]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
                                                anneal_strategy="linear")
    n_batch = args.face_batch if geo is not None else args.erp_batch
    eval_at = {int(x) for x in args.eval_at.split(",") if x}
    for step in range(1, args.steps + 1):
        model.train()
        loss = loss_fn(*batch_of(n_batch))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step} loss {loss.item():.4f} ({time.time() - start:.0f}s)", flush=True)
        if step in eval_at:
            history.append({"step": step, **evaluate(f"step {step}")})
    history.append({"step": args.steps, **evaluate("fine-tuned")})

    first, last = history[0], history[-1]
    print(f"\nP14 {args.repr} seed {args.seed}: " + " | ".join(
        f"{v} {first[v]['deg']:.4f} -> {last[v]['deg']:.4f} deg" for v in args.eval_views))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out / "raft_small.pth")
    C.save(args.output_dir, {"args": vars(args), "history": history}, "p14_faces_finetune.json")


if __name__ == "__main__":
    main()
