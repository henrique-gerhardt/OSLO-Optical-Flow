"""Hierarchy as the flow representation: a frozen model's field plus residuals on finer NESTED levels.

Direction under test (reports/Horizontes HEALPix para fluxo óptico.md, item 3): the works where
HEALPix pays off use the NESTED hierarchy as the representation (a coarse field plus per-level
residuals that a network updates), not equal area. OSLO-RAFT uses the hierarchy only for pooling:
it estimates at r4 and convex-upsamples to r6. This probe keeps a trained predictor frozen and
adds the residual levels on top of its field:

  level L:  x = [frame1(L), frame2 warped by the current flow (L), their difference, flow / node spacing]
            flow_L = flow_in + node_spacing_L * net_L(x)          (SDPAConv net, zero-initialised output)
  to L+1:   the field is read at the child nodes with the HEALPix stencil and the next level repeats.

The flow is detached between levels, so each level learns a correction of the field it is given
(the SpyNet scheme on the NESTED grid). Zero residual = "same as the parent".

--levels "6 7"   the hierarchy: correct at r6, then at its children (r7)
--levels "6 6"   flat control with the same parameters: two corrections, both at r6
--predictor      oslo (node flow at its supervision grid), raft (ERP flow read at the r6 nodes),
                 zero (wiring test only)

Metric: sphere-mean geodesic error in degrees over valid nodes of the first level (r6, equal
area), as in scripts/orientation_suite.sh. A finer field is brought back to r6 by the NESTED mean
of its children. Rows: zero, base (the frozen predictor), s1, s2, ... (after each level).

Pre-registered reading on flowscape:test, fs30k pairs (OSLO noso3_erppx vs RAFT-small fs, seeds 7/11):
  H1 hierarchy helps OSLO      OSLO "6 7" final <= 0.90 x OSLO base, on both seeds
  H2 it is the hierarchy       OSLO "6 7" final lower than OSLO "6 6" final by >= 3 %, on both seeds
  H3 parity on the sphere      OSLO "6 7" final <= RAFT-small base, on both seeds
  H4 specific or generic       relative gain of OSLO "6 7" larger than RAFT-small "6 7" on both seeds
                               = specific; otherwise the levels are a generic refinement (as the
                               band head and the static gate were, RAFT_PARITY 11.25)
Prior, stated before the run: H1 likely, H4 likely generic.

Limits: the trunk is frozen and was not trained for this output; two levels; one head seed; the
head sees RGB, not the trunk's features.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.erp_readout import bilinear_node_weights, erp_pixel_directions, node_weights_at_dirs, nodes_to_erp_flow
from spherical_flow.geometry import expmap, logmap, points_to_equirectangular_pixels, tangent_components_to_3d
from spherical_flow.healpix_pyramid import healpix_neighbor_graph
from spherical_flow.shard_dataset import _to_chw_free_float, sample_pair_to_nodes
from spherical_flow.models import SDPAConv


# ----------------------------------------------------------------------------- geometry
class Level:
    def __init__(self, res: int, device, frame_pool: int):
        self.res = res
        self.points = healpix_unit_vectors(res)
        self.east, self.north = tangent_basis(self.points)
        idx, wgt, _ = healpix_neighbor_graph(res, self.points, fill_missing=True)
        self.spacing = math.sqrt(4.0 * math.pi / self.points.shape[0])  # rad
        self.frame_pool = frame_pool
        self.lat = torch.asin(self.points[:, 2].clamp(-1.0, 1.0))
        self.dev = {k: v.to(device) for k, v in dict(points=self.points, east=self.east, north=self.north,
                                                    idx=idx, wgt=wgt).items()}


def frame_pool_for(res: int, height: int) -> int:
    """Area-pooling factor of the ERP frame so one pixel is about one node spacing or finer."""
    spacing_deg = math.degrees(math.sqrt(4.0 * math.pi / (12 * 4 ** res)))
    return max(1, int(spacing_deg / (180.0 / height)))


def sample_erp(img: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of ERP fields [B, H, W, C] at unit directions [B, M, 3] -> [B, M, C]."""
    b, h, w, c = img.shape
    m = dirs.shape[1]
    u, v = points_to_equirectangular_pixels(dirs.reshape(-1, 3), h, w)
    u, v = u.reshape(b, m), v.reshape(b, m).clamp(0.0, h - 1.0)
    u0, v0 = torch.floor(u), torch.floor(v)
    fu, fv = (u - u0).unsqueeze(-1), (v - v0).unsqueeze(-1)
    u0 = u0.long() % w
    u1 = (u0 + 1) % w
    v0 = v0.long().clamp(0, h - 1)
    v1 = (v0 + 1).clamp(max=h - 1)
    flat = img.reshape(b, h * w, c)

    def take(vv, uu):
        return torch.gather(flat, 1, (vv * w + uu).unsqueeze(-1).expand(-1, -1, c))

    top = take(v0, u0) * (1 - fu) + take(v0, u1) * fu
    bot = take(v1, u0) * (1 - fu) + take(v1, u1) * fu
    return top * (1 - fv) + bot * fv


def endpoints(level: Level, flow: torch.Tensor) -> torch.Tensor:
    d = level.dev
    return expmap(d["points"], tangent_components_to_3d(flow, d["east"], d["north"]))


def to_components(level: Level, ambient: torch.Tensor) -> torch.Tensor:
    """Ambient vectors [B, N, 3] at the level's nodes -> tangent components [B, N, 2]."""
    d = level.dev
    return torch.stack([(ambient * d["east"]).sum(-1), (ambient * d["north"]).sum(-1)], dim=-1)


def lift(flow: torch.Tensor, src: Level, dst: Level, stencil) -> torch.Tensor:
    """Tangent flow on src nodes -> tangent flow on dst nodes (HEALPix stencil of the benchmark readout)."""
    if src.res == dst.res:
        return flow
    idx, w = stencil
    amb = tangent_components_to_3d(flow, src.dev["east"], src.dev["north"])       # [B, Ns, 3]
    amb = (amb[:, idx] * w[None, :, :, None]).sum(dim=1)                          # [B, Nd, 3]
    return to_components(dst, amb)


def reduce(flow: torch.Tensor, src: Level, dst: Level) -> torch.Tensor:
    """Tangent flow on a finer level -> the coarser level, by the NESTED mean of the children."""
    if src.res == dst.res:
        return flow
    amb = tangent_components_to_3d(flow, src.dev["east"], src.dev["north"])
    amb = amb.reshape(amb.shape[0], dst.points.shape[0], -1, 3).mean(dim=2)
    return to_components(dst, amb)


def geodesic_deg(level: Level, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between the endpoints of two tangent flows [B, N, 2] -> [B, N] degrees (haversine form)."""
    chord = (endpoints(level, a) - endpoints(level, b)).norm(dim=-1)
    return torch.rad2deg(2.0 * torch.asin((0.5 * chord).clamp(max=1.0)))


# ----------------------------------------------------------------------------- model
class ResidualNet(nn.Module):
    def __init__(self, channels: int, layers: int):
        super().__init__()
        dims = [11] + [channels] * layers
        self.convs = nn.ModuleList(SDPAConv(a, b, kernel_size=9, node_dim=1) for a, b in zip(dims[:-1], dims[1:]))
        self.out = SDPAConv(channels, 2, kernel_size=9, node_dim=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, idx, wgt):
        for conv in self.convs:
            x = F.relu(conv(x, idx, wgt))
        return self.out(x, idx, wgt)


class Hierarchy(nn.Module):
    def __init__(self, levels, stencils, channels: int, layers: int):
        super().__init__()
        self.levels, self.stencils = levels, stencils
        self.nets = nn.ModuleList(ResidualNet(channels, layers) for _ in levels)

    def forward(self, img1, img2, flow):
        """img1/img2 [B, H, W, 3] in [0, 1]; flow [B, N0, 2] on levels[0]. Returns the flow after each level."""
        outs = []
        for i, level in enumerate(self.levels):
            if i > 0:
                flow = lift(flow, self.levels[i - 1], level, self.stencils[i])
            flow = flow.detach()
            a, b = img1, img2
            if level.frame_pool > 1:
                a = F.avg_pool2d(a.permute(0, 3, 1, 2), level.frame_pool).permute(0, 2, 3, 1)
                b = F.avg_pool2d(b.permute(0, 3, 1, 2), level.frame_pool).permute(0, 2, 3, 1)
            pts = level.dev["points"].unsqueeze(0).expand(flow.shape[0], -1, -1)
            f1 = sample_erp(a, pts)
            f2 = sample_erp(b, endpoints(level, flow))
            x = torch.cat([f1, f2, f1 - f2, flow / level.spacing], dim=-1)
            flow = flow + level.spacing * self.nets[i](x, level.dev["idx"], level.dev["wgt"])
            outs.append(flow)
        return outs


# ----------------------------------------------------------------------------- predictors
def build_oslo(args, first: Level):
    sys.argv = ["x", "--predictor", "oslo", "--init-checkpoint", args.init_checkpoint,
                "--pyramid-cache", args.pyramid_cache, "--eval-iters", str(args.eval_iters), "--device", args.device]
    eargs = C.ee.parse_args()
    C.ee.adopt_checkpoint_grids(eargs)
    if eargs.resolution != first.res:
        raise SystemExit(f"the checkpoint outputs r{eargs.resolution}; --levels must start there")
    device = torch.device(args.device)
    model, pyramid = C.ee.build_oslo(eargs, device)
    model.eval()
    ret = pyramid.retina_level.points.cpu()
    rbe, rbn = tangent_basis(ret)

    def run(rec, f1, f2, gt, valid):
        nodes = sample_pair_to_nodes(f1, f2, gt, valid, ret, rbe, rbn, target_points=first.points)
        with torch.no_grad():
            return model(nodes["frame1"][None].to(device), nodes["frame2"][None].to(device), pyramid,
                         iters=eargs.eval_iters)[-1][0].float().cpu(), None

    return run


def build_raft(args, first: Level):
    from p10_prediction_probe import erp_flow_to_node_endpoints
    from p6_polar_band_head import build_raft_runner

    raft = build_raft_runner(args)
    grid = first.points.double()

    def run(rec, f1, f2, gt, valid):
        erp = raft(rec)
        end = erp_flow_to_node_endpoints(erp.numpy(), grid).float()
        return logmap(first.points, end, first.east, first.north)[0], erp

    return run


def prepare(rec, predictor, levels):
    """One record -> cached tensors (CPU): uint8 frames, init flow on levels[0], GT flow/valid per level."""
    f1, f2 = _to_chw_free_float(rec["frame1"]), _to_chw_free_float(rec["frame2"])
    gt = torch.from_numpy(np.ascontiguousarray(rec["flow"])).float()
    valid = torch.from_numpy(np.ascontiguousarray(rec["valid"]))
    out = {"f1": torch.from_numpy(np.ascontiguousarray(rec["frame1"])),
           "f2": torch.from_numpy(np.ascontiguousarray(rec["frame2"])), "gt": [], "valid": []}
    done = {}
    for level in levels:
        if level.res not in done:
            n = sample_pair_to_nodes(f1, f2, gt, valid, level.points, level.east, level.north)
            done[level.res] = (n["flow"].float(), n["valid"].bool())
        out["gt"].append(done[level.res][0])
        out["valid"].append(done[level.res][1])
    if predictor is None:
        out["init"], out["erp"] = torch.zeros_like(out["gt"][0]), None
    else:
        out["init"], out["erp"] = predictor(rec, f1, f2, gt, valid)
    return out


def collate(items, device):
    img = lambda k: torch.stack([it[k] for it in items]).to(device).float().div(255.0)
    return (img("f1"), img("f2"), torch.stack([it["init"] for it in items]).to(device),
            [torch.stack([it["gt"][i] for it in items]).to(device) for i in range(len(items[0]["gt"]))],
            [torch.stack([it["valid"][i] for it in items]).to(device) for i in range(len(items[0]["valid"]))])


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--predictor", default="oslo", choices=["oslo", "raft", "zero"])
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--tv-checkpoint", default="")
    p.add_argument("--raft-model", default="raft_small", choices=["raft_small", "raft_large"])
    p.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--levels", type=int, nargs="+", default=[6, 7])
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--train-split", default="train")
    p.add_argument("--train-pairs", type=int, default=600)
    p.add_argument("--train-stride", type=int, default=7)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--static-deg", type=float, default=0.25)
    p.add_argument("--score-stride", type=int, default=4, help="ERP-px benchmark rows on every k-th eval pair (0 = off)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if any(b < a for a, b in zip(args.levels[:-1], args.levels[1:])):
        raise SystemExit("--levels must not decrease")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    tr = argparse.Namespace(**{**vars(args), "sources": args.train_sources, "split": args.train_split,
                               "stride": args.train_stride, "max_pairs": args.train_pairs})
    first_rec = next(iter(C.iter_pairs(tr)))
    height = first_rec["flow"].shape[0]
    levels = [Level(r, device, frame_pool_for(r, height)) for r in args.levels]
    stencils = [None] + [
        tuple(t.to(device) for t in node_weights_at_dirs(a.res, b.points.double().numpy())) if a.res != b.res else None
        for a, b in zip(levels[:-1], levels[1:])]
    first, last = levels[0], levels[-1]
    print("levels: " + ", ".join(f"r{l.res} ({l.points.shape[0]} nodes, frame pool {l.frame_pool})" for l in levels),
          flush=True)

    predictor = {"oslo": build_oslo, "raft": build_raft, "zero": lambda a, f: None}[args.predictor](args, first)

    start = time.time()
    train = []
    for k, rec in enumerate(C.iter_pairs(tr)):
        train.append(prepare(rec, predictor, levels))
        train[-1]["erp"] = None  # the dense ERP flow is only needed for the benchmark rows of the evaluation
        if (k + 1) % 100 == 0:
            print(f"cached {k + 1} train pairs ({time.time() - start:.0f}s)", flush=True)
    print(f"train pairs: {len(train)}", flush=True)

    model = Hierarchy(levels, stencils, args.channels, args.layers).to(device)
    params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    start = time.time()
    for step in range(1, args.steps + 1):
        img1, img2, init, gts, valids = collate([rng.choice(train) for _ in range(args.batch_size)], device)
        outs = model(img1, img2, init)
        loss, parts = 0.0, []
        for level, out, gt, valid in zip(levels, outs, gts, valids):
            err = torch.rad2deg((out - gt).norm(dim=-1))
            part = (err * valid).sum() / valid.sum().clamp_min(1)
            parts.append(part.item())
            loss = loss + part
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                base = (torch.rad2deg((init - gts[0]).norm(dim=-1)) * valids[0]).sum() / valids[0].sum().clamp_min(1)
            print(f"step {step} base {base.item():.4f} " + " ".join(f"s{i + 1} {v:.4f}" for i, v in enumerate(parts))
                  + f" deg ({time.time() - start:.0f}s)", flush=True)
    del train

    # ---------------------------------------------------------------- evaluation on levels[0] nodes
    model.eval()
    rows = ["zero", "base"] + [f"s{i + 1}" for i in range(len(levels))]
    poles = (first.lat.abs() > math.radians(45.0)).to(device)
    sums = {r: {} for r in rows}
    per_pair = {r: [] for r in rows}
    fine = [0.0, 0.0]  # final field scored on the last level's own nodes
    scorer, erp_cache = C.Scorer(), {}
    n_eval = 0
    for k, rec in enumerate(C.iter_pairs(args)):
        item = prepare(rec, predictor, levels)
        img1, img2, init, gts, valids = collate([item], device)
        with torch.no_grad():
            outs = model(img1, img2, init)
        gt0, valid0 = gts[0], valids[0]
        fields = {"zero": torch.zeros_like(init), "base": init}
        for i, out in enumerate(outs):
            fields[f"s{i + 1}"] = reduce(out, levels[i], first)
        gt_mag = torch.rad2deg(gt0.norm(dim=-1))[0]
        masks = {"all": valid0[0], "poles": valid0[0] & poles, "equator": valid0[0] & ~poles,
                 "static": valid0[0] & (gt_mag < args.static_deg), "moving": valid0[0] & (gt_mag >= args.static_deg)}
        for r in rows:
            err = geodesic_deg(first, fields[r], gt0)[0]
            for name, m in masks.items():
                s = sums[r].setdefault(name, [0.0, 0.0])
                s[0] += err[m].sum().item()
                s[1] += m.sum().item()
            per_pair[r].append((err[masks["all"]].mean()).item())
        err = geodesic_deg(last, outs[-1], gts[-1])[0][valids[-1][0]]
        fine[0] += err.sum().item()
        fine[1] += err.numel()
        if args.score_stride and k % args.score_stride == 0:
            h, w = rec["flow"].shape[:2]
            if (h, w) not in erp_cache:
                erp_cache[(h, w)] = (erp_pixel_directions(h, w),
                                     {l.res: bilinear_node_weights(l.res, h, w) for l in (first, last)})
            dirs, wts = erp_cache[(h, w)]
            read = lambda flow, l: nodes_to_erp_flow(flow[0].float().cpu(), l.points, l.east, l.north, h, w,
                                                     wts[l.res], pixel_dirs=dirs)
            scorer.add("base_nodes", read(init, first), rec["flow"])
            scorer.add("final", read(outs[-1], last), rec["flow"])
            if item["erp"] is not None:
                scorer.add("native_erp", item["erp"], rec["flow"])
        n_eval += 1
        if n_eval % 100 == 0:
            print(f"evaluated {n_eval} pairs ({time.time() - start:.0f}s)", flush=True)

    table = {r: {name: s / n for name, (s, n) in sums[r].items() if n > 0} for r in rows}
    base_pp = np.array(per_pair["base"])
    for r in rows:
        d = np.array(per_pair[r]) - base_pp
        table[r]["paired_delta_vs_base"] = float(d.mean())
        table[r]["paired_se"] = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        table[r]["improved_fraction"] = float((d < 0).mean())
    final = rows[-1]
    gain = 100.0 * (1.0 - table[final]["all"] / table["base"]["all"])
    erp = scorer.summary() if args.score_stride else {}
    print(f"\npredictor={args.predictor} levels={args.levels} head_params={params} eval_pairs={n_eval}")
    print(f"{'row':<8}" + "".join(f"{c:>10}" for c in ("all", "equator", "poles", "static", "moving", "d_base", "se", "better")))
    for r in rows:
        t = table[r]
        print(f"{r:<8}" + "".join(f"{t.get(c, float('nan')):>10.4f}" for c in
                                   ("all", "equator", "poles", "static", "moving", "paired_delta_vs_base", "paired_se",
                                    "improved_fraction")))
    print(f"final on its own r{last.res} nodes: {fine[0] / max(fine[1], 1):.4f} deg; gain of {final} over base: {gain:+.1f} %")
    if erp:
        scorer.print_table(erp)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "levels": args.levels}, out / "nested_residual.pt")
    C.save(args.output_dir, {"args": vars(args), "head_params": params, "eval_pairs": n_eval, "table": table,
                             "final_row": final, "gain_pct": gain, "final_own_level_deg": fine[0] / max(fine[1], 1),
                             "erp": erp}, "p13_nested_residual.json")


if __name__ == "__main__":
    main()
