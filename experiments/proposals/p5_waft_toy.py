"""Proposal 5, minimal: does a cost-volume-free (WAFT-style) update learn on HEALPix?

A small, isolated training experiment (not OSLO-RAFT). Same tiny graph encoder for both arms;
frames sampled at r6, features pooled to the estimation grid r5 (NESTED // 4); flow kept as an
ambient tangent field t at r5, endpoint e = expmap(x, t), 6 update iterations, geodesic loss.

  corr  local cost volume: <f1(x), f2(expmap(e, offset))> over a 5x5 tangent grid of one
        node spacing (frame-1 basis parallel-transported to e), + context + flow -> MLP
        with one neighbour aggregation
  waft  no cost volume: f2 sampled AT the current endpoint (warping), concatenated with f1,
        context and flow -> 2 window-attention blocks over NESTED windows of 64 nodes
        (second block shifted by 32 indices) -> delta

f2 is read at arbitrary directions through a 128x256 ERP map of the node features (fixed
HEALPix stencil, then seam-wrapped bilinear); no gradient through the coordinates, as in RAFT. Both arms train the same steps on the same pairs and
are scored on held-out pairs with the benchmark metrics (zero = floor).

Reading: waft within the seed noise of corr or better, in 0-5 deg and at the poles, means the
update without a cost volume is viable on HEALPix and worth a full-scale arm; clearly worse
means it needs more data/pretraining than this budget (HEAL-SWIN's warning).
"""

from __future__ import annotations

import argparse
import itertools
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
from run_tta_refine_probe import sample_erp
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.erp_readout import bilinear_node_weights, erp_pixel_directions, nodes_to_erp_flow
from spherical_flow.geometry import expmap, logmap
from spherical_flow.healpix_pyramid import healpix_neighbor_graph
from spherical_flow.shard_dataset import sample_pair_to_nodes, _to_chw_free_float


SAMPLE_HW = (128, 256)  # ERP map used to read node features at arbitrary directions


def transport(vec, a, b):
    axis = torch.linalg.cross(a, b)
    s = axis.norm(dim=-1, keepdim=True)
    c = (a * b).sum(-1, keepdim=True)
    k = axis / s.clamp_min(1e-12)
    rot = vec * c + torch.linalg.cross(k, vec) * s + k * (k * vec).sum(-1, keepdim=True) * (1 - c)
    return torch.where(s < 1e-12, vec, rot)


class GraphConv(nn.Module):
    def __init__(self, cin, cout, index):
        super().__init__()
        self.register_buffer("index", torch.cat([torch.arange(index.shape[0]).unsqueeze(1), index], 1))
        self.lin = nn.Linear(9 * cin, cout)

    def forward(self, x):  # [B, N, C]
        g = x[:, self.index]  # [B, N, 9, C]
        return F.gelu(self.lin(g.flatten(2)))


class Encoder(nn.Module):
    def __init__(self, idx6, idx5, c=48, out=64):
        super().__init__()
        self.a1, self.a2 = GraphConv(3, c, idx6), GraphConv(c, c, idx6)
        self.b1, self.b2 = GraphConv(c, out, idx5), GraphConv(out, out, idx5)

    def forward(self, img):  # [B, N6, 3]
        x = self.a2(self.a1(img))
        x = x.reshape(x.shape[0], -1, 4, x.shape[-1]).mean(2)  # NESTED pool r6 -> r5
        return self.b2(self.b1(x))


class WindowAttention(nn.Module):
    def __init__(self, dim, heads=4, win=64, shift=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.win, self.shift = win, shift

    def forward(self, x):  # [B, N, D], N divisible by win
        b, n, d = x.shape
        y = torch.roll(self.n1(x), -self.shift, 1).reshape(b * n // self.win, self.win, d)
        y = self.attn(y, y, y, need_weights=False)[0].reshape(b, n, d)
        x = x + torch.roll(y, self.shift, 1)
        return x + self.mlp(self.n2(x))


class FlowNet(nn.Module):
    def __init__(self, arm, idx6, idx5, pts5, radius=2, dim=64):
        super().__init__()
        self.arm, self.radius = arm, radius
        self.fnet, self.cnet = Encoder(idx6, idx5), Encoder(idx6, idx5)
        self.register_buffer("pts", pts5)
        be, bn = tangent_basis(pts5)
        self.register_buffer("be", be)
        self.register_buffer("bn", bn)
        self.spacing = math.sqrt(4 * math.pi / pts5.shape[0])
        idx, w = bilinear_node_weights(5, SAMPLE_HW[0], SAMPLE_HW[1])
        self.register_buffer("erp_idx", idx)
        self.register_buffer("erp_w", w)
        k = (2 * radius + 1) ** 2
        if arm == "corr":
            self.inp = nn.Linear(k + 64 + 2, dim)
            self.mix = GraphConv(dim, dim, idx5)
            self.blocks = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        else:
            self.inp = nn.Linear(64 + 64 + 64 + 2, dim)
            self.blocks = nn.Sequential(WindowAttention(dim, shift=0), WindowAttention(dim, shift=32))
        self.out = nn.Linear(dim, 2)

    def to_erp(self, feat):
        """feat [N, C] on r5 nodes -> small ERP map [C, H, W] (fixed HEALPix stencil)."""
        erp = (feat[self.erp_idx] * self.erp_w[..., None]).sum(0)
        return erp.t().reshape(feat.shape[-1], SAMPLE_HW[0], SAMPLE_HW[1])

    @staticmethod
    def sample(erp, dirs):
        """Seam-wrapped bilinear read of the ERP feature map at dirs [M, 3]; no gradient
        through the coordinates."""
        return sample_erp(erp, dirs.detach())

    def forward(self, img1, img2, iters=6):  # [N6, 3] each (batch 1)
        f1, f2 = self.fnet(img1[None])[0], self.fnet(img2[None])[0]
        ctx = self.cnet(img1[None])[0]
        f2 = self.to_erp(f2)
        n = self.pts.shape[0]
        t = torch.zeros(n, 3, device=img1.device)
        outs = []
        for _ in range(iters):
            e = expmap(self.pts, t[None])[0]
            comps = torch.stack([(t * self.be).sum(-1), (t * self.bn).sum(-1)], -1) / self.spacing
            if self.arm == "corr":
                E, N = transport(self.be, self.pts, e), transport(self.bn, self.pts, e)
                costs = []
                for i in range(-self.radius, self.radius + 1):
                    for j in range(-self.radius, self.radius + 1):
                        cand = expmap(e, ((i * E + j * N) * self.spacing)[None])[0]
                        costs.append((f1 * self.sample(f2, cand)).sum(-1) / math.sqrt(f1.shape[-1]))
                h = self.inp(torch.cat([torch.stack(costs, -1), ctx, comps], -1))
                h = self.blocks(self.mix(h[None])[0])
            else:
                h = self.inp(torch.cat([f1, self.sample(f2, e), ctx, comps], -1))
                h = self.blocks(h[None])[0]
            d = self.out(h) * self.spacing
            t = t.detach() + d[:, :1] * self.be + d[:, 1:] * self.bn
            t = t - (t * self.pts).sum(-1, keepdim=True) * self.pts
            outs.append(t)
        return outs


def angle(a, b):
    return torch.atan2(torch.linalg.cross(a, b).norm(dim=-1), (a * b).sum(-1))


def prep(rec, pts6, be6, bn6, pts5):
    gt = torch.from_numpy(np.ascontiguousarray(rec["flow"])).float()
    nodes = sample_pair_to_nodes(_to_chw_free_float(rec["frame1"]), _to_chw_free_float(rec["frame2"]), gt,
                                 torch.from_numpy(np.ascontiguousarray(rec["valid"])), pts6, be6, bn6,
                                 target_points=pts5)
    return nodes


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--train-split", default="train")
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--arms", default="corr,waft")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-pairs", type=int, default=400, help="distinct training pairs cached in memory")
    args = p.parse_args()
    device = torch.device(args.device)
    pts6, pts5 = healpix_unit_vectors(6), healpix_unit_vectors(5)
    be6, bn6 = tangent_basis(pts6)
    idx6 = healpix_neighbor_graph(6, sort_cut_deg=-157.5, fill_missing=True)[0]
    idx5 = healpix_neighbor_graph(5, sort_cut_deg=-157.5, fill_missing=True)[0]

    targs = argparse.Namespace(**{**vars(args), "split": args.train_split, "sources": args.train_sources,
                                  "stride": 1, "max_pairs": args.train_pairs})
    print("caching training pairs ...", flush=True)
    train = [prep(r, pts6, be6, bn6, pts5) for r in C.iter_pairs(targs)]
    test = [(r, prep(r, pts6, be6, bn6, pts5)) for r in C.iter_pairs(args)]
    print(f"train {len(train)} pairs, test {len(test)} pairs", flush=True)

    scorer = C.Scorer()
    be5, bn5 = tangent_basis(pts5)
    wcache = {}
    for rec, _ in test:
        h, w = rec["flow"].shape[:2]
        scorer.add("zero", torch.zeros(h, w, 2), rec["flow"])
    params = {}
    for arm in args.arms.split(","):
        torch.manual_seed(args.seed)
        net = FlowNet(arm, idx6, idx5, pts5).to(device)
        params[arm] = sum(q.numel() for q in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
        rng = np.random.default_rng(args.seed)
        start = time.time()
        for step in range(args.steps):
            s = train[rng.integers(len(train))]
            outs = net(s["frame1"].to(device), s["frame2"].to(device), args.iters)
            gt_end, valid = s["endpoint"].to(device), s["valid"].to(device)
            loss = sum(0.8 ** (len(outs) - 1 - i) * angle(expmap(net.pts, o[None])[0], gt_end)[valid].mean()
                       for i, o in enumerate(outs))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 100 == 0 or step == args.steps - 1:
                print(f"[{arm}] step {step} loss {loss.item():.5f} ({time.time() - start:.0f}s)", flush=True)
        net.eval()
        with torch.no_grad():
            for rec, s in test:
                h, w = rec["flow"].shape[:2]
                if (h, w) not in wcache:
                    wcache[(h, w)] = (bilinear_node_weights(5, h, w), erp_pixel_directions(h, w))
                t = net(s["frame1"].to(device), s["frame2"].to(device), args.iters)[-1].cpu()
                comps = logmap(pts5, expmap(pts5, t[None])[0], be5, bn5)[0]
                erp = nodes_to_erp_flow(comps, pts5, be5, bn5, h, w, wcache[(h, w)][0], pixel_dirs=wcache[(h, w)][1])
                scorer.add(arm, erp, rec["flow"])
    summary = scorer.summary(base_row="zero")
    print(f"\nP5 toy: corr vs waft update on HEALPix r5 ({args.steps} steps, params {params}). " + C.REFERENCE)
    scorer.print_table(summary)
    C.save(args.output_dir, {"args": vars(args), "params": params, "rows": summary}, "p5_waft_toy.json")


if __name__ == "__main__":
    main()
