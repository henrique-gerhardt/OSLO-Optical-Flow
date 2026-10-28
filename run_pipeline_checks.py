"""End-to-end pipeline checks on real shards and a trained OSLO-RAFT-R checkpoint.

Complements ``run_sanity_checks.py`` (operators with synthetic inputs). Here the input
is real data as the training loader yields it, and the model is a trained checkpoint.
PASS/FAIL per check, summary at the end, exit code 1 if any check fails.

    python run_pipeline_checks.py --shards ../sphereflow-dataprep/shards \
        --checkpoint outputs/oslo_raft_retina_stageA/oslo_raft.pt --sources replica360:val

Checks:
  warp <src>        GT flow is photometrically right: frame 2 sampled at the GT endpoint
                    matches frame 1 better than at zero flow and better than at the
                    sign-flipped endpoint (a flipped convention would invert this)
  warp_loader       the same on the samples the training loader yields (retina frames,
                    supervision targets), without and with SO(3) augmentation: the
                    rotation of frames and targets must stay paired
  batch_indep       a pair's prediction does not depend on the other pairs in the batch
  ckpt_equiv        gradient checkpointing on/off gives the same output and gradients
  grad_reach        one backward reaches every parameter (no dead module)
  yaw_equiv         rotating the input 90 deg about the vertical axis (an exact HEALPix
                    symmetry) rotates the prediction the same way: the model is a
                    convolution on the sphere, not a function of absolute position
  static            identical frames -> (near) zero predicted flow
  vs_zero           the trained model beats the zero-flow baseline on real pairs, in
                    the metric the training run reports
  iters             error per update iteration (should fall, then flatten)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from run_sanity_checks import RESULTS, report
from spherical_flow.flow360 import bilinear_sample_erp
from spherical_flow.geometry import (geodesic_distance, healpix_unit_vectors, logmap,
                                     points_to_equirectangular_pixels, set_geodesic_mode,
                                     tangent_basis, endpoint_from_tangent_flow)
from spherical_flow.healpix_pyramid import build_healpix_pyramid, pool_features
from spherical_flow.oslo_raft import sequence_geodesic_loss
from spherical_flow.oslo_raft_retina import OSLORAFTRetina
from spherical_flow.shard_dataset import ShardFlowDataset, _import_sfprep, _to_chw_free_float


def parse_sources(text: str) -> List[Tuple[str, str]]:
    return [tuple(s.strip().split(":")) for s in text.split(",") if s.strip()]  # type: ignore


def raw_records(shards: Path, source: Tuple[str, str], count: int) -> List[dict]:
    iter_shard, list_shards = _import_sfprep()
    out: List[dict] = []
    for shard in list_shards(shards, *source):
        for record in iter_shard(shard):
            out.append(record)
            if len(out) >= count:
                return out
    return out


def photometric(img_hw3: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    h, w = img_hw3.shape[:2]
    u, v = points_to_equirectangular_pixels(dirs, h, w)
    return bilinear_sample_erp(img_hw3, u, v)


def nearest_nodes(q: torch.Tensor, pts: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    out = []
    for i in range(0, q.size(0), chunk):
        out.append((q[i:i + chunk] @ pts.T).argmax(-1))
    return torch.cat(out)


def flipped_endpoint(points: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
    """Endpoint of the sign-flipped tangent flow (the inverted-convention hypothesis)."""
    e, n = tangent_basis(points)
    f = logmap(points, endpoint[None], e, n)
    return endpoint_from_tangent_flow(points, -f, e, n)[0]


def warp_ratios(f1: torch.Tensor, f2_at, points: torch.Tensor, endpoint: torch.Tensor,
                mask: torch.Tensor) -> Tuple[float, float]:
    """(residual GT / zero, residual GT / flipped) on ``mask``; f2_at(dirs) samples frame 2."""
    r_gt = (f1 - f2_at(endpoint)).abs().mean(-1)[mask].mean()
    r_zero = (f1 - f2_at(points)).abs().mean(-1)[mask].mean()
    r_flip = (f1 - f2_at(flipped_endpoint(points, endpoint))).abs().mean(-1)[mask].mean()
    return (r_gt / r_zero).item(), (r_gt / r_flip).item()


# --------------------------------------------------------------------------- #
# data checks
# --------------------------------------------------------------------------- #
def check_warp_raw(shards: Path, sources: List[Tuple[str, str]], sup: int, pairs: int) -> None:
    from spherical_flow.shard_dataset import sample_pair_to_nodes

    pts = healpix_unit_vectors(sup).float()
    e, n = tangent_basis(pts)
    for src in sources:
        try:
            recs = raw_records(shards, src, pairs)
        except Exception as exc:  # source not materialized on this machine
            print(f"skip warp {src[0]}:{src[1]}: {exc}")
            continue
        if not recs:
            print(f"skip warp {src[0]}:{src[1]}: no shards here")
            continue
        g, f = [], []
        for rec in recs:
            f1e, f2e = _to_chw_free_float(rec["frame1"]), _to_chw_free_float(rec["frame2"])
            flow = torch.from_numpy(rec["flow"]).float()
            valid = torch.from_numpy(rec["valid"])
            t = sample_pair_to_nodes(f1e, f2e, flow, valid, pts, e, n)
            px = 2 * math.pi / f1e.shape[1]
            moving = t["valid"] & (t["flow"].norm(dim=-1) > 0.5 * px)
            if moving.sum() < 100:
                continue
            gz, gf = warp_ratios(t["frame1"], lambda d: photometric(f2e, d), pts,
                                 t["endpoint"], moving)
            g.append(gz)
            f.append(gf)
        if not g:
            report(f"warp {src[0]}:{src[1]}", True, "no pair with > 100 moving nodes")
            continue
        gz, gf = sum(g) / len(g), sum(f) / len(f)
        report(f"warp {src[0]}:{src[1]}", gz < 0.9 and gf < 0.9,
               f"{len(g)} pairs, moving nodes: residual GT/zero {gz:.2f}, GT/flipped {gf:.2f} "
               f"(both < 1 = GT warps frame 2 onto frame 1)")


def check_warp_loader(shards: Path, source: Tuple[str, str], pyr, pairs: int) -> None:
    ret, sup = pyr.retina_level, pyr.fine_level
    # supervision node i = parent of retina nodes 4^k i ... (nested): frame 1 at sup nodes
    k = pyr.retina_resolution - pyr.fine_resolution
    for so3 in (0.0, 1.0):
        ds = ShardFlowDataset(shards, ret.points.cpu(), source, shuffle_shards=False,
                              shuffle_buffer=0, seed=3, max_pairs=pairs, so3_prob=so3,
                              target_points=sup.points.cpu())
        gz_l, gf_l = [], []
        for s in ds:
            f1_sup = s["frame1"][None]
            for _ in range(k):
                f1_sup = f1_sup.reshape(1, -1, 4, 3).mean(2)
            f1_sup = f1_sup[0]
            f2 = s["frame2"]
            moving = s["valid"] & (s["flow"].norm(dim=-1) > math.radians(0.3))
            if moving.sum() < 100:
                continue
            gz, gf = warp_ratios(f1_sup, lambda d: f2[nearest_nodes(d, ret.points)],
                                 sup.points, s["endpoint"], moving)
            gz_l.append(gz)
            gf_l.append(gf)
        if not gz_l:
            report(f"warp_loader so3={so3:g}", True, "no pair with enough motion")
            continue
        gz, gf = sum(gz_l) / len(gz_l), sum(gf_l) / len(gf_l)
        report(f"warp_loader so3={so3:g}", gz < 0.9 and gf < 0.9,
               f"{len(gz_l)} {source[0]} samples, moving nodes: residual GT/zero {gz:.3f}, "
               f"GT/flipped {gf:.3f}")


def check_warp_planar(shards: Path, source: Tuple[str, str], pairs: int) -> None:
    """The planar RAFT loader (run_erp_raft_train.ERPPairStream): roll, flip, SO(3) and
    resize must keep frames and flow paired, as in the OSLO loader."""
    import random
    import types
    import importlib.util
    if importlib.util.find_spec("torchvision") is None:  # local venv; the jitter is unused
        sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    from run_erp_raft_train import ERPPairStream

    recs = raw_records(shards, source, pairs)
    h, w = recs[0]["valid"].shape
    for name, kw in (("plain", {}), ("roll+flip", {"roll": True, "hflip_prob": 1.0}),
                     ("so3", {"so3_prob": 1.0}), ("resize 320x640", {"resize": (320, 640)})):
        crop = kw.get("resize", (h, w))
        ds = ERPPairStream.__new__(ERPPairStream)
        ds.crop, ds.photo_prob = crop, 0.0
        ds.roll, ds.hflip_prob = kw.get("roll", False), kw.get("hflip_prob", 0.0)
        ds.so3_prob, ds.so3_max_angle_deg = kw.get("so3_prob", 0.0), 180.0
        ds.resize = kw.get("resize")
        rng, gen = random.Random(0), torch.Generator().manual_seed(0)
        ratios = []
        for rec in recs:
            b = ds._augment(rec, rng, gen, None)
            f1 = b["frame1"].permute(1, 2, 0).float() / 255
            f2 = b["frame2"].permute(1, 2, 0).float() / 255
            fl, va = b["flow"], b["valid"]
            hh, ww = va.shape
            v, u = torch.meshgrid(torch.arange(hh).float(), torch.arange(ww).float(), indexing="ij")
            m = va & (fl.norm(dim=0) > 1.0) & (v > 0.1 * hh) & (v < 0.9 * hh)
            if m.sum() < 100:
                continue

            def at(du, dv):
                return bilinear_sample_erp(f2, torch.remainder(u + du, ww).reshape(-1),
                                           (v + dv).clamp(0, hh - 1).reshape(-1)).reshape(hh, ww, 3)
            r_gt = (f1 - at(fl[0], fl[1])).abs().mean(-1)[m].mean()
            r_zero = (f1 - at(0, 0)).abs().mean(-1)[m].mean()
            ratios.append((r_gt / r_zero).item())
        if not ratios:
            report(f"warp_planar {name}", True, "no pair with enough motion")
            continue
        r = sum(ratios) / len(ratios)
        report(f"warp_planar {name}", r < 0.9,
               f"{len(ratios)} {source[0]} pairs, moving px: residual GT/zero {r:.3f}")


# --------------------------------------------------------------------------- #
# model checks
# --------------------------------------------------------------------------- #
def load_model(pyr, checkpoint: str) -> Tuple[OSLORAFTRetina, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    a = payload.get("args", {})
    model = OSLORAFTRetina(
        pyr, hidden_channels=a.get("hidden_channels", 96), context_dim=a.get("context_dim", 64),
        flow_scale=a.get("flow_scale", 0.5), lookup_rings=a.get("lookup_rings", 2),
        lookup_ring_points=a.get("lookup_ring_points", 8),
        interp_mode=a.get("lookup_interp") or "idw",
        flow_unit=(math.sqrt(4 * math.pi / pyr.num_estimation_nodes)
                   if a.get("flow_unit", "rad") == "node" else 1.0),
        corr_norm=a.get("corr_norm") or "cosine", update_block=a.get("update_block") or "small",
        encoder_blocks_per_level=a.get("encoder_blocks_per_level") or 1,
        encoder_est_blocks=a.get("encoder_est_blocks") or 0,
    )
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.eval()
    return model, a


def loader_samples(shards: Path, source, pyr, count: int) -> List[dict]:
    ds = ShardFlowDataset(shards, pyr.retina_level.points.cpu(), source, shuffle_shards=False,
                          shuffle_buffer=0, seed=3, max_pairs=count,
                          target_points=pyr.fine_level.points.cpu())
    return list(ds)


def stack(samples: List[dict], key: str) -> torch.Tensor:
    return torch.stack([s[key] for s in samples])


def geo_err(pred: torch.Tensor, samples: List[dict], sup) -> Tuple[torch.Tensor, torch.Tensor]:
    end = endpoint_from_tangent_flow(sup.points, pred, sup.basis_east, sup.basis_north)
    gt = stack(samples, "endpoint")
    valid = stack(samples, "valid")
    err = torch.rad2deg(geodesic_distance(end, gt))
    zero = torch.rad2deg(geodesic_distance(sup.points.expand_as(gt), gt))
    return err[valid], zero[valid]


def yaw90_permutation(points: torch.Tensor) -> torch.Tensor:
    """perm with points[perm[i]] = Rz(90) points[i] (exact on HEALPix)."""
    rz = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    perm = nearest_nodes(points @ rz.T, points)
    gap = torch.rad2deg((points[perm] - points @ rz.T).norm(dim=-1).max())  # chord: acos is coarse in fp32
    if gap > 1e-3 or perm.unique().numel() != perm.numel():
        raise RuntimeError(f"yaw 90 is not a node permutation here (gap {gap:.2e} deg)")
    return perm


def check_model(model: OSLORAFTRetina, trained: dict, pyr, samples: List[dict], iters: int) -> None:
    sup = pyr.fine_level
    f1, f2 = stack(samples, "frame1"), stack(samples, "frame2")

    # batch independence
    with torch.no_grad():
        both = model(f1[:2], f2[:2], pyr, iters=iters)[-1]
        one = model(f1[1:2], f2[1:2], pyr, iters=iters)[-1]
    rel = ((both[1] - one[0]).norm(dim=-1).max() / one[0].norm(dim=-1).mean()).item()
    report("batch_indep", rel < 1e-3, f"pair 2 alone vs in a batch of 2: max diff / mean |flow| {rel:.1e}")

    # checkpointing on/off, and gradient reach
    def loss_and_grads(ckpt: bool):
        model.zero_grad(set_to_none=True)
        model.use_checkpoint = ckpt
        model.fnet.use_checkpoint = model.cnet.use_checkpoint = ckpt
        model.train()
        preds = model(f1[:1], f2[:1], pyr, iters=3)
        loss = sequence_geodesic_loss(preds, stack(samples[:1], "endpoint"), sup,
                                      stack(samples[:1], "valid"))
        loss.backward()
        model.eval()
        return preds[-1].detach(), {k: (None if p.grad is None else p.grad.clone())
                                    for k, p in model.named_parameters()}
    torch.manual_seed(0)
    out_a, g_a = loss_and_grads(True)
    torch.manual_seed(0)
    out_b, g_b = loss_and_grads(False)
    d_out = (out_a - out_b).abs().max().item()
    d_grad = max(((g_a[k] - g_b[k]).abs().max() / g_b[k].abs().max().clamp_min(1e-12)).item()
                 for k in g_b if g_b[k] is not None)
    report("ckpt_equiv", d_out < 1e-5 and d_grad < 1e-3,
           f"checkpointing on vs off: max output diff {d_out:.1e}, max relative grad diff {d_grad:.1e}")
    dead = [k for k, g in g_b.items() if g is None or g.abs().max() == 0]
    report("grad_reach", not dead,
           f"{len(g_b) - len(dead)}/{len(g_b)} parameter tensors get a gradient"
           + (f"; none: {', '.join(dead[:8])}" if dead else ""))

    # yaw 90 equivariance (east/north bases are invariant under a z rotation)
    p_ret = yaw90_permutation(pyr.retina_level.points)
    p_sup = yaw90_permutation(sup.points)
    rf1 = torch.empty_like(f1[:1]); rf1[:, p_ret] = f1[:1]
    rf2 = torch.empty_like(f2[:1]); rf2[:, p_ret] = f2[:1]
    with torch.no_grad():
        out = model(f1[:1], f2[:1], pyr, iters=iters)[-1]
        rout = model(rf1, rf2, pyr, iters=iters)[-1]
    diff = (rout[0, p_sup] - out[0]).norm(dim=-1)
    rel = (diff.mean() / out[0].norm(dim=-1).mean()).item()
    frac = (diff > 0.1 * out[0].norm(dim=-1).mean()).float().mean().item()
    report("yaw_equiv", rel < 0.05,
           f"input rotated 90 deg about z: mean |pred diff| / mean |flow| {rel:.3f}, "
           f"nodes off by > 10%: {frac:.1%}")

    # static input
    with torch.no_grad():
        stat = model(f1, f1, pyr, iters=iters)[-1]
    gt_mag = torch.rad2deg(stack(samples, "flow").norm(dim=-1))[stack(samples, "valid")]
    st = torch.rad2deg(stat.norm(dim=-1))
    report("static", st.mean() < 0.25 * gt_mag.mean(),
           f"frame 2 = frame 1: mean |pred| {st.mean():.3f} deg (p99 {st.quantile(0.99):.3f}) "
           f"vs mean GT motion {gt_mag.mean():.3f} deg")

    # vs zero, and per iteration
    errs = {}
    with torch.no_grad():
        preds = model(f1, f2, pyr, iters=max(24, iters))
    for t, pred in enumerate(preds, 1):
        err, zero = geo_err(pred, samples, sup)
        errs[t] = (err.mean().item(), zero.mean().item())
    err, zero = errs[iters]
    imp = 100 * (1 - err / zero)
    report("vs_zero", imp > 0,
           f"{len(samples)} pairs at {iters} iters (trained with {trained.get('iters', '?')}): "
           f"geo error {err:.3f} deg vs zero {zero:.3f} deg = {imp:+.1f}%")
    curve = " ".join(f"{t}:{errs[t][0]:.3f}" for t in (1, 2, 4, 8, 12, 16, 24) if t in errs)
    best = min(errs, key=lambda t: errs[t][0])
    report("iters", errs[24][0] <= 1.1 * errs[best][0],
           f"geo error (deg) by iteration {curve}; best at {best}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards", default="/data/shards")
    p.add_argument("--sources", default="replica360:val,mpf:val,flow360:val",
                   help="Sources for the raw warp check; the first one feeds the loader "
                        "and model checks.")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--retina", type=int, default=7)
    p.add_argument("--sup", type=int, default=6)
    p.add_argument("--est", type=int, default=4)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--sort-cut-deg", type=float, default=None)
    p.add_argument("--fill-missing-neighbors", action="store_true")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    set_geodesic_mode("haversine")
    shards, sources = Path(args.shards), parse_sources(args.sources)

    t0 = time.time()
    check_warp_raw(shards, sources, args.sup, args.pairs)
    pyr = build_healpix_pyramid(fine_resolution=args.sup, estimation_resolution=args.est,
                                corr_pool_levels=3, retina_resolution=args.retina,
                                sort_cut_deg=args.sort_cut_deg,
                                fill_missing_neighbors=args.fill_missing_neighbors)
    check_warp_loader(shards, sources[0], pyr, args.pairs)
    check_warp_planar(shards, sources[0], args.pairs)
    if args.checkpoint:
        model, trained = load_model(pyr, args.checkpoint)
        samples = loader_samples(shards, sources[0], pyr, args.pairs)
        check_model(model, trained, pyr, samples, args.iters)
    print(f"({time.time() - t0:.0f}s)")
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\npipeline: {len(RESULTS) - len(failed)}/{len(RESULTS)} passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
