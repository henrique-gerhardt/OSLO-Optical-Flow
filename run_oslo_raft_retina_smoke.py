"""CPU smoke + unit suite for OSLO-RAFT-R (the retina model) — plan §9.1/§9.2.

Everything here runs healpy-free on a synthetic nested pyramid (the
`run_oslo_raft_pyramid_smoke.py` builder extended with a retina range), so the full
suite gates GPU work from a laptop. Sections:

  1. pyramid structure    — trivial lookups above the supervision grid, pool_index
                            covers both the encoder and correlation ranges, retina_level.
  2. data seam            — target-grid split is backward compatible bit-for-bit;
                            SO(3) passthrough matches the direct single-grid result.
  3. synth rotations      — conventions match `SyntheticRotationFlowDataset`
                            (endpoint = R p, frame2 = texture(R^-1 p)); yaw sanity.
  4. lookup properties    — §5.4: continuous in sub-node flow where the old snap
                            lookup is bit-identical (the bug being fixed); K=1/center
                            stencil == the validated snap lookup; cold-start center
                            column == the all-pairs diagonal; finite at the poles.
  5. model                — cold-start zero at r_sup, finite grads, checkpointed ==
                            non-checkpointed grads, live heads, param budget.
  6. sub-node recovery    — THE decisive test (§9.2d): train ~300 steps on analytic
                            rotations of 0.3 x est-node-spacing (sub-node at the
                            estimation grid, multi-node at the retina). The full model
                            must recover the flow direction (cos-sim > 0.9 on held-out
                            rotations); the corr-ablated control must fail (< 0.3).
                            If this fails, the retina thesis is wrong — stop and debug
                            before spending any GPU time.

    python run_oslo_raft_retina_smoke.py                 # full suite (~minutes on CPU)
    python run_oslo_raft_retina_smoke.py --skip-recovery # wiring-only (~seconds)
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from spherical_flow.geometry import (
    _normalize,
    endpoint_from_tangent_flow,
    fibonacci_unit_vectors,
    logmap,
    rotate_points,
    tangent_basis,
)
from spherical_flow.healpix_pyramid import (
    SpherePyramid,
    _build_level,
    nested_children_index,
    nested_descendant_index,
)
from spherical_flow.oslo_raft import sequence_geodesic_loss
from spherical_flow.oslo_raft_pyramid import (
    AllPairsCorrelation,
    build_correlation_pyramid,
    pyramid_lookup,
)
from spherical_flow.oslo_raft_retina import (
    OSLORAFTRetina,
    build_feature_pyramid,
    build_lookup_offsets,
    interp_pyramid_lookup,
    lookup_stencil,
    stencil_match_loss,
)
from spherical_flow.shard_dataset import sample_pair_to_nodes, synth_rotation_record
from spherical_flow.so3_augment import rotation_matrix, sample_rotation, so3_augment_pair
from spherical_flow.synthetic import analytic_sphere_texture


def synthetic_retina_pyramid(
    estimation_resolution: int = 4,
    fine_resolution: int = 5,
    retina_resolution: int = 7,
    corr_pool_levels: int = 2,
    coarsest_nodes: int = 12,
    conv_neighbors: int = 8,
    lookup_neighbors: int = 24,
) -> SpherePyramid:
    """Quasi-uniform nested pyramid with a retina range (no healpy).

    Each finer level subdivides its parent 2x2 in the parent's tangent plane (children
    at the four sub-cell centers, +-h/4 along east/north for a cell of width ``h``), so
    node spacing HALVES per level like real HEALPix and the nominal spacing
    ``sqrt(4 pi / N)`` matches the true local spacing. This matters: the earlier
    random-perturbation construction clustered points ~3x tighter than nominal, which
    silently degraded every spacing-scaled quantity (lookup stencils, aux-loss soft
    targets, "sub-node" angles) and made the learning test unwinnable by construction.
    Nested contiguity (children of ``i`` are ``4i..4i+3``) holds by construction; the
    retina levels above the supervision grid carry a trivial lookup (mirroring the
    production builder).
    """
    corr_res = [
        estimation_resolution - k
        for k in range(corr_pool_levels + 1)
        if estimation_resolution - k >= 0
    ]
    needed = sorted(set(range(estimation_resolution, retina_resolution + 1)) | set(corr_res))
    base = fibonacci_unit_vectors(coarsest_nodes)
    points = {needed[0]: base}
    cell = math.sqrt(4.0 * math.pi / coarsest_nodes)  # coarsest cell width (radians)
    for r in range(needed[0] + 1, needed[-1] + 1):
        parent = points[r - 1]
        east, north = tangent_basis(parent)
        h = cell / (2 ** (r - needed[0] - 1))          # parent-level cell width
        offs = torch.tensor(
            [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]]
        ) * h
        children = (
            parent.unsqueeze(1)
            + offs[:, 0].view(1, 4, 1) * east.unsqueeze(1)
            + offs[:, 1].view(1, 4, 1) * north.unsqueeze(1)
        ).reshape(-1, 3)
        points[r] = _normalize(children)
    levels = {
        r: _build_level(
            points[r], conv_neighbors,
            lookup_neighbors if r <= fine_resolution else 0, knn_chunk=1024,
        )
        for r in needed
    }
    pool_index = {
        r: nested_children_index(levels[r].num_nodes) for r in needed if (r + 1) in levels
    }
    descendant_index = nested_descendant_index(
        levels[estimation_resolution].num_nodes, fine_resolution - estimation_resolution
    )
    est = levels[estimation_resolution]
    upsample_neighbors = torch.cat(
        [torch.arange(est.num_nodes, dtype=torch.long).unsqueeze(1), est.conv_index], dim=1
    )
    return SpherePyramid(
        levels=levels,
        estimation_resolution=estimation_resolution,
        fine_resolution=fine_resolution,
        corr_resolutions=corr_res,
        pool_index=pool_index,
        descendant_index=descendant_index,
        upsample_neighbors=upsample_neighbors,
        retina_resolution=retina_resolution,
    )


# --------------------------------------------------------------------------- #
# 1. pyramid structure
# --------------------------------------------------------------------------- #
def test_pyramid_structure(pyr: SpherePyramid) -> None:
    est, fine, ret = pyr.estimation_resolution, pyr.fine_resolution, pyr.retina_resolution
    assert ret is not None and est < fine <= ret
    for r, lvl in pyr.levels.items():
        if r > fine:
            n = lvl.num_nodes
            assert lvl.lookup_index.shape == (n, 1), f"retina level {r} lookup not trivial"
            assert (lvl.lookup_index.squeeze(1) == torch.arange(n)).all()
        else:
            assert lvl.lookup_index.size(1) > 1, f"level {r} lookup unexpectedly trivial"
    # pool_index must cover the encoder chain (ret..est) AND the corr chain (est..est-k)
    for r in list(range(est, ret)) + [c for c in pyr.corr_resolutions if c < est]:
        assert r in pyr.pool_index, f"pool_index missing level {r}"
        assert pyr.pool_index[r].shape == (pyr.levels[r].num_nodes, 4)
    assert pyr.retina_level.num_nodes == pyr.levels[ret].num_nodes
    print(f"PASS pyramid structure: est={est} fine={fine} retina={ret}, trivial lookups "
          f"above fine, pool_index covers encoder+corr ranges")


# --------------------------------------------------------------------------- #
# 2. data seam
# --------------------------------------------------------------------------- #
def test_data_seam() -> None:
    torch.manual_seed(0)
    h, w = 64, 128
    f1, f2 = torch.rand(h, w, 3), torch.rand(h, w, 3)
    flow, valid = torch.randn(h, w, 2) * 2.0, torch.ones(h, w)
    sup = fibonacci_unit_vectors(500)
    ret = fibonacci_unit_vectors(2000)
    e_s, n_s = tangent_basis(sup)

    old = sample_pair_to_nodes(f1, f2, flow, valid, sup, e_s, n_s)
    new = sample_pair_to_nodes(f1, f2, flow, valid, sup, e_s, n_s, target_points=None)
    for k in old:
        assert torch.equal(old[k], new[k]), f"backward-compat break in {k}"

    split = sample_pair_to_nodes(
        f1, f2, flow, valid, ret, *tangent_basis(ret), target_points=sup,
        target_basis_east=e_s, target_basis_north=n_s,
    )
    assert split["frame1"].shape == (2000, 3) and split["flow"].shape == (500, 2)
    for k in ("flow", "endpoint", "valid"):
        assert torch.allclose(split[k].float(), old[k].float(), atol=1e-6), k

    rot = sample_rotation(torch.Generator().manual_seed(3), max_angle_deg=30.0)
    direct = so3_augment_pair(f1, f2, flow, valid, sup, rot, e_s, n_s)
    passthrough = so3_augment_pair(
        f1, f2, flow, valid, ret, rot, *tangent_basis(ret),
        target_points=sup, target_basis_east=e_s, target_basis_north=n_s,
    )
    assert torch.allclose(direct["flow"], passthrough["flow"], atol=1e-6)
    assert torch.equal(direct["valid"], passthrough["valid"])

    try:
        sample_pair_to_nodes(f1, f2, flow, valid, ret, *tangent_basis(ret),
                             endpoint_rotation=rot, target_points=sup)
        raise AssertionError("missing target_query_points guard did not fire")
    except ValueError:
        pass
    print("PASS data seam: single-grid bit-identical; split-grid targets == single-grid; "
          "SO(3) passthrough consistent; rotation guard fires")


# --------------------------------------------------------------------------- #
# 3. synth-rotation conventions
# --------------------------------------------------------------------------- #
def _texture_erp(h: int, w: int) -> torch.Tensor:
    from spherical_flow.geometry import equirectangular_pixels_to_unit_vectors

    v, u = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    dirs = equirectangular_pixels_to_unit_vectors(u.reshape(-1), v.reshape(-1), h, w)
    return analytic_sphere_texture(dirs).reshape(h, w, 3)


def test_synth_rotation() -> None:
    sup = fibonacci_unit_vectors(500)
    e_s, n_s = tangent_basis(sup)
    erp = _texture_erp(256, 512)

    axis = torch.tensor([0.2, 0.4, 1.0]); axis = axis / axis.norm()
    ang = torch.tensor(math.radians(8.0))
    rec = synth_rotation_record(erp, sup, sup, e_s, n_s, rotation_matrix(axis, ang))
    # endpoint convention: e = R p (matches SyntheticRotationFlowDataset's rotate +angle)
    assert torch.allclose(rec["endpoint"], rotate_points(sup, axis, ang), atol=1e-6)
    # frame2 convention: frame2(p) = texture(R^-1 p), up to ERP bilinear error
    ref = analytic_sphere_texture(rotate_points(sup, axis, -ang))
    err = (rec["frame2"] - ref).abs().max().item()
    assert err < 0.05, err
    assert bool(rec["valid"].all())
    # |flow| equals the angular displacement
    gt = logmap(sup, rec["endpoint"], e_s, n_s).squeeze(0)
    assert torch.allclose(rec["flow"], gt, atol=1e-6)

    # pure yaw at the equator: east flow ~= +angle, north ~= 0
    yaw = rotation_matrix(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.radians(5.0)))
    eq = sup[sup[:, 2].abs() < 0.1]
    rec2 = synth_rotation_record(erp, eq, eq, *tangent_basis(eq), yaw)
    east = math.degrees(rec2["flow"][:, 0].mean())
    north = math.degrees(rec2["flow"][:, 1].abs().mean())
    assert abs(east - 5.0) < 0.15 and north < 0.15, (east, north)
    print(f"PASS synth rotations: endpoint=R p, frame2 texture err {err:.4f}, "
          f"equator yaw east flow {east:.3f} deg (target 5)")


# --------------------------------------------------------------------------- #
# 4. lookup properties (§5.4)
# --------------------------------------------------------------------------- #
def test_lookup_properties(pyr: SpherePyramid) -> None:
    est = pyr.estimation_level
    n = est.num_nodes
    b, c = 1, 32
    torch.manual_seed(2)
    f1 = F.normalize(torch.randn(b, n, c), dim=-1)
    f2 = F.normalize(torch.randn(b, n, c), dim=-1)
    f2_levels = build_feature_pyramid(f2, pyr)
    offsets = build_lookup_offsets(pyr, rings=2, ring_points=8)
    s_est = math.sqrt(4 * math.pi / n)

    corr_pyr = build_correlation_pyramid(AllPairsCorrelation()(f1, f2), pyr)
    def new_lk(fl): return interp_pyramid_lookup(f1, f2_levels, fl, pyr, offsets)
    def old_lk(fl): return pyramid_lookup(corr_pyr, fl, pyr)

    # Continuity, anchored at flow=0 (endpoints start exactly at the nodes). "Sub-node"
    # must be measured against each node's OWN cell on this irregular synthetic grid
    # (its clustered construction makes true nearest-neighbor distances ~3x smaller than
    # sqrt(4pi/N)), so the perturbation is scaled per node to 0.2x / 0.1x its nearest-
    # neighbor distance — strictly inside every est-level Voronoi cell. Then the old
    # snap lookup's est-level slice must be BIT-IDENTICAL (that is the §1 bug: exactly
    # zero response to sub-cell flow), while the interpolated lookup responds broadly,
    # with finite differences shrinking with the step (piecewise-smooth; candidate-set
    # switches keep the ratio below the smooth 2.0).
    zero = torch.zeros(b, n, 2)
    torch.manual_seed(3)
    dirn = torch.randn(b, n, 2); dirn = dirn / dirn.norm(dim=-1, keepdim=True)
    d_nn = (est.points - est.points[est.lookup_index[:, 1]]).norm(dim=-1)  # [N] chord ~ geodesic
    step = d_nn.view(1, n, 1) * dirn
    d1 = (new_lk(0.2 * step) - new_lk(zero)).norm().item()
    d2 = (new_lk(0.1 * step) - new_lk(zero)).norm().item()
    assert d1 > 1e-4 and d2 > 1e-5 and d1 > d2, (d1, d2)
    ratio = d1 / d2
    assert 1.2 < ratio < 3.5, ratio
    n_est_cols = pyr.levels[pyr.corr_resolutions[0]].lookup_index.size(1)
    old_est_diff = (
        (old_lk(0.1 * step) - old_lk(zero))[..., :n_est_cols].abs().max().item()
    )
    assert old_est_diff == 0.0, (
        f"old est-level snap gather responded to sub-cell flow ({old_est_diff})"
    )
    new_frac = ((new_lk(0.1 * step) - new_lk(zero)).abs() > 1e-7).float().mean().item()
    assert new_frac > 0.5, f"interp lookup changed only {new_frac:.1%} of entries"
    print(f"PASS lookup continuity: interp |df| {d1:.4f}->{d2:.4f} (ratio {ratio:.2f}), "
          f"{new_frac:.1%} of entries responding; old est-level snap gather bit-identical "
          f"under the same sub-cell flow (the §1 piecewise-constant bug)")

    # Snap-parity anchor: K_int=1 + center-only stencil == old lookup's center column
    # (up to the deliberate scale change: the interp lookup is in cosine units, the old
    # AllPairsCorrelation path divides the normalized dot by sqrt(C) — see the lookup).
    torch.manual_seed(4)
    flow0 = 0.3 * s_est * torch.randn(b, n, 2)
    off0 = torch.zeros(len(pyr.corr_resolutions), 1, 2)
    new_c = interp_pyramid_lookup(f1, f2_levels, flow0, pyr, off0, k_interp=1)
    old_full = old_lk(flow0)
    sizes = [pyr.levels[r].lookup_index.size(1) for r in pyr.corr_resolutions]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    old_c = torch.stack([old_full[..., s] for s in starts], dim=-1) * math.sqrt(c)
    assert torch.allclose(new_c, old_c, atol=1e-5), (new_c - old_c).abs().max()
    print("PASS snap-parity anchor: K_int=1 + center stencil == validated snap lookup "
          "(cosine units)")

    # Cold start: at flow=0 the center column equals the all-pairs diagonal (cosine).
    cold = interp_pyramid_lookup(f1, f2_levels, zero, pyr, off0, k_interp=3)[..., 0]
    diag = torch.diagonal(AllPairsCorrelation()(f1, f2), dim1=1, dim2=2) * math.sqrt(c)
    err = (cold - diag).abs().max().item()
    assert err < 1e-5, err
    print(f"PASS cold-start: interp center == all-pairs diagonal (max err {err:.1e})")

    # Pole safety: large random flows (endpoints crossing +-z) stay finite.
    out = new_lk(0.5 * s_est * torch.randn(b, n, 2))
    assert torch.isfinite(out).all()
    st = lookup_stencil(2, 8)
    assert st.shape == (17, 2) and (st[0] == 0).all()
    assert offsets.shape == (len(pyr.corr_resolutions), 17, 2)
    print("PASS lookup finite at poles; stencil M=17/level, level-scaled")


# --------------------------------------------------------------------------- #
# 5. model wiring
# --------------------------------------------------------------------------- #
def test_model(pyr: SpherePyramid) -> None:
    torch.manual_seed(0)
    model = OSLORAFTRetina(pyr)
    n_ret, n_fine = pyr.retina_level.num_nodes, pyr.num_fine_nodes
    b, iters = 2, 4
    f1, f2 = torch.rand(b, n_ret, 3), torch.rand(b, n_ret, 3)

    preds = model(f1, f2, pyr, iters=iters)
    assert len(preds) == iters and preds[-1].shape == (b, n_fine, 2)
    assert torch.count_nonzero(preds[0]) == 0, "cold-start prediction must be exactly zero"

    gt = endpoint_from_tangent_flow(
        pyr.fine_level.points, torch.zeros(b, n_fine, 2),
        pyr.fine_level.basis_east, pyr.fine_level.basis_north,
    )
    valid = torch.ones(b, n_fine, dtype=torch.bool)
    loss = sequence_geodesic_loss(preds, gt, pyr.fine_level, valid)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.flow_conv2.weight.grad is not None and model.flow_conv2.weight.grad.abs().sum() > 0
    assert model.upsample_head.conv2.weight.grad is not None

    # checkpointed (default) == non-checkpointed gradients
    g_ck = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    model.use_checkpoint = False
    model.fnet.use_checkpoint = model.cnet.use_checkpoint = False
    loss2 = sequence_geodesic_loss(model(f1, f2, pyr, iters=iters), gt, pyr.fine_level, valid)
    loss2.backward()
    for k, p in model.named_parameters():
        if p.grad is not None and k in g_ck:
            assert torch.allclose(p.grad, g_ck[k], atol=1e-6), k
    model.use_checkpoint = True
    model.fnet.use_checkpoint = model.cnet.use_checkpoint = True

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_000_000 <= n_params <= 3_000_000, f"param count {n_params:,} outside 1-3M budget"
    print(f"PASS model: cold-start zero at r_sup, finite grads, checkpoint parity, "
          f"params={n_params:,} (1-3M), loss={loss.item():.4f}")


# --------------------------------------------------------------------------- #
# 6. THE decisive test: sub-est-node motion recovery (§9.2d)
# --------------------------------------------------------------------------- #
def _recovery_texture(points: torch.Tensor) -> torch.Tensor:
    """Multi-band texture: the smooth analytic base PLUS ~9 deg-wavelength content.

    `analytic_sphere_texture` alone is too smooth for this test (45-70 deg wavelengths
    — the estimation grid already Nyquist-samples it, so a retina adds nothing and the
    correlation landscape is nearly flat). The added bands (|k| ~ 38-45 rad^-1, ~8-10
    deg wavelength) are resolvable at retina sampling but sub-Nyquist at the est grid —
    exactly the content regime (FLOW360-like fine texture) the retina thesis is about.
    """
    x, y, z = points.unbind(dim=-1)
    base = analytic_sphere_texture(points)
    mid = torch.stack(
        [
            torch.sin(29.0 * x - 17.0 * y + 23.0 * z),
            torch.sin(19.0 * x + 31.0 * y - 11.0 * z),
            torch.cos(-13.0 * x + 27.0 * y + 33.0 * z),
        ],
        dim=-1,
    )
    return 0.5 * base + 0.25 * (mid + 1.0)


def _rotation_batch(
    pyr: SpherePyramid, angle_lo: float, angle_hi: float, gen: torch.Generator, batch: int
):
    """Random-axis rotations of the recovery texture: frames at the retina, GT at r_sup.

    The angle is drawn uniformly in [angle_lo, angle_hi] per sample (a fixed angle when
    lo == hi).
    """
    ret, sup, est = pyr.retina_level, pyr.fine_level, pyr.estimation_level
    f1s, f2s, ends, ends_est = [], [], [], []
    for _ in range(batch):
        axis = torch.randn(3, generator=gen)
        axis = axis / axis.norm().clamp_min(1e-8)
        u = float(torch.rand((), generator=gen))
        ang = torch.tensor(angle_lo + u * (angle_hi - angle_lo))
        f1s.append(_recovery_texture(ret.points))
        f2s.append(_recovery_texture(rotate_points(ret.points, axis, -ang)))
        ends.append(rotate_points(sup.points, axis, ang))
        ends_est.append(rotate_points(est.points, axis, ang))
    return torch.stack(f1s), torch.stack(f2s), torch.stack(ends), torch.stack(ends_est)


def _direction_cos_sim(pred: torch.Tensor, endpoints: torch.Tensor, pyr: SpherePyramid):
    sup = pyr.fine_level
    gt = logmap(sup.points, endpoints, sup.basis_east, sup.basis_north)
    mask = gt.norm(dim=-1) > 0.5 * gt.norm(dim=-1).amax(dim=1, keepdim=True)  # skip near-axis
    cos = F.cosine_similarity(pred, gt, dim=-1)
    return (cos * mask).sum() / mask.sum()


def test_subnode_recovery(pyr: SpherePyramid, steps: int, ablate: bool) -> float:
    """Train on random rotations, eval held-out at 0.3 x est spacing (sub-node).

    Every sample is a fresh random-axis rotation, so the context path cannot memorize a
    field — only correlation can solve this. The protocol encodes what the failure
    ladder measured on the way here (each item was individually necessary):

      - **Angle diversity** (uniform 0.2-1.2 x est spacing per sample): the fixed
        sub-node angle alone never escaped the context-only optimum. Eval stays
        strictly sub-node (held-out axes, 0.3 x spacing).
      - **Aux matching warmup** (`stencil_match_loss`, first quarter of the budget
        aux-only): direct matching supervision reaches its soft-target entropy floor in
        ~300 steps (match accuracy 0.90-0.99 across angles) where end-to-end training
        never develops matchable features at all — the §8 Stage-A bootstrap, made
        explicit. Joint phase keeps 0.5x aux.
      - **The corr skip in the model**: with matching solved, the corr stencil is
        LINEARLY decodable to flow (probe cos-sim 0.99) yet the motion-encoder->GRU
        path alone never aligned to it on this budget; with the skip the flow loss
        drops to ~0.1x the zero baseline within ~450 joint steps.
      - **OneCycle to 1e-3** (flat 1e-3 plateaued; 2.5e-3 diverged).
    """
    s_est = math.sqrt(4 * math.pi / pyr.num_estimation_nodes)
    eval_angle = 0.3 * s_est
    warmup = max(1, steps // 4)
    torch.manual_seed(11)
    model = OSLORAFTRetina(pyr)
    model.ablate_corr = ablate
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    gen = torch.Generator().manual_seed(21)
    valid = torch.ones(2, pyr.num_fine_nodes, dtype=torch.bool)

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        f1, f2, ends, ends_est = _rotation_batch(pyr, 0.2 * s_est, 1.2 * s_est, gen, batch=2)
        if step <= warmup:
            _, (f1e, f2e) = model(f1, f2, pyr, iters=1, return_features=True)
            total = stencil_match_loss(f1e, f2e, ends_est, pyr.estimation_level)
            desc = f"aux={total.item():.5f}"
        else:
            preds, (f1e, f2e) = model(f1, f2, pyr, iters=4, return_features=True)
            loss = sequence_geodesic_loss(preds, ends, pyr.fine_level, valid)
            aux = stencil_match_loss(f1e, f2e, ends_est, pyr.estimation_level)
            total = loss + 0.5 * aux
            desc = f"flow={loss.item():.5f} aux={aux.item():.5f}"
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps:
            print(f"    [{'ablated' if ablate else 'full   '}] step {step:3d} {desc} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    gen_eval = torch.Generator().manual_seed(1234)  # held-out rotations, sub-node angle
    sims = []
    with torch.no_grad():
        for _ in range(4):
            f1, f2, ends, _ = _rotation_batch(pyr, eval_angle, eval_angle, gen_eval, batch=2)
            pred = model(f1, f2, pyr, iters=8)[-1]
            sims.append(_direction_cos_sim(pred, ends, pyr).item())
    return sum(sims) / len(sims)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-recovery", action="store_true",
                    help="Skip the ~minutes sub-node recovery training (wiring-only run).")
    ap.add_argument("--recovery-steps", type=int, default=600)
    args = ap.parse_args()

    print("[oslo-raft retina smoke]")
    pyr = synthetic_retina_pyramid()
    test_pyramid_structure(pyr)
    test_data_seam()
    test_synth_rotation()
    test_lookup_properties(pyr)
    test_model(pyr)

    if args.skip_recovery:
        print("SKIP sub-node recovery (--skip-recovery)")
        print("ALL OK (wiring)")
        return

    s_est_deg = math.degrees(math.sqrt(4 * math.pi / pyr.num_estimation_nodes))
    print(f"sub-node recovery: train angles 0.2-1.2 x est spacing, held-out eval at "
          f"0.3 x ({0.3 * s_est_deg:.2f} deg), {args.recovery_steps} steps each ...")
    sim_full = test_subnode_recovery(pyr, args.recovery_steps, ablate=False)
    sim_ablated = test_subnode_recovery(pyr, args.recovery_steps, ablate=True)
    print(f"held-out direction cos-sim: full={sim_full:.3f}  corr-ablated={sim_ablated:.3f}")
    assert sim_full > 0.9, (
        f"FULL MODEL FAILED sub-node recovery (cos-sim {sim_full:.3f} <= 0.9) — the retina "
        "thesis does not hold on this wiring; debug §5/§6 before any GPU run"
    )
    assert sim_ablated < 0.3, (
        f"ABLATED CONTROL DID NOT FAIL (cos-sim {sim_ablated:.3f} >= 0.3) — the task is "
        "solvable without correlation; the test is not measuring matching"
    )
    print("PASS sub-node recovery: correlation is load-bearing for sub-est-node motion")
    print("ALL OK")


if __name__ == "__main__":
    main()
