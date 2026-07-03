#!/usr/bin/env bash
# OSLO-RAFT GPU container smoke test.
#
# Two tiers:
#   1. Data-free: build a real HEALPix level (proves astropy-healpix), then one
#      forward/backward of OSLO-RAFT on random frames (proves the model + CUDA +
#      SDPAConv path). Always runs; needs no mounted data.
#   2. Integration: if shards are mounted at $OSLO_SHARDS, run the full
#      run_oslo_raft.py --grid healpix --smoke-test (one train step + one eval),
#      which additionally exercises the sfprep data pipeline and metrics.
#
# Exit non-zero on the first failure so `docker compose run` surfaces it.
set -euo pipefail

OSLO_SHARDS="${OSLO_SHARDS:-/data/shards}"
RESOLUTION="${OSLO_SMOKE_RESOLUTION:-4}"
# Retina smoke geometry (OSLO-RAFT-R). Defaults exercise the production r7/r6/r4
# stack; export OSLO_SMOKE_RETINA=6 OSLO_SMOKE_RETINA_SUP=5 for a faster pass on
# weak/emulated hardware.
RETINA_RES="${OSLO_SMOKE_RETINA:-7}"
RETINA_SUP="${OSLO_SMOKE_RETINA_SUP:-6}"
RETINA_EST="${OSLO_SMOKE_RETINA_EST:-4}"
# DataLoader workers / GRU iterations for the tier-2 smokes (0 workers + 4 iters for
# memory-tight/emulated environments).
SMOKE_WORKERS="${OSLO_SMOKE_WORKERS:-2}"
SMOKE_ITERS="${OSLO_SMOKE_ITERS:-8}"

echo "=================================================================="
echo " OSLO-RAFT container smoke"
echo "=================================================================="
python - <<'PY'
import torch
print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device      {torch.cuda.get_device_name(0)}")
try:
    import astropy_healpix
    print(f"astropy-healpix  {astropy_healpix.__version__}")
except Exception as exc:  # pragma: no cover - smoke
    raise SystemExit(f"astropy-healpix import failed: {exc}")
PY

echo
echo "[tier 1] data-free HEALPix forward/backward ..."
python - <<'PY'
import torch
from spherical_flow.geometry import healpix_unit_vectors
from spherical_flow.oslo_raft import OSLORAFT, build_knn_level, sequence_geodesic_loss
from spherical_flow.geometry import endpoint_from_tangent_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = 2  # nside=4 -> 192 nodes; tiny but real nested HEALPix geometry
conv_neighbors, lookup_neighbors = 8, 24

points = healpix_unit_vectors(res)            # proves astropy-healpix nested path
level = build_knn_level(points, conv_neighbors, lookup_neighbors).to(device)
model = OSLORAFT(
    kernel_size=conv_neighbors + 1,
    lookup_neighbors=lookup_neighbors + 1,     # same contract as run_oslo_raft.py
).to(device)

b, n = 2, level.num_nodes
f1 = torch.rand(b, n, 3, device=device)
f2 = torch.rand(b, n, 3, device=device)
preds = model(f1, f2, level, iters=4)
assert len(preds) == 4 and preds[-1].shape == (b, n, 2), "bad prediction shape"

# zero-init head => cold-start flow is exactly zero
assert torch.count_nonzero(preds[0]) == 0, "cold-start flow should be zero"

gt = endpoint_from_tangent_flow(level.points, torch.zeros(b, n, 2, device=device),
                                level.basis_east, level.basis_north)
valid = torch.ones(b, n, dtype=torch.bool, device=device)
loss = sequence_geodesic_loss(preds, gt, level, valid)
loss.backward()
finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
assert finite, "non-finite gradient"
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  nodes={n} params={params:,} loss={loss.item():.4f} device={device.type}  OK")
PY

echo
echo "[tier 1.5] nested-HEALPix pyramid foundation (index + transport + real geometry) ..."
python run_healpix_pyramid_smoke.py

echo
echo "[tier 1.6] multi-res OSLORAFTPyramid forward/backward (real r5/r4 geometry) ..."
python - <<'PY'
import torch
from spherical_flow.healpix_pyramid import build_healpix_pyramid
from spherical_flow.oslo_raft_pyramid import OSLORAFTPyramid
from spherical_flow.oslo_raft import sequence_geodesic_loss
from spherical_flow.geometry import endpoint_from_tangent_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fine=5/est=4 keeps it quick while exercising the r5 chunked build + the full multi-res
# forward/backward on real nested geometry (the 3-level r6 shape is covered by the CPU smoke).
pyr = build_healpix_pyramid(fine_resolution=5, estimation_resolution=4, corr_pool_levels=3).to(device)
model = OSLORAFTPyramid(pyr, feature_channels=(32, 64), context_channels=(32, 64)).to(device)

b, n = 1, pyr.num_fine_nodes
f1 = torch.rand(b, n, 3, device=device)
f2 = torch.rand(b, n, 3, device=device)
preds = model(f1, f2, pyr, iters=4)
assert preds[-1].shape == (b, n, 2), "bad fine prediction shape"
assert torch.count_nonzero(preds[0]) == 0, "cold-start flow should be zero at the fine grid"

gt = endpoint_from_tangent_flow(pyr.fine_level.points, torch.zeros(b, n, 2, device=device),
                                pyr.fine_level.basis_east, pyr.fine_level.basis_north)
valid = torch.ones(b, n, dtype=torch.bool, device=device)
loss = sequence_geodesic_loss(preds, gt, pyr.fine_level, valid)
loss.backward()
finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
assert finite, "non-finite gradient"
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  est_nodes={pyr.num_estimation_nodes} fine_nodes={n} params={params:,} "
      f"loss={loss.item():.4f} device={device.type}  OK")
PY

echo
echo "[tier 1.65] OSLO-RAFT-R CPU wiring suite (synthetic pyramid, no data) ..."
python run_oslo_raft_retina_smoke.py --skip-recovery

echo
echo "[tier 1.7] OSLO-RAFT-R real geometry: fast graph parity, pyramid cache, model ..."
RETINA_RES="$RETINA_RES" RETINA_SUP="$RETINA_SUP" RETINA_EST="$RETINA_EST" python - <<'PY'
import math, os, time
import torch

from spherical_flow.geometry import healpix_unit_vectors
from spherical_flow.healpix_pyramid import (
    build_healpix_pyramid, chunked_directional_knn_graph, healpix_neighbor_graph,
    load_pyramid, save_pyramid,
)
from spherical_flow.oslo_raft import sequence_geodesic_loss
from spherical_flow.geometry import endpoint_from_tangent_flow
from spherical_flow.oslo_raft_retina import OSLORAFTRetina

ret = int(os.environ["RETINA_RES"]); sup = int(os.environ["RETINA_SUP"]); est = int(os.environ["RETINA_EST"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (a) fast-graph sanity at r4: the topological 8-neighborhood vs kNN 8-nearest.
# Identical-set parity measured at 77.6% (kNN and HEALPix topology genuinely disagree
# on the marginal 8th neighbor at many pixels), so per plan §3.3's fallback the fast
# path serves ONLY N>100k levels (r7+), where kNN is unaffordable anyway. What SDPAConv
# needs from the graph is asserted strictly: valid indices, near-complete rows, high
# per-node overlap with the nearest set, and all neighbors within the local cell scale.
import math as _math
pts4 = healpix_unit_vectors(4)
n4g = pts4.size(0)
idx_f, wgt_f, val_f = healpix_neighbor_graph(4, pts4)
idx_k, _, _ = chunked_directional_knn_graph(pts4, 8, 2048)
assert idx_f.shape == (n4g, 8) and (idx_f >= 0).all() and (idx_f < n4g).all()
assert val_f.float().sum(dim=1).min() >= 7, "a pixel has < 7 valid topological neighbors"
overlap = sum(
    len(set(idx_f[i][val_f[i]].tolist()) & set(idx_k[i].tolist())) for i in range(n4g)
) / (8.0 * n4g)
d = (pts4.unsqueeze(1) - pts4[idx_f]).norm(dim=-1)[val_f]
spacing = _math.sqrt(4 * _math.pi / n4g)
assert d.max().item() < 2.5 * spacing, "topological neighbor beyond the local cell scale"
assert overlap >= 0.85, f"fast-graph/kNN overlap {overlap:.1%} < 85% at r4"
same = sum(
    1 for i in range(n4g)
    if set(idx_f[i][val_f[i]].tolist()) == set(idx_k[i].tolist())
)
print(f"  (a) fast-graph at r4: overlap {overlap:.1%}, identical sets {same / n4g:.1%}, "
      f"all neighbors < 2.5x spacing, rows valid  OK (fast path serves N>100k only)")

# (b) strict snap-constancy on REAL HEALPix (the irregular synthetic grid can't
# assert this bit-exactly; regular cells can): sub-half-node flow leaves the old
# est-level snap gather bit-identical.
from spherical_flow.oslo_raft_pyramid import AllPairsCorrelation, build_correlation_pyramid, pyramid_lookup
import torch.nn.functional as F
pyr4 = build_healpix_pyramid(fine_resolution=5, estimation_resolution=4, corr_pool_levels=2)
n4 = pyr4.num_estimation_nodes
f1 = F.normalize(torch.randn(1, n4, 32), dim=-1); f2 = F.normalize(torch.randn(1, n4, 32), dim=-1)
cp = build_correlation_pyramid(AllPairsCorrelation()(f1, f2), pyr4)
s4 = math.sqrt(4 * math.pi / n4)
dirn = torch.randn(1, n4, 2); dirn = dirn / dirn.norm(dim=-1, keepdim=True)
diff = (pyramid_lookup(cp, 0.3 * s4 * dirn, pyr4) - pyramid_lookup(cp, torch.zeros(1, n4, 2), pyr4))
n_est_cols = pyr4.levels[4].lookup_index.size(1)
assert diff[..., :n_est_cols].abs().max().item() == 0.0
print("  (b) real-HEALPix snap gather bit-identical under 0.3-node flow (the fixed bug)  OK")

# (c) retina pyramid build + disk cache round-trip
t0 = time.time()
pyr = build_healpix_pyramid(
    fine_resolution=sup, estimation_resolution=est, corr_pool_levels=3, retina_resolution=ret,
)
build_s = time.time() - t0
path = "/outputs/pyramid_cache/pyramid_ret%d_sup%d_est%d_cp3_cn8_ln24.pt" % (ret, sup, est)
save_pyramid(pyr, path)
t0 = time.time()
pyr2 = load_pyramid(path)
load_s = time.time() - t0
for r, lvl in pyr.levels.items():
    assert torch.equal(lvl.points, pyr2.levels[r].points)
    assert torch.equal(lvl.conv_index, pyr2.levels[r].conv_index)
    assert torch.equal(lvl.lookup_index, pyr2.levels[r].lookup_index)
assert pyr2.retina_resolution == ret
probe = torch.randn(4, 3)
assert torch.equal(pyr.levels[est].ang2pix(probe), pyr2.levels[est].ang2pix(probe))
print(f"  (c) pyramid ret={ret} sup={sup} est={est} built in {build_s:.1f}s, "
      f"cache round-trip OK ({load_s:.1f}s load, {os.path.getsize(path)/1e6:.0f} MB)")

# (d) one retina forward/backward on random frames (AMP on cuda); report peak VRAM
pyr = pyr.to(device)
model = OSLORAFTRetina(pyr).to(device)
n_ret, n_sup = pyr.retina_level.num_nodes, pyr.num_fine_nodes
b = 1
fr1 = torch.rand(b, n_ret, 3, device=device); fr2 = torch.rand(b, n_ret, 3, device=device)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
amp = device.type == "cuda"
with torch.amp.autocast(device_type="cuda", enabled=amp):
    preds = model(fr1, fr2, pyr, iters=4)
    assert preds[-1].shape == (b, n_sup, 2)
    assert torch.count_nonzero(preds[0]) == 0, "cold-start flow must be zero at r_sup"
    gt = endpoint_from_tangent_flow(pyr.fine_level.points, torch.zeros(b, n_sup, 2, device=device),
                                    pyr.fine_level.basis_east, pyr.fine_level.basis_north)
    loss = sequence_geodesic_loss(preds, gt, pyr.fine_level, torch.ones(b, n_sup, dtype=torch.bool, device=device))
loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if device.type == "cuda":
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    assert peak_gb < 20.0, f"peak VRAM {peak_gb:.1f} GB >= 20 GB"
    print(f"  (d) fwd/bwd B={b} AMP: params={params:,} loss={loss.item():.4f} peak VRAM {peak_gb:.2f} GB  OK")
else:
    print(f"  (d) fwd/bwd B={b} cpu: params={params:,} loss={loss.item():.4f}  OK")
PY
echo "[tier 1.7] OK"

echo
if [ -d "$OSLO_SHARDS" ] && [ -n "$(ls -A "$OSLO_SHARDS" 2>/dev/null)" ]; then
    echo "[tier 2] integration smoke on shards at $OSLO_SHARDS ..."
    DEVICE_ARGS="--device cpu"
    if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        DEVICE_ARGS="--device cuda --amp"
    fi
    python run_oslo_raft.py \
        --grid healpix --resolution "$RESOLUTION" \
        --shards "$OSLO_SHARDS" \
        $DEVICE_ARGS \
        --num-workers "$SMOKE_WORKERS" --batch-size 2 \
        --max-val-pairs 16 \
        --output-dir /outputs/oslo_raft_smoke \
        --smoke-test
    echo "[tier 2] OK"

    echo
    echo "[tier 2.7] OSLO-RAFT-R data integration: loader throughput + --retina smoke ..."
    # Loader probe (plan §4.5): retina-grid sampling is ~1.6M bilinear ERP lookups per
    # pair at r8 (a quarter at r7) on CPU workers — measure pairs/s before training.
    RETINA_RES="$RETINA_RES" RETINA_SUP="$RETINA_SUP" OSLO_SHARDS="$OSLO_SHARDS" python - <<'PY'
import os, time
import torch
from spherical_flow.geometry import healpix_unit_vectors
from spherical_flow.shard_dataset import ShardFlowDataset

ret = int(os.environ["RETINA_RES"]); sup = int(os.environ["RETINA_SUP"])
ds = ShardFlowDataset(
    os.environ["OSLO_SHARDS"], healpix_unit_vectors(ret), ("replica360", "train"),
    shuffle_shards=False, shuffle_buffer=0, max_pairs=8,
    so3_prob=1.0, synth_rot_prob=0.5,
    target_points=healpix_unit_vectors(sup),
)
it = iter(ds)
first = next(it)  # warm-up record (shard open)
t0 = time.time(); n = 0
for s in it:
    n += 1
pairs_s = n / max(time.time() - t0, 1e-9)
print(f"  loader: retina r{ret} frames {tuple(first['frame1'].shape)}, "
      f"targets {tuple(first['flow'].shape)}, {pairs_s:.2f} pairs/s/worker")
PY
    python run_oslo_raft.py \
        --grid healpix --retina \
        --retina-resolution "$RETINA_RES" --resolution "$RETINA_SUP" \
        --estimation-resolution "$RETINA_EST" \
        --pyramid-cache /outputs/pyramid_cache \
        --shards "$OSLO_SHARDS" \
        $DEVICE_ARGS \
        --num-workers "$SMOKE_WORKERS" --batch-size 1 \
        --iters "$SMOKE_ITERS" --eval-iters "$SMOKE_ITERS" \
        --train-sources replica360:train --val-sources replica360:val \
        --synth-rot-prob 0.5 --max-val-pairs 8 \
        --output-dir /outputs/oslo_raft_retina_smoke \
        --smoke-test
    echo "[tier 2.7] OK"
else
    echo "[tier 2] skipped (no shards at $OSLO_SHARDS) -- mount them to run the"
    echo "         full data + metrics integration smoke."
fi

echo
echo "container smoke complete."
