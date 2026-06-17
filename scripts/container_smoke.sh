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
        --num-workers 2 --batch-size 2 \
        --max-val-pairs 16 \
        --output-dir /outputs/oslo_raft_smoke \
        --smoke-test
    echo "[tier 2] OK"
else
    echo "[tier 2] skipped (no shards at $OSLO_SHARDS) -- mount them to run the"
    echo "         full data + metrics integration smoke."
fi

echo
echo "container smoke complete."
