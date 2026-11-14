#!/usr/bin/env bash
# Downloads the released weights of the Dual-Projection Fusion upright estimator (arXiv 2512.00911,
# MIT License) used by the *ref views of scripts/orientation_suite.sh. 993 MB, checked by SHA-256.
# If the container has no network, download the URL below on the host into the same folder.
set -euo pipefail
DEST="${OUTPUT_DIR:-/outputs}/external_weights/dpf_imp_align"
URL="https://github.com/YuhaoShine/DualProjectionFusion/releases/download/DPF_UpPanoGeneration_Imp-Align/model.pth"
SHA="6ac13fd243df342b89cfca65db019792a782d695f89d9de299772d0dbbf3dca7"
mkdir -p "$DEST"
python - "$DEST/model.pth" "$URL" "$SHA" <<'PY'
import hashlib, os, sys, urllib.request

path, url, sha = sys.argv[1:]


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


if os.path.isfile(path) and digest(path) == sha:
    print(f"present and verified: {path}")
    sys.exit(0)
print(f"downloading {url}", flush=True)
urllib.request.urlretrieve(url, path + ".part")
got = digest(path + ".part")
if got != sha:
    os.remove(path + ".part")
    sys.exit(f"SHA-256 mismatch: {got}")
os.replace(path + ".part", path)
print(f"saved and verified: {path}")
PY
