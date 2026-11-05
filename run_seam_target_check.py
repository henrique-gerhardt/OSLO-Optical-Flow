"""How many supervision-node targets does the seam corrupt, per source?

Some shards store GT du unwrapped (a pixel next to the +-180 deg seam that moves one
pixel across it can read du = +-(W-1)). ``sample_pair_to_nodes`` bilinearly samples
du, so a node whose stencil mixes a wrapped and an unwrapped neighbor gets a du near
+-W/2: an endpoint on the far side of the sphere. This compares the targets built
from the raw flow against targets built after wrapping du to [-W/2, W/2).

Read-only; prints per source: fraction of pixels with |du| > W/2, fraction of valid
nodes whose target moves by > 1 deg and > 90 deg, and the mean target change.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from run_raft_shard_baseline import iter_source_records, parse_sources
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.shard_dataset import _to_chw_free_float, sample_pair_to_nodes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards", default="/data/shards")
    p.add_argument("--sources", default="flowscape:train,flowscape:test,chairs360:train,replica360:train")
    p.add_argument("--resolution", type=int, default=6)
    p.add_argument("--max-pairs", type=int, default=50)
    args = p.parse_args()
    pts = healpix_unit_vectors(args.resolution)
    be, bn = tangent_basis(pts)
    for source in parse_sources(args.sources):
        wide = off1 = off90 = err = nodes = pixels = 0.0
        seen = 0
        for r in iter_source_records(Path(args.shards), [source]):
            f = torch.from_numpy(np.ascontiguousarray(r["flow"])).float()
            width = f.shape[1]
            val = torch.from_numpy(np.ascontiguousarray(r["valid"]))
            f1, f2 = _to_chw_free_float(r["frame1"]), _to_chw_free_float(r["frame2"])
            raw = sample_pair_to_nodes(f1, f2, f, val, pts, be, bn)
            fw = f.clone()
            fw[..., 0] = torch.remainder(fw[..., 0] + width / 2, width) - width / 2
            wrapped = sample_pair_to_nodes(f1, f2, fw, val, pts, be, bn)
            ang = torch.rad2deg(torch.acos((raw["endpoint"] * wrapped["endpoint"]).sum(-1).clamp(-1, 1)))
            m = raw["valid"] & wrapped["valid"]
            wide += (f[..., 0].abs() > width / 2).sum().item()
            pixels += f[..., 0].numel()
            off1 += (ang[m] > 1).sum().item()
            off90 += (ang[m] > 90).sum().item()
            err += ang[m].sum().item()
            nodes += m.sum().item()
            seen += 1
            if seen >= args.max_pairs:
                break
        if seen == 0:
            print(f"{source[0]}:{source[1]}: no pairs")
            continue
        print(f"{source[0]}:{source[1]} ({seen} pairs, r{args.resolution}): pixels |du|>W/2 "
              f"{100 * wide / pixels:.3f}%  nodes off >1deg {100 * off1 / nodes:.3f}%  "
              f">90deg {100 * off90 / nodes:.3f}%  mean target change {err / nodes:.4f} deg",
              flush=True)


if __name__ == "__main__":
    main()
