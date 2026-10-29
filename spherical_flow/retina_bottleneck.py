"""ERP -> HEALPix retina -> ERP: what OSLO-RAFT-R sees, handed back to a planar network.

OSLO samples each ERP frame bilinearly at the HEALPix nodes of its retina (r7 = 196,608
nodes) and never sees anything else. ``RetinaBottleneck`` does the same sampling and then
rebuilds the ERP raster from the node values with HEALPix bilinear interpolation, so a
planar RAFT fed the result gets exactly the retina's information on its usual grid. Used
by ``run_erp_raft_train.py --retina-bottleneck`` and the RAFT eval paths.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .flow360 import bilinear_sample_erp
from .geometry import healpix_unit_vectors, points_to_equirectangular_pixels


class RetinaBottleneck:
    def __init__(self, height: int, width: int, resolution: int = 7) -> None:
        import astropy.units as un
        from astropy_healpix import HEALPix

        self.height, self.width, self.resolution = height, width, resolution
        nodes = healpix_unit_vectors(resolution, nest=True, dtype=torch.float64)
        nu, nv = points_to_equirectangular_pixels(nodes, height, width)
        self.node_u, self.node_v = nu.float(), nv.float()
        # ERP pixel centers, same convention as equirectangular_pixels_to_unit_vectors
        v, u = np.meshgrid(np.arange(height, dtype=np.float64),
                           np.arange(width, dtype=np.float64), indexing="ij")
        lon = ((u + 0.5) / width) * 2.0 * np.pi - np.pi
        lat = np.pi / 2.0 - ((v + 0.5) / height) * np.pi
        hp = HEALPix(nside=1 << resolution, order="nested")
        idx, w = hp.bilinear_interpolation_weights(  # astropy wants lon in [0, 2 pi)
            np.mod(lon.ravel(), 2.0 * np.pi) * un.rad, lat.ravel() * un.rad)
        self.index = torch.as_tensor(np.asarray(idx).T.astype(np.int64))      # [HW, 4]
        self.weight = torch.as_tensor(np.asarray(w).T.astype(np.float32))     # [HW, 4]

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """``[C, H, W]`` or ``[B, C, H, W]`` frames (any dtype) -> same shape and dtype."""
        single = frames.dim() == 3
        x = frames[None] if single else frames
        b, c, h, w = x.shape
        if (h, w) != (self.height, self.width):
            raise ValueError(f"bottleneck built for {self.height}x{self.width}, got {h}x{w}")
        dev = x.device
        index, weight = self.index.to(dev), self.weight.to(dev)
        nu, nv = self.node_u.to(dev), self.node_v.to(dev)
        out = []
        for img in x.float():
            nodes = bilinear_sample_erp(img.permute(1, 2, 0), nu, nv)            # [N, C]
            erp = (nodes[index] * weight[..., None]).sum(dim=1)                  # [HW, C]
            out.append(erp.t().reshape(c, h, w))
        y = torch.stack(out)
        if not frames.dtype.is_floating_point:
            y = y.round().clamp(0, 255)
        y = y.to(frames.dtype)
        return y[0] if single else y


@lru_cache(maxsize=4)
def get_bottleneck(height: int, width: int, resolution: int) -> RetinaBottleneck:
    return RetinaBottleneck(height, width, resolution)
