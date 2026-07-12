"""Bridge from the standard sphereflow-dataprep shards to HEALPix-node samples.

The `sphereflow-dataprep` project (sibling repo) normalizes every dataset into one
uniform ERP shard format and exposes a stable read contract::

    from sfprep import iter_split  # yields frame1/frame2/flow/valid/meta (ERP)

This module wraps that boundary and turns each ERP pair into HEALPix-node samples
with tangent-plane flow targets, reusing exactly the sampling path validated for
FLOW360 in :mod:`spherical_flow.flow360` (sample frames/flow at node pixels, build
the endpoint, ``logmap`` to tangent flow at the source node). The result is a
dataset-agnostic stream: FLOW360, Replica-360, and MPF all arrive in one format
with their diagnosed flow conventions already applied by the materializer.

SO(3) rotation augmentation (Phase 0, Week 1) plugs in at :func:`sample_pair_to_nodes`
via the ``query_points`` seam: sample frames + GT at rotated directions, then
re-express the target at the unrotated nodes. The plain path leaves it ``None``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .flow360 import bilinear_sample_erp
from .geometry import (
    equirectangular_pixels_to_unit_vectors,
    logmap,
    points_to_equirectangular_pixels,
    tangent_basis,
)

# Sibling location of the data-prep project, overridable via env var.
_DEFAULT_SFPREP_DIR = Path(__file__).resolve().parents[2] / "sphereflow-dataprep"

Source = Tuple[str, str]  # (dataset, split)


def _import_sfprep():
    """Import the shard reader from the sibling sphereflow-dataprep project.

    The read contract (``iter_shard`` / ``list_shards``) is the intended coupling
    point between the two repos; we add the sibling dir to ``sys.path`` only if the
    package is not already importable (e.g. not ``pip install -e``'d).
    """
    try:
        from sfprep.shard_reader import iter_shard, list_shards  # type: ignore
        return iter_shard, list_shards
    except ImportError:
        pass

    candidate = Path(os.environ.get("SPHEREFLOW_DATAPREP", _DEFAULT_SFPREP_DIR))
    if (candidate / "sfprep").is_dir():
        sys.path.insert(0, str(candidate))
        try:
            from sfprep.shard_reader import iter_shard, list_shards  # type: ignore
            return iter_shard, list_shards
        except ImportError:
            pass
    raise ImportError(
        "Could not import sfprep. Either `pip install -e ../sphereflow-dataprep`, "
        "or set SPHEREFLOW_DATAPREP to its path. Looked in: "
        f"{candidate}"
    )


def _to_chw_free_float(frame: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] ERP frame -> float32 [H,W,3] in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(frame)).float() / 255.0


def sample_pair_to_nodes(
    frame1_erp: torch.Tensor,
    frame2_erp: torch.Tensor,
    flow_erp: torch.Tensor,
    valid_erp: torch.Tensor,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    query_points: Optional[torch.Tensor] = None,
    endpoint_rotation: Optional[torch.Tensor] = None,
    target_points: Optional[torch.Tensor] = None,
    target_basis_east: Optional[torch.Tensor] = None,
    target_basis_north: Optional[torch.Tensor] = None,
    target_query_points: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Sample one ERP pair onto HEALPix nodes and build tangent-flow targets.

    Mirrors ``Flow360Dataset.__getitem__`` so every dataset shares one validated
    sampling path. ``flow_erp`` is assumed canonical (the materializer already
    applied each dataset's diagnosed convention).

    Args:
        frame1_erp, frame2_erp: float32 ``[H, W, 3]`` ERP frames in ``[0, 1]``.
        flow_erp: float32 ``[H, W, 2]`` ERP pixel displacement (frame1 -> frame2).
        valid_erp: ``[H, W]`` validity (bool or float); bilinearly resampled.
        points: ``[N, 3]`` frame node directions (the model's input/retina grid).
        basis_east, basis_north: tangent basis at ``points`` (``tangent_basis``).
        query_points: ``[N, 3]`` directions at which to *sample* frames. Defaults
            to ``points`` (no augmentation); the SO(3) seam passes rotated directions
            ``q = points @ R``.
        endpoint_rotation: ``[3, 3]`` rotation ``R``. The world-frame endpoint ``e(q)``
            (and thus the validity check) is computed in the unrotated frames; the
            target endpoint is rotated into the augmented world as ``e' = e @ R.T``
            before ``logmap`` at the target nodes. ``None`` leaves it untouched.
        target_points: ``[N_sup, 3]`` node directions where the flow target is sampled
            and *expressed* (OSLO-RAFT-R decouples this supervision grid from the frame
            grid). ``None`` keeps today's behavior bit-for-bit: one grid for both.
        target_basis_east, target_basis_north: tangent basis at ``target_points``
            (computed if omitted).
        target_query_points: rotated target directions under SO(3) augmentation
            (``target_points @ R``). Required whenever both ``target_points`` and
            ``endpoint_rotation`` are given — rotating the frames but sampling GT at
            unrotated targets would silently mis-pair them.

    Returns:
        ``frame1``/``frame2`` on the frame grid; ``flow``/``endpoint``/``valid`` on the
        target grid (= frame grid when ``target_points`` is None).
    """
    height, width = frame1_erp.shape[:2]
    if frame2_erp.shape[:2] != (height, width):
        raise ValueError("frame size mismatch between frame1 and frame2")
    if flow_erp.shape[:2] != (height, width):
        raise ValueError(
            f"flow size {tuple(flow_erp.shape[:2])} != frame size {(height, width)}"
        )

    sample_dirs = points if query_points is None else query_points
    u, v = points_to_equirectangular_pixels(sample_dirs, height, width)

    frame1 = bilinear_sample_erp(frame1_erp, u, v)
    frame2 = bilinear_sample_erp(frame2_erp, u, v)

    if target_points is None:
        # Single-grid path (pre-retina models): targets share the frame samples.
        t_points, t_east, t_north = points, basis_east, basis_north
        tu, tv = u, v
    else:
        if endpoint_rotation is not None and target_query_points is None:
            raise ValueError(
                "target_query_points is required when combining target_points with "
                "endpoint_rotation (pass target_points @ R)"
            )
        t_points = target_points
        if target_basis_east is None or target_basis_north is None:
            target_basis_east, target_basis_north = tangent_basis(target_points)
        t_east, t_north = target_basis_east, target_basis_north
        t_dirs = target_points if target_query_points is None else target_query_points
        tu, tv = points_to_equirectangular_pixels(t_dirs, height, width)

    sampled_flow = bilinear_sample_erp(flow_erp, tu, tv)

    valid_map = valid_erp.float().unsqueeze(-1) if valid_erp.ndim == 2 else valid_erp.float()
    sampled_valid = bilinear_sample_erp(valid_map, tu, tv).squeeze(-1) > 0.999

    endpoint_u = tu + sampled_flow[:, 0]
    endpoint_v = tv + sampled_flow[:, 1]
    inside_vertical = (endpoint_v >= 0.0) & (endpoint_v <= float(height - 1))
    valid = sampled_valid & inside_vertical

    endpoint = equirectangular_pixels_to_unit_vectors(endpoint_u, endpoint_v, height, width)
    if endpoint_rotation is not None:
        # Carry the world-frame endpoint into the augmented world: e' = e @ R.T.
        endpoint = endpoint @ endpoint_rotation.transpose(-1, -2)
    # Express the target as tangent flow at the (unrotated) target nodes.
    flow = logmap(t_points, endpoint, t_east, t_north).squeeze(0)

    return {
        "frame1": frame1,
        "frame2": frame2,
        "flow": flow,
        "endpoint": endpoint,
        "valid": valid,
    }


def synth_rotation_record(
    frame1_erp: torch.Tensor,
    frame_points: torch.Tensor,
    target_points: torch.Tensor,
    target_basis_east: torch.Tensor,
    target_basis_north: torch.Tensor,
    rotation: torch.Tensor,
    view_rotation: Optional[torch.Tensor] = None,
    frame2_erp: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Synthesize an exact-motion pair from one real ERP frame (§4.4, the bootstrap data).

    Frame 2 *is* frame 1 seen after the world rotates: perfect brightness constancy,
    exact correspondence at any magnitude — the in-house FlyingChairs that exercises
    matching by construction. Conventions match ``SyntheticRotationFlowDataset``
    (``R = rotation_matrix(axis, angle)``, i.e. ``R @ v`` rotates ``v``):

      - world endpoint of direction ``d`` is ``e = R d``  (rows: ``d @ R.T``);
      - frame 2 observed at ``d`` shows what frame 1 held at ``R^{-1} d``  (rows:
        ``d @ R``), so ``frame2 = frame1_erp`` sampled at ``frame_points @ R``.

    ``view_rotation`` composes the usual SO(3) *viewpoint* augmentation ``R_v`` (a
    different thing: it rotates where the fixed nodes look, not the world between
    frames): frames are sampled at ``q = p @ R_v`` and the endpoint is carried back
    with ``e' = e @ R_v.T``, exactly as the ``sample_pair_to_nodes`` SO(3) seam.

    Returns the same dict shape as :func:`sample_pair_to_nodes`; ``valid`` is all True
    (the rotation is defined everywhere on the sphere).
    """
    height, width = frame1_erp.shape[:2]

    q = frame_points if view_rotation is None else frame_points @ view_rotation
    u1, v1 = points_to_equirectangular_pixels(q, height, width)
    frame1 = bilinear_sample_erp(frame1_erp, u1, v1)
    u2, v2 = points_to_equirectangular_pixels(q @ rotation, height, width)
    # frame2_erp lets the nuisance levers corrupt the raster frame 2 is observed
    # from (edge-anchored corruption needs spatial structure); None keeps the exact
    # brightness-constancy pair bit-for-bit.
    frame2 = bilinear_sample_erp(frame1_erp if frame2_erp is None else frame2_erp, u2, v2)

    qt = target_points if view_rotation is None else target_points @ view_rotation
    endpoint = qt @ rotation.transpose(-1, -2)
    if view_rotation is not None:
        endpoint = endpoint @ view_rotation.transpose(-1, -2)
    flow = logmap(target_points, endpoint, target_basis_east, target_basis_north).squeeze(0)

    return {
        "frame1": frame1,
        "frame2": frame2,
        "flow": flow,
        "endpoint": endpoint,
        "valid": torch.ones(target_points.size(0), dtype=torch.bool),
    }


def _record_tensors(record: Dict[str, object]):
    """Decode a raw shard record into (frame1, frame2, flow, valid) ERP tensors."""
    frame1_erp = _to_chw_free_float(record["frame1"])  # type: ignore[index]
    frame2_erp = _to_chw_free_float(record["frame2"])  # type: ignore[index]
    flow_erp = torch.from_numpy(np.ascontiguousarray(record["flow"])).float()  # type: ignore[index]
    valid_erp = torch.from_numpy(np.ascontiguousarray(record["valid"]))  # type: ignore[index]
    return frame1_erp, frame2_erp, flow_erp, valid_erp


def _attach_meta(sample: Dict[str, object], record: Dict[str, object]) -> Dict[str, object]:
    meta = record.get("meta", {})  # type: ignore[union-attr]
    sample["uid"] = record["uid"]  # type: ignore[index]
    sample["dataset"] = meta.get("dataset", "")
    sample["sequence"] = meta.get("sequence", "")
    sample["direction"] = meta.get("direction", "")
    return sample


def _sample_record(
    record: Dict[str, object],
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    target_points: Optional[torch.Tensor] = None,
    target_basis_east: Optional[torch.Tensor] = None,
    target_basis_north: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Turn one raw shard record into a node-sample dict (+ provenance)."""
    frame1_erp, frame2_erp, flow_erp, valid_erp = _record_tensors(record)
    sample = sample_pair_to_nodes(
        frame1_erp, frame2_erp, flow_erp, valid_erp, points, basis_east, basis_north,
        target_points=target_points,
        target_basis_east=target_basis_east,
        target_basis_north=target_basis_north,
    )
    return _attach_meta(sample, record)


class ShardFlowDataset(IterableDataset):
    """Streaming dataset over standard shards, yielding HEALPix-node samples.

    Streaming (``IterableDataset``) is the natural fit for the WebDataset-style
    shards: shard order is shuffled per epoch, shards are split across DataLoader
    workers, and an optional in-memory buffer gives sample-level shuffle. For small
    fixed sets (overfit / deterministic val) use :func:`load_shard_subset` instead.
    """

    def __init__(
        self,
        shards_dir: str | Path,
        points: torch.Tensor,
        sources: Source | Sequence[Source],
        *,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 0,
        direction: Optional[str] = None,
        seed: int = 0,
        max_pairs: Optional[int] = None,
        so3_prob: float = 0.0,
        so3_max_angle_deg: float = 180.0,
        so3_uniform: bool = False,
        target_points: Optional[torch.Tensor] = None,
        synth_rot_prob: float = 0.0,
        synth_rot_min_deg: float = 1.0,
        synth_rot_max_deg: float = 15.0,
        synth_photo_scale: float = 0.0,
        synth_photo_noise_std: float = 0.0,
        synth_edge_corrupt_delta: float = 0.0,
    ) -> None:
        if points.ndim != 2 or points.size(-1) != 3:
            raise ValueError("points must have shape [N, 3]")
        self.shards_dir = Path(shards_dir)
        self.points = points.detach().cpu().float()
        self.basis_east, self.basis_north = tangent_basis(self.points)
        self.sources: List[Source] = (
            [sources] if isinstance(sources, tuple) else list(sources)
        )
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = int(shuffle_buffer)
        self.direction = direction
        self.seed = seed
        self.epoch = 0
        self.max_pairs = max_pairs  # cap samples yielded per epoch (debug/overfit)
        # SO(3) augmentation is applied at sampling time (needs ERP frames + GT).
        self.so3_prob = float(so3_prob)
        self.so3_max_angle_deg = float(so3_max_angle_deg)
        self.so3_uniform = bool(so3_uniform)
        # OSLO-RAFT-R: frames at `points` (retina), targets at `target_points` (supervision).
        if target_points is not None:
            target_points = target_points.detach().cpu().float()
            self.target_basis_east, self.target_basis_north = tangent_basis(target_points)
        else:
            self.target_basis_east = self.target_basis_north = None
        self.target_points = target_points
        # Synthetic-rotation motion source (§4.4): with this probability a record's
        # frame2/targets are REPLACED by an exact rotation of its own frame1.
        self.synth_rot_prob = float(synth_rot_prob)
        self.synth_rot_min_deg = float(synth_rot_min_deg)
        self.synth_rot_max_deg = float(synth_rot_max_deg)
        # P2C nuisance axis: asymmetric photometric jitter on the synthetic frame 2
        # (RAFT-parity ranges x scale). Params are drawn AFTER the rotation draws, so
        # runs differing only in scale share identical rotations/frames — and scale 0
        # adds no RNG draws, keeping existing runs bit-identical.
        self.synth_photo_scale = float(synth_photo_scale)
        # P0b: per-pixel iid Gaussian noise on the synthetic frame 2, std in 1/255
        # units — the spatially-unstructured nuisance axis (global jitter above is
        # the spatially-coherent one).
        self.synth_photo_noise_std = float(synth_photo_noise_std)
        # Edge-modulated corruption (the *measured* real-nuisance shape): applied to
        # the RASTER copy that synth frame 2 is sampled from, so it needs spatial
        # structure (gradient envelope + correlated noise) unavailable node-wise.
        self.synth_edge_corrupt_delta = float(synth_edge_corrupt_delta)

        self._iter_shard, self._list_shards = _import_sfprep()
        self._shards: List[Path] = []
        for dataset, split in self.sources:
            self._shards.extend(self._list_shards(self.shards_dir, dataset, split))
        if not self._shards:
            raise RuntimeError(
                f"No shards found in {self.shards_dir} for sources={self.sources}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _worker_shards(self) -> List[Path]:
        shards = list(self._shards)
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(shards)
        info = get_worker_info()
        if info is not None:
            shards = shards[info.id :: info.num_workers]
        return shards

    def _raw_stream(self) -> Iterator[Dict[str, object]]:
        for shard in self._worker_shards():
            for record in self._iter_shard(shard):
                if self.direction is not None:
                    if record.get("meta", {}).get("direction") != self.direction:  # type: ignore[union-attr]
                        continue
                yield record

    def __iter__(self) -> Iterator[Dict[str, object]]:
        stream = self._raw_stream()
        if self.shuffle_buffer > 1:
            stream = _shuffle_buffer(stream, self.shuffle_buffer, self.seed + self.epoch)

        gen = None
        so3 = None
        photo_gen = None
        if self.so3_prob > 0.0 or self.synth_rot_prob > 0.0:
            # Lazy import avoids a shard_dataset <-> so3_augment import cycle.
            from .so3_augment import sample_rotation, so3_augment_pair

            info = get_worker_info()
            worker_id = info.id if info is not None else 0
            gen = torch.Generator().manual_seed(self.seed + self.epoch * 131 + worker_id)
            so3 = (sample_rotation, so3_augment_pair)
            if (self.synth_photo_scale > 0.0 or self.synth_photo_noise_std > 0.0
                    or self.synth_edge_corrupt_delta > 0.0):
                # Dedicated stream: jitter/noise draws must never advance the main
                # gen, or runs differing only in scale would see different rotations
                # from the second record on. Same raw draws at every scale/std -> the
                # nuisance curves are nested (same directions, scaled magnitudes).
                photo_gen = torch.Generator().manual_seed(
                    self.seed + self.epoch * 131 + worker_id + 777_000_000
                )

        yielded = 0
        for record in stream:
            if self.max_pairs is not None and yielded >= self.max_pairs:
                break
            if (
                self.synth_rot_prob > 0.0
                and float(torch.rand((), generator=gen)) < self.synth_rot_prob
            ):
                yield self._synth_record(record, gen, so3[0], photo_gen)
            elif (
                self.so3_prob > 0.0
                and float(torch.rand((), generator=gen)) < self.so3_prob
            ):
                yield self._augment_record(record, gen, so3)
            else:
                yield _sample_record(
                    record, self.points, self.basis_east, self.basis_north,
                    self.target_points, self.target_basis_east, self.target_basis_north,
                )
            yielded += 1

    def _augment_record(self, record, gen, so3) -> Dict[str, object]:
        sample_rotation, so3_augment_pair = so3
        frame1_erp, frame2_erp, flow_erp, valid_erp = _record_tensors(record)
        rotation = sample_rotation(
            gen, max_angle_deg=self.so3_max_angle_deg, uniform_so3=self.so3_uniform
        )
        sample = so3_augment_pair(
            frame1_erp, frame2_erp, flow_erp, valid_erp,
            self.points, rotation, self.basis_east, self.basis_north,
            target_points=self.target_points,
            target_basis_east=self.target_basis_east,
            target_basis_north=self.target_basis_north,
        )
        return _attach_meta(sample, record)

    def _synth_record(self, record, gen, sample_rotation, photo_gen=None) -> Dict[str, object]:
        """Replace a record's motion by an exact rotation of its own frame 1 (§4.4)."""
        from .so3_augment import rotation_matrix

        frame1_erp = _to_chw_free_float(record["frame1"])  # type: ignore[index]
        axis = torch.randn(3, generator=gen)
        angle_deg = self.synth_rot_min_deg + float(torch.rand((), generator=gen)) * (
            self.synth_rot_max_deg - self.synth_rot_min_deg
        )
        rotation = rotation_matrix(axis, torch.tensor(angle_deg * torch.pi / 180.0))
        # Compose with the usual viewpoint augmentation (independent draw, same knob).
        view_rotation = None
        if self.so3_prob > 0.0 and float(torch.rand((), generator=gen)) < self.so3_prob:
            view_rotation = sample_rotation(
                gen, max_angle_deg=self.so3_max_angle_deg, uniform_so3=self.so3_uniform
            )
        if self.target_points is not None:
            t_points, t_east, t_north = (
                self.target_points, self.target_basis_east, self.target_basis_north
            )
        else:
            t_points, t_east, t_north = self.points, self.basis_east, self.basis_north
        frame2_src = frame1_erp
        if photo_gen is not None and self.synth_edge_corrupt_delta > 0.0:
            from .photometric import edge_corruption

            frame2_src = edge_corruption(
                frame1_erp, photo_gen, self.synth_edge_corrupt_delta
            )
        sample = synth_rotation_record(
            frame1_erp, self.points, t_points, t_east, t_north, rotation, view_rotation,
            frame2_erp=frame2_src,
        )
        if photo_gen is not None:
            from .photometric import apply_jitter, apply_noise, sample_jitter_params

            if self.synth_photo_scale > 0.0:
                params = sample_jitter_params(photo_gen, self.synth_photo_scale)
                sample["frame2"] = apply_jitter(sample["frame2"], params)
            if self.synth_photo_noise_std > 0.0:
                sample["frame2"] = apply_noise(
                    sample["frame2"], photo_gen, self.synth_photo_noise_std
                )
        return _attach_meta(sample, record)


def _shuffle_buffer(
    stream: Iterator[Dict[str, object]], size: int, seed: int
) -> Iterator[Dict[str, object]]:
    """Reservoir-style shuffle buffer for streamed records."""
    rng = np.random.default_rng(seed)
    buffer: List[Dict[str, object]] = []
    for item in stream:
        if len(buffer) < size:
            buffer.append(item)
            continue
        j = int(rng.integers(0, size))
        yield buffer[j]
        buffer[j] = item
    rng.shuffle(buffer)
    yield from buffer


def load_shard_subset(
    shards_dir: str | Path,
    points: torch.Tensor,
    sources: Source | Sequence[Source],
    *,
    max_pairs: Optional[int] = None,
    direction: Optional[str] = None,
    target_points: Optional[torch.Tensor] = None,
) -> List[Dict[str, object]]:
    """Eagerly materialize node samples into a list (overfit / deterministic val).

    Random-access friendly and order-stable; only use it for small caps — each
    sample holds two ``[N, 3]`` node frames in memory.
    """
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape [N, 3]")
    points = points.detach().cpu().float()
    basis_east, basis_north = tangent_basis(points)
    target_basis_east = target_basis_north = None
    if target_points is not None:
        target_points = target_points.detach().cpu().float()
        target_basis_east, target_basis_north = tangent_basis(target_points)
    iter_shard, list_shards = _import_sfprep()

    out: List[Dict[str, object]] = []
    src_list: List[Source] = [sources] if isinstance(sources, tuple) else list(sources)
    for dataset, split in src_list:
        for shard in list_shards(Path(shards_dir), dataset, split):
            for record in iter_shard(shard):
                if direction is not None and record.get("meta", {}).get("direction") != direction:
                    continue
                out.append(_sample_record(
                    record, points, basis_east, basis_north,
                    target_points, target_basis_east, target_basis_north,
                ))
                if max_pairs is not None and len(out) >= max_pairs:
                    return out
    return out
