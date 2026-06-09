import math
import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _load_sdpa_conv_class():
    module_path = Path(__file__).resolve().parents[1] / "spherical_models" / "sdpa_conv.py"
    spec = importlib.util.spec_from_file_location("_oslo_sdpa_conv", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SDPAConv from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SDPAConv


SDPAConv = _load_sdpa_conv_class()


class SphereConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = SDPAConv(in_channels, out_channels, kernel_size=kernel_size, node_dim=1)
        self.activation = torch.nn.ReLU(inplace=True) if activation else None

    def forward(
        self,
        x: torch.Tensor,
        index: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        valid_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv(x, neighbors_indices=index, neighbors_weights=weight, valid_index=valid_index)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SphereFeatureEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                SphereConvBlock(in_channels, hidden_channels),
                SphereConvBlock(hidden_channels, hidden_channels),
                SphereConvBlock(hidden_channels, feature_channels, activation=False),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        weight: torch.Tensor,
        valid_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, index, weight, valid_index)
        return x


class LocalCostVolume(torch.nn.Module):
    """Local HEALPix correlation volume over center + directional neighbors."""

    def __init__(self, num_neighbors: int = 8, normalize_features: bool = True) -> None:
        super().__init__()
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        self.num_neighbors = num_neighbors
        self.normalize_features = normalize_features

    @property
    def out_channels(self) -> int:
        return 1 + self.num_neighbors

    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        neighbors_indices: torch.Tensor,
        valid_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.normalize_features:
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)

        bsz, n_nodes, channels = features1.shape
        scale = math.sqrt(channels)
        center = (features1 * features2).sum(dim=-1, keepdim=True) / scale

        if neighbors_indices.size(1) < self.num_neighbors:
            raise ValueError(
                f"Cost volume needs {self.num_neighbors} neighbors, "
                f"but graph only has {neighbors_indices.size(1)}"
            )
        idx = neighbors_indices[:, : self.num_neighbors].to(features1.device)
        flat_idx = idx.reshape(-1)
        gathered = features2.index_select(1, flat_idx)
        gathered = gathered.reshape(bsz, n_nodes, self.num_neighbors, channels)
        neighbor_corr = (features1.unsqueeze(2) * gathered).sum(dim=-1) / scale

        if valid_index is not None:
            if valid_index.size(1) < self.num_neighbors:
                raise ValueError(
                    f"Cost volume needs {self.num_neighbors} valid flags, "
                    f"but mask only has {valid_index.size(1)}"
                )
            valid = valid_index[:, : self.num_neighbors].to(features1.device)
            neighbor_corr = neighbor_corr.masked_fill(~valid.unsqueeze(0), 0.0)

        return torch.cat([center, neighbor_corr], dim=-1)


class SphericalFlowMVP(torch.nn.Module):
    """Small OSLO-style spherical optical-flow baseline.

    It predicts a local tangent vector [east, north] in radians for each
    HEALPix node. This is intentionally compact: the goal is to validate whether
    OSLO features and local spherical correlations are promising before porting
    a full RAFT/PWC update stack.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        feature_channels: int = 48,
        max_flow_rad: float = 0.25,
        include_points: bool = True,
        zero_init_flow_head: bool = True,
        cost_num_neighbors: int = 8,
        use_displacement_prior: bool = False,
        cost_prior_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.include_points = include_points
        self.max_flow_rad = max_flow_rad
        self.use_displacement_prior = use_displacement_prior
        self.cost_prior_temperature = cost_prior_temperature
        self.encoder = SphereFeatureEncoder(in_channels, hidden_channels, feature_channels)
        self.cost_volume = LocalCostVolume(num_neighbors=cost_num_neighbors)

        motion_channels = feature_channels * 2 + self.cost_volume.out_channels + in_channels
        if use_displacement_prior:
            motion_channels += 2 + self.cost_volume.out_channels * 2
        if include_points:
            motion_channels += 3

        out_channels = 3 if use_displacement_prior else 2
        self.predictor = torch.nn.ModuleList(
            [
                SphereConvBlock(motion_channels, hidden_channels),
                SphereConvBlock(hidden_channels, hidden_channels),
                SphereConvBlock(hidden_channels, hidden_channels // 2),
                SphereConvBlock(hidden_channels // 2, out_channels, activation=False),
            ]
        )
        if zero_init_flow_head:
            self._zero_init_flow_head()

    def _zero_init_flow_head(self) -> None:
        final_conv = self.predictor[-1].conv
        torch.nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            torch.nn.init.zeros_(final_conv.bias)
            if self.use_displacement_prior:
                final_conv.bias.data[2].fill_(-8.0)

    def _build_displacement_prior(
        self,
        cost: torch.Tensor,
        cost_offsets: torch.Tensor,
        cost_candidate_valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cost_offsets.size(1) != cost.size(-1):
            raise ValueError(
                f"cost_offsets has {cost_offsets.size(1)} candidates, "
                f"but cost volume has {cost.size(-1)}"
            )
        cost_offsets = cost_offsets.to(device=cost.device, dtype=cost.dtype)
        logits = cost / max(float(self.cost_prior_temperature), 1e-6)
        if cost_candidate_valid is not None:
            valid = cost_candidate_valid.to(cost.device)
            if valid.size(1) != cost.size(-1):
                raise ValueError(
                    f"cost_candidate_valid has {valid.size(1)} candidates, "
                    f"but cost volume has {cost.size(-1)}"
                )
            logits = logits.masked_fill(~valid.unsqueeze(0), -1e4)
        probs = torch.softmax(logits.float(), dim=-1).to(dtype=cost.dtype)
        flow_prior = (probs.unsqueeze(-1) * cost_offsets.unsqueeze(0)).sum(dim=2)
        weighted_offsets = (probs.unsqueeze(-1) * cost_offsets.unsqueeze(0)).reshape(
            cost.size(0),
            cost.size(1),
            -1,
        )
        return flow_prior, weighted_offsets, probs

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        index: torch.Tensor,
        weight: torch.Tensor,
        valid_index: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        cost_index: Optional[torch.Tensor] = None,
        cost_valid_index: Optional[torch.Tensor] = None,
        cost_offsets: Optional[torch.Tensor] = None,
        cost_candidate_valid: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        f1 = self.encoder(frame1, index, weight, valid_index)
        f2 = self.encoder(frame2, index, weight, valid_index)
        if cost_index is None:
            cost_index = index
        if cost_valid_index is None:
            cost_valid_index = valid_index
        cost = self.cost_volume(f1, f2, cost_index, cost_valid_index)

        inputs = [f1, f2, cost, frame2 - frame1]
        debug = {"features1": f1, "features2": f2, "cost": cost}
        flow_prior = None
        if self.use_displacement_prior:
            if cost_offsets is None:
                raise ValueError("cost_offsets must be provided when use_displacement_prior=True")
            flow_prior, weighted_offsets, cost_prob = self._build_displacement_prior(
                cost,
                cost_offsets,
                cost_candidate_valid,
            )
            inputs.extend([flow_prior, weighted_offsets])
            debug.update(
                {
                    "flow_prior": flow_prior,
                    "cost_prob": cost_prob,
                }
            )
        if self.include_points:
            if points is None:
                raise ValueError("points must be provided when include_points=True")
            inputs.append(points.to(frame1.device).unsqueeze(0).expand(frame1.size(0), -1, -1))

        x = torch.cat(inputs, dim=-1)
        for block in self.predictor:
            x = block(x, index, weight, valid_index)
        if self.use_displacement_prior:
            if flow_prior is None:
                raise RuntimeError("flow_prior was not built")
            residual = self.max_flow_rad * torch.tanh(x[..., :2])
            gate = torch.sigmoid(x[..., 2:3])
            flow = residual + gate * flow_prior
            debug.update({"residual": residual, "gate": gate})
        else:
            flow = self.max_flow_rad * torch.tanh(x)

        if return_debug:
            return flow, debug
        return flow


class RaftResidualCorrector(torch.nn.Module):
    """OSLO-style HEALPix residual head conditioned on frozen RAFT flow.

    The model predicts a small tangent-space delta added to a cached RAFT
    tangent flow. Its final layer is zero-initialized, so the initial full
    prediction is exactly the RAFT baseline.
    """

    def __init__(
        self,
        hidden_channels: int = 48,
        residual_max_rad: float = 0.05,
        include_points: bool = True,
        zero_init_residual_head: bool = True,
    ) -> None:
        super().__init__()
        self.residual_max_rad = residual_max_rad
        self.include_points = include_points
        in_channels = 3 + 3 + 3 + 2
        if include_points:
            in_channels += 3
        self.blocks = torch.nn.ModuleList(
            [
                SphereConvBlock(in_channels, hidden_channels),
                SphereConvBlock(hidden_channels, hidden_channels),
                SphereConvBlock(hidden_channels, hidden_channels // 2),
                SphereConvBlock(hidden_channels // 2, 2, activation=False),
            ]
        )
        if zero_init_residual_head:
            self._zero_init_residual_head()

    def _zero_init_residual_head(self) -> None:
        final_conv = self.blocks[-1].conv
        torch.nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            torch.nn.init.zeros_(final_conv.bias)

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        raft_flow: torch.Tensor,
        index: torch.Tensor,
        weight: torch.Tensor,
        valid_index: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        inputs = [frame1, frame2, frame2 - frame1, raft_flow]
        if self.include_points:
            if points is None:
                raise ValueError("points must be provided when include_points=True")
            inputs.append(points.to(frame1.device).unsqueeze(0).expand(frame1.size(0), -1, -1))
        x = torch.cat(inputs, dim=-1)
        for block in self.blocks:
            x = block(x, index, weight, valid_index)
        residual = self.residual_max_rad * torch.tanh(x)
        pred = raft_flow + residual
        if return_residual:
            return pred, residual
        return pred
