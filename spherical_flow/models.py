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
    ) -> None:
        super().__init__()
        self.include_points = include_points
        self.max_flow_rad = max_flow_rad
        self.encoder = SphereFeatureEncoder(in_channels, hidden_channels, feature_channels)
        self.cost_volume = LocalCostVolume(num_neighbors=cost_num_neighbors)

        motion_channels = feature_channels * 2 + self.cost_volume.out_channels + in_channels
        if include_points:
            motion_channels += 3

        self.predictor = torch.nn.ModuleList(
            [
                SphereConvBlock(motion_channels, hidden_channels),
                SphereConvBlock(hidden_channels, hidden_channels),
                SphereConvBlock(hidden_channels, hidden_channels // 2),
                SphereConvBlock(hidden_channels // 2, 2, activation=False),
            ]
        )
        if zero_init_flow_head:
            self._zero_init_flow_head()

    def _zero_init_flow_head(self) -> None:
        final_conv = self.predictor[-1].conv
        torch.nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            torch.nn.init.zeros_(final_conv.bias)

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
        if self.include_points:
            if points is None:
                raise ValueError("points must be provided when include_points=True")
            inputs.append(points.to(frame1.device).unsqueeze(0).expand(frame1.size(0), -1, -1))

        x = torch.cat(inputs, dim=-1)
        for block in self.predictor:
            x = block(x, index, weight, valid_index)
        flow = self.max_flow_rad * torch.tanh(x)

        if return_debug:
            return flow, {"features1": f1, "features2": f2, "cost": cost}
        return flow
