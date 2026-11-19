"""Error metrics over predicted futures (B, K, 13) vs targets (B, K, 13); k = 1 … K."""
from __future__ import annotations

import torch

from physiworld_uav.models import quat


def position_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred[..., 0:3] - target[..., 0:3]).norm(dim=-1)


def velocity_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred[..., 3:6] - target[..., 3:6]).norm(dim=-1)


def attitude_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Quaternion geodesic angle in degrees."""
    return torch.rad2deg(quat.geodesic_angle(pred[..., 6:10], target[..., 6:10]))


def angular_velocity_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred[..., 10:13] - target[..., 10:13]).norm(dim=-1)
