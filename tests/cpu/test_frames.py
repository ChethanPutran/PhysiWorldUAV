"""Frame conventions (plan §5-A11, §6.1-F4): Hamilton wxyz, body→world; ω_B = R_WBᵀ ω_W."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from physiworld_uav.core.state_extractor import (
    angular_velocity_body_np, angular_velocity_body_torch, compute_imu_from_state,
    quat_rotate_inverse_np, quat_rotate_inverse_torch, quat_rotate_np, quat_rotate_torch,
    quat_wxyz_to_matrix, quat_wxyz_to_matrix_np,
)

from conftest import omni_drones_available


def _rand_q(n, seed=0):
    return torch.nn.functional.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_rotation_helpers_agree():
    q, v = _rand_q(128), torch.randn(128, 3)
    R = quat_wxyz_to_matrix(q)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(128, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(128), atol=1e-5)
    assert torch.allclose(quat_rotate_torch(q, v), (R @ v.unsqueeze(-1)).squeeze(-1), atol=1e-5)
    assert torch.allclose(quat_rotate_inverse_torch(q, v), (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1),
                          atol=1e-5)
    qn, vn = q.numpy().astype(np.float64), v.numpy().astype(np.float64)
    assert np.allclose(quat_rotate_np(qn, vn), quat_rotate_torch(q, v).numpy(), atol=1e-5)
    assert np.allclose(quat_rotate_inverse_np(qn, vn), quat_rotate_inverse_torch(q, v).numpy(), atol=1e-5)
    assert np.allclose(quat_wxyz_to_matrix_np(qn[0]), R[0].numpy(), atol=1e-6)


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_matches_omnidrones_convention():
    from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
    q, v = _rand_q(64, 1), torch.randn(64, 3)
    assert torch.allclose(quat_rotate_torch(q, v), quat_rotate(q, v), atol=1e-5)
    assert torch.allclose(quat_rotate_inverse_torch(q, v), quat_rotate_inverse(q, v), atol=1e-5)


def test_body_rate_from_world_rate():
    yaw = math.pi / 2
    q = torch.tensor([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]])
    w_B = torch.tensor([[0.7, 0.0, 0.0]])                       # roll rate about body x
    w_W = quat_rotate_torch(q, w_B)
    assert torch.allclose(w_W, torch.tensor([[0.0, 0.7, 0.0]]), atol=1e-6)   # body x = world y at yaw 90°
    state = torch.cat([torch.zeros(1, 3), q, torch.zeros(1, 3), w_W], dim=-1)
    assert torch.allclose(angular_velocity_body_torch(state), w_B, atol=1e-6)
    assert np.allclose(angular_velocity_body_np(state.numpy()), w_B.numpy(), atol=1e-6)


def test_imu_fallback_specific_force_and_body_gyro():
    yaw = 0.6
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    w_W = np.array([0.0, 0.0, 0.3])
    s = np.concatenate([[0, 0, 2], q, [0, 0, 0], w_W])
    acc, gyr = compute_imu_from_state(s, np.zeros(3), 0.005)
    assert np.allclose(acc, [0, 0, 9.81])                       # at rest: +g on body z
    assert np.allclose(gyr, quat_rotate_inverse_np(q, w_W))


# -- IMU transport correction (2026-11-24) --------------------------------------------------------------------
def test_imu_transport_correction_restores_the_specific_force():
    import torch
    from physiworld_uav.core.state_extractor import (ANGVEL, POS, QUAT, VEL, angular_velocity_body_torch,
                                                     imu_transport_correction_torch, quat_rotate_inverse_torch)
    rng = np.random.default_rng(0)
    n = 6
    q = torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    s = torch.zeros(n, 13, dtype=torch.float64)
    s[:, QUAT], s[:, VEL] = q, torch.tensor(rng.normal(0, 3, (n, 3)))
    s[:, ANGVEL] = torch.tensor(rng.normal(0, 1, (n, 3)))
    f_true = torch.tensor(rng.normal(0, 5, (n, 3)))
    w_B, v_B = angular_velocity_body_torch(s), quat_rotate_inverse_torch(s[:, QUAT], s[:, VEL])
    raw = f_true - torch.cross(w_B, v_B, dim=-1)                                  # what the Isaac sensor reports
    assert not torch.allclose(raw, f_true, atol=1e-3)
    assert torch.allclose(imu_transport_correction_torch(raw, s), f_true, atol=1e-12)
    zero = s.clone(); zero[:, VEL] = 0
    assert torch.equal(imu_transport_correction_torch(raw, zero), raw)             # at rest (v = 0) nothing changes
    assert imu_transport_correction_torch(raw[0], s[0]).shape == (3,)
