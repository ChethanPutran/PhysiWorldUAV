"""Analytic tests of the aerodynamic models, the injector and binding checks (plan §6.1, §5)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from physiworld_uav.config import load_template
from physiworld_uav.core.state_extractor import quat_wxyz_to_matrix
from physiworld_uav.drones.sim_binding import check_against_plan, controller_view
from physiworld_uav.forces import AeroForceSystem, WrenchInjector
from physiworld_uav.forces import models as M

RHO = 1.225


def _q(yaw=0.0, roll=0.0):
    cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
    return torch.tensor([[cy * cr, cy * sr, sy * sr, sy * cr]])      # yaw ∘ roll


def _state(v=(0, 0, 0), q=None, w_W=(0, 0, 0), z=2.0):
    q = _q() if q is None else q
    return torch.cat([torch.tensor([[0.0, 0.0, z]]), q, torch.tensor([v], dtype=torch.float32),
                      torch.tensor([w_W], dtype=torch.float32)], dim=1)


def test_body_drag_antiparallel_quadratic_and_equivariant():
    p = M.BodyDragParams(cd=1.1, area_m2=(0.007, 0.0076, 0.018))
    I = torch.eye(3).unsqueeze(0)
    for v in ([3.0, 0, 0], [0, 2.0, 0], [1.0, -2.0, 0.5]):
        F, tau = M.body_drag(p, I, torch.tensor([v]), torch.zeros(1, 3))
        vv = torch.tensor(v)
        assert torch.allclose(torch.linalg.cross(F[0], vv), torch.zeros(3), atol=1e-7)   # anti-parallel
        assert float(F[0] @ vv) < 0 and torch.allclose(tau, torch.zeros(1, 3))
        F2, _ = M.body_drag(p, I, 2 * torch.tensor([v]), torch.zeros(1, 3))
        assert torch.allclose(F2, 4 * F, rtol=1e-5)                                      # ∝ v²
    F, _ = M.body_drag(p, I, torch.tensor([[4.0, 0, 0]]), torch.zeros(1, 3))
    assert F[0, 0].item() == pytest.approx(-0.5 * RHO * 1.1 * 0.007 * 16, rel=1e-6)
    # wind-only: stationary vehicle in +y wind is pushed along +y
    F, _ = M.body_drag(p, I, torch.zeros(1, 3), torch.tensor([[0.0, 3.0, 0.0]]))
    assert F[0, 1] > 0
    # equivariance: rotating body and flow together rotates the body-frame result identically
    R = quat_wxyz_to_matrix(_q(yaw=0.7, roll=0.3))
    v_W = torch.tensor([[1.0, 2.0, -0.3]])
    F_B_rot, _ = M.body_drag(p, R, v_W, torch.zeros(1, 3))
    F_B_ref, _ = M.body_drag(p, torch.eye(3).unsqueeze(0), (R.transpose(1, 2) @ v_W.unsqueeze(-1)).squeeze(-1),
                             torch.zeros(1, 3))
    assert torch.allclose(F_B_rot, F_B_ref, atol=1e-6)


def test_body_drag_per_axis_cd():
    p = M.BodyDragParams(cd=(1.1, 1.2, 1.3), area_m2=(0.007, 0.0076, 0.018))
    I = torch.eye(3).unsqueeze(0)
    for i, (c, a) in enumerate(zip((1.1, 1.2, 1.3), (0.007, 0.0076, 0.018))):
        v = torch.zeros(1, 3); v[0, i] = 4.0
        F, _ = M.body_drag(p, I, v, torch.zeros(1, 3))
        assert F[0, i].item() == pytest.approx(-0.5 * RHO * c * a * 16, rel=1e-6)
    v = torch.tensor([[1.0, -2.0, 0.5]])
    F, _ = M.body_drag(p, I, v, torch.zeros(1, 3))
    assert torch.allclose(torch.linalg.cross(F[0], v[0]), torch.zeros(3), atol=1e-7)     # still anti-parallel
    # scalar cd equals the equivalent per-axis tuple
    Fs, _ = M.body_drag(M.BodyDragParams(cd=1.1, area_m2=p.area_m2), I, v, torch.zeros(1, 3))
    Ft, _ = M.body_drag(M.BodyDragParams(cd=(1.1,) * 3, area_m2=p.area_m2), I, v, torch.zeros(1, 3))
    assert torch.allclose(Fs, Ft)


def test_body_drag_terminal_velocity():
    """Falling body: integrate m dv/dt = −m g − F_drag ⇒ v_t = sqrt(2 m g / (ρ C_D A_z))."""
    p = M.BodyDragParams(cd=1.1, area_m2=(0.007, 0.0076, 0.018))
    m, v = 0.716, torch.zeros(1, 3)
    for _ in range(40000):
        F, _ = M.body_drag(p, torch.eye(3).unsqueeze(0), v, torch.zeros(1, 3))
        v = v + 0.001 * (torch.tensor([[0, 0, -9.81]]) + F / m)
    vt = math.sqrt(2 * m * 9.81 / (RHO * 1.1 * 0.018))
    assert -v[0, 2].item() == pytest.approx(vt, rel=1e-3)


def test_rotor_drag_linear_inplane_and_omega_proportional(drone):
    r_B = torch.tensor(drone.rotor_positions_B, dtype=torch.float32)
    p = M.RotorDragParams(k_rd=1e-4)
    om = torch.full((1, 4), 450.0)
    I = torch.eye(3).unsqueeze(0)
    w0 = torch.zeros(1, 4, 3)
    F1, t1 = M.rotor_drag(p, I, torch.tensor([[2.0, 0, 0]]), torch.zeros(1, 3), om, w0, r_B)
    F2, _ = M.rotor_drag(p, I, torch.tensor([[4.0, 0, 0]]), torch.zeros(1, 3), om, w0, r_B)
    F3, _ = M.rotor_drag(p, I, torch.tensor([[2.0, 0, 0]]), torch.zeros(1, 3), 2 * om, w0, r_B)
    Fz, _ = M.rotor_drag(p, I, torch.tensor([[0, 0, 3.0]]), torch.zeros(1, 3), om, w0, r_B)
    assert F1[0, 0].item() == pytest.approx(-1e-4 * 4 * 450 * 2.0, rel=1e-6)
    assert torch.allclose(F2, 2 * F1) and torch.allclose(F3, 2 * F1)            # linear in v and Ω
    assert torch.allclose(Fz, torch.zeros(1, 3), atol=1e-9)                      # no axial component
    # uniform flow, level: no roll torque; pitch torque only from the 1 cm hub height (τ_y = r_z ΣF_x)
    assert abs(t1[0, 0].item()) < 1e-9
    assert t1[0, 1].item() == pytest.approx(drone.rotor_height * F1[0, 0].item(), rel=1e-5)
    # yaw rotation → damping torque about z (ω × r gives tangential hub velocity)
    _, tz = M.rotor_drag(p, I, torch.zeros(1, 3), torch.tensor([[0, 0, 2.0]]), om, w0, r_B)
    assert tz[0, 2].item() < 0


def test_rotor_drag_matches_faessler_d_at_hover(drone):
    sysm = AeroForceSystem.from_config({"rotor_drag": {"enabled": True, "d_hover_per_s": 0.30}}, drone, drone)
    s = _state(v=(1.5, 0, 0))
    om = torch.as_tensor(drone.hover_rotor_speed, dtype=torch.float32).reshape(1, 4)
    res = sysm.compute(s, om, torch.zeros(1, 3), torch.zeros(1, 4, 3))
    assert res.total_force_W[0, 0].item() / drone.mass == pytest.approx(-0.30 * 1.5, rel=1e-5)


def test_ground_effect_limits(drone):
    p = M.GroundEffectParams(rotor_radius=0.1)
    r_B = torch.tensor(drone.rotor_positions_B, dtype=torch.float32)
    kf = torch.tensor(drone.force_constants, dtype=torch.float32)
    om = torch.full((1, 4), 450.0)
    T = (kf * om ** 2).sum().item()
    F_hi, _ = M.ground_effect(p, torch.eye(3).unsqueeze(0), torch.tensor([[0, 0, 50.0]]), om, kf, r_B)
    assert F_hi[0, 2].item() < 1e-5 * T
    h = 0.2 + 0.01                                                    # hub height
    F, _ = M.ground_effect(p, torch.eye(3).unsqueeze(0), torch.tensor([[0, 0, 0.2]]), om, kf, r_B)
    assert F[0, 2].item() == pytest.approx((1 / (1 - (0.1 / (4 * h)) ** 2) - 1) * T, rel=1e-5)


def test_aero_torque_opposes_rotation_frame_invariant():
    p = M.AeroTorqueParams(k1=(0.002, 0.002, 0.002), k2=0.001)
    w = torch.tensor([[0.3, -1.2, 2.0]])
    _, tau = M.aero_torque(p, w)
    assert float((tau * w).sum()) < 0
    assert torch.allclose(torch.linalg.cross(tau[0], w[0]), torch.zeros(3), atol=1e-7)


def test_system_world_frame_and_applicability(drone):
    cfg = load_template("forces", "truth_default")
    sysm = AeroForceSystem.from_config(cfg, drone, drone, {"rotor_drag_scale": 1.1, "body_drag_scale": 0.9})
    assert sysm.rotor_drag.scale == 1.1 and sysm.body_drag.scale == 0.9
    q = _q(yaw=math.pi / 2)
    s = _state(v=(2.0, 0, 0), q=q)
    om = torch.as_tensor(drone.hover_rotor_speed, dtype=torch.float32).reshape(1, 4)
    res = sysm.compute(s, om, torch.zeros(1, 3), torch.zeros(1, 4, 3))
    assert res.total_force_W[0, 0] < 0 and abs(res.total_force_W[0, 1]) < 1e-6         # opposes world v
    fast = sysm.compute(_state(v=(20.0, 0, 0)), om, torch.zeros(1, 3), torch.zeros(1, 4, 3))
    assert bool(fast.violations["rotor_drag"]) and bool(fast.violations["body_drag"])
    assert not bool(res.violations["body_drag"])


def test_injector_adds_into_single_call():
    calls = []

    class View:
        count = 1

        def apply_forces_and_torques_at_pos(self, forces, torques, is_global=True, indices=None):
            calls.append((forces.clone(), torques.clone()))

        def get_masses(self):
            return torch.tensor([0.68])

    inj = WrenchInjector(View(), 1)
    inj.set(torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[0.0, 0.0, 0.5]]))
    inj.apply_forces_and_torques_at_pos(torch.tensor([[0.1, 0, 0]]), torch.tensor([[0, 0, 0.2]]), is_global=True)
    assert torch.allclose(calls[0][0], torch.tensor([[1.1, 2.0, 3.0]]))
    assert torch.allclose(calls[0][1], torch.tensor([[0, 0, 0.7]]))
    inj.apply_forces_and_torques_at_pos(torch.zeros(1, 3), torch.zeros(1, 3), is_global=True)
    assert torch.allclose(calls[1][0], torch.zeros(1, 3))                       # cleared after use
    assert torch.allclose(inj.get_masses(), torch.tensor([0.68]))               # delegation
    with pytest.raises(Exception):
        inj.apply_forces_and_torques_at_pos(torch.zeros(1, 3), torch.zeros(1, 3), is_global=False)
    with pytest.raises(Exception):
        inj.set(torch.tensor([[float("nan"), 0, 0]]), torch.zeros(1, 3))


def test_binding_checks(drone):
    real = {"total_mass_kg": drone.mass, "base_inertia_diag_kgm2": list(drone.inertia_diag),
            "composite_inertia_effective_kgm2": np.diag([0.0077, 0.0077, 0.01304]).tolist(),
            "force_constants": list(drone.force_constants), "moment_constants": list(drone.moment_constants),
            "motor_time_constant_s": drone.effective_motor_time_constant, "drag_coef_omnidrones": 0.0}
    assert check_against_plan(drone, real) == []
    assert check_against_plan(drone, {**real, "total_mass_kg": drone.mass * 1.01})
    cv = controller_view(drone, real)
    assert cv.inertia_diag == pytest.approx((0.0077, 0.0077, 0.01304)) and cv.mass == drone.mass
