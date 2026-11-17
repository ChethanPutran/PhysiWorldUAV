"""Bind a domain's physical parameters to the running OmniDrones/PhysX vehicle (plan §6.4-R1, R4; §5-A4/A5/A9).

Uses the same setters as OmniDrones' own ``_randomize`` (no OmniDrones edits):

* base-link mass  = total mass − Σ rotor-link masses (rotor links kept as in the USD)
* base-link inertia = ``DroneParams.inertia_diag`` (OmniDrones semantics: base-link inertia)
* rotor thrust/moment constants: ``KF = k_F Ω_max²``, ``KM = k_M Ω_max²`` (OmniDrones parameters)
* motor lag: per-step blend α = 1 − exp(−dt/τ_m) written to ``tau_up``/``tau_down``
* OmniDrones' own drag coefficient is asserted to be 0 (its sign is +, audit A5)

``realised()`` reads everything back from PhysX/OmniDrones and computes the
composite inertia about the CoM by the parallel-axis theorem. Two composites
are reported: ``rigid`` (rotors locked) and ``effective`` (each rotor's own
inertia replaced by its axisymmetric equivalent about x/y and excluded about z,
because the rotor joints are free about the rotor axis). Run 4 measurements
(torque probes in Isaac): I_xx 0.00771, I_yy 0.00768, I_zz 0.01304 vs effective
0.00768 / 0.00768 / 0.01304. Note: the rotor joints do not spin in this setup
(joint velocities 0), so there is no rotor gyroscopic coupling to model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .params import DroneParams


class BindingError(RuntimeError):
    pass


def _rotor_link_masses(drone: Any) -> np.ndarray:
    m = drone._view.get_body_masses().reshape(-1).detach().cpu().numpy()
    names = list(drone._view.body_names)
    return np.array([m[names.index(f"rotor_{i}")] for i in range(sum(n.startswith("rotor_") for n in names))])


def bind(drone: Any, params: DroneParams, dt: float) -> None:
    """Write ``params`` into the simulated vehicle (single environment)."""
    dev = drone.device
    rotor_m = _rotor_link_masses(drone)
    m_base = params.mass - float(rotor_m.sum())
    if m_base <= 0:
        raise BindingError(f"total mass {params.mass} ≤ rotor-link masses {rotor_m.sum()}")
    drone.base_link.set_masses(torch.tensor([m_base], dtype=torch.float32, device=dev))
    Ixx, Iyy, Izz = params.inertia_diag
    Ixy, Ixz, Iyz = params.inertia_offdiag
    J = torch.tensor([[Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz]], dtype=torch.float32, device=dev)
    drone.base_link.set_inertias(J)
    drone.masses[:] = m_base
    drone.gravity[:] = drone.masses * 9.81

    om2 = torch.as_tensor(params.max_rotor_speed, dtype=torch.float32, device=dev) ** 2
    drone.KF[:] = torch.as_tensor(params.force_constants, dtype=torch.float32, device=dev) * om2
    drone.KM[:] = torch.as_tensor(params.moment_constants, dtype=torch.float32, device=dev) * om2
    alpha = params.motor_alpha(dt)
    drone.rotor_params["tau_up"][:] = alpha
    drone.rotor_params["tau_down"][:] = alpha
    if float(drone.drag_coef.abs().max()) != 0.0:
        raise BindingError("OmniDrones drag_coef must stay 0 (project drag lives in forces/)")


def realised(drone: Any, dt: float) -> dict[str, Any]:
    """Physical parameters actually in the simulator (recorded as ``drone_realised``)."""
    v = drone._view
    names = list(v.body_names)
    m = v.get_body_masses().reshape(-1).detach().cpu().numpy().astype(np.float64)
    I_local = v.get_body_inertias().reshape(-1, 3, 3).detach().cpu().numpy().astype(np.float64)
    # link positions in the body frame
    pos_base, rot_base = drone.base_link.get_world_poses()
    from physiworld_uav.core.state_extractor import quat_rotate_inverse_np
    qb = rot_base.reshape(-1).detach().cpu().numpy()
    pb = pos_base.reshape(-1).detach().cpu().numpy()
    rp, _ = drone.rotors_view.get_world_poses()
    r = np.zeros((len(names), 3))
    rotor_pos = rp.reshape(-1, 3).detach().cpu().numpy()
    for i, n in enumerate(names):
        if n.startswith("rotor_"):
            r[i] = quat_rotate_inverse_np(qb, rotor_pos[int(n.split("_")[1])] - pb)
    M = m.sum()
    c = (m[:, None] * r).sum(axis=0) / M

    def composite(effective: bool) -> np.ndarray:
        J = np.zeros((3, 3))
        for i, n in enumerate(names):
            Ii = I_local[i].copy()                    # rotor link frames are aligned with the body (verified)
            if effective and n.startswith("rotor_"):
                # rotor joints are free about the rotor axis (body z): the rotor's own z-inertia
                # does not couple to yaw; about x/y the rotor acts as its axisymmetric equivalent
                # (I_xx + I_yy)/2. Validated in Isaac: roll/pitch/yaw within 0.3 % (Run 4).
                avg = 0.5 * (Ii[0, 0] + Ii[1, 1])
                Ii = np.diag([avg, avg, 0.0])
            d = r[i] - c
            J += Ii + m[i] * (d @ d * np.eye(3) - np.outer(d, d))
        return J

    J_eff, J_rig = composite(True), composite(False)
    KF = drone.KF.reshape(-1).detach().cpu().numpy()
    KM = drone.KM.reshape(-1).detach().cpu().numpy()
    alpha = drone.rotor_params["tau_up"].reshape(-1).detach().cpu().numpy()
    om_max = drone.MAX_ROT_VEL.reshape(-1).detach().cpu().numpy() if hasattr(drone, "MAX_ROT_VEL") else None
    out = {
        "total_mass_kg": float(M),
        "link_masses_kg": dict(zip(names, m.round(9).tolist())),
        "base_inertia_diag_kgm2": np.diag(I_local[names.index("base_link")]).tolist(),
        "composite_inertia_effective_kgm2": J_eff.round(9).tolist(),
        "composite_inertia_rigid_kgm2": J_rig.round(9).tolist(),
        "com_offset_B_m": c.tolist(),
        "KF_max_thrust_N": KF.tolist(),
        "KM_max_moment_Nm": KM.tolist(),
        "motor_alpha": alpha.tolist(),
        "motor_time_constant_s": float(-dt / np.log(1.0 - float(alpha.mean()))),
        "drag_coef_omnidrones": float(drone.drag_coef.abs().max()),
    }
    if om_max is not None:
        out["force_constants"] = (KF / om_max ** 2).tolist()
        out["moment_constants"] = (KM / om_max ** 2).tolist()
    return out


def controller_view(params: DroneParams, real: dict[str, Any]) -> DroneParams:
    """Parameters a 'true'-policy controller should use: realised total mass and the
    effective composite inertia (principal diagonal)."""
    J = np.asarray(real["composite_inertia_effective_kgm2"])
    return params.scaled(mass=real["total_mass_kg"], inertia_diag=tuple(np.diag(J)))


def check_against_plan(params: DroneParams, real: dict[str, Any], rtol: float = 1e-4) -> list[str]:
    """Hard check: what PhysX simulates equals what the planner sampled (plan R4)."""
    issues = []
    if not np.isclose(real["total_mass_kg"], params.mass, rtol=rtol):
        issues.append(f"mass {real['total_mass_kg']} != planned {params.mass}")
    if not np.allclose(real["base_inertia_diag_kgm2"], params.inertia_diag, rtol=rtol):
        issues.append(f"base inertia {real['base_inertia_diag_kgm2']} != planned {params.inertia_diag}")
    if "force_constants" in real and not np.allclose(real["force_constants"], params.force_constants, rtol=rtol):
        issues.append("force constants differ from plan")
    if "moment_constants" in real and not np.allclose(real["moment_constants"], params.moment_constants, rtol=rtol):
        issues.append("moment constants differ from plan")
    if not np.isclose(real["motor_time_constant_s"], params.effective_motor_time_constant, rtol=1e-3):
        issues.append(f"motor τ {real['motor_time_constant_s']} != planned {params.effective_motor_time_constant}")
    if real["drag_coef_omnidrones"] != 0.0:
        issues.append("OmniDrones drag_coef non-zero")
    return issues
