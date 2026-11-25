"""Execute one manifest row in the simulator and record it (plan §4; Run 4 physics).

Per episode: bind the domain to PhysX (hard check of the read-back values),
build controllers from the realised parameters (policy "true", Q2), reset to
the planned state with motors at the true-mass hover trim, then run.

Per physics step k: wind at the CoM (once per instant) and at each rotor hub;
project aerodynamic wrench from the enabled models (forces/system.py), injected
into OmniDrones' single base-link call (forces/injection.py) and logged per
component; control every ``control_decimation`` steps with the command held.

Guards (refuse rather than mislabel): force configs not marked ``validated``
(unless explicitly allowed for tests/probes); non-zero wind without any force
model (wind would have no physical effect).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from physiworld_uav.control.controllers import ControlLaw
from physiworld_uav.core.state_extractor import (
    ANGVEL, POS, QUAT, VEL, angular_velocity_body_torch, imu_transport_correction_torch, quat_rotate_inverse_torch, quat_rotate_torch,
    quat_wxyz_to_matrix,
)
from physiworld_uav.drones.sim_binding import check_against_plan, controller_view
from physiworld_uav.forces.system import AeroForceSystem
from physiworld_uav.data.metadata import CounterfactualInfo, EpisodeMetadata
from physiworld_uav.data.recorder import EpisodeRecorder
from physiworld_uav.drones.params import G, DroneParams
from physiworld_uav.excitation.open_loop import OpenLoopProgram
from physiworld_uav.excitation.reference_perturbation import reference_from_dict
from physiworld_uav.randomization.parameter_sampler import Domain
from physiworld_uav.wind.wind_manager import WindManager



class SimBackend(Protocol):
    device: torch.device

    def reset_episode(self, position, quat_wxyz, velocity_W, angular_velocity_W, throttle) -> None: ...
    def get_state(self) -> torch.Tensor: ...             # (1, 13) simulator layout
    def throttle(self) -> torch.Tensor: ...              # (1, R)
    def apply_rotor_command(self, cmd: torch.Tensor) -> None: ...
    def step(self) -> None: ...
    def read_imu(self): ...                              # (accel_B, gyro_B, valid) | None
    def bind_domain(self, params, dt: float) -> dict: ...   # realised parameters
    def set_external_wrench(self, force_W: torch.Tensor, torque_W: torch.Tensor) -> None: ...
    def last_external_wrench(self): ...                  # (force_W, torque_W) forwarded last step


class RunnerGuardError(NotImplementedError):
    pass


@dataclass(frozen=True)
class Termination:
    z_min: float = 0.1
    tilt_max_deg: float = 80.0
    speed_max_mps: float = 15.0
    workspace_min: tuple[float, float, float] = (-40.0, -40.0, 0.0)
    workspace_max: tuple[float, float, float] = (40.0, 40.0, 20.0)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "Termination":
        if not d:
            return cls()
        kw = dict(d)
        for k in ("workspace_min", "workspace_max"):
            if k in kw:
                kw[k] = tuple(float(x) for x in kw[k])
        return cls(**kw)

    def check(self, state: torch.Tensor) -> str | None:
        s = state.reshape(-1)
        if not torch.isfinite(s).all():
            return "terminated_divergence"
        p = s[POS]
        if float(p[2]) < self.z_min:
            return "terminated_ground"
        q = s[QUAT]
        up_z = 1.0 - 2.0 * float(q[1] ** 2 + q[2] ** 2)          # R[2,2]
        if math.degrees(math.acos(max(-1.0, min(1.0, up_z)))) > self.tilt_max_deg:
            return "terminated_envelope"
        if float(torch.linalg.norm(s[VEL])) > self.speed_max_mps:
            return "terminated_envelope"
        lo, hi = self.workspace_min, self.workspace_max
        if any(float(p[i]) < lo[i] or float(p[i]) > hi[i] for i in range(3)):
            return "terminated_envelope"
        return None


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def check_guards(plan: Mapping[str, Any], *, allow_unvalidated: bool = False) -> None:
    forces = plan["forces"]
    if forces["name"] != "none" and not forces.get("config", {}).get("validated", False) and not allow_unvalidated:
        raise RunnerGuardError(f"force config {forces['name']!r} is not validated (decision Q3 pending)")
    if plan["wind"]["stats"].get("peak_speed", 0.0) > 0.0 and forces["name"] == "none":
        raise RunnerGuardError("non-zero wind with forces=none would have no physical effect")


def run_episode(sim: SimBackend, plan: Mapping[str, Any], nominal: DroneParams, *,
                termination: Termination, config_hash: str, dataset_version: str,
                replay_commands: np.ndarray | None = None, enforce_guards: bool = True,
                allow_unvalidated_forces: bool = False,
                sim_settings: Mapping[str, Any] | None = None,
                ) -> tuple[dict[str, np.ndarray], EpisodeMetadata]:
    if enforce_guards:
        check_guards(plan, allow_unvalidated=allow_unvalidated_forces)
    dev = sim.device
    dt = float(plan["sim"]["physics_dt"])
    D = int(plan["sim"]["control_decimation"])
    n_steps = int(plan["n_steps"])
    n_ctrl = n_steps // D
    replay_truncated = False
    if plan["controller"]["mode"] == "replay":
        # A baseline that ended early (e.g. ground contact) only provides that many commands:
        # replay what exists and flag it, instead of failing the episode.
        if replay_commands is None or replay_commands.shape[0] < 1:
            raise ValueError(f"replay of {plan.get('replay_of')} has no baseline commands")
        if replay_commands.shape[0] < n_ctrl:
            n_ctrl = int(replay_commands.shape[0])
            n_steps = n_ctrl * D
            replay_truncated = True

    dom = plan["domain"]
    drone = Domain(domain_id=dom["domain_id"], region=dom["region"], factors=dom["factors"]).drone(nominal)
    if hasattr(sim, "reset_to_canonical_pose"):      # the realised parameters must not depend on where the previous episode ended
        sim.reset_to_canonical_pose()
    realised = sim.bind_domain(drone, dt)
    bad = check_against_plan(drone, realised)
    if bad:
        raise RunnerGuardError(f"PhysX parameters differ from the plan: {bad}")
    ctrl = plan["controller"]
    mode = ctrl["mode"]
    law = None
    if mode in ("position", "attitude", "rate"):
        gains = dict(ctrl.get("gains") or {})
        if mode == "attitude" and ctrl.get("attitude_hold"):
            gains["attitude_hold"] = ctrl["attitude_hold"]
        ctrl_params = controller_view(drone, realised) if ctrl["policy"] == "true" else nominal
        law = ControlLaw(mode, ctrl_params, dev, gains=gains)
    aero = None
    if plan["forces"]["name"] != "none":
        aero = AeroForceSystem.from_config(plan["forces"]["config"], drone, nominal, dom["factors"])
    components = aero.enabled if aero is not None else ()
    violations = {c: 0 for c in components}
    ref = reference_from_dict(plan["reference"]) if plan.get("reference") else None
    prog = OpenLoopProgram.from_dict(plan["open_loop"]) if plan.get("open_loop") else None
    wind = WindManager(dict(plan["wind"]["spec"]), seed=int(plan["wind"]["seed"])).configure(
        seed=int(plan["wind"]["seed"]))

    omega_max = torch.as_tensor(drone.max_rotor_speed, dtype=torch.float32, device=dev)
    kf = torch.as_tensor(drone.force_constants, dtype=torch.float32, device=dev)
    fields = [
        "time/t", "ground_truth/position_W", "ground_truth/orientation_wxyz", "ground_truth/velocity_W",
        "ground_truth/angular_velocity_B", "ground_truth/angular_velocity_W",
        "ground_truth/rotor_speed", "ground_truth/rotor_thrust",
        "ground_truth/wrench/external_force_W", "ground_truth/wrench/external_torque_B",
        "observations/imu_accel_B", "observations/imu_gyro_B", "observations/imu_valid",
        "actions/rotor_cmd", "actions/rotor_cmd_raw", "actions/saturated", "actions/rotor_speed_cmd",
        "actions/collective_thrust_cmd", "actions/body_rate_cmd",
        "environment/wind_W", "environment/wind_applied_W", "environment/wind_rotor_W",
    ]
    fields += [f"ground_truth/wrench/{c}_{q}" for c in components for q in ("force_W", "torque_B")]
    if prog is not None and prog.kind == "body_wrench":
        fields += ["ground_truth/wrench/program_force_W", "ground_truth/wrench/program_torque_B"]
    if plan["data_type"] == "closed_loop":
        fields += ["control/reference/position_W", "control/reference/velocity_W",
                   "control/reference/acceleration_W", "control/reference/yaw",
                   "control/reference/roll", "control/reference/pitch",
                   "control/tracking_error/position_W", "control/tracking_error/velocity_W"]
    rec = EpisodeRecorder(n_steps, fields, device=dev)

    # ---- reset to the planned initial state (motors at the true-mass hover trim) ----
    ic = plan["initial_conditions"]
    q0 = _quat_from_yaw(math.radians(ic["yaw_deg"]))
    w_B0 = torch.as_tensor(ic["angular_velocity_B"], dtype=torch.float32).reshape(1, 3)
    w_W0 = quat_rotate_torch(torch.as_tensor(q0, dtype=torch.float32).reshape(1, 4), w_B0).reshape(3).numpy()
    if ic.get("motor", "hover_trim") != "hover_trim":
        raise ValueError(f"unsupported initial motor state {ic['motor']!r}")
    trim_throttle = drone.hover_throttle
    trim_cmd = torch.as_tensor(drone.hover_command, dtype=torch.float32, device=dev).reshape(1, -1)
    reset_args = (np.asarray(ic["position"]), q0, np.asarray(ic["velocity_W"]), w_W0, trim_throttle)
    sim.reset_episode(*reset_args)
    # settle steps at trim so the IMU holds a fresh frame (and the solver's cached constraint impulses are the ones of the
    # hover state, not of the previous episode), then restore the exact IC. ``settle_steps`` > 1 is the order-dependence fix.
    for _ in range(max(1, int(plan["sim"].get("settle_steps", 1)))):
        sim.apply_rotor_command(trim_cmd)
        sim.step()
    imu0 = sim.read_imu()
    sim.reset_episode(*reset_args)
    ic_at_rest = not np.any(ic["velocity_W"]) and not np.any(ic["angular_velocity_B"])
    if imu0 is not None and ic_at_rest:
        # The t = 0 sample is NOT the settle-step sensor reading: that value depended on the solver state left by the previous episode in
        # the process (4.6e-3 m/s² lateral after one settle step). It is the specific force of a vehicle at rest (a = 0) at the initial
        # attitude, R(q0)ᵀ(0, 0, g), with zero gyro: deterministic and physically what the sensor reports at hover trim.
        g_vec = torch.tensor([[0.0, 0.0, G]], dtype=torch.float64)
        acc0 = quat_rotate_inverse_torch(torch.as_tensor(q0, dtype=torch.float64).reshape(1, 4), g_vec).reshape(3).numpy()
        imu0 = (acc0, np.zeros(3), True)

    hold_point = torch.as_tensor(ic["position"], dtype=torch.float32, device=dev).reshape(1, 3)
    prev_q = None
    status, reason = "completed", None
    cmd = trim_cmd
    out = None
    saturation = 0
    k_end = n_steps

    last_wind = np.zeros(3)
    last_omega = torch.zeros(1, drone.num_rotors, device=dev)

    imu_corrected = bool(plan["sim"].get("imu_transport_correction", False))

    def record_inst(k: int, state: torch.Tensor, imu, clean: bool = False) -> None:
        nonlocal prev_q, last_wind, last_omega
        q = state[:, QUAT].clone()
        if prev_q is not None and float((q * prev_q).sum()) < 0:
            q = -q
        prev_q = q
        rec.put("time/t", k, k * dt)
        rec.put("ground_truth/position_W", k, state[:, POS])
        rec.put("ground_truth/orientation_wxyz", k, q)
        rec.put("ground_truth/velocity_W", k, state[:, VEL])
        rec.put("ground_truth/angular_velocity_W", k, state[:, ANGVEL])
        rec.put("ground_truth/angular_velocity_B", k, angular_velocity_body_torch(state))
        omega = sim.throttle().to(dev) * omega_max
        last_omega = omega
        rec.put("ground_truth/rotor_speed", k, omega)
        rec.put("ground_truth/rotor_thrust", k, kf * omega ** 2)
        # evaluated exactly once per instant: stateful components (OU, events) advance once,
        # matching the planner's offline series (wind/regimes.evaluate_series)
        last_wind = wind.get_wind_state(state[0, POS].detach().cpu().numpy(), k * dt, dt).total_W
        rec.put("environment/wind_W", k, last_wind)
        if imu is None:
            rec.put("observations/imu_accel_B", k, float("nan"))
            rec.put("observations/imu_gyro_B", k, float("nan"))
            rec.put("observations/imu_valid", k, False)
        else:
            acc, gyr, valid = imu
            if imu_corrected and valid and not clean:    # remove the sensor's -ω×v transport term (see core/state_extractor)
                acc = imu_transport_correction_torch(torch.as_tensor(np.asarray(acc), dtype=torch.float64, device=state.device).reshape(1, 3),
                                                     state.to(torch.float64)).reshape(3).cpu().numpy()
            rec.put("observations/imu_accel_B", k, acc)
            rec.put("observations/imu_gyro_B", k, gyr)
            rec.put("observations/imu_valid", k, bool(valid))

    state = sim.get_state().reshape(1, -1).to(dev)
    record_inst(0, state, imu0 if ic_at_rest else None, clean=True)
    state0 = state

    for k in range(n_steps):
        t = k * dt
        if k % D == 0:
            j = k // D
            if mode == "position":
                rp = ref.at(t)
                out = law.position(state, [rp])
            elif mode == "attitude":
                rp = ref.at(t)
                out = law.attitude(state, [rp], hold_point if ctrl.get("hold_position", True) else None)
            elif mode == "rate":
                coll, rates = prog.collective_rate(t)
                out = law.rate(state, torch.tensor([coll], device=dev),
                               torch.as_tensor(rates, dtype=torch.float32, device=dev).reshape(1, 3))
            elif mode == "rotor_command":
                if prog.kind == "body_wrench":          # rotors held at trim; wrench injected below
                    out = _passthrough(trim_cmd)
                else:
                    applied, raw = prog.rotor_command(drone, t)
                    out = _passthrough(torch.as_tensor(raw, dtype=torch.float32, device=dev).reshape(1, -1))
            elif mode == "replay":
                out = _passthrough(torch.as_tensor(replay_commands[j], dtype=torch.float32,
                                                   device=dev).reshape(1, -1))
            else:
                raise ValueError(f"unknown controller mode {mode!r}")
            cmd = out.cmd
            saturation += int((out.cmd_raw.abs() > 1.0).any())
            if ref is not None:
                if mode == "attitude":
                    # attitude manoeuvres track a hold point (or nothing) in position
                    held = ctrl.get("hold_position", True)
                    rp_pos = hold_point[0].cpu().numpy() if held else np.full(3, np.nan)
                    rp_vel = np.zeros(3) if held else np.full(3, np.nan)
                    rp_acc = np.full(3, np.nan)
                else:
                    rp_pos, rp_vel, rp_acc = rp.position, rp.velocity, rp.acceleration
                ref_vals = {
                    "control/reference/position_W": rp_pos, "control/reference/velocity_W": rp_vel,
                    "control/reference/acceleration_W": rp_acc, "control/reference/yaw": rp.yaw,
                    "control/reference/roll": rp.roll if rp.roll is not None else float("nan"),
                    "control/reference/pitch": rp.pitch if rp.pitch is not None else float("nan"),
                    "control/tracking_error/position_W": state[0, POS].cpu().numpy() - rp_pos,
                    "control/tracking_error/velocity_W": state[0, VEL].cpu().numpy() - rp_vel,
                }
        # held over [t_k, t_{k+1})
        rec.put("actions/rotor_cmd", k, cmd)
        rec.put("actions/rotor_cmd_raw", k, out.cmd_raw)
        rec.put("actions/saturated", k, out.cmd_raw.abs() > 1.0)
        rec.put("actions/rotor_speed_cmd", k, torch.sqrt((cmd + 1.0) / 2.0) * omega_max)
        rec.put("actions/collective_thrust_cmd", k, out.collective_thrust)
        rec.put("actions/body_rate_cmd", k, out.body_rate_cmd)
        if ref is not None:
            for path, val in ref_vals.items():
                rec.put(path, k, val)
        rec.put("environment/wind_applied_W", k, last_wind)      # wind at (p_k, t_k), held
        R_WB = quat_wxyz_to_matrix(state[:, QUAT])
        f_W = torch.zeros(1, 3, device=dev)
        tau_W = torch.zeros(1, 3, device=dev)
        hubs_W = (state[:, POS].unsqueeze(1)
                  + torch.einsum("nij,kj->nki", R_WB, torch.as_tensor(drone.rotor_positions_B,
                                                                     dtype=torch.float32, device=dev)))
        delta = (hubs_W - state[:, POS].unsqueeze(1)).detach().cpu().numpy()
        wind_rot = torch.as_tensor(last_wind + wind.spatial_offsets(delta), dtype=torch.float32,
                                   device=dev).reshape(1, -1, 3)
        rec.put("environment/wind_rotor_W", k, wind_rot)
        if aero is not None:
            res = aero.compute(state, last_omega,
                               torch.as_tensor(last_wind, dtype=torch.float32, device=dev).reshape(1, 3), wind_rot)
            for c in components:
                rec.put(f"ground_truth/wrench/{c}_force_W", k, res.force_W[c])
                rec.put(f"ground_truth/wrench/{c}_torque_B", k, res.torque_B[c])
                violations[c] += int(res.violations[c].any())
            f_W = f_W + res.total_force_W
            tau_W = tau_W + res.total_torque_W
        if prog is not None and prog.kind == "body_wrench":
            pf_B, pt_B = prog.body_wrench(t)
            pf_B = torch.as_tensor(pf_B, dtype=torch.float32, device=dev).reshape(1, 3)
            pt_B = torch.as_tensor(pt_B, dtype=torch.float32, device=dev).reshape(1, 3)
            rec.put("ground_truth/wrench/program_force_W", k, torch.einsum("nij,nj->ni", R_WB, pf_B))
            rec.put("ground_truth/wrench/program_torque_B", k, pt_B)
            f_W = f_W + torch.einsum("nij,nj->ni", R_WB, pf_B)
            tau_W = tau_W + torch.einsum("nij,nj->ni", R_WB, pt_B)
        sim.set_external_wrench(f_W, tau_W)

        sim.apply_rotor_command(cmd)
        applied = sim.last_external_wrench()
        if applied is None:
            raise RunnerGuardError("external wrench was not forwarded to the simulator")
        rec.put("ground_truth/wrench/external_force_W", k, applied[0])
        rec.put("ground_truth/wrench/external_torque_B", k,
                torch.einsum("nji,nj->ni", R_WB, applied[1].to(dev)))
        sim.step()
        state = sim.get_state().reshape(1, -1).to(dev)
        imu_k1 = sim.read_imu()
        if k == 0 and imu_k1 is not None:
            # The sensor differences the velocity against the one it stored at the settle step, which carries solver state from the previous
            # episode (≈ 1e-9 … 4.6e-3 m/s²). Row 1 is therefore derived from the ground truth of this episode instead: R(q₁)ᵀ((v₁ − v₀)/Δt + g ẑ).
            dv = (state[:, VEL].to(torch.float64) - state0[:, VEL].to(torch.float64)) / dt
            acc1 = quat_rotate_inverse_torch(state[:, QUAT].to(torch.float64), dv + torch.tensor([[0.0, 0.0, G]], dtype=torch.float64, device=dv.device))
            imu_k1 = (acc1.reshape(3).cpu().numpy(), angular_velocity_body_torch(state).reshape(3).to(torch.float64).cpu().numpy(), True)
            record_inst(k + 1, state, imu_k1, clean=True)
        else:
            record_inst(k + 1, state, imu_k1)
        reason = termination.check(state)
        if reason is not None:
            status = reason
            k_end = ((k + 1) // D) * D
            break

    if k_end < n_steps:
        rec.truncate(max(k_end, D))
    arrays = rec.to_numpy()
    _add_derived(arrays, dt)

    final = {"position_W": arrays["ground_truth/position_W"][-1].tolist(),
             "velocity_W": arrays["ground_truth/velocity_W"][-1].tolist(),
             "orientation_wxyz": arrays["ground_truth/orientation_wxyz"][-1].tolist(),
             "angular_velocity_B": arrays["ground_truth/angular_velocity_B"][-1].tolist()}
    cf = plan.get("counterfactual")
    meta = EpisodeMetadata(
        dataset_id=plan["dataset_id"], dataset_version=dataset_version,
        experiment_id=plan["experiment_id"], episode_id=plan["episode_id"],
        scenario_id=plan["scenario_id"], domain_id=dom["domain_id"], group_id=plan["group_id"],
        split_role=plan["split_role"], data_type=plan["data_type"], ood_category=plan["ood_category"],
        experiment_group=plan.get("experiment_group"), mode=plan.get("mode"),
        scenario_template=plan.get("scenario_template"),
        counterfactual=None if not cf else CounterfactualInfo(
            group_id=cf["group_id"], mode=cf["mode"], variable=cf["variable"],
            sibling_index=cf["sibling_index"], is_baseline=cf["is_baseline"], diff=cf.get("diff", {})),
        plan=dict(plan), drone=drone.summary(), drone_realised=realised,
        controller={**ctrl, "built_from": "realised_domain" if ctrl["policy"] == "true" else "nominal"},
        forces={"name": plan["forces"]["name"], "applied": aero is not None,
                **({"system": aero.summary()} if aero is not None else {})},
        sim={**plan["sim"], **dict(sim_settings or {})},
        seeds=dict(plan["seeds"]),
        initial_conditions=dict(ic), terminal_conditions=final,
        status=status, termination_reason=reason,
        n_steps=int(arrays["time/t"].shape[0] - 1),
        duration_s=float(arrays["time/t"][-1]),
        stats={"saturated_control_fraction": saturation / max(1, n_ctrl),
               "applicability_violation_frac": {c: violations[c] / max(1, k_end) for c in components},
               "applicability_ok": all(violations[c] / max(1, k_end)
                                       <= float(plan["forces"].get("config", {}).get("max_violation_fraction", 0.05))
                                       for c in components),
               "imu_valid_fraction": float(arrays["observations/imu_valid"].mean()),
               "imu_k0_from_settle_step": bool(ic_at_rest),
               "imu_transport_corrected": imu_corrected,
               "replay_truncated_to_baseline": replay_truncated},
        config_hash=config_hash,
        generated_utc=datetime.now(timezone.utc).isoformat(),
    )
    return arrays, meta


def _passthrough(raw: torch.Tensor):
    from physiworld_uav.control.controllers import ControlOutput
    n = raw.shape[0]
    return ControlOutput(cmd=raw.clamp(-1.0, 1.0), cmd_raw=raw,
                         collective_thrust=torch.full((n,), float("nan"), device=raw.device),
                         body_rate_cmd=torch.full((n, 3), float("nan"), device=raw.device))


def _add_derived(arrays: dict[str, np.ndarray], dt: float) -> None:
    """Derived accelerations (flagged DERIVED in the schema): central differences."""
    v = arrays["ground_truth/velocity_W"].astype(np.float64)
    w = arrays["ground_truth/angular_velocity_B"].astype(np.float64)
    if v.shape[0] >= 3:
        arrays["ground_truth/linear_acceleration_W"] = np.gradient(v, dt, axis=0).astype(np.float32)
        arrays["ground_truth/angular_acceleration_B"] = np.gradient(w, dt, axis=0).astype(np.float32)
