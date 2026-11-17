"""Isolated physics experiments inside Isaac Sim (Run 4 validation, plan §7 TS5).

Run by tests/isaac/test_physics_validation.py in a subprocess (Isaac's
SimulationApp.close() ends the interpreter). Writes a JSON result file.

usage: python tests/isaac/physics_probe.py <out.json>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiworld_uav.config import load_template  # noqa: E402
from physiworld_uav.core.simulation_interface import SimulationConfig, SimulationInterface  # noqa: E402
from physiworld_uav.core.state_extractor import angular_velocity_body_torch  # noqa: E402
from physiworld_uav.drones import load_drone  # noqa: E402
from physiworld_uav.drones.sim_binding import check_against_plan  # noqa: E402
from physiworld_uav.excitation.open_loop import allocation_matrix  # noqa: E402
from physiworld_uav.forces import AeroForceSystem  # noqa: E402

DT = 0.005
out_path = Path(sys.argv[1])
R: dict = {}
nominal = load_drone(ROOT / "config/drone/hummingbird.yaml")
sim = SimulationInterface(SimulationConfig(dt=DT, device="cuda:0", render=False, headless=True,
                                           init_position=[0, 0, 3.0], hydra_config={"headless": True}))
sim.ensure_ready()
dev = sim.device


def state():
    return sim.get_state().reshape(1, -1).to(dev)


def cmd_of(drone, thrusts):
    return torch.as_tensor(2 * np.asarray(thrusts) / drone.max_thrust_per_rotor - 1, dtype=torch.float32,
                           device=dev).reshape(1, -1)


def reset(drone, pos=(0, 0, 3.0), yaw=0.0, v=(0, 0, 0), w_W=(0, 0, 0)):
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    sim.reset_episode(np.asarray(pos, float), q, np.asarray(v, float), np.asarray(w_W, float), drone.hover_throttle)


def run(drone, cmd, n, ext=None):
    s0 = state()
    imu = []
    for _ in range(n):
        if ext is not None:
            sim.set_external_wrench(*ext)
        sim.apply_rotor_command(cmd)
        sim.step()
        r = sim.read_imu()
        if r is not None:
            imu.append(np.concatenate([r[0], r[1]]))
    return s0, state(), np.array(imu)


try:
    real = sim.bind_domain(nominal, DT)
    R["settings"] = sim.physics_settings()
    R["realised_nominal"] = real
    J_eff = real["composite_inertia_effective_kgm2"]
    hover = torch.as_tensor(nominal.hover_command, dtype=torch.float32, device=dev).reshape(1, -1)

    # 1 hover equilibrium (true-mass trim)
    reset(nominal)
    s0, s1, imu = run(nominal, hover, 200)
    R["hover"] = {"a_W": ((s1[0, 7:10] - s0[0, 7:10]) / (200 * DT)).tolist(),
                  "imu_accel_mean": imu[-50:, :3].mean(0).tolist()}

    # 2 free fall: rotor command −1 (throttle decays with the motor lag)
    reset(nominal)
    run(nominal, torch.full((1, 4), -1.0, device=dev), 40)
    s0, s1, imu = run(nominal, torch.full((1, 4), -1.0, device=dev), 60)
    R["free_fall"] = {"a_z": float((s1[0, 9] - s0[0, 9]) / (60 * DT)), "speed_end": float(-s1[0, 9]),
                      "imu_accel_mean": imu.mean(0)[:3].tolist()}

    # 3 horizontal coast at hover trim, no project forces, damping 0
    reset(nominal, v=(2.0, 0, 0))
    s0, s1, _ = run(nominal, hover, 400)
    R["coast"] = {"v0": float(s0[0, 7]), "v_end": float(s1[0, 7])}

    # 4 injector: 1 N along +y
    reset(nominal)
    s0, s1, imu = run(nominal, hover, 20, ext=(torch.tensor([[0, 1.0, 0]]), torch.zeros(1, 3)))
    R["inject_1N"] = {"a_y": float((s1[0, 8] - s0[0, 8]) / (20 * DT)), "imu_y": float(imu[-5:, 1].mean()),
                      "expected": 1 / nominal.mass}

    # 5 motor step response: throttle(k) − target = (1 − α)^k (throttle(0) − target)
    reset(nominal)
    T_hover = nominal.weight / 4
    step_cmd = cmd_of(nominal, [T_hover * 1.3] * 4)
    target = math.sqrt((float(step_cmd[0, 0]) + 1) / 2)
    th0 = float(sim.throttle()[0, 0])
    ths = []
    for _ in range(10):
        sim.apply_rotor_command(step_cmd)
        ths.append(float(sim.throttle()[0, 0]))
        sim.step()
    alpha = real["motor_alpha"][0]
    pred = [target + (th0 - target) * (1 - alpha) ** (k + 1) for k in range(10)]
    R["motor_step"] = {"max_abs_err": float(np.max(np.abs(np.array(ths) - pred))), "alpha": alpha}

    # 6/7 yaw and roll torque through the exact allocation (open loop)
    A = allocation_matrix(nominal)
    for name, wrench, axis in (("yaw_torque", [0, 0, 0.01, nominal.weight], 2),
                               ("roll_torque", [0.01, 0, 0, nominal.weight], 0),
                               ("pitch_torque", [0, 0.01, 0, nominal.weight], 1)):
        I_eff = J_eff[axis][axis]
        reset(nominal)
        run(nominal, hover, 1)                           # settle motors at trim
        reset(nominal)
        Tm = np.linalg.solve(A, np.asarray(wrench, float))
        c = cmd_of(nominal, Tm)
        # wait for the motor lag to converge, then measure angular acceleration over 20 steps
        run(nominal, c, 15)
        sa = state()
        w0 = angular_velocity_body_torch(sa)[0, axis].item()
        _, sb, _ = run(nominal, c, 20)
        w1 = angular_velocity_body_torch(sb)[0, axis].item()
        R[name] = {"ang_acc": (w1 - w0) / (20 * DT), "expected": 0.01 / I_eff, "I_eff": I_eff}

    # 8 gyro vs ω_B (yawed 90°, initial body roll rate)
    q_yaw = torch.tensor([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
    w_W = (0.0, 0.5, 0.0)                                # body x = world y at yaw 90°
    reset(nominal, yaw=math.pi / 2, w_W=w_W)
    _, s1, imu = run(nominal, hover, 3)
    R["gyro_vs_omegaB"] = {"omega_B": angular_velocity_body_torch(s1)[0].tolist(), "gyro": imu[-1, 3:].tolist()}

    # 9 articulation coupling (A7, measurement only): roll acceleration while pitching at 1 rad/s,
    #   symmetric vs yaw-differential rotor thrusts (rotor joints do not spin in this setup)
    out = {}
    for label, wrench in (("symmetric", [0, 0, 0, nominal.weight]), ("yaw_differential", [0, 0, 0.02, nominal.weight])):
        reset(nominal)
        Tm = np.linalg.solve(A, np.asarray(wrench, float))
        c = cmd_of(nominal, Tm)
        run(nominal, c, 15)
        sa = state()
        sim.reset_episode(sa[0, 0:3].cpu().numpy(), np.array([1.0, 0, 0, 0]), np.zeros(3),
                          np.array([0.0, 1.0, 0.0]), sim.throttle().cpu().numpy().reshape(-1))
        s0, s1, _ = run(nominal, c, 10)
        wb0, wb1 = angular_velocity_body_torch(s0)[0], angular_velocity_body_torch(s1)[0]
        out[label] = {"roll_acc": float((wb1[0] - wb0[0]) / (10 * DT)),
                      "rotor_speed_spread": float(np.ptp(sim.throttle().cpu().numpy()) * nominal.max_rotor_speed[0])}
    R["articulation_coupling"] = out

    # 10 domain binding: heavier, weaker rotors, slower motors → hover at the new true trim
    heavy = nominal.scaled(mass=nominal.mass * 1.1, inertia_diag=tuple(x * 1.1 for x in nominal.inertia_diag),
                           force_scale=0.95, moment_scale=0.95,
                           motor_time_constant=nominal.effective_motor_time_constant * 1.2)
    real_h = sim.bind_domain(heavy, DT)
    reset(heavy)
    s0, s1, imu = run(heavy, torch.as_tensor(heavy.hover_command, dtype=torch.float32, device=dev).reshape(1, -1), 200)
    R["domain_binding"] = {"check_issues": check_against_plan(heavy, real_h), "realised": real_h,
                           "hover_a_W": ((s1[0, 7:10] - s0[0, 7:10]) / (200 * DT)).tolist()}
    real = sim.bind_domain(nominal, DT)

    # 11 wind-induced acceleration vs model (truth_default coefficients, unvalidated allowed here)
    cfg = load_template("forces", "truth_default")
    aero = AeroForceSystem.from_config(cfg, nominal, nominal)
    wind = torch.tensor([[3.0, 0.0, 0.0]])
    reset(nominal)
    s = state()
    om = sim.throttle().to(dev) * torch.as_tensor(nominal.max_rotor_speed, dtype=torch.float32, device=dev)
    res = aero.compute(s, om, wind, wind.reshape(1, 1, 3).expand(1, 4, 3))
    s0, s1, imu = run(nominal, hover, 1, ext=(res.total_force_W, res.total_torque_W))
    R["wind_accel"] = {"a_x": float((s1[0, 7] - s0[0, 7]) / DT), "expected": float(res.total_force_W[0, 0] / nominal.mass),
                       "components_N": {k: v[0].tolist() for k, v in res.force_W.items()}}

    # 12 determinism: identical closed-loop-free segment twice
    traces = []
    for _ in range(2):
        reset(nominal)
        c = cmd_of(nominal, np.linalg.solve(A, np.array([0.005, -0.004, 0.002, nominal.weight * 1.02])))
        tr = []
        for _ in range(200):
            sim.apply_rotor_command(c)
            sim.step()
            tr.append(state()[0].cpu().numpy())
        traces.append(np.array(tr))
    R["determinism_max_abs_diff"] = float(np.abs(traces[0] - traces[1]).max())
except Exception as exc:  # noqa: BLE001
    import traceback
    R["error"] = traceback.format_exc()
out_path.write_text(json.dumps(R, indent=1, default=float))
print("PROBE_DONE", flush=True)
sim.close()
