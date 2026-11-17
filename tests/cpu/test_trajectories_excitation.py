"""Reference library, feasibility, excitation signals and open-loop programs (plan T4–T13)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from physiworld_uav.config import load_template
from physiworld_uav.excitation import signals as S
from physiworld_uav.excitation.open_loop import allocation_matrix, make_program
from physiworld_uav.excitation.reference_perturbation import perturb, reference_from_dict
from physiworld_uav.trajectories.feasibility import FeasibilityLimits, check
from physiworld_uav.trajectories.gp_kernel import make_composite
from physiworld_uav.trajectories.gp_sampler import joint_covariance, sample_joint_gp
from physiworld_uav.trajectories.library import Anchor, Reference, make_reference

from conftest import omni_drones_available

ANCHOR = Anchor((0.3, -0.2, 2.0), math.radians(40))
POSITION_SCENARIOS = ["hover", "vertical", "translation", "line", "accel_decel", "circle", "helix",
                      "figure_eight", "sinusoidal", "waypoints", "combined", "yaw_step", "yaw_sine"]


def _ref(name, seed=0, duration=12.0):
    sc = load_template("scenario", name)
    return make_reference(sc["family"], sc.get("params", {}), sc.get("yaw", {}), ANCHOR, duration,
                          np.random.default_rng(seed))


@pytest.mark.parametrize("name", POSITION_SCENARIOS)
def test_position_references_start_at_rest_on_anchor(name):
    r = _ref(name)
    p0 = r.at(0.0)
    assert np.allclose(p0.position, ANCHOR.p, atol=1e-9)
    assert np.allclose(p0.velocity, 0.0, atol=1e-3)
    assert np.allclose(p0.acceleration, 0.0, atol=2e-2)


@pytest.mark.parametrize("name", POSITION_SCENARIOS + ["roll_doublet", "pitch_doublet", "attitude_combined"])
def test_scenario_templates_feasible_and_roundtrip(name, drone):
    for seed in range(3):
        r = _ref(name, seed)
        rep = check(r, drone, FeasibilityLimits())
        assert rep.feasible, rep.violations
        r2 = Reference.from_dict(r.to_dict())
        for t in (0.0, 1.3, 7.7):
            a, b = r.at(t), r2.at(t)
            if r.kind == "position":
                assert np.allclose(a.position, b.position)
            assert a.yaw == pytest.approx(b.yaw)


def test_accel_decel_ends_at_rest():
    r = Reference("accel_decel", {"azimuth_deg": 30.0, "speed_mps": 2.0, "ramp_s": 1.0, "cruise_s": 2.0},
                  ANCHOR, 10.0)
    q = r.at(6.0)
    assert np.allclose(q.velocity, 0.0, atol=1e-6)
    assert np.linalg.norm(r.at(2.0).velocity) == pytest.approx(2.0, rel=1e-6)


def test_heading_relative_translation():
    r = Reference("line", {"heading_offset_deg": 0.0, "speed_mps": 1.0, "ramp_s": 1.0}, ANCHOR, 10.0)
    v = r.at(5.0).velocity
    assert np.allclose(v[:2] / np.linalg.norm(v[:2]), [math.cos(ANCHOR.yaw), math.sin(ANCHOR.yaw)])


def test_yaw_sine_is_a_true_sinusoid_after_ramp():
    r = Reference("hover", {}, ANCHOR, 10.0,
                  {"type": "sine", "amplitude_deg": 30.0, "frequency_hz": 0.2, "ramp_s": 1.0})
    ts = np.linspace(2.0, 9.0, 200)
    y = np.array([r.yaw(t) for t in ts]) - ANCHOR.yaw
    assert np.allclose(y, math.radians(30) * np.sin(2 * math.pi * 0.2 * ts), atol=1e-12)


def test_feasibility_flags_aggressive_reference(drone):
    r = Reference("sinusoidal", {"azimuth_deg": 0.0, "amplitude_m": 2.0, "frequency_hz": 1.5, "ramp_s": 0.5},
                  ANCHOR, 6.0)
    rep = check(r, drone, FeasibilityLimits())
    assert not rep.feasible and any("accel" in v for v in rep.violations)


# -- signals / perturbations ---------------------------------------------------

@pytest.mark.parametrize("tmpl", [
    {"type": "multisine", "amplitude": 0.3, "f_min_hz": 0.1, "f_max_hz": 1.0, "n_tones": 8, "ramp_s": 1.0},
    {"type": "chirp", "amplitude": 0.2, "f0_hz": 0.1, "f1_hz": 1.0, "sweep_s": 6.0, "ramp_s": 1.0},
    {"type": "prbs", "amplitude": 0.2, "clock_s": 1.0, "edge_s": 0.5},
    {"type": "smooth_random", "amplitude": 0.2, "n_tones": 10, "ramp_s": 1.0},
    {"type": "doublet", "amplitude": 0.5, "start_s": 1.0, "width_s": 0.3},
])
def test_signals_bounded_and_deterministic(tmpl):
    a = S.sample_signal(tmpl, np.random.default_rng(0), 10.0)
    b = S.sample_signal(tmpl, np.random.default_rng(0), 10.0)
    assert a == b
    v = np.array([S.evaluate(a, t) for t in np.linspace(0, 10, 2001)])
    assert np.all(np.abs(v) <= tmpl["amplitude"] + 1e-12)


def test_prbs_starts_from_zero_and_edge_capped():
    s = S.sample_signal({"type": "prbs", "amplitude": 1.0, "clock_s": 0.5, "edge_s": 2.0},
                        np.random.default_rng(1), 5.0)
    assert s["edge_s"] == 0.5
    assert S.evaluate(s, 0.0) == 0.0


def test_perturbation_must_start_at_rest():
    base = Reference("hover", {}, ANCHOR, 8.0)
    with pytest.raises(ValueError):
        perturb(base, {"x": {"type": "multisine", "amplitude": 0.2}}, np.random.default_rng(0))
    ok = perturb(base, load_template("excitation", "ref_smooth")["channels"], np.random.default_rng(0))
    assert np.allclose(ok.at(0.0).position, ANCHOR.p, atol=1e-9)
    assert np.allclose(reference_from_dict(ok.to_dict()).position(3.3), ok.position(3.3))


# -- open loop ---------------------------------------------------------------------

def test_rotor_program_trim_is_hover_command(drone):
    prog = make_program("rotor_command", {}, drone, np.random.default_rng(0), 3.0)
    applied, raw = prog.rotor_command(drone, 1.0)
    assert np.allclose(raw, drone.hover_command, atol=1e-12)
    assert np.allclose(applied, raw)


def test_rotor_program_torque_maps_through_allocation(drone):
    prog = make_program("rotor_command", {"tau_x": {"type": "step", "amplitude": 0.02}}, drone,
                        np.random.default_rng(0), 3.0)
    T = prog.rotor_thrusts(drone, 1.0)
    wrench = allocation_matrix(drone) @ T
    assert np.allclose(wrench, [0.02, 0.0, 0.0, drone.weight], atol=1e-12)


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_allocation_matches_omnidrones(drone):
    import torch
    from omni_drones.controllers.lee_position_controller import compute_parameters
    p = drone.to_omnidrones_params()
    I = torch.diag_embed(torch.tensor([*drone.inertia_diag, 1.0]))
    mixer = compute_parameters(p["rotor_configuration"], I).numpy()
    A = allocation_matrix(drone)
    # OmniDrones mixer = Aᵀ(AAᵀ)⁻¹ I  ⇒  A · mixer = I (rows of A identical up to float32)
    assert np.allclose(A @ mixer, np.diag([*drone.inertia_diag, 1.0]), atol=1e-6)


# -- GP sampler (plan T7) -----------------------------------------------------------------

def test_gp_sampler_works_on_low_rank_covariance():
    k = make_composite(1.0, [1.3, 3.0, 4.0], [10.0, 20.0, 30.0])
    t = np.linspace(0, 25, 300)
    a = sample_joint_gp(k, t, mean=[0, 0, 1.5], rng=np.random.default_rng(0))
    b = sample_joint_gp(k, t, mean=[0, 0, 1.5], rng=np.random.default_rng(0))
    for x, y in zip(a, b):
        assert x.shape == (300, 3) and np.array_equal(x, y)
    C = joint_covariance(k, t, 0.0)
    w = np.linalg.eigvalsh(C)
    assert w.min() >= -1e-9 * w.max()                              # PSD up to round-off
