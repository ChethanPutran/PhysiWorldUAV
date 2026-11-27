"""Priority-1 diversity additions (2026-11-24): Dryden and composite wind profiles, air-density factor, controller gain randomisation,
perturbed starts, longer episodes."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from physiworld_uav.config import compose, to_plain
from physiworld_uav.config.compose import load_template
from physiworld_uav.drones.params import DroneParams
from physiworld_uav.experiments.plan import PlanError, _apply_controller_randomisation, plan_dataset
from physiworld_uav.forces.system import AeroForceSystem
from physiworld_uav.randomization.parameter_sampler import DomainSamplerConfig, sample_domains
from physiworld_uav.wind.regimes import PROFILE_TYPES, WindRegime, evaluate_series, sample_wind

from conftest import omni_drones_available


def _regime(profiles, limits=None, mean_speed=(1.0, 2.0)):
    return WindRegime.from_dict({"name": "t", "category": "id", "mean_speed": list(mean_speed), "azimuth_deg": [0, 360],
                                 "elevation_deg": [0, 0], "profiles": profiles,
                                 "limits": limits or {"peak_speed_max": 8.0, "vertical_max": 2.0, "mean_horizontal": [0.0, 8.0]}})


# ------------------------------------------------------------------ wind
def test_new_profile_types_are_registered_and_unknown_ones_rejected():
    assert "dryden" in PROFILE_TYPES and "composite" in PROFILE_TYPES
    with pytest.raises(ValueError, match="unknown wind profile"):
        _regime([{"type": "nonsense"}])


def test_dryden_profile_adds_turbulence_with_the_requested_intensity_and_is_reproducible():
    reg = _regime([{"type": "dryden", "params": {"sigma_uv_mps": [0.5, 0.5], "sigma_w_ratio": 0.5, "length_scale_uv_m": [20.0, 20.0],
                                                 "length_scale_w_m": [3.0, 3.0]}}])
    p = sample_wind(reg, np.random.default_rng(3), duration=60.0, dt=0.02)
    t = p.spec["turbulence"]
    assert t["enabled"] is True and t["model"] == "dryden" and t["intensities_mps"] == [0.5, 0.5, 0.25] and t["length_scales_m"] == [20.0, 20.0, 3.0]
    assert p.labels["temporal_class"] == "turbulent" and "gust" not in p.spec
    s1 = evaluate_series(p.spec, p.seed, 60.0, 0.02); s2 = evaluate_series(p.spec, p.seed, 60.0, 0.02)
    assert np.array_equal(s1, s2)                                                    # same seed, same realisation
    fluct = s1 - s1.mean(0)
    assert 0.2 < fluct[:, 0].std() < 0.9 and fluct[:, 2].std() < fluct[:, 0].std()   # σ_u ≈ 0.5 (one finite realisation), σ_w smaller


def test_composite_profile_switches_the_wind_level_up_and_back_down():
    reg = _regime([{"type": "composite", "params": {"components": [
        {"type": "step", "params": {"delta_mps": [2.0, 2.0], "rise_time_s": [0.5, 0.5], "onset_fraction": [0.3, 0.3]}},
        {"type": "step", "params": {"delta_mps": [-2.0, -2.0], "rise_time_s": [0.5, 0.5], "onset_fraction": [0.7, 0.7]}}]}}],
        mean_speed=(1.0, 1.0))
    p = sample_wind(reg, np.random.default_rng(1), duration=20.0, dt=0.05)
    assert len(p.spec["gust"]) == 2 and p.labels["temporal_class"] == "composite"
    s = np.linalg.norm(evaluate_series(p.spec, p.seed, 20.0, 0.05), axis=1)
    early, mid, late = s[: int(0.2 * 400)].mean(), s[int(0.4 * 400): int(0.65 * 400)].mean(), s[int(0.8 * 400):].mean()
    assert mid > early + 1.5 and abs(late - early) < 0.3                             # up in the middle, back to the start level at the end


def test_composite_may_overlay_dryden_but_not_nest_and_limits_still_hold():
    reg = _regime([{"type": "composite", "params": {"components": [
        {"type": "step", "params": {"delta_mps": [0.5, 1.0]}}, {"type": "dryden", "params": {"sigma_uv_mps": [0.2, 0.3]}}]}}],
        limits={"peak_speed_max": 4.0, "vertical_max": 1.0, "mean_horizontal": [0.0, 4.0]})
    p = sample_wind(reg, np.random.default_rng(5), duration=15.0, dt=0.02)
    assert "turbulence" in p.spec and "gust" in p.spec and p.stats["peak_speed"] <= 4.0 and p.stats["max_abs_vertical"] <= 1.0
    for bad in ({"type": "composite", "params": {"components": [{"type": "composite", "params": {}}]}},
                {"type": "composite", "params": {"components": []}}):
        with pytest.raises(ValueError):
            sample_wind(_regime([bad]), np.random.default_rng(0), duration=5.0, dt=0.05)


def test_id_diverse_regime_loads_and_every_draw_respects_the_id_limits():
    cfg = to_plain(load_template("wind/regimes", "id_diverse")) if False else None
    from physiworld_uav.config.compose import load_template as lt
    reg = WindRegime.from_dict(lt("wind/regimes", "id_diverse"))
    kinds = set()
    rng = np.random.default_rng(0)
    for _ in range(40):
        p = sample_wind(reg, rng, duration=10.0, dt=0.02)
        kinds.add(p.profile)
        assert p.stats["peak_speed"] <= 4.0 + 1e-9 and p.stats["max_abs_vertical"] <= 1.0 + 1e-9
    assert {"dryden", "composite"} <= kinds


# ------------------------------------------------------------------ air density
def test_air_density_factor_is_not_a_design_dimension_and_stays_in_range():
    base = load_template("domain", "randomized")
    env = load_template("domain", "randomized_env")
    nominal = DroneParams.from_yaml("config/drone/hummingbird.yaml")
    a = sample_domains(DomainSamplerConfig.from_dict(base), nominal, 12, "id", np.random.default_rng(7))
    b = sample_domains(DomainSamplerConfig.from_dict(env), nominal, 12, "id", np.random.default_rng(7))
    for x, y in zip(a, b):                                                           # every design factor is unchanged by the extra factor
        assert x.factors == {k: v for k, v in y.factors.items() if k != "air_density_scale"}
    rho = np.array([d.factors["air_density_scale"] for d in b])
    assert rho.min() >= 0.9 - 1e-9 and rho.max() <= 1.1 + 1e-9 and rho.std() > 0.02
    assert "air_density_scale" not in a[0].factors
    o = sample_domains(DomainSamplerConfig.from_dict(env), nominal, 12, "ood_physics", np.random.default_rng(7))
    assert all(0.1 - 1e-9 <= abs(d.factors["air_density_scale"] - 1.0) <= 0.2 + 1e-9 for d in o)    # OOD shell
    with pytest.raises(ValueError, match="unknown extra"):
        DomainSamplerConfig.from_dict({**env, "extra_factors": {"bogus": {"id_halfwidth": 0.1}}})


def test_air_density_scales_body_and_rotor_drag_linearly():
    import torch
    cfg = load_template("forces", "truth_default")
    nominal = DroneParams.from_yaml("config/drone/hummingbird.yaml")
    state = torch.zeros(1, 13); state[0, 3] = 1.0; state[0, 7] = 4.0                  # identity attitude, moving at 4 m/s
    omega = torch.full((1, 4), 450.0)
    wind0 = torch.zeros(1, 3); wind_rot = torch.zeros(1, 4, 3)
    f1 = AeroForceSystem.from_config(cfg, nominal, nominal, {}).compute(state, omega, wind0, wind_rot).total_force_W
    f2 = AeroForceSystem.from_config(cfg, nominal, nominal, {"air_density_scale": 1.1}).compute(state, omega, wind0, wind_rot).total_force_W
    assert torch.allclose(f2, 1.1 * f1, rtol=1e-5) and float(f1.norm()) > 0.05


# ------------------------------------------------------------------ controller gains, starts, durations (planner)
@pytest.fixture(scope="module")
def div_plan():
    cfg = to_plain(compose(["experiment=diversity_check"]))
    return cfg, plan_dataset(cfg)


def test_diversity_plan_is_deterministic_and_contains_every_feature(div_plan):
    cfg, m = div_plan
    m2 = plan_dataset(copy.deepcopy(cfg))
    import json
    assert json.dumps(m.episodes, sort_keys=True) == json.dumps(m2.episodes, sort_keys=True)
    rows = m.episodes
    by = lambda part: [r for r in rows if r["part"] == part]                         # noqa: E731
    assert all(np.linalg.norm(r["initial_conditions"]["velocity_W"]) > 0 for r in by("div_start"))       # perturbed starts
    assert all(np.any(r["initial_conditions"]["angular_velocity_B"]) for r in by("div_start"))
    assert all(19.0 <= r["duration_s"] <= 31.0 for r in by("div_long")) and all(r["duration_s"] < 20 for r in by("div_wind"))
    assert {"dryden", "composite"} <= {r["wind"]["profile"] for r in by("div_wind")}
    assert all("air_density_scale" in r["domain"]["factors"] for r in by("div_env"))
    assert all("air_density_scale" not in r["domain"]["factors"] for r in by("div_start"))
    for r in by("div_gain"):
        rnd = r["controller"]["randomisation"]
        if r["controller"]["mode"] == "position":
            g = r["controller"]["gains"]
            assert 0.7 - 1e-9 <= rnd["position_gain_scale"] <= 1.3 + 1e-9 and np.allclose(g["position_gain"], 4.0 * rnd["position_gain_scale"])
            assert np.allclose(g["velocity_gain"], 2.2 * rnd["velocity_gain_scale"])
        else:
            assert 0.7 - 1e-9 <= rnd["attitude_hold_scale"] <= 1.3 + 1e-9
    assert all(r["controller"].get("randomisation") is None for r in by("div_start") + by("div_wind"))   # off unless the part asks for it


def test_controller_randomisation_needs_nominal_gains_and_leaves_other_modes_alone():
    rng = np.random.default_rng(0)
    with pytest.raises(PlanError, match="nominal_gains"):
        _apply_controller_randomisation({"mode": "position"}, {"position_gain_scale": [0.8, 1.2]}, {"position": {"gains": None}}, rng)
    c = {"mode": "rate", "gains": None}
    _apply_controller_randomisation(c, {"position_gain_scale": [0.5, 2.0]}, {}, rng)
    assert c["gains"] is None and c["randomisation"] == {}


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_runner_accepts_randomised_gains_and_perturbed_starts(div_plan):
    from test_planner_runner import MockSim
    from physiworld_uav.experiments.runner import Termination, run_episode
    cfg, m = div_plan
    nominal = DroneParams.from_dict(cfg["drone"])

    class Mock(MockSim):
        def close(self): pass

    for row in [next(r for r in m.episodes if r["part"] == "div_gain" and r["controller"]["mode"] == "position"),
                next(r for r in m.episodes if r["part"] == "div_start")]:
        row = copy.deepcopy(row); row["n_steps"] = 80; row["duration_s"] = 80 * row["sim"]["physics_dt"]
        arrays, meta = run_episode(Mock(nominal, cfg["physics"]["dt"]), row, nominal, termination=Termination(), config_hash="h",
                                   dataset_version="0", enforce_guards=False)
        assert meta.n_steps == 80
        if row["part"] == "div_start":                                                # moving start: the t = 0 IMU sample is not recorded
            assert not bool(arrays["observations/imu_valid"][0])
        else:
            assert meta.controller["gains"] == row["controller"]["gains"]
