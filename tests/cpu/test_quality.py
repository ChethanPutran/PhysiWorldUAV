"""IMU noise transform, run tracking, and the statistical / physical / coverage validator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_models import PH, synthetic_episode  # noqa: E402

from physiworld_uav.data import statistical_validator as sv  # noqa: E402
from physiworld_uav.data.dataset_builder import BuilderConfig, ModelEpisode  # noqa: E402
from physiworld_uav.data.dataset import ModelReadyDataset  # noqa: E402
from physiworld_uav.data.noise import ImuNoise, ImuNoiseConfig  # noqa: E402
from physiworld_uav.training.tracking import Tracker, load_runs  # noqa: E402


# ----------------------------------------------------------------- noise
def _sample(i=0, ep="e0"):
    return {"obs_history": torch.zeros(10, 6), "state_history": torch.ones(10, 13), "episode_id": ep, "time_index": i}


def test_noise_none_is_identity_and_unknown_preset_rejected():
    s = _sample()
    assert ImuNoise("none")(s) is s
    with pytest.raises(ValueError):
        ImuNoiseConfig.preset("bogus")


def test_noise_is_reproducible_per_window_and_touches_only_the_imu():
    n = ImuNoise("nominal", seed=3)
    a, b, c = n(_sample(5)), n(_sample(5)), n(_sample(6))
    assert torch.equal(a["obs_history"], b["obs_history"]) and not torch.equal(a["obs_history"], c["obs_history"])
    assert torch.equal(a["state_history"], torch.ones(10, 13)) and a["episode_id"] == "e0"
    assert ImuNoise("nominal", seed=4)(_sample(5))["obs_history"].abs().sum() != a["obs_history"].abs().sum()
    d = ImuNoise("nominal", deterministic=False)
    assert not torch.equal(d(_sample(5))["obs_history"], d(_sample(5))["obs_history"])


def test_noise_statistics_match_the_preset():
    cfg = ImuNoiseConfig(accel_white=0.2, accel_bias=0.1, gyro_white=0.02, gyro_bias=0.01)
    n = ImuNoise(cfg, seed=1)
    obs = torch.stack([n(_sample(i, f"e{i % 7}"))["obs_history"] for i in range(3000)])           # (N,10,6)
    within = obs.std(dim=1).mean(0)                                                                # white noise (bias is constant in a window)
    across = obs.mean(dim=1).std(dim=0)                                                            # bias + white/√10
    assert torch.allclose(within[:3], torch.full((3,), 0.2), rtol=0.1) and torch.allclose(within[3:], torch.full((3,), 0.02), rtol=0.1)
    assert torch.allclose(across[:3], torch.full((3,), float(np.sqrt(0.1 ** 2 + 0.2 ** 2 / 10))), rtol=0.1)


def test_noise_works_as_a_dataset_transform():
    ep = synthetic_episode("a", 1)
    ds = ModelReadyDataset([ep], BuilderConfig(history=10, horizon=5), transform=ImuNoise("high", seed=0))
    clean = ModelReadyDataset([ep], BuilderConfig(history=10, horizon=5))
    assert not torch.equal(ds[3]["obs_history"], clean[3]["obs_history"])
    assert torch.equal(ds[3]["future_states"], clean[3]["future_states"])


# ----------------------------------------------------------------- tracking
def test_tracker_registers_start_and_end(tmp_path):
    t = Tracker(tmp_path / "exp" / "runA", "runA", {"model": "b2", "dataset": "d_v1", "n_params": 5})
    t.end({"epochs": 3, "best_metric": 0.1, "stop_reason": "epochs_done"})
    Tracker(tmp_path / "exp" / "runB", "runB", {"model": "b3"})
    runs = {r["run"]: r for r in load_runs(tmp_path / "exp")}
    assert runs["runA"]["status"] == "finished" and runs["runA"]["epochs"] == 3 and runs["runB"]["status"] == "running"


# ----------------------------------------------------------------- validator
def eps_and_rows(n=6, **kw):
    eps = [synthetic_episode(f"e{i}", i, **kw) for i in range(n)]
    rows = {e.episode_id: dict(episode_id=e.episode_id, group_id=f"g{i}", split_role="train", data_type="closed_loop", seeds={"episode": i},
                               domain={"domain_id": f"d{i}", "region": "id", "factors": {}}, scenario_template="hover", scenario_id=f"s{i}",
                               wind={"regime": "id", "stats": {}}) for i, e in enumerate(eps)}
    return eps, rows


def test_range_check_flags_frozen_channels_and_ground_penetration():
    eps, _ = eps_and_rows()
    f, st = sv.check_range(eps)
    assert not [x for x in f if x["level"] == "fail"]
    bad = [ModelEpisode(e.episode_id, None, e.t, e.states.copy(), e.obs, e.actions.copy(), e.dt, {}) for e in eps]
    for e in bad:
        e.actions[:, 3] = 450.0; e.states[:, 2] = -0.5
    f, _ = sv.check_range(bad)
    assert any(x["check"] == "range.coverage" and x["level"] == "fail" for x in f)
    assert any(x["check"] == "range.pz" and x["level"] == "warn" for x in f)


def test_duplicate_check_finds_copies_and_passes_distinct_episodes():
    eps, rows = eps_and_rows()
    assert all(x["level"] == "pass" for x in sv.check_duplicates(eps, rows))
    copy = ModelEpisode("copy", None, eps[0].t, eps[0].states.copy(), eps[0].obs, eps[0].actions.copy(), eps[0].dt, {})
    rows["copy"] = dict(rows["e0"], episode_id="copy", group_id="gX", seeds={"episode": 99})
    f = sv.check_duplicates(eps + [copy], rows)
    assert any(x["check"] == "duplicates.exact" and x["level"] == "fail" for x in f)
    rows["e1"]["seeds"]["episode"] = rows["e0"]["seeds"]["episode"]
    assert any(x["check"] == "duplicates.seeds" and x["level"] == "fail" for x in sv.check_duplicates(eps, rows))


def test_physical_check_passes_consistent_and_fails_corrupted_data():
    eps, _ = eps_and_rows(3)
    f, summ = sv.check_physical(eps)
    assert [x for x in f if x["check"] in ("physical.pos_vel", "physical.quaternion_norm", "physical.quaternion_sign")] and \
        all(x["level"] == "pass" for x in f if x["check"] in ("physical.pos_vel", "physical.quaternion_norm", "physical.quaternion_sign", "physical.gyro"))
    bad = ModelEpisode("x", None, eps[0].t, eps[0].states.copy(), eps[0].obs, eps[0].actions, eps[0].dt, {})
    bad.states[:, 0:3] += np.random.default_rng(0).normal(0, 0.05, (len(bad.states), 3))          # positions inconsistent with velocities
    bad.states[10:, 6:10] *= -1                                                                    # quaternion sign flip
    f, _ = sv.check_physical([bad])
    assert any(x["check"] == "physical.pos_vel" and x["level"] == "warn" for x in f)
    assert any(x["check"] == "physical.quaternion_sign" and x["level"] == "fail" for x in f)


def _raw(T_scale: float, n=50, mass=0.716, dt=0.005):
    q = np.tile([1.0, 0, 0, 0], (n + 1, 1)); rot = np.full((n + 1, 4), mass * 9.81 / 4 * T_scale)
    return {"ground_truth/velocity_W": np.zeros((n + 1, 3)), "ground_truth/orientation_wxyz": q, "ground_truth/rotor_thrust": rot,
            "ground_truth/wrench/external_force_W": np.zeros((n, 3))}


def test_force_balance_residual_is_zero_at_hover_trim_and_detects_excess_thrust():
    assert sv.force_balance_residual_raw(_raw(1.0), 0.716, 0.005).max() < 1e-6
    assert sv.force_balance_residual_raw(_raw(1.1), 0.716, 0.005).mean() == pytest.approx(0.981, rel=1e-3)


def test_correlation_check_flags_actions_that_encode_the_wind():
    rng = np.random.default_rng(0)
    eps, rows = eps_and_rows(8)
    for e in eps:
        e.aux["environment/wind_applied_W"] = rng.normal(0, 1, (len(e.actions), 3)).astype(np.float32)
    f, res = sv.check_correlation(eps, rows)
    assert next(x for x in f if x["check"] == "correlation.action_wind.closed_loop")["level"] == "pass"
    for e in eps:
        e.actions[:, 0] = 450 + 30 * e.aux["environment/wind_applied_W"][:, 0]
    f, res = sv.check_correlation(eps, rows)
    assert next(x for x in f if x["check"] == "correlation.action_wind.closed_loop")["level"] == "warn"
    assert res["action_wind"]["closed_loop"]["max_abs"] > 0.9


def test_residual_vs_wind_identifiability_check_runs_with_physics():
    rng = np.random.default_rng(1)
    eps, rows = eps_and_rows(4, wind_acc=(0.0, 0.0, 0.0))
    for e in eps:
        e.aux["environment/wind_applied_W"] = rng.uniform(0, 5, (len(e.actions), 3)).astype(np.float32)
    f, res = sv.check_correlation(eps, rows, physics=PH)
    assert "residual_vs_wind" in res and "closed_loop" in res["residual_vs_wind"]
    assert any(x["check"].startswith("identifiability.residual_vs_wind") for x in f)


def test_distribution_check_fails_on_wind_outside_regime_and_domain_outside_range():
    rows = {"a": dict(episode_id="a", part="p", scenario_template="hover", replay_of=None,
                      wind={"regime": "id", "stats": {"mean_horizontal_speed": 5.0, "peak_speed": 5.5}, "limits": {"mean_horizontal": [0, 4], "peak_speed_max": 4.0}},
                      domain={"domain_id": "d0", "region": "id", "factors": {"mass": 1.25}})}
    cfg = {"parameters": {"mass": {"id_halfwidth": 0.1, "ood_shell": [0.1, 0.2]}}}
    f, _ = sv.check_distribution(rows, cfg, None)
    assert any(x["check"] == "distribution.wind.id" and x["level"] == "fail" for x in f)
    assert any(x["check"] == "distribution.domain_ranges" and x["level"] == "fail" for x in f)


def test_coverage_check_reports_sparse_roles_and_occupancy():
    rows = {}
    for i in range(12):
        rows[f"e{i}"] = dict(episode_id=f"e{i}", group_id=f"g{i % 3}", split_role="train" if i < 8 else "val_ood_gust", scenario_template="hover", scenario_id=f"s{i}", replay_of=None,
                             domain={"domain_id": f"d{i}", "region": "id", "factors": {"mass": 0.9 + 0.2 * i / 11}},
                             wind={"stats": {"mean_horizontal_speed": 0.7 * i, "mean_x": 1.0, "mean_y": 0.3 * i, "mean_elevation_deg": 0.0}, "profile": "steady", "labels": {}},
                             initial_conditions={"position": [0, 0, 1.5], "yaw_deg": 30.0 * i})
    eps = [synthetic_episode(f"e{i}", i) for i in range(12)]
    f, res = sv.check_coverage(rows, eps, {"parameters": {"mass": {"id_halfwidth": 0.1}}})
    assert res["domain_1d"]["mass"]["occupied_fraction"] >= 0.9
    assert any(x["check"] == "coverage.groups_per_role" and x["level"] == "warn" for x in f)
    assert "novelty_vs_train" in res and res["effective"]["episodes"] == 12
