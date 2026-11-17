"""Wind regimes and seeding (plan W1, W4, W7, W9; decisions D1, D2)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from physiworld_uav.config import load_template
from physiworld_uav.wind import StepGust, WindManager
from physiworld_uav.wind.regimes import (
    WindRegime, check_limits, evaluate_series, realised_stats, rotated_plan, sample_wind, scaled_plan,
)

DT, DUR = 0.005, 8.0


def _regime(name):
    return WindRegime.from_dict(load_template("wind/regimes", name))


@pytest.mark.parametrize("name", ["id", "ood_magnitude", "ood_direction", "ood_temporal", "ood_gust"])
def test_regime_samples_satisfy_limits(name):
    reg = _regime(name)
    rng = np.random.default_rng(0)
    for _ in range(15):
        p = sample_wind(reg, rng, duration=DUR, dt=DT)
        assert check_limits(p.stats, reg.limits) == []
        # the stored stats are exactly what the runner's evaluation reproduces
        again = realised_stats(evaluate_series(p.spec, p.seed, DUR, DT))
        assert again == pytest.approx(p.stats, abs=1e-12)


def test_id_and_ood_magnitude_are_disjoint():
    rng = np.random.default_rng(1)
    id_peaks = [sample_wind(_regime("id"), rng, duration=DUR, dt=DT).stats["peak_speed"] for _ in range(20)]
    ood_means = [sample_wind(_regime("ood_magnitude"), rng, duration=DUR, dt=DT).stats["mean_horizontal_speed"]
                 for _ in range(20)]
    assert max(id_peaks) <= 4.0
    assert min(ood_means) > 4.0 and max(ood_means) <= 8.0


def test_temporal_labels():
    rng = np.random.default_rng(2)
    labels = {sample_wind(_regime("ood_temporal"), rng, duration=DUR, dt=DT).labels["temporal_class"]
              for _ in range(20)}
    assert labels <= {"fast", "sharp"}


def test_scaled_and_rotated_plans_share_the_realisation():
    p = sample_wind(_regime("id"), np.random.default_rng(3), duration=DUR, dt=DT, profile="ou")
    base = evaluate_series(p.spec, p.seed, DUR, DT)
    half = scaled_plan(p, 0.5, duration=DUR, dt=DT)
    assert np.allclose(evaluate_series(half.spec, half.seed, DUR, DT), 0.5 * base, atol=1e-12)
    rot = rotated_plan(p, math.pi / 2, duration=DUR, dt=DT)
    r = evaluate_series(rot.spec, rot.seed, DUR, DT)
    assert np.allclose(r[:, 0], -base[:, 1]) and np.allclose(r[:, 1], base[:, 0])
    assert np.allclose(r[:, 2], base[:, 2])


def test_components_get_independent_seeds():
    """Before plan W4 both components shared one seed (correlation exactly 1).
    τ = 0.05 s, 4000 samples at 0.01 s ⇒ ~400 effective samples, σ_ρ ≈ 0.05."""
    spec = {"gust": [{"type": "ou", "sigma_mps": 1.0, "tau_s": 0.05},
                     {"type": "ou", "sigma_mps": 1.0, "tau_s": 0.05}]}
    mgr = WindManager(spec, seed=11).configure(seed=11)
    a, b = [], []
    for k in range(4000):
        s = mgr.get_wind_state(np.zeros(3), k * 0.01, 0.01)
        a.append(s.by_component["gust_0"][0])
        b.append(s.by_component["gust_1"][0])
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.2


def test_step_gust_rise_time():
    g = StepGust(0.0, 2.0, step_time_s=1.0, direction="north", rise_time_s=0.5)
    assert np.allclose(g.contribution(None, 0.99), 0.0)
    assert np.allclose(g.contribution(None, 1.25), [1.0, 0, 0])
    assert np.allclose(g.contribution(None, 1.6), [2.0, 0, 0])
