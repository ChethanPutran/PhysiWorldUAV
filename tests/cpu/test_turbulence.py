"""Dryden turbulence (plan §6.2-W3) and temporal scaling (W5)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from physiworld_uav.wind import WindManager
from physiworld_uav.wind.turbulence import DrydenTurbulence, low_altitude


@pytest.mark.parametrize("L", [2.5, 10.0, 200.0])
def test_dryden_stationary_from_t0(L):
    first = np.array([DrydenTurbulence(5.0, (1, 1, 1), (L, L, L), seed=s).contribution(None, 0.0)
                      for s in range(1500)])
    assert np.allclose(first.std(axis=0), 1.0, atol=0.07)


def test_dryden_long_run_std_and_spectrum():
    d = DrydenTurbulence(5.0, (0.8, 0.8, 0.5), (10.0, 10.0, 10.0), seed=3)
    X = np.array([d.contribution(None, k * 0.005, 0.005) for k in range(200000)])
    assert np.allclose(X.std(axis=0), [0.8, 0.8, 0.5], rtol=0.1)
    f, P = welch(X[:, 1] / 0.8, fs=200, nperseg=8192)
    T = 10.0 / 5.0
    w = 2 * np.pi * f
    analytic = (T / np.pi) * (1 + 3 * (T * w) ** 2) / (1 + (T * w) ** 2) ** 2 * 2 * np.pi
    m = (f > 0.02) & (f < 10)
    assert np.median(P[m] / analytic[m]) == pytest.approx(1.0, abs=0.1)


def test_dryden_mean_wind_frame():
    """u along the mean wind: with σ_v = σ_w = 0 the turbulence is parallel to the mean wind."""
    d = DrydenTurbulence(4.0, (1.0, 0.0, 0.0), (10, 10, 10), seed=0, mean_direction_W=np.array([0.0, 1.0, 0.0]))
    X = np.array([d.contribution(None, k * 0.01, 0.01) for k in range(500)])
    assert np.allclose(X[:, 0], 0, atol=1e-12) and np.allclose(X[:, 2], 0, atol=1e-12) and X[:, 1].std() > 0.3


def test_low_altitude_parameters():
    sig, L = low_altitude(2.0, 6.0)                 # clamped to 10 ft
    assert L[2] == pytest.approx(3.048) and sig[2] == pytest.approx(0.6)
    assert sig[0] / sig[2] == pytest.approx(1 / (0.177 + 0.000823 * 10) ** 0.4)


def test_manager_builds_low_altitude_turbulence_advected_by_mean_wind():
    spec = {"steady": {"enabled": True, "speed_mps": 3.0, "direction": [1.0, 0.0, 0.0]},
            "turbulence": {"enabled": True, "w20_mps": 5.0, "altitude_m": 2.0}}
    mgr = WindManager(spec, seed=1).configure(seed=1)
    turb = mgr._field.components["turbulence"]
    assert turb.V == pytest.approx(3.0)


def test_temporal_scale_applies_to_stateful_components():
    def series(scale):
        spec = {"gust": [{"type": "ou", "sigma_mps": 1.0, "tau_s": 1.0}], "ood": {"temporal_scale": scale}}
        m = WindManager(spec, seed=5).configure(seed=5)
        return np.array([m.get_wind_state(np.zeros(3), k * 0.01, 0.01).total_W[0] for k in range(20000)])
    def corr_time(x):
        x = x - x.mean()
        ac = np.correlate(x[:4000], x[:4000], "full")[3999:]
        return np.argmax(ac / ac[0] < np.exp(-1)) * 0.01
    assert corr_time(series(4.0)) < 0.5 * corr_time(series(1.0))
