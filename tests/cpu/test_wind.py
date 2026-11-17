import numpy as np
import pytest

from physiworld_uav.wind import (
    CosineGust, DrydenTurbulence, OUGust, SpatialGradient, SteadyWind,
    StepGust, SinusoidalGust, WindField, WindManager,
)


# ── Steady ────────────────────────────────────────────────────────────────
def test_steady_magnitude_and_direction():
    w = SteadyWind(speed_mps=3.0, direction="north")
    assert np.allclose(w.contribution(np.zeros(3), 0.0), [3.0, 0.0, 0.0])


def test_steady_vector_direction():
    w = SteadyWind(speed_mps=2.0, direction=[0.0, 0.0, 1.0])
    assert np.allclose(w.contribution(np.zeros(3), 0.0), [0.0, 0.0, 2.0])


# ── Gradient ─────────────────────────────────────────────────────────────
def test_gradient_zero_at_p0():
    G = np.eye(3) * 0.1
    g = SpatialGradient(G=G, p0=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(g.contribution(np.array([1.0, 2.0, 3.0]), 0.0), 0.0)


def test_gradient_linear_response():
    G = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
    g = SpatialGradient(G=G, p0=np.zeros(3))
    assert np.allclose(g.contribution(np.array([10.0, 5.0, 2.0]), 0.0), [1.0, 1.0, 0.6])


# ── Cosine gust ──────────────────────────────────────────────────────────
def test_cosine_gust_shape():
    g = CosineGust(peak_mps=5.0, start_time_s=2.0, rise_time_s=0.5, direction="north")
    # Before gust
    assert np.allclose(g.contribution(np.zeros(3), 1.9), 0.0)
    # At peak (t_start + T_rise)
    assert np.allclose(g.contribution(np.zeros(3), 2.5), [5.0, 0.0, 0.0], atol=1e-9)
    # At end (t_start + 2*T_rise)
    assert np.allclose(g.contribution(np.zeros(3), 3.0), 0.0, atol=1e-9)
    # Past end
    assert np.allclose(g.contribution(np.zeros(3), 3.5), 0.0)


# ── Step gust ────────────────────────────────────────────────────────────
def test_step_gust_timing():
    g = StepGust(before_mps=1.0, after_mps=4.0, step_time_s=5.0, direction="east")
    assert np.allclose(g.contribution(np.zeros(3), 4.9), [0.0, 1.0, 0.0])
    assert np.allclose(g.contribution(np.zeros(3), 5.0), [0.0, 4.0, 0.0])
    assert np.allclose(g.contribution(np.zeros(3), 10.0), [0.0, 4.0, 0.0])


# ── Sinusoidal ───────────────────────────────────────────────────────────
def test_sinusoidal_frequency():
    g = SinusoidalGust(amplitude_mps=1.0, frequency_hz=0.5, mean_mps=0.0, direction="north")
    # Quarter period = 0.5 s → sin(pi/2) = 1
    assert np.allclose(g.contribution(np.zeros(3), 0.5), [1.0, 0.0, 0.0], atol=1e-9)
    # Half period = 1.0 s → sin(pi) = 0
    assert np.allclose(g.contribution(np.zeros(3), 1.0), 0.0, atol=1e-9)
    # 3/4 period = 1.5 s → sin(3pi/2) = -1
    assert np.allclose(g.contribution(np.zeros(3), 1.5), [-1.0, 0.0, 0.0], atol=1e-9)


# ── OU gust statistics ───────────────────────────────────────────────────
def test_ou_stationary_std():
    g = OUGust(sigma_mps=1.0, tau_s=2.0, seed=0)
    dt = 0.01
    samples = []
    for k in range(50000):
        v = g.contribution(np.zeros(3), k * dt, dt)
        if k > 2000:
            samples.append(v)
    samples = np.array(samples)
    std = samples.std(axis=0)
    assert np.allclose(std, [1.0, 1.0, 0.5], atol=0.1), std      # vertical_ratio 0.5 (plan W2)


def test_ou_is_stationary_from_t0():
    """Plan W2: the state starts from the stationary distribution, not from zero."""
    first = np.array([OUGust(sigma_mps=1.0, tau_s=2.0, seed=s).contribution(np.zeros(3), 0.0) for s in range(2000)])
    assert np.allclose(first.std(axis=0), [1.0, 1.0, 0.5], atol=0.06)


def test_ou_autocorrelation_time():
    g = OUGust(sigma_mps=1.0, tau_s=2.0, seed=0)
    dt = 0.01
    samples = []
    for k in range(20000):
        samples.append(g.contribution(np.zeros(3), k * dt, dt)[0])
    samples = np.array(samples[2000:])
    # Autocorrelation at lag tau (~2 s = 200 steps) ≈ exp(-1) ≈ 0.37
    lag = 200
    r = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
    assert 0.25 < r < 0.5, r


# ── Dryden statistics ────────────────────────────────────────────────────
def test_dryden_stationary_std():
    turb = DrydenTurbulence(
        airspeed_mps=10.0,
        intensities_mps=(1.0, 1.0, 1.0),
        length_scales_m=(200.0, 200.0, 200.0),
        seed=0,
    )
    dt = 0.05
    samples = []
    for k in range(20000):
        v = turb.contribution(np.zeros(3), k * dt, dt)
        if k > 1000:
            samples.append(v)
    samples = np.array(samples)
    std = samples.std(axis=0)
    assert np.allclose(std, 1.0, atol=0.15), std


# ── Field composition ────────────────────────────────────────────────────
def test_field_sums_components():
    field = WindField(
        components={
            "steady": SteadyWind(2.0, "north"),
            "gradient": SpatialGradient(np.zeros((3, 3))),
        },
    )
    sample = field.evaluate(np.zeros(3), 0.0)
    assert np.allclose(sample.total_W, [2.0, 0.0, 0.0])
    assert np.isclose(sample.speed_mps, 2.0)
    assert "steady" in sample.by_component


def test_field_wind_scale():
    field = WindField({"steady": SteadyWind(2.0, "north")}, wind_scale=2.5)
    sample = field.evaluate(np.zeros(3), 0.0)
    assert np.allclose(sample.total_W, [5.0, 0.0, 0.0])




def test_manager_reproducible_sampling():
    spec = {
        "steady": {"enabled": True, "speed_mps": [0.0, 10.0],
                   "direction": ["north", "south", "east", "west"]},
    }
    a = WindManager(spec, seed=7).configure()
    b = WindManager(spec, seed=7).configure()
    for k in range(5):
        sa = a.get_wind_state(np.zeros(3), k * 0.1)
        sb = b.get_wind_state(np.zeros(3), k * 0.1)
        assert np.allclose(sa.total_W, sb.total_W)


def test_manager_direction_override_removed():
    """Plan W6: the output-stage override rectified zero-mean components; it now raises."""
    spec = {
        "steady": {"enabled": True, "speed_mps": 3.0, "direction": "north"},
        "ood": {"direction_override": "east"},
    }
    with pytest.raises(ValueError, match="direction_override was removed"):
        WindManager(spec, seed=0)


def test_manager_rotate_z_rotates_whole_field():
    spec = {"steady": {"enabled": True, "speed_mps": 3.0, "direction": "north"},
            "ood": {"rotate_z_rad": float(np.pi / 2)}}
    sample = WindManager(spec, seed=0).configure().get_wind_state(np.zeros(3), 0.0)
    assert np.allclose(sample.total_W, [0.0, 3.0, 0.0])


def test_manager_reset_clears_ou_state():
    spec = {"gust": [{"enabled": True, "type": "ou", "sigma_mps": 1.0, "tau_s": 1.0}]}
    mgr = WindManager(spec, seed=42).configure()
    mgr.get_wind_state(np.zeros(3), 0.0)
    v1 = mgr.get_wind_state(np.zeros(3), 0.1, dt=0.1).total_W
    mgr.configure(seed=42)  # reset
    mgr.get_wind_state(np.zeros(3), 0.0)
    v2 = mgr.get_wind_state(np.zeros(3), 0.1, dt=0.1).total_W
    assert np.allclose(v1, v2)