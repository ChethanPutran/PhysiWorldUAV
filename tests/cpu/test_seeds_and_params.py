"""Seeds (plan RP1), drone parameters (F2, S3) and domain sampling (R2, R3, D9)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from physiworld_uav.core.seeds import child_seed, root
from physiworld_uav.drones.params import DroneParams, check_against_omnidrones
from physiworld_uav.randomization.parameter_sampler import (
    PARAMETERS, DomainSamplerConfig, design_discrepancy, nominal_domain, sample_domains,
)


# -- seeds -------------------------------------------------------------------

def test_seed_tree_is_deterministic_and_key_stable():
    a = root(42).descend("part:x", "group:00001", "wind")
    b = root(42).descend("part:x", "group:00001", "wind")
    assert a.seed == b.seed
    assert np.array_equal(a.rng().random(5), b.rng().random(5))
    # independent of sibling creation order / existence
    _ = root(42).descend("part:x", "group:00000")
    assert root(42).descend("part:x", "group:00001", "wind").seed == a.seed


def test_seed_tree_distinct_keys_and_roots():
    r = root(1)
    seeds = {r.child(f"k{i}").seed for i in range(200)}
    assert len(seeds) == 200
    assert root(1).child("a").seed != root(2).child("a").seed
    assert child_seed(5, "x") == root(5).child("x").seed
    with pytest.raises(ValueError):
        r.child("")


# -- drone parameters -------------------------------------------------------------

def test_drone_matches_omnidrones(drone):
    assert check_against_omnidrones(drone) == []


def test_rotor_geometry_in_xy_plane(drone):
    r = drone.rotor_positions_B
    assert r.shape == (4, 3)
    assert np.allclose(r[:, 2], drone.rotor_height)
    assert np.allclose(r[0], [0.17, 0.0, 0.01], atol=1e-9)          # "+" layout, rotor 0 on +x
    assert np.allclose(np.linalg.norm(r[:, :2], axis=1), 0.17)
    assert np.allclose(drone.thrust_axis_B, [0, 0, 1])


def test_hover_quantities_consistent(drone):
    # OmniDrones: cmd = 2 T/T_max − 1  ⇒  Σ T = m g at the hover command
    T = (drone.hover_command + 1) / 2 * drone.max_thrust_per_rotor
    assert math.isclose(T.sum(), drone.mass * 9.81, rel_tol=1e-9)
    assert math.isclose(drone.max_thrust, 4 * 8.54858e-6 * 838 ** 2, rel_tol=1e-9)
    assert drone.thrust_to_weight > 3.0
    assert math.isclose(drone.motor_alpha(drone.motor_alpha_reference_dt), drone.motor_alpha_nominal,
                        rel_tol=1e-9)


def test_drone_validation_rejects_bad_values(drone):
    with pytest.raises(ValueError):
        drone.scaled(mass=-1.0)
    with pytest.raises(ValueError):
        drone.scaled(inertia_diag=(0.001, 0.001, 0.01))        # triangle inequality
    with pytest.raises(ValueError):
        drone.scaled(force_scale=0.2)                          # cannot hover
    back = DroneParams.from_dict({**drone.to_omnidrones_params(), "rotor_height": 0.01})
    assert np.allclose(back.force_constants, drone.force_constants)


# -- domain sampling ---------------------------------------------------------------

CFG = {
    "region": "id", "sampler": "sobol", "min_thrust_to_weight": 2.0,
    "parameters": {
        "mass": {"id_halfwidth": 0.10, "ood_shell": [0.10, 0.20]},
        "inertia_shape_x": {"id_halfwidth": 0.05}, "inertia_shape_y": {"id_halfwidth": 0.05},
        "inertia_shape_z": {"id_halfwidth": 0.05},
        "force_scale": {"id_halfwidth": 0.10, "ood_shell": [0.10, 0.20]},
        "km_kf_ratio": {"id_halfwidth": 0.05},
        "motor_time_constant": {"id_halfwidth": 0.20, "ood_shell": [0.20, 0.40]},
        "rotor_drag_scale": {"id_halfwidth": 0.10}, "body_drag_scale": {"id_halfwidth": 0.10},
    },
    "holdout_corners": [{"mass": 0.5, "force_scale": -0.5}],
}


def _sample(drone, region, n=64, seed=0, **over):
    cfg = DomainSamplerConfig.from_dict({**CFG, **over})
    return sample_domains(cfg, drone, n, region, np.random.default_rng(seed))


def test_id_domains_within_ranges_and_coupled(drone):
    doms = _sample(drone, "id")
    for d in doms:
        assert abs(d.factors["mass"] - 1) <= 0.10 + 1e-12
        assert abs(d.factors["force_scale"] - 1) <= 0.10 + 1e-12
        p = d.drone(drone)
        ratio = p.mass / drone.mass
        for J, J0, ax in zip(p.inertia_diag, drone.inertia_diag, "xyz"):
            assert math.isclose(J, J0 * ratio * d.factors[f"inertia_shape_{ax}"], rel_tol=1e-12)
        km_kf = np.asarray(p.moment_constants) / np.asarray(p.force_constants)
        km_kf0 = np.asarray(drone.moment_constants) / np.asarray(drone.force_constants)
        assert np.allclose(km_kf / km_kf0, d.factors["km_kf_ratio"])
        assert p.thrust_to_weight >= 2.0
    assert design_discrepancy(doms) < 0.05                       # space filling


def test_holdout_corner_excluded_from_id_and_used_by_ood_combination(drone):
    in_corner = lambda d: d.coordinates["mass"] > 0.5 and d.coordinates["force_scale"] < -0.5  # noqa: E731
    assert not any(in_corner(d) for d in _sample(drone, "id", n=128))
    comb = _sample(drone, "ood_combination", n=16)
    assert all(in_corner(d) for d in comb)
    assert all(abs(d.factors["mass"] - 1) <= 0.10 + 1e-12 for d in comb)   # each param still ID


def test_ood_physics_in_shell(drone):
    for d in _sample(drone, "ood_physics", n=32):
        assert 0.10 - 1e-12 <= abs(d.factors["mass"] - 1) <= 0.20 + 1e-12
        assert 0.20 - 1e-12 <= abs(d.factors["motor_time_constant"] - 1) <= 0.40 + 1e-12
        assert abs(d.factors["km_kf_ratio"] - 1) <= 0.05 + 1e-12          # no shell → ID


def test_domain_sampling_deterministic(drone):
    a = [d.factors for d in _sample(drone, "id", seed=3)]
    b = [d.factors for d in _sample(drone, "id", seed=3)]
    assert a == b
    assert nominal_domain().factors == {k: 1.0 for k in PARAMETERS}


def test_invalid_domain_config_rejected():
    with pytest.raises(ValueError):
        DomainSamplerConfig.from_dict({"parameters": {"mass": {"id_halfwidth": 0.2, "ood_shell": [0.1, 0.3]}}})
    with pytest.raises(ValueError):
        DomainSamplerConfig.from_dict({"parameters": {"bogus": {"id_halfwidth": 0.1}}})
