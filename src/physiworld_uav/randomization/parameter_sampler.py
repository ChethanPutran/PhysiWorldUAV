"""Physical-domain sampling with coupling, space-filling designs and ID/OOD regions.

Implements plan §6.4 (R2, R3) and decision D9.

Parameterisation
----------------
Every randomised quantity is a *relative factor* f around the nominal value,
driven by a coordinate ``s ∈ [-1, 1)`` from a space-filling design:

* region ``id``          f = 1 + h · g(s)        (h = ID half-width, e.g. 0.10)
* region ``ood``         f = 1 + sign(s)·(a + (b − a)|s|)   (shell a < |δ| ≤ b)
* region ``nominal``     f = 1

``g`` is the identity for ``uniform``; for ``truncnorm`` / ``lognormal`` the
coordinate is mapped through the corresponding inverse CDF restricted to the
same support, so the support (and therefore the ID/OOD boundary) never moves.

Coupling (D9)
-------------
* Inertia:   J_i = J0_i · (m/m0) · s_J,i     (mass scaling × small shape factor)
* Torque:    k_M = k_M0 · f_F · f_ratio      (k_M/k_F ratio varies only by f_ratio)
* Feasibility: thrust-to-weight ≥ ``min_thrust_to_weight`` and a valid inertia,
  otherwise the design point is rejected and the next one is used.

Held-out combinations ("unseen combinations of known parameters")
----------------------------------------------------------------
A corner such as ``{mass: +0.5, force_scale: -0.5}`` means s_mass > 0.5 and
s_force < −0.5. ID sampling *rejects* points inside any held-out corner;
``ood_combination`` sampling draws *only* inside it. Training data therefore
never contains these combinations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.stats import qmc

from physiworld_uav.drones.params import DroneParams

#: Order of design dimensions (fixed so designs are reproducible).
PARAMETERS: tuple[str, ...] = (
    "mass", "inertia_shape_x", "inertia_shape_y", "inertia_shape_z",
    "force_scale", "km_kf_ratio", "motor_time_constant",
    "rotor_drag_scale", "body_drag_scale",
)
REGIONS = ("nominal", "id", "ood_physics", "ood_combination")
# Factors that are randomised but are NOT dimensions of the space-filling design: drawn from their own generator AFTER the design, so
# enabling them leaves every other factor of every domain unchanged (existing manifests stay reproducible).
EXTRA_PARAMETERS: tuple[str, ...] = ("air_density_scale",)   # multiplies ρ in the body drag and the rotor-drag coefficient (∝ ρ)


@dataclass(frozen=True)
class ParameterSpec:
    id_halfwidth: float = 0.0                     # h; 0 → parameter not randomised in ID
    ood_shell: tuple[float, float] | None = None  # (a, b) relative; None → stays ID in OOD
    distribution: str = "uniform"                 # uniform | truncnorm | lognormal

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ParameterSpec":
        shell = d.get("ood_shell")
        spec = cls(
            id_halfwidth=float(d.get("id_halfwidth", 0.0)),
            ood_shell=None if shell is None else (float(shell[0]), float(shell[1])),
            distribution=str(d.get("distribution", "uniform")),
        )
        if spec.id_halfwidth < 0:
            raise ValueError("id_halfwidth must be >= 0")
        if spec.ood_shell is not None:
            a, b = spec.ood_shell
            if not (0 <= a < b < 1):
                raise ValueError(f"ood_shell must satisfy 0 <= a < b < 1, got {spec.ood_shell}")
            if a < spec.id_halfwidth - 1e-12:
                raise ValueError(f"ood_shell inner bound {a} overlaps the ID range ±{spec.id_halfwidth}")
        if spec.distribution not in ("uniform", "truncnorm", "lognormal"):
            raise ValueError(f"unknown distribution {spec.distribution!r}")
        return spec


@dataclass(frozen=True)
class DomainSamplerConfig:
    parameters: dict[str, ParameterSpec]
    sampler: str = "sobol"                        # sobol | lhs | random
    min_thrust_to_weight: float = 2.0
    holdout_corners: tuple[dict[str, float], ...] = ()
    max_rejection_factor: int = 64
    extras: dict[str, "ParameterSpec"] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DomainSamplerConfig":
        params = {k: ParameterSpec.from_dict(v) for k, v in (d.get("parameters") or {}).items()}
        unknown = set(params) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"unknown domain parameters: {sorted(unknown)}")
        for name in PARAMETERS:
            params.setdefault(name, ParameterSpec())
        extras = {k: ParameterSpec.from_dict(v) for k, v in (d.get("extra_factors") or {}).items()}
        bad = set(extras) - set(EXTRA_PARAMETERS)
        if bad:
            raise ValueError(f"unknown extra domain factors {sorted(bad)}; allowed {EXTRA_PARAMETERS}")
        corners = tuple(dict(c) for c in (d.get("holdout_corners") or ()))
        for c in corners:
            for k, v in c.items():
                if k not in PARAMETERS or not -1.0 < float(v) < 1.0:
                    raise ValueError(f"invalid holdout corner {c}")
        sampler = str(d.get("sampler", "sobol"))
        if sampler not in ("sobol", "lhs", "random"):
            raise ValueError(f"unknown sampler {sampler!r}")
        return cls(parameters=params, sampler=sampler,
                   min_thrust_to_weight=float(d.get("min_thrust_to_weight", 2.0)),
                   holdout_corners=corners,
                   max_rejection_factor=int(d.get("max_rejection_factor", 64)),
                   extras=extras)


@dataclass(frozen=True)
class Domain:
    """One fixed physical configuration (sampled values; realised values are
    read back from the simulator and stored separately in Run 4)."""

    domain_id: str
    region: str
    factors: dict[str, float]                     # relative factor per PARAMETERS entry
    coordinates: dict[str, float] = field(default_factory=dict)   # s ∈ [-1, 1) per parameter
    seed_path: str = ""

    def drone(self, nominal: DroneParams) -> DroneParams:
        f = self.factors
        mass = nominal.mass * f["mass"]
        mass_ratio = mass / nominal.mass
        inertia = tuple(
            J0 * mass_ratio * f[f"inertia_shape_{ax}"]
            for J0, ax in zip(nominal.inertia_diag, "xyz")
        )
        return nominal.scaled(
            mass=mass,
            inertia_diag=inertia,
            force_scale=f["force_scale"],
            moment_scale=f["force_scale"] * f["km_kf_ratio"],
            motor_time_constant=nominal.effective_motor_time_constant * f["motor_time_constant"],
        )

    @property
    def rotor_drag_scale(self) -> float:
        return self.factors["rotor_drag_scale"]

    @property
    def body_drag_scale(self) -> float:
        return self.factors["body_drag_scale"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def nominal_domain() -> Domain:
    return Domain(domain_id="nominal", region="nominal",
                  factors={k: 1.0 for k in PARAMETERS},
                  coordinates={k: 0.0 for k in PARAMETERS})


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _unit_design(n: int, d: int, sampler: str, rng: np.random.Generator) -> np.ndarray:
    """(n, d) points in [0, 1)^d."""
    if sampler == "random":
        return rng.random((n, d))
    seed = int(rng.integers(0, 2**31 - 1))
    if sampler == "sobol":
        engine = qmc.Sobol(d=d, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        return engine.random_base2(m)[:n]
    engine = qmc.LatinHypercube(d=d, seed=seed)
    return engine.random(n)


def _shape(s: float, distribution: str) -> float:
    """Map s ∈ [-1, 1) to g ∈ [-1, 1) with the requested in-support distribution."""
    u = 0.5 * (s + 1.0)                                    # ∈ [0, 1)
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    if distribution == "uniform":
        return s
    # truncnorm and lognormal: standard normal truncated at ±2σ, rescaled to [-1, 1).
    # For lognormal the caller maps g linearly in log-factor space (same support).
    return float(stats.truncnorm.ppf(u, -2.0, 2.0) / 2.0)


def _factor(spec: ParameterSpec, s: float, use_ood: bool) -> float:
    if use_ood and spec.ood_shell is not None:
        a, b = spec.ood_shell
        mag = a + (b - a) * abs(s)
        return float(1.0 + np.sign(s if s != 0 else 1.0) * mag)
    if spec.id_halfwidth == 0.0:
        return 1.0
    g = _shape(s, spec.distribution)
    if spec.distribution == "lognormal":
        h = spec.id_halfwidth
        lo, hi = np.log(1 - h), np.log(1 + h)              # log-space support
        return float(np.exp(lo + (hi - lo) * 0.5 * (g + 1.0)))
    return float(1.0 + spec.id_halfwidth * g)


def _in_corner(coords: Mapping[str, float], corner: Mapping[str, float]) -> bool:
    return all((coords[k] > v) if v > 0 else (coords[k] < v) for k, v in corner.items())


def _corner_coordinates(s: np.ndarray, corner: Mapping[str, float]) -> np.ndarray:
    """Map a full design point into the held-out corner (only cornered dims change)."""
    out = s.copy()
    for k, v in corner.items():
        i = PARAMETERS.index(k)
        u = 0.5 * (s[i] + 1.0)                              # [0,1)
        out[i] = v + (1.0 - v) * u if v > 0 else -1.0 + (v + 1.0) * u
    return out


def sample_domains(
    cfg: DomainSamplerConfig,
    nominal: DroneParams,
    n: int,
    region: str,
    rng: np.random.Generator,
    *,
    id_prefix: str = "dom",
) -> list[Domain]:
    """Sample ``n`` feasible domains in ``region`` (deterministic given ``rng``)."""
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
    if region == "nominal":
        return [nominal_domain()]
    if region == "ood_combination" and not cfg.holdout_corners:
        raise ValueError("ood_combination requires holdout_corners in the domain config")
    if region == "ood_physics" and not any(p.ood_shell for p in cfg.parameters.values()):
        raise ValueError("ood_physics requires at least one parameter with an ood_shell")

    d = len(PARAMETERS)
    budget = n * cfg.max_rejection_factor
    design = 2.0 * _unit_design(budget, d, cfg.sampler, rng) - 1.0   # [-1, 1)
    extra_s: dict[str, np.ndarray] = {}
    if cfg.extras:                                   # own generator, seeded from rng AFTER the design (design draws are unchanged)
        erng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        extra_s = {k: 2.0 * erng.random(budget) - 1.0 for k in sorted(cfg.extras)}
    out: list[Domain] = []
    for j, s in enumerate(design):
        if region == "ood_combination":
            corner = cfg.holdout_corners[j % len(cfg.holdout_corners)]
            s = _corner_coordinates(s, corner)
        coords = {k: float(v) for k, v in zip(PARAMETERS, s)}
        if region == "id" and any(_in_corner(coords, c) for c in cfg.holdout_corners):
            continue
        factors = {
            k: _factor(cfg.parameters[k], coords[k], use_ood=(region == "ood_physics"))
            for k in PARAMETERS
        }
        for k, spec in cfg.extras.items():
            factors[k] = _factor(spec, float(extra_s[k][j]), use_ood=(region == "ood_physics" and spec.ood_shell is not None))
            coords[k] = float(extra_s[k][j])
        dom = Domain(domain_id=f"{id_prefix}_{len(out):05d}", region=region,
                     factors=factors, coordinates=coords)
        try:
            params = dom.drone(nominal)
        except ValueError:
            continue
        if params.thrust_to_weight < cfg.min_thrust_to_weight:
            continue
        out.append(dom)
        if len(out) == n:
            return out
    raise RuntimeError(
        f"only {len(out)}/{n} feasible domains in region {region!r} after {budget} "
        f"design points; relax constraints or increase max_rejection_factor"
    )


def design_discrepancy(domains: Sequence[Domain]) -> float:
    """Centered L2 discrepancy of the domains' design coordinates (coverage metric)."""
    if len(domains) < 2:
        return float("nan")
    pts = np.array([[0.5 * (d.coordinates[k] + 1.0) for k in PARAMETERS] for d in domains])
    return float(qmc.discrepancy(np.clip(pts, 0.0, 1.0 - 1e-12)))
