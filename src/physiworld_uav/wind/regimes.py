"""Wind regimes: ID / OOD samplers with realised-statistics guarantees (plan §6.2-W7, D1, D2).

A *regime* (``config/wind/regimes/<name>.yaml``) describes a distribution of
wind specs. ``sample_wind`` draws one concrete spec (only scalars and explicit
unit vectors — no ranges left), evaluates it offline with the exact same
``dt``, seed and component construction the simulator run will use, and
accepts it only if the **realised** statistics satisfy the regime limits.

Because every wind component is position-independent unless a spatial
gradient is enabled (gradients are rejected here), the offline series equals
the series the vehicle experiences; the dataset validator re-checks this
against the logged wind (plan §6.5-DS9).

Units: m/s (D1). Directions: world ENU; azimuth from +x towards +y,
elevation positive upward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .wind_manager import WindManager

PROFILE_TYPES = ("none", "steady", "cosine_gust", "step", "sinusoid", "ou", "random_gusts", "dryden", "composite")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    @classmethod
    def of(cls, v: Any) -> "Range":
        if isinstance(v, (int, float)):
            return cls(float(v), float(v))
        lo, hi = float(v[0]), float(v[1])
        if hi < lo:
            raise ValueError(f"range upper bound < lower bound: {v}")
        return cls(lo, hi)

    def draw(self, rng: np.random.Generator) -> float:
        return self.lo if self.hi == self.lo else float(rng.uniform(self.lo, self.hi))

    def contains(self, x: float, tol: float = 1e-9) -> bool:
        return self.lo - tol <= x <= self.hi + tol


@dataclass(frozen=True)
class Limits:
    peak_speed_max: float                       # m/s, max over episode of |w_t|
    vertical_max: float | None                  # m/s, max |w_z|; None → unconstrained
    mean_horizontal: Range                      # m/s, |mean(w_xy)|
    vertical_min: float | None = None           # m/s, min of max|w_z| (directional OOD)
    mean_elevation_min_deg: float | None = None # |elevation of mean wind| lower bound

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Limits":
        return cls(
            peak_speed_max=float(d["peak_speed_max"]),
            vertical_max=None if d.get("vertical_max") is None else float(d["vertical_max"]),
            mean_horizontal=Range.of(d.get("mean_horizontal", [0.0, 1e9])),
            vertical_min=None if d.get("vertical_min") is None else float(d["vertical_min"]),
            mean_elevation_min_deg=(None if d.get("mean_elevation_min_deg") is None
                                    else float(d["mean_elevation_min_deg"])),
        )


@dataclass(frozen=True)
class WindRegime:
    name: str
    category: str                               # id | ood_magnitude | ood_direction | ...
    mean_speed: Range                           # m/s, steady (mean) wind speed
    azimuth_deg: Range
    elevation_deg: tuple[Range, ...]            # one or more allowed bands
    profiles: tuple[tuple[str, float, dict], ...]   # (type, weight, params)
    limits: Limits
    max_tries: int = 200
    speed_strata: int = 0                       # >0: stratified mean speed, see ``speed_bands`` / ``assign_strata`` (0 = i.i.d. uniform)
    speed_strata_weights: tuple[float, ...] = ()   # relative share of the groups per band (empty = equal); higher bands usually get more

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WindRegime":
        elev = d.get("elevation_deg", [0.0, 0.0])
        bands = (tuple(Range.of(b) for b in elev) if isinstance(elev[0], (list, tuple))
                 else (Range.of(elev),))
        profiles = []
        for p in d["profiles"]:
            ptype = str(p["type"])
            if ptype not in PROFILE_TYPES:
                raise ValueError(f"unknown wind profile type {ptype!r} (allowed {PROFILE_TYPES})")
            profiles.append((ptype, float(p.get("weight", 1.0)), dict(p.get("params", {}))))
        if not profiles:
            raise ValueError("a wind regime needs at least one profile")
        return cls(
            name=str(d["name"]), category=str(d["category"]),
            mean_speed=Range.of(d.get("mean_speed", [0.0, 0.0])),
            azimuth_deg=Range.of(d.get("azimuth_deg", [0.0, 360.0])),
            elevation_deg=bands,
            profiles=tuple(profiles),
            limits=Limits.from_dict(d["limits"]),
            max_tries=int(d.get("max_tries", 200)),
            speed_strata=int(d.get("speed_strata", 0)),
            speed_strata_weights=tuple(float(x) for x in d.get("speed_strata_weights", ())),
        )

    def speed_bands(self) -> list[tuple[float, float]]:
        """``speed_strata`` equal-width bands covering ``mean_speed`` (empty when stratification is off)."""
        k = self.speed_strata
        lo, hi = self.mean_speed.lo, self.mean_speed.hi
        return [(lo + i * (hi - lo) / k, lo + (i + 1) * (hi - lo) / k) for i in range(k)] if k > 0 else []


def _largest_remainder(weights: Sequence[float], n: int) -> list[int]:
    w = np.asarray(weights, dtype=float)
    if w.shape[0] == 0 or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"speed_strata_weights must be non-negative with a positive sum, got {list(weights)}")
    ideal = w / w.sum() * n
    counts = np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal - counts), kind="stable")[: n - int(counts.sum())]:
        counts[i] += 1
    if n >= len(counts):                         # no empty band when there are enough groups: take the groups from the fullest band
        for i in np.where((counts == 0) & (w > 0))[0]:
            counts[int(np.argmax(counts))] -= 1
            counts[i] += 1
    return counts.tolist()


def assign_strata(n: int, k: int, rng: np.random.Generator, weights: Sequence[float] = ()) -> list[int]:
    """Band index of each of ``n`` groups. Without ``weights`` every band gets ``floor(n/k)`` or ``ceil(n/k)`` groups; with ``weights``
    (length k) the counts follow the weights by largest remainder (band i gets about n·w_i/Σw groups). Either way in a random order.

    Weighted bands: the counterfactual siblings (wind × 0.0 and × 0.5) pile up at low speed, so equal group counts still give most
    episodes below 1 m/s; more groups in the upper bands flatten the distribution of the episodes.

    Stratified (not i.i.d.) sampling of the episode-mean wind speed: with few groups per role, i.i.d. uniform draws leave bands empty
    by chance. Deterministic given ``rng`` (a child of the part's seed node), so the plan stays reproducible."""
    if weights:
        if len(weights) != k:
            raise ValueError(f"speed_strata_weights has {len(weights)} entries for {k} bands")
        bands = [i for i, c in enumerate(_largest_remainder(weights, n)) for _ in range(c)]
        return [bands[int(j)] for j in rng.permutation(n)]
    return [int(r) % k for r in rng.permutation(n)]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class WindPlan:
    """A concrete, reproducible wind realisation description."""

    regime: str
    category: str
    profile: str
    spec: dict[str, Any]                        # WindManager spec (concrete)
    seed: int
    stats: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"regime": self.regime, "category": self.category, "profile": self.profile,
                "spec": self.spec, "seed": self.seed, "stats": self.stats, "labels": self.labels}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def unit_vector(azimuth_deg: float, elevation_deg: float) -> list[float]:
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]


def _perp_horizontal(d: list[float]) -> list[float]:
    h = math.hypot(d[0], d[1])
    if h < 1e-9:
        return [1.0, 0.0, 0.0]
    return [-d[1] / h, d[0] / h, 0.0]


def _component_spec(ptype: str, params: Mapping[str, Any], mean_dir: list[float],
                    duration: float, rng: np.random.Generator,
                    extra: dict[str, Any] | None = None) -> tuple[list[dict], dict[str, str]]:
    """Return (gust spec list, labels) for one profile type.

    ``extra`` receives spec entries that are not gusts (``extra["turbulence"]`` for ``dryden``)."""
    r = lambda key, default: Range.of(params.get(key, default)).draw(rng)   # noqa: E731
    labels: dict[str, str] = {}
    if ptype in ("none", "steady"):
        return [], {"temporal_class": "constant"}
    if ptype == "dryden":
        # stationary Dryden turbulence (MIL-F-8785C form) with explicit intensities and length scales; the frozen-turbulence speed is the
        # episode's mean horizontal wind (wind_manager), so counterfactual pairs share realisations.
        if extra is None:
            raise ValueError("a dryden profile needs the spec to carry a turbulence entry")
        sig = r("sigma_uv_mps", [0.2, 0.6])
        extra["turbulence"] = {"enabled": True, "model": "dryden",
                               "intensities_mps": [sig, sig, sig * float(params.get("sigma_w_ratio", 0.5))],
                               "length_scales_m": [r("length_scale_uv_m", [10.0, 30.0])] * 2 + [r("length_scale_w_m", [2.0, 5.0])]}
        return [], {"temporal_class": "turbulent", "gust_class": "turbulence"}
    if ptype == "composite":
        # several components in one episode: e.g. a wind level that steps up and back down (a regime "switch") with an OU or Dryden
        # overlay. Components are added; each draws its own parameters from the same rng, in order (deterministic).
        subs = params.get("components")
        if not subs:
            raise ValueError("a composite profile needs a non-empty `components` list")
        gusts_all: list[dict] = []
        for sub in subs:
            st = str(sub["type"])
            if st in ("composite", "none"):
                raise ValueError(f"composite components cannot be {st!r}")
            g, lab = _component_spec(st, sub.get("params", {}), mean_dir, duration, rng, extra)
            gusts_all += g
            labels.update({k: v for k, v in lab.items() if k != "temporal_class"})
        labels["temporal_class"] = "composite"
        return gusts_all, labels
    if ptype == "cosine_gust":
        rise = r("rise_time_s", [0.5, 1.5])
        onset = r("onset_fraction", [0.1, 0.8]) * duration
        rel = params.get("direction", "mean")
        d = mean_dir if rel == "mean" else unit_vector(r("azimuth_deg", [0, 360]), 0.0)
        labels["gust_class"] = "impulsive" if rise < 0.2 else "smooth"
        labels["temporal_class"] = "sharp" if rise < 0.2 else "slow"
        return [{"enabled": True, "type": "cosine", "peak_mps": r("peak_mps", [0.5, 1.5]),
                 "start_time_s": onset, "rise_time_s": rise, "direction": d}], labels
    if ptype == "step":
        rise = r("rise_time_s", [0.5, 2.0])
        labels["temporal_class"] = "sharp" if rise < 0.2 else "slow"
        return [{"enabled": True, "type": "step", "before_mps": 0.0,
                 "after_mps": r("delta_mps", [-1.0, 1.0]),
                 "step_time_s": r("onset_fraction", [0.2, 0.7]) * duration,
                 "rise_time_s": rise, "direction": mean_dir}], labels
    if ptype == "sinusoid":
        f = r("frequency_hz", [0.05, 0.3])
        labels["temporal_class"] = "fast" if f > 0.5 else "slow"
        axis = params.get("axis", "mean")
        d = mean_dir if axis == "mean" else _perp_horizontal(mean_dir)
        return [{"enabled": True, "type": "sinusoidal", "amplitude_mps": r("amplitude_mps", [0.3, 1.0]),
                 "frequency_hz": f, "mean_mps": 0.0, "direction": d}], labels
    if ptype == "ou":
        tau = r("tau_s", [2.0, 5.0])
        labels["temporal_class"] = "fast" if tau < 1.0 else "slow"
        return [{"enabled": True, "type": "ou", "sigma_mps": r("sigma_mps", [0.2, 0.6]),
                 "tau_s": tau, "mean_mps": [0.0, 0.0, 0.0]}], labels
    if ptype == "random_gusts":
        rise = Range.of(params.get("rise_range_s", [0.3, 1.2]))
        peak = Range.of(params.get("peak_range_mps", [0.5, 1.5]))
        labels["gust_class"] = "impulsive" if rise.lo < 0.2 else "smooth"
        labels["temporal_class"] = "sharp" if rise.lo < 0.2 else "slow"
        return [{"enabled": True, "type": "random_events", "rate_hz": r("rate_hz", [0.05, 0.2]),
                 "peak_range_mps": [peak.lo, peak.hi],
                 "rise_range_s": [rise.lo, rise.hi], "duration_s": duration}], labels
    raise ValueError(ptype)


def evaluate_series(spec: Mapping[str, Any], seed: int, duration: float, dt: float,
                    position=(0.0, 0.0, 1.5)) -> np.ndarray:
    """Wind series (N+1, 3) at t_k = k·dt, exactly as the runner evaluates it."""
    mgr = WindManager(dict(spec), seed=seed).configure(seed=seed)
    n = int(round(duration / dt))
    pos = np.asarray(position, dtype=float)
    return np.stack([mgr.get_wind_state(pos, k * dt, dt).total_W for k in range(n + 1)])


def realised_stats(series: np.ndarray) -> dict[str, float]:
    mean = series.mean(axis=0)
    speed = np.linalg.norm(series, axis=1)
    mean_h = float(np.linalg.norm(mean[:2]))
    mean_norm = float(np.linalg.norm(mean))
    elev = math.degrees(math.atan2(mean[2], mean_h)) if mean_norm > 1e-9 else 0.0
    return {
        "mean_x": float(mean[0]), "mean_y": float(mean[1]), "mean_z": float(mean[2]),
        "mean_horizontal_speed": mean_h,
        "mean_speed": float(speed.mean()),
        "peak_speed": float(speed.max()),
        "rms_fluctuation": float(np.sqrt(((series - mean) ** 2).sum(axis=1).mean())),
        "max_abs_vertical": float(np.abs(series[:, 2]).max()),
        "mean_elevation_deg": float(elev),
    }


def check_limits(stats: Mapping[str, float], lim: Limits) -> list[str]:
    bad = []
    if stats["peak_speed"] > lim.peak_speed_max + 1e-9:
        bad.append(f"peak {stats['peak_speed']:.3f} > {lim.peak_speed_max}")
    if lim.vertical_max is not None and stats["max_abs_vertical"] > lim.vertical_max + 1e-9:
        bad.append(f"|w_z| {stats['max_abs_vertical']:.3f} > {lim.vertical_max}")
    if lim.vertical_min is not None and stats["max_abs_vertical"] < lim.vertical_min - 1e-9:
        bad.append(f"|w_z| {stats['max_abs_vertical']:.3f} < {lim.vertical_min}")
    if not lim.mean_horizontal.contains(stats["mean_horizontal_speed"]):
        bad.append(f"mean horizontal {stats['mean_horizontal_speed']:.3f} not in "
                   f"[{lim.mean_horizontal.lo}, {lim.mean_horizontal.hi}]")
    if (lim.mean_elevation_min_deg is not None
            and abs(stats["mean_elevation_deg"]) < lim.mean_elevation_min_deg - 1e-9):
        bad.append(f"|elevation| {abs(stats['mean_elevation_deg']):.1f} < {lim.mean_elevation_min_deg}")
    return bad


def sample_wind(regime: WindRegime, rng: np.random.Generator, *, duration: float, dt: float,
                profile: str | None = None, speed_range: tuple[float, float] | None = None) -> WindPlan:
    """Draw one concrete wind realisation satisfying ``regime.limits``.

    ``profile`` forces a profile type (used by experiment mixes); otherwise it
    is drawn by weight. ``speed_range`` restricts the draw of the mean speed to one stratum (uniform inside it);
    None keeps the regime's whole range. Raises ``RuntimeError`` if no valid realisation is found.
    """
    types = [p[0] for p in regime.profiles]
    weights = np.array([p[1] for p in regime.profiles], dtype=float)
    weights = weights / weights.sum()
    last: list[str] = []
    for _ in range(regime.max_tries):
        if profile is None:
            idx = int(rng.choice(len(types), p=weights))
        else:
            if profile not in types:
                raise ValueError(f"profile {profile!r} not offered by regime {regime.name!r}")
            idx = types.index(profile)
        ptype, _, params = regime.profiles[idx]
        if ptype == "none":
            speed = 0.0
        elif speed_range is not None:
            speed = float(rng.uniform(speed_range[0], speed_range[1]))
        else:
            speed = regime.mean_speed.draw(rng)
        band = regime.elevation_deg[int(rng.integers(len(regime.elevation_deg)))]
        direction = unit_vector(regime.azimuth_deg.draw(rng), band.draw(rng))
        extra: dict[str, Any] = {}
        gusts, labels = _component_spec(ptype, params, direction, duration, rng, extra)
        spec: dict[str, Any] = {"steady": {"enabled": speed > 0.0, "speed_mps": speed,
                                           "direction": direction}}
        if gusts:
            spec["gust"] = gusts
        spec.update(extra)                                  # e.g. "turbulence" for dryden profiles
        seed = int(rng.integers(0, 2**31 - 1))
        stats = realised_stats(evaluate_series(spec, seed, duration, dt))
        last = check_limits(stats, regime.limits)
        if not last:
            labels.setdefault("gust_class", "none")
            labels.setdefault("temporal_class", "constant")
            return WindPlan(regime=regime.name, category=regime.category, profile=ptype,
                            spec=spec, seed=seed, stats=stats, labels=labels)
    raise RuntimeError(f"regime {regime.name!r}: no valid realisation in {regime.max_tries} "
                       f"tries; last violations: {last}")


def scaled_plan(plan: WindPlan, scale: float, *, duration: float, dt: float) -> WindPlan:
    """Same realisation (same seed, components), amplitude × ``scale`` (counterfactual)."""
    spec = dict(plan.spec)
    ood = dict(spec.get("ood", {}))
    ood["wind_scale"] = float(ood.get("wind_scale", 1.0)) * float(scale)
    spec["ood"] = ood
    stats = realised_stats(evaluate_series(spec, plan.seed, duration, dt))
    return WindPlan(regime=plan.regime, category=plan.category, profile=plan.profile,
                    spec=spec, seed=plan.seed, stats=stats, labels=dict(plan.labels))


def rotated_plan(plan: WindPlan, angle_rad: float, *, duration: float, dt: float) -> WindPlan:
    """Same realisation rotated about world z by ``angle_rad`` (counterfactual)."""
    spec = dict(plan.spec)
    ood = dict(spec.get("ood", {}))
    ood["rotate_z_rad"] = float(ood.get("rotate_z_rad", 0.0)) + float(angle_rad)
    spec["ood"] = ood
    stats = realised_stats(evaluate_series(spec, plan.seed, duration, dt))
    return WindPlan(regime=plan.regime, category=plan.category, profile=plan.profile,
                    spec=spec, seed=plan.seed, stats=stats, labels=dict(plan.labels))
