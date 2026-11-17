"""Pure-time reference trajectories (plan §6.3-T4, T5, T8, T14).

A reference is a deterministic function of time, fully defined by:

* its ``family`` (hover, vertical, line, accel_decel, circle, helix,
  figure_eight, sinusoidal, waypoints, attitude_doublet, attitude_sine, …),
* a dict of **concrete** parameters (sampled once per scenario from the
  trajectory seed stream and stored in the manifest), and
* an ``Anchor`` (initial position and yaw).

Properties guaranteed by construction (tested):

* Every position reference starts **at** the anchor with zero velocity and
  zero acceleration (minimum-jerk ramps, no t=0 steps).
* Position is C³, so velocity/acceleration/jerk used for feed-forward and
  feasibility checks are continuous.
* Sinusoid parameters are fixed for the whole episode (no per-step resampling).

Velocity/acceleration/jerk are obtained by central finite differences of the
analytic position (h = 1e-3 s); for these C³ signals the truncation error is
O(h²·p⁽⁵⁾) ≈ 1e-6 relative, negligible for feed-forward and feasibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

_H = 1e-3

POSITION_FAMILIES = ("hover", "vertical", "line", "accel_decel", "circle", "helix",
                     "figure_eight", "sinusoidal", "waypoints")
ATTITUDE_FAMILIES = ("attitude_doublet", "attitude_sine", "attitude_combined")
YAW_PROFILES = ("constant", "step", "sine", "rate")


# ---------------------------------------------------------------------------
# Smooth primitives
# ---------------------------------------------------------------------------


def smoothstep(x: float) -> float:
    """Minimum-jerk transition 0→1 on x∈[0,1] (C², zero vel/acc at ends)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def smoothstep_integral(x: float) -> float:
    """∫₀ˣ smoothstep(u) du (= 0.5 at x = 1); linear continuation for x > 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 0.5 + (x - 1.0)
    return 2.5 * x ** 4 - 3.0 * x ** 5 + x ** 6


def ramped_phase(t: float, rate: float, ramp: float) -> float:
    """Phase whose derivative ramps smoothly from 0 to ``rate`` over ``ramp`` s."""
    if ramp <= 0.0:
        return rate * t
    return rate * ramp * smoothstep_integral(t / ramp)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    position: tuple[float, float, float]
    yaw: float = 0.0

    @property
    def p(self) -> np.ndarray:
        return np.asarray(self.position, dtype=float)


@dataclass
class RefPoint:
    """Reference at one instant. ``None`` fields are not commanded."""

    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    jerk: np.ndarray | None = None
    yaw: float | None = None
    yaw_rate: float | None = None
    roll: float | None = None
    pitch: float | None = None


@dataclass(frozen=True)
class Reference:
    family: str
    params: dict[str, Any]
    anchor: Anchor
    duration: float
    yaw_profile: dict[str, Any] = field(default_factory=lambda: {"type": "constant"})

    # -- kind -------------------------------------------------------------
    @property
    def kind(self) -> str:
        return "attitude" if self.family in ATTITUDE_FAMILIES else "position"

    # -- evaluation ---------------------------------------------------------
    def position(self, t: float) -> np.ndarray:
        if self.kind != "position":
            raise TypeError(f"{self.family} has no position reference")
        return _POSITION_FNS[self.family](self.params, self.anchor, t)

    def yaw(self, t: float) -> float:
        return _yaw(self.yaw_profile, self.anchor.yaw, t)

    def attitude(self, t: float) -> tuple[float, float]:
        if self.kind != "attitude":
            raise TypeError(f"{self.family} has no attitude reference")
        return _ATTITUDE_FNS[self.family](self.params, t)

    def at(self, t: float) -> RefPoint:
        return evaluate_point(self, t)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "params": self.params,
                "anchor": {"position": list(self.anchor.position), "yaw": self.anchor.yaw},
                "duration": self.duration, "yaw_profile": self.yaw_profile}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Reference":
        a = d["anchor"]
        return cls(family=d["family"], params=dict(d["params"]),
                   anchor=Anchor(tuple(a["position"]), float(a["yaw"])),
                   duration=float(d["duration"]), yaw_profile=dict(d["yaw_profile"]))


def evaluate_point(ref: Any, t: float) -> RefPoint:
    """Evaluate any reference-like object (``kind``, ``position``, ``yaw``,
    ``attitude``) at ``t`` with central finite differences for derivatives."""
    yaw = ref.yaw(t)
    yaw_rate = (ref.yaw(t + _H) - ref.yaw(t - _H)) / (2 * _H)
    if ref.kind == "attitude":
        roll, pitch = ref.attitude(t)
        return RefPoint(roll=roll, pitch=pitch, yaw=yaw, yaw_rate=yaw_rate)
    p = ref.position
    p0, pp, pm = p(t), p(t + _H), p(t - _H)
    pp2, pm2 = p(t + 2 * _H), p(t - 2 * _H)
    v = (pp - pm) / (2 * _H)
    a = (pp - 2 * p0 + pm) / _H ** 2
    j = (pp2 - 2 * pp + 2 * pm - pm2) / (2 * _H ** 3)
    return RefPoint(position=p0, velocity=v, acceleration=a, jerk=j, yaw=yaw, yaw_rate=yaw_rate)


# ---------------------------------------------------------------------------
# Position families (anchor-relative; p(0) = anchor, v(0) = a(0) = 0)
# ---------------------------------------------------------------------------


def _dir(azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def _move_dir(p: Mapping[str, Any], a: Anchor) -> np.ndarray:
    """World azimuth (``azimuth_deg``) or body-relative heading (``heading_offset_deg``:
    0 forward, 180 backward, ±90 lateral) from the anchor yaw."""
    if "heading_offset_deg" in p:
        az = math.degrees(a.yaw) + float(p["heading_offset_deg"])
    else:
        az = float(p["azimuth_deg"])
    return _dir(az, p.get("elevation_deg", 0.0))


def _hover(p, a: Anchor, t):
    return a.p.copy()


def _vertical(p, a: Anchor, t):
    s = smoothstep((t - p["start_s"]) / p["move_s"])
    return a.p + np.array([0.0, 0.0, p["dz_m"] * s])


def _cruise_distance(t: float, v: float, ramp: float) -> float:
    return v * ramp * smoothstep_integral(t / ramp)


def _line(p, a: Anchor, t):
    d = _move_dir(p, a)
    return a.p + d * _cruise_distance(t, p["speed_mps"], p["ramp_s"])


def _accel_decel(p, a: Anchor, t):
    """Accelerate (ramp) → cruise → decelerate (ramp) → hold."""
    v, ramp, cruise = p["speed_mps"], p["ramp_s"], p["cruise_s"]
    d = _move_dir(p, a)
    t_dec = ramp + cruise
    up = _cruise_distance(min(t, t_dec), v, ramp)
    if t <= t_dec:
        return a.p + d * up
    # deceleration mirrors the acceleration: distance = v·τ − v·ramp·S∫(τ/ramp)
    tau = min(t - t_dec, ramp)
    down = v * tau - v * ramp * smoothstep_integral(tau / ramp)
    return a.p + d * (up + down)


def _circle(p, a: Anchor, t):
    R, w, ramp, th0 = p["radius_m"], p["omega_radps"], p["ramp_s"], math.radians(p["start_angle_deg"])
    center = a.p - R * np.array([math.cos(th0), math.sin(th0), 0.0])
    th = th0 + ramped_phase(t, w, ramp)
    return center + R * np.array([math.cos(th), math.sin(th), 0.0])


def _helix(p, a: Anchor, t):
    base = _circle(p, a, t)
    base[2] += _cruise_distance(t, p["climb_rate_mps"], p["ramp_s"])
    return base


def _figure_eight(p, a: Anchor, t):
    """Lemniscate of Gerono, rotated by ``azimuth_deg``; starts at its centre."""
    phi = ramped_phase(t, 2 * math.pi * p["frequency_hz"], p["ramp_s"])
    x = p["amp_x_m"] * math.sin(phi)
    y = p["amp_y_m"] * math.sin(phi) * math.cos(phi)
    c, s = math.cos(math.radians(p["azimuth_deg"])), math.sin(math.radians(p["azimuth_deg"]))
    return a.p + np.array([c * x - s * y, s * x + c * y, 0.0])


def _sinusoidal(p, a: Anchor, t):
    phi = ramped_phase(t, 2 * math.pi * p["frequency_hz"], p["ramp_s"])
    d = _move_dir(p, a)
    return a.p + d * p["amplitude_m"] * math.sin(phi)


def _waypoints(p, a: Anchor, t):
    """Min-jerk segments between anchor-relative waypoints (each segment rest-to-rest)."""
    pts = [np.zeros(3)] + [np.asarray(w, dtype=float) for w in p["offsets_m"]]
    seg = p["segment_s"]
    k = int(min(t // seg, len(pts) - 1)) if t > 0 else 0
    if k >= len(pts) - 1:
        return a.p + pts[-1]
    s = smoothstep((t - k * seg) / seg)
    return a.p + pts[k] + (pts[k + 1] - pts[k]) * s


_POSITION_FNS: dict[str, Callable] = {
    "hover": _hover, "vertical": _vertical, "line": _line, "accel_decel": _accel_decel,
    "circle": _circle, "helix": _helix, "figure_eight": _figure_eight,
    "sinusoidal": _sinusoidal, "waypoints": _waypoints,
}


# ---------------------------------------------------------------------------
# Attitude families (roll, pitch in rad); short manoeuvres around level
# ---------------------------------------------------------------------------


def _doublet(t: float, amp: float, start: float, hold: float, edge: float) -> float:
    """+amp for ``hold`` s, then −amp for ``hold`` s, then 0; smooth edges of ``edge`` s."""
    up = smoothstep((t - start) / edge)
    flip = smoothstep((t - start - hold) / edge)
    down = smoothstep((t - start - 2 * hold) / edge)
    return amp * (up - 2 * flip + down)


def _attitude_doublet(p, t):
    amp = math.radians(p["amplitude_deg"])
    val = _doublet(t, amp, p["start_s"], p["hold_s"], p["edge_s"])
    return (val, 0.0) if p["axis"] == "roll" else (0.0, val)


def _attitude_sine(p, t):
    amp = math.radians(p["amplitude_deg"]) * smoothstep(t / p["ramp_s"])
    val = amp * math.sin(2 * math.pi * p["frequency_hz"] * t)
    return (val, 0.0) if p["axis"] == "roll" else (0.0, val)


def _attitude_combined(p, t):
    env = smoothstep(t / p["ramp_s"])
    w = 2 * math.pi * p["frequency_hz"]
    return (math.radians(p["roll_amplitude_deg"]) * env * math.sin(w * t),
            math.radians(p["pitch_amplitude_deg"]) * env * math.sin(w * t + math.radians(p["phase_deg"])))


_ATTITUDE_FNS: dict[str, Callable] = {
    "attitude_doublet": _attitude_doublet, "attitude_sine": _attitude_sine,
    "attitude_combined": _attitude_combined,
}


# ---------------------------------------------------------------------------
# Yaw profiles
# ---------------------------------------------------------------------------


def _yaw(profile: Mapping[str, Any], yaw0: float, t: float) -> float:
    kind = profile.get("type", "constant")
    if kind == "constant":
        return yaw0
    if kind == "step":
        return yaw0 + math.radians(profile["delta_deg"]) * smoothstep(
            (t - profile["start_s"]) / profile["transition_s"])
    if kind == "sine":
        env = smoothstep(t / profile["ramp_s"])
        return yaw0 + math.radians(profile["amplitude_deg"]) * env * math.sin(
            2 * math.pi * profile["frequency_hz"] * t)
    if kind == "rate":
        return yaw0 + ramped_phase(t, math.radians(profile["rate_degps"]), profile["ramp_s"])
    raise ValueError(f"unknown yaw profile {kind!r}")


# ---------------------------------------------------------------------------
# Parameter sampling from scenario templates
# ---------------------------------------------------------------------------


def _draw(v: Any, rng: np.random.Generator) -> Any:
    """Concrete value from a template entry: scalar, [lo, hi] range, or {choice: [...]}."""
    if isinstance(v, Mapping) and "choice" in v:
        options = list(v["choice"])
        return options[int(rng.integers(len(options)))]
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return float(rng.uniform(float(v[0]), float(v[1])))
    if isinstance(v, (list, tuple)):
        return [_draw(x, rng) for x in v]
    return v


def sample_params(template: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Draw concrete parameters (keys iterated in sorted order for determinism)."""
    out: dict[str, Any] = {}
    for key in sorted(template):
        val = template[key]
        if key == "offsets_m":          # waypoints: n points within a box
            n = int(_draw(val["count"], rng))
            box = np.asarray(val["box_m"], dtype=float)
            out[key] = [list(rng.uniform(-box, box)) for _ in range(n)]
            continue
        out[key] = _draw(val, rng)
    return out


def make_reference(family: str, params_template: Mapping[str, Any],
                   yaw_template: Mapping[str, Any], anchor: Anchor, duration: float,
                   rng: np.random.Generator) -> Reference:
    if family not in _POSITION_FNS and family not in _ATTITUDE_FNS:
        raise ValueError(f"unknown trajectory family {family!r}")
    params = sample_params(params_template, rng)
    yaw_profile = {"type": "constant"}
    if yaw_template:
        yaw_profile = {"type": yaw_template.get("type", "constant"),
                       **sample_params({k: v for k, v in yaw_template.items() if k != "type"}, rng)}
        if yaw_profile["type"] not in YAW_PROFILES:
            raise ValueError(f"unknown yaw profile {yaw_profile['type']!r}")
    return Reference(family=family, params=params, anchor=anchor, duration=float(duration),
                     yaw_profile=yaw_profile)
