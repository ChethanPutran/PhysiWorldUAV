"""Plan-time feasibility of reference trajectories (plan §6.3-T13).

For a position reference, quadrotor differential flatness gives (nominal,
no drag, no wind):

    f(t)  = a(t) + g e₃                 (specific thrust vector, m/s²)
    T(t)  = m ‖f(t)‖                     required collective thrust
    tilt  = ∠(f, e₃)                     required tilt
    ‖ω_xy‖ ≈ ‖ j_⊥ ‖ / ‖f‖                tilt rate, j_⊥ = j − (j·f̂) f̂

These are *necessary* conditions checked with a safety margin, because wind
and drag increase the real demand; run-time saturation is logged separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from physiworld_uav.drones.params import G, DroneParams

from .library import Reference


@dataclass(frozen=True)
class FeasibilityLimits:
    thrust_fraction: tuple[float, float] = (0.15, 0.85)   # of T_max
    max_tilt_deg: float = 35.0
    max_tilt_rate_radps: float = 3.0
    max_yaw_rate_radps: float = 2.0
    max_speed_mps: float = 5.0
    max_accel_mps2: float = 5.0
    max_jerk_mps3: float = 30.0
    workspace_min: tuple[float, float, float] = (-30.0, -30.0, 0.5)
    workspace_max: tuple[float, float, float] = (30.0, 30.0, 10.0)
    sample_hz: float = 100.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "FeasibilityLimits":
        if not d:
            return cls()
        kw = dict(d)
        for k in ("thrust_fraction", "workspace_min", "workspace_max"):
            if k in kw:
                kw[k] = tuple(float(x) for x in kw[k])
        return cls(**kw)


@dataclass
class FeasibilityReport:
    feasible: bool
    violations: list[str] = field(default_factory=list)
    peaks: dict[str, float] = field(default_factory=dict)


def check(ref: Reference, drone: DroneParams, limits: FeasibilityLimits) -> FeasibilityReport:
    ts = np.arange(0.0, ref.duration + 1e-9, 1.0 / limits.sample_hz)
    v = []
    peaks: dict[str, float] = {}
    yaw = np.array([ref.yaw(t) for t in ts])
    yaw_rate = np.abs(np.gradient(yaw, ts)) if ts.size > 1 else np.zeros(1)
    peaks["yaw_rate"] = float(yaw_rate.max())
    if peaks["yaw_rate"] > limits.max_yaw_rate_radps:
        v.append(f"yaw rate {peaks['yaw_rate']:.2f} > {limits.max_yaw_rate_radps} rad/s")

    if ref.kind == "attitude":
        att = np.array([ref.attitude(t) for t in ts])
        tilt = np.degrees(np.arccos(np.clip(np.cos(att[:, 0]) * np.cos(att[:, 1]), -1, 1)))
        rates = np.abs(np.gradient(att, ts, axis=0)).max(axis=1) if ts.size > 1 else np.zeros(1)
        peaks.update(tilt_deg=float(tilt.max()), tilt_rate=float(rates.max()))
        if peaks["tilt_deg"] > limits.max_tilt_deg:
            v.append(f"tilt {peaks['tilt_deg']:.1f} > {limits.max_tilt_deg} deg")
        if peaks["tilt_rate"] > limits.max_tilt_rate_radps:
            v.append(f"attitude rate {peaks['tilt_rate']:.2f} > {limits.max_tilt_rate_radps} rad/s")
        return FeasibilityReport(feasible=not v, violations=v, peaks=peaks)

    pts = [ref.at(t) for t in ts]
    p = np.array([q.position for q in pts])
    vel = np.array([q.velocity for q in pts])
    acc = np.array([q.acceleration for q in pts])
    jerk = np.array([q.jerk for q in pts])
    f = acc + np.array([0.0, 0.0, G])
    fn = np.linalg.norm(f, axis=1)
    thrust = drone.mass * fn
    tilt = np.degrees(np.arccos(np.clip(f[:, 2] / np.maximum(fn, 1e-9), -1, 1)))
    fhat = f / np.maximum(fn, 1e-9)[:, None]
    j_perp = jerk - (jerk * fhat).sum(axis=1, keepdims=True) * fhat
    tilt_rate = np.linalg.norm(j_perp, axis=1) / np.maximum(fn, 1e-9)

    peaks.update(
        thrust_min_frac=float(thrust.min() / drone.max_thrust),
        thrust_max_frac=float(thrust.max() / drone.max_thrust),
        tilt_deg=float(tilt.max()), tilt_rate=float(tilt_rate.max()),
        speed=float(np.linalg.norm(vel, axis=1).max()),
        accel=float(np.linalg.norm(acc, axis=1).max()),
        jerk=float(np.linalg.norm(jerk, axis=1).max()),
    )
    lo, hi = limits.thrust_fraction
    if peaks["thrust_min_frac"] < lo or peaks["thrust_max_frac"] > hi:
        v.append(f"thrust fraction [{peaks['thrust_min_frac']:.2f}, {peaks['thrust_max_frac']:.2f}] "
                 f"outside [{lo}, {hi}]")
    for key, lim, unit in (("tilt_deg", limits.max_tilt_deg, "deg"),
                           ("tilt_rate", limits.max_tilt_rate_radps, "rad/s"),
                           ("speed", limits.max_speed_mps, "m/s"),
                           ("accel", limits.max_accel_mps2, "m/s²"),
                           ("jerk", limits.max_jerk_mps3, "m/s³")):
        if peaks[key] > lim:
            v.append(f"{key} {peaks[key]:.2f} > {lim} {unit}")
    lo_ws, hi_ws = np.asarray(limits.workspace_min), np.asarray(limits.workspace_max)
    if np.any(p < lo_ws) or np.any(p > hi_ws):
        v.append(f"leaves workspace: min {p.min(axis=0).round(2).tolist()} max {p.max(axis=0).round(2).tolist()}")
    return FeasibilityReport(feasible=not v, violations=v, peaks=peaks)
