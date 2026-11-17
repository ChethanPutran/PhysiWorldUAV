"""Bounded excitation signals (plan §6.3-T10/T11; design doc §11.2).

Every signal is a deterministic function of time defined by *concrete*
parameters. ``sample_signal`` turns a template (ranges) into concrete
parameters with the excitation seed stream — including PRBS bit sequences and
multisine phases — so the manifest fully specifies the excitation and
counterfactual siblings replay it exactly.

Signal types:
  zero, step, pulse, doublet, multisine (Schroeder phases), chirp (linear/log),
  prbs (clocked ±1 sequence), smooth_random (random-phase band-limited sum).

``edge_s`` > 0 replaces discontinuities by minimum-jerk edges (use for
references fed to a tracking controller); ``edge_s = 0`` keeps ideal edges
(use for actuator-level open-loop inputs, where the motor lag smooths them).
All signals are bounded by construction: |s(t)| ≤ ``amplitude``.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from physiworld_uav.trajectories.library import _draw, smoothstep

SIGNAL_TYPES = ("zero", "step", "pulse", "doublet", "multisine", "chirp", "prbs", "smooth_random")


def _edge(t: float, t0: float, edge: float) -> float:
    if edge <= 0.0:
        return 1.0 if t >= t0 else 0.0
    return smoothstep((t - t0) / edge)


def sample_signal(template: Mapping[str, Any], rng: np.random.Generator, duration: float) -> dict[str, Any]:
    """Concrete parameters for one signal (deterministic given ``rng``)."""
    kind = str(template.get("type", "zero"))
    if kind not in SIGNAL_TYPES:
        raise ValueError(f"unknown signal type {kind!r}")
    p: dict[str, Any] = {"type": kind}
    for key in sorted(k for k in template if k != "type"):
        p[key] = _draw(template[key], rng)
    p.setdefault("amplitude", 0.0)
    p.setdefault("edge_s", 0.0)
    if kind == "multisine":
        n = int(p.get("n_tones", 8))
        f = np.linspace(float(p.get("f_min_hz", 0.1)), float(p.get("f_max_hz", 2.0)), n)
        k = np.arange(1, n + 1)
        p["freqs_hz"] = f.tolist()
        p["phases"] = (-math.pi * k * (k - 1) / n).tolist()     # Schroeder: low crest factor
    elif kind == "prbs":
        clock = float(p.get("clock_s", 0.5))
        # an edge longer than the clock would restart mid-transition (discontinuity)
        p["edge_s"] = min(float(p["edge_s"]), clock)
        nbits = int(math.ceil(duration / clock)) + 1
        p["bits"] = (2 * rng.integers(0, 2, size=nbits) - 1).tolist()
    elif kind == "smooth_random":
        n = int(p.get("n_tones", 12))
        p["freqs_hz"] = rng.uniform(float(p.get("f_min_hz", 0.05)), float(p.get("f_max_hz", 1.0)), n).tolist()
        p["phases"] = rng.uniform(0, 2 * math.pi, n).tolist()
    return p


def evaluate(sig: Mapping[str, Any], t: float) -> float:
    kind = sig["type"]
    A = float(sig.get("amplitude", 0.0))
    edge = float(sig.get("edge_s", 0.0))
    t0 = float(sig.get("start_s", 0.0))
    if kind == "zero" or A == 0.0:
        return 0.0
    if kind == "step":
        return A * _edge(t, t0, edge)
    if kind == "pulse":
        w = float(sig["width_s"])
        return A * (_edge(t, t0, edge) - _edge(t, t0 + w, edge))
    if kind == "doublet":
        w = float(sig["width_s"])
        return A * (_edge(t, t0, edge) - 2 * _edge(t, t0 + w, edge) + _edge(t, t0 + 2 * w, edge))
    tau = t - t0
    if tau < 0:
        return 0.0
    # Oscillatory signals start at t0 and fade in over ``ramp_s`` (min-jerk envelope).
    ramp = float(sig.get("ramp_s", 0.0))
    env = smoothstep(tau / ramp) if ramp > 0.0 else 1.0
    if kind == "multisine":
        f = np.asarray(sig["freqs_hz"])
        ph = np.asarray(sig["phases"])
        s = np.cos(2 * np.pi * f * tau + ph).sum()
        return A * env * float(s) / len(f)                        # |s| ≤ A
    if kind == "chirp":
        f0, f1, T = float(sig["f0_hz"]), float(sig["f1_hz"]), float(sig["sweep_s"])
        tt = min(tau, T)
        if sig.get("method", "linear") == "log":
            k = (f1 / f0) ** (1.0 / T)
            phase = 2 * math.pi * f0 * (k ** tt - 1.0) / math.log(k)
        else:
            phase = 2 * math.pi * (f0 * tt + 0.5 * (f1 - f0) / T * tt * tt)
        return A * env * math.sin(phase) if tau <= T else 0.0
    if kind == "prbs":
        clock = float(sig["clock_s"])
        bits = sig["bits"]
        k = min(int(tau // clock), len(bits) - 1)
        level = bits[k]
        prev = bits[k - 1] if k > 0 else 0          # the sequence starts from zero at t0
        if edge > 0.0 and prev != level:
            frac = smoothstep((tau - k * clock) / edge)
            return A * (prev + (level - prev) * frac)
        return A * level
    if kind == "smooth_random":
        f = np.asarray(sig["freqs_hz"])
        ph = np.asarray(sig["phases"])
        return A * env * float(np.sin(2 * np.pi * f * tau + ph).sum()) / len(f)
    raise ValueError(kind)


def sample_channels(templates: Mapping[str, Mapping[str, Any]], rng: np.random.Generator,
                    duration: float) -> dict[str, dict[str, Any]]:
    """Sample one signal per named channel (channels iterated in sorted order)."""
    return {ch: sample_signal(templates[ch], rng, duration) for ch in sorted(templates)}
