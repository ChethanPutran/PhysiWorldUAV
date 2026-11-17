"""Bounded perturbations added to tracking references (plan §6.3-T10; design doc §11.2 2–4, 9).

``PerturbedReference`` wraps a position reference and adds signals on the
channels ``x``, ``y``, ``z`` (metres, world frame) and ``yaw`` (degrees).
Velocity/acceleration/jerk feed-forward of the perturbation come from the
same finite-difference evaluation as the base reference, so the controller
receives a consistent (p, v, a, yaw) target.

Signals fed to a tracking controller should use ``edge_s > 0`` (smooth edges);
``sample_signal`` keeps whatever the template specifies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from physiworld_uav.trajectories.library import RefPoint, Reference, evaluate_point

from . import signals as S

CHANNELS = ("x", "y", "z", "yaw")


@dataclass(frozen=True)
class PerturbedReference:
    base: Reference
    channels: dict[str, dict[str, Any]]           # channel -> concrete signal params

    def __post_init__(self) -> None:
        bad = set(self.channels) - set(CHANNELS)
        if bad:
            raise ValueError(f"unknown perturbation channels {sorted(bad)}")
        if self.base.kind != "position" and set(self.channels) - {"yaw"}:
            raise ValueError("position perturbations require a position reference")

    @property
    def kind(self) -> str:
        return self.base.kind

    @property
    def duration(self) -> float:
        return self.base.duration

    @property
    def anchor(self):
        return self.base.anchor

    def _sig(self, ch: str, t: float) -> float:
        sig = self.channels.get(ch)
        return 0.0 if sig is None else S.evaluate(sig, t)

    def position(self, t: float) -> np.ndarray:
        return self.base.position(t) + np.array([self._sig("x", t), self._sig("y", t), self._sig("z", t)])

    def yaw(self, t: float) -> float:
        return self.base.yaw(t) + math.radians(self._sig("yaw", t))

    def attitude(self, t: float) -> tuple[float, float]:
        return self.base.attitude(t)

    def at(self, t: float) -> RefPoint:
        return evaluate_point(self, t)

    def to_dict(self) -> dict[str, Any]:
        return {**self.base.to_dict(), "perturbation": self.channels}


def perturb(base: Reference, templates: Mapping[str, Mapping[str, Any]] | None,
            rng: np.random.Generator) -> Reference | PerturbedReference:
    """Add sampled perturbations. Every channel must start at zero with zero slope
    (use ``ramp_s``, ``edge_s`` or ``start_s > 0``) so the reference stays anchored
    to the initial state (plan T5); otherwise a ``ValueError`` is raised."""
    if not templates:
        return base
    channels = S.sample_channels(templates, rng, base.duration)
    h = 1e-3
    for ch, sig in channels.items():
        v0, v1 = S.evaluate(sig, 0.0), S.evaluate(sig, h)
        if abs(v0) > 1e-9 or abs(v1 - v0) / h > 1e-2 * max(1.0, abs(float(sig.get("amplitude", 0.0)))):
            raise ValueError(f"perturbation channel {ch!r} ({sig['type']}) does not start at rest; "
                             f"add ramp_s / edge_s / start_s")
    return PerturbedReference(base, channels)


def reference_from_dict(d: Mapping[str, Any]) -> Reference | PerturbedReference:
    base = Reference.from_dict(d)
    pert = d.get("perturbation")
    return PerturbedReference(base, dict(pert)) if pert else base
