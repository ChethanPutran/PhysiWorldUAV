"""Inject project forces into OmniDrones' single base-link force call (plan §5-A8, A10; §6.1-F1).

OmniDrones applies rotor reaction torque (and its own drag/downwash buffer)
on the base link with exactly one call per ``apply_action``::

    self.base_link.apply_forces_and_torques_at_pos(forces, torques, is_global=True)

A second call on the same view in the same physics step may *overwrite* the
first on the GPU tensor API (measured in Run 4, see docs/models/force_accounting.md).
``WrenchInjector`` therefore replaces ``drone.base_link`` with a proxy that adds
the project's external wrench to OmniDrones' buffers inside that single call —
correct under both overwrite and accumulate semantics, without editing OmniDrones.

The external wrench must be set every physics step (``set``); it is cleared after
being forwarded so a stale force can never be applied twice. The last wrench
actually forwarded is kept for logging (``last_applied``), so logged forces are
exactly the applied ones (plan F9).
"""
from __future__ import annotations

from typing import Any

import torch


class InjectionError(RuntimeError):
    pass


class WrenchInjector:
    """Proxy around a ``RigidPrimView``; all other attributes are delegated."""

    def __init__(self, view: Any, n_bodies: int) -> None:
        if not hasattr(view, "apply_forces_and_torques_at_pos"):
            raise InjectionError(f"{type(view).__name__} has no apply_forces_and_torques_at_pos")
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "_n", int(n_bodies))
        object.__setattr__(self, "_force_W", None)
        object.__setattr__(self, "_torque_W", None)
        object.__setattr__(self, "last_applied", None)
        object.__setattr__(self, "calls", 0)

    # -- delegation -----------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_view"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._view, name, value)

    # -- API --------------------------------------------------------------------
    def set(self, force_W: torch.Tensor, torque_W: torch.Tensor) -> None:
        """World-frame force [N] and torque [N m] for the next physics step, shape (n, 3)."""
        f = torch.as_tensor(force_W, dtype=torch.float32).reshape(self._n, 3)
        t = torch.as_tensor(torque_W, dtype=torch.float32).reshape(self._n, 3)
        if not (torch.isfinite(f).all() and torch.isfinite(t).all()):
            raise InjectionError("non-finite external wrench")
        object.__setattr__(self, "_force_W", f)
        object.__setattr__(self, "_torque_W", t)

    def apply_forces_and_torques_at_pos(self, forces=None, torques=None, positions=None,
                                        indices=None, is_global: bool = True):
        if not is_global:
            raise InjectionError("OmniDrones base-link call expected with is_global=True")
        if positions is not None:
            raise InjectionError("external wrench injection assumes forces at the body origin")
        dev = (forces if forces is not None else torques).device
        f_ext = self._force_W if self._force_W is not None else torch.zeros(self._n, 3)
        t_ext = self._torque_W if self._torque_W is not None else torch.zeros(self._n, 3)
        f_ext, t_ext = f_ext.to(dev), t_ext.to(dev)
        forces = f_ext if forces is None else forces.reshape(self._n, 3) + f_ext
        torques = t_ext if torques is None else torques.reshape(self._n, 3) + t_ext
        object.__setattr__(self, "last_applied", (f_ext.clone(), t_ext.clone()))
        object.__setattr__(self, "_force_W", None)
        object.__setattr__(self, "_torque_W", None)
        object.__setattr__(self, "calls", self.calls + 1)
        return self._view.apply_forces_and_torques_at_pos(forces, torques, is_global=True,
                                                          indices=indices)


def install(drone: Any) -> WrenchInjector:
    """Replace ``drone.base_link`` by an injector (idempotent)."""
    if isinstance(drone.base_link, WrenchInjector):
        return drone.base_link
    n = int(torch.as_tensor(drone.base_link.count if hasattr(drone.base_link, "count") else 1))
    inj = WrenchInjector(drone.base_link, n)
    drone.base_link = inj
    return inj
