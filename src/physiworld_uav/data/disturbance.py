"""Disturbance targets for analysis (probe) and oracle models.

Two targets (decision: use both, ``.idea/choices.md`` #2):

1. **Total injected aero wrench** (recorded): ``ground_truth/wrench/external_force_W`` (world, N) and ``external_torque_B``
   (body, N·m), the interval mean of what the force models (rotor drag, body drag, …) applied during model step j. This is what
   ``f_phys`` does not contain. It is **not** wind only: drag also acts with zero wind (it depends on the vehicle's own velocity).
2. **Wind-attributable part** (reconstructed here): total(wind) − total(zero wind), evaluated with the same force models, drone
   parameters and domain factors as the generator, at the state and rotor speeds of the model step.

The reconstruction works at the **model rate**: state at the start of the interval, mean rotor speed ``u_j`` and the interval-mean
wind, whereas the generator evaluates the wrench at the 5 ms physics rate. Both the "with wind" and "zero wind" terms use the same
approximation, so most of the error cancels in the difference. ``reconstruction_error`` measures how well the with-wind
reconstruction reproduces the logged total (relative error of the force norm), which bounds the quality of the wind-attributable part.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from physiworld_uav.core.state_extractor import ANGVEL, POS, QUAT, VEL
from physiworld_uav.drones.params import DroneParams
from physiworld_uav.forces.system import AeroForceSystem
from physiworld_uav.models import quat
from physiworld_uav.randomization.parameter_sampler import Domain

TOTAL_FORCE = "ground_truth/wrench/external_force_W"
TOTAL_TORQUE = "ground_truth/wrench/external_torque_B"
WIND_FORCE = "derived/wind_attrib_force_W"
WIND_TORQUE = "derived/wind_attrib_torque_B"
RECON_FORCE = "derived/recon_total_force_W"
WIND_APPLIED = "environment/wind_applied_W"
WIND_ROTOR = "environment/wind_rotor_W"


def nominal_drone(path: str | Path | None = None) -> DroneParams:
    return DroneParams.from_yaml(path or Path(__file__).resolve().parents[3] / "config" / "drone" / "hummingbird.yaml")


def _sim_state(states: np.ndarray) -> torch.Tensor:
    """Model state p|v|q|ω_B (13) → simulator layout (p|q|v|ω_W) used by ``AeroForceSystem.compute``."""
    s = torch.as_tensor(states, dtype=torch.float32)
    q = s[:, 6:10]
    w_W = torch.einsum("nij,nj->ni", quat.to_rotation_matrix(quat.normalize(q)), s[:, 10:13])
    out = torch.zeros_like(s)
    out[:, POS], out[:, QUAT], out[:, VEL], out[:, ANGVEL] = s[:, 0:3], q, s[:, 3:6], w_W
    return out


def aero_system(ep, nominal: DroneParams | None = None) -> AeroForceSystem | None:
    plan = ep.meta.plan
    forces = plan.get("forces") or {}
    if forces.get("name", "none") == "none":
        return None
    nominal = nominal or nominal_drone()
    dom = plan["domain"]
    drone = Domain(domain_id=dom["domain_id"], region=dom["region"], factors=dom["factors"]).drone(nominal)
    return AeroForceSystem.from_config(forces["config"], drone, nominal, dom["factors"])


@torch.no_grad()
def reconstruct(ep, nominal: DroneParams | None = None) -> dict[str, np.ndarray]:
    """Per model step j (m rows): reconstructed total and wind-attributable force_W / torque_B."""
    aero = aero_system(ep, nominal)
    m = len(ep.actions)
    if aero is None:
        z = np.zeros((m, 3), np.float32)
        return {"total_force_W": z, "total_torque_B": z, "wind_force_W": z, "wind_torque_B": z}
    state = _sim_state(ep.states[:m])
    omega = torch.as_tensor(ep.actions, dtype=torch.float32)                          # rotor_speed_mean (interval mean)
    w_cm = torch.as_tensor(np.asarray(ep.aux[WIND_APPLIED])[:m], dtype=torch.float32)
    w_rot = (torch.as_tensor(np.asarray(ep.aux[WIND_ROTOR])[:m], dtype=torch.float32) if WIND_ROTOR in ep.aux
             else w_cm[:, None, :].expand(m, omega.shape[1], 3).contiguous())
    with_wind = aero.compute(state, omega, w_cm, w_rot)
    no_wind = aero.compute(state, omega, torch.zeros_like(w_cm), torch.zeros_like(w_rot))
    return {"total_force_W": with_wind.total_force_W.numpy(), "total_torque_B": with_wind.total_torque_B.numpy(),
            "wind_force_W": (with_wind.total_force_W - no_wind.total_force_W).numpy(),
            "wind_torque_B": (with_wind.total_torque_B - no_wind.total_torque_B).numpy()}


def attach(ep, nominal: DroneParams | None = None) -> None:
    """Add the wind-attributable target (and the with-wind reconstruction) to ``ep.aux`` under ``derived/``."""
    r = reconstruct(ep, nominal)
    ep.aux[WIND_FORCE], ep.aux[WIND_TORQUE], ep.aux[RECON_FORCE] = r["wind_force_W"], r["wind_torque_B"], r["total_force_W"]


def reconstruction_error(episodes: Sequence, nominal: DroneParams | None = None) -> dict:
    """How well the model-rate reconstruction of the TOTAL force reproduces the logged one (per-step relative norm error)."""
    rel, share = [], []
    for ep in episodes:
        r = reconstruct(ep, nominal)
        logged = np.asarray(ep.aux[TOTAL_FORCE])[: len(r["total_force_W"])]
        n = np.linalg.norm(logged, axis=1)
        ok = n > 1e-3
        if ok.any():
            rel.append(np.linalg.norm(r["total_force_W"] - logged, axis=1)[ok] / n[ok])
            share.append(np.linalg.norm(r["wind_force_W"], axis=1)[ok] / n[ok])
    if not rel:
        return {"n_steps": 0}
    rel, share = np.concatenate(rel), np.concatenate(share)
    return {"n_steps": int(len(rel)), "rel_error_median": float(np.median(rel)), "rel_error_p90": float(np.percentile(rel, 90)),
            "wind_share_of_total_median": float(np.median(share))}
