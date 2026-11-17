"""Counterfactual sibling construction (plan §6.3-T12, §6.2-W9; design doc §10).

A counterfactual group shares initial state, reference, excitation, seeds and
initial motor state; exactly **one variable class** differs between siblings.
The baseline is sibling 0. The planner calls ``build_siblings`` with the
baseline episode plan (a plain dict) and receives one plan per sibling plus
the exact ``diff`` against the baseline, which is verified to touch only the
paths owned by the variable.

Modes:
* ``closed_loop`` (A): every sibling runs the same controller on the same reference.
* ``fixed_action`` (B): the baseline runs closed loop; siblings replay the
  baseline's *applied* rotor commands (``controller.mode = "replay"``).
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping

from physiworld_uav.wind.regimes import Limits, WindPlan, check_limits, rotated_plan, scaled_plan

VARIABLES = {
    # variable -> manifest paths a sibling may change
    "wind_magnitude": ("wind",),
    "wind_direction": ("wind",),
    "wind_on_off": ("wind",),
    "mass": ("domain",),
    "inertia": ("domain",),
    "rotor_drag": ("domain",),
    "body_drag": ("domain",),
    "motor_time_constant": ("domain",),
    "aero_model": ("forces",),
}
DOMAIN_FACTORS = {
    "mass": ("mass",), "inertia": ("inertia_shape_x", "inertia_shape_y", "inertia_shape_z"),
    "rotor_drag": ("rotor_drag_scale",), "body_drag": ("body_drag_scale",),
    "motor_time_constant": ("motor_time_constant",),
}
#: Bookkeeping paths that always differ between siblings.
IGNORED = ("episode_id", "counterfactual", "controller", "replay_of")
IGNORED_PATHS = ("seeds/episode",)          # per-episode id seed; all random streams are shared


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, Mapping):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}/"))
        return out or {prefix.rstrip("/"): {}}
    return {prefix.rstrip("/"): d}


def plan_diff(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, list[Any]]:
    fb, fo = _flatten(base), _flatten(other)
    keys = sorted(set(fb) | set(fo))
    return {k: [fb.get(k), fo.get(k)] for k in keys
            if fb.get(k) != fo.get(k) and k.split("/")[0] not in IGNORED
            and not k.startswith(IGNORED_PATHS)}


def build_siblings(baseline: Mapping[str, Any], template: Mapping[str, Any], *,
                   duration: float, dt: float, group_id: str,
                   id_halfwidths: Mapping[str, float] | None = None,
                   drone_summary: Callable[[Mapping[str, Any]], dict] | None = None,
                   forces_loader: Callable[[str], dict] | None = None) -> list[dict[str, Any]]:
    """Return sibling plans (sibling 0 is the baseline itself, with metadata filled in)."""
    variable = str(template["variable"])
    if variable not in VARIABLES:
        raise ValueError(f"unknown counterfactual variable {variable!r}")
    mode = str(template.get("mode", "closed_loop"))
    values = list(template["values"])
    if len(values) < 2:
        raise ValueError("a counterfactual group needs at least 2 values")

    sibs: list[dict[str, Any]] = []
    base_wind = WindPlan(**{k: baseline["wind"][k] for k in
                            ("regime", "category", "profile", "spec", "seed", "stats", "labels")})
    limits = baseline["wind"].get("limits")
    for i, value in enumerate(values):
        plan = copy.deepcopy(dict(baseline))
        if variable in ("wind_magnitude", "wind_on_off"):
            scale = float(value) if variable == "wind_magnitude" else (1.0 if value else 0.0)
            w = scaled_plan(base_wind, scale, duration=duration, dt=dt)
            plan["wind"] = {**w.to_dict(), "limits": limits}
        elif variable == "wind_direction":
            w = rotated_plan(base_wind, math.radians(float(value)), duration=duration, dt=dt)
            plan["wind"] = {**w.to_dict(), "limits": limits}
        elif variable in DOMAIN_FACTORS:
            factors = dict(plan["domain"]["factors"])
            for f in DOMAIN_FACTORS[variable]:
                factors[f] = factors[f] * float(value)
                hw = (id_halfwidths or {}).get(f)
                if hw is not None and plan["domain"]["region"] in ("id", "nominal") \
                        and abs(factors[f] - 1.0) > hw + 1e-12:
                    raise ValueError(f"counterfactual {variable}×{value} leaves the ID range for {f}")
            region = plan["domain"]["region"]
            if region == "nominal" and i > 0:
                region = "id"          # a perturbed nominal vehicle is no longer nominal
            plan["domain"] = {**plan["domain"], "factors": factors, "region": region,
                              "domain_id": plan["domain"]["domain_id"] + (f"_cf{i}" if i else "")}
            if drone_summary is None:
                raise ValueError("domain counterfactuals need a drone_summary callback")
            plan["domain"]["drone"] = drone_summary(plan["domain"])
        elif variable == "aero_model":
            if forces_loader is None:
                raise ValueError("aero_model counterfactuals need a forces_loader callback")
            plan["forces"] = {"name": str(value), "config": forces_loader(str(value))}
        if limits and variable.startswith("wind"):
            bad = check_limits(plan["wind"]["stats"], Limits.from_dict(limits))
            if bad:
                raise ValueError(f"counterfactual sibling {i} ({variable}={value}) violates the "
                                 f"wind regime limits: {bad}")
        plan["counterfactual"] = {"group_id": group_id, "mode": mode, "variable": variable,
                                  "sibling_index": i, "is_baseline": i == 0, "value": value}
        if mode == "fixed_action" and i > 0:
            plan["controller"] = {**plan["controller"], "mode": "replay"}
            plan["replay_of"] = None          # filled with the baseline episode id by the planner
        sibs.append(plan)

    allowed = VARIABLES[variable]
    for i, s in enumerate(sibs[1:], start=1):
        diff = plan_diff(sibs[0], s)
        stray = [k for k in diff if k.split("/")[0] not in allowed]
        if stray:
            raise AssertionError(f"counterfactual sibling {i} changes paths outside {allowed}: {stray[:5]}")
        s["counterfactual"]["diff"] = {k: v for k, v in diff.items()}
    sibs[0]["counterfactual"]["diff"] = {}
    return sibs
