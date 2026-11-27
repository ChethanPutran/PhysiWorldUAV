"""Episode labels for evaluation splits: wind vs no wind, counterfactual mode, counterfactual group."""
from __future__ import annotations

from typing import Sequence

import numpy as np

NO_WIND_PEAK_MPS = 0.05          # realised peak wind speed below this: the episode is a "no wind" episode
WIND_KEY = "environment/wind_applied_W"


def peak_wind(ep) -> float:
    """Realised peak wind speed [m/s] (from the metadata statistics; the logged wind when there is no metadata)."""
    if getattr(ep, "meta", None) is not None:
        st = (ep.meta.plan.get("wind") or {}).get("stats") or {}
        if "peak_speed" in st:
            return float(st["peak_speed"])
    if WIND_KEY in getattr(ep, "aux", {}):
        return float(np.linalg.norm(np.asarray(ep.aux[WIND_KEY]), axis=-1).max())
    return float("nan")


def wind_regime(ep) -> str:
    p = peak_wind(ep)
    return "no_wind" if p < NO_WIND_PEAK_MPS else "wind"


def cf_mode(ep) -> str:
    cf = getattr(getattr(ep, "meta", None), "counterfactual", None)
    return cf.mode if cf is not None else "none"          # closed_loop (Mode A) | fixed_action (Mode B) | none


def cf_group(ep) -> str | None:
    cf = getattr(getattr(ep, "meta", None), "counterfactual", None)
    return cf.group_id if cf is not None else None


def cf_variable(ep) -> str | None:
    cf = getattr(getattr(ep, "meta", None), "counterfactual", None)
    return cf.variable if cf is not None else None


def select(episodes: Sequence, wind: str | None = None, mode: str | None = None) -> list:
    """Filter by wind regime ("wind" | "no_wind") and/or counterfactual mode ("closed_loop" | "fixed_action" | "none")."""
    out = list(episodes)
    if wind is not None:
        out = [e for e in out if wind_regime(e) == wind]
    if mode is not None:
        out = [e for e in out if cf_mode(e) == mode]
    return out
