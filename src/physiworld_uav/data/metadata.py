"""Episode metadata, schema v2 (plan §6.5-DS10; prompt §11–§12).

Every episode stores one ``EpisodeMetadata`` (as a JSON attribute in its HDF5
group and as a row of the dataset index). It must be sufficient to identify
and **reproduce** the episode: the manifest row (``plan``) holds every sampled
value, ``seeds`` holds every stream seed, and ``config_hash`` +
``provenance_ref`` tie it to the exact resolved config and code version.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .schema import DATA_TYPES, SCHEMA_VERSION, SPLIT_ROLES

STATUS_VALUES = ("completed", "terminated_ground", "terminated_envelope",
                 "terminated_divergence", "failed_sim", "aborted_user")
COUNTERFACTUAL_MODES = ("closed_loop", "fixed_action")


@dataclass
class CounterfactualInfo:
    group_id: str
    mode: str                                 # closed_loop (Mode A) | fixed_action (Mode B)
    variable: str                             # wind_magnitude | wind_direction | wind_on_off | mass | ...
    sibling_index: int
    is_baseline: bool
    diff: dict[str, list[Any]] = field(default_factory=dict)   # path -> [baseline, this]

    def __post_init__(self) -> None:
        if self.mode not in COUNTERFACTUAL_MODES:
            raise ValueError(f"counterfactual mode must be one of {COUNTERFACTUAL_MODES}")


@dataclass
class EpisodeMetadata:
    # identity -----------------------------------------------------------------
    dataset_id: str
    dataset_version: str
    experiment_id: str
    episode_id: str
    scenario_id: str
    domain_id: str
    group_id: str                             # split unit (counterfactual group or scenario)
    split_role: str
    data_type: str                            # closed_loop | open_loop | physics_validation
    ood_category: str | None = None           # None for ID roles
    experiment_group: str | None = None       # design-doc §11.3 / §12.2 group label
    mode: str | None = None                   # identifiability mode (hover, translation, …)
    scenario_template: str | None = None
    counterfactual: CounterfactualInfo | None = None

    # what was simulated ---------------------------------------------------------
    plan: dict[str, Any] = field(default_factory=dict)          # full manifest row
    drone: dict[str, Any] = field(default_factory=dict)         # true parameters simulated
    drone_realised: dict[str, Any] | None = None                # PhysX readback (Run 4)
    controller: dict[str, Any] = field(default_factory=dict)    # mode, policy, gains
    forces: dict[str, Any] = field(default_factory=dict)        # force-model config + flags
    sim: dict[str, Any] = field(default_factory=dict)           # dt, decimation, settings
    seeds: dict[str, Any] = field(default_factory=dict)         # stream -> seed / path
    initial_conditions: dict[str, Any] = field(default_factory=dict)
    terminal_conditions: dict[str, Any] = field(default_factory=dict)

    # outcome --------------------------------------------------------------------
    status: str = "completed"
    termination_reason: str | None = None
    n_steps: int = 0
    duration_s: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)          # saturation, envelope, wind realised

    # provenance -------------------------------------------------------------------
    schema_version: str = SCHEMA_VERSION
    config_hash: str = ""
    provenance_ref: str = "provenance.json"
    generated_utc: str = ""

    def __post_init__(self) -> None:
        if self.split_role not in SPLIT_ROLES:
            raise ValueError(f"split_role {self.split_role!r} not in {SPLIT_ROLES}")
        if self.data_type not in DATA_TYPES:
            raise ValueError(f"data_type {self.data_type!r} not in {DATA_TYPES}")
        if self.status not in STATUS_VALUES:
            raise ValueError(f"status {self.status!r} not in {STATUS_VALUES}")
        is_ood = "_ood_" in self.split_role
        if is_ood and not self.ood_category:
            raise ValueError("OOD split roles require ood_category")
        if not is_ood and self.ood_category:
            raise ValueError(f"ood_category set for non-OOD role {self.split_role!r}")
        if isinstance(self.counterfactual, Mapping):
            self.counterfactual = CounterfactualInfo(**self.counterfactual)

    # -- (de)serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=_json_default)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EpisodeMetadata":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown metadata keys: {sorted(unknown)}")
        return cls(**dict(d))

    @classmethod
    def from_json(cls, text: str) -> "EpisodeMetadata":
        return cls.from_dict(json.loads(text))

    def index_row(self) -> dict[str, Any]:
        """Compact row for the dataset index (split/leakage bookkeeping)."""
        cf = self.counterfactual
        return {
            "episode_id": self.episode_id, "scenario_id": self.scenario_id,
            "domain_id": self.domain_id, "group_id": self.group_id,
            "split_role": self.split_role, "ood_category": self.ood_category,
            "data_type": self.data_type, "experiment_group": self.experiment_group,
            "mode": self.mode, "status": self.status, "n_steps": self.n_steps,
            "counterfactual_group": None if cf is None else cf.group_id,
            "counterfactual_variable": None if cf is None else cf.variable,
            "wind_regime": self.plan.get("wind", {}).get("regime"),
            "seed_episode": self.seeds.get("episode"),
        }


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")
