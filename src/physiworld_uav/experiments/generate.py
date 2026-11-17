"""Dataset generation CLI (plan §4, §6.6).

Usage (inside the isaac-sim container, venv activated)::

    physiworld-generate plan       experiment=pilot             # manifest only (CPU)
    physiworld-generate run        experiment=smoke             # plan + simulate + validate
    physiworld-generate validate   --dataset data/datasets/smoke_v0.3.0
    physiworld-generate regenerate --dataset <dir> --episode <episode_id>

Outputs in ``<output_root>/<dataset_id>_v<version>/``::

    resolved_config.yaml   fully resolved config
    provenance.json        versions, git state, config hash, timestamps
    manifest.jsonl         one fully specified plan row per episode
    shard_XXX.h5           episodes (schema v2)
    index.json             episode rows, layout, validation summary, ``complete`` flag
    validation.json        full validation report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf

from physiworld_uav.config.compose import compose, to_plain
from physiworld_uav.core import provenance
from physiworld_uav.data.dataset_builder import model_layout
from physiworld_uav.data.dataset_validator import validate_all
from physiworld_uav.data.schema import SCHEMA_VERSION
from physiworld_uav.data.writer import ShardWriter, iter_episodes, read_episode
from physiworld_uav.drones.params import DroneParams

from .plan import Manifest, plan_dataset
from .runner import Termination, run_episode

log = logging.getLogger("physiworld.generate")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def dataset_dir(cfg: dict[str, Any]) -> Path:
    exp = cfg["experiment"]
    root = Path(cfg["dataset"]["output_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / f"{exp['dataset_id']}_v{exp['dataset_version']}"


def write_plan(cfg_dc) -> tuple[Path, Manifest, dict[str, Any]]:
    cfg = to_plain(cfg_dc)
    out = dataset_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    manifest = plan_dataset(cfg)
    manifest.write(out / "manifest.jsonl")
    (out / "resolved_config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    prov = provenance.collect(cfg)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
    log.info("planned %d episodes → %s (%s)", len(manifest.episodes), out, manifest.by_role())
    return out, manifest, prov


def _sim_settings(sim) -> dict[str, Any]:
    """Settings read back from the running simulator (plan §1-V4)."""
    return sim.physics_settings()


def _make_sim(phys: dict[str, Any], first: dict[str, Any]):
    from physiworld_uav.core.simulation_interface import SimulationConfig, SimulationInterface
    if phys.get("physics_pipeline", "gpu_dynamics_cpu_readback") == "gpu":
        raise ValueError("physics_pipeline 'gpu' disables readback: Isaac IMUSensor then reports "
                         "gravity-only frames (Run 4); use gpu_dynamics_cpu_readback or cpu")
    damping = phys.get("rigid_body_damping") or {}
    sim_cfg = SimulationConfig(
        dt=float(phys["dt"]), device=str(phys["device"]), render=False,
        headless=bool(phys.get("headless", True)), drone_model="Hummingbird",
        controller="RateController", init_position=list(first["initial_conditions"]["position"]),
        hydra_config={"headless": bool(phys.get("headless", True))},
        physics_pipeline=str(phys.get("physics_pipeline", "gpu_dynamics_cpu_readback")),
        enhanced_determinism=bool(phys.get("enhanced_determinism", True)),
        linear_damping=damping.get("linear"), angular_damping=damping.get("angular"),
    )
    sim = SimulationInterface(sim_cfg)
    sim.ensure_ready()
    return sim


def run(cfg_dc, *, limit: int | None = None, episodes: Sequence[str] | None = None,
        allow_unvalidated_forces: bool = False) -> Path:
    out, manifest, prov = write_plan(cfg_dc)
    cfg = to_plain(cfg_dc)
    phys = cfg["physics"]
    nominal = DroneParams.from_dict(cfg["drone"])
    term = Termination.from_dict(cfg.get("limits", {}).get("termination"))
    rows = manifest.episodes
    if episodes:
        rows = [r for r in rows if r["episode_id"] in set(episodes)]
    rows = sorted(rows, key=lambda r: r.get("replay_of") is not None)   # Mode B baselines first
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit("no episodes selected")
    per_shard = int(cfg["dataset"]["storage"].get("episodes_per_shard", 200))
    storage = cfg["dataset"]["storage"]
    sim = _make_sim(phys, rows[0])
    settings = _sim_settings(sim)
    index_rows: list[dict[str, Any]] = []
    where: dict[str, str] = {}
    t0 = time.time()
    writer = None
    try:
        for i, row in enumerate(rows):
            if i % per_shard == 0:
                if writer:
                    writer.close()
                shard = f"shard_{i // per_shard:03d}.h5"
                writer = ShardWriter(out / shard, compression=storage.get("compression"),
                                     compression_opts=storage.get("compression_opts"))
            replay = None
            if row.get("replay_of"):
                base_arrays, _ = read_episode(out / where[row["replay_of"]], row["replay_of"])
                D = int(row["sim"]["control_decimation"])
                replay = base_arrays["actions/rotor_cmd"][::D]
            ts = time.time()
            arrays, meta = run_episode(sim, row, nominal, termination=term,
                                       config_hash=prov["config_hash"],
                                       dataset_version=manifest.dataset_version,
                                       replay_commands=replay, sim_settings=settings,
                                       allow_unvalidated_forces=allow_unvalidated_forces)
            writer.write(arrays, meta)
            where[meta.episode_id] = shard
            index_rows.append({**meta.index_row(), "shard": shard})
            log.info("[%d/%d] %s %s %d steps in %.1fs", i + 1, len(rows), meta.episode_id,
                     meta.status, meta.n_steps, time.time() - ts)
    except BaseException:
        if writer:
            writer.close()
        sim.close()
        raise
    if writer:
        writer.close()
    # Validate and write the index BEFORE closing Isaac: SimulationApp.close()
    # terminates the interpreter, so nothing after it is guaranteed to run.
    report = validate_dataset_dir(out, index_rows)
    _write_index(out, manifest, index_rows, report, time.time() - t0)
    log.info("dataset %s complete=%s", out, json.loads((out / "index.json").read_text())["complete"])
    sim.close()
    return out


def validate_dataset_dir(out: Path, index_rows: list[dict[str, Any]] | None = None):
    shards = sorted({r["shard"] for r in index_rows}) if index_rows else \
        sorted(p.name for p in out.glob("shard_*.h5"))
    report = validate_all(ep for s in shards for ep in iter_episodes(out / s))
    (out / "validation.json").write_text(json.dumps(report.to_dict(), indent=2))
    log.info("validation: ok=%s hard=%d soft=%d", report.ok, len(report.hard), len(report.soft))
    for h in report.hard[:20]:
        log.error("  %s", h)
    return report


def _write_index(out: Path, manifest: Manifest, rows: list[dict[str, Any]], report, wall_s: float) -> None:
    index = {
        "dataset_id": manifest.dataset_id, "dataset_version": manifest.dataset_version,
        "experiment_id": manifest.experiment_id, "schema_version": SCHEMA_VERSION,
        "complete": bool(report.ok) and len(rows) == len(manifest.episodes),
        "n_planned": len(manifest.episodes), "n_written": len(rows),
        "validation": {"ok": report.ok, "hard": len(report.hard), "soft": len(report.soft)},
        "layout": model_layout(), "generation_wall_s": wall_s,
        "episodes": rows,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))


def regenerate(out: Path, episode_id: str) -> dict[str, float]:
    """Re-run one manifest row and compare with the stored episode (plan RP3)."""
    cfg = OmegaConf.to_container(OmegaConf.load(out / "resolved_config.yaml"), resolve=True)
    rows = {r["episode_id"]: r for r in Manifest.read_rows(out / "manifest.jsonl")}
    index = json.loads((out / "index.json").read_text())
    where = {r["episode_id"]: r["shard"] for r in index["episodes"]}
    row = rows[episode_id]
    stored, smeta = read_episode(out / where[episode_id], episode_id)
    replay = None
    if row.get("replay_of"):
        base, _ = read_episode(out / where[row["replay_of"]], row["replay_of"])
        replay = base["actions/rotor_cmd"][::int(row["sim"]["control_decimation"])]
    sim = _make_sim(cfg["physics"], row)
    try:
        arrays, _ = run_episode(sim, row, DroneParams.from_dict(cfg["drone"]),
                                termination=Termination.from_dict(cfg["limits"]["termination"]),
                                config_hash=smeta.config_hash, dataset_version=smeta.dataset_version,
                                replay_commands=replay, sim_settings=_sim_settings(sim))
    except BaseException:
        sim.close()
        raise
    diffs = {}
    for k, v in stored.items():
        if k in arrays and np.issubdtype(v.dtype, np.floating):
            a = arrays[k]
            if a.shape == v.shape:
                diffs[k] = float(np.nanmax(np.abs(a.astype(np.float64) - v.astype(np.float64))))
            else:
                diffs[k] = float("inf")
    (out / f"regenerate_{episode_id}.json").write_text(json.dumps(diffs, indent=2))
    print(json.dumps(diffs, indent=2), flush=True)
    sim.close()          # terminates the interpreter (Isaac); report already written
    return diffs


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="physiworld-generate")
    p.add_argument("command", choices=["plan", "run", "validate", "regenerate"])
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. experiment=pilot")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--episodes", nargs="*", default=None)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--episode", default=None)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--allow-unvalidated-forces", action="store_true",
                   help="development only: run force configs not marked validated (dataset is never complete)")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if a.command == "plan":
        write_plan(compose(a.overrides))
    elif a.command == "run":
        print(run(compose(a.overrides), limit=a.limit, episodes=a.episodes,
                  allow_unvalidated_forces=a.allow_unvalidated_forces))
    elif a.command == "validate":
        r = validate_dataset_dir(a.dataset)
        return 0 if r.ok else 1
    else:
        regenerate(a.dataset, a.episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
