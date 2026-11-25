"""Dataset generation CLI (plan §4, §6.6).

Usage (inside the isaac-sim container, venv activated)::

    physiworld-generate plan       experiment=pilot             # manifest only (CPU)
    physiworld-generate run        experiment=smoke             # plan + simulate + validate
    physiworld-generate validate   --dataset data/datasets/smoke_v0.3.0
    physiworld-generate regenerate --dataset <dir> --episode <episode_id>

Parallel generation (one process per worker; CPU-bound, so several workers per GPU are fine)::

    physiworld-generate run   experiment=pilot --shard 0/4        # worker 0 of 4 (CUDA_VISIBLE_DEVICES picks the GPU)
    physiworld-generate run   experiment=pilot --shard 1/4        # ... and so on, concurrently
    physiworld-generate merge experiment=pilot                    # after all workers finished: validate + index.json
    # or: scripts/generate_parallel.sh --workers 8 --gpus 0,1,2 -- experiment=pilot

Outputs in ``<output_root>/<dataset_id>_v<version>/``::

    resolved_config.yaml   fully resolved config
    provenance.json        versions, git state, config hash, timestamps
    manifest.jsonl         one fully specified plan row per episode
    shard_XXX.h5           episodes (schema v2); parallel workers write shard_wII_XXX.h5 and index.workerII.json
    index.json             episode rows, layout, validation summary, ``complete`` flag
    validation.json        full validation report
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf

from physiworld_uav.config.compose import compose, to_plain
from physiworld_uav.core import provenance
from physiworld_uav.data.dataset_builder import model_layout
from physiworld_uav.data.dataset_validator import ValidationReport, validate_all
from physiworld_uav.data.schema import SCHEMA_VERSION
from physiworld_uav.data.writer import ShardWriter, add_missing_checksums, iter_episodes, read_episode, verify_episode_group, verify_shard
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


def write_plan(cfg_dc, *, only_missing: bool = False) -> tuple[Path, Manifest, dict[str, Any]]:
    """Plan the dataset and write manifest / resolved config / provenance.

    ``only_missing`` (parallel workers): a file that already exists is left alone. Every file is written through a tmp file
    that is unique to the process and then renamed, so concurrent workers never see a half-written file."""
    cfg = to_plain(cfg_dc)
    out = dataset_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    manifest = plan_dataset(cfg)
    prov = provenance.collect(cfg)

    def put(name: str, write) -> None:
        path = out / name
        if only_missing and path.exists():
            return
        tmp = path.with_name(f"{name}.{os.getpid()}.tmp")
        write(tmp)
        os.replace(tmp, path)

    put("manifest.jsonl", manifest.write)
    put("resolved_config.yaml", lambda t: t.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg))))
    put("provenance.json", lambda t: t.write_text(json.dumps(prov, indent=2, sort_keys=True)))
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


# ---------------------------------------------------------------------------------------------
# Parallel generation: manifest rows are split across worker processes (plan §6.8, Run 6).
# A counterfactual group stays on ONE worker: Mode B siblings replay the baseline's commands, which they read from the
# baseline's shard. Assignment is deterministic (longest-processing-time on the planned steps), so every worker computes
# the same split from the manifest alone and no coordination is needed. Episode content never depends on the worker.
# ---------------------------------------------------------------------------------------------
def group_key(row: dict[str, Any]) -> str:
    return str(row.get("group_id") or row["episode_id"])


def assign_workers(rows: Sequence[dict[str, Any]], n_workers: int) -> dict[str, int]:
    """group key → worker index, balancing planned physics steps (largest group first, least-loaded worker)."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    cost: dict[str, float] = {}
    for r in rows:
        cost[group_key(r)] = cost.get(group_key(r), 0.0) + float(r.get("n_steps") or r.get("duration_s") or 1.0)
    load = [0.0] * n_workers
    out: dict[str, int] = {}
    for key in sorted(cost, key=lambda k: (-cost[k], k)):
        w = min(range(n_workers), key=lambda i: (load[i], i))
        out[key] = w
        load[w] += cost[key]
    return out


def worker_rows(rows: Sequence[dict[str, Any]], worker: int, n_workers: int) -> list[dict[str, Any]]:
    owner = assign_workers(rows, n_workers)
    return [r for r in rows if owner[group_key(r)] == worker]


def shard_prefix(shard: tuple[int, int] | None) -> str:
    return "shard_" if shard is None or shard[1] == 1 else f"shard_w{shard[0]:02d}_"


def _scan_shards(out: Path, order: dict[str, int], prefix: str = "shard_") -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rebuild the index rows and the episode→shard map from the shards (the source of truth).

    An episode group without metadata is a write that was interrupted; it is removed so the
    episode is generated again.
    """
    import h5py
    from physiworld_uav.data.metadata import EpisodeMetadata
    rows: list[dict[str, Any]] = []
    where: dict[str, str] = {}
    pattern = "shard_*.h5" if prefix == "shard_" else f"{prefix}*.h5"      # a worker only ever opens its own shard files
    for shard in sorted(out.glob(pattern)):
        with h5py.File(shard, "a") as f:
            eps = f.require_group("episodes")
            for eid in list(eps.keys()):
                if "metadata" not in eps[eid].attrs:
                    log.warning("removing incomplete episode %s from %s", eid, shard.name)
                    del eps[eid]
                    continue
                status = verify_episode_group(eps[eid])
                if status in ("mismatch", "unreadable"):
                    log.warning("removing corrupted episode %s from %s (checksum %s); it will be generated again", eid, shard.name, status)
                    del eps[eid]
                    continue
                meta = EpisodeMetadata.from_json(eps[eid].attrs["metadata"])
                rows.append({**meta.index_row(), "shard": shard.name})
                where[eid] = shard.name
    rows.sort(key=lambda r: order.get(r["episode_id"], len(order)))
    return rows, where


def run(cfg_dc, *, limit: int | None = None, episodes: Sequence[str] | None = None,
        allow_unvalidated_forces: bool = False, resume: bool = False,
        shard: tuple[int, int] | None = None) -> Path:
    """Generate the dataset, or one worker's share of it (``shard=(i, n)``: this process handles worker i of n).

    A worker writes only its own ``shard_wII_NNN.h5`` files and ``index.workerII.json``; it does not validate or write
    ``index.json``. Run :func:`merge` after all workers have finished."""
    cfg = to_plain(cfg_dc)
    if shard is not None and not (0 <= shard[0] < shard[1]):
        raise SystemExit(f"--shard {shard[0]}/{shard[1]}: need 0 <= i < n")
    prefix = shard_prefix(shard)
    existing = sorted(dataset_dir(cfg).glob("shard_*.h5" if prefix == "shard_" else f"{prefix}*.h5"))
    if existing and not resume:
        raise SystemExit(f"{dataset_dir(cfg)} already contains {len(existing)} shard(s). Use --resume to "
                         "continue (skips episodes already written) or change dataset_version/dataset_id.")
    out, manifest, prov = write_plan(cfg_dc, only_missing=shard is not None)
    phys = cfg["physics"]
    nominal = DroneParams.from_dict(cfg["drone"])
    term = Termination.from_dict(cfg.get("limits", {}).get("termination"))
    rows = manifest.episodes
    order = {r["episode_id"]: i for i, r in enumerate(rows)}
    if shard is not None:
        rows = worker_rows(rows, shard[0], shard[1])
        log.info("worker %d/%d: %d of %d episodes", shard[0], shard[1], len(rows), len(order))
    n_assigned = len(rows)
    if episodes:
        rows = [r for r in rows if r["episode_id"] in set(episodes)]
    rows = sorted(rows, key=lambda r: r.get("replay_of") is not None)   # Mode B baselines first
    index_rows: list[dict[str, Any]] = []
    where: dict[str, str] = {}
    if resume:
        index_rows, where = _scan_shards(out, order, prefix)
        log.info("resume: %d episodes already written", len(index_rows))
        rows = [r for r in rows if r["episode_id"] not in where]
    if limit is not None:
        rows = rows[:limit]
    if not rows and shard is not None:
        _write_worker_index(out, shard, n_assigned, index_rows, [], 0.0)
        log.info("worker %d/%d: nothing to generate", *shard)
        return out
    if not rows:
        if resume:                                              # nothing left: just finalise the index
            report = validate_dataset_dir(out, index_rows)
            _write_index(out, manifest, index_rows, report, 0.0, [])
            log.info("nothing to generate; index refreshed")
            return out
        raise SystemExit("no episodes selected")
    per_shard = int(cfg["dataset"]["storage"].get("episodes_per_shard", 200))
    storage = cfg["dataset"]["storage"]
    ckpt_every = max(1, int(storage.get("index_checkpoint_every", 1)))
    sim = _make_sim(phys, rows[0])
    settings = _sim_settings(sim)
    failures: list[dict[str, Any]] = []
    t0 = time.time()

    def checkpoint() -> None:
        if shard is not None:
            _write_worker_index(out, shard, n_assigned, index_rows, failures, time.time() - t0, in_progress=True)
        else:
            _write_index(out, manifest, index_rows, None, time.time() - t0, failures, in_progress=True)

    try:
        for i, row in enumerate(rows):
            ts = time.time()
            try:
                replay = None
                if row.get("replay_of"):
                    if row["replay_of"] not in where:
                        raise RuntimeError(f"baseline {row['replay_of']} was not written (it failed or is missing)")
                    base_arrays, _ = read_episode(out / where[row["replay_of"]], row["replay_of"])
                    D = int(row["sim"]["control_decimation"])
                    replay = base_arrays["actions/rotor_cmd"][::D]
                arrays, meta = run_episode(sim, row, nominal, termination=term,
                                           config_hash=prov["config_hash"],
                                           dataset_version=manifest.dataset_version,
                                           replay_commands=replay, sim_settings=settings,
                                           allow_unvalidated_forces=allow_unvalidated_forces)
            except Exception as exc:                    # one bad episode must not end the run
                tb = traceback.format_exc()
                log.error("[%d/%d] %s FAILED: %s: %s\n%s", i + 1, len(rows), row["episode_id"],
                          type(exc).__name__, exc, tb)
                failures.append({"episode_id": row["episode_id"], "error": type(exc).__name__,
                                 "message": str(exc), "traceback": tb})
                checkpoint()
                continue
            shard_file = f"{prefix}{len(index_rows) // per_shard:03d}.h5"
            # The shard is closed after every episode, so a crash between episodes cannot leave it
            # half-written; only the episode being written at that instant can be lost.
            with ShardWriter(out / shard_file, compression=storage.get("compression"),
                             compression_opts=storage.get("compression_opts")) as writer:
                writer.write(arrays, meta)
            where[meta.episode_id] = shard_file
            index_rows.append({**meta.index_row(), "shard": shard_file})
            log.info("[%d/%d] %s %s %d steps in %.1fs", i + 1, len(rows), meta.episode_id,
                     meta.status, meta.n_steps, time.time() - ts)
            if len(index_rows) % ckpt_every == 0:
                checkpoint()
    except BaseException:
        # SimulationApp.close() ends the process, so log the traceback and save the index BEFORE closing.
        log.exception("generation aborted after %d written episodes", len(index_rows))
        try:
            checkpoint()
        except Exception:
            log.exception("could not write the checkpoint index")
        sim.close()
        raise
    if failures:
        log.warning("%d episode(s) failed and are missing from the dataset: %s", len(failures),
                    ", ".join(f["episode_id"] for f in failures))
    if shard is not None:        # a worker only reports its own progress; `merge` validates the whole dataset
        _write_worker_index(out, shard, n_assigned, index_rows, failures, time.time() - t0)
        log.info("worker %d/%d finished: %d written, %d failed", shard[0], shard[1], len(index_rows), len(failures))
        sim.close()
        return out
    # Validate and write the index BEFORE closing Isaac: SimulationApp.close()
    # terminates the interpreter, so nothing after it is guaranteed to run.
    report = validate_dataset_dir(out, index_rows)
    _write_index(out, manifest, index_rows, report, time.time() - t0, failures)
    log.info("dataset %s complete=%s", out, json.loads((out / "index.json").read_text())["complete"])
    sim.close()
    return out


def _write_worker_index(out: Path, shard: tuple[int, int], n_assigned: int, rows: list[dict[str, Any]],
                        failures: Sequence[dict[str, Any]], wall_s: float, in_progress: bool = False) -> None:
    doc = {"worker": shard[0], "n_workers": shard[1], "n_assigned": n_assigned, "n_written": len(rows),
           "in_progress": bool(in_progress), "failures": list(failures), "generation_wall_s": wall_s,
           "episodes": [r["episode_id"] for r in rows]}
    path = out / f"index.worker{shard[0]:02d}.json"
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
    os.replace(tmp, path)


def merge(cfg_dc, *, force: bool = False) -> Path:
    """Combine the shards of all workers into one dataset: scan the shards (the source of truth), validate the whole
    dataset and write ``index.json`` / ``validation.json``. Safe to repeat. Refuses while a worker still reports
    ``in_progress`` (a crashed worker leaves that flag set: ``force=True`` overrides after you checked it is dead)."""
    cfg = to_plain(cfg_dc)
    out = dataset_dir(cfg)
    manifest = plan_dataset(cfg)
    order = {r["episode_id"]: i for i, r in enumerate(manifest.episodes)}
    workers = [json.loads(p.read_text()) for p in sorted(out.glob("index.worker*.json"))]
    busy = [w["worker"] for w in workers if w["in_progress"]]
    if busy and not force:
        raise SystemExit(f"workers {busy} still report in_progress (running, or crashed); use --force to merge anyway")
    index_rows, where = _scan_shards(out, order)
    failures = [f for w in workers for f in w["failures"] if f["episode_id"] not in where]
    counts: dict[str, int] = {}
    for r in index_rows:
        counts[r["episode_id"]] = counts.get(r["episode_id"], 0) + 1
    dup = [e for e, c in counts.items() if c > 1]
    if dup:
        raise SystemExit(f"episodes written by more than one worker: {dup[:5]}")
    report = validate_dataset_dir(out, index_rows)
    wall = max((w["generation_wall_s"] for w in workers), default=0.0)
    _write_index(out, manifest, index_rows, report, wall, failures)
    idx = json.loads((out / "index.json").read_text())
    log.info("merged %d workers: %d/%d episodes, complete=%s", len(workers), len(index_rows), len(manifest.episodes), idx["complete"])
    return out


def validate_dataset_dir(out: Path, index_rows: list[dict[str, Any]] | None = None):
    shards = sorted({r["shard"] for r in index_rows}) if index_rows else \
        sorted(p.name for p in out.glob("shard_*.h5"))
    integrity = {s: verify_shard(out / s) for s in shards}
    bad = [f"{s}:{e} checksum {st}" for s, d in integrity.items() for e, st in d.items() if st in ("mismatch", "unreadable")]
    legacy = sum(1 for d in integrity.values() for st in d.values() if st == "missing")
    if bad:                      # do not try to interpret corrupted data: report the integrity failures only
        report = ValidationReport()
        report.hard.extend(f"integrity: {b}" for b in bad)
    else:
        report = validate_all(ep for s in shards for ep in iter_episodes(out / s))
    if legacy:
        log.warning("%d episode(s) have no checksum (written before checksums); run `generate checksums --dataset %s`", legacy, out)
    (out / "validation.json").write_text(json.dumps(report.to_dict(), indent=2))
    log.info("validation: ok=%s hard=%d soft=%d", report.ok, len(report.hard), len(report.soft))
    for h in report.hard[:20]:
        log.error("  %s", h)
    return report


def _write_index(out: Path, manifest: Manifest, rows: list[dict[str, Any]], report, wall_s: float,
                 failures: Sequence[dict[str, Any]] = (), in_progress: bool = False) -> None:
    """Write index.json atomically. ``report=None`` means validation has not run yet (checkpoint)."""
    validation = ({"ok": report.ok, "hard": len(report.hard), "soft": len(report.soft)}
                  if report is not None else {"ok": None, "hard": None, "soft": None})
    index = {
        "dataset_id": manifest.dataset_id, "dataset_version": manifest.dataset_version,
        "experiment_id": manifest.experiment_id, "schema_version": SCHEMA_VERSION,
        "complete": bool(report is not None and report.ok and not failures and not in_progress
                         and len(rows) == len(manifest.episodes)),
        "in_progress": bool(in_progress),
        "failures": list(failures),
        "n_planned": len(manifest.episodes), "n_written": len(rows),
        "validation": validation,
        "layout": model_layout(), "generation_wall_s": wall_s,
        "episodes": rows,
    }
    tmp = out / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
    os.replace(tmp, out / "index.json")          # atomic: readers never see a half-written index


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
    p.add_argument("command", choices=["plan", "run", "merge", "validate", "regenerate", "checksums"])
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. experiment=pilot")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--episodes", nargs="*", default=None)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--episode", default=None)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--resume", action="store_true",
                   help="continue an existing dataset: keep episodes already in the shards, generate only the rest")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="parallel generation: run worker I of N (0-based). Each worker needs its own process (and GPU via "
                        "CUDA_VISIBLE_DEVICES); afterwards run `merge` once")
    p.add_argument("--force", action="store_true", help="merge: proceed although a worker still reports in_progress")
    p.add_argument("--allow-unvalidated-forces", action="store_true",
                   help="development only: run force configs not marked validated (dataset is never complete)")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if a.command == "plan":
        write_plan(compose(a.overrides))
    elif a.command == "run":
        shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
        print(run(compose(a.overrides), limit=a.limit, episodes=a.episodes,
                  allow_unvalidated_forces=a.allow_unvalidated_forces, resume=a.resume, shard=shard))
    elif a.command == "checksums":
        d = a.dataset
        added = {sh.name: add_missing_checksums(sh) for sh in sorted(d.glob("shard_*.h5"))}
        status = {sh.name: verify_shard(sh) for sh in sorted(d.glob("shard_*.h5"))}
        bad = {sh: {e: st for e, st in v.items() if st != "ok"} for sh, v in status.items()}
        print(json.dumps({"added": added, "not_ok": {k: v for k, v in bad.items() if v}}, indent=2))
        return 0 if not any(bad.values()) else 1
    elif a.command == "merge":
        print(merge(compose(a.overrides), force=a.force))
    elif a.command == "validate":
        r = validate_dataset_dir(a.dataset)
        return 0 if r.ok else 1
    else:
        regenerate(a.dataset, a.episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
