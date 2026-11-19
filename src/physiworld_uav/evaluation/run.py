"""CLI: evaluate baselines per split role and write a JSON + Markdown report.

    python -m physiworld_uav.evaluation.run --dataset DIR --models b0 b1 b2=outputs/experiments/b2_pilot/checkpoints/best.pt \\
        --out outputs/eval/pilot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, load_model_episodes
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.models.world_model import build_model
from physiworld_uav.training import checkpoint as ckpt
from physiworld_uav.training.trainer import build_from_checkpoint, pick_device
from .evaluator import evaluate

ROLE_GROUPS = {
    "val_id": ["val_id"], "test_id": ["test_id"],
    "test_ood_magnitude": ["test_ood_magnitude"], "test_ood_direction": ["test_ood_direction"],
    "test_ood_temporal": ["test_ood_temporal"], "test_ood_gust": ["test_ood_gust"],
    "test_ood_physics": ["test_ood_physics"], "test_ood_trajectory": ["test_ood_trajectory"],
    "val_ood_magnitude": ["val_ood_magnitude"], "val_ood_direction": ["val_ood_direction"],
    "val_ood_temporal": ["val_ood_temporal"], "val_ood_gust": ["val_ood_gust"], "val_ood_physics": ["val_ood_physics"],
}


def physics_validation_cases(episodes):
    """G10: Case A = baseline of a counterfactual group (nominal vehicle, no wind); Case C = its mass sibling."""
    a = [e for e in episodes if e.meta.counterfactual is None or e.meta.counterfactual.is_baseline]
    c = [e for e in episodes if e.meta.counterfactual is not None and not e.meta.counterfactual.is_baseline]
    return {"physics_validation:case_A_no_wind": a, "physics_validation:case_C_mass_plus5pct": c}


def load_model(spec: str, dataset_dir: Path, device):
    name, _, path = spec.partition("=")
    if name in ("b0", "b1"):
        return name, build_model(name)
    state = ckpt.load(path, map_location=device)
    return name, build_from_checkpoint(state)


def markdown(results: dict, ks) -> str:
    lines = []
    for role in next(iter(results.values())).keys():
        lines += [f"\n### {role}\n", "| model | windows | episodes | " + " | ".join(f"pos@{k} [m]" for k in ks) + " | att@last [deg] | reliable [steps] |",
                  "|---|---|---|" + "---|" * (len(ks) + 2)]
        for m, per in results.items():
            r = per[role]
            if not r.get("n_windows"):
                lines.append(f"| {m} | 0 | 0 |" + " |" * (len(ks) + 2)); continue
            att = r["att_deg"][str(ks[-1])]
            lines.append(f"| {m} | {r['n_windows']} | {r['n_episodes']} | " + " | ".join(f"{r['pos_m'][str(k)]:.4f}" for k in ks)
                         + f" | {att:.2f} | {r['reliable_horizon_mean']:.1f} |")
    return "\n".join(lines)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="evaluate", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True, help="b0 | b1 | b2=<checkpoint.pt>")
    p.add_argument("--roles", nargs="*", default=list(ROLE_GROUPS) + ["physics_validation"])
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--initial-state", default="given", choices=["given", "decoder"])
    p.add_argument("--limit-episodes", type=int, default=0, help="use only the first N episodes (sorted by id) of each role, e.g. the tiny-overfit set")
    p.add_argument("--noise", default="none", help="IMU noise preset for evaluation (none|low|nominal|high)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    device = pick_device(a.device)
    bcfg = BuilderConfig(decimation=4, history=10, horizon=a.horizon)
    groups: dict[str, list] = {}
    for role in a.roles:
        if role == "physics_validation":
            groups.update(physics_validation_cases(load_model_episodes(a.dataset, bcfg, roles=["physics_validation"])))
        else:
            groups[role] = load_model_episodes(a.dataset, bcfg, roles=ROLE_GROUPS.get(role, [role]))
        if a.limit_episodes:
            groups[role] = sorted(groups[role], key=lambda e: e.episode_id)[: a.limit_episodes]
    results: dict = {}
    for spec in a.models:
        name, model = load_model(spec, a.dataset, device)
        init = a.initial_state if name not in ("b0", "b1") else "given"
        results[name] = {role: evaluate(model, ModelReadyDataset(eps, bcfg, transform=ImuNoise(a.noise, 777) if a.noise != "none" else None), horizon=a.horizon, device=device,
                                        initial_state=init, ks=[k for k in (1, 5, 10, 20, 50, 100) if k <= a.horizon])
                         for role, eps in groups.items() if eps or True}
    ks = [k for k in (1, 5, 10, 20, 50, 100) if k <= a.horizon]
    md = markdown(results, ks)
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))
        a.out.with_suffix(".md").write_text(md)
    return results


if __name__ == "__main__":
    main()
