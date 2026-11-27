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
from physiworld_uav.data.dataset_builder import (IMU_RAW_WARNING, BuilderConfig, dataset_decimation, dataset_sim, imu_transport_status,
                                                 load_model_episodes)
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.models.world_model import build_model, check_obs_mode, set_phi_noise
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
    """(name, model, window settings stored in the checkpoint or None for B0/B1)."""
    name, _, path = spec.partition("=")
    dt = dataset_sim(dataset_dir)["model_dt"]                        # 0.02 s for the 50 Hz pilots, 0.05 s for a 20 Hz dataset
    if name in ("b0", "b1", "b6a_future", "b6a_current"):          # nothing to load: no learned parameters
        return name, build_model(name, dt=dt), None
    if name in ("b1_kv", "b6a_future_kv", "b6a_current_kv"):       # same physics with the TRUE vehicle parameters of each episode
        return name, build_model(name[:-3], dt=dt, known_vehicle=True), None
    state = ckpt.load(path, map_location=device)
    if abs(float(state["dt"]) - dt) > 1e-9:
        raise SystemExit(f"{name}: the checkpoint was trained at a model step of {state['dt']} s, the dataset has {dt} s; "
                         f"a model must be evaluated at the rate it was trained at")
    return name, build_from_checkpoint(state), checkpoint_window(state)


def checkpoint_window(state: dict) -> dict:
    """History length and observation mode a checkpoint was trained with (older checkpoints: from train_cfg, else 10/instant)."""
    tc = state.get("train_cfg", {})
    return {"history": int(state.get("history", tc.get("history", 10))), "obs_mode": state.get("obs_mode", tc.get("obs_mode", "instant"))}


def common_window(windows: list, default_history: int | None) -> dict:
    """One window setting for all models in a comparison: all checkpoints must agree (fair comparison)."""
    ws = [w for w in windows if w]
    if not ws:
        return {"history": default_history or 10, "obs_mode": "instant"}
    if len({(w["history"], w["obs_mode"]) for w in ws}) > 1:
        raise SystemExit(f"checkpoints were trained with different windows {ws}; evaluate them separately")
    if default_history and default_history != ws[0]["history"]:
        raise SystemExit(f"--history {default_history} differs from the checkpoint's {ws[0]['history']}")
    return ws[0]


def markdown(results: dict, ks) -> str:
    lines = []
    for role in next(iter(results.values())).keys():
        lines += [f"\n### {role}\n", "| model | windows | episodes | " + " | ".join(f"pos RMSE@{k} [m]" for k in ks) + " | att RMSE@last [deg] | reliable [steps] |",
                  "|---|---|---|" + "---|" * (len(ks) + 2)]
        for m, per in results.items():
            r = per[role]
            if not r.get("n_windows"):
                lines.append(f"| {m} | 0 | 0 |" + " |" * (len(ks) + 2)); continue
            att = r["att_rmse_deg"][str(ks[-1])]
            lines.append(f"| {m} | {r['n_windows']} | {r['n_episodes']} | " + " | ".join(f"{r['pos_rmse_m'][str(k)]:.4f}" for k in ks)
                         + f" | {att:.2f} | {r['reliable_horizon_mean']:.1f} |")
    return "\n".join(lines)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="evaluate", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="evaluation defaults (config/evaluation/default.yaml); flags override")
    p.add_argument("--models", nargs="+", required=True, help="b0 | b1 | b6a_future | b6a_current | b1_kv | b6a_future_kv | b6a_current_kv (true vehicle parameters) | <kind>=<checkpoint.pt> (b2, b3, b4, b5a, b5b, b6b_*)")
    p.add_argument("--roles", nargs="*", default=list(ROLE_GROUPS) + ["physics_validation"])
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--history", type=int, default=None, help="default: read from the checkpoint(s); 10 for b0/b1 only")
    p.add_argument("--initial-state", default="given", choices=["given", "decoder"])
    p.add_argument("--limit-episodes", type=int, default=0, help="use only the first N episodes (sorted by id) of each role, e.g. the tiny-overfit set")
    p.add_argument("--noise", default="nominal", help="IMU noise preset for evaluation (none|low|nominal|high)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None)
    pre = argparse.ArgumentParser(add_help=False)               # --config first: its values become the defaults of the other flags
    pre.add_argument("--config", type=Path, default=None)
    from physiworld_uav.training.config_io import eval_defaults
    p.set_defaults(**eval_defaults(pre.parse_known_args(argv)[0].config))
    a = p.parse_args(argv)
    device = pick_device(a.device)
    loaded = [load_model(spec, a.dataset, device) for spec in a.models]
    win = common_window([w for _, _, w in loaded], a.history)
    bcfg = BuilderConfig(decimation=dataset_decimation(a.dataset), history=win["history"], horizon=a.horizon, obs_mode=win["obs_mode"])
    groups: dict[str, list] = {}
    for role in a.roles:
        if role == "physics_validation":
            groups.update(physics_validation_cases(load_model_episodes(a.dataset, bcfg, roles=["physics_validation"])))
        else:
            groups[role] = load_model_episodes(a.dataset, bcfg, roles=ROLE_GROUPS.get(role, [role]))
    if a.limit_episodes:
        groups = {k: sorted(v, key=lambda e: e.episode_id)[: a.limit_episodes] for k, v in groups.items()}
    results: dict = {}
    all_eps = [e for v in groups.values() for e in v]
    if any(hasattr(m, "synth_imu") for _, m, _ in loaded) and imu_transport_status(all_eps) in ("raw", "mixed"):
        print("WARNING:", IMU_RAW_WARNING)
    for name, model, _ in loaded:
        check_obs_mode(model, bcfg.obs_mode)
        set_phi_noise(model, a.noise)       # φ's synthetic IMU gets the same white noise as the real IMU
        init = a.initial_state if name not in ("b0", "b1", "b6a_future", "b6a_current", "b1_kv", "b6a_future_kv", "b6a_current_kv") else "given"
        results[name] = {role: evaluate(model, ModelReadyDataset(eps, bcfg, transform=ImuNoise(a.noise, 777) if a.noise != "none" else None,
                                                          oracle=getattr(model, "needs_oracle", False)), horizon=a.horizon, device=device,
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
