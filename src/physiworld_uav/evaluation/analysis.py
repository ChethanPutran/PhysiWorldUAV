"""Evaluation analyses beyond the per-role table of ``evaluation.run``.

    python -m physiworld_uav.evaluation.analysis regime   --dataset D --models b1 b3=CKPT b6a_future --gap b3:b6a_future b3:b1
    python -m physiworld_uav.evaluation.analysis anchor   --dataset D --models b3=CKPT b1 --every 1 5 10 --mode h
    python -m physiworld_uav.evaluation.analysis residual --dataset D --models b3=CKPT
    python -m physiworld_uav.evaluation.analysis shuffle  --dataset D --models b1 b2=CKPT b3=CKPT b4=CKPT

* ``regime``  : RMSE per horizon step for wind and no-wind episodes (error vs step), and paired gaps between two models with
                an episode-bootstrap CI (the B3–B6 gap: without wind it should be small).
* ``anchor``  : re-observation every K steps (``--mode h``: real IMU into the latent only; ``--mode state``: also the true
                state). Reports RMSE vs step, RMSE by steps since the last real observation (staleness), and the cost of the
                synthetic IMU φ against the real IMU at the same step (the observation→imagination "boundary jump").
* ``residual``: size of d_θ against f_phys per regime and the degradation when d_θ is zeroed (no-wind data: small).
* ``shuffle`` : shuffled-u test on Mode A wind-magnitude groups; read the jump of each model relative to B1.

All commands: ``--roles`` (default val_id), ``--cf-mode closed_loop|fixed_action|none`` (Mode A / Mode B filter),
``--noise`` (default nominal), ``--horizon`` (default 20), ``--out file.json``.
Mode B (fixed-action) episodes exist only in the train and val_id roles of the pilot; test roles have Mode A only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, dataset_decimation, load_model_episodes
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.models.world_model import Anchor, check_obs_mode, set_phi_noise
from physiworld_uav.training.trainer import pick_device
from . import regimes as R
from .evaluator import collect_errors
from .gap import paired_gap, residual_diagnostics, residual_off, rmse_curve, summarize_residual
from .leakage import ActionSwap, partner_map
from .run import ROLE_GROUPS, common_window, load_model


def make_ds(eps, bcfg, model, noise: str, future_obs: bool = False) -> ModelReadyDataset:
    return ModelReadyDataset(eps, bcfg, transform=ImuNoise(noise, 777) if noise != "none" else None,
                             oracle=getattr(model, "needs_oracle", False), future_obs=future_obs)


def ks_for(horizon: int) -> list[int]:
    return [k for k in (1, 2, 5, 10, 20, 50, 100) if k <= horizon]


def _fmt(row: Sequence[float]) -> str:
    return " | ".join(f"{v:.4f}" for v in row)


# ----------------------------------------------------------------------------------- analyses
def regime_analysis(models: dict, episodes: Sequence, bcfg, noise: str, horizon: int, device, gaps: Sequence[str], n_boot: int) -> dict:
    ks = ks_for(horizon)
    out: dict = {"ks": ks, "regimes": {}}
    for reg in ("wind", "no_wind"):
        eps = R.select(episodes, wind=reg)
        if not eps:
            continue
        errs, eids = {}, None
        for name, m in models.items():
            set_phi_noise(m, noise)
            E = collect_errors(m, make_ds(eps, bcfg, m, noise), horizon=horizon, device=device)
            errs[name], eids = E["pe"], E["eids"]
        block = {"n_episodes": len(eps), "n_windows": int(len(eids)),
                 "pos_rmse_m": {n: rmse_curve(e, ks) for n, e in errs.items()}, "gaps": {}}
        for g in gaps:
            a, b = g.split(":")
            if a in errs and b in errs:
                block["gaps"][g] = paired_gap(errs[a], errs[b], eids, ks, n_boot)
        out["regimes"][reg] = block
    return out


def anchor_analysis(models: dict, episodes: Sequence, bcfg, noise: str, horizon: int, device, every: Sequence[int], mode: str) -> dict:
    ks = list(range(1, horizon + 1))
    out: dict = {"mode": mode, "every": list(every), "horizon": horizon, "models": {}}
    for name, m in models.items():
        has_latent = hasattr(m, "observe")
        if mode == "h" and not has_latent:
            continue
        set_phi_noise(m, noise)
        ds = make_ds(episodes, bcfg, m, noise, future_obs=(mode == "h"))
        open_pe = collect_errors(m, ds, horizon=horizon, device=device)["pe"]
        res = {"open_loop": rmse_curve(open_pe, ks), "n_windows": int(len(open_pe)), "anchored": {}}
        for K in every:
            pe = collect_errors(m, ds, horizon=horizon, device=device, anchor=Anchor(K, state=(mode == "state")))["pe"]
            curve = rmse_curve(pe, ks)
            since = {}                                   # steps since the last real observation (obs at steps K, 2K, …)
            for j in ks:
                since.setdefault(((j - 1) % K) + 1, []).append(pe[:, j - 1] ** 2)
            res["anchored"][str(K)] = {
                "rmse": curve,
                "rmse_by_steps_since_observation": {str(s): float(np.sqrt(np.mean(np.concatenate(v)))) for s, v in sorted(since.items())},
                "gain_vs_open_loop": {j: res["open_loop"][j] - curve[j] for j in ks}}
        if mode == "h" and "1" in res["anchored"]:
            # cost of the synthetic IMU φ against the real IMU at the same step (K = 1 re-observes at every step)
            res["phi_vs_real_imu_boundary"] = {j: res["open_loop"][j] - res["anchored"]["1"]["rmse"][j] for j in ks}
        out["models"][name] = res
    return out


def residual_analysis(models: dict, episodes: Sequence, bcfg, noise: str, horizon: int, device) -> dict:
    ks = ks_for(horizon)
    out: dict = {"ks": ks, "models": {}}
    for name, m in models.items():
        if not hasattr(m, "residual") or not hasattr(m, "observe"):
            continue
        set_phi_noise(m, noise)
        blocks = {}
        for reg in ("wind", "no_wind"):
            eps = R.select(episodes, wind=reg)
            if not eps:
                continue
            ds = make_ds(eps, bcfg, m, noise)
            diag = residual_diagnostics(m, ds, horizon=horizon, device=device)
            on = collect_errors(m, ds, horizon=horizon, device=device)["pe"]
            with residual_off(m):
                off = collect_errors(m, ds, horizon=horizon, device=device)["pe"]
            blocks[reg] = {**summarize_residual(diag), "pos_rmse_with_residual": rmse_curve(on, ks),
                           "pos_rmse_residual_zeroed": rmse_curve(off, ks),
                           "degradation_ratio": {k: rmse_curve(off, ks)[k] / max(rmse_curve(on, ks)[k], 1e-12) for k in ks}}
        out["models"][name] = blocks
    return out


def shuffle_analysis(models: dict, episodes: Sequence, bcfg, noise: str, horizon: int, device) -> dict:
    ks = ks_for(horizon)
    partners = partner_map(episodes)
    out: dict = {"ks": ks, "n_episodes_with_partner": len(partners), "models": {}}
    for name, m in models.items():
        set_phi_noise(m, noise)
        base = make_ds(episodes, bcfg, m, noise)
        orig, swapped = ActionSwap(base, partners, swap=False), ActionSwap(base, partners, swap=True)
        if len(orig) == 0:
            out["models"][name] = {"n_windows": 0}
            continue
        a = collect_errors(m, orig, horizon=horizon, device=device)["pe"]
        b = collect_errors(m, swapped, horizon=horizon, device=device)["pe"]
        ra, rb = rmse_curve(a, ks), rmse_curve(b, ks)
        out["models"][name] = {"n_windows": int(len(a)), "rmse_original_u": ra, "rmse_shuffled_u": rb,
                               "jump_ratio": {k: rb[k] / max(ra[k], 1e-12) for k in ks}}
    return out


# ----------------------------------------------------------------------------------- CLI
def markdown(kind: str, res: dict) -> str:
    L = []
    if kind == "regime":
        ks = res["ks"]
        for reg, b in res["regimes"].items():
            L += [f"\n### {reg} ({b['n_episodes']} episodes, {b['n_windows']} windows): position RMSE [m]\n",
                  "| model | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
            L += [f"| {n} | " + _fmt([c[k] for k in ks]) + " |" for n, c in b["pos_rmse_m"].items()]
            for g, v in b["gaps"].items():
                ci = v.get("gap_ci95")
                L.append(f"\ngap {g} (RMSE_a − RMSE_b): " + ", ".join(
                    f"k={k}: {v['gap'][i]:+.4f}" + (f" [{ci[0][i]:+.4f}, {ci[1][i]:+.4f}]" if ci else "") for i, k in enumerate(v["ks"])))
    elif kind == "anchor":
        for n, r in res["models"].items():
            L.append(f"\n### {n}: re-observation, mode={res['mode']} (position RMSE [m] at step 5 / {res['horizon']})")
            L.append(f"open loop: {r['open_loop'][5]:.4f} / {r['open_loop'][res['horizon']]:.4f}")
            for K, a in r["anchored"].items():
                L.append(f"every {K}: {a['rmse'][5]:.4f} / {a['rmse'][res['horizon']]:.4f}; by steps since observation: "
                         + ", ".join(f"{s}: {v:.4f}" for s, v in a["rmse_by_steps_since_observation"].items()))
            if "phi_vs_real_imu_boundary" in r:
                L.append("cost of φ vs real IMU (open − every-1) at k=2,5,10: "
                         + ", ".join(f"{r['phi_vs_real_imu_boundary'][k]:+.4f}" for k in (2, 5, 10) if k in r["phi_vs_real_imu_boundary"]))
    elif kind == "residual":
        for n, b in res["models"].items():
            for reg, v in b.items():
                L.append(f"\n### {n} / {reg}: |d_θ,v̇|/|specific force| at k=1: {v['ratio_r_acc_to_f_spec'][0]:.4f}, "
                         f"|d_θ,ω̇|/|ω̇_phys| at k=1: {v['ratio_r_ang_to_f_ang'][0]:.3f}; residual zeroed → RMSE ×"
                         + ", ".join(f"{v['degradation_ratio'][k]:.2f}" for k in sorted(v["degradation_ratio"])))
    else:
        for n, v in res["models"].items():
            if v.get("n_windows"):
                L.append(f"{n}: {v['n_windows']} windows; jump ratio (shuffled/original RMSE) at k="
                         + ", ".join(f"{k}: {v['jump_ratio'][k]:.2f}" for k in res["ks"]))
            else:
                L.append(f"{n}: no windows with a partner")
    return "\n".join(L)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="analysis", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["regime", "anchor", "residual", "shuffle"])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="evaluation defaults (config/evaluation/default.yaml); flags override")
    p.add_argument("--models", nargs="+", required=True, help="b1 | b6a_future | <kind>=<checkpoint.pt>")
    p.add_argument("--roles", nargs="*", default=["val_id"])
    p.add_argument("--cf-mode", default=None, choices=["closed_loop", "fixed_action", "none"])
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--noise", default="nominal")
    p.add_argument("--gap", nargs="*", default=[], help="pairs a:b of model names for the paired gap (regime command)")
    p.add_argument("--every", nargs="*", type=int, default=[1, 5, 10])
    p.add_argument("--mode", default="h", choices=["h", "state"])
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-episodes", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    pre = argparse.ArgumentParser(add_help=False)               # --config first: its values become the defaults of the other flags
    pre.add_argument("--config", type=Path, default=None)
    from physiworld_uav.training.config_io import eval_defaults
    p.set_defaults(**eval_defaults(pre.parse_known_args(argv)[0].config))
    a = p.parse_args(argv)
    device = pick_device(a.device)
    loaded = [load_model(s, a.dataset, device) for s in a.models]
    win = common_window([w for _, _, w in loaded], None)
    bcfg = BuilderConfig(decimation=dataset_decimation(a.dataset), history=win["history"], horizon=a.horizon, obs_mode=win["obs_mode"])
    models = {n: m for n, m, _ in loaded}
    for m in models.values():
        check_obs_mode(m, bcfg.obs_mode)
    eps = []
    for role in a.roles:
        eps += load_model_episodes(a.dataset, bcfg, roles=ROLE_GROUPS.get(role, [role]))
    if a.cf_mode:
        eps = R.select(eps, mode=a.cf_mode)
    if a.limit_episodes:
        eps = sorted(eps, key=lambda e: e.episode_id)[: a.limit_episodes]
    if not eps:
        raise SystemExit("no episodes selected")
    if a.command == "regime":
        res = regime_analysis(models, eps, bcfg, a.noise, a.horizon, device, a.gap, a.n_boot)
    elif a.command == "anchor":
        res = anchor_analysis(models, eps, bcfg, a.noise, a.horizon, device, a.every, a.mode)
    elif a.command == "residual":
        res = residual_analysis(models, eps, bcfg, a.noise, a.horizon, device)
    else:
        res = shuffle_analysis(models, eps, bcfg, a.noise, a.horizon, device)
    print(markdown(a.command, res))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
