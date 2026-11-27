"""Post-hoc probe: does the latent ``h`` carry the disturbance? (choice 4(c); no gradient into the model).

The recorded wrench is used **only here**, as regression targets for a frozen ridge probe fitted on the ``train`` role and
scored on other roles. Nothing privileged is a model input or a training target (thesis: "recorded for analysis and for oracle
experiments"). Targets (decision: both):

* ``total_force_W`` / ``total_torque_B`` : the recorded **total injected aero wrench**: what ``f_phys`` is missing. Not wind only
  (drag acts with zero wind too).
* ``wind_force_W`` / ``wind_torque_B`` : the **wind-attributable part** = total(wind) − total(zero wind), reconstructed with the
  same force models at the model rate (``data/disturbance.py``; check its error with ``reconstruction_error``).

Force is in the world frame, torque in the body frame.

For each rollout step k the probe predicts each target at step t+k from

* ``state``  : the model's own physical state before step k (v, q, ω; position-free) and u_k   (what physics already knows)
* ``h``      : the latent h_k
* ``h+state``: both

If ``h`` adds R² over ``state`` the latent carries disturbance information beyond the state; R² vs k shows how fast
that information goes stale in imagination (choice 6). Run it per noise preset: a clean IMU reveals the aero force almost
directly, so report the noisy setting.

    python -m physiworld_uav.evaluation.probe --dataset DIR --checkpoint outputs/experiments/b3/checkpoints/best.pt \\
        --noise nominal --out outputs/eval/probe_b3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physiworld_uav.data import disturbance as D
from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, dataset_decimation, dataset_sim, load_model_episodes
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.models.world_model import set_phi_noise
from physiworld_uav.training import checkpoint as ckpt
from physiworld_uav.training.trainer import BATCH_KEYS, build_from_checkpoint, pick_device
from physiworld_uav.models.world_model import check_obs_mode
from .run import checkpoint_window

TARGETS = {"total_force_W": D.TOTAL_FORCE, "total_torque_B": D.TOTAL_TORQUE,
           "wind_force_W": D.WIND_FORCE, "wind_torque_B": D.WIND_TORQUE}
FEATURES = ("state", "h", "h+state")


def ensure_targets(episodes) -> list[str]:
    """Attach the reconstructed wind-attributable target where possible; return the targets present in every episode."""
    for ep in episodes:
        if ep.meta is not None and D.WIND_FORCE not in ep.aux and D.TOTAL_FORCE in ep.aux:
            D.attach(ep)
    return [n for n, key in TARGETS.items() if all(key in ep.aux for ep in episodes)]


@torch.no_grad()
def collect(model, episodes, bcfg: BuilderConfig, ks, noise: str, device, max_windows: int, targets, seed: int = 0) -> dict:
    """Features per k and the disturbance targets, for up to ``max_windows`` windows (random subset, fixed seed).

    Alignment: the rollout step k goes from t+k to t+k+1 and the logged (interval-mean) wrench index t+k is the wrench acting
    during that step, so the latent h_k (before step k) is paired with the wrench of step k."""
    ds = ModelReadyDataset(episodes, bcfg, transform=ImuNoise(noise, 777) if noise != "none" else None)
    n = len(ds)
    idx = np.random.default_rng(seed).permutation(n)[: min(n, max_windows)]
    feats = {f: {k: [] for k in ks} for f in FEATURES}
    target = {t: {k: [] for k in ks} for t in targets}
    model.eval().to(device)
    H = max(ks) + 1
    for lo in range(0, len(idx), 256):
        sel = [int(i) for i in idx[lo: lo + 256]]
        items = [ds[i] for i in sel]
        batch = {k: torch.stack([it[k] for it in items]).to(device) for k in BATCH_KEYS}
        win = [ds.window(i) for i in sel]
        tg = {t: torch.stack([torch.from_numpy(np.asarray(ep.aux[TARGETS[t]][tt: tt + H])) for ep, tt in win]).float() for t in targets}
        out = model(batch, H, "given")
        s0 = batch["state_history"][:, -1:]
        s_before = torch.cat([s0, out["pred"][:, :-1]], 1)                                      # state before step k
        st = torch.cat([s_before[..., 3:6], s_before[..., 6:10], s_before[..., 10:13], batch["future_actions"][:, :H]], -1)
        for k in ks:
            hk, sk = out["h_traj"][:, k].cpu(), st[:, k].cpu()
            feats["state"][k].append(sk); feats["h"][k].append(hk); feats["h+state"][k].append(torch.cat([hk, sk], -1))
            for t in targets:
                target[t][k].append(tg[t][:, k])
    return {"feats": {f: {k: torch.cat(v).numpy() for k, v in d.items()} for f, d in feats.items()},
            "target": {t: {k: torch.cat(v).numpy() for k, v in d.items()} for t, d in target.items()}, "n": len(idx)}


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple:
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-8)
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    A = Z.T @ Z + alpha * np.diag([1.0] * (Z.shape[1] - 1) + [0.0])
    return mu, sd, np.linalg.solve(A, Z.T @ Y)


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    mu, sd, W = model
    return np.c_[(X - mu) / sd, np.ones(len(X))] @ W


def r2(pred: np.ndarray, y: np.ndarray) -> float | None:
    sst = float(((y - y.mean(0)) ** 2).sum())
    return None if sst < 1e-12 else float(1.0 - ((pred - y) ** 2).sum() / sst)


def run_probe(model, train_eps, role_eps: dict, bcfg, ks, noise: str, device, alpha: float = 1.0, max_windows: int = 20000) -> dict:
    set_phi_noise(model, noise)
    targets = ensure_targets(list(train_eps) + [e for v in role_eps.values() for e in v])
    tr = collect(model, train_eps, bcfg, ks, noise, device, max_windows, targets)
    fitted = {(t, f, k): ridge_fit(tr["feats"][f][k], tr["target"][t][k], alpha) for t in targets for f in FEATURES for k in ks}
    res = {"noise": noise, "ks": list(ks), "alpha": alpha, "n_train_windows": tr["n"], "targets": targets, "roles": {}}
    for role, eps in role_eps.items():
        if not eps:
            continue
        ev = collect(model, eps, bcfg, ks, noise, device, max_windows, targets)
        res["roles"][role] = {"n_windows": ev["n"], "r2": {t: {f: {str(k): r2(ridge_predict(fitted[(t, f, k)], ev["feats"][f][k]), ev["target"][t][k])
                                                                 for k in ks} for f in FEATURES} for t in targets}}
    return res


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="probe", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--roles", nargs="*", default=["val_id", "test_id", "test_ood_magnitude"])
    p.add_argument("--ks", nargs="*", type=int, default=[0, 5, 10, 20])
    p.add_argument("--noise", default="nominal")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-windows", type=int, default=20000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    device = pick_device(a.device)
    state = ckpt.load(a.checkpoint, map_location=device)
    model = build_from_checkpoint(state)
    win = checkpoint_window(state)                  # history / obs_mode the model was trained with
    if model.name not in ("b2", "b3", "b5a", "b5b"):
        p.error("the probe needs a latent model without an oracle input (b2, b3, b5a, b5b)")
    if abs(float(state["dt"]) - dataset_sim(a.dataset)["model_dt"]) > 1e-9:
        p.error(f"the checkpoint was trained at a model step of {state['dt']} s, the dataset has {dataset_sim(a.dataset)['model_dt']} s")
    bcfg = BuilderConfig(decimation=dataset_decimation(a.dataset), history=win["history"], horizon=max(a.ks) + 1, obs_mode=win["obs_mode"])
    check_obs_mode(model, bcfg.obs_mode)
    train_eps = load_model_episodes(a.dataset, bcfg, roles=["train"])
    role_eps = {r: load_model_episodes(a.dataset, bcfg, roles=[r]) for r in a.roles}
    res = run_probe(model, train_eps, role_eps, bcfg, a.ks, a.noise, device, a.alpha, a.max_windows)
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
