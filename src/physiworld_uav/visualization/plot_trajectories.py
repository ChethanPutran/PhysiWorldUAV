"""Actual vs predicted trajectories, with the wind of every episode (magnitude and direction) in the legend. Run INSIDE the container.

    python -m physiworld_uav.visualization trajectories --dataset data/datasets/physiworld_pilot_v7_v0.7.0 \\
        --models b1 st_v07_b3_s0=outputs/experiments/st_v07_b3_s0/checkpoints/best.pt --roles test_id test_ood_magnitude test_ood_direction \\
        --out outputs/reports/trajectories_v7

Each prediction is an open-loop rollout of ``--horizon`` steps started from the TRUE state at every ``--stride``-th step of the episode
(so a figure shows how the model drifts from the truth over one horizon, many times, not one long rollout). Per episode one PNG:
top view (x-y) with the wind arrow, altitude vs time, and position error vs time. The legend names the wind speed [m/s] and the
azimuth [deg, from +x towards +y]. ``summary.png`` shows, for every selected episode, the position error at the last horizon step
against the wind speed (magnitude) and the wind azimuth (direction), coloured by role."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, dataset_decimation, load_model_episodes
from physiworld_uav.data.noise import ImuNoise

PAL = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7B4EA3"]
ROLE_PAL = {"val_id": "#56B4E9", "test_id": "#009E73", "test_ood_magnitude": "#D55E00", "test_ood_direction": "#CC79A7",
            "test_ood_temporal": "#888888", "test_ood_gust": "#B8860B", "test_ood_physics": "#7B4EA3", "test_ood_trajectory": "#444444"}


def wind_of(ep) -> tuple[float, float]:
    """(mean speed [m/s], azimuth [deg]) of the steady wind of an episode's plan; (0, nan) without wind."""
    w = ep.meta.plan.get("wind") or {}
    st = (w.get("spec") or {}).get("steady") or {}
    if not st.get("enabled", False) or float(st.get("speed_mps", 0.0)) <= 0.0:
        return 0.0, float("nan")
    d = st["direction"]
    return float(st["speed_mps"]), float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def wind_label(ep) -> str:
    s, az = wind_of(ep)
    return "no wind" if s == 0.0 else f"wind {s:.1f} m/s, {az:.0f}°"


def pick_episodes(eps: list, n: int) -> list:
    """n episodes of a role spread over the wind speed (lowest, highest and evenly between), deterministic."""
    eps = sorted(eps, key=lambda e: (wind_of(e)[0], e.episode_id))
    if len(eps) <= n:
        return eps
    return [eps[int(round(i))] for i in np.linspace(0, len(eps) - 1, n)]


@torch.no_grad()
def rollouts(model, ds: ModelReadyDataset, ep, horizon: int, stride: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Predicted positions (W, horizon, 3) and the start indices (W,) of the windows of ``ep`` (every ``stride``-th)."""
    idx = [i for i in range(len(ds)) if ds.window(i)[0] is ep]
    idx = idx[::stride]
    batch = torch.utils.data.default_collate([ds[i] for i in idx])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.eval().to(device)
    init = "given"
    pred = model.predict(batch, horizon, init)[..., :3].cpu().numpy()
    return pred, np.array([ds.window(i)[1] for i in idx])


def plot_episode(ep, preds: dict, starts: np.ndarray, dt: float, horizon: int, role: str, path: Path) -> None:
    p = ep.states[:, :3]
    t = np.arange(len(p)) * dt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.2, 1.1, 1.1]})
    ax[0].plot(p[:, 0], p[:, 1], color="black", lw=1.6, label="actual")
    ax[1].plot(t, p[:, 2], color="black", lw=1.6, label="actual")
    s, az = wind_of(ep)
    for c, (name, pr) in zip(PAL, preds.items()):
        errs = []
        for j, t0 in enumerate(starts):
            seg = np.vstack([p[t0], pr[j]])
            ax[0].plot(seg[:, 0], seg[:, 1], color=c, lw=0.9, alpha=0.8, label=name if j == 0 else None)
            ax[1].plot(t[t0:t0 + horizon + 1], seg[:, 2], color=c, lw=0.9, alpha=0.8)
            errs.append(np.linalg.norm(pr[j] - p[t0 + 1:t0 + horizon + 1], axis=1))
        err = np.mean(errs, axis=0)
        ax[2].plot((np.arange(1, horizon + 1)) * dt, err, color=c, lw=1.8, label=f"{name} (k={horizon}: {err[-1]:.3f} m)")
    if s > 0:                                                                        # wind arrow, drawn at the start of the path
        span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
        L = 0.25 * span
        ux, uy = np.cos(np.radians(az)), np.sin(np.radians(az))
        ax[0].annotate("", xy=(p[0, 0] + L * ux, p[0, 1] + L * uy), xytext=(p[0, 0], p[0, 1]),
                       arrowprops=dict(arrowstyle="-|>", color="#1b6ca8", lw=2.5))
    ax[0].plot([], [], color="#1b6ca8", lw=2.5, label=f"{wind_label(ep)} (arrow = wind direction)")
    ax[0].set(xlabel="x [m] (east)", ylabel="y [m] (north)", title="top view"); ax[0].set_aspect("equal", "datalim")
    ax[1].set(xlabel="time [s]", ylabel="z [m]", title="altitude")
    ax[2].set(xlabel="time into the rollout [s]", ylabel="position error [m]", title=f"mean error over {len(starts)} rollouts")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8, loc="best")
    fig.suptitle(f"{ep.episode_id} — {role} — {wind_label(ep)}", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_summary(rows: list[dict], models: list[str], path: Path) -> None:
    """Final-step position error of each selected episode against wind speed and wind azimuth."""
    fig, ax = plt.subplots(len(models), 2, figsize=(12, 3.6 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for r in rows:
            c = ROLE_PAL.get(r["role"], "#444444")
            ax[i, 0].scatter(r["speed"], r["err"][m], color=c, s=28)
            if not np.isnan(r["az"]):
                ax[i, 1].scatter(r["az"], r["err"][m], color=c, s=28)
        ax[i, 0].set(xlabel="wind speed [m/s] (magnitude)", ylabel=f"{m}\nposition error at last step [m]")
        ax[i, 1].set(xlabel="wind azimuth [deg] (direction)"); ax[i, 1].set_xlim(0, 360)
        for a in ax[i]:
            a.grid(alpha=0.3)
    for role, c in ROLE_PAL.items():
        if any(r["role"] == role for r in rows):
            ax[0, 0].scatter([], [], color=c, label=role)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    from physiworld_uav.evaluation.run import checkpoint_window, common_window, load_model
    from physiworld_uav.models.world_model import set_phi_noise
    from physiworld_uav.training.trainer import pick_device
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True, help="b0 | b1 | <name>=<checkpoint.pt>")
    ap.add_argument("--roles", nargs="+", default=["test_id", "test_ood_magnitude", "test_ood_direction"])
    ap.add_argument("--per-role", type=int, default=3, help="episodes per role, spread over wind speed")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=20, help="steps between two rollout starts")
    ap.add_argument("--noise", default="nominal")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    device = pick_device(a.device)
    loaded = [load_model(s, a.dataset, device) for s in a.models]
    win = common_window([w for _, _, w in loaded], None)
    bcfg = BuilderConfig(decimation=dataset_decimation(a.dataset), history=win["history"], horizon=a.horizon, obs_mode=win["obs_mode"])
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for role in a.roles:
        eps = pick_episodes(load_model_episodes(a.dataset, bcfg, roles=[role]), a.per_role)
        if not eps:
            print("no episodes in", role); continue
        tf = ImuNoise(a.noise, 777) if a.noise != "none" else None
        for ep in eps:
            preds, starts = {}, None
            for name, model, _ in loaded:
                set_phi_noise(model, a.noise)
                ds = ModelReadyDataset([ep], bcfg, transform=tf, oracle=getattr(model, "needs_oracle", False))
                preds[name], starts = rollouts(model, ds, ep, a.horizon, a.stride, device)
            plot_episode(ep, preds, starts, ep.dt, a.horizon, role, a.out / f"{role}_{ep.episode_id}.png")
            s, az = wind_of(ep)
            rows.append({"role": role, "speed": s, "az": az, "err": {
                n: float(np.mean([np.linalg.norm(pr[j, -1] - ep.states[t0 + a.horizon, :3]) for j, t0 in enumerate(starts)])) for n, pr in preds.items()}})
            print("wrote", role, ep.episode_id, wind_label(ep))
    if rows:
        plot_summary(rows, [n for n, _, _ in loaded], a.out / "summary.png")


if __name__ == "__main__":
    main()
