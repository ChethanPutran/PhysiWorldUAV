#!/usr/bin/env python3
"""Distribution of the wind in a dataset: per time step, per episode, per role. Run INSIDE the container.

    python scripts/plot_wind_distribution.py --dataset data/datasets/physiworld_pilot_v0.6.0 --out outputs/reports/wind_distribution

Wind = environment/wind_applied_W (world frame, m/s) at the model rate (50 Hz). Writes figures/*.png and tables/*.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physiworld_uav.data.dataset_builder import BuilderConfig, load_model_episodes

ROLES = ["train", "val_id", "test_id", "val_ood_magnitude", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal",
         "test_ood_gust", "test_ood_physics", "test_ood_trajectory"]
PAL = {"train": "#0072B2", "val_id": "#56B4E9", "test_id": "#009E73", "val_ood_magnitude": "#E69F00", "test_ood_magnitude": "#D55E00",
       "test_ood_direction": "#CC79A7", "test_ood_temporal": "#888888", "test_ood_gust": "#B8860B", "test_ood_physics": "#7B4EA3", "test_ood_trajectory": "#444444"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (a.out / "figures").mkdir(parents=True, exist_ok=True); (a.out / "tables").mkdir(exist_ok=True)
    bcfg = BuilderConfig(decimation=4, history=10, horizon=20, obs_mode="instant")
    data, rows = {}, []
    for role in ROLES:
        try:
            eps = load_model_episodes(a.dataset, bcfg, roles=[role])
        except Exception as e:  # noqa: BLE001
            print("skip", role, e); continue
        sp, per = [], []
        for e in eps:
            w = np.asarray(e.aux["environment/wind_applied_W"], dtype=float)
            s = np.linalg.norm(w, axis=1); sp.append(s)
            per.append((s.mean(), s.max(), float(np.linalg.norm(w.mean(0))), np.degrees(np.arctan2(w[:, 2], np.linalg.norm(w[:, :2], axis=1))).mean() if s.max() > 0.1 else 0.0, len(s)))
        if sp:
            data[role] = (np.concatenate(sp), np.array(per))
    qs = (50, 90, 99)
    for role, (s, per) in data.items():
        rows.append([role, len(per), len(s), f"{(s < 0.1).mean():.3f}", f"{(s < 1).mean():.3f}", f"{(s > 3).mean():.3f}", f"{(s > 6).mean():.3f}", f"{(s > 8).mean():.3f}"]
                    + [f"{np.percentile(s, q):.2f}" for q in qs] + [f"{s.max():.2f}", f"{(per[:, 1] < 0.1).mean():.2f}"])
    hdr = ["role", "episodes", "time steps", "frac <0.1 m/s", "frac <1 m/s", "frac >3 m/s", "frac >6 m/s", "frac >8 m/s", "p50", "p90", "p99", "max", "episodes with no wind"]
    with open(a.out / "tables" / "wind_by_role.csv", "w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr)); [print("| " + " | ".join(map(str, r)) + " |") for r in rows]

    bins = np.linspace(0, 10, 51)
    # 1: per time step, density, log y
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for role, (s, _) in data.items():
        h, _ = np.histogram(s, bins=bins, density=False); ax[0].plot(0.5 * (bins[1:] + bins[:-1]), h / max(len(s), 1), color=PAL[role], lw=1.8, label=f"{role} ({len(s) // 1000}k steps)")
        ax[1].plot(np.sort(s), 1 - np.arange(1, len(s) + 1) / len(s), color=PAL[role], lw=1.8)
    ax[0].set_yscale("log"); ax[0].set_xlabel("wind speed [m/s]"); ax[0].set_ylabel("fraction of time steps per 0.2 m/s bin"); ax[0].set_title("Wind speed per time step, by role (log y)")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=0.3)
    ax[1].set_yscale("log"); ax[1].set_xlim(0, 10); ax[1].set_xlabel("wind speed [m/s]"); ax[1].set_ylabel("fraction of time steps above this speed"); ax[1].set_title("Exceedance (log y)"); ax[1].grid(alpha=0.3)
    for x in (4, 6, 8):
        ax[1].axvline(x, color="#999", ls=":", lw=1)
    fig.tight_layout(); fig.savefig(a.out / "figures" / "wind_speed_per_timestep.png", dpi=140); plt.close(fig)
    # 2: per episode mean vs peak, by role
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for role, (_, per) in data.items():
        ax[0].scatter(per[:, 0], per[:, 1], s=26, color=PAL[role], alpha=0.85, label=role, edgecolors="none")
    ax[0].set_xlabel("episode mean wind speed [m/s]"); ax[0].set_ylabel("episode peak wind speed [m/s]"); ax[0].set_title("One dot per episode"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=2)
    allp = np.concatenate([p[:, 1] for _, p in data.values()])
    ax[1].hist([p[:, 1] for _, p in data.values()], bins=np.linspace(0, 10, 21), stacked=True, color=[PAL[r] for r in data], label=list(data))
    ax[1].set_xlabel("episode peak wind speed [m/s]"); ax[1].set_ylabel("episodes"); ax[1].set_title(f"Episodes by peak wind ({len(allp)} episodes; {(allp < 0.1).sum()} with no wind)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out / "figures" / "wind_per_episode.png", dpi=140); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
