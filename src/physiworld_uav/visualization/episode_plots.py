"""Quick-look plots of one recorded episode (matplotlib, headless).

Panels: position vs reference, tracking error, applied wind, aerodynamic force
by component, rotor commands (with saturation marks) and tilt angle. Reads the
HDF5 shard read-only, so it also works on a shard that is still being written.

    python -m physiworld_uav.visualization.episode_plots --dataset DIR --list
    python -m physiworld_uav.visualization.episode_plots --dataset DIR --episode id_cl_g00000_e0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Read-only access must not fight the generator's write lock.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _shards(dataset_dir: Path) -> list[Path]:
    shards = sorted(Path(dataset_dir).glob("shard_*.h5"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.h5 in {dataset_dir}")
    return shards


def list_episodes(dataset_dir: Path) -> list[str]:
    ids: list[str] = []
    for s in _shards(dataset_dir):
        with h5py.File(s, "r") as f:
            ids += list(f["episodes"].keys())
    return ids


def _open_episode(dataset_dir: Path, episode_id: str):
    for s in _shards(dataset_dir):
        f = h5py.File(s, "r")
        if episode_id in f["episodes"]:
            return f, f["episodes"][episode_id]
        f.close()
    raise KeyError(f"episode {episode_id!r} not found in {dataset_dir}")


def _get(g, path: str):
    return np.asarray(g[path]) if path in g else None


def _tilt_deg(q_wxyz: np.ndarray) -> np.ndarray:
    """Angle between body z and world z for Hamilton wxyz body→world quaternions."""
    w, x, y, z = q_wxyz.T
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))


def plot_episode(dataset_dir: str | Path, episode_id: str, out: str | Path | None = None):
    dataset_dir = Path(dataset_dir)
    f, g = _open_episode(dataset_dir, episode_id)
    try:
        t = _get(g, "time/t")
        pos = _get(g, "ground_truth/position_W")
        q = _get(g, "ground_truth/orientation_wxyz")
        ref = _get(g, "control/reference/position_W")
        err = _get(g, "control/tracking_error/position_W")
        wind = _get(g, "environment/wind_applied_W")
        rot_cmd = _get(g, "actions/rotor_cmd")
        sat = _get(g, "actions/saturated")
        rd = _get(g, "ground_truth/wrench/rotor_drag_force_W")
        bd = _get(g, "ground_truth/wrench/body_drag_force_W")
    finally:
        f.close()

    tc = t[:-1]                                           # per-control-step series have N samples; states have N+1
    fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle(episode_id)

    a = ax[0, 0]
    for i, name in enumerate("xyz"):
        a.plot(t, pos[:, i], label=name)
        if ref is not None and not np.isnan(ref[:, i]).all():
            a.plot(tc, ref[:, i], "--", color=a.lines[-1].get_color(), alpha=0.6)
    a.set(ylabel="position [m]", title="Position (dashed: reference)")
    a.legend(ncol=3, loc="best")

    a = ax[0, 1]
    if err is not None and not np.isnan(err).all():
        a.plot(tc, np.linalg.norm(err, axis=1), color="C0")
        a.set(title="Position tracking error")
    else:
        a.text(0.5, 0.5, "no reference (open loop)", ha="center", va="center", transform=a.transAxes)
        a.set(title="Position tracking error")
    a.set(ylabel="‖e‖ [m]")

    a = ax[1, 0]
    if wind is not None:
        for i, name in enumerate("xyz"):
            a.plot(tc, wind[:, i], label=f"w_{name}")
        a.plot(tc, np.linalg.norm(wind, axis=1), "k", lw=1.2, label="|w|")
        a.legend(ncol=4, loc="best")
    a.set(ylabel="wind [m/s]", title="Applied wind (world frame)")

    a = ax[1, 1]
    if rd is not None:
        a.plot(tc, np.linalg.norm(rd, axis=1), label="rotor drag")
    if bd is not None:
        a.plot(tc, np.linalg.norm(bd, axis=1), label="body drag")
    a.set(ylabel="force [N]", title="Aerodynamic force by component")
    a.legend(loc="best")

    a = ax[2, 0]
    if rot_cmd is not None:
        for k in range(rot_cmd.shape[1]):
            a.plot(tc, rot_cmd[:, k], lw=0.9, label=f"rotor {k}")
        if sat is not None and sat.any():
            idx = np.where(sat.any(axis=1))[0]
            a.plot(tc[idx], np.zeros(len(idx)) + float(rot_cmd.min()), "r|", label="saturated")
        a.legend(ncol=5, loc="best", fontsize=8)
    a.set(ylabel="command", xlabel="time [s]", title="Rotor commands")

    a = ax[2, 1]
    a.plot(t, _tilt_deg(q), color="C3")
    a.set(ylabel="tilt [deg]", xlabel="time [s]", title="Tilt from vertical")

    for a in ax.flat:
        a.grid(alpha=0.25)
    if out is not None:
        out = Path(out)
        fig.savefig(out, dpi=130)
    return fig


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="episode-plots", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episode", default=None, help="episode id (see --list)")
    p.add_argument("--list", action="store_true", help="list episode ids in the dataset and exit")
    p.add_argument("--out", type=Path, default=None, help="PNG path (default: <dataset>/plots/<episode>.png)")
    a = p.parse_args(argv)
    if a.list:
        print("\n".join(list_episodes(a.dataset)))
        return
    if not a.episode:
        p.error("--episode is required unless --list is given")
    out = a.out or (a.dataset / "plots" / f"{a.episode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_episode(a.dataset, a.episode, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
