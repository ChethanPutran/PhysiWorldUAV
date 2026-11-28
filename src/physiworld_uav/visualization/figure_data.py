"""Data for the report figures that needs the HDF5 shards or checkpoints (run INSIDE the container; torch, h5py).

    python -m physiworld_uav.visualization figure-data [--dataset data/datasets/<v7>] [--cache outputs/figure_cache] [--only traj extra errors]

Writes into the cache folder: ``traj_data.npz`` (open-loop rollouts of B1, B2, B3, B4 on chosen test episodes), ``extra.npz`` (trajectory
coverage, wind time series per profile, clean vs noisy IMU, counterfactual siblings) and ``err_data.json`` (per-episode errors of B1 (true
parameters), B2, B3, B4 and constant velocity on the v7 test roles, with the wind of each episode). ``figures`` then plots them.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, dataset_decimation, load_model_episodes
from physiworld_uav.data.noise import ImuNoise, ImuNoiseConfig
from physiworld_uav.evaluation.run import common_window, load_model
from physiworld_uav.models.world_model import set_phi_noise
from physiworld_uav.training.trainer import pick_device
from physiworld_uav.visualization.plot_trajectories import pick_episodes, rollouts, wind_of

KV_MODELS = ("b2", "b3", "b4_rw")      # known-vehicle runs (seed 0) used for the figures
TEST_ROLES = ["test_id", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory"]


class Shards:
    """episode id -> HDF5 shard, from the dataset index."""

    def __init__(self, ds: Path) -> None:
        idx = json.load(open(ds / "index.json"))["episodes"]
        self.ds, self.shard = ds, {e["episode_id"]: e["shard"] for e in idx}

    def has(self, ep: str) -> bool:
        return ep in self.shard

    def get(self, ep: str, path: str) -> np.ndarray:
        with h5py.File(self.ds / self.shard[ep], "r") as h:
            return h[f"episodes/{ep}/{path}"][()]


def _loaded(ds: Path, dev, kv_head: str):
    specs = [kv_head] + [f"{n}=outputs/experiments/kv_v07_{n}_s0/checkpoints/best.pt" for n in KV_MODELS]
    loaded = [load_model(s, ds, dev) for s in specs]
    win = common_window([w for _, _, w in loaded], None)
    return loaded, BuilderConfig(decimation=dataset_decimation(ds), history=win["history"], horizon=20, obs_mode=win["obs_mode"])


def build_traj(ds: Path, cache: Path) -> None:
    """Open-loop rollouts (1 s, started every second) of B1 (nominal parameters), B2, B3, B4 on three episodes of three roles."""
    dev = pick_device("auto"); loaded, bcfg = _loaded(ds, dev, "b1"); out = {}
    for role in ("test_id", "test_ood_magnitude", "test_ood_direction"):
        for ep in pick_episodes(load_model_episodes(ds, bcfg, roles=[role]), 3):
            key = f"{role}|{ep.episode_id}"; preds = {}
            for name, model, _ in loaded:
                set_phi_noise(model, "nominal")
                d = ModelReadyDataset([ep], bcfg, transform=ImuNoise("nominal", 777), oracle=getattr(model, "needs_oracle", False))
                preds[name], starts = rollouts(model, d, ep, 20, 20, dev)
            s, az = wind_of(ep)
            out[key + "|truth"], out[key + "|starts"], out[key + "|dt"], out[key + "|wind"] = ep.states[:, :3], starts, np.array(ep.dt), np.array([s, az])
            for k, v in preds.items():
                out[key + "|pred_" + k] = v
    np.savez(cache / "traj_data.npz", **out); print("wrote traj_data.npz")


def build_extra(ds: Path, cache: Path) -> None:
    man = [json.loads(l) for l in open(ds / "manifest.jsonl")]; sh = Shards(ds); man = [m for m in man if sh.has(m["episode_id"])]
    out, rng = {}, np.random.default_rng(0)
    # trajectory coverage: 4 training episodes per closed-loop template
    by_t = collections.defaultdict(list)
    for m in man:
        if m["split_role"] == "train" and m["data_type"] == "closed_loop":
            by_t[m["scenario_template"]].append(m["episode_id"])
    for t, ids in by_t.items():
        for j, e in enumerate(rng.choice(ids, size=min(4, len(ids)), replace=False)):
            out[f"cov|{t}|{j}"] = sh.get(e, "ground_truth/position_W")[::10]
    # wind time series, one episode per profile (the one with the median peak among those with wind above 1 m/s)
    for prof in ("steady", "ou", "cosine_gust", "step", "sinusoid", "random_gusts"):
        ids = [m["episode_id"] for m in man if m["wind"]["profile"] == prof and m["wind"]["stats"]["peak_speed"] > 1.0 and m["split_role"] in ("train", "test_id", "test_ood_gust", "test_ood_temporal")]
        best, best_ptp = None, -1.0
        for e in rng.choice(ids, size=min(60, len(ids)), replace=False):      # the example with the most visible time variation
            w = sh.get(e, "environment/wind_W")[::5]; ptp = float(np.ptp(np.linalg.norm(w, axis=1)))
            if ptp > best_ptp or (prof == "steady" and best is None):
                best, best_ptp, bw = e, ptp, w
        out[f"wind|{prof}"] = bw; out[f"wind|{prof}|id"] = np.array(best)
    # counterfactual siblings: same group, wind scaled 0 / 0.5 / 1.0
    g = collections.defaultdict(list)
    for m in man:
        g[m["group_id"]].append(m)
    cands = [v for v in g.values() if len(v) == 3 and v[0]["data_type"] == "closed_loop" and max(x["wind"]["stats"]["mean_speed"] for x in v) > 5.0
             and v[0]["scenario_template"] in ("circle", "translation", "line", "combined")]
    grp = cands[0] if cands else [v for v in g.values() if len(v) == 3][0]
    for m in grp:
        cf = m["counterfactual"] or {}; diff = cf.get("diff", {})
        scale = float(diff["wind/spec/ood/wind_scale"][1]) if "wind/spec/ood/wind_scale" in diff else (0.0 if cf.get("is_baseline") else 1.0)   # the baseline has wind scale 0
        out[f"cf|{m['episode_id']}|pos"] = sh.get(m["episode_id"], "ground_truth/position_W")[::5]
        out[f"cf|{m['episode_id']}|wind"] = np.array([m["wind"]["stats"]["mean_speed"], scale]); out[f"cf|{m['episode_id']}|az"] = np.array(wind_of_plan(m))
    out["cf|group"] = np.array(grp[0]["group_id"])
    # dynamics-parameter siblings (nominal vs perturbed vehicle, same reference and wind)
    dg = collections.defaultdict(list)
    for m in man:
        if str(m["domain"]["domain_id"]).startswith("nominal"):
            dg[m["group_id"]].append(m)
    for k, v in dg.items():
        if len(v) >= 3 and v[0]["data_type"] != "open_loop":
            for m in v:
                out[f"cfd|{m['episode_id']}|pos"] = sh.get(m["episode_id"], "ground_truth/position_W")[::5]
                out[f"cfd|{m['episode_id']}|mass"] = np.array(m["domain"]["drone"]["mass_kg"]); out[f"cfd|{m['episode_id']}|dom"] = np.array(m["domain"]["domain_id"])
            out["cfd|group"] = np.array(k); break
    # clean vs noisy IMU (20 Hz model rate), same noise model as the dataset transform (bias per 10-sample window + white noise)
    dev = pick_device("auto"); _, bcfg = _loaded(ds, dev, "b1")
    ep = [e for e in load_model_episodes(ds, bcfg, roles=["test_id"]) if e.meta.plan["scenario_template"] == "circle"][0]
    obs = torch.tensor(ep.obs, dtype=torch.float32); cfg = ImuNoiseConfig.preset("nominal"); noisy = obs.clone(); gen = torch.Generator().manual_seed(1)
    sw = torch.tensor([cfg.accel_white] * 3 + [cfg.gyro_white] * 3); sb = torch.tensor([cfg.accel_bias] * 3 + [cfg.gyro_bias] * 3)
    for s0 in range(0, len(obs), 10):
        n = min(10, len(obs) - s0); noisy[s0:s0 + n] += torch.randn(n, 6, generator=gen) * sw + torch.randn(6, generator=gen) * sb
    out["imu|clean"], out["imu|noisy"], out["imu|dt"], out["imu|ep"] = obs.numpy(), noisy.numpy(), np.array(ep.dt), np.array(ep.episode_id)
    np.savez(cache / "extra.npz", **out); print("wrote extra.npz")


def wind_of_plan(m: dict) -> float:
    st = (m["wind"]["spec"].get("steady") or {})
    if not st.get("enabled", False):
        return float("nan")
    d = st["direction"]; return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def build_errors(ds: Path, cache: Path, stride: int = 10, horizon: int = 20) -> None:
    """Position / velocity error at the end of the horizon per episode (windows every ``stride`` steps) for B1 (true parameters), B2, B3, B4 and constant velocity."""
    dev = pick_device("auto"); loaded, bcfg = _loaded(ds, dev, "b1_kv"); man = {json.loads(l)["episode_id"]: json.loads(l) for l in open(ds / "manifest.jsonl")}; rows = []
    for role in TEST_ROLES:
        eps = load_model_episodes(ds, bcfg, roles=[role])
        for ep in eps:
            s_, az = wind_of(ep); w = man[ep.episode_id]["wind"]["stats"]; res = {}
            for name, model, _ in loaded:
                set_phi_noise(model, "nominal")
                d = ModelReadyDataset([ep], bcfg, transform=ImuNoise("nominal", 777), oracle=getattr(model, "needs_oracle", False))
                idx = [i for i in range(len(d)) if d.window(i)[0] is ep][::stride]
                if not idx:
                    continue
                batch = torch.utils.data.default_collate([d[i] for i in idx]); batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                model.eval().to(dev)
                with torch.no_grad():
                    pr = model.predict(batch, horizon, "given").cpu().numpy()
                st = np.array([d.window(i)[1] for i in idx]); tru = np.stack([ep.states[s0 + 1: s0 + horizon + 1] for s0 in st])
                res[name] = (np.linalg.norm(pr[..., :3] - tru[..., :3], axis=-1), np.linalg.norm(pr[..., 3:6] - tru[..., 3:6], axis=-1))
                if name == "b1_kv":                 # constant-velocity reference on the same windows
                    k = (np.arange(1, horizon + 1) * ep.dt)[None, :, None]; cvp = ep.states[st, None, :3] + ep.states[st, None, 3:6] * k
                    res["cv"] = (np.linalg.norm(cvp - tru[..., :3], axis=-1), np.linalg.norm(ep.states[st, None, 3:6] - tru[..., 3:6], axis=-1))
            for name, (pe, ve) in res.items():
                r = lambda a: float(np.sqrt((a ** 2).mean()))   # noqa: E731
                rows.append({"role": role, "ep": ep.episode_id, "model": name, "wind": float(w["mean_speed"]), "peak": float(w["peak_speed"]), "az": None if az != az else float(az),
                             "pos_rmse_20": r(pe[:, -1]), "pos_rmse_10": r(pe[:, 9]), "pos_rmse_5": r(pe[:, 4]), "vel_rmse_20": r(ve[:, -1]), "pos_max_20": float(pe[:, -1].max()), "n": int(len(pe))})
        print(role, len(eps), flush=True)
    json.dump(rows, open(cache / "err_data.json", "w")); print("wrote err_data.json", len(rows))



def build_failure(ds: Path, cache: Path) -> None:
    """Rollouts (1 s, every second) of B1, B2, B3, B4 on the worst B3 episode of the OOD-magnitude role (needs err_data.json from ``errors``)."""
    rows = json.load(open(cache / "err_data.json"))
    worst = max((r for r in rows if r["model"] == "b3" and r["role"] == "test_ood_magnitude"), key=lambda r: r["pos_rmse_20"])["ep"]
    dev = pick_device("auto")
    specs = ["b1"] + [f"{n}=outputs/experiments/kv_v07_{n}_s0/checkpoints/best.pt" for n in KV_MODELS]
    loaded = [load_model(s, ds, dev) for s in specs]
    win = common_window([w for _, _, w in loaded], None)
    bcfg = BuilderConfig(decimation=dataset_decimation(ds), history=win["history"], horizon=20, obs_mode=win["obs_mode"])
    ep = next(e for e in load_model_episodes(ds, bcfg, roles=["test_ood_magnitude"]) if e.episode_id == worst)
    out, key = {}, f"fail|{worst}"
    for name, model, _ in loaded:
        set_phi_noise(model, "nominal")
        d = ModelReadyDataset([ep], bcfg, transform=ImuNoise("nominal", 777), oracle=getattr(model, "needs_oracle", False))
        out[f"{key}|pred_{name}"], starts = rollouts(model, d, ep, 20, 20, dev)
    out[key + "|truth"], out[key + "|starts"], out[key + "|dt"] = ep.states[:, :3], starts, np.array(ep.dt)
    np.savez(cache / "traj_fail.npz", **out); print("wrote traj_fail.npz", worst)


def build_quality(ds: Path, cache: Path) -> None:
    """Histogram counts of the state / IMU / action channels and of the physical-consistency residuals by split family (model-ready v7 episodes), and the wind speed per time step by role group."""
    from physiworld_uav.data.statistical_validator import physical_residuals
    man = {json.loads(l)["episode_id"]: json.loads(l) for l in open(ds / "manifest.jsonl")}
    _, bcfg = _loaded(ds, pick_device("auto"), "b1")
    fam = lambda r: "train" if r == "train" else ("ID (val/test)" if r in ("val_id", "test_id") else ("physics check" if r == "physics_validation" else "OOD"))   # noqa: E731
    chans = {"speed": (lambda e: np.linalg.norm(e.states[:, 3:6], axis=1), np.linspace(0, 12, 61)), "z": (lambda e: e.states[:, 2], np.linspace(0, 6, 61)),
             "tilt": (lambda e: np.degrees(np.arccos(np.clip(1 - 2 * (e.states[:, 7] ** 2 + e.states[:, 8] ** 2), -1, 1))), np.linspace(0, 70, 61)),
             "omega": (lambda e: np.linalg.norm(e.states[:, 10:13], axis=1), np.linspace(0, 6, 61)), "fz": (lambda e: e.obs[:, 2], np.linspace(6, 14, 61)), "rotor0": (lambda e: e.actions[:, 0], np.linspace(330, 580, 61))}
    res_bins = {"pos_vel": np.linspace(0, 0.03, 61), "acc_imu": np.linspace(0, 2.0, 61), "gyro": np.linspace(0, 3e-4, 61), "att_rate": np.linspace(0, 0.03, 61)}
    out, H = {}, {}
    for e in load_model_episodes(ds, bcfg):
        f = fam(man[e.episode_id]["split_role"])
        for c, (fn, b) in chans.items():
            H.setdefault(("ch", c, f), np.zeros(len(b) - 1)); H[("ch", c, f)] += np.histogram(fn(e), b)[0]
        r = physical_residuals(e)
        for k, b in res_bins.items():
            H.setdefault(("res", k, "all"), np.zeros(len(b) - 1)); H[("res", k, "all")] += np.histogram(r[k], b)[0]
    for (kind, c, f), v in H.items():
        out[f"{kind}|{c}|{f}"] = v
    for c, (_, b) in chans.items():
        out[f"edges|{c}"] = b
    for k, b in res_bins.items():
        out[f"edges|{k}"] = b
    # wind speed per time step by role group (all written episodes)
    sh = Shards(ds); wb = np.linspace(0, 10, 51); WH = {}
    grp = lambda r: "train" if r == "train" else ("ID (val/test)" if r in ("val_id", "test_id") else ("OOD magnitude" if "magnitude" in r else ("OOD other" if r.startswith(("val_ood", "test_ood")) else None)))   # noqa: E731
    for ep, m in man.items():
        g = grp(m["split_role"])
        if g is None or not sh.has(ep):
            continue
        w = np.linalg.norm(sh.get(ep, "environment/wind_W")[::10], axis=1); WH.setdefault(g, np.zeros(len(wb) - 1)); WH[g] += np.histogram(w, wb)[0]
    for g, v in WH.items():
        out[f"windstep|{g}"] = v
    out["windstep|edges"] = wb
    np.savez(cache / "quality.npz", **out); print("wrote quality.npz")

STEPS = {"quality": build_quality, "traj": build_traj, "extra": build_extra, "errors": build_errors, "failure": build_failure}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=Path("data/datasets/physiworld_pilot_v7_v0.7.0"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/figure_cache"))
    ap.add_argument("--only", nargs="+", choices=sorted(STEPS), default=["quality", "traj", "extra", "errors", "failure"])
    a = ap.parse_args(argv)
    a.cache.mkdir(parents=True, exist_ok=True)
    for k in a.only:
        STEPS[k](a.dataset, a.cache)


if __name__ == "__main__":
    main()
