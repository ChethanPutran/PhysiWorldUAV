"""Sanity baseline: constant-velocity extrapolation (zero acceleration), evaluated exactly like B0/B1 on the v0.6 roles."""
import json, torch, torch.nn as nn
from pathlib import Path
from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, load_model_episodes
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.evaluation.evaluator import evaluate
from physiworld_uav.evaluation.run import ROLE_GROUPS

class ConstVel(nn.Module):
    name = "cv"
    def predict(self, batch, horizon, initial_state="given", anchor=None):
        s = batch["state_history"][:, -1]
        out = s[:, None, :].repeat(1, horizon, 1).clone()
        t = (torch.arange(1, horizon + 1, dtype=s.dtype) * 0.02)[None, :, None]
        out[..., 0:3] = s[:, None, 0:3] + s[:, None, 3:6] * t
        return out
ds = Path("data/datasets/physiworld_pilot_v0.6.0")
bcfg = BuilderConfig(decimation=4, history=10, horizon=20, obs_mode="instant")
res = {}
for role in ["val_id", "test_id", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory"]:
    eps = load_model_episodes(ds, bcfg, roles=ROLE_GROUPS.get(role, [role]))
    r = evaluate(ConstVel(), ModelReadyDataset(eps, bcfg, transform=ImuNoise("nominal", 777)), horizon=20, device=torch.device("cpu"), initial_state="given", ks=[1, 5, 10, 20])
    res[role] = r["pos_rmse_m"]["20"]; print(role, "%.4f" % r["pos_rmse_m"]["20"], "k=1 %.5f" % r["pos_rmse_m"]["1"], flush=True)
json.dump(res, open("outputs/eval/constant_velocity_v06.json", "w"), indent=1)
