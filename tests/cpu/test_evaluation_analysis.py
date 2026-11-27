"""Evaluation analyses: regimes, paired gaps, re-observation (Anchor), residual diagnostics, shuffled-u, future IMU."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.evaluation import analysis as AN
from physiworld_uav.evaluation import regimes as R
from physiworld_uav.evaluation.evaluator import collect_errors, evaluate
from physiworld_uav.evaluation.gap import paired_gap, residual_diagnostics, residual_off, rmse_curve, summarize_residual
from physiworld_uav.evaluation.leakage import ActionSwap, partner_map
from physiworld_uav.models.world_model import Anchor, ModelConfig, build_model, fit_stats
from test_models import BCFG, DT, PH, synthetic_episode

WIND = (0.8, -0.4, 0.2)


def _meta(peak, mode=None, group=None, variable="wind_magnitude", idx=0):
    cf = SimpleNamespace(mode=mode, group_id=group, variable=variable, sibling_index=idx, is_baseline=idx == 0) if mode else None
    return SimpleNamespace(plan={"wind": {"stats": {"peak_speed": peak}}}, counterfactual=cf)


def _ep(eid, seed, wind=WIND, peak=None, **kw):
    e = synthetic_episode(eid, seed, n=90, wind_acc=wind)
    m = len(e.actions)
    e.aux = {"ground_truth/wrench/external_force_W": np.tile(np.array(wind, np.float32) * PH.mass, (m, 1)),
             "ground_truth/wrench/external_torque_B": np.zeros((m, 3), np.float32)}
    e.meta = _meta(float(np.linalg.norm(wind)) if peak is None else peak, **kw)
    return e


def _b3(eps, **kw):
    return build_model("b3", stats=fit_stats(eps), cfg=ModelConfig(latent=16, embed=8, hidden=24, **kw), dt=DT)


def _batch(ds, n=8, **extra):
    items = [ds[i] for i in range(n)]
    keys = ("state_history", "obs_history", "action_history", "future_actions", "future_states") + extra.get("more", ())
    return {k: torch.stack([it[k] for it in items]) for k in keys}


# ------------------------------------------------------------------ regimes
def test_regime_labels_and_selection():
    a, b, c = _ep("a", 0, peak=0.0), _ep("b", 1, peak=2.0), _ep("c", 2, peak=1.0, mode="fixed_action", group="g", idx=1)
    assert [R.wind_regime(e) for e in (a, b, c)] == ["no_wind", "wind", "wind"]
    assert [R.cf_mode(e) for e in (a, b, c)] == ["none", "none", "fixed_action"]
    assert R.select([a, b, c], wind="wind") == [b, c] and R.select([a, b, c], mode="fixed_action") == [c]
    assert R.select([a, b, c], wind="wind", mode="none") == [b]


# ------------------------------------------------------------------ paired gap
def test_paired_gap_exact_and_requires_identical_windows():
    rng = np.random.default_rng(0)
    eids = np.array(["e0"] * 6 + ["e1"] * 4 + ["e2"] * 5)
    b = rng.uniform(0.1, 0.2, (15, 5))
    a = b + 0.05
    r = paired_gap(a, b, eids, [1, 5], n_boot=200)
    assert np.allclose(r["gap"], [rmse_curve(a, [1, 5])[k] - rmse_curve(b, [1, 5])[k] for k in (1, 5)])
    assert all(g > 0 for g in r["gap"]) and r["n_episodes"] == 3 and r["n_windows"] == 15
    lo, hi = r["gap_ci95"]
    assert all(l <= g + 1e-9 and g - 1e-9 <= h for l, g, h in zip(lo, r["gap"], hi))
    with pytest.raises(ValueError):
        paired_gap(a[:10], b, eids, [1])
    same = paired_gap(b, b, eids, [1], n_boot=10)
    assert same["gap"] == [0.0] and same["ratio"] == [1.0]


# ------------------------------------------------------------------ Anchor semantics
def test_anchor_schedule():
    a = Anchor(3)
    assert [k for k in range(10) if a.fires(k, 10)] == [2, 5, 8]               # after steps 3, 6, 9 (k is 0-based), never after the last
    assert not Anchor(1).fires(9, 10) and Anchor(1).fires(8, 10)
    with pytest.raises(ValueError):
        Anchor(0)


def test_state_anchor_stops_error_growth_for_physics_only():
    eps = [_ep("w0", 0), _ep("w1", 1)]
    ds = ModelReadyDataset(eps, BuilderConfig(decimation=4, history=10, horizon=12))
    b1 = build_model("b1")
    open_ = rmse_curve(collect_errors(b1, ds, horizon=12)["pe"], [1, 12])
    anch = rmse_curve(collect_errors(b1, ds, horizon=12, anchor=Anchor(1, state=True))["pe"], [1, 12])
    assert anch[12] < 0.5 * open_[12] and anch[12] < 2.5 * anch[1]               # resetting to the true state every step: no growth
    assert anch[1] == pytest.approx(open_[1])                                    # the first step is not affected


def test_latent_reobservation_changes_only_later_steps_and_needs_future_obs():
    eps = [_ep("w0", 0), _ep("w1", 1)]
    m = _b3(eps)
    for p in m.resid.parameters():
        torch.nn.init.normal_(p, std=0.2)
    bc = BuilderConfig(decimation=4, history=10, horizon=8)
    ds = ModelReadyDataset(eps, bc, future_obs=True)
    b = _batch(ds, more=("future_obs",))
    m.eval()
    with torch.no_grad():
        open_ = m.predict(b, 8)
        every2 = m.predict(b, 8, "given", Anchor(2))
        st = m.predict(b, 8, "given", Anchor(2, state=True))
    assert torch.equal(open_[:, :2], every2[:, :2])                              # nothing re-observed before step 2 is consumed
    assert not torch.allclose(open_[:, 3:], every2[:, 3:])
    assert not torch.allclose(every2[:, 3:], st[:, 3:]) and torch.isfinite(st).all()
    nofo = {k: v for k, v in b.items() if k != "future_obs"}
    for anchor in (Anchor(2), Anchor(2, state=True)):                            # a latent model always needs the real IMU
        with pytest.raises(ValueError, match="future_obs"):
            m.predict(nofo, 8, "given", anchor)
    with torch.no_grad():                                                         # models without a latent: state anchor, no IMU
        build_model("b1").predict(nofo, 8, "given", Anchor(2, state=True))


def test_future_obs_channel_and_noise_does_not_change_history_noise():
    ep = _ep("w0", 0)
    plain = ModelReadyDataset([ep], BCFG)
    fo = ModelReadyDataset([ep], BCFG, future_obs=True)
    assert "future_obs" not in plain[0]
    i = 5
    ep_, t = fo.window(i)
    assert fo[i]["future_obs"].shape == (BCFG.horizon, 6)
    assert np.array_equal(fo[i]["future_obs"].numpy(), ep.obs[t + 1: t + 1 + BCFG.horizon])
    noisy_plain = ModelReadyDataset([ep], BCFG, transform=ImuNoise("nominal", 3))[i]["obs_history"]
    noisy_fo = ModelReadyDataset([ep], BCFG, transform=ImuNoise("nominal", 3), future_obs=True)[i]
    assert torch.equal(noisy_plain, noisy_fo["obs_history"])                      # adding the channel never changes the history noise
    assert not torch.equal(noisy_fo["future_obs"], fo[i]["future_obs"])
    again = ModelReadyDataset([ep], BCFG, transform=ImuNoise("nominal", 3), future_obs=True)[i]["future_obs"]
    assert torch.equal(again, noisy_fo["future_obs"])                             # deterministic


# ------------------------------------------------------------------ residual
def test_residual_off_zeroes_and_restores_and_diagnostics_shapes():
    eps = [_ep("w0", 0), _ep("w1", 1)]
    m = _b3(eps)
    for p in m.resid.parameters():
        torch.nn.init.normal_(p, std=0.3)
    ds = ModelReadyDataset(eps, BCFG)
    b = _batch(ds)
    m.eval()
    with torch.no_grad():
        on = m.predict(b, 5)
        with residual_off(m):
            assert float(m.res_mask.abs().sum()) == 0.0
            off = m.predict(b, 5)
        again = m.predict(b, 5)
    assert not torch.allclose(on, off) and torch.equal(on, again)
    d = residual_diagnostics(m, ds, horizon=5)
    assert d["r_acc"].shape == (len(ds), 5) and (d["r_acc"] >= 0).all() and len(d["eids"]) == len(ds)
    assert (np.abs(d["f_spec"].mean() - PH.g) < 3.0)                              # specific force ≈ g in these near-hover rollouts
    s = summarize_residual(d)
    assert len(s["ratio_r_acc_to_f_spec"]) == 5 and s["n_windows"] == len(ds)
    with residual_off(build_model("b1")) if False else pytest.raises(ValueError):
        residual_diagnostics(build_model("b1"), ds, horizon=5)


# ------------------------------------------------------------------ shuffled u
def test_partner_map_and_action_swap():
    e0 = _ep("g_e0", 0, peak=0.0, wind=(0, 0, 0), mode="closed_loop", group="g", idx=0)
    e1 = _ep("g_e1", 1, peak=2.0, mode="closed_loop", group="g", idx=1)
    e2 = _ep("g_e2", 2, peak=1.0, mode="closed_loop", group="g", idx=2)
    mass = _ep("m_e0", 3, peak=2.0, mode="closed_loop", group="m", variable="mass", idx=0)
    modeB = _ep("b_e1", 4, peak=2.0, mode="fixed_action", group="b", idx=1)
    pm = partner_map([e0, e1, e2, mass, modeB])
    assert pm["g_e0"] is e1 and pm["g_e1"] is e0 and pm["g_e2"] is e0              # the sibling with the most different wind
    assert "m_e0" not in pm and "b_e1" not in pm                                    # mass groups and Mode B are skipped
    ds = ModelReadyDataset([e0, e1, mass], BCFG)
    orig, swap = ActionSwap(ds, pm, swap=False), ActionSwap(ds, pm, swap=True)
    assert len(orig) == len(swap) > 0 and all(ds.window(i)[0].episode_id != "m_e0" for i, _ in orig.index)
    for j in (0, len(swap) - 1):
        i, q = swap.index[j]
        ep, t = ds.window(i)
        a, b = orig[j], swap[j]
        assert torch.equal(a["future_states"], b["future_states"]) and torch.equal(a["state_history"], b["state_history"])
        assert torch.equal(b["future_actions"], torch.from_numpy(q.actions[t: t + BCFG.horizon]))
        assert torch.equal(b["action_history"], torch.from_numpy(q.actions[t - BCFG.history: t]))
        assert not torch.equal(a["future_actions"], b["future_actions"])


# ------------------------------------------------------------------ end-to-end on synthetic data
def test_regime_analysis_b1_vs_oracle_gap_is_positive_with_wind_and_zero_without():
    windy = [_ep("w0", 0), _ep("w1", 1)]
    calm = [_ep("c0", 2, wind=(0, 0, 0), peak=0.0), _ep("c1", 3, wind=(0, 0, 0), peak=0.0)]
    bc = BuilderConfig(decimation=4, history=10, horizon=10)
    models = {"b1": build_model("b1"), "b6a_future": build_model("b6a_future")}
    res = AN.regime_analysis(models, windy + calm, bc, "none", 10, torch.device("cpu"), ["b1:b6a_future"], 50)
    gw = res["regimes"]["wind"]["gaps"]["b1:b6a_future"]; gc = res["regimes"]["no_wind"]["gaps"]["b1:b6a_future"]
    assert gw["gap"][-1] > 1e-2 and gw["gap_ci95"][0][-1] >= 0 and abs(gc["gap"][-1]) < 1e-3
    text = AN.markdown("regime", res)
    assert "no_wind" in text and "gap b1:b6a_future" in text


def test_anchor_residual_and_shuffle_analyses_run_on_a_latent_model():
    e0 = _ep("g_e0", 0, peak=0.0, wind=(0, 0, 0), mode="closed_loop", group="g", idx=0)
    e1 = _ep("g_e1", 1, peak=2.0, mode="closed_loop", group="g", idx=1)
    eps = [e0, e1]
    m = _b3(eps); b1 = build_model("b1")
    bc = BuilderConfig(decimation=4, history=10, horizon=10)
    dev = torch.device("cpu")
    a = AN.anchor_analysis({"b3": m, "b1": b1}, eps, bc, "nominal", 10, dev, [1, 5], "h")
    assert set(a["models"]) == {"b3"}                                              # B1 has no latent: skipped in h mode
    r = a["models"]["b3"]
    assert set(r["anchored"]) == {"1", "5"} and set(r["anchored"]["5"]["rmse_by_steps_since_observation"]) == {"1", "2", "3", "4", "5"}
    assert "phi_vs_real_imu_boundary" in r and "boundary" not in AN.markdown("anchor", a).lower().split("cost")[0]
    s = AN.anchor_analysis({"b1": b1}, eps, bc, "none", 10, dev, [2], "state")
    assert "b1" in s["models"]
    rr = AN.residual_analysis({"b3": m, "b1": b1}, eps, bc, "nominal", 10, dev)
    assert set(rr["models"]) == {"b3"} and set(rr["models"]["b3"]) == {"wind", "no_wind"}
    sh = AN.shuffle_analysis({"b1": b1, "b3": m}, eps, bc, "none", 10, dev)
    assert sh["n_episodes_with_partner"] == 2 and sh["models"]["b1"]["n_windows"] > 0
    assert all(v >= 0 for v in sh["models"]["b1"]["jump_ratio"].values())
