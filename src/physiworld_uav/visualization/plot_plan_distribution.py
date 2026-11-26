"""Distribution plots of a planned or generated dataset, from its manifest only (no simulation, no HDF5).

    python -m physiworld_uav.visualization plan --dataset data/datasets/physiworld_pilot_v7_v0.7.0 --out outputs/reports/data_distribution_v7

Figures: 1 sizes per role; 2 wind speed (mean and peak per episode, by role group, with the ID / OOD limits) and direction; 3 vehicle
(domain) factors, ID against OOD roles, one point per distinct vehicle; 4 scenarios, data type, wind profile and counterfactual modes;
5 initial conditions. Tables: tables/*.csv."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = [("train", ["train"], "#0072B2"), ("val ID", ["val_id"], "#56B4E9"), ("test ID", ["test_id"], "#009E73"),
          ("OOD magnitude", ["val_ood_magnitude", "test_ood_magnitude"], "#D55E00"),
          ("other OOD (direction, temporal, gust, physics, trajectory)", ["val_ood_direction", "test_ood_direction", "val_ood_temporal", "test_ood_temporal",
           "val_ood_gust", "test_ood_gust", "val_ood_physics", "test_ood_physics", "test_ood_trajectory"], "#CC79A7"),
          ("physics validation", ["physics_validation"], "#888888")]
ROLE_COL = {}
for lab, roles, col in GROUPS:
    for r in roles:
        ROLE_COL[r] = col


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id-peak-limit", type=float, default=6.2)
    ap.add_argument("--ood-peak-limit", type=float, default=9.0)
    a = ap.parse_args(argv)
    (a.out / "figures").mkdir(parents=True, exist_ok=True); (a.out / "tables").mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (a.dataset / "manifest.jsonl").read_text().splitlines() if l.strip()]
    sim = rows[0]["sim"]; dec = int(sim["control_decimation"]); mdt = float(sim["model_dt"])
    title = f"{a.dataset.name}: {len(rows)} episodes, model step {mdt * 1000:.0f} ms ({1 / mdt:.0f} Hz)"
    grp_of = {r: lab for lab, rs, _ in GROUPS for r in rs}
    col_of = {lab: col for lab, _, col in GROUPS}
    steps = lambda r: r["n_steps"] / dec  # noqa: E731

    # ---- tables
    per_role = defaultdict(lambda: {"episodes": 0, "steps": 0.0, "groups": set(), "domains": set()})
    for r in rows:
        d = per_role[r["split_role"]]; d["episodes"] += 1; d["steps"] += steps(r); d["groups"].add(r["group_id"]); d["domains"].add(r["domain"]["domain_id"])
    with open(a.out / "tables" / "per_role.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["role", "episodes", "model_steps", "groups", "domains"])
        for k, v in sorted(per_role.items(), key=lambda kv: -kv[1]["steps"]):
            w.writerow([k, v["episodes"], int(v["steps"]), len(v["groups"]), len(v["domains"])])

    # ---- fig 1: sizes per role
    roles = [k for k, _ in sorted(per_role.items(), key=lambda kv: -kv[1]["steps"])]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for i, (key, lab) in enumerate((("episodes", "episodes"), ("steps", "model steps"))):
        ax[i].barh(roles[::-1], [per_role[r][key] for r in roles[::-1]], color=[ROLE_COL[r] for r in roles[::-1]])
        ax[i].set_title(f"{lab} per role"); ax[i].grid(axis="x", alpha=0.3)
        for y, r in enumerate(roles[::-1]):
            ax[i].text(per_role[r][key], y, f" {int(per_role[r][key]):,}", va="center", fontsize=8)
    ax[2].barh(roles[::-1], [len(per_role[r]["domains"]) for r in roles[::-1]], color=[ROLE_COL[r] for r in roles[::-1]])
    ax[2].set_title("distinct vehicles (domains) per role"); ax[2].grid(axis="x", alpha=0.3)
    for y, r in enumerate(roles[::-1]):
        ax[2].text(len(per_role[r]["domains"]), y, f" {len(per_role[r]['domains'])}", va="center", fontsize=8)
    fig.suptitle(title + f"; {sum(v['steps'] for v in per_role.values()) / 1e6:.2f}M model steps"); fig.tight_layout(); fig.savefig(a.out / "figures" / "1_sizes_per_role.png", dpi=140); plt.close(fig)

    # ---- fig 2: wind
    def ws(r, key): return r["wind"]["stats"][key]
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.6))
    mb = np.linspace(0, 9, 46)
    for lab, rs, col in GROUPS[:5]:
        sel = [r for r in rows if r["split_role"] in rs]
        if not sel:
            continue
        ax[0, 0].hist([ws(r, "mean_horizontal_speed") for r in sel], bins=mb, histtype="step", lw=2, color=col, label=f"{lab} (n={len(sel)})")
        ax[0, 1].hist([ws(r, "peak_speed") for r in sel], bins=mb, histtype="step", lw=2, color=col)
    for x, t in ((a.id_peak_limit, "ID peak limit"), (a.ood_peak_limit, "OOD peak limit")):
        ax[0, 1].axvline(x, color="#444", ls="--", lw=1); ax[0, 1].text(x, ax[0, 1].get_ylim()[1] * 0.95, " " + t, fontsize=8, rotation=90, va="top")
    ax[0, 0].axvline(a.id_peak_limit, color="#444", ls="--", lw=1)
    ax[0, 0].set_title("episode mean wind speed"); ax[0, 0].set_xlabel("m/s"); ax[0, 0].set_ylabel("episodes"); ax[0, 0].legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.92)); ax[0, 0].grid(alpha=0.3)
    ax[0, 1].set_title("episode peak wind speed"); ax[0, 1].set_xlabel("m/s"); ax[0, 1].grid(alpha=0.3)
    # time-step weighted speed is not in the manifest; use mean speed weighted by steps
    for lab, rs, col in GROUPS[:5]:
        sel = [r for r in rows if r["split_role"] in rs]
        if sel:
            ax[0, 2].hist([ws(r, "mean_horizontal_speed") for r in sel], bins=mb, weights=[steps(r) for r in sel], histtype="step", lw=2, color=col, density=True)
    ax[0, 2].set_title("mean wind speed weighted by model steps (density)"); ax[0, 2].set_xlabel("m/s"); ax[0, 2].grid(alpha=0.3)
    az = np.array([np.degrees(np.arctan2(r["wind"]["spec"]["steady"]["direction"][1], r["wind"]["spec"]["steady"]["direction"][0])) for r in rows if r["wind"]["spec"].get("steady", {}).get("enabled")])
    el = np.array([np.degrees(np.arcsin(np.clip(r["wind"]["spec"]["steady"]["direction"][2], -1, 1))) for r in rows if r["wind"]["spec"].get("steady", {}).get("enabled")])
    ax[1, 0].hist(az, bins=np.linspace(-180, 180, 37), color="#0072B2"); ax[1, 0].set_title(f"wind azimuth ({len(az)} episodes with wind)"); ax[1, 0].set_xlabel("deg"); ax[1, 0].grid(alpha=0.3)
    ax[1, 1].hist(el, bins=20, color="#009E73"); ax[1, 1].set_title("wind elevation"); ax[1, 1].set_xlabel("deg"); ax[1, 1].grid(alpha=0.3)
    prof = Counter(r["wind"]["profile"] for r in rows); ax[1, 2].barh(list(prof)[::-1], [prof[k] for k in list(prof)[::-1]], color="#E69F00")
    ax[1, 2].set_title("wind profile type (episodes)"); ax[1, 2].grid(axis="x", alpha=0.3)
    fig.suptitle(title + " : wind"); fig.tight_layout(); fig.savefig(a.out / "figures" / "2_wind.png", dpi=140); plt.close(fig)

    # ---- fig 3: vehicle factors (one point per distinct vehicle)
    dom = {}
    for r in rows:
        dom.setdefault(r["domain"]["domain_id"], (grp_of[r["split_role"]], r["domain"]["factors"]))
    names = list(next(iter(dom.values()))[1])
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for ax_, nme in zip(axs.ravel(), names):
        for lab, _, col in GROUPS[:5]:
            v = [f[nme] for g, f in dom.values() if g == lab]
            if v:
                ax_.hist(v, bins=24, histtype="step", lw=1.8, color=col, density=True, label=lab if nme == names[0] else None)
        ax_.set_title(nme); ax_.grid(alpha=0.3)
    axs.ravel()[0].legend(fontsize=6)
    fig.suptitle(title + f" : vehicle factors relative to nominal ({len(dom)} distinct vehicles; density)"); fig.tight_layout(); fig.savefig(a.out / "figures" / "3_vehicle_factors.png", dpi=140); plt.close(fig)

    # ---- fig 4: scenarios, data type, counterfactual
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    sc = Counter(r["scenario_template"] for r in rows); ks = [k for k, _ in sc.most_common()][::-1]
    ax[0].barh(ks, [sc[k] for k in ks], color="#0072B2"); ax[0].set_title("episodes per scenario template"); ax[0].grid(axis="x", alpha=0.3)
    cats = ["train", "val ID", "test ID", "OOD magnitude", "other OOD (direction, temporal, gust, physics, trajectory)"]
    dtypes = sorted({r["data_type"] for r in rows}); bottom = np.zeros(len(cats))
    for dt_, col in zip(dtypes, ["#0072B2", "#E69F00", "#009E73"]):
        v = np.array([sum(1 for r in rows if grp_of[r["split_role"]] == c and r["data_type"] == dt_) for c in cats])
        ax[1].bar(range(len(cats)), v, bottom=bottom, color=col, label=dt_); bottom += v
    ax[1].set_xticks(range(len(cats))); ax[1].set_xticklabels(["train", "val ID", "test ID", "OOD\nmagnitude", "other\nOOD"], fontsize=8); ax[1].set_title("data type (closed / open loop)"); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=0.3)
    def cf(r):
        c = r.get("counterfactual")
        return "none" if not c else f"{c['mode']} (variable {c.get('variable', '?')})"
    cfc = Counter(cf(r) for r in rows); ax[2].barh(list(cfc)[::-1], [cfc[k] for k in list(cfc)[::-1]], color="#CC79A7"); ax[2].set_title("counterfactual episodes\n(Mode A closed loop / Mode B fixed action)", fontsize=10); ax[2].grid(axis="x", alpha=0.3)
    modeB = Counter(r["split_role"] for r in rows if r.get("counterfactual") and r["counterfactual"].get("mode") == "fixed_action")
    ax[2].text(0.02, 0.02, "Mode B by role: " + ", ".join(f"{k} {v}" for k, v in modeB.items()), transform=ax[2].transAxes, fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(a.out / "figures" / "4_scenarios_modes.png", dpi=140); plt.close(fig)

    # ---- fig 5: initial conditions and duration
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    ic = [r["initial_conditions"] for r in rows]
    ax[0].hist([i["position"][2] for i in ic], bins=20, color="#0072B2"); ax[0].set_title("initial height [m]")
    ax[1].hist([np.linalg.norm(i["velocity_W"]) for i in ic], bins=20, color="#009E73"); ax[1].set_title("initial speed [m/s]")
    ax[2].hist([np.linalg.norm(i["angular_velocity_B"]) for i in ic], bins=20, color="#E69F00"); ax[2].set_title("initial angular rate [rad/s]")
    ax[3].hist([r["duration_s"] for r in rows], bins=20, color="#CC79A7"); ax[3].set_title("episode duration [s]")
    for x in ax:
        x.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(a.out / "figures" / "5_initial_conditions.png", dpi=140); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
