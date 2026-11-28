"""Figures of the midterm report and slides, drawn from the project's own data (reusable for later reports).

    python -m physiworld_uav.visualization figures [--only wind vehicle ...] [--out outputs/figures] [--formats pdf png jpg]
                                                   [--dataset data/datasets/<v7>] [--copy-to report/figures_new] [--list]

Every figure is one function ``fig_<name>`` registered in ``FIGURES``; each writes ``<name>.pdf`` (vector), ``.png`` and optionally
``.jpg`` into ``--out``. Inputs: the dataset manifest and index, the evaluation JSONs in ``outputs/eval``, the training logs in
``outputs/experiments``, and a cache of data that needs the HDF5 shards or the checkpoints (``python -m physiworld_uav.visualization
figure-data``, container) in ``outputs/figure_cache``.
Nothing in the figures is typed in by hand except labels; numbers come from the files above (the schematic diagrams excepted).
"""
from __future__ import annotations

import argparse
import glob
import json
import collections
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "figures"
CACHE = ROOT / "outputs" / "figure_cache"
DS = ROOT / "data/datasets/physiworld_pilot_v7_v0.7.0"
FORMATS = ["pdf", "png"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#e3e7ea", "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.titleweight": "bold", "legend.frameon": False})
W1, W2 = 3.45, 7.16
# Optional (``--page-size``): ``PAGE_WIDTH`` maps a figure name to the fraction of the 7.16 in text width it should occupy; sizes written for
# the full width are then scaled. Off by default: the figures are drawn full width and shrunk by LaTeX/pptx.
PAGE_WIDTH = {"wind": .62, "error_accum": .75, "coverage": .6, "wind_series": .5, "dr_grid": .5, "counterfactual": .52, "imu": .52, "dataset_versions": .52, "dataset_dist": .62,
              "train_loss": .5, "versions_perf": .56, "id_ood": .66, "failure": .66, "timeline": .62, "pipeline": .7, "horizon": .7, "traj": .5, "arch": .82, "modes": .7, "frames_forces": .6}
_SCALE = [1.0]
_plt_subplots, _plt_figure = plt.subplots, plt.figure
def _scaled(fs):
    f = _SCALE[0]
    return fs if f == 1.0 or fs is None else (fs[0] * f, fs[1] * f * 1.12)
def _subplots(*a, **k):
    if "figsize" in k: k["figsize"] = _scaled(k["figsize"])
    return _plt_subplots(*a, **k)
def _figure(*a, **k):
    if "figsize" in k: k["figsize"] = _scaled(k["figsize"])
    return _plt_figure(*a, **k)
plt.subplots, plt.figure = _subplots, _figure

C = {"B0": "#9a9a9a", "CV": "#555555", "B1": "#CC79A7", "B2": "#0072B2", "B3": "#009E73", "B3s": "#66c2a5", "B4": "#56B4E9", "B4w": "#E69F00",
     "B5a": "#8c6d46", "B5b": "#b3a28a", "B6": "#7B4EA3"}
GC = {"train": "#7f8c99", "ID": "#0072B2", "OOD": "#E69F00"}

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext != "pdf" else None, **({"pil_kwargs": {"quality": 92}} if ext == "jpg" else {}))
    plt.close(fig); print("wrote", name, "->", OUT)
def tag(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right")

man: list = []


def load_manifest(dataset: Path) -> None:
    global man, DS
    DS = Path(dataset); man = [json.loads(l) for l in open(DS / "manifest.jsonl")]
def grp(r):
    if r == "train": return "train"
    if r in ("val_id", "test_id"): return "ID"
    if r.startswith(("val_ood", "test_ood")): return "OOD"
    return None

# ---------------------------------------------------------------- wind
def fig_wind():
    g = collections.defaultdict(lambda: {"mean": [], "peak": [], "az": []})
    for m in man:
        k = grp(m["split_role"]); 
        if k is None: continue
        if k == "OOD" and "magnitude" in m["split_role"]: k = "OOD magnitude"
        elif k == "OOD": k = "OOD other"
        st = m["wind"]["stats"]; g[k]["mean"].append(st["mean_horizontal_speed"]); g[k]["peak"].append(st["peak_speed"])
        d = (m["wind"]["spec"].get("steady") or {}).get("direction")
        if d and (m["wind"]["spec"]["steady"].get("enabled")): g[k]["az"].append(np.degrees(np.arctan2(d[1], d[0])) % 360)
    cols = {"train": GC["train"], "ID": GC["ID"], "OOD other": "#d9a679", "OOD magnitude": "#D55E00"}
    fig = plt.figure(figsize=(W2, 4.9))
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2); ax3 = fig.add_subplot(2, 2, 3, projection="polar"); ax4 = fig.add_subplot(2, 2, 4)
    bins = np.arange(0, 10.01, 0.5)
    for k in ("train", "ID", "OOD other", "OOD magnitude"):
        ax1.hist(g[k]["mean"], bins, histtype="stepfilled", alpha=.55, color=cols[k], label=k, density=True, lw=0)
        ax2.hist(g[k]["peak"], bins, histtype="stepfilled", alpha=.55, color=cols[k], density=True, lw=0)
    for ax, lim in ((ax1, 6.2), (ax2, 6.2)):
        ax.axvline(lim, color="k", ls="--", lw=.8); ax.set_xlabel("m/s"); ax.set_ylabel("density")
    ax1.set_title("Mean horizontal speed"); ax2.set_title("Peak speed")
    h_, l_ = ax1.get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.01), fontsize=7.5)
    az = np.radians(np.concatenate([g[k]["az"] for k in g])); b = np.linspace(0, 2 * np.pi, 25)
    ax3.hist(az, b, color="#0072B2", alpha=.8); ax3.set_title("Steady-wind azimuth", pad=12); ax3.set_yticklabels([]); ax3.tick_params(labelsize=6); ax3.grid(True)
    prof = collections.Counter(m["wind"]["profile"] for m in man if m["wind"]["profile"] != "none")
    names = {"steady": "steady", "ou": "OU", "cosine_gust": "1-cos gust", "step": "step", "sinusoid": "sinusoid", "random_gusts": "random gusts"}
    ks = sorted(prof, key=prof.get); ax4.barh([names[k] for k in ks], [prof[k] for k in ks], color="#0072B2", height=.65)
    for i, k in enumerate(ks): ax4.text(prof[k] + 15, i, str(prof[k]), va="center", fontsize=6.5)
    ax4.set_title("Wind profile type"); ax4.set_xlabel("episodes"); ax4.grid(axis="y", visible=False); ax4.set_xlim(0, 1300)
    for ax, s in zip((ax1, ax2, ax3, ax4), "abcd"): tag(ax, s)
    fig.tight_layout(w_pad=2.5, h_pad=2.0, rect=(0, 0, 1, .96)); save(fig, "wind")

# ---------------------------------------------------------------- vehicle
def fig_vehicle():
    names = ["mass", "force_scale", "inertia_shape_x", "inertia_shape_y", "inertia_shape_z", "km_kf_ratio", "motor_time_constant", "rotor_drag_scale", "body_drag_scale"]
    hw = {"mass": .10, "force_scale": .10, "inertia_shape_x": .05, "inertia_shape_y": .05, "inertia_shape_z": .05, "km_kf_ratio": .05, "motor_time_constant": .20, "rotor_drag_scale": .10, "body_drag_scale": .10}
    X = np.array([[(m["domain"]["factors"][n] - 1) * 100 for n in names] for m in man if m["split_role"] != "physics_validation"])
    roles = [m["split_role"] for m in man if m["split_role"] != "physics_validation"]
    dom = [m["domain"]["domain_id"] for m in man if m["split_role"] != "physics_validation"]
    ph = np.array([r.endswith("ood_physics") for r in roles]); idm = np.array([grp(r) != "OOD" for r in roles])
    fig, ax2d = plt.subplots(2, 2, figsize=(W2, 5.4)); ax = ax2d.ravel()[[0, 1, 2]]; axd = ax2d.ravel()[3]
    a = ax[0]
    a.scatter(X[~ph, 0], X[~ph, 1], s=3, c="#7f8c99", alpha=.35, lw=0, label="train / ID / other OOD")
    a.scatter(X[ph, 0], X[ph, 1], s=6, c="#E69F00", alpha=.9, lw=0, label="OOD physics")
    a.set_xlabel("mass deviation [%]"); a.set_ylabel("thrust-scale deviation [%]"); a.legend(loc="upper center", bbox_to_anchor=(.5, -.28), markerscale=2.2, ncol=1); a.set_title("Mass vs thrust scale"); a.set_aspect("equal")
    a = ax[1]; j = names.index("motor_time_constant")
    b = np.linspace(-42, 42, 29)
    a.hist(X[~ph, j], b, color="#7f8c99", alpha=.8, label="other"); a.hist(X[ph, j], b, color="#E69F00", alpha=.9, label="OOD physics")
    a.set_xlabel("motor time-constant deviation [%]"); a.set_ylabel("episodes"); a.set_title("Actuator lag"); a.legend()
    a = ax[2]
    # one row per distinct vehicle for correlations
    uniq = {}
    for d_, x in zip(dom, X): uniq[d_] = x
    U = np.array(list(uniq.values())); R = np.corrcoef(U.T)
    lab = ["m", "$k_F$", "$J_x$", "$J_y$", "$J_z$", "$k_M/k_F$", "$\\tau_m$", "$d_{rot}$", "$C_{D}$"]
    im = a.imshow(R, cmap="RdBu_r", vmin=-.3, vmax=.3); a.grid(False)
    a.set_xticks(range(9)); a.set_xticklabels(lab, rotation=90); a.set_yticks(range(9)); a.set_yticklabels(lab)
    off = np.abs(R - np.eye(9)).max(); a.set_title("Factor correlation")
    cb = fig.colorbar(im, ax=a, fraction=.046, pad=.03); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(.4)
    lbl = ["mass", "thrust", "$J_x$", "$J_y$", "$J_z$", "$k_M/k_F$", "motor $\\tau$", "rotor drag", "body drag"]
    nph = ~ph
    bp1 = axd.boxplot([X[nph, k] for k in range(9)], positions=np.arange(9) - .17, widths=.3, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=.8))
    bp2 = axd.boxplot([X[ph, k] for k in range(9)], positions=np.arange(9) + .17, widths=.3, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=.8))
    for b_ in bp1["boxes"]: b_.set(fc="#7f8c99", ec="none", alpha=.85)
    for b_ in bp2["boxes"]: b_.set(fc="#E69F00", ec="none", alpha=.95)
    axd.set_xticks(range(9)); axd.set_xticklabels(lbl, rotation=35, ha="right"); axd.set_ylabel("deviation from nominal [%]"); axd.set_title("All nine factors"); axd.axhline(0, color="k", lw=.5)
    axd.legend([bp1["boxes"][0], bp2["boxes"][0]], ["other roles", "OOD physics"], loc="upper left", ncol=2, fontsize=6.5)
    for x_, s in zip(list(ax) + [axd], "abcd"): tag(x_, s)
    fig.tight_layout(w_pad=2.0, h_pad=2.0); save(fig, "vehicle")
    print("vehicles", len(uniq), "max |rho|", off)

# ---------------------------------------------------------------- sizes
ROLE_ORDER = ["train", "val_id", "test_id", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory",
              "val_ood_magnitude", "val_ood_direction", "val_ood_temporal", "val_ood_gust", "val_ood_physics", "physics_validation"]
def fig_sizes():
    cnt = collections.Counter(m["split_role"] for m in man)
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    ys = np.arange(len(ROLE_ORDER))[::-1]
    for y, r in zip(ys, ROLE_ORDER):
        c = GC.get(grp(r) or "", "#bbbbbb")
        ax.barh(y, cnt[r], color=c, height=.7); ax.text(cnt[r] + 25, y, f"{cnt[r]:,}", va="center", fontsize=6.5)
    ax.set_yticks(ys); ax.set_yticklabels([r.replace("_", " ").replace("ood", "OOD").replace("id", "ID") for r in ROLE_ORDER]); ax.set_xlabel("episodes")
    ax.set_xlim(0, 2600); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GC["train"], label="train"), Patch(color=GC["ID"], label="ID"), Patch(color=GC["OOD"], label="OOD"), Patch(color="#bbbbbb", label="physics check")], loc="center right")
    save(fig, "sizes")

# ---------------------------------------------------------------- scenarios
FAM = {"hover": "hover", "vertical": "translation", "translation": "translation", "line": "translation", "accel_decel": "translation",
       "circle": "complex", "helix": "complex", "figure_eight": "complex", "sinusoidal": "complex", "waypoints": "complex", "combined": "complex",
       "yaw_step": "attitude / yaw", "yaw_sine": "attitude / yaw", "roll_doublet": "attitude / yaw", "pitch_doublet": "attitude / yaw", "attitude_combined": "attitude / yaw",
       "open_loop_rotor": "open loop", "open_loop_collective_rate": "open loop"}
FC = {"hover": "#9a9a9a", "translation": "#0072B2", "complex": "#009E73", "attitude / yaw": "#E69F00", "open loop": "#7B4EA3"}
def fig_scen():
    G = ["train", "ID", "OOD"]
    tc = collections.defaultdict(lambda: collections.Counter())
    for m in man:
        k = grp(m["split_role"])
        if k: tc[m["scenario_template"]][k] += 1
    order = sorted(tc, key=lambda t: (list(FC).index(FAM[t]), -sum(tc[t].values())))
    fig = plt.figure(figsize=(W2, 3.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=.55)
    a = fig.add_subplot(gs[0]); ys = np.arange(len(order))[::-1]; left = np.zeros(len(order))
    for g_ in G:
        v = np.array([tc[t][g_] for t in order]); a.barh(ys, v, left=left, color=GC[g_], height=.72, label=g_); left += v
    for y, t in zip(ys, order): a.text(sum(tc[t].values()) + 12, y, f"{sum(tc[t].values())}", va="center", fontsize=6.3)
    a.set_yticks(ys); a.set_yticklabels([t.replace("_", " ") for t in order])
    for tl, t in zip(a.get_yticklabels(), order): tl.set_color(FC[FAM[t]])
    a.set_xlabel("episodes"); a.legend(loc="lower right"); a.grid(axis="y", visible=False); a.set_xlim(0, 720); a.set_title("Scenario templates (18)"); tag(a, "a")
    b = fig.add_subplot(gs[1]); roles = ["train", "val_id", "test_id", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory"]
    M = np.array([[sum(1 for m in man if m["split_role"] == r and FAM.get(m["scenario_template"]) == f) for f in FC] for r in roles], float)
    M = M / M.sum(1, keepdims=True) * 100; left = np.zeros(len(roles)); yy = np.arange(len(roles))[::-1]
    for j, f in enumerate(FC):
        b.barh(yy, M[:, j], left=left, color=FC[f], height=.72, label=f)
        for y, v, l_ in zip(yy, M[:, j], left):
            if v > 9: b.text(l_ + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white")
        left += M[:, j]
    b.set_yticks(yy); b.set_yticklabels([r.replace("_", " ").replace("ood", "OOD").replace("id", "ID") for r in roles])
    b.set_xlabel("share of episodes in role [%]"); b.set_xlim(0, 100); b.grid(False); b.set_title("Scenario family per role"); tag(b, "b")
    b.legend(ncol=3, loc="upper center", bbox_to_anchor=(.45, -.2), columnspacing=1, handlelength=1)
    save(fig, "scenarios")
    # numbers for the text
    tot = collections.Counter(FAM[m["scenario_template"]] for m in man if m["data_type"] == "closed_loop" and m["split_role"] == "train")
    print("train closed-loop families", tot, sum(tot.values()))
    print("OOD traj", collections.Counter(m["scenario_template"] for m in man if m["split_role"] == "test_ood_trajectory"))

# ---------------------------------------------------------------- training curves
def curve(name):
    r = [json.loads(l) for l in open(ROOT / "outputs/experiments" / name / "metrics.jsonl")]
    return np.array([x["epoch"] for x in r]), np.array([x["val"]["pos_rmse_m"]["20"] for x in r]), np.array([x["horizon"] for x in r])
def fig_curves():
    sets = [("Stage 1: nominal physics prior", "st_v07", [("b2", "B2"), ("b3", "B3"), ("b4_rw", "B4w"), ("b5a", "B5a"), ("b5b", "B5b")]),
            ("Known-vehicle physics prior", "kv_v07", [("b2", "B2"), ("b3", "B3"), ("b4_rw", "B4w"), ("b5a", "B5a"), ("b5b", "B5b"), ("b6b_future", "B6")])]
    lab = {"B2": "B2 pure latent", "B3": "B3 proposed", "B4w": "B4 (B3 width)", "B5a": "B5a", "B5b": "B5b", "B6": "B6b oracle (future)"}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
    for ax, (ttl, pre, ms) in zip(axs, sets):
        for k, c in ms:
            e, v, h = curve(f"{pre}_{k}_s0"); ax.plot(e, v, color=C[c], lw=1.2, label=lab[c]); i = v.argmin(); ax.plot(e[i], v[i], "o", ms=3.5, color=C[c], mec="white", mew=.5)
        e, v, h = curve(f"{pre}_b3_s0")
        for hh in (5, 10, 20):
            ep = e[np.argmax(h >= hh)]; ax.axvline(ep, color="#cccccc", lw=.7, ls=":"); ax.text(ep + .3, 6.5, f"K={hh}", fontsize=6, color="#888888", va="bottom", rotation=0)
        ax.set_yscale("log"); ax.set_ylim(.02, 8); ax.set_xlabel("epoch"); ax.set_title(ttl, pad=11)
    axs[0].set_ylabel("val ID position RMSE at 1 s [m]")
    h_, l_ = axs[1].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=6, bbox_to_anchor=(.5, -.07), fontsize=7)
    tag(axs[0], "a"); tag(axs[1], "b")
    fig.tight_layout(rect=(0, .04, 1, 1)); save(fig, "curves")

# ---------------------------------------------------------------- eval
def load_eval():
    E = {}
    for f in glob.glob(str(ROOT / "outputs/eval/kv_v07_*.json")):
        for m, r in json.load(open(f)).items(): E[m] = r
    return E
MODELS = [("b1_kv", "B1 physics only", "B1"), ("kv_v07_b2_s0", "B2 pure latent", "B2"), ("kv_v07_b3_s0", "B3 proposed", "B3"), ("kv_v07_b4_rw_s0", "B4 (B3 width)", "B4w"),
          ("kv_v07_b4_s0", "B4 (226k)", "B4"), ("kv_v07_b5a_s0", "B5a learned field", "B5a"), ("kv_v07_b5b_s0", "B5b learned dynamics", "B5b"), ("b6a_future_kv", "B6a oracle", "B6"), ("kv_v07_b6b_future_s0", "B6b oracle input", "B6")]
def fig_horizon():
    E = load_eval(); cv = json.load(open(ROOT / "outputs/eval/constant_velocity_v07.json"))
    roles = [("test_id", "Test ID"), ("test_ood_magnitude", "OOD wind magnitude"), ("test_ood_direction", "OOD wind direction")]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 3.0), sharey=True); t = np.array([1, 5, 10, 20]) * .05
    show = ["b1_kv", "kv_v07_b2_s0", "kv_v07_b3_s0", "kv_v07_b4_rw_s0", "kv_v07_b4_s0", "kv_v07_b5a_s0", "kv_v07_b5b_s0", "kv_v07_b6b_future_s0"]
    for ax, (r, ttl) in zip(axs, roles):
        for key, lab, c in MODELS:
            if key not in show: continue
            y = [E[key][r]["pos_rmse_m"][str(k)] for k in (1, 5, 10, 20)]
            ax.plot(t, y, marker={"B4": "s", "B4w": "s", "B5a": "^", "B5b": "v"}.get(c, "o"), ms=3, lw=1.5 if c == "B3" else 1.0, color=C[c], label=lab, zorder=5 if c == "B3" else 3, ls="--" if c == "B6" else ("-." if c in ("B5a", "B5b") else "-"))
        ax.plot([1.0], [cv[r]], "D", ms=5, color=C["CV"], label="constant velocity (1 s)", zorder=6, mfc="white", mew=1.2)
        ax.set_yscale("log"); ax.set_xlabel("rollout time [s]"); ax.set_title(ttl)
    axs[0].set_ylabel("position RMSE [m]"); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=5, bbox_to_anchor=(.5, -.05), fontsize=6.8, columnspacing=1.2)
    for ax, s in zip(axs, "abc"): tag(ax, s)
    fig.tight_layout(rect=(0, .1, 1, 1)); save(fig, "horizon")

ROLES13 = ["test_id", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory",
           "val_id", "val_ood_magnitude", "val_ood_direction", "val_ood_temporal", "val_ood_gust", "val_ood_physics"]
def fig_heat(metric="pos_rmse_m", fname="heatmap", unit="position RMSE at 1 s [m]"):
    E = load_eval(); cv = json.load(open(ROOT / "outputs/eval/constant_velocity_v07.json"))
    lab = {k: l for k, l, _ in MODELS}
    rows = [("constant velocity", None), (lab["b1_kv"], "b1_kv"), (lab["kv_v07_b2_s0"], "kv_v07_b2_s0"), (lab["kv_v07_b3_s0"], "kv_v07_b3_s0"), ("B3 small (60k)", "kv_v07_b3_small_s0"),
            (lab["kv_v07_b4_rw_s0"], "kv_v07_b4_rw_s0"), (lab["kv_v07_b4_s0"], "kv_v07_b4_s0"), (lab["kv_v07_b5a_s0"], "kv_v07_b5a_s0"), (lab["kv_v07_b5b_s0"], "kv_v07_b5b_s0"),
            (lab["b6a_future_kv"], "b6a_future_kv"), (lab["kv_v07_b6b_future_s0"], "kv_v07_b6b_future_s0")]
    Mx = np.full((len(rows), len(ROLES13)), np.nan)
    for i, (lab, key) in enumerate(rows):
        for j, r in enumerate(ROLES13):
            if key is None: Mx[i, j] = cv.get(r, np.nan) if metric == "pos_rmse_m" else np.nan
            else: Mx[i, j] = E[key][r][metric]["20"]
    fig, ax = plt.subplots(figsize=(W2, 3.4))
    im = ax.imshow(Mx, cmap="viridis_r", norm=LogNorm(vmin=.005 if metric == "pos_rmse_m" else .03, vmax=2.5 if metric == "pos_rmse_m" else 40), aspect="auto"); ax.grid(False)
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            v = Mx[i, j]
            if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=6, color="#888"); continue
            ax.text(j, i, f"{v:.3f}" if v < .1 else f"{v:.2f}", ha="center", va="center", fontsize=6.2, color="white" if (v > .35 or v < .02) else "black")
    ax.set_xticks(range(len(ROLES13))); ax.set_xticklabels([r.replace("_", " ").replace("ood", "OOD").replace("id", "ID").replace("test ", "").replace("val ", "val ") for r in ROLES13], rotation=40, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(6.5, color="white", lw=2); ax.text(3, -1.0, "test roles", ha="center", fontsize=7.5, fontweight="bold"); ax.text(9.5, -1.0, "validation twins", ha="center", fontsize=7.5, fontweight="bold")
    for k in range(4): ax.spines[["left", "bottom", "right", "top"]][0].set_visible(False) if False else None
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01); cb.set_label(unit, fontsize=7); cb.ax.tick_params(labelsize=6)
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, fname)


# ---------------------------------------------------------------- architecture diagram
def fig_arch():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Patch
    fig, ax = plt.subplots(figsize=(W2, 4.1)); ax.set_xlim(0, 14.2); ax.set_ylim(-1.25, 6.75); ax.axis("off"); ax.grid(False)
    LRN, PRI, AUX, DAT, RES = "#0072B2", "#009E73", "#E69F00", "#6b7886", "#7B4EA3"
    ax.add_patch(FancyBboxPatch((0.05, 3.95), 14.1, 2.75, boxstyle="round,pad=0,rounding_size=0.15", fc="#eef4fa", ec="#c9dcec", lw=.8, zorder=0))
    ax.add_patch(FancyBboxPatch((0.05, -0.55), 14.1, 4.2, boxstyle="round,pad=0,rounding_size=0.15", fc="#eef7f2", ec="#c4e3d3", lw=.8, zorder=0))
    ax.text(0.3, 6.55, "Latent path (memory)", fontsize=8, fontweight="bold", color=LRN, va="top")
    ax.text(0.3, 3.5, "Physical transition (physics prior + learned residual)", fontsize=8, fontweight="bold", color="#007a5a", va="top")
    def box(x, y, w, h, title, sub, fc, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none", zorder=3))
        ax.text(x + w / 2, y + h * .64, title, ha="center", va="center", fontsize=8.3, fontweight="bold", color=tc, zorder=4)
        ax.text(x + w / 2, y + h * .25, sub, ha="center", va="center", fontsize=6.4, color=tc, zorder=4)
    def arr(p, q, c="#222", ls="-", lw=1.1, rad=0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=c, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))
    def lab(x, y, s, c="#333", ha="center", fs=6.6): ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va="center", zorder=5, bbox=dict(fc="white", ec="none", pad=.6, alpha=.85))
    # ---- row 1
    yA, hA = 4.45, 1.1
    box(0.2, yA, 2.3, hA, "IMU source (6)", "real IMU  or  $\\varphi(\\tilde s)$", DAT)
    box(3.05, yA, 2.2, hA, "Encoder $E$", "MLP 6 → 128 → 64", LRN)
    box(6.75, yA, 2.3, hA, "Latent $F$", "GRU 68 → 128", LRN)
    box(9.95, yA, 2.1, hA, "Decoder $D$", "128 → 256 → 13", LRN)
    box(12.45, yA, 1.55, hA, "$\\hat s_t$", "state (13)", DAT)
    arr((2.5, yA + .55), (3.05, yA + .55)); arr((5.25, yA + .55), (6.0, yA + .55)); lab(5.62, yA + .85, "$e_t$ (64)", fs=6.3)
    ax.add_patch(Circle((6.2, yA + .55), .2, fc="white", ec="#222", lw=1, zorder=4)); ax.text(6.2, yA + .55, "+", ha="center", va="center", fontsize=9, zorder=5)
    arr((6.4, yA + .55), (6.75, yA + .55)); arr((6.2, yA + 1.45), (6.2, yA + .77)); ax.text(6.32, yA + 1.35, "$u_{t-1}$ (4)", ha="left", fontsize=6.6, color="#333")
    arr((9.05, yA + .55), (9.95, yA + .55)); lab(9.5, yA + .85, "$h_t$ (128)", c=LRN, fs=6.3); arr((12.05, yA + .55), (12.45, yA + .55))
    ax.add_patch(FancyArrowPatch((8.6, yA + hA), (7.4, yA + hA), arrowstyle="-|>", mutation_scale=7, lw=1, color=LRN, connectionstyle="arc3,rad=.9", zorder=2))
    ax.text(8.0, yA + hA + .62, "$h_{t-1}$", ha="center", fontsize=6.4, color=LRN)
    # ---- row 2
    box(0.45, 1.25, 1.9, 1.0, "$s$", "current state (13)", DAT)
    box(4.1, 2.05, 2.8, 0.95, "Residual $d_\\theta$", "142 → 256 → 256 → 12", RES)
    box(4.1, 0.4, 2.8, 0.95, "Prior $f_{\\mathrm{phys}}$", "exact rigid body, no aero", PRI)
    box(8.0, 1.2, 2.1, 1.1, "$\\Sigma$  →  RK4", "$\\dot s = f_{\\mathrm{phys}} + d_\\theta$", "#34495e")
    box(11.4, 1.2, 2.2, 1.1, "$\\tilde s_{t+1}$", "next state (13)", DAT)
    arr((2.35, 1.95), (4.1, 2.5), rad=-.06); arr((2.35, 1.55), (4.1, .9), rad=.06)
    arr((6.9, 2.5), (8.0, 2.0)); arr((6.9, .9), (8.0, 1.5)); arr((10.1, 1.75), (11.4, 1.75))
    # h_t into residual
    ax.plot([9.5, 9.5], [yA, 3.25], color=LRN, lw=1.1, zorder=2); ax.plot([9.5, 5.5], [3.25, 3.25], color=LRN, lw=1.1, zorder=2); arr((5.5, 3.25), (5.5, 3.0), LRN)
    lab(7.6, 3.25, "$h_t$ (128)", c=LRN, fs=6.3)
    ax.text(3.2, 0.1, "action $u$ (4) enters $d_\\theta$ and $f_{\\mathrm{phys}}$", fontsize=6.4, color="#444", ha="center")
    # synthetic IMU feedback
    box(7.6, -1.15, 4.6, 0.85, "$\\varphi$: synthetic IMU", "$[R(\\hat q)^{\\top}(\\dot v - g),\\ \\hat\\omega]$, no parameters", AUX, tc="black")
    arr((12.5, 1.2), (12.5, -.72)); ax.plot([12.5, 12.2], [-.72, -.72], color="#222", lw=1.1, zorder=2)
    ax.plot([7.6, 1.25], [-.72, -.72], color=AUX, lw=1.5, ls="--", zorder=2); ax.plot([1.25, 1.25], [-.72, 3.7], color=AUX, lw=1.5, ls="--", zorder=2); arr((1.25, 3.7), (1.25, yA), AUX, ls="--", lw=1.5)
    lab(3.9, -.72, "$\\varphi(\\tilde s)$ is fed back in place of the real IMU", c="#9a6b00", fs=6.5)
    fig.legend(handles=[Patch(fc=LRN, label="learned (latent path)"), Patch(fc=RES, label="learned residual"), Patch(fc=PRI, label="fixed physics prior"), Patch(fc=AUX, label="no parameters"), Patch(fc=DAT, label="data / state")],
               loc="lower center", ncol=5, bbox_to_anchor=(.5, -.03), fontsize=7, handlelength=1.2, columnspacing=1.5)
    save(fig, "arch")


# ---------------------------------------------------------------- training vs open-loop imagination
def fig_modes():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    LRN, PRI, DAT, LOSS, OUT = "#0072B2", "#009E73", "#6b7886", "#C0392B", "#2E8B57"
    fig, axs = plt.subplots(2, 1, figsize=(W2, 6.6))
    def setup(ax, ttl, col):
        ax.set_xlim(0, 14.2); ax.set_ylim(-1.35, 5.55); ax.axis("off"); ax.grid(False)
        ax.add_patch(FancyBboxPatch((0.05, -1.3), 14.1, 6.8, boxstyle="round,pad=0,rounding_size=0.15", fc="#fafbfc", ec="#d5dbe0", lw=.8, zorder=0))
        ax.text(0.3, 5.35, ttl, fontsize=8.5, fontweight="bold", color=col, va="top")
    def box(ax, x, y, w, h, title, sub, fc, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none", zorder=3))
        ax.text(x + w / 2, y + h * .66, title, ha="center", va="center", fontsize=7.8, fontweight="bold", color=tc, zorder=4)
        ax.text(x + w / 2, y + h * .27, sub, ha="center", va="center", fontsize=6.5, color=tc, zorder=4, linespacing=1.15)
    def arr(ax, p, q, c="#222", ls="-", lw=1.1, rad=0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=c, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))
    xs = [4.2, 6.7, 11.4]; w = 2.0; yS = 1.55; hS = 1.35
    def common(ax):
        box(ax, 0.3, 1.55, 2.7, 1.35, "Context", "history, 10 steps:\nreal IMU + $u$ → $E$ → $F$", LRN)
        box(ax, 0.3, -0.15, 2.7, 0.95, "Start state $s_t$", "true state (given)", DAT)
        for x, k in zip(xs, ("$k=1$", "$k=2$", "$k=K$")): box(ax, x, yS, w, hS, f"Step {k}", "action $u_k$\n$f_{\\mathrm{phys}}+d_\\theta$ → RK4\n$\\varphi$ → $E$, $F$", PRI)
        ax.text(9.55, yS + hS / 2, "…", fontsize=14, ha="center", va="center")
        arr(ax, (3.0, yS + .65), (4.2, yS + .65)); arr(ax, (6.2, yS + .65), (6.7, yS + .65)); arr(ax, (8.7, yS + .65), (9.2, yS + .65)); arr(ax, (9.9, yS + .65), (11.4, yS + .65))
        ax.text(3.6, yS + .95, "$h_t$", fontsize=6.6, color=LRN, ha="center")
        arr(ax, (3.0, .32), (4.5, yS), rad=.15); ax.text(3.15, .75, "$s_t$", fontsize=6.6, color="#444")
    # ---- (a) training
    ax = axs[0]; setup(ax, "(a) Training: context, then free-running rollout with a loss at every step", LOSS); common(ax)
    box(ax, 0.3, 3.55, 2.7, 0.95, "$\\mathcal{L}_{\\mathrm{dec}}$", "$D(h_j)$ vs true $s_j$, all 10 history steps", LOSS)
    arr(ax, (1.65, 2.9), (1.65, 3.55), LOSS)
    for x, k in zip(xs, ("1", "2", "K")):
        box(ax, x, 3.55, w, 0.95, f"$\\mathcal{{L}}_{{\\mathrm{{state}},{k}}}$", f"$\\tilde s_{{t+{k}}}$ vs true $s_{{t+{k}}}$,  $\\gamma^{{{('0' if k == '1' else ('1' if k == '2' else 'K-1'))}}}$", LOSS)
        arr(ax, (x + w / 2, yS + hS), (x + w / 2, 3.55), LOSS)
    box(ax, 4.2, -0.15, 9.2, 0.95, "$\\mathcal{L}_{\\mathrm{res}}$ = mean $\\|d_\\theta\\|^2$ (dimensionless output)", "keeps the residual small: the prior should explain as much as possible", LOSS)
    for x in xs: arr(ax, (x + w / 2, yS), (x + w / 2, .8), LOSS, ls=":", lw=.9)
    ax.text(7.1, -.95, "$\\mathcal{L}=\\lambda_{\\mathrm{dec}}\\mathcal{L}_{\\mathrm{dec}}+\\frac{\\lambda_{\\mathrm{roll}}}{\\sum\\gamma^{k-1}}\\sum_k\\gamma^{k-1}\\mathcal{L}_{\\mathrm{state},k}+\\lambda_d\\mathcal{L}_{\\mathrm{res}}$;   gradients flow back through the whole rollout;  horizon $K$: 1 → 5 → 10 → 20 (curriculum)", fontsize=6.8, ha="center", color=LOSS)
    arr(ax, (12.4, 3.3), (1.9, 3.3), "#C0392B", ls="--", lw=.9, rad=0); ax.text(7.1, 3.12, "gradients (backpropagation through the rollout)", fontsize=6.2, color="#C0392B", ha="center", style="italic")
    # ---- (b) imagination
    ax = axs[1]; setup(ax, "(b) Open-loop imagination (evaluation and use): no targets, no loss, no real IMU in the rollout", OUT); common(ax)
    box(ax, 0.3, 3.55, 2.7, 0.95, "Inputs", "history window + true $s_t$\n+ future actions $u_{1..K}$ only", DAT)
    arr(ax, (1.65, 2.9), (1.65, 3.55), DAT)
    for x, k in zip(xs, ("1", "2", "K")):
        box(ax, x, 3.55, w, 0.95, f"$\\tilde s_{{t+{k}}}$", "predicted state", OUT)
        arr(ax, (x + w / 2, yS + hS), (x + w / 2, 3.55), OUT)
    ax.text(7.1, -.75, "Output: predicted trajectory over horizon $K$ (up to 50 steps).  The model never sees the real IMU or the true future states after the context window.", fontsize=6.8, ha="center", color="#333")
    ax.text(7.1, -1.1, "Evaluation then compares it with ground truth offline: position RMSE against horizon, reliable imagination horizon.", fontsize=6.8, ha="center", color=OUT)
    for a, s_ in zip(axs, "ab"): pass
    fig.tight_layout(h_pad=1.0); save(fig, "modes")

# ---------------------------------------------------------------- trajectories
def fig_traj():
    Z = np.load(CACHE / "traj_data.npz"); keys = sorted({k.rsplit("|", 1)[0] for k in Z.files})
    pick = [("test_id|id_cl_g01323_e0", "ID, no wind"), ("test_ood_magnitude|ood_mag_g00193_e0", "OOD wind magnitude 8.1 m/s")]
    mods = [("b1", "B1", "B1"), ("b2", "B2", "B2"), ("b3", "B3 proposed", "B3"), ("b4_rw", "B4 (B3 width)", "B4w")]
    fig, axs = plt.subplots(3, 2, figsize=(W2, 7.0))
    for row, (k, ttl) in enumerate(pick):
        tr = Z[k + "|truth"]; st = Z[k + "|starts"]; dt = float(Z[k + "|dt"]); wnd = Z[k + "|wind"]
        t = np.arange(len(tr)) * dt; ax0, ax1, ax2 = axs[0][row], axs[1][row], axs[2][row]
        ax0.plot(tr[:, 0], tr[:, 1], color="k", lw=1.6, label="ground truth", zorder=10); ax1.plot(t, tr[:, 2], color="k", lw=1.6, zorder=10)
        errs = {}
        for m, lab, c in mods:
            P = Z[f"{k}|pred_{m}"]; e = []
            for j, s0 in enumerate(st):
                p = P[j]; x = np.arange(1, p.shape[0] + 1)
                ax0.plot(p[:, 0], p[:, 1], color=C[c], lw=.8, alpha=.9, label=lab if j == 0 else None)
                ax1.plot((s0 + x) * dt, p[:, 2], color=C[c], lw=.8, alpha=.9)
                e.append(np.linalg.norm(p - tr[s0 + x, :3], axis=1))
            errs[m] = np.mean(e, 0); ax2.plot(np.arange(1, len(errs[m]) + 1) * dt, errs[m], color=C[c], lw=1.4, label=lab)
        ax0.set_aspect("equal", "datalim"); ax0.set_xlabel("x [m] (east)"); ax0.set_ylabel("y [m] (north)"); ax1.set_xlabel("time [s]"); ax1.set_ylabel("z [m]")
        ax2.set_xlabel("time into rollout [s]"); ax2.set_ylabel("mean position error [m]")
        if wnd[0] > 0.5:
            u = np.array([np.cos(np.radians(wnd[1])), np.sin(np.radians(wnd[1]))]); x0, x1 = ax0.get_xlim(); y0, y1 = ax0.get_ylim()
            ax0.annotate("", xy=(x0 + .12 * (x1 - x0) + .18 * (x1 - x0) * u[0], y1 - .14 * (y1 - y0) + .18 * (y1 - y0) * u[1]), xytext=(x0 + .12 * (x1 - x0), y1 - .14 * (y1 - y0)),
                         arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.6)); ax0.text(x0 + .12 * (x1 - x0), y1 - .02 * (y1 - y0), f"wind {wnd[0]:.1f} m/s", color="#D55E00", fontsize=7, va="top")
        ax0.set_title(f"{ttl}: top view", fontsize=8); ax1.set_title("altitude"); ax2.set_title("error growth")
        tag(ax0, "ab"[row])
        print(ttl, {m: round(float(v[-1]), 3) for m, v in errs.items()})
    h_, l_ = axs[0][0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=5, bbox_to_anchor=(.5, -.01), fontsize=7.5)
    fig.tight_layout(h_pad=1.4, w_pad=1.5, rect=(0, .04, 1, 1)); save(fig, "trajectories")

# ======================================================================= more figures (report checklist)
def _cache(name):
    f = CACHE / name
    if not f.exists():
        raise SystemExit(f"{f} missing: run `python -m physiworld_uav.visualization figure-data` in the container first")
    return np.load(f, allow_pickle=True) if name.endswith(".npz") else json.load(open(f))
def _ev(*files):
    E = {}
    for f in files:
        for m, r in json.load(open(ROOT / "outputs/eval" / f)).items(): E[m] = r
    return E
OOD6 = ["test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory"]
MC = {"cv": C["CV"], "b1_kv": C["B1"], "b2": C["B2"], "b3": C["B3"], "b4_rw": C["B4w"], "b4": C["B4"], "b5b": C["B5b"]}
MN = {"cv": "constant velocity", "b1_kv": "B1 physics", "b2": "B2 pure latent", "b3": "B3 proposed", "b4_rw": "B4 (B3 width)", "b4": "B4", "b5b": "B5b"}

def fig_error_accum():
    """Plot 1: ground truth and open-loop predictions of increasing horizon (0.25 / 0.5 / 1 s) on one OOD episode."""
    Z = _cache("traj_data.npz"); k = "test_ood_magnitude|ood_mag_g00193_e0"
    if k + "|truth" not in Z.files: k = [x.rsplit("|", 1)[0] for x in Z.files if x.startswith("test_ood_magnitude") and x.endswith("|truth")][-1]
    tr, st, dt, wnd = Z[k + "|truth"], Z[k + "|starts"], float(Z[k + "|dt"]), Z[k + "|wind"]; P = Z[k + "|pred_b3"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    a = ax[0]; a.plot(tr[:, 0], tr[:, 1], "k", lw=1.8, label="ground truth", zorder=10)
    for hh, c, lab, lw_, z in ((20, "#D55E00", "long: 1.0 s", 1.3, 3), (10, "#E69F00", "medium: 0.5 s", 1.8, 4), (5, "#56B4E9", "short: 0.25 s", 2.4, 5)):
        for j in range(len(st)): a.plot(P[j, :hh, 0], P[j, :hh, 1], color=c, lw=lw_, alpha=.95, label=lab if j == 0 else None, zorder=z)
    a.set_aspect("equal", "datalim"); a.set_xlabel("x [m] (east)"); a.set_ylabel("y [m] (north)"); a.legend(loc="lower right", fontsize=6.5); a.set_title(f"B3 predictions, wind {wnd[0]:.1f} m/s"); tag(a, "a")
    b = ax[1]; errs = []
    for j, s0 in enumerate(st): errs.append(np.linalg.norm(P[j] - tr[s0 + 1: s0 + 21], axis=1))
    errs = np.array(errs); t = np.arange(1, 21) * dt
    b.fill_between(t, np.percentile(errs, 10, 0), np.percentile(errs, 90, 0), color=C["B3"], alpha=.2, lw=0); b.plot(t, errs.mean(0), color=C["B3"], lw=1.6, label="mean over rollouts")
    b.set_xlabel("time into the rollout [s]"); b.set_ylabel("position error [m]"); b.legend(); b.set_title("Error accumulates with the horizon"); tag(b, "b")
    fig.tight_layout(w_pad=2); save(fig, "error_accumulation")

def fig_coverage():
    """Plot 3: trajectory coverage, one small 3D plot per scenario template (4 training episodes each, coloured by episode; dot = start)."""
    X = _cache("extra.npz"); temps = sorted({k.split("|")[1] for k in X.files if k.startswith("cov|")}, key=lambda t: (list(FC).index(FAM[t]), t))
    n = len(temps); cols = 4; rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(W2, 1.75 * rows)); pal = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for i, t in enumerate(temps):
        a = fig.add_subplot(rows, cols, i + 1, projection="3d"); allp = []
        for j in range(4):
            k = f"cov|{t}|{j}"
            if k not in X.files: continue
            p = X[k]; allp.append(p); a.plot(p[:, 0], p[:, 1], p[:, 2], color=pal[j], lw=1.0); a.plot([p[0, 0]], [p[0, 1]], [p[0, 2]], "o", ms=2.5, color=pal[j])
        P = np.concatenate(allp); lo, hi = P.min(0), P.max(0); c = (lo + hi) / 2; r_ = max((hi - lo).max() / 2, 0.15)
        a.set_xlim(c[0] - r_, c[0] + r_); a.set_ylim(c[1] - r_, c[1] + r_); a.set_zlim(c[2] - r_, c[2] + r_)
        a.set_title(t.replace("_", " "), fontsize=7.5, color=FC[FAM[t]], pad=0, fontweight="bold"); a.view_init(22, -55)
        a.tick_params(labelsize=5, pad=-3); a.set_box_aspect((1, 1, 1))
        for ax_ in (a.xaxis, a.yaxis, a.zaxis): ax_.set_major_locator(plt.MaxNLocator(3))
    fig.text(.5, .005, "3D trajectories (x east, y north, z up, metres; each panel has its own scale, equal on all axes). Title colour = scenario family: "
             + ", ".join(f"{k}" for k in FC if k != "open loop"), ha="center", fontsize=6.5, style="italic")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.03, top=0.96, wspace=0.0, hspace=0.12); save(fig, "coverage")

def fig_wind_series():
    """Plot 4 / Fig 8: wind speed against time for each wind profile type (one example episode each)."""
    X = _cache("extra.npz"); names = [("steady", "steady"), ("ou", "Ornstein-Uhlenbeck"), ("cosine_gust", "1-cos gust"), ("step", "step"), ("sinusoid", "sinusoid"), ("random_gusts", "random gust events")]
    fig, axs = plt.subplots(3, 2, figsize=(W2, 5.6), sharex=True)
    for a, (k, ttl), s_ in zip(axs.ravel(), names, "abcdef"):
        w = X[f"wind|{k}"]; t = np.arange(len(w)) * 0.025; sp = np.linalg.norm(w, axis=1)
        a.plot(t, sp, color="#0072B2", lw=1.1, label="speed"); a.plot(t, w[:, 0], color="#E69F00", lw=.7, alpha=.8, label="$w_x$"); a.plot(t, w[:, 1], color="#009E73", lw=.7, alpha=.8, label="$w_y$")
        a.set_title(ttl); a.set_ylabel("m/s"); tag(a, s_)
    axs[2, 0].set_xlabel("time [s]"); axs[2, 1].set_xlabel("time [s]"); axs[0, 0].legend(ncol=3, fontsize=6.5, loc="upper left"); fig.tight_layout(h_pad=1.0); save(fig, "wind_series")

def fig_dr_grid():
    """Plot 5: histograms of the six randomised vehicle parameters (nominal = 0, ID box dashed)."""
    rows = [m["domain"]["factors"] for m in man if m["split_role"] != "physics_validation"]; ph = np.array([m["split_role"].endswith("ood_physics") for m in man if m["split_role"] != "physics_validation"])
    spec = [("mass", "Mass", 10), ("force_scale", "Thrust coefficient $k_F$", 10), ("km_kf_ratio", "Torque ratio $k_M/k_F$", 5), ("motor_time_constant", "Motor time constant", 20), ("body_drag_scale", "Body drag scale", 10), ("rotor_drag_scale", "Rotor drag scale", 10)]
    fig, axs = plt.subplots(3, 2, figsize=(W2, 5.2))
    for a, (k, ttl, hw), s_ in zip(axs.ravel(), spec, "abcdef"):
        v = np.array([(r[k] - 1) * 100 for r in rows]); lim = max(hw * 2.2, 5); b = np.linspace(-lim, lim, 33)
        a.hist(v[~ph], b, color="#7f8c99", alpha=.85, label="other roles"); a.hist(v[ph], b, color="#E69F00", alpha=.95, label="OOD physics")
        a.axvline(-hw, color="k", ls="--", lw=.7); a.axvline(hw, color="k", ls="--", lw=.7); a.set_title(ttl); a.set_xlabel("deviation from nominal [%]"); a.set_ylabel("episodes"); tag(a, s_)
    jx = np.array([[(r[k] - 1) * 100 for k in ("inertia_shape_x", "inertia_shape_y", "inertia_shape_z")] for r in rows]); axs[0, 0].legend(fontsize=6.5)
    fig.tight_layout(h_pad=1.0, w_pad=2); save(fig, "domain_randomization")

def _traj(X, key): return X[f"cf|{key}|pos"]
def fig_counterfactual():
    """Plot 6: counterfactual siblings: same reference and initial state, wind scaled 0 / 0.5 / 1 (a, b) and perturbed vehicle (c, d)."""
    X = _cache("extra.npz"); ids = sorted({k.split("|")[1] for k in X.files if k.startswith("cf|") and k.endswith("|pos")}, key=lambda i: float(X[f"cf|{i}|wind"][1]))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.4)); cols = ["#555555", "#56B4E9", "#D55E00"]
    base = _traj(X, ids[0])
    for i, e in enumerate(ids):
        p = _traj(X, e); sc, spd = float(X[f"cf|{e}|wind"][1]), float(X[f"cf|{e}|wind"][0]); lab = f"wind $\\times${sc:g} (mean {spd:.1f} m/s)" if spd > 0 else "no wind"
        axs[0, 0].plot(p[:, 0], p[:, 1], color=cols[i], lw=1.3, label=lab); n = min(len(p), len(base)); t = np.arange(n) * 0.025
        axs[0, 1].plot(t, np.linalg.norm(p[:n, :2] - base[:n, :2], axis=1), color=cols[i], lw=1.3, label=lab)
    axs[0, 0].set_aspect("equal", "datalim"); axs[0, 0].set_xlabel("x [m]"); axs[0, 0].set_ylabel("y [m]"); axs[0, 0].legend(fontsize=6.3); axs[0, 0].set_title("Same start and reference, different wind"); tag(axs[0, 0], "a")
    axs[0, 1].set_xlabel("time [s]"); axs[0, 1].set_ylabel("horizontal deviation from baseline [m]"); axs[0, 1].set_title("Effect of the wind"); tag(axs[0, 1], "b")
    dyn = sorted({k.split("|")[1] for k in X.files if k.startswith("cfd|") and k.endswith("|pos")}, key=lambda i: str(X[f"cfd|{i}|dom"]))
    for i, e in enumerate(dyn):
        p = X[f"cfd|{e}|pos"]; m = float(X[f"cfd|{e}|mass"]); axs[1, 0].plot(p[:, 0], p[:, 1], color=["#555555", "#009E73", "#CC79A7", "#E69F00"][i % 4], lw=1.3, label=f"{str(X[f'cfd|{e}|dom'])}: m = {m:.3f} kg")
        if i == 0: b0 = p
        t = np.arange(min(len(p), len(b0))) * 0.025; axs[1, 1].plot(t, np.linalg.norm(p[:len(t)] - b0[:len(t)], axis=1), color=["#555555", "#009E73", "#CC79A7", "#E69F00"][i % 4], lw=1.3)
    axs[1, 0].set_aspect("equal", "datalim"); axs[1, 0].set_xlabel("x [m]"); axs[1, 0].set_ylabel("y [m]"); axs[1, 0].legend(fontsize=6.3); axs[1, 0].set_title("Same wind and reference, perturbed vehicle"); tag(axs[1, 0], "c")
    axs[1, 1].set_xlabel("time [s]"); axs[1, 1].set_ylabel("deviation from nominal vehicle [m]"); axs[1, 1].set_title("Effect of the dynamics"); tag(axs[1, 1], "d")
    fig.tight_layout(h_pad=1.2, w_pad=2); save(fig, "counterfactual")

def fig_imu():
    """Plot 7: ground-truth (clean) IMU and the noisy measurement used by the models (nominal preset; no filter is applied)."""
    X = _cache("extra.npz"); c, n = X["imu|clean"], X["imu|noisy"]; dt = float(X["imu|dt"]); t = np.arange(len(c)) * dt
    chans = [(2, "accelerometer $z$ [m/s$^2$]"), (0, "accelerometer $x$ [m/s$^2$]"), (5, "gyroscope $z$ [rad/s]"), (3, "gyroscope $x$ [rad/s]")]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.4), sharex=True)
    for a, (j, ttl), s_ in zip(axs.ravel(), chans, "abcd"):
        a.plot(t, n[:, j], color="#E69F00", lw=.6, alpha=.9, label="noisy (model input)"); a.plot(t, c[:, j], color="#0072B2", lw=1.2, label="clean (ground truth)"); a.set_ylabel(ttl); tag(a, s_)
    axs[1, 0].set_xlabel("time [s]"); axs[1, 1].set_xlabel("time [s]"); axs[0, 0].legend(fontsize=6.5, loc="upper right")
    fig.text(.5, -.01, "Noise: bias (constant per 0.5 s window) + white noise; accelerometer $\\sigma_w$=0.10 m/s$^2$, $\\sigma_b$=0.05; gyroscope $\\sigma_w$=0.010 rad/s, $\\sigma_b$=0.005. No filtering is applied.", ha="center", fontsize=6.5, style="italic")
    fig.tight_layout(h_pad=1.0, rect=(0, .02, 1, 1)); save(fig, "imu")

VERS = [("smoke_v0.3.0", "Smoke\nv0.3"), ("physiworld_pilot_v0.4.0-dev", "v0.4\n(partial)"), ("physiworld_pilot_v0.5.0", "v0.5"), ("physiworld_pilot_v0.6.0", "v0.6"), ("physiworld_pilot_v7_v0.7.0", "v7")]
def _vers():
    out = []
    for d, lab in VERS:
        ix = ROOT / "data/datasets" / d / "index.json"
        if not ix.exists(): continue
        j = json.load(open(ix)); eps = j["episodes"]; st = collections.Counter(e["status"] for e in eps)
        size = sum(f.stat().st_size for f in (ROOT / "data/datasets" / d).rglob("*") if f.is_file()) / 1e6
        steps = sum(e["n_steps"] for e in eps) * 0.005 / 3600
        out.append(dict(d=d, lab=lab, n=len(eps), hours=steps, status=st, wall=j.get("generation_wall_s"), mb=size, roles=collections.Counter(e["split_role"] for e in eps)))
    return out
def fig_dataset_versions():
    """Plots 8 and 10: dataset size by version (trajectories, simulated time) and generation effort."""
    V = _vers(); x = np.arange(len(V)); fig, axs = plt.subplots(2, 2, figsize=(W2, 5.2))
    a = axs[0, 0]; a.bar(x, [v["n"] for v in V], color="#0072B2", width=.65); a.set_yscale("log")
    for i, v in enumerate(V): a.text(i, v["n"] * 1.15, f"{v['n']:,}", ha="center", fontsize=7)
    a.set_title("Trajectories"); a.set_ylabel("episodes (log)"); tag(a, "a")
    b = axs[0, 1]; b.bar(x, [v["hours"] for v in V], color="#009E73", width=.65); b.set_yscale("log")
    for i, v in enumerate(V): b.text(i, v["hours"] * 1.15, f"{v['hours']:.2f}", ha="center", fontsize=7)
    b.set_title("Total simulated time"); b.set_ylabel("hours (log)"); tag(b, "b")
    c = axs[1, 0]; bot = np.zeros(len(V))
    for stt, col in (("completed", "#009E73"), ("terminated_ground", "#E69F00"), ("terminated_envelope", "#D55E00")):
        vv = np.array([v["status"].get(stt, 0) / v["n"] * 100 for v in V]); c.bar(x, vv, bottom=bot, color=col, width=.65, label=stt.replace("_", " ")); bot += vv
    c.set_title("Episode outcome"); c.set_ylabel("share of episodes [%]"); c.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3); tag(c, "c")
    d_ = axs[1, 1]; ok = [(i, v) for i, v in enumerate(V) if v["wall"] and v["n"] >= 100 and v["d"] != "physiworld_pilot_v0.5.0"]
    d_.bar([i for i, _ in ok], [v["n"] / (v["wall"] / 3600) for _, v in ok], color="#7B4EA3", width=.65)
    for i, v in ok: d_.text(i, v["n"] / (v["wall"] / 3600) * 1.03, f"{v['n'] / (v['wall'] / 3600):.0f}\n({v['wall'] / 3600:.1f} h)", ha="center", fontsize=7)
    d_.set_title("Generation throughput"); d_.set_ylabel("episodes / hour"); tag(d_, "d")
    for a_ in axs.ravel(): a_.set_xticks(x); a_.set_xticklabels([v["lab"] for v in V], fontsize=7)
    fig.tight_layout(h_pad=1.2, w_pad=2); save(fig, "dataset_versions")

def fig_dataset_dist():
    """Plot 9: how the wind distribution changed across dataset versions (training wind, and the OOD-magnitude evaluation wind)."""
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7)); cols = {"physiworld_pilot_v0.5.0": "#9a9a9a", "physiworld_pilot_v0.6.0": "#56B4E9", "physiworld_pilot_v7_v0.7.0": "#D55E00"}
    labs = {"physiworld_pilot_v0.5.0": "v0.5", "physiworld_pilot_v0.6.0": "v0.6", "physiworld_pilot_v7_v0.7.0": "v7"}; b = np.arange(0, 10.01, .5)
    for d, col in cols.items():
        f = ROOT / "data/datasets" / d / "manifest.jsonl"
        if not f.exists(): continue
        M = [json.loads(l) for l in open(f)]
        tr = [m["wind"]["stats"]["peak_speed"] for m in M if m["split_role"] == "train" and m["wind"]["profile"] != "none"]; oo = [m["wind"]["stats"]["peak_speed"] for m in M if m["split_role"] == "test_ood_magnitude"]
        ls = "--" if d.endswith("v0.5.0") else "-"; axs[0].hist(tr, b, histtype="step", color=col, lw=1.6, ls=ls, density=True, label=f"{labs[d]} (n={len(tr)})"); axs[1].hist(oo, b, histtype="step", color=col, lw=1.6, ls=ls, density=True, label=f"{labs[d]} (n={len(oo)})")
    axs[0].set_title("Training wind: peak speed"); axs[1].set_title("OOD-magnitude test wind: peak speed")
    for a, s_ in zip(axs, "ab"): a.set_xlabel("peak wind speed [m/s]"); a.set_ylabel("density"); a.legend(fontsize=6.5); tag(a, s_)
    fig.tight_layout(w_pad=2); save(fig, "dataset_distribution")

def fig_train_loss():
    """Plot 11: training loss and validation error per epoch (proposed model and baselines), stage 1 and known vehicle."""
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.2), sharex="col")
    for col, (pre, ttl) in enumerate((("st_v07", "stage 1"), ("kv_v07", "known vehicle"))):
        for k, c in (("b2", "B2"), ("b3", "B3"), ("b4_rw", "B4w")):
            r = [json.loads(l) for l in open(ROOT / "outputs/experiments" / f"{pre}_{k}_s0" / "metrics.jsonl")]
            e = [x["epoch"] for x in r]; lab = {"B2": "B2 pure latent", "B3": "B3 proposed", "B4w": "B4 (B3 width)"}[c]
            axs[0, col].plot(e, [x["train"]["roll"] for x in r], color=C[c], lw=1.3, label=lab); axs[1, col].plot(e, [x["val"]["pos_rmse_m"]["20"] for x in r], color=C[c], lw=1.3)
        axs[0, col].set_yscale("log"); axs[1, col].set_yscale("log"); axs[0, col].set_title(f"Training rollout loss ({ttl})"); axs[1, col].set_title(f"Validation position RMSE at 1 s ({ttl})")
        axs[1, col].set_xlabel("epoch"); tag(axs[0, col], "ac"[col]); tag(axs[1, col], "bd"[col])
    axs[0, 0].set_ylabel("$\\mathcal{L}_{roll}$ (Appendix B)"); axs[1, 0].set_ylabel("RMSE [m]"); axs[0, 0].legend(fontsize=6.5)
    axs[0, 0].text(.5, -.1, "", transform=axs[0, 0].transAxes)
    fig.text(.5, -.01, "The rollout loss changes scale when the curriculum horizon K increases (K = 1, 5, 10, 20), so jumps are expected; validation uses the fixed 1 s horizon.", ha="center", fontsize=6.5, style="italic")
    fig.tight_layout(h_pad=1.0, w_pad=2, rect=(0, .02, 1, 1)); save(fig, "train_loss")

def fig_versions_perf():
    """Plot 12: model performance across dataset versions (note: v0.5/v0.6 horizon = 0.4 s, v7 = 1 s, different data and vehicles)."""
    sets = [("v0.5", _ev("v05.json"), {"b2": "b2", "b3": "b3", "b4_rw": "b4_rw", "b5b": "b5b"}), ("v0.6", _ev("v06.json"), {"b2": "b2", "b3": "b3", "b4_rw": "b4_rw", "b5b": "b5b"}),
            ("v7 stage 1", _ev("st_v07_c0.json", "st_v07_c1.json", "st_v07_c2.json"), {k: f"st_v07_{k}_s0" for k in ("b2", "b3", "b4_rw", "b5b")}),
            ("v7 known veh.", _ev("kv_v07_c1.json", "kv_v07_d_c1.json", "kv_v07_c2.json"), {k: f"kv_v07_{k}_s0" for k in ("b2", "b3", "b4_rw", "b5b")})]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9)); w = .2; x = np.arange(len(sets)); names = [("b2", "B2"), ("b3", "B3"), ("b4_rw", "B4w"), ("b5b", "B5b")]
    for ax, role, ttl, s_ in zip(axs, ("test_id", "test_ood_magnitude"), ("Test ID", "OOD wind magnitude"), "ab"):
        for i, (k, c) in enumerate(names):
            vals = [E[key[k]][role]["pos_rmse_m"]["20"] for _, E, key in sets]; ax.bar(x + (i - 1.5) * w, vals, w, color=C[c], label={"B2": "B2", "B3": "B3 proposed", "B4w": "B4 (B3 width)", "B5b": "B5b"}[c])
        ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({'0.4' if n in ('v0.5', 'v0.6') else '1.0'} s)" for n, _, _ in sets], fontsize=7); ax.set_title(ttl); ax.set_ylabel("position RMSE at last step [m]"); tag(ax, s_)
    axs[0].legend(fontsize=6.3, ncol=2, loc="upper left"); fig.tight_layout(w_pad=2); save(fig, "versions_performance")

def fig_id_ood():
    """Plot 14: ID vs OOD error and the degradation Delta = E_OOD - E_ID (known-vehicle runs, position RMSE at 1 s)."""
    E = _ev("kv_v07_c0.json", "kv_v07_c1.json", "kv_v07_c2.json", "kv_v07_d_c1.json", "kv_v07_d_c2.json"); cv = json.load(open(ROOT / "outputs/eval/constant_velocity_v07.json"))
    mods = [("cv", None), ("b1_kv", "b1_kv"), ("b2", "kv_v07_b2_s0"), ("b3", "kv_v07_b3_s0"), ("b4_rw", "kv_v07_b4_rw_s0"), ("b5b", "kv_v07_b5b_s0")]
    def val(key, role): return cv[role] if key is None else E[key][role]["pos_rmse_m"]["20"]
    idv = [val(k, "test_id") for _, k in mods]; ood = [np.mean([val(k, r) for r in OOD6 if (k is not None or r in cv)]) for _, k in mods]; mag = [val(k, "test_ood_magnitude") for _, k in mods]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0)); x = np.arange(len(mods)); w = .27
    axs[0].bar(x - w, idv, w, color="#0072B2", label="ID"); axs[0].bar(x, ood, w, color="#E69F00", label="OOD (mean of 6 roles)"); axs[0].bar(x + w, mag, w, color="#D55E00", label="OOD wind magnitude")
    axs[0].set_xticks(x); axs[0].set_xticklabels([MN[m].replace(" ", "\n", 1) for m, _ in mods], fontsize=6.5); axs[0].set_ylabel("position RMSE at 1 s [m]"); axs[0].legend(fontsize=6.3, loc="upper left"); axs[0].set_title("Absolute error"); tag(axs[0], "a")
    d1 = np.array(ood) - np.array(idv); d2 = np.array(mag) - np.array(idv); axs[1].bar(x - w / 2, d1, w, color="#E69F00", label="$\\Delta$ OOD (mean)"); axs[1].bar(x + w / 2, d2, w, color="#D55E00", label="$\\Delta$ OOD magnitude")
    axs[1].axhline(0, color="k", lw=.6); axs[1].set_xticks(x); axs[1].set_xticklabels([MN[m].replace(" ", "\n", 1) for m, _ in mods], fontsize=6.5); axs[1].set_ylabel("$E_{OOD}-E_{ID}$ [m]"); axs[1].legend(fontsize=6.3); axs[1].set_title("Degradation under distribution shift"); tag(axs[1], "b")
    fig.tight_layout(w_pad=2); save(fig, "id_vs_ood")

def _err():
    return _cache("err_data.json")
def fig_severity():
    """Plot 15: prediction error against OOD severity (mean wind speed of the episode), binned, per model."""
    R = _err(); fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9)); bins = np.array([0, 2, 4, 6, 7, 8, 9]); mids = (bins[:-1] + bins[1:]) / 2
    for ax, key, ttl, s_ in zip(axs, ("pos_rmse_20", "vel_rmse_20"), ("Position RMSE at 1 s", "Velocity RMSE at 1 s"), "ab"):
        for m in ("cv", "b1_kv", "b2", "b3", "b4_rw"):
            rr = [(r["wind"], r[key]) for r in R if r["model"] == m and r["role"] in ("test_id", "test_ood_magnitude")]
            w = np.array([a for a, _ in rr]); e = np.array([b for _, b in rr]); y = []; q1 = []; q3 = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                v = e[(w >= lo) & (w < hi)]; y.append(np.median(v) if len(v) > 3 else np.nan); q1.append(np.percentile(v, 25) if len(v) > 3 else np.nan); q3.append(np.percentile(v, 75) if len(v) > 3 else np.nan)
            ax.plot(mids, y, "-o", ms=3, color=MC[m], lw=1.4 if m == "b3" else 1.0, label=MN[m]); ax.fill_between(mids, q1, q3, color=MC[m], alpha=.08, lw=0)
        ax.axvline(6.2, color="k", ls="--", lw=.7); ax.set_yscale("log"); ax.set_xlabel("episode mean wind speed [m/s]"); ax.set_ylabel("median error per episode [m, resp. m/s]"); ax.set_title(ttl); tag(ax, s_)
    axs[0].legend(fontsize=6.3, loc="upper left"); axs[0].text(6.3, axs[0].get_ylim()[0] * 1.3, "ID limit", fontsize=6.5, rotation=90, va="bottom")
    fig.tight_layout(w_pad=2); save(fig, "ood_severity")

def fig_error_dist():
    """Plot 17: distribution of per-episode errors (box plots) for ID and OOD, at two horizons."""
    R = _err(); mods = ("cv", "b1_kv", "b2", "b3", "b4_rw"); fig, axs = plt.subplots(2, 2, figsize=(W2, 5.2), sharey="row")
    for r_, (key, hz) in enumerate((("pos_rmse_10", "0.5 s"), ("pos_rmse_20", "1.0 s"))):
        for c_, (roles, ttl) in enumerate((((["test_id"]), "ID (test)"), (OOD6, "OOD (six test roles)"))):
            a = axs[r_, c_]; data = [[r[key] for r in R if r["model"] == m and r["role"] in roles] for m in mods]
            bp = a.boxplot(data, whis=(5, 95), patch_artist=True, showfliers=True, flierprops=dict(marker=".", ms=2, alpha=.4), medianprops=dict(color="k", lw=.9), widths=.6)
            for b_, m in zip(bp["boxes"], mods): b_.set(fc=MC[m], ec="none", alpha=.85)
            a.set_yscale("log"); a.set_xticks(range(1, len(mods) + 1)); a.set_xticklabels([MN[m].replace(" ", "\n", 1) for m in mods], fontsize=6.5); a.set_title(f"{ttl}, horizon {hz}"); tag(a, "abcd"[r_ * 2 + c_])
        axs[r_, 0].set_ylabel("position RMSE per episode [m]")
    fig.tight_layout(h_pad=1.2, w_pad=2); save(fig, "error_distribution")

def fig_failure():
    """Plot 18: the worst B3 episode (largest 1 s error among OOD magnitude episodes), all models, with the failure annotated."""
    R = _err(); w = max([r for r in R if r["model"] == "b3" and r["role"] == "test_ood_magnitude"], key=lambda r: r["pos_rmse_20"]); ep = w["ep"]
    Z = _cache("traj_fail.npz"); k = f"fail|{ep}"
    tr, st, dt = Z[k + "|truth"], Z[k + "|starts"], float(Z[k + "|dt"]); fig, axs = plt.subplots(1, 2, figsize=(W2, 3.2))
    a = axs[0]; a.plot(tr[:, 0], tr[:, 1], "k", lw=1.8, label="ground truth", zorder=10)
    for m, c, lab in (("b1", "B1", "B1"), ("b2", "B2", "B2"), ("b3", "B3", "B3 proposed"), ("b4_rw", "B4w", "B4 (B3 width)")):
        P = Z[f"{k}|pred_{m}"]
        for j in range(len(st)): a.plot(P[j, :, 0], P[j, :, 1], color=C[c], lw=.9, alpha=.9, label=lab if j == 0 else None)
    a.set_aspect("equal", "datalim"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.legend(fontsize=6.3); a.set_title(f"Worst B3 case: wind {w['wind']:.1f} m/s, azimuth {w['az']:.0f}°"); tag(a, "a")
    b = axs[1]
    for m, c, lab in (("b1", "B1", "B1"), ("b2", "B2", "B2"), ("b3", "B3", "B3 proposed"), ("b4_rw", "B4w", "B4 (B3 width)")):
        P = Z[f"{k}|pred_{m}"]; e = np.array([np.linalg.norm(P[j] - tr[s0 + 1: s0 + 21, :3], axis=1)[-1] for j, s0 in enumerate(st)])
        b.plot((st + 20) * dt, e, "-o", ms=2.5, color=C[c], lw=1.2, label=lab)
    b.set_xlabel("time in the episode [s]"); b.set_ylabel("position error after 1 s [m]"); b.set_title("Error of each 1 s rollout along the episode"); b.legend(fontsize=6.3); tag(b, "b")
    fig.text(.5, -.02, f"Episode {ep}. What happened: strong unseen wind ({w['wind']:.1f} m/s, steady); where it began: the error is already large in the first rollout and then stays; which model failed: B3 ({w['pos_rmse_20']:.2f} m RMSE at 1 s), B4 and B2 do better here.", ha="center", fontsize=6.5, style="italic")
    fig.tight_layout(w_pad=2, rect=(0, .02, 1, 1)); save(fig, "failure_case")

def fig_frames_forces():
    """Figures 3 and 4 (schematics, not data): coordinate frames and the forces acting on the vehicle."""
    from matplotlib.patches import FancyArrowPatch, Circle
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.3))
    for a in axs: a.set_xlim(-1.6, 1.6); a.set_ylim(-1.3, 1.5); a.set_aspect("equal"); a.axis("off"); a.grid(False)
    def ar(a, p, q, c, lw=1.6, ls="-"): a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw, color=c, ls=ls, zorder=3))
    a = axs[0]; ar(a, (-1.3, -1.0), (-0.3, -1.0), "#0072B2"); ar(a, (-1.3, -1.0), (-1.3, 0.0), "#009E73"); ar(a, (-1.3, -1.0), (-0.75, -0.55), "#D55E00")
    a.text(-0.25, -1.05, "$x_W$ east", fontsize=8, color="#0072B2"); a.text(-1.28, 0.05, "$y_W$ north", fontsize=8, color="#009E73"); a.text(-0.7, -0.5, "$z_W$ up", fontsize=8, color="#D55E00"); a.text(-1.5, -1.25, "World frame ENU", fontsize=8, fontweight="bold")
    for dx in (-.55, .55): a.add_patch(Circle((.2 + dx, .15), .22, fc="#dde6ee", ec="#555", lw=.8)); a.add_patch(Circle((.2, .15 + dx), .22, fc="#dde6ee", ec="#555", lw=.8))
    a.plot([-.35, .75], [.15, .15], color="#333", lw=2.5); a.plot([.2, .2], [-.4, .7], color="#333", lw=2.5)
    ar(a, (.2, .15), (1.05, .15), "#0072B2", 1.4); ar(a, (.2, .15), (.2, .95), "#009E73", 1.4); a.text(1.07, .1, "$x_B$ front", fontsize=7.5, color="#0072B2"); a.text(.25, 1.0, "$y_B$ left", fontsize=7.5, color="#009E73"); a.text(1.55, -.45, "Body frame FLU, $z_B$ up\n$q$: Hamilton wxyz,\nbody$\\to$world;\n$\\omega$ in body frame", fontsize=7, ha="right", va="top")
    b = axs[1]; b.add_patch(Circle((0, .1), .26, fc="#dde6ee", ec="#555")); b.plot([-.9, .9], [.1, .1], color="#333", lw=3)
    for x in (-.9, .9): b.add_patch(Circle((x, .1), .2, fc="#cfe3f5", ec="#555", lw=.8)); ar(b, (x, .3), (x, .95), "#0072B2", 1.4)
    b.text(.0, 1.1, "rotor thrust $k_F\\Omega_i^2$ (along $z_B$), reaction torque $k_M\\Omega_i^2$", fontsize=7, ha="center", color="#0072B2")
    ar(b, (0, -.1), (0, -.9), "#555", 1.6); b.text(.07, -.8, "gravity $mg$", fontsize=7.5); ar(b, (-1.5, -.2), (-.4, -.2), "#D55E00", 1.6); b.text(-1.5, -.05, "wind $w$", fontsize=7.5, color="#D55E00")
    ar(b, (.45, .1), (1.4, .1), "#7B4EA3", 1.4, ls="--"); b.text(1.4, .17, "drag", fontsize=7.5, color="#7B4EA3", ha="right"); b.text(0, -1.22, "drag from $v-w$:  rotor drag $-k_{rd}\\Omega_iP_\\perp v_{rel}$,  body drag $-\\frac{1}{2}\\rho C_DA\\|v_{rel}\\|v_{rel}$", fontsize=6.8, ha="center")
    tag(axs[0], "a"); tag(axs[1], "b"); fig.tight_layout(); save(fig, "frames_forces")

def fig_timeline():
    """Figure 24: project timeline (phases and durations from the thesis plan; status from HANDOFF.md)."""
    ph = [("1  Literature, simulator setup", 2.5, "done"), ("2  Dynamics, dataset generation", 2.5, "done"), ("3  Physics calibration, baseline", 2, "done"), ("4  Pure neural world model", 2.5, "done"), ("5  Physics + residual model", 3, "done"),
          ("6  Latent memory, imagination", 3, "done"), ("7  Primary OOD wind experiments", 3, "ongoing"), ("8  Ablations, reliability analysis", 3, "planned"), ("9  Uncertainty / vision (optional)", 2.5, "planned"), ("10  Final analysis, writing", 3.5, "planned")]
    col = {"done": "#009E73", "ongoing": "#E69F00", "planned": "#b8c2cc"}; fig, ax = plt.subplots(figsize=(W2, 3.3)); t = 0
    for i, (n, d, s_) in enumerate(ph):
        y = len(ph) - 1 - i; ax.barh(y, d, left=t, color=col[s_], height=.62); ax.text(t + d + .15, y, f"{d:g} wk", va="center", fontsize=6.5, color="#555"); t += d
    ax.set_yticks(range(len(ph))); ax.set_yticklabels([n for n, _, _ in ph][::-1]); ax.set_xlabel("cumulative weeks from the start of the plan (durations from the proposal; calendar dates not fixed)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col[k], label=k) for k in col], loc="upper right"); ax.grid(axis="y", visible=False); ax.set_xlim(0, t + 3); save(fig, "timeline")


def fig_pipeline():
    """Figure 2: end-to-end methodology (data side in blue/teal, model side in green, evaluation in amber)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    steps = [("UAV dynamics\nmodel", "#0072B2"), ("Scenario\ndesign", "#0072B2"), ("Wind + dynamics\nrandomisation", "#E69F00"), ("Trajectory\ngeneration, control", "#0072B2"), ("Simulation\nrollouts", "#0072B2"),
             ("Sensor / data\nprocessing", "#1F8A8A"), ("Dataset\nvalidation", "#1F8A8A"), ("World-model\ntraining", "#009E73"), ("ID / OOD\nevaluation", "#E69F00"), ("Reliability\nanalysis", "#009E73")]
    fig, ax = plt.subplots(figsize=(W2, 2.2)); ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.6); ax.axis("off"); ax.grid(False)
    for i, (t, c) in enumerate(steps):
        r, k = divmod(i, 5); x = .1 + k * 2.06; y = 2.1 - r * 1.75
        ax.add_patch(FancyBboxPatch((x, y), 1.8, 1.15, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec="none")); ax.text(x + .9, y + .575, f"{i + 1}  " + t, ha="center", va="center", color="white", fontsize=6.4, fontweight="bold")
        if k < 4: ax.add_patch(FancyArrowPatch((x + 1.8, y + .575), (x + 2.06, y + .575), arrowstyle="-|>", mutation_scale=8, lw=1, color="#222"))
    ax.text(5.2, .12, "data side (1-7): build and validate the evidence      model side (8-10): the controlled comparison", ha="center", fontsize=7, color="#555", style="italic")
    save(fig, "pipeline")

def fig_heat_attitude():
    """Attitude RMSE (deg) at 1 s on all 13 roles, known-vehicle runs."""
    fig_heat("att_rmse_deg", "heatmap_attitude", "attitude RMSE at 1 s [deg]")



# ======================================================================= data-quality figures (house style, same content as the quality report)
QUALITY_JSON = ROOT / "outputs/reports/physiworld_pilot_v7_quality/quality.json"
GCOL = {"train": "#7f8c99", "ID (val/test)": "#0072B2", "OOD": "#E69F00", "physics check": "#009E73", "OOD magnitude": "#D55E00", "OOD other": "#CC79A7"}

def _quality():
    if not QUALITY_JSON.exists():
        raise SystemExit(f"{QUALITY_JSON} missing: run `python -m physiworld_uav.data.statistical_validator --dataset <v7>` first")
    return json.load(open(QUALITY_JSON))["metrics"]

def fig_q_correlation():
    """Data quality: action-wind correlations (closed / open loop), domain-factor correlations and the residual-vs-wind identifiability curve."""
    m = _quality()["correlation"]; fig, axs = plt.subplots(2, 2, figsize=(W2, 5.8), constrained_layout=True); aw = m["action_wind"]
    for a, dt, s_ in zip(axs[0], ("closed_loop", "open_loop"), "ab"):
        im = a.imshow(np.array(aw[dt]["matrix"]), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto"); a.grid(False)
        a.set_xticks([0, 1, 2]); a.set_xticklabels(["$w_x$", "$w_y$", "$w_z$"]); a.set_yticks(range(4)); a.set_yticklabels([f"rotor {i}" for i in range(4)])
        a.set_title(f"Rotor speed vs wind, {dt.replace('_', ' ')}"); tag(a, s_)
        for i in range(4):
            for j in range(3): a.text(j, i, f"{aw[dt]['matrix'][i][j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=list(axs[0]), fraction=.03, pad=.02).ax.tick_params(labelsize=6)
    d = m["domain"]; a = axs[1, 0]; im = a.imshow(np.array(d["matrix"]), vmin=-.3, vmax=.3, cmap="RdBu_r"); a.grid(False)
    nm = [n.replace("_scale", "").replace("inertia_shape_", "J").replace("motor_time_constant", "tau_m").replace("km_kf_ratio", "km/kf") for n in d["names"]]
    a.set_xticks(range(len(nm))); a.set_xticklabels(nm, rotation=90, fontsize=6); a.set_yticks(range(len(nm))); a.set_yticklabels(nm, fontsize=6); a.set_title("Correlation between domain factors"); tag(a, "c")
    fig.colorbar(im, ax=a, fraction=.046, pad=.03).ax.tick_params(labelsize=6)
    a = axs[1, 1]
    for dt, c in (("closed_loop", "#0072B2"), ("open_loop", "#E69F00")):
        v = m["residual_vs_wind"][dt]; a.plot([(b["lo"] + b["hi"]) / 2 for b in v["bins"]], [b["mean_residual_ms2"] for b in v["bins"]], "-o", ms=3.5, color=c, label=f"{dt.replace('_', ' ')} ($\\rho$ = {v['corr']:.2f})")
    a.set_xlabel("wind speed [m/s]"); a.set_ylabel("mean $\\|\\Delta v\\|$ residual [m/s$^2$]"); a.set_title("Nominal-physics residual vs wind"); a.legend(); tag(a, "d")
    save(fig, "q_correlation")

def fig_q_coverage():
    """Data quality: wind-speed coverage, domain-factor coverage, scenario x role matrix, initial heights."""
    cov = _quality()["coverage"]; fig, axs = plt.subplots(2, 2, figsize=(W2, 6.2))
    a = axs[0, 0]; occ = cov["wind"]["speed_occupancy"][1]; a.bar(np.arange(len(occ)) * .5 + .25, occ, width=.45, color="#0072B2"); a.set_xlabel("mean wind speed [m/s]"); a.set_ylabel("episodes"); a.set_title("Episodes per mean-wind bin"); tag(a, "a")
    d1 = cov["domain_1d"]; a = axs[0, 1]; M1 = np.array([v["counts"] for v in d1.values()]); im = a.imshow(M1, cmap="Blues", aspect="auto"); a.grid(False)
    a.set_yticks(range(len(d1))); a.set_yticklabels([k.replace("_scale", "").replace("inertia_shape_", "J") for k in d1], fontsize=6.5); a.set_xlabel("bin across the ID range (low to high)"); a.set_title("ID vehicles per factor bin")
    fig.colorbar(im, ax=a, fraction=.046, pad=.03).ax.tick_params(labelsize=6); tag(a, "b")
    cells = cov["scenario_role"]; scn = sorted({k.split("|")[0] for k in cells}); roles = sorted({k.split("|")[1] for k in cells})
    a = axs[1, 0]; M = np.array([[cells.get(f"{s_}|{r}", 0) for r in roles] for s_ in scn]); im = a.imshow(M, cmap="Blues", aspect="auto"); a.grid(False)
    a.set_xticks(range(len(roles))); a.set_xticklabels([r.replace("_", " ").replace("ood", "OOD") for r in roles], rotation=90, fontsize=5.5); a.set_yticks(range(len(scn))); a.set_yticklabels([x.replace("_", " ") for x in scn], fontsize=5.5)
    a.set_title("Baseline episodes: scenario and role"); fig.colorbar(im, ax=a, fraction=.046, pad=.03).ax.tick_params(labelsize=6); tag(a, "c")
    a = axs[1, 1]; gp = cov["groups_per_role"]; ks = sorted(gp, key=gp.get); a.barh([k.replace("_", " ").replace("ood", "OOD") for k in ks], [gp[k] for k in ks], color="#009E73", height=.7); a.set_xlabel("independent groups"); a.set_title("Independent groups per role"); a.grid(axis="y", visible=False); tag(a, "d")
    fig.tight_layout(h_pad=1.2, w_pad=1.5); save(fig, "q_coverage")

def fig_q_channels():
    """Data quality: distributions of six state / IMU / action channels by split family (log density)."""
    Q = _cache("quality.npz"); ch = [("speed", "speed [m/s]"), ("z", "altitude [m]"), ("tilt", "tilt [deg]"), ("omega", "$\\|\\omega\\|$ [rad/s]"), ("fz", "IMU $f_z$ [m/s$^2$]"), ("rotor0", "rotor 0 [rad/s]")]
    fig, axs = plt.subplots(3, 2, figsize=(W2, 6.6))
    for a, (k, ttl), s_ in zip(axs.ravel(), ch, "abcdef"):
        e = Q[f"edges|{k}"]; mid = (e[1:] + e[:-1]) / 2
        for g in ("train", "ID (val/test)", "OOD", "physics check"):
            key = f"ch|{k}|{g}"
            if key in Q.files and Q[key].sum() > 0: v = Q[key] / Q[key].sum() / np.diff(e); a.step(mid, np.where(v > 0, v, np.nan), where="mid", color=GCOL[g], lw=1.2, label=g)
        a.set_yscale("log"); a.set_xlabel(ttl); a.set_ylabel("density"); tag(a, s_)
    axs[0, 0].legend(fontsize=6.5); fig.tight_layout(h_pad=1.0, w_pad=1.5); save(fig, "q_channels")

def fig_q_physical():
    """Data quality: physical-consistency residuals of the ground truth with the validator limits."""
    Q = _cache("quality.npz"); ch = [("pos_vel", "$\\Delta p/\\Delta t-\\bar v$ [m/s]", .10), ("acc_imu", "$\\Delta v/\\Delta t-(Rf+g)$ [m/s$^2$]", None), ("gyro", "IMU gyro $-\\omega_B$ [rad/s]", .05), ("att_rate", "angle $-\\|\\omega\\|\\Delta t$ [rad]", .02)]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.6))
    for a, (k, ttl, lim), s_ in zip(axs.ravel(), ch, "abcd"):
        e = Q[f"edges|{k}"]; v = Q[f"res|{k}|all"]; a.bar((e[1:] + e[:-1]) / 2, np.where(v > 0, v, np.nan), width=np.diff(e), color="#0072B2"); a.set_yscale("log"); a.set_xlabel(ttl); a.set_ylabel("samples (model-rate steps)"); tag(a, s_)
        if lim and lim < e[-1]: a.axvline(lim, color="#D55E00", ls="--", lw=.9); a.text(lim, a.get_ylim()[1] * .5, " limit (p99)", color="#D55E00", fontsize=6.5)
    fig.tight_layout(h_pad=1.0, w_pad=1.5); save(fig, "q_physical")

def fig_q_wind():
    """Data quality: wind actually stored: episode peak against mean speed, per-time-step speed density and exceedance, by role group."""
    Q = _cache("quality.npz"); fig, axs = plt.subplots(2, 2, figsize=(W2, 5.4)); g_of = lambda r: "train" if r == "train" else ("ID (val/test)" if r in ("val_id", "test_id") else ("OOD magnitude" if "magnitude" in r else ("OOD other" if r.startswith(("val_ood", "test_ood")) else None)))
    a = axs[0, 0]
    for g in ("train", "ID (val/test)", "OOD other", "OOD magnitude"):
        pts = [(m_["wind"]["stats"]["mean_horizontal_speed"], m_["wind"]["stats"]["peak_speed"]) for m_ in man if g_of(m_["split_role"]) == g]
        if pts: a.scatter(*zip(*pts), s=3, color=GCOL[g], alpha=.5, lw=0, label=g)
    a.set_xlabel("episode mean wind speed [m/s]"); a.set_ylabel("episode peak speed [m/s]"); a.legend(fontsize=6.3, markerscale=2.5); a.set_title("One dot per episode"); tag(a, "a")
    a = axs[0, 1]; b = np.linspace(0, 10, 41)
    for g in ("train", "ID (val/test)", "OOD other", "OOD magnitude"):
        pk = [m_["wind"]["stats"]["peak_speed"] for m_ in man if g_of(m_["split_role"]) == g]; a.hist(pk, b, histtype="step", lw=1.3, color=GCOL[g], density=True)
    a.set_yscale("log"); a.set_xlabel("episode peak speed [m/s]"); a.set_ylabel("density"); a.set_title("Peak wind speed by role"); tag(a, "b")
    e = Q["windstep|edges"]; mid = (e[1:] + e[:-1]) / 2; a = axs[1, 0]; c = axs[1, 1]
    for g in ("train", "ID (val/test)", "OOD other", "OOD magnitude"):
        v = Q[f"windstep|{g}"]; fr = v / v.sum(); a.step(mid, np.where(fr > 0, fr, np.nan), where="mid", color=GCOL[g], lw=1.3, label=g); c.plot(e[:-1], 1 - np.cumsum(np.r_[0, fr[:-1]]), color=GCOL[g], lw=1.3)
    a.set_yscale("log"); a.set_xlabel("wind speed per time step [m/s]"); a.set_ylabel("fraction of time steps"); a.set_title("Wind speed per time step"); a.legend(fontsize=6.3); tag(a, "c")
    c.set_yscale("log"); c.set_ylim(1e-4, 1.2); c.set_xlabel("speed [m/s]"); c.set_ylabel("fraction of steps above this speed"); c.set_title("Exceedance"); tag(c, "d")
    fig.tight_layout(h_pad=1.2, w_pad=1.5); save(fig, "q_wind")

def fig_q_initial():
    """Data quality: initial conditions and episode length (early-ended episodes)."""
    idx = json.load(open(DS / "index.json"))["episodes"]; fig, axs = plt.subplots(2, 2, figsize=(W2, 4.8))
    a = axs[0, 0]; xy = np.array([m_["initial_conditions"]["position"][:2] for m_ in man]); a.scatter(xy[:, 0], xy[:, 1], s=2, color="#0072B2", alpha=.5, lw=0); a.set_aspect("equal"); a.set_xlabel("initial x [m]"); a.set_ylabel("initial y [m]"); a.set_title("Initial horizontal position"); tag(a, "a")
    a = axs[0, 1]; a.hist([m_["initial_conditions"]["position"][2] for m_ in man], np.linspace(1.4, 2.6, 25), color="#009E73"); a.set_xlabel("initial altitude [m]"); a.set_ylabel("episodes"); a.set_title("Initial altitude"); tag(a, "b")
    a = axs[1, 0]; a.hist([m_["initial_conditions"]["yaw_deg"] for m_ in man], np.linspace(-180, 180, 25), color="#CC79A7"); a.set_xlabel("initial yaw [deg]"); a.set_ylabel("episodes"); a.set_title("Initial yaw (8 sectors all covered)"); tag(a, "c")
    a = axs[1, 1]; L = np.array([e["n_steps"] * .005 for e in idx]); a.hist(L, np.linspace(0, 10.2, 35), color="#E69F00"); a.set_yscale("log"); a.set_xlabel("episode length [s]"); a.set_ylabel("episodes"); a.set_title("Episode length (10 s planned; early ends)"); tag(a, "d")
    fig.tight_layout(h_pad=1.2, w_pad=1.5); save(fig, "q_initial")


FIGURES = {n[4:]: f for n, f in list(globals().items()) if n.startswith("fig_")}

def main(argv: list[str] | None = None) -> None:
    global OUT, FORMATS, CACHE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", help="figure names (default: all); --list shows them")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--page-size", dest="page_size", action="store_true", help="scale each figure to the width it has on the page (PAGE_WIDTH); text gets crowded below about 0.7 of the page width"); ap.add_argument("--dataset", type=Path, default=DS)
    ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--formats", nargs="+", default=FORMATS, choices=["pdf", "png", "jpg", "svg"]); ap.add_argument("--copy-to", type=Path, default=None, help="also copy the files here (e.g. report/figures_new)")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(sorted(FIGURES))); return
    OUT, FORMATS, CACHE = a.out, a.formats, a.cache; load_manifest(a.dataset)
    for n in (a.only or list(FIGURES)):
        _SCALE[0] = PAGE_WIDTH.get(n, 1.0) if a.page_size else 1.0
        try:
            FIGURES[n]()
        except SystemExit as e:
            print("skipped", n, "-", e)
    if a.copy_to:
        a.copy_to.mkdir(parents=True, exist_ok=True)
        for f in OUT.iterdir():
            if f.suffix.lstrip(".") in FORMATS: shutil.copy(f, a.copy_to / f.name)


if __name__ == "__main__":
    main()
