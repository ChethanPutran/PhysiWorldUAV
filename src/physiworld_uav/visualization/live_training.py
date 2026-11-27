"""Live view of a training sweep: progress, timing, CPU, GPU and memory per run (stdlib only). Run on the HOST.

    python3 -m physiworld_uav.visualization.live_training --sweep outputs/sweeps/fix_v1            # refresh every 5 s, Ctrl-C to stop
    python3 -m physiworld_uav.visualization.live_training --sweep outputs/sweeps/fix_v1 --once      # print once (also for logs / pasting)
    python3 -m physiworld_uav.visualization.live_training --sweep outputs/sweeps/fix_v1 --serve --port 60194   # live page with charts (writer + web server in one process)

In-place refresh needs a real terminal (ssh, tmux, a normal shell): the screen is cleared with ANSI codes. Where the output is not
a terminal (Claude Code's `!` panel, a pipe, a file) every refresh would be appended, so the script then refuses to loop and tells
you to use --html (open the file in a browser or the IDE; it reloads every `interval` seconds) or `watch -n 5 ... --once`.

Per run (from outputs/experiments/<run>/metrics.jsonl, the process in /proc and nvidia-smi):
  state, GPU, epoch, curriculum horizon, elapsed, training seconds per epoch, per training episode and episodes/s,
  validation seconds, windows/s, validation position RMSE at the validation horizon, best so far (epoch),
  CPU % of one core, resident memory, GPU memory of the process. Above the table: per-GPU utilisation and memory, host load,
  job counts. Training time per epoch / per episode excludes validation (validation is shown separately).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TICKS = os.sysconf("SC_CLK_TCK")


def procs() -> dict[str, dict]:
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = [x.decode(errors="replace") for x in (p / "cmdline").read_bytes().split(b"\0")]
        except OSError:
            continue
        if not any("physiworld_uav.training.run" in x for x in args) or (args and args[0].endswith(("bash", "sh"))):
            continue
        d = dict(zip(args, args[1:]))
        if "--name" in d:
            out[d["--name"]] = {"pid": int(p.name)}
    return out


def cpu_ticks(pid: int):
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return float(f[11]) + float(f[12])
    except (OSError, IndexError):
        return None


def rss_mb(pid: int):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def smi(query: str, kind: str):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return [[c.strip() for c in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def last_metrics(path: Path):
    try:
        lines = path.read_text().strip().splitlines()
    except OSError:
        return None, 0
    return (json.loads(lines[-1]) if lines else None), len(lines)


RUN_RE = re.compile(r"^(?:(?:fx|st)_(v0\d)|kv_short|kv_v07)_(b2|b3|b4_rw|b4|b5a|b5b|b6b_future|b6b_current)(?:_(.+?))?_s(\d+)$")


def run_series(exp_root: Path, st: dict, refs=("v06_b3",)) -> list[dict]:
    """Per-epoch curves of every run of the sweep (and reference runs), for the charts of the live page."""
    out = []
    names = list(st) + [r for r in refs if r not in st]
    for name in names:
        m = RUN_RE.match(name)
        if m:
            ds, model, seed = m.group(1) or "v07", m.group(2), int(m.group(4))
            var = m.group(3) or ("kv" if m.group(1) is None else "")          # kv_short_* = known-vehicle short runs
        elif name in refs:                                   # an earlier run used as a reference: <ds>_<model>
            ds, model = name.split("_", 1); var, seed = "ref", 0
        else:
            continue
        f = exp_root / name / "metrics.jsonl"
        if not f.exists():
            continue
        ep = []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                                   # a half-written last line
            v = r.get("val", {}).get("pos_rmse_m", {})
            ep.append({"e": r["epoch"], "h": r["horizon"], "roll": r["train"].get("roll"), "total": r["train"].get("total"),
                       "v1": v.get("1"), "vk": r.get("metric"), "best": r.get("best_global"), "t": r.get("epoch_s"), "vt": r.get("val_s")})
        out.append({"name": name, "ds": ds, "model": model, "variant": var, "seed": seed, "ref": name in refs and not m,
                    "state": st.get(name, {}).get("state", "done"), "epochs": ep})
    return out


def hms(s):
    s = int(s or 0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(sweep: Path, exp_root: Path, prev: dict) -> tuple[str, dict]:
    st = json.loads((sweep / "status.json").read_text()) if (sweep / "status.json").exists() else {}
    pr, now = procs(), time.time()
    gpu_of_pid = {int(r[0]): float(r[1]) for r in smi("pid,used_memory", "compute-apps") if r and r[0].isdigit()}
    gpus = smi("index,utilization.gpu,memory.used,memory.total", "gpu")
    cnt = {k: sum(v["state"] == k for v in st.values()) for k in ("running", "queued", "done", "failed")}
    head = [f"sweep {sweep.name}   {time.strftime('%H:%M:%S')}   host load {os.getloadavg()[0]:.0f}   jobs: " +
            "  ".join(f"{k} {v}" for k, v in cnt.items()) + f"   of {len(st)}"]
    head.append("GPU  " + "   ".join(f"[{g[0]}] util {g[1]:>3}%  mem {float(g[2]) / 1024:5.1f}/{float(g[3]) / 1024:.0f} GB" for g in gpus))
    cols = ["run", "state", "gpu", "ep", "h", "elapsed", "train s/ep", "s/episode", "ep/s", "val s", "windows/s", "val k RMSE", "best (ep)", "CPU %", "RSS MB", "GPU MB"]
    rows, new_prev = [], {}
    order = {"running": 0, "failed": 1, "queued": 3, "done": 2}
    for name, j in sorted(st.items(), key=lambda kv: (order.get(kv[1]["state"], 9), kv[0])):
        m, n = last_metrics(exp_root / name / "metrics.jsonl")
        p = pr.get(name)
        cpu = None
        if p:
            tk = cpu_ticks(p["pid"])
            if tk is not None:
                new_prev[name] = (now, tk)
                if name in prev and now > prev[name][0]:
                    cpu = 100.0 * (tk - prev[name][1]) / TICKS / (now - prev[name][0])
        if j["state"] == "queued":
            rows.append([name, "queued"] + [""] * 14); continue
        elapsed = (now - j["t0"]) if j["state"] == "running" and "t0" in j else j.get("wall_s")
        rows.append([name, j["state"], j.get("gpu", ""), n or "", m["horizon"] if m else "", hms(elapsed),
                     f"{m['epoch_s']:.1f}" if m else "", f"{m['s_per_episode']:.2f}" if m and "s_per_episode" in m else "",
                     f"{m['episodes_per_s']:.1f}" if m and "episodes_per_s" in m else "", f"{m['val_s']:.1f}" if m and "val_s" in m else "",
                     f"{m['windows_per_s']:.0f}" if m else "", f"{m['metric']:.4f}" if m else "",
                     (f"{m['best_global']:.4f}" if m and "best_global" in m else (f"{m['best']:.4f}" if m else "")) if m else "",
                     f"{cpu:.0f}" if cpu is not None else "", f"{rss_mb(p['pid']):.0f}" if p and rss_mb(p["pid"]) else "",
                     f"{gpu_of_pid.get(p['pid'], 0):.0f}" if p and p["pid"] in gpu_of_pid else ""])
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    line = lambda r: "  ".join(str(x).rjust(w[i]) if i else str(x).ljust(w[i]) for i, x in enumerate(r))  # noqa: E731
    return "\n".join(head + ["", line(cols), "  ".join("-" * x for x in w)] + [line(r) for r in rows]), new_prev


def structured(sweep: Path, exp_root: Path, prev: dict) -> tuple[dict, dict]:
    """Everything the dashboard shows, as data: job counts, per-job state and resources, GPUs, host, elapsed time and an estimated time left."""
    st = json.loads((sweep / "status.json").read_text()) if (sweep / "status.json").exists() else {}
    pr, now = procs(), time.time()
    gpu_mem = {int(r[0]): float(r[1]) for r in smi("pid,used_memory", "compute-apps") if r and r[0].isdigit()}
    gpus = [{"index": int(g[0]), "util": float(g[1]), "mem_used_mb": float(g[2]), "mem_total_mb": float(g[3])} for g in smi("index,utilization.gpu,memory.used,memory.total", "gpu")]
    for g, extra in zip(gpus, smi("index,temperature.gpu,power.draw", "gpu")):
        g["temp_c"] = float(extra[1]) if extra[1].replace(".", "").isdigit() else None
        g["power_w"] = float(extra[2]) if extra[2].replace(".", "").isdigit() else None
    jobs, new_prev = [], {}
    for name, j in st.items():
        m, n = last_metrics(exp_root / name / "metrics.jsonl")
        p = pr.get(name); cpu = None
        if p:
            tk = cpu_ticks(p["pid"])
            if tk is not None:
                new_prev[name] = (now, tk)
                if name in prev and now > prev[name][0]:
                    cpu = 100.0 * (tk - prev[name][1]) / TICKS / (now - prev[name][0])
        elapsed = (now - j["t0"]) if j["state"] == "running" and "t0" in j else j.get("wall_s")
        jobs.append({"name": name, "state": j["state"], "gpu": j.get("gpu"), "model": j.get("model"), "epoch": n, "horizon": m["horizon"] if m else None,
                     "elapsed_s": elapsed, "epoch_s": m.get("epoch_s") if m else None, "windows_per_s": m.get("windows_per_s") if m else None,
                     "val": m.get("metric") if m else None, "best": (m.get("best_global") if m and m.get("best_global") is not None else (m.get("best") if m else None)),
                     "cpu_pct": cpu, "rss_mb": rss_mb(p["pid"]) if p else None, "gpu_mb": gpu_mem.get(p["pid"]) if p else None})
    cnt = {k: sum(v["state"] == k for v in st.values()) for k in ("running", "queued", "done", "failed")}
    walls = [v["wall_s"] for v in st.values() if v["state"] == "done" and v.get("wall_s")]
    run_el = [j["elapsed_s"] for j in jobs if j["state"] == "running" and j["elapsed_s"] is not None]
    eta = None
    if walls:
        mw = sum(walls) / len(walls); slots = max(cnt["running"], 1)
        eta = cnt["queued"] * mw / slots + (max(0.0, mw - sum(run_el) / len(run_el)) if run_el else 0.0)
    starts = [v["t0"] for v in st.values() if "t0" in v]
    ends = [v["t0"] + v["wall_s"] for v in st.values() if v.get("wall_s") and "t0" in v]
    elapsed_all = ((now if cnt["running"] or cnt["queued"] else max(ends, default=now)) - min(starts)) if starts else None
    return ({"sweep": sweep.name, "updated": time.strftime("%H:%M:%S"), "counts": cnt, "total": len(st), "elapsed_s": elapsed_all, "eta_s": eta,
             "host": {"cores": os.cpu_count(), "load1": os.getloadavg()[0]}, "gpus": gpus, "jobs": jobs,
             "ours": {"cpu_cores": sum(j["cpu_pct"] or 0 for j in jobs) / 100, "rss_gb": sum(j["rss_mb"] or 0 for j in jobs) / 1024,
                      "gpu_gb": sum(j["gpu_mb"] or 0 for j in jobs) / 1024}}, new_prev)


PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training live</title>
<style>
:root{--bg:#fff;--fg:#1c1c1c;--mut:#666;--grid:#e6e6e6;--axis:#999;--card:#f6f6f6}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#e6e6e6;--mut:#9a9a9a;--grid:#2a2a2a;--axis:#666;--card:#1b1b1b}}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif;padding:16px}
h1{font-size:16px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:10px}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:8px 0 6px}
select{background:var(--card);color:var(--fg);border:1px solid var(--axis);padding:3px 6px;border-radius:4px}
.chip{display:inline-flex;align-items:center;gap:6px;cursor:pointer;padding:2px 8px;border-radius:10px;background:var(--card);margin:0 6px 4px 0;user-select:none}
.chip i{display:inline-block;width:14px;height:3px;border-radius:2px}.chip.off{opacity:.35}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}
.card{background:var(--card);border-radius:8px;padding:8px 10px}.card h2{font-size:13px;margin:0 0 2px}
svg{width:100%;height:240px;display:block}svg text{fill:var(--mut);font-size:10px}
pre{background:var(--card);padding:10px;border-radius:8px;overflow:auto;font:12px/1.35 ui-monospace,monospace;margin-top:12px}
#tip{position:fixed;pointer-events:none;background:var(--bg);border:1px solid var(--axis);padding:4px 8px;border-radius:4px;display:none;font-size:12px;z-index:9}

/* ---- dashboard (default view): one screen on a desktop, one scrolling column on a phone; same design as the generation report page */
body.dash,body.dash *{box-sizing:border-box}
body.dash{margin:0;padding:0;height:100vh;overflow:hidden;font-size:clamp(10px,1.25vh,15px);line-height:1.35}
#view-dash{display:none}body.dash #view-dash{display:grid;height:100vh;padding:.8vh .8vw;gap:.8vh .6vw;grid-template-columns:repeat(12,minmax(0,1fr));grid-template-rows:auto auto auto minmax(0,1.1fr) minmax(0,1fr)}
body.dash #view-curves{display:none}body.dash #tip{display:none}
#tabs{position:fixed;right:.8vw;top:.6vh;z-index:30;display:flex;gap:6px}
#tabs button{font:inherit;font-size:max(10px,1.1vh);padding:.3vh .9vh;border:1px solid var(--axis);border-radius:6px;background:var(--card);color:var(--mut);cursor:pointer}
#tabs button.on{background:#0072B2;color:#fff;border-color:#0072B2}
.d-panel{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:.7vh .7vw;min-width:0;min-height:0;overflow:hidden;display:flex;flex-direction:column}
.d-panel h3{font-size:1.05em;font-weight:600;margin:0 0 .4vh;display:flex;justify-content:space-between;align-items:baseline}.d-panel h3 small{font-weight:400;color:var(--mut)}
#d-head{grid-column:1/13;display:flex;align-items:baseline;gap:1vw;padding-right:11vw}#d-head h1{font-size:1.7em;margin:0}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.85em;background:var(--grid);color:var(--mut)}
#d-bar{grid-column:1/13;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:.6vw;align-items:center}
#d-bar .bar{height:1.7vh;background:var(--grid);border-radius:1vh;overflow:hidden;display:flex}#d-bar .bar i{display:block;height:100%}
#d-tiles{grid-column:1/13;display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:.5vw}
#d-tiles .t{background:var(--card);border:1px solid var(--grid);border-radius:7px;padding:.4vh .6vw;align-self:start}#d-tiles .t b{display:block;font-size:1.7em;line-height:1.1;font-variant-numeric:tabular-nums}#d-tiles .t span{color:var(--mut);font-size:.85em}
#d-runs{grid-column:1/9}#d-val{grid-column:9/13}#d-gpu{grid-column:1/5}#d-loss{grid-column:5/9}#d-best{grid-column:9/13}
.d-panel table{width:100%;border-collapse:collapse;font-size:.92em;font-variant-numeric:tabular-nums}.d-panel th,.d-panel td{padding:.3vh .5vw;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap}.d-panel th{color:var(--mut);font-weight:500}
.mini{display:inline-block;width:5.5vw;height:1.1vh;background:var(--grid);border-radius:1vh;overflow:hidden;vertical-align:middle;margin-right:.3vw}.mini i{display:block;height:100%;background:#0072B2}
.ok{color:#009E73}.bad{color:#D55E00}.hostk{display:flex;gap:.5vw;flex-wrap:wrap;margin-top:.4vh}.hostk div{background:var(--bg);border:1px solid var(--grid);border-radius:6px;padding:.2vh .5vw}
.d-panel svg.dc{flex:1;min-height:0;width:100%;height:100%}.d-leg{display:flex;flex-wrap:wrap;gap:.2vh .8vw;font-size:.85em;color:var(--mut)}.d-leg i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:4px;vertical-align:middle}
@media (max-width:900px),(max-aspect-ratio:1/1){
  body.dash{height:auto;min-height:100vh;overflow-y:auto;overflow-x:hidden;font-size:14px}
  body.dash #view-dash{height:auto;display:flex;flex-direction:column;gap:10px;padding:10px 10px 28px}
  #tabs{position:sticky;top:0;right:auto;justify-content:flex-end;padding:6px 10px;background:var(--bg);border-bottom:1px solid var(--grid)}#tabs button{font-size:14px;padding:7px 12px}
  #d-head{padding-right:0;flex-wrap:wrap}#d-head h1{font-size:20px}#d-bar{grid-template-columns:1fr}#d-bar .bar{height:14px}
  #d-tiles{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}#d-tiles .t b{font-size:22px}#d-tiles .t span{font-size:12px}
  .d-panel{padding:10px;overflow:visible;display:block}.d-panel>div{overflow-x:auto}.d-panel table{font-size:12.5px}.d-panel th,.d-panel td{padding:6px 8px}
  .mini{width:60px;height:9px}.d-panel svg.dc{height:210px}.hostk div{padding:4px 8px;font-size:13px}
  #d-rt th:nth-child(3),#d-rt td:nth-child(3),#d-rt th:nth-child(5),#d-rt td:nth-child(5),#d-rt th:nth-child(9),#d-rt td:nth-child(9),#d-rt th:nth-child(11),#d-rt td:nth-child(11){display:none}
}
@media (max-width:480px){#d-rt th:nth-child(2),#d-rt td:nth-child(2){display:none}.mini{width:40px}}
</style>
<div id="tabs"><button id="tab-dash" class="on">Dashboard</button><button id="tab-curves">Curves</button></div>
<div id="view-dash">
  <div id="d-head"><h1 id="d-title">Training</h1><span class="badge" id="d-state">waiting</span><span id="d-sub" style="color:var(--mut)"></span></div>
  <div id="d-bar"><div class="bar" id="d-barfill"></div><b id="d-pct"></b></div>
  <div id="d-tiles"></div>
  <div class="d-panel" id="d-runs"><h3>Runs <small id="d-runs-note"></small></h3><div style="min-height:0;overflow:hidden"><table id="d-rt"></table></div></div>
  <div class="d-panel" id="d-val"><h3>Validation error (position RMSE) <small id="d-val-note"></small></h3><svg class="dc" id="d-valsvg"></svg><div class="d-leg" id="d-leg"></div></div>
  <div class="d-panel" id="d-gpu"><h3>GPUs and host</h3><table id="d-gt"></table><div class="hostk" id="d-host"></div></div>
  <div class="d-panel" id="d-loss"><h3>Training rollout loss <small>running runs</small></h3><svg class="dc" id="d-losssvg"></svg></div>
  <div class="d-panel" id="d-best"><h3>Best validation so far <small>lowest first</small></h3><div style="min-height:0;overflow:hidden"><table id="d-bt"></table></div></div>
</div>
<div id="view-curves">
<h1 id="title">Training live</h1><div class="sub" id="sub"></div>
<div class="ctl"><label>Model <select id="model"></select></label><label>Dataset <select id="ds"></select></label>
<label>Runs <select id="which"><option value="all">all of the selection</option><option value="running">running only</option></select></label>
<label>Y scale <select id="scale"><option value="log">log</option><option value="lin">linear</option></select></label>
<span>dashed = seed 1, dotted = seed 2, grey = reference run (earlier)</span></div>
<div id="chips"></div>
<div class="grid" id="charts"></div>
<pre id="tbl"></pre></div><div id="tip"></div>
<script>
const COL=["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#888888"];
const CH=[{k:"roll",t:"Training rollout loss (weighted, per epoch)"},{k:"vk",t:"Validation position RMSE at the validation horizon [m]"},
          {k:"v1",t:"Validation position RMSE, one step (k=1) [m]"},{k:"t",t:"Training seconds per epoch"}];
let DATA=null,off=new Set();const $=id=>document.getElementById(String(id).replace(/^#/,""));
function nice(a,b,n){const r=(b-a)||1,s=Math.pow(10,Math.floor(Math.log10(r/n))),e=r/n/s,m=e>5?10:e>2?5:e>1?2:1,st=m*s;const o=[];for(let v=Math.ceil(a/st)*st;v<=b+1e-12;v+=st)o.push(v);return o}
function fmt(v){return v==null?"":(Math.abs(v)>=1000||(v!==0&&Math.abs(v)<0.01))?v.toExponential(1):(+v.toPrecision(3)).toString()}
function groups(runs){const g=[];runs.forEach(r=>{const k=r.ref?"reference":(r.variant||"base")+" ("+r.ds+")";if(!g.includes(k))g.push(k)});return g}
function colorOf(g,gs){return g==="reference"?"#888888":COL[gs.filter(x=>x!=="reference").indexOf(g)%COL.length]}
function draw(card,ch,runs,gs,log){
  const W=card.clientWidth-20,H=240,m={l:52,r:8,t:6,b:22};const lines=[];
  runs.forEach(r=>{const g=r.ref?"reference":(r.variant||"base")+" ("+r.ds+")";if(off.has(g))return;
    const pts=r.epochs.filter(e=>e[ch.k]!=null&&(!log||ch.k==="t"&&false||e[ch.k]>0)).map(e=>[e.e,e[ch.k],e.h]);
    if(pts.length)lines.push({r,g,pts,c:colorOf(g,gs),d:r.seed===1?"6 3":r.seed===2?"2 3":""})});
  const all=lines.flatMap(l=>l.pts);let svg=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  if(!all.length){card.querySelector(".plot").innerHTML=svg+`<text x="${m.l}" y="40">no data yet</text></svg>`;return}
  const L=ch.k!=="t"&&log,tr=v=>L?Math.log10(v):v;
  let x0=1,x1=Math.max(...all.map(p=>p[0])),y0=Math.min(...all.map(p=>tr(p[1]))),y1=Math.max(...all.map(p=>tr(p[1])));
  if(!L)y0=Math.min(0,y0);if(y1===y0)y1=y0+1;const px=x=>m.l+(x-x0)/Math.max(x1-x0,1)*(W-m.l-m.r),py=y=>H-m.b-(tr(y)-y0)/(y1-y0)*(H-m.t-m.b);
  nice(y0,y1,4).forEach(t=>{const y=H-m.b-(t-y0)/(y1-y0)*(H-m.t-m.b);svg+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" stroke="var(--grid)"/><text x="${m.l-4}" y="${y+3}" text-anchor="end">${fmt(L?Math.pow(10,t):t)}</text>`});
  nice(x0,x1,6).forEach(t=>{svg+=`<text x="${px(t)}" y="${H-6}" text-anchor="middle">${t}</text>`});
  svg+=`<line x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}" stroke="var(--axis)"/><line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H-m.b}" stroke="var(--axis)"/>`;
  lines.forEach(l=>{svg+=`<polyline fill="none" stroke="${l.c}" stroke-width="2" ${l.d?`stroke-dasharray="${l.d}"`:""} points="${l.pts.map(p=>px(p[0]).toFixed(1)+","+py(p[1]).toFixed(1)).join(" ")}"/>`;
    const q=l.pts[l.pts.length-1];if(l.r.state==="running")svg+=`<circle cx="${px(q[0])}" cy="${py(q[1])}" r="3.5" fill="${l.c}" stroke="var(--bg)" stroke-width="2"/>`});
  const box=card.querySelector(".plot");box.innerHTML=svg+"</svg>";
  box.onmousemove=ev=>{const b=box.getBoundingClientRect(),mx=(ev.clientX-b.left)/b.width*W,my=(ev.clientY-b.top)/b.height*H;let best=null,bd=1e9;
    lines.forEach(l=>l.pts.forEach(p=>{const d=Math.hypot(px(p[0])-mx,py(p[1])-my);if(d<bd){bd=d;best={l,p}}}));const tip=$("tip");
    if(best&&bd<30){tip.style.display="block";tip.style.left=ev.clientX+12+"px";tip.style.top=ev.clientY+12+"px";tip.innerHTML=`<b>${best.l.r.name}</b><br>epoch ${best.p[0]} · horizon ${best.p[2]}<br>${ch.k==="t"?"":""}${fmt(best.p[1])}`}else tip.style.display="none"};
  box.onmouseleave=()=>$("tip").style.display="none"}
function render(){
  if(!DATA)return;const runs=DATA.runs;
  const ms=[...new Set(runs.map(r=>r.model))],dss=[...new Set(runs.map(r=>r.ds))];
  const fill=(id,vals,all)=>{const el=$(id),cur=el.value;if(el.options.length!==vals.length+(all?1:0)){el.innerHTML=(all?'<option value="*">all</option>':"")+vals.map(v=>`<option>${v}</option>`).join("");if(cur&&[...el.options].some(o=>o.value===cur))el.value=cur}};
  fill("model",ms,true);fill("ds",dss,true);
  const sel=runs.filter(r=>($("model").value==="*"||r.model===$("model").value||r.ref&&r.model===$("model").value)&&($("ds").value==="*"||r.ds===$("ds").value)&&($("which").value==="all"||r.state==="running"));
  const gs=groups(sel);$("chips").innerHTML=gs.map(g=>`<span class="chip ${off.has(g)?"off":""}" data-g="${g}"><i style="background:${colorOf(g,gs)}"></i>${g}</span>`).join("");
  document.querySelectorAll(".chip").forEach(c=>c.onclick=()=>{off.has(c.dataset.g)?off.delete(c.dataset.g):off.add(c.dataset.g);render()});
  if(!$("charts").children.length)$("charts").innerHTML=CH.map((c,i)=>`<div class="card" id="c${i}"><h2>${c.t}</h2><div class="plot"></div></div>`).join("");
  CH.forEach((c,i)=>draw($("c"+i),c,sel,gs,$("scale").value==="log"));
  $("title").textContent="Training live: sweep "+DATA.sweep;$("sub").textContent=DATA.header;$("tbl").textContent=DATA.table}

/* ---- dashboard */
const dF=(v,d=1)=>v==null||!isFinite(v)?'—':(+v).toFixed(d),dT=s=>{s=Math.max(0,Math.round(s||0));const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?h+'h '+m+'m':m?m+'m '+(s%60)+'s':s+'s'};
const DCOL=["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#B8860B","#7B4EA3","#444444"];
function lineChart(svg,series,key,log,W0,H0){
  const W=svg.clientWidth||W0||400,H=svg.clientHeight||H0||200,m={l:46,r:8,t:6,b:20};
  const lines=series.map((r,i)=>({r,c:DCOL[i%DCOL.length],pts:r.epochs.filter(e=>e[key]!=null&&(!log||e[key]>0)).map(e=>[e.e,e[key]])})).filter(l=>l.pts.length);
  if(!lines.length){svg.innerHTML='<text x="12" y="30" style="fill:var(--mut)">no data yet</text>';return}
  const all=lines.flatMap(l=>l.pts),tr=v=>log?Math.log10(v):v;let x1=Math.max(...all.map(p=>p[0])),y0=Math.min(...all.map(p=>tr(p[1]))),y1=Math.max(...all.map(p=>tr(p[1])));
  if(!log)y0=Math.min(0,y0);if(y1===y0)y1=y0+1;
  const px=x=>m.l+(x-1)/Math.max(x1-1,1)*(W-m.l-m.r),py=y=>H-m.b-(tr(y)-y0)/(y1-y0)*(H-m.t-m.b);
  let o='';nice(y0,y1,4).forEach(t=>{const y=H-m.b-(t-y0)/(y1-y0)*(H-m.t-m.b);o+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" stroke="var(--grid)"/><text x="${m.l-4}" y="${y+3}" text-anchor="end">${fmt(log?Math.pow(10,t):t)}</text>`});
  nice(1,x1,5).forEach(t=>{o+=`<text x="${px(t)}" y="${H-5}" text-anchor="middle">${t}</text>`});
  lines.forEach(l=>{o+=`<polyline fill="none" stroke="${l.c}" stroke-width="2" ${l.r.seed===1?'stroke-dasharray="6 3"':l.r.seed===2?'stroke-dasharray="2 3"':''} points="${l.pts.map(p=>px(p[0]).toFixed(1)+','+py(p[1]).toFixed(1)).join(' ')}"/>`;const q=l.pts[l.pts.length-1];if(l.r.state==='running')o+=`<circle cx="${px(q[0])}" cy="${py(q[1])}" r="3.5" fill="${l.c}" stroke="var(--bg)" stroke-width="2"/>`});
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=o;return lines;
}
function renderDash(){
  const d=DATA;if(!d||d.counts===undefined)return;const c=d.counts,total=d.total,n=c.done,run=c.running,q=c.queued,f=c.failed;
  $('#d-title').textContent=d.sweep;$('#d-state').textContent=run||q?'running':(f?'finished with failures':'finished');
  $('#d-sub').textContent=run+' running · elapsed '+dT(d.elapsed_s)+(d.eta_s?' · ETA ~'+dT(d.eta_s):'')+' · updated '+d.updated;
  $('#d-barfill').innerHTML=`<i style="width:${100*n/total}%;background:#0072B2"></i><i style="width:${100*run/total}%;background:#E69F00"></i><i style="width:${100*f/total}%;background:#D55E00"></i><i style="width:${100*q/total}%;background:var(--grid)"></i>`;
  $('#d-pct').textContent=n+' / '+total+' runs done ('+dF(100*n/total,0)+' %)';
  const best=Math.min(...d.jobs.map(j=>j.best).filter(v=>v!=null),Infinity),ep=d.jobs.filter(j=>j.state==='running').reduce((a,j)=>a+(j.epoch||0),0);
  const T=(b,t)=>`<div class="t"><b>${b}</b><span>${t}</span></div>`,o=d.ours;
  $('#d-tiles').innerHTML=T(n,'done')+T(run,'running now')+T(q,'queued')+T(f,'failed')+T(d.eta_s?dT(d.eta_s):'—','time left (estimate)')+T(isFinite(best)?dF(best,4):'—','best validation [m]')+T(ep,'epochs, running runs')+T(dF(o.cpu_cores,0)+' cores','CPU, our runs')+T(dF(o.rss_gb,1)+' GB','RAM, our runs')+T(dF(o.gpu_gb,1)+' GB','GPU memory, our runs');
  const order={running:0,failed:1,done:2,queued:3},jobs=[...d.jobs].sort((a,b)=>order[a.state]-order[b.state]||(a.name<b.name?-1:1)),show=jobs.filter(j=>j.state==='running'||j.state==='failed').concat(jobs.filter(j=>j.state==='done').slice(0,4));
  $('#d-runs-note').textContent=q+' queued, '+n+' done';
  $('#d-rt').innerHTML='<tr><th>Run</th><th>GPU</th><th>State</th><th>Epoch</th><th>Horizon</th><th>Elapsed</th><th>s / epoch</th><th>Windows/s</th><th>Val RMSE</th><th>Best</th><th>CPU</th><th>RAM</th><th>GPU MB</th></tr>'+
    show.map(j=>`<tr><td title="${esc(j.name)}">${esc(j.name.replace(/^(fx|st)_/,''))}</td><td>${j.gpu==null?'—':j.gpu}</td><td class="${j.state==='running'?'ok':j.state==='failed'?'bad':''}">${j.state}</td><td>${j.epoch||'—'}</td><td>${j.horizon||'—'}</td><td>${j.elapsed_s==null?'—':dT(j.elapsed_s)}</td><td>${dF(j.epoch_s,1)}</td><td>${dF(j.windows_per_s,0)}</td><td>${dF(j.val,4)}</td><td>${dF(j.best,4)}</td><td>${j.cpu_pct==null?'—':dF(j.cpu_pct,0)+' %'}</td><td>${j.rss_mb==null?'—':dF(j.rss_mb/1024,1)}</td><td>${j.gpu_mb==null?'—':dF(j.gpu_mb,0)}</td></tr>`).join('');
  $('#d-gt').innerHTML='<tr><th>GPU</th><th>Util</th><th>Memory</th><th>Temp</th><th>Power</th><th>Our runs</th></tr>'+d.gpus.map(g=>{const rs=d.jobs.filter(j=>j.state==='running'&&j.gpu===String(g.index)||j.state==='running'&&j.gpu===g.index);return `<tr><td>${g.index}</td><td>${dF(g.util,0)} %</td><td>${dF(g.mem_used_mb/1024,1)}/${dF(g.mem_total_mb/1024,0)}</td><td>${g.temp_c==null?'—':dF(g.temp_c,0)+'°'}</td><td>${g.power_w==null?'—':dF(g.power_w,0)+' W'}</td><td>${rs.length}</td></tr>`}).join('');
  $('#d-host').innerHTML=`<div>load <b>${dF(d.host.load1,0)}</b> / ${d.host.cores} cores</div><div>our CPU <b>${dF(o.cpu_cores,0)}</b> cores</div><div>our RAM <b>${dF(o.rss_gb,1)}</b> GB</div>`;
  // charts: the running runs (all runs if none runs)
  const names=new Set(jobs.filter(j=>j.state==='running').map(j=>j.name));let series=DATA.runs.filter(r=>names.has(r.name));if(!series.length)series=DATA.runs.filter(r=>r.state==='done').slice(-6);
  const lines=lineChart($('#d-valsvg'),series,'vk',$('#d-valsvg').dataset.log!=='0');lineChart($('#d-losssvg'),series,'roll',true);
  $('#d-leg').innerHTML=(lines||[]).map(l=>`<span><i style="background:${l.c}"></i>${esc(l.r.name.replace(/^(fx|st)_/,''))}</span>`).join('');
  $('#d-val-note').textContent=series.length+' run'+(series.length===1?'':'s')+', x = epoch';
  const fin=d.jobs.filter(j=>j.best!=null).sort((a,b)=>a.best-b.best).slice(0,8);
  $('#d-bt').innerHTML='<tr><th>Run</th><th>Best val [m]</th><th>State</th></tr>'+fin.map(j=>`<tr><td title="${esc(j.name)}">${esc(j.name.replace(/^(fx|st)_/,''))}</td><td>${dF(j.best,4)}</td><td>${j.state}</td></tr>`).join('');
}
function setView(v){document.body.classList.toggle('dash',v==='dash');$('#tab-dash').classList.toggle('on',v==='dash');$('#tab-curves').classList.toggle('on',v!=='dash');if(v==='dash')renderDash();else render();try{history.replaceState(null,'',v==='dash'?'#dashboard':'#curves')}catch(e){}}
$('#tab-dash').onclick=()=>setView('dash');$('#tab-curves').onclick=()=>setView('curves');addEventListener('resize',()=>{if(document.body.classList.contains('dash'))renderDash()});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function tick(){try{DATA=await (await fetch("live.json?"+Date.now())).json();if(document.body.classList.contains("dash"))renderDash();else render()}catch(e){$("sub").textContent="waiting for live.json ..."}}
["model","ds","which","scale"].forEach(id=>$(id).onchange=render);addEventListener("resize",()=>{if(!document.body.classList.contains("dash"))render()});setView(location.hash==="#curves"?"curves":"dash");tick();setInterval(tick,5000);
</script>"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--exp-root", type=Path, default=Path("outputs/experiments"))
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--html", type=Path, help="write the page (live.html + live.json) to this file every interval instead of printing")
    ap.add_argument("--serve", action="store_true", help="write the page into the sweep folder AND serve that folder over HTTP (one process); open /live.html")
    ap.add_argument("--host", default="127.0.0.1", help="bind address for --serve (default: local only; reach it through an SSH tunnel)")
    ap.add_argument("--port", type=int, default=60194, help="port for --serve (0 picks a free one; this is a shared server)")
    ap.add_argument("--allow", nargs="*", default=None, metavar="CIDR",
                    help="client networks that may open the page (default: this machine, 10.0.0.0/8 and 100.64.0.0/10 (Tailscale); others get 403)")
    a = ap.parse_args(argv)
    if a.serve:
        import functools
        import threading
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        page = a.sweep / "live.html"
        threading.Thread(target=main, args=(["--sweep", str(a.sweep), "--exp-root", str(a.exp_root), "--interval", str(a.interval), "--html", str(page)],), daemon=True).start()
        from physiworld_uav.visualization.access import DEFAULT_ALLOW, allowed
        nets = tuple(a.allow) if a.allow else DEFAULT_ALLOW

        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args, **kw) -> None:                      # keep the terminal quiet
                pass

            def _deny(self) -> bool:
                if allowed(self.client_address[0], nets):
                    return False
                self.send_error(403, "forbidden: this page is limited to the campus network")
                return True

            def do_GET(self) -> None:  # noqa: N802
                if not self._deny():
                    super().do_GET()

            def do_HEAD(self) -> None:  # noqa: N802
                if not self._deny():
                    super().do_HEAD()

        handler = functools.partial(Quiet, directory=str(a.sweep))
        srv = ThreadingHTTPServer((a.host, a.port), handler)
        print(f"serving {a.sweep} at http://{a.host}:{srv.server_address[1]}/live.html  (Ctrl+C to stop)", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
        return
    prev: dict = {}
    if a.once:                                           # two samples 2 s apart so the CPU % is defined
        _, prev = render(a.sweep, a.exp_root, prev); time.sleep(2)
        print(render(a.sweep, a.exp_root, prev)[0]); return
    if a.html:
        a.html.write_text(PAGE)                              # static page: it fetches live.json every 5 s and redraws in place
        jpath = a.html.with_name("live.json"); prev2: dict = {}
        try:
            while True:
                text, prev = render(a.sweep, a.exp_root, prev)
                struct, prev2 = structured(a.sweep, a.exp_root, prev2)
                st = json.loads((a.sweep / "status.json").read_text()) if (a.sweep / "status.json").exists() else {}
                lines = text.split("\n")
                payload = {**struct, "header": lines[0] + "\n" + lines[1], "table": "\n".join(lines[3:]), "runs": run_series(a.exp_root, st)}
                tmp = jpath.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, separators=(",", ":"))); tmp.replace(jpath)
                time.sleep(a.interval)
        except KeyboardInterrupt:
            return
    if not sys.stdout.isatty():
        print("stdout is not a terminal: an in-place refresh is impossible here (each refresh would be appended).\n"
              "Run this in a real terminal, or use  --html live.html  (open it in a browser / the IDE), or  watch -n 5 'python3 -m physiworld_uav.visualization.live_training --sweep "
              f"{a.sweep} --once'.\n")
        print(render(a.sweep, a.exp_root, prev)[0]); return
    try:
        while True:
            text, prev = render(a.sweep, a.exp_root, prev)
            print("\033[2J\033[H" + text + "\n\n(Ctrl-C to stop; the training continues)", flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
