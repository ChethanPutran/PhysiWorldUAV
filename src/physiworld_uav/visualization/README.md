# Visualization: live pages and plots

Everything that shows data (live generation page, live training page, dataset plots) lives here. One entry point:

```
python -m physiworld_uav.visualization <command> [options]      # needs PYTHONPATH=<repo>/src; --help shows the options of a command
```

| Command | What it does | Needs |
|---|---|---|
| `generation` | **Live dataset report** while a dataset is generated: running / queued / pending / done per episode, per-worker progress, CPU, RAM, GPU, disk, charts of the plan | python3 (stdlib) |
| `report` | The same report written once as a static `report.html` (no server) | python3 |
| `progress` | Generation progress and ETA printed in the terminal | python3 |
| `training` | **Live training page**: loss, validation error, epoch time per run, table with CPU/GPU/RAM, one sweep at a time | python3 |
| `plan` | Distribution plots of a planned or generated dataset from its manifest (sizes per role, wind, vehicle factors, scenarios) | numpy, matplotlib (container) |
| `wind` | Distribution plots of the wind stored in a dataset (per time step, per episode) | numpy, matplotlib, h5py (container) |
| `trajectories` | Actual vs predicted trajectories per episode (top view, altitude, error vs time); wind speed (magnitude) and azimuth (direction) in the legend and a wind arrow; `summary.png` = error vs wind speed / azimuth per role | torch, checkpoints (container) |
| `figures` | **All figures of the report and slides** (error accumulation, coverage, wind, domain randomisation, counterfactuals, IMU, dataset versions, training curves, error vs horizon, ID vs OOD, severity, error distribution, qualitative and failure cases, architecture and training/imagination diagrams, timeline). Writes pdf + png (+ jpg, svg) per figure to `outputs/figures`; `--only`, `--out`, `--formats`, `--copy-to`, `--list` | numpy, matplotlib; cached data from `figure-data` |
| `figure-data` | Extracts what the figures need from the HDF5 shards and checkpoints (rollouts, coverage, wind series, IMU, counterfactual siblings, per-episode errors) into `outputs/figure_cache` | torch, h5py, checkpoints (container) |
| `tables` | LaTeX tables computed from the evaluation files and datasets (main results, dataset statistics) | numpy |
| `evaluation` | Figures from evaluation JSON files: error vs horizon per role, model x role heatmaps (position, attitude), OOD gap | numpy, matplotlib |

Convenience launcher for the two servers, run on the **host**: `scripts/start_report_server.sh` (`generation` / `training` / `status` / `stop`).

## Who can open the pages

The two servers have no login. Started with `--public` they listen on all interfaces but answer only clients from this machine, the campus network `10.0.0.0/8` and Tailscale `100.64.0.0/10`; everyone else gets HTTP 403 (`access.py`, option `--allow <CIDR> ...` to change it). Your own address does not matter, only the server's: 10.24.36.121 (static; it also has the Tailscale address 100.115.191.34). From another network (a hostel, a phone on mobile data) the page opens only if that network can reach the server; otherwise use an SSH tunnel (works wherever SSH works) or Tailscale.

## 1. Live data generation

```
scripts/start_report_server.sh generation data/datasets/physiworld_pilot_v7_v0.7.0            # port 60193
# or directly:  PYTHONPATH=$PWD/src python3 -m physiworld_uav.visualization generation --dataset <dir> --port 60193
```

The server listens on `127.0.0.1` only. From your laptop open a tunnel and a browser:

```
ssh -L 3000:127.0.0.1:60193 <user>@<server>          # keep it open
http://localhost:3000/                                # use http, not https
```

The page has two views (tabs at the top right; `#dashboard` / `#report` in the URL): the **Dashboard**, which fits on one screen without scrolling (font and panel sizes follow the window height; checked at 1920x1080 and 1366x768), and **Plan & episodes**, the long report with the wind, vehicle and scenario charts and the filterable episode table. The Dashboard shows a progress bar (done / running / queued), ten tiles (done, running, queued, pending, failed, ended early, episodes per hour, time left, disk, CPU), the worker table, a chart of episodes written over time with a dashed projection to the finish, GPUs and host, progress by split role and the latest episodes. A static `report.html` (no server) opens on the report view.

**Phones and tablets:** on a narrow or portrait screen (under 900 px wide, or taller than wide) the Dashboard becomes one scrolling column: two tiles per row, the panels stacked, the worker table reduced to the columns that fit (progress, time on episode, done, per hour, RAM) and tables that scroll sideways. The Plan & episodes view stacks its panels and lets the wide charts scroll sideways so that their text stays readable. Checked with headless Chromium at 390x844 (phone), 768x1024 (tablet), 1366x768 and 1920x1080.

What the page shows (it polls every 5 s; the plan itself is embedded, so it works from the moment `plan` has run):

* **Generation status**: episodes done / running now / queued / pending / failed / ended early, elapsed time, estimated time left; a worker table (GPU, done, queued, assigned, running episode); the latest finished episodes; the *Status* column and filter of the episode table.
  A parallel generation has no `index.json` until the final merge, so the server assembles the progress from the per-worker `index.workerNN.json` files. *Running* and *queued* are inferred from the generator's deterministic worker assignment (`experiments.generate.assign_workers`, the same rule) and its processing order, for the workers that are alive (`ps`); they are not reported by the workers.
* **Resources** (`/stats.json`, read from `/proc`, `nvidia-smi` and the dataset folder): dataset size on disk and the projected final size, free space, host load, RAM of our workers and available RAM, per-GPU utilisation / memory / temperature / power, per worker PID, GPU, RAM, CPU %, threads, GPU memory, episodes, episodes per hour, disk.
* **Per-episode progress**: *Now running*, *Progress* (`112 / 200 steps (56 %)`) and *On it / typical* (time on the current episode / mean time per episode). The exact step needs the worker to write `progress.workerNN.json` (generators started after 2026-11-26 do, once per second). For older workers the page estimates the percentage from the time since the worker's last finished episode and marks it `(estimate)`.

To stop a worker without losing work: `scripts/stop_after_episode.py <dataset> --experiment <name> --workers 0,3,6` waits for each worker to finish its current episode, then stops it. Restart with the **same** worker count (see `HANDOFF.md`).

Terminal alternative (no browser): `python3 -m physiworld_uav.visualization progress <dataset> --watch 60`.

## 2. Live training

```
scripts/start_report_server.sh training outputs/sweeps/fix_v1             # port 60194
ssh -L 3001:127.0.0.1:60194 <user>@<server>   ->   http://localhost:3001/live.html
```

Reads `<sweep>/status.json` (written by `scripts/run_sweep.py`) and each run's `outputs/experiments/<run>/metrics.jsonl`. Four charts per selection (training rollout loss, validation error at the validation horizon, one-step validation error, seconds per epoch), selectors for model / dataset / running only, log or linear scale, one colour per variant (dashed = seed 1, dotted = seed 2), and a table with state, GPU, epoch, horizon, time per epoch and per episode, windows/s, CPU, RAM and GPU memory. Terminal version (needs a real terminal; the screen is redrawn in place): `python3 -m physiworld_uav.visualization.live_training --sweep <dir>`; one snapshot with `--once`.

## 3. Plots

Run inside the container (they need numpy and matplotlib; `wind` also h5py):

```
docker exec isaac-sim bash -c 'cd /workspace/PhysiWorldUAV && source omnidrones_env/bin/activate && export PYTHONPATH=$PWD/src:$PYTHONPATH && \
  python -m physiworld_uav.visualization plan --dataset data/datasets/physiworld_pilot_v7_v0.7.0 --out outputs/reports/data_distribution_v7'
```

* `plan` (manifest only, also for a dataset that is not generated yet): `figures/1_sizes_per_role.png`, `2_wind.png` (mean / peak speed by role with the ID and OOD limits, azimuth, elevation, profile types), `3_vehicle_factors.png`, `4_scenarios_modes.png`, `5_initial_conditions.png`, `tables/per_role.csv`. Options `--id-peak-limit` and `--ood-peak-limit` draw the limits.
* `wind` (reads the stored wind of the episodes): `figures/wind_speed_per_timestep.png`, `wind_per_episode.png`, `tables/wind_by_role.csv`.

Other plots live in this folder as well: `dataset_report.py` (the report), `episode_plots.py` (a quick look at one episode: `python -m physiworld_uav.visualization.episode_plots --dataset <dir> --episode <id>`), `templates/` (the report page). Evaluation figures and tables of trained models are made by `evaluation/report.py`, not here.

## Tests

`tests/cpu/test_visualization.py` (assembled index, live states, resource figures, progress). Run with `python -m pytest tests/cpu/test_visualization.py -q`.


## Report and slide figures (reusable)

```
# 1. in the container: data that needs HDF5 / checkpoints (about an hour for the per-episode errors on v7)
python -m physiworld_uav.visualization figure-data                 # --only traj extra errors failure
# 2. anywhere with matplotlib: draw everything (pdf + png), copy into the report folder
python -m physiworld_uav.visualization figures --formats pdf png --copy-to report/figures_new
python -m physiworld_uav.visualization figures --list              # names; --only error_accumulation horizon ...
python -m physiworld_uav.visualization tables --out report/auto_tables.tex
```

Each figure is one function `fig_<name>` in `report_figures.py` (style, palette and `save()` at the top); to add a figure, write a function that draws and calls `save(fig, "<name>")`. For a new dataset version pass `--dataset data/datasets/<name>` and change the model names in `figure_data.py` (`KV_MODELS`).
