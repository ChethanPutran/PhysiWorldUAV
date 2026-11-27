# PhysiWorldUAV

Physics-prior latent world models for reliable long-horizon quadrotor prediction under out-of-distribution (OOD) wind. This is M.Tech thesis code.

The model has the form

```
ds/dt = f_phys(s, u) + d_θ(h, s, u)
```

where `f_phys` is a nominal rigid-body quadrotor model and `d_θ` is a learned residual.

**Current scope.** The repository has two parts:

1. **Data-generation pipeline.** Isaac Sim 4.1 and OmniDrones (Hummingbird quadrotor) produce physically validated, reproducible trajectories with IMU, controls, wind, domain parameters and counterfactual pairs.
2. **World-model code.** Baselines B0 to B6 (`models/`), a curriculum trainer (`training/`), losses, per-role evaluation and analyses (`evaluation/`), and plots and live pages (`visualization/`).

The thesis question: does a known-physics prior make long-horizon prediction under OOD wind more reliable than a pure learned latent model? Results so far are mixed (see `MASTER_README.md` and `docs/results/`).

## Status (2026-11-27)

| Item | Status |
|---|---|
| Runs 1 to 5: audit, plan, config architecture, validated physics, pilot dataset | Done |
| Pilot v0.5.0 / v0.6.0 (147 episodes, 50 Hz), 16 baseline runs and repeat sweep `fix_v1` (29 runs) | Done, `docs/results/` |
| Pilot v7 (4,530 episodes, 20 Hz model step, wide stratified wind, 473 vehicles) | Generated and validated (1 quality failure, open decision B24) |
| Stage-1 training on v7 (B2, B3, B3 small, B4, B4 at B3 width, B5a, B5b) and evaluation on all roles | Done, `outputs/reports/v7_stage1/` |
| Known-vehicle physics prior (true mass, k_F, k_M, inertia given to `f_phys`) | Implemented; 9 training runs (`kv_v07_*`) in progress |
| Evaluation of the known-vehicle runs, seeds, reliable-horizon experiment, OOD-failure tests | Next |
| Midterm report and slides | `report/midterm_report.pdf`, `report/slides/PhysiWorldUAV_midterm.pptx` |

Tests: 296 CPU tests and 17 Isaac tests pass. Details and open decisions: `HANDOFF.md`, `docs/TODO.md`, `docs/problems_log.md`.

## How it works

```
config/ (Hydra groups) → experiments/plan.py → manifest.jsonl
        → experiments/runner.py (Isaac Sim + OmniDrones) → HDF5 shards
        → data/dataset_validator.py → index.json
        → data/dataset_builder.py (model-rate view: decimate, windows, future actions)
        → training/run.py (B2 to B6, horizon curriculum) → evaluation/run.py (per-role tables)
```

* **Plan, then simulate.** Every random choice (domain, initial condition, reference, excitation, wind, counterfactual sibling, split role) is made at planning time from a seed tree. An episode is a pure function of one manifest row.
* **Truth, observations and actions are stored separately.** IMU noise is applied at load time, not at generation.
* **Physics:** PhysX at 200 Hz, OmniDrones rotor model, plus project aerodynamics (rotor drag, body drag with per-axis C_D and orientation-dependent area, optional ground effect) injected as one wrench. Wind comes from steady, gust, OU and Dryden components.
* **Fail loudly.** The runner refuses unvalidated force models, wind without force models, and schema violations.

Conventions: world ENU, body FLU, Hamilton quaternion wxyz (body→world). Physics step 5 ms; model step 20 ms (pilot v0.5/v0.6) or 50 ms (pilot v7). Model state is `p | v | q | ω_B`.

## Repository layout

| Path | Contents |
|---|---|
| `src/physiworld_uav/` | Data: `core`, `drones`, `control`, `trajectories`, `excitation`, `wind`, `randomization`, `forces`, `experiments`, `data`. Models: `models`, `baselines`, `losses`, `training`, `evaluation`, `visualization` |
| `config/` | Hydra groups: drone, physics, limits, controller, dataset, scenario, excitation, wind, domain, counterfactual, forces, experiment |
| `tests/cpu`, `tests/isaac` | Unit tests and Isaac physics-validation tests |
| `docs/guides/` | Architecture, configuration, environment, dataset schema, scenarios, assumptions and limitations |
| `docs/models/` | Force accounting, aerodynamics, drone and wind models, validation numbers |
| `docs/audit/` | Run reports and the implementation plan |
| `docs/results/` | Baseline comparisons, sweep results, 20 Hz tracking check |
| `docs/thesis.tex`, `docs/problems_log.md`, `docs/TODO.md` | Thesis text, every problem met and its resolution, remaining work |
| `report/` | Midterm report (LaTeX, PDF), `MIDTERM_GUIDE.md`, slides in `report/slides/` |
| `.idea/` | Design notes and specifications (`.idea/specs/`) |
| `docker/`, `scripts/` | Container image and helper scripts (generation, sweeps, evaluation, live pages, `upload_to_gdrive.sh`) |
| `CLAUDE.md`, `HANDOFF.md`, `MASTER_README.md` | Working notes; the project story and results to date |

## Requirements

* Linux with an NVIDIA GPU (developed on RTX A6000) and GLIBC ≥ 2.34
* Docker with the NVIDIA container toolkit
* Isaac Sim 4.1.0, Isaac Lab v1.0.0 and OmniDrones, all provided by the container image
* Python 3.10 (Isaac Sim 4.1 requires it)

Isaac Sim and OmniDrones are not resolvable with plain `pip`. Everything that touches the simulator runs inside the container.

## Setup

### 1. Build the image

`docker/Dockerfile` starts from `nvcr.io/nvidia/isaac-sim:4.1.0` and adds the noVNC stack, a Python 3.10 `uv` environment, Isaac Lab v1.0.0, TensorDict/TorchRL and OmniDrones.

```bash
docker build -t isaac-sim-omnidrones:4.1.0 -f docker/Dockerfile .
```

### 2. Start the container

`scripts/build.sh` is a reference `docker run` command. Adjust the mounts to your machine. The repository must be mounted at `/workspace/PhysiWorldUAV`.

```bash
docker start -ai isaac-sim        # an existing container
```

Without `--rm`, your installed packages survive restarts. If you create the container yourself, mount the repository at `/workspace/PhysiWorldUAV`.

### 3. Manual environment setup (only if you did not bake it into the image)

Inside the container:

```bash
apt-get update && apt-get install -y --no-install-recommends git build-essential cmake wget curl
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

cd /workspace/PhysiWorldUAV
uv venv --python 3.10 --seed omnidrones_env
source omnidrones_env/bin/activate

git clone https://github.com/btx0424/OmniDrones.git
git clone https://github.com/isaac-sim/IsaacLab.git && (cd IsaacLab && git checkout v1.0.0)
git clone https://github.com/btx0424/tensordict.git
git clone https://github.com/pytorch/rl.git

export ISAACSIM_PATH=/isaac-sim
uv pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash
(cd IsaacLab && ./isaaclab.sh --install)
(cd tensordict && uv pip install -e .)
(cd rl && uv pip install -e .)
(cd OmniDrones && uv pip install -e .)
```

Use Isaac Lab **v1.0.0**. The latest Isaac Lab needs Python 3.12 and does not work with Isaac Sim 4.1.

### 4. Verify

```bash
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')"
python -c "import omni.isaac.lab; print('Isaac Lab OK')"
python -c "import omni_drones; print('OmniDrones OK')"
```

## Running

Every command runs in the container, in the activated environment, with `src` **appended** to `PYTHONPATH`. Overwriting `PYTHONPATH` hides Isaac's `isaacsim` module.

```bash
docker exec isaac-sim bash -c 'cd /workspace/PhysiWorldUAV && source omnidrones_env/bin/activate && export PYTHONPATH=$PWD/src:$PYTHONPATH && <command>'
```

| `<command>` | Purpose |
|---|---|
| `python -m pytest tests/cpu -q` | CPU tests (about 5 min with `OMP_NUM_THREADS=4`, 296 tests) |
| `python -m pytest tests/isaac --isaac -q` | Isaac physics validation (17 tests, needs a GPU) |
| `python -m physiworld_uav.experiments.generate plan experiment=pilot` | Write the manifest only |
| `python -m physiworld_uav.experiments.generate run experiment=smoke` | Generate a small dataset |
| `python -m physiworld_uav.experiments.generate run experiment=pilot` | Generate the pilot dataset (`experiment=pilot_v7` for v7; use `scripts/generate_parallel.sh --workers 8 --gpus 0,1,2 -- experiment=pilot_v7` for parallel generation) |
| `python -m physiworld_uav.experiments.generate validate --dataset data/datasets/<name>` | Validate a dataset |
| `python -m physiworld_uav.experiments.generate regenerate --dataset data/datasets/<name> --episode <id>` | Reproduce one episode |
| `python -m physiworld_uav.visualization.dataset_report --dataset data/datasets/<name>` | HTML report of the plan and index (`report.html` in the dataset folder) |
| `python -m physiworld_uav.visualization.episode_plots --dataset data/datasets/<name> --episode <id>` | PNG quick-look of one episode (`--list` shows ids) |
| `python -m physiworld_uav.data.statistical_validator --dataset data/datasets/<name>` | Data quality report: range, distribution, duplicates, correlation, physical consistency, coverage (`outputs/reports/<name>_quality/`) |
| `python -m physiworld_uav.training.run --dataset data/datasets/<name> --name <run> --model b3 --config config/training/default.yaml config/training/compare.yaml` | Train a model: `b2`, `b3`, `b4`, `b4_rw`, `b5a`, `b5b`, `b6b_future`, `b6b_current`. Add `config/training/known_vehicle.yaml` for the known-vehicle prior and `config/training/v7_openloop.yaml` for v7. `--resume` continues a run |
| `python scripts/run_sweep.py --jobs outputs/sweeps/<sweep>/jobs.json --gpus 0,1,2 --per-gpu 3` | Run many trainings (job lists: `scripts/sweep_jobs_*.py`); status in `status.json`. Use only free GPUs |
| `python -m physiworld_uav.evaluation.run --dataset data/datasets/<name> --models b0 b1 b1_kv b6a_future b6a_future_kv b3=<checkpoint>` | Per-role evaluation (JSON + Markdown). `_kv` names use each episode's true vehicle parameters |
| `python -m physiworld_uav.evaluation.analysis regime\|anchor\|residual\|shuffle --dataset ... --models ...` | Wind/no-wind gap, re-observation, residual diagnostics, shuffled-action test |
| `python -m physiworld_uav.visualization trajectories\|evaluation\|wind\|plan\|report\|training ...` | Trajectory plots (actual vs predicted, wind in the legend), evaluation figures, wind and plan distributions, dataset report, live training page. `python -m physiworld_uav.visualization --help` lists all |
| `scripts/start_report_server.sh training <sweep>` | Live training dashboard (also `scripts/start_generation.sh` for generation) |
| `scripts/upload_to_gdrive.sh --dry-run` | Upload datasets and outputs to Google Drive with rclone (needs `rclone config` first) |

Hydra overrides go after the command, e.g. `experiment=pilot`. Datasets are written to `data/datasets/<id>_v<version>/` and are not committed, nor are `outputs/` and `logs/`. Training and large jobs need the supervisor's approval each time; do not use GPU 3 on the shared host.

Notes:
* `SimulationApp.close()` ends the process, so the CLI validates and writes `index.json` before closing Isaac.
* Files created in the container are owned by root. Fix with `chown 1003:1003 <path>`.
* The Isaac IMU needs the sensor created before `sim.reset()` and the `gpu_dynamics_cpu_readback` physics pipeline. This is already handled in the code.

## Viewing the Isaac Sim GUI (optional)

The container includes Xvfb, x11vnc and noVNC.

1. In the container, start a virtual display and Isaac Sim:
   ```bash
   Xvfb :99 -screen 0 1920x1080x24 &
   export DISPLAY=:99
   ./runheadless.native.sh --no-window     # or ./isaac-sim.sh for the full GUI
   ```
2. In a second container shell (`docker exec -it isaac-sim bash`):
   ```bash
   x11vnc -display :99 -forever -nopw -shared -rfbport 5900 &
   websockify --web /usr/share/novnc/ 6080 localhost:5900 &
   ```
3. On your laptop, open a tunnel (replace the placeholders):
   ```bash
   ssh -L 6080:localhost:6080 <user>@<server-ip>
   ```
4. Browse to `http://localhost:6080/vnc.html` and click **Connect**.

Do not expose port 6080 to the internet. The first launch builds the shader cache and can take several minutes.

## Documentation

* Start here: `MASTER_README.md` (project story, results, problems), `HANDOFF.md` (current position), `docs/TODO.md` (remaining work)
* Architecture: `docs/guides/architecture.md`; configuration: `docs/guides/configuration.md`; scenarios: `docs/guides/scenarios_and_experiment_types.md`
* Dataset schema: `docs/guides/dataset_schema.md`; reproducibility: `docs/guides/reproducibility_and_domains.md`; glossary: `docs/guides/glossary.md`
* Aerodynamic and drone models: `docs/models/aerodynamics.md`, `docs/models/drone_and_wind.md`, `docs/models/force_accounting.md`
* Assumptions and limitations: `docs/guides/assumptions_limitations.md`
* Results: `docs/results/`; problems and fixes: `docs/problems_log.md`; run reports: `docs/audit/`
* Model specification: `.idea/specs/model_specification.md`; thesis text: `docs/thesis.tex`
* Midterm: `report/MIDTERM_GUIDE.md`, `report/midterm_report.tex`, `report/slides/`
