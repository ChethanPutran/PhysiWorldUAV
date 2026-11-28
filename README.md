# PhysiWorldUAV

Physics-prior latent world models for reliable long-horizon quadrotor prediction under out-of-distribution (OOD) wind.

The project studies one question: does a known-physics prior make long-horizon, open-loop prediction of a quadrotor more reliable than a pure learned latent model when the wind differs from the training wind? The model has the form

$$
\dot{s} = f_{\mathrm{phys}}(s, u) + d_\theta(h, s, u)
$$

where $s = (p, v, q, \omega_B)$ is the physical state (position, velocity, attitude, body rate), $u$ the rotor-speed action, $f_{\mathrm{phys}}$ a rigid-body quadrotor model with no aerodynamics, $d_\theta$ a learned residual and $h$ a recurrent latent memory built from IMU history.

The physics prior is the rigid-body model

$$
\dot{p} = v, \qquad m\,\dot{v} = m g + R f_T + f_{\mathrm{ext}}, \qquad \dot{R} = R\,[\omega]_\times, \qquad J\,\dot{\omega} = \tau - \omega \times J\omega + \tau_{\mathrm{ext}},
$$

with rotor thrust $f_T = \sum_i k_F \Omega_i^2$; drag and wind enter only through the learned residual. Evaluation reports the open-loop position error at horizon $k$,

$$
E_p(k) = \sqrt{\tfrac{1}{N}\sum_{i=1}^{N} \lVert \hat{p}_{i,t+k} - p_{i,t+k} \rVert^2 }.
$$

The repository contains everything needed to run the study end to end:

1. **Data generation.** A simulation pipeline (Isaac Sim 4.1 and OmniDrones, Hummingbird quadrotor) that produces physically validated, reproducible episodes with ground truth, IMU, controls, wind, vehicle parameters and counterfactual pairs.
2. **World models.** Baselines B0 to B6, the proposed model (B3), a curriculum trainer and the loss.
3. **Evaluation.** Open-loop evaluation per ID and OOD role, error against horizon, and diagnostics.
4. **Visualization.** Plots, live pages and report figures drawn from the data and the results.

## Models

All learned models share the same data interface, loss, horizon curriculum and optimiser, so differences come from the model only.

| Name | Model | Purpose |
|---|---|---|
| B0 | hold the last state | no-model floor |
| B1 | physics only ($f_{\mathrm{phys}}$, RK4) | what the known equations give alone |
| B2 | pure neural latent model (same encoder, GRU and decoder as B3) | pure-learned side of the comparison |
| **B3** | **$f_{\mathrm{phys}}$ + residual $d_\theta$ + latent memory** | **proposed model** |
| B4 | $f_{\mathrm{phys}}$ + residual, no latent memory | is the memory what helps? |
| B5a / B5b | learned vector field / exact kinematics + learned dynamics, no $f_{\mathrm{phys}}$ | is the physics split what helps? |
| B6a / B6b | $f_{\mathrm{phys}}$ + recorded disturbance wrench / B3 given the wrench | upper bound if the disturbance were known |

Names ending in `_kv` (B1, B6a, and the `known_vehicle` training option) give $f_{\mathrm{phys}}$ each episode's true mass, thrust and torque constants and inertia.

Evaluation roles: `test_id` and six OOD test roles (`test_ood_magnitude`, `_direction`, `_temporal`, `_gust`, `_physics`, `_trajectory`), each with a validation twin.

## How it works

```
config/ (Hydra groups) → experiments/plan.py → manifest.jsonl
        → experiments/runner.py (Isaac Sim + OmniDrones) → HDF5 shards
        → data/dataset_validator.py → index.json
        → data/dataset_builder.py (model-rate view: decimate, windows, future actions)
        → training/run.py (horizon curriculum) → evaluation/run.py (per-role tables)
        → visualization/ (figures, live pages)
```

* **Plan, then simulate.** Every random choice (vehicle, initial condition, reference, excitation, wind, counterfactual sibling, split role) is made at planning time from a seed tree, so an episode is a pure function of one manifest row and can be regenerated exactly.
* **Truth, observations and actions are stored separately.** IMU noise is applied at load time, not at generation.
* **Physics.** PhysX at 200 Hz with the OmniDrones rotor model, plus project aerodynamics (rotor drag, body drag with per-axis drag coefficients, optional ground effect) injected as one wrench. Wind comes from steady, gust, step, sinusoid, Ornstein-Uhlenbeck and Dryden components with realised-statistics checks.
* **Fail loudly.** The runner refuses unvalidated force models, wind without force models and schema violations.

Conventions: world ENU, body FLU, Hamilton quaternion wxyz (body to world). Physics step 5 ms; model step $\Delta t$ = 20 ms or 50 ms. The model state is $s = (p, v, q, \omega_B)$, with $\omega_B$ in the body frame.

## Repository structure

| Path | Contents |
|---|---|
| `src/physiworld_uav/core`, `drones`, `control` | Simulator wrapper, frames, seeds, provenance; vehicle parameters and domain binding to PhysX; Lee, attitude and rate controllers |
| `.../trajectories`, `excitation`, `wind`, `randomization`, `forces` | Reference library and feasibility checks; bounded excitation; wind models and regimes; Sobol/LHS sampling with ID and OOD shells; aerodynamic models and the wrench injector |
| `.../experiments`, `data` | Planner, counterfactual siblings, runner and the `generate` CLI; schema, recorder, HDF5 writer, validators, resampler, dataset builder, splits |
| `.../models`, `losses`, `training`, `evaluation` | B0 to B6, loss functions, curriculum trainer, per-role evaluation and analyses |
| `.../visualization` | Plots, live pages, report figures and tables (see `visualization/README.md`) |
| `config/` | Hydra groups: drone, physics, limits, controller, dataset, scenario, excitation, wind, domain, counterfactual, forces, experiment, model, training, evaluation |
| `tests/cpu`, `tests/isaac` | Unit tests; Isaac physics-validation tests |
| `scripts/` | Parallel generation, training and evaluation launchers, sweeps, live-page servers, upload |
| `docker/` | Container image |
| `docs/` | `guides/` (architecture, configuration, environment, dataset schema, scenarios, assumptions), `models/` (force accounting, aerodynamics, drone and wind), `results/` |
| `data/`, `outputs/`, `logs/` | Generated datasets, checkpoints, evaluation files, figures, logs (not committed) |

## Requirements

* Linux with an NVIDIA GPU and GLIBC ≥ 2.34
* Docker with the NVIDIA container toolkit
* Isaac Sim 4.1.0, Isaac Lab v1.0.0 and OmniDrones, all provided by the container image
* Python 3.10 (required by Isaac Sim 4.1)

Isaac Sim and OmniDrones cannot be installed with plain `pip`. Everything that touches the simulator runs inside the container. The model, evaluation and plotting code needs only Python, PyTorch, NumPy, h5py and matplotlib.

## Setup

### 1. Build the image

```bash
docker build -t isaac-sim-omnidrones:4.1.0 -f docker/Dockerfile .
```

`docker/Dockerfile` starts from `nvcr.io/nvidia/isaac-sim:4.1.0` and adds the noVNC stack, a Python 3.10 `uv` environment, Isaac Lab v1.0.0, TensorDict/TorchRL and OmniDrones.

### 2. Start the container

`scripts/build.sh` is a reference `docker run` command; adjust the mounts to your machine. The repository must be mounted at `/workspace/PhysiWorldUAV`.

```bash
docker start -ai isaac-sim        # an existing container
```

### 3. Manual environment setup (only if it is not baked into the image)

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

Use Isaac Lab **v1.0.0**; later versions need Python 3.12 and do not work with Isaac Sim 4.1.

### 4. Verify

```bash
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')"
python -c "import omni.isaac.lab; print('Isaac Lab OK')"
python -c "import omni_drones; print('OmniDrones OK')"
```

## Usage

Every command runs in the container, in the activated environment, with `src` **appended** to `PYTHONPATH` (overwriting it hides Isaac's `isaacsim` module):

```bash
docker exec isaac-sim bash -c 'cd /workspace/PhysiWorldUAV && source omnidrones_env/bin/activate && export PYTHONPATH=$PWD/src:$PYTHONPATH && <command>'
```

Hydra overrides go after the command, for example `experiment=pilot_v7`. Datasets are written to `data/datasets/<id>_v<version>/`.

### Generate a dataset

An experiment config in `config/experiment/` defines the plan: `smoke` (one episode), `pilot`, `pilot_v7` (wide stratified wind, 20 Hz model step), `pilot_wide_wind`, `stability_20hz`, `diversity_check`, `wind_envelope_check`.

```bash
python -m physiworld_uav.experiments.generate plan     experiment=pilot_v7     # write the manifest only
python -m physiworld_uav.experiments.generate run      experiment=smoke        # simulate and write HDF5 shards
python -m physiworld_uav.experiments.generate validate --dataset data/datasets/<name>
python -m physiworld_uav.experiments.generate regenerate --dataset data/datasets/<name> --episode <id>   # reproduce one episode
scripts/generate_parallel.sh --workers 8 --gpus 0,1,2 -- experiment=pilot_v7  # parallel workers, then merge
python -m physiworld_uav.data.statistical_validator --dataset data/datasets/<name>   # quality report in outputs/reports/<name>_quality/
```

Generation is CPU-bound, so run several workers per GPU and choose GPUs with `CUDA_VISIBLE_DEVICES`. Runs are crash-safe and resumable (`--resume`).

### Train models

```bash
python -m physiworld_uav.training.run --dataset data/datasets/<name> --name <run> --model b3 \
       --config config/training/default.yaml config/training/compare.yaml
```

`--model` is one of `b2`, `b3`, `b4`, `b4_rw` (B4 at B3 width), `b5a`, `b5b`, `b6b_future`, `b6b_current`. Add `config/training/known_vehicle.yaml` for the known-vehicle prior and `config/training/v7_openloop.yaml` for v7 training windows; `--resume` continues a run. Model sizes and other options are in `config/training/default.yaml` and `config/model/`. Checkpoints and metrics go to `outputs/experiments/<run>/`.

```bash
scripts/train_all_baselines.sh <dataset> <tag> 0,1,2           # all trainable baselines, spread over GPUs
python scripts/run_sweep.py --jobs outputs/sweeps/<sweep>/jobs.json --gpus 0,1,2 --per-gpu 3   # job lists: scripts/sweep_jobs_*.py
```

### Evaluate

Evaluation is open loop: each model gets the history window, starts from the true state and is rolled forward with the recorded actions. It reports position, velocity, attitude and angular-velocity error per rollout length for every role, and the reliable imagination horizon.

```bash
python -m physiworld_uav.evaluation.run --dataset data/datasets/<name> \
       --models b0 b1 b1_kv b6a_future b3=outputs/experiments/<run>/checkpoints/best.pt --horizon 20 --noise nominal --out outputs/eval/<name>.json
scripts/eval_baselines.sh <dataset> <tag>                      # all baselines trained with the same tag
python -m physiworld_uav.evaluation.analysis regime|anchor|residual|shuffle --dataset ... --models ...
```

The analyses cover the wind and no-wind gap, re-observation, residual diagnostics and a shuffled-action test. Output is JSON and Markdown in `outputs/eval/`.

### Visualize

```bash
python -m physiworld_uav.visualization --help          # all commands
```

| Command | Output |
|---|---|
| `generation` / `training` | live pages (generation progress; training curves and resources) |
| `report` | static HTML report of a dataset plan and index |
| `plan`, `wind` | distribution plots of a planned or generated dataset |
| `trajectories` | actual against predicted trajectories per episode, with the wind in the legend |
| `evaluation` | error against horizon, model by role heat maps, OOD gap from evaluation JSONs |
| `figure-data` | extracts what figures need from the HDF5 shards and checkpoints into `outputs/figure_cache` (container) |
| `figures` | all report and slide figures as pdf and png (`--only`, `--list`, `--formats`, `--copy-to`) |
| `tables` | LaTeX tables computed from the evaluation files and datasets |

```bash
python -m physiworld_uav.visualization figure-data                 # once per dataset / set of runs
python -m physiworld_uav.visualization figures --copy-to <folder>  # draw everything
python -m physiworld_uav.visualization.episode_plots --dataset data/datasets/<name> --episode <id>   # quick look at one episode
scripts/start_report_server.sh training <sweep>                    # live training dashboard
```

Details, options and how to add a figure are in `src/physiworld_uav/visualization/README.md`.

### Tests

```bash
python -m pytest tests/cpu -q                  # CPU suite: config, planning, wind, force equations, data, models
python -m pytest tests/isaac --isaac -q        # Isaac physics validation (needs a GPU)
```

Without the container, only the pure-Python parts (config, planning, wind, force equations, models, evaluation, plotting) are testable.

### Notes

* `SimulationApp.close()` ends the process, so the CLI validates and writes `index.json` before closing Isaac.
* Files created in the container are owned by root; fix with `chown <uid>:<gid> <path>`.
* The Isaac IMU needs the sensor created before `sim.reset()` and the `gpu_dynamics_cpu_readback` physics pipeline; the code already does this.
* `scripts/upload_to_gdrive.sh --dry-run` uploads datasets and outputs to Google Drive with rclone (run `rclone config` first).

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
3. On your laptop, open a tunnel: `ssh -L 6080:localhost:6080 <user>@<server-ip>`
4. Browse to `http://localhost:6080/vnc.html` and click **Connect**.

Do not expose port 6080 to the internet. The first launch builds the shader cache and can take several minutes.

## Documentation

* Architecture: `docs/guides/architecture.md`; configuration: `docs/guides/configuration.md`; scenarios and experiment types: `docs/guides/scenarios_and_experiment_types.md`
* Dataset schema: `docs/guides/dataset_schema.md`; reproducibility: `docs/guides/reproducibility_and_domains.md`; glossary: `docs/guides/glossary.md`
* Aerodynamic and drone models: `docs/models/aerodynamics.md`, `docs/models/drone_and_wind.md`, `docs/models/force_accounting.md`
* Assumptions and limitations: `docs/guides/assumptions_limitations.md`
* Measured results: `docs/results/`
* Visualization commands: `src/physiworld_uav/visualization/README.md`
