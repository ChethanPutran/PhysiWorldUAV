# PhysiWorldUAV

Physics-prior latent world models for reliable long-horizon quadrotor prediction under out-of-distribution (OOD) wind. This is M.Tech thesis code.

The model has the form

```
ds/dt = f_phys(s, u) + d_θ(h, s, u)
```

where `f_phys` is a nominal rigid-body quadrotor model and `d_θ` is a learned residual.

**Current scope.** The repository contains the *data-generation pipeline*: Isaac Sim 4.1 and OmniDrones (Hummingbird quadrotor) produce physically validated, reproducible trajectories with IMU, controls, wind, domain parameters and counterfactual pairs. The world-model code (`models/`, `training/`, `losses/`, `baselines/`, `evaluation/`) is out of scope until the dataset is finished. `training/train.py` is known to be broken.

## Status

| Run | Scope | Status |
|---|---|---|
| 1 | Repository audit and scientific review | Done |
| 2 | Implementation plan | Done |
| 3 | Config and experiment architecture | Done |
| 4 | Physical models and validation in Isaac | Done (17 Isaac + 132 CPU tests pass) |
| 5 | Pilot dataset and quality inspection | In progress |
| 6 | Parallel scale-up | Not started |

See `HANDOFF.md` for the detailed checklist and `docs/prompt.md` for the six-run specification.

## How it works

```
config/ (Hydra groups) → experiments/plan.py → manifest.jsonl
        → experiments/runner.py (Isaac Sim + OmniDrones) → HDF5 shards
        → data/dataset_validator.py → index.json
        → data/dataset_builder.py (model-rate view: decimate, windows, future actions)
```

* **Plan, then simulate.** Every random choice (domain, initial condition, reference, excitation, wind, counterfactual sibling, split role) is made at planning time from a seed tree. An episode is a pure function of one manifest row.
* **Truth, observations and actions are stored separately.** IMU noise is applied at load time, not at generation.
* **Physics:** PhysX at 200 Hz, OmniDrones rotor model, plus project aerodynamics (rotor drag, body drag with per-axis C_D and orientation-dependent area, optional ground effect) injected as one wrench. Wind comes from steady, gust, OU and Dryden components.
* **Fail loudly.** The runner refuses unvalidated force models, wind without force models, and schema violations.

Conventions: world ENU, body FLU, Hamilton quaternion wxyz (body→world). Physics step 5 ms, model step 20 ms.

## Repository layout

| Path | Contents |
|---|---|
| `src/physiworld_uav/` | `core`, `drones`, `control`, `trajectories`, `excitation`, `wind`, `randomization`, `forces`, `experiments`, `data` |
| `config/` | Hydra groups: drone, physics, limits, controller, dataset, scenario, excitation, wind, domain, counterfactual, forces, experiment |
| `tests/cpu`, `tests/isaac` | Unit tests and Isaac physics-validation tests |
| `docs/guides/` | Architecture, configuration, environment, dataset schema, scenarios, assumptions and limitations |
| `docs/models/` | Force accounting, aerodynamics, drone and wind models, validation numbers |
| `docs/audit/` | Run reports and the implementation plan |
| `docker/`, `scripts/` | Container image and helper scripts |
| `CLAUDE.md`, `HANDOFF.md` | Working notes for AI-assisted development |

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
| `python -m pytest tests/cpu -q` | CPU tests (~100 s) |
| `python -m pytest tests/isaac --isaac -q` | Isaac physics validation (~4 min, needs a GPU) |
| `python -m physiworld_uav.experiments.generate plan experiment=pilot` | Write the manifest only |
| `python -m physiworld_uav.experiments.generate run experiment=smoke` | Generate a small dataset |
| `python -m physiworld_uav.experiments.generate run experiment=pilot` | Generate the pilot dataset |
| `python -m physiworld_uav.experiments.generate validate --dataset data/datasets/<name>` | Validate a dataset |
| `python -m physiworld_uav.experiments.generate regenerate --dataset data/datasets/<name> --episode <id>` | Reproduce one episode |

Hydra overrides go after the command, e.g. `experiment=pilot`. Datasets are written to `data/datasets/<id>_v<version>/` and are not committed.

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

* Architecture: `docs/guides/architecture.md`
* Configuration and scenarios: `docs/guides/configuration.md`, `docs/guides/scenarios.md`
* Dataset schema: `docs/guides/dataset_schema.md`
* Aerodynamic and drone models: `docs/models/aerodynamics.md`, `docs/models/drone_and_wind.md`
* Assumptions and limitations: `docs/guides/assumptions_limitations.md`
* Run reports: `docs/audit/`
