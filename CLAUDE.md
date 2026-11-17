# CLAUDE.md — PhysiWorldUAV

## What this is
M.Tech thesis code: a **physics-prior latent world model** for long-horizon quadrotor prediction under **out-of-distribution (OOD) wind**, ds/dt = f_phys(s,u) + d_θ(h,s,u).
This repo currently contains the **data-generation pipeline** (Isaac Sim 4.1 + OmniDrones, Hummingbird quadrotor). World-model code (`models/ training/ losses/ baselines/ evaluation/`) is out of scope until after Run 6; `training/train.py` is known-broken (imports a deleted module).

Work follows the six-run sequence in `docs/prompt.md`: (1) audit, (2) plan, (3) config/experiment architecture, (4) physical models + validation, (5) pilot dataset + quality inspection, (6) scale-up with parallel generation. See `HANDOFF.md` for the current position. Audit/plan/reports: `docs/audit/run{1..4}_*.md`.

## Architecture (`src/physiworld_uav/`)
Plan → simulate → write → validate. A manifest row fully determines an episode.

| Package | Role |
|---|---|
| `config/`, `config/*.yaml` (repo root `config/`) | Hydra groups: drone, physics, limits, controller, dataset, scenario, excitation, wind/regimes, domain, counterfactual, forces, experiment |
| `core/` | `simulation_interface` (Isaac wrapper, IMU, wrench), state layout/frames, seed tree, provenance |
| `drones/` | `DroneParams`; `sim_binding` (domain → PhysX, read-back, effective inertia) |
| `control/` | Control laws over OmniDrones Lee/Attitude/Rate controllers |
| `trajectories/`, `excitation/` | Reference library + feasibility; bounded excitation signals |
| `wind/` | OU, Dryden (MIL-F-8785C), gusts, regimes with realised-statistics guarantees |
| `randomization/` | Sobol/LHS physical-domain sampling, ID/OOD shells, held-out corners |
| `forces/` | Aero models (rotor drag, body drag with per-axis C_D, ground effect, rotational damping), applicability registry, `WrenchInjector` |
| `experiments/` | `plan.py`, counterfactual siblings, `runner.py`, CLI `generate.py` |
| `data/` | Schema v2, recorder, HDF5 writer, resampler, dataset builder/validator, splits |

More detail: `docs/guides/architecture.md`, `docs/models/*.md`, `docs/guides/assumptions_limitations.md`.

## Install / run / test
Isaac Sim 4.1 and OmniDrones are **not pip-installable**; everything runs inside the docker container `isaac-sim` (Isaac at `/isaac-sim`, repo mounted at `/workspace/PhysiWorldUAV`, 4× RTX A6000 on the host). `OmniDrones/`, `IsaacLab/`, `omnidrones_env/`, `lagacy/` are git-ignored and live only on the host/container.

```bash
docker exec isaac-sim bash -c 'cd /workspace/PhysiWorldUAV && source omnidrones_env/bin/activate && export PYTHONPATH=$PWD/src:$PYTHONPATH && <command>'
```
Commands (`<command>`):
```bash
python -m pytest tests/cpu -q                       # CPU suite (~100 s), 132 tests
python -m pytest tests/isaac --isaac -q             # Isaac validation (~2+ min, needs GPU), 17 tests
python -m physiworld_uav.experiments.generate plan     experiment=pilot
python -m physiworld_uav.experiments.generate run      experiment=smoke
python -m physiworld_uav.experiments.generate validate   --dataset data/datasets/<name>
python -m physiworld_uav.experiments.generate regenerate --dataset data/datasets/<name> --episode <id>
```
Without the container, only pure-Python parts (config, planning, wind, force equations) are testable, and only if `omni_drones`/`torch` import.

## Gotchas
* **Append** to `PYTHONPATH` (`export PYTHONPATH=$PWD/src:$PYTHONPATH`); overwriting hides Isaac's `isaacsim` and breaks `omni_drones` imports.
* `SimulationApp.close()` **kills the Python process** — finish validation/writing before closing. Isaac tests drive the CLI in a subprocess.
* Container-written files are root-owned: `chown 1003:1003`.
* Isaac IMU needs the sensor prim created **before** `sim.reset()` and the pipeline `gpu_dynamics_cpu_readback` (default GPU pipeline gives gravity-only frames). CLI rejects `gpu`.
* Reset rotor joint pos/vel between episodes (determinism).
* `.gitignore` entries `/data/` and `/outputs/` are anchored on purpose — `src/physiworld_uav/data/` is source code.

## Conventions and decisions
* Frames: world ENU, body FLU, quaternion Hamilton **wxyz** body→world. Simulator state p|q|v|ω_W; model state p|v|q|ω_B (ω in body frame everywhere in models).
* Physics dt 5 ms, decimation 4 → model Δt 20 ms. Action = `rotor_speed_mean`. IMU noise applied at **load time**, not stored.
* PhysX damping is 0; OmniDrones `drag_coef` must be 0 (asserted). Project aero wrench goes in through `WrenchInjector` only.
* Controllers use the **true** vehicle mass and effective inertia (decision Q2).
* Counterfactual Mode A = closed loop, Mode B = fixed-action replay. Splits are assigned per group; seeds via `SeedSequence` tree with stable keys.
* Force configs must be `validated: true` or the runner refuses them (`--allow-unvalidated-forces` is dev-only and such datasets never validate). Episodes outside any force-model envelope are excluded from model-ready data.
* Approved aero truth (`config/forces/truth_default.yaml`): rotor drag d = 0.30 s⁻¹ (linear in Ω, in-plane), body drag with **per-axis C_D = (1.1, 1.1, 1.3)** and projected areas (0.0070, 0.0076, 0.0180) m², R = 0.10 m, ρ = 1.225; rotor-drag envelope μ ≤ **0.20** (widened from 0.15; fidelity reduced 0.15–0.20); OOD wind (4, 8] m/s.
* Dryden frozen-turbulence speed = episode mean wind (not vehicle airspeed) so counterfactual pairs share realisations.
* Legacy code is in `lagacy/` (sic, ignored); superseded force models in `lagacy/src_superseded/`.

## How the user wants to work
* Do **not** silently resolve scientific ambiguities — present options with a recommendation and ask (esp. physical correctness, dataset validity, coefficients).
* Never fabricate results; report measured numbers and failures plainly.
* Stay within the current run's scope; don't start the next run until asked.
* Run everything inside the `isaac-sim` container.
* Prefer recommended options when the user says so; keep documents in `docs/` updated each run (reports, `assumptions_limitations.md`).
