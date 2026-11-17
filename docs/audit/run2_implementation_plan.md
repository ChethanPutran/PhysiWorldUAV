# PhysiWorld UAV — Implementation Plan for the Run 1 Audit Findings

Date: 2026-11-16 · Input: [run1_repository_audit.md](run1_repository_audit.md) · Spec: `docs/prompt.md`, `docs/requirement.md`, `docs/datasets/problems/identifiability.md`, `docs/phase1/output.md` (design doc §10–§17)

This document explains **how** each audit finding will be fixed: what changes, where, by which method, how it is validated, and in which run of the six-run sequence.
- Section numbers mirror the audit (§0–§9).
- Every audit point has a stable ID; the traceability table in §12 maps every ID to a work item.
- Section 9 decisions use the recommended options. Where the audit listed options without marking one, this document picks one and says why.

**Status: plan only.** No code is changed by this document. Implementation starts in Run 3 after you approve.

---

## Contents

- [A. Run mapping and conventions](#a-run-mapping-and-conventions)
- [B. Changes since the audit (found while preparing this plan)](#b-changes-since-the-audit)
- [§9 first: settled decisions D1–D10](#9-settled-decisions-d1d10-recommended-options)
- [§0 Executive-summary items E1–E8](#0-executive-summary-items)
- [§1 Environment and versions](#1-environment-and-versions)
- [§2 Folder structure and module responsibilities](#2-folder-structure-and-module-responsibilities)
- [§3 Configuration architecture](#3-configuration-architecture)
- [§4 Data-generation pipeline](#4-data-generation-pipeline)
- [§5 Simulator physics accounting](#5-simulator-physics-accounting)
- [§6.1 Force and torque models](#61-force-and-torque-models)
- [§6.2 Wind models](#62-wind-models)
- [§6.3 Trajectories, controller and excitation](#63-trajectories-controller-and-excitation)
- [§6.4 Domain randomization](#64-domain-randomization)
- [§6.5 Dataset schema, splits and validation](#65-dataset-schema-splits-and-validation)
- [§6.6 Reproducibility](#66-reproducibility)
- [§6.7 Identifiability](#67-identifiability)
- [§6.8 Computational efficiency](#68-computational-efficiency)
- [§7 Tests](#7-tests)
- [§8 Missing vs redundant functionality](#8-missing-vs-redundant-functionality)
- [§10 File plan: create / modify / retain / deprecate](#10-file-plan)
- [§11 Documentation deliverables](#11-documentation-deliverables)
- [§12 Traceability matrix](#12-traceability-matrix)
- [§13 New decisions raised by this plan](#13-new-decisions-raised-by-this-plan)

---

## A. Run mapping and conventions

| Run | Scope | Exit criterion (must be demonstrated, not claimed) |
|---|---|---|
| **Run 2** (now) | Review this plan, the physics assumptions and the dataset schema together. Settle §13. | You approve this document (with edits). |
| **Run 3** | Configuration architecture, experiment-plan (manifest) generation, seeds and provenance, scenario/controller/excitation integration, storage schema skeleton, split-by-design, pytest setup, legacy cleanup. | A CPU-only test suite passes. A manifest for a pilot experiment is generated deterministically. A 1-episode Isaac run writes the new schema with complete metadata. |
| **Run 4** | Physical models: force injection, frames, mass/inertia, damping, motor reset, drag models, wind fixes, domain application to PhysX, applicability registry, physics-validation tests. | Every physics-validation test in §7 passes **inside Isaac Sim** with recorded numbers. |
| **Run 5** | Pilot dataset (about 1–2% of full volume across all data types and regimes). Validation report. Identifiability and decorrelation diagnostics. Storage benchmark. Profiling. | The validation report is green, or every failure is explained. You review the diagnostics. |
| **Run 6** | Multi-env and multi-GPU generation, benchmarks, full dataset generation, final report. | Full dataset passes automated validation; benchmark table recorded; final project review document. |

**Conventions in this plan:**
- **Item format:** each item lists *Action → Method → Files → Validation → Run*.
- **Frames:** world = ENU (z up); body = FLU; quaternion = Hamilton, `[w,x,y,z]`, body→world (verified to be Isaac's convention). R_WB maps body vectors to world.
- **Units:** SI everywhere. Rotor speed in rad/s.
- **Code style:** dataclasses with type hints. Reuse OmniDrones/Isaac functionality and do not reimplement it.
- **Physics validation:** physics code is validated with isolated analytic tests before full-dynamics tests.

---

## B. Changes since the audit

These were found while preparing this plan and are added as new items.

| ID | Finding | Action | Run |
|---|---|---|---|
| **N1** | `src/physiworld_uav/data/dataset.py:17` now reads `from ....lagacy.wind import generate_wind_sequence`. A relative import above the top-level package always raises `ImportError` (verified in the container). `dataset_validator.validate_primary_aux_separation` imports `dataset.py`, so **`assert_valid` and example0's dataset build are broken again.** | Move `SyntheticUAVDataset` (the only user of the legacy wind generator) into `lagacy/synthetic_dataset.py` and remove the import from `dataset.py`. Nothing in the live path uses `SyntheticUAVDataset`. **Proposed as the very first Run 3 commit; it is a one-line fix if you want it earlier.** | 3 |
| **N2** | Legacy code moved to `lagacy/`: `collect_data*.py`, `collect_data_wind_hover.{py,yaml}`, `control_library.py`, `dataset_generation.yaml`, `generate_dataset.py`, `utils.py`, `wind.py`. Old scripts and orphan configs are deleted. | Treat `lagacy/` as read-only reference. Add a `lagacy/README.md` stating what each file was and what replaces it. Exclude `lagacy/` from packaging and tests. | 3 |
| **N3** | `docs/requirement.md` gained four requirements (see the list below this table). | Each is addressed: experiment design → §6.7; three data types → §6.3/§6.5 (`data_type` metadata); applicability registry → §6.1-F11; truth/observation/action separation → §6.5 schema. | 3–5 |
| **N4** | `docs/phase1/output.md` is the design document the code cites (§10 counterfactual, §11 action decorrelation, §12 identifiability groups A–F, §13 hierarchy, §14 logging, §16 split experiments A–G, §17 ablations). | Kept as the design reference. Every one of its experiment groups is represented in the experiment plan (§4, §6.7). | 3–5 |
| **N5** | `docs/datasets/wind.md` lists Beaufort ranges: light breeze 6–11 km/h ≈ 1.7–3.1 m/s; gentle breeze 12–19 km/h ≈ 3.3–5.3 m/s; moderate breeze 20–29 km/h ≈ 5.6–8.1 m/s. | Used to justify the D1/D2 regime boundaries: ID 0–4 m/s ≈ calm to light/gentle breeze; OOD 4–8 m/s ≈ gentle to moderate breeze. Recorded in the wind documentation. | 3 |

**The four `requirement.md` additions (N3):**
1. An identifiability-driven experiment-design module.
2. Three distinct data types: closed-loop, open-loop and counterfactual.
3. A model applicability registry.
4. Separation of ground truth, observations and actions.

---

## 9. Settled decisions D1–D10 (recommended options)

These are applied throughout the plan. Numeric boundaries live in config, so they can be changed without code changes.

### D1 — Wind unit: **m/s of wind velocity**
- The thesis and `requirement.md` define the OOD split in wind velocity, and velocity is the controllable physical quantity.
- The disturbance *force* depends on the drag model and on relative airspeed, so it cannot define the regime. It is logged as auxiliary ground truth.
- **Inconsistency to fix:** `identifiability.md` writes "≤ 4 N". Run 3 adds a short erratum note at the top of that file stating that its N values should be read as m/s. I will not rewrite the document; the note goes in for your approval.

### D2 — ID/OOD statistic: **realised wind statistics computed from the logged wind, with both mean and peak constraints**
Definitions (horizontal speed U_h = ‖w_xy‖, vertical w_z, from the logged `wind_W` at the vehicle CoM over the episode):

| Regime | Rule (all must hold) |
|---|---|
| **ID (train / val_id / test_id)** | peak ‖w_t‖ ≤ 4.0 m/s; \|w_z\| ≤ 1.0 m/s ("limited vertical" per `requirement.md` §4); temporal class ∈ ID set; turbulence σ ≤ ID limit |
| **OOD-magnitude** | mean U_h ∈ (4, 8] m/s; peak ‖w_t‖ ≤ 8 m/s + gust allowance (configurable, default 10 m/s); same direction/temporal classes as ID; \|w_z\| ≤ 1 m/s |
| **OOD-direction** | elevation of the mean wind > 15° (vertical/diagonal flow), magnitude within the ID range |
| **OOD-temporal** | sinusoid f > 0.5 Hz, or sharp step changes, or OU τ < 1 s, with magnitude within the ID range |
| **OOD-gust/turbulence** | Dryden σ above the ID limit, or impulsive gusts (rise < 0.2 s), with magnitude bounds stated per experiment |
| **OOD-joint** | Two or more OOD factors combined |

Why these rules:
- Using **peak** for ID guarantees that no training sample ever sees |w| > 4 m/s. This prevents the leakage found in the audit (median peak was 8.5 m/s).
- Using **mean** for OOD-magnitude matches the thesis's "mean wind magnitude" wording.
- Regimes are **assigned at plan time** (before generation) from the sampled parameters. The validator then **re-checks them against the realised logged wind**, and any episode violating its designation fails validation (§6.5).
- **The OOD-temporal and OOD-gust numeric limits are proposed defaults** to be confirmed in §13.

### D3 — Control input u: **log all action representations; model input = rotor angular velocity Ω (post-motor-lag)**
- Thesis eq. 22 defines u_t = [Ω₁…Ω₄].
- In OmniDrones' `RotorGroup`, `throttle` ∝ Ω/Ω_max and thrust = KF·throttle². So Ω_i = throttle_i · Ω_max,i (Ω_max = 838 rad/s). This identity is verified against source in Run 4.
- Also logged:
  - the clamped rotor command `cmd ∈ [−1,1]` that was actually applied;
  - the raw (pre-clamp) controller output, with a saturation flag;
  - collective thrust and body-rate or torque commands;
  - reference and tracking errors.
- The model-ready dataset exposes `rotor_speed` as the primary action and keeps the others as alternative action fields (`requirement.md` §1.3 allows the thrust + torque alternative).

### D4 — Rates: **physics 200 Hz; control decimation 4 → model Δt = 0.02 s (50 Hz); states decimated, never filtered**
- The motor lag (≈ 8.9 ms) and rate-loop dynamics are resolved at 200 Hz.
- Holding the rotor command for 4 physics steps makes u_t exactly piecewise-constant over every model interval. This is the zero-order-hold assumption used by the RK4 transition in thesis eq. 28.
- 50 Hz is chosen over the existing 20 Hz (Δt = 50 ms, coarse relative to attitude dynamics with a rate-controlled inner loop) and over 100 Hz (2× more samples per second of flight, with a shorter horizon in seconds for the same step count).
- At 50 Hz the evaluation horizon of 100 steps = 2 s, and the training horizons 5–20 steps = 0.1–0.4 s.
- `control_decimation` is configurable (2 → 100 Hz, 10 → 20 Hz).
- **Risk:** the closed-loop rate controller now updates at 50 Hz. Run 4 validates closed-loop stability at 50 Hz. If tracking degrades, the fallback is decimation 2 (100 Hz). That fallback would be raised with you, not silently applied.
- Rotor speed still varies inside the interval because of motor lag. Ω is logged at 200 Hz, and the model-rate dataset stores both Ω at the interval start and the interval mean.

### D5 — Simulator ground-truth aerodynamics: **a minimal calibrated set with literature-based coefficients**

| Model | In sim ("truth") | In nominal f_phys | Notes |
|---|---|---|---|
| Rotor-induced drag (blade flapping + induced drag), linear D-matrix ∝ Ω̄ | **on** | off | Dominant small-quadrotor drag at low speed (Faessler et al. 2018; Kai et al. 2017; Bauersfeld et al. 2021 NeuroBEM) |
| Body parasitic quadratic drag, projected areas from the mesh | **on** | off | Dominant at higher airspeed |
| Rotational damping | via rotor-drag ω×r only | off | Separate `aerodynamic_torque` model **off** by default (it double-counts); kept for ablation |
| Ground effect (Cheeseman–Bennett) | off by default | off | Negligible at the ≥ 0.5 m flight altitudes; enabled only by near-ground scenarios |
| Wind-on-thrust (translational lift, axial inflow) | **not implemented** | — | Documented limitation. The current λ_wind/λ_DL formulas are physically inverted and are deleted (§6.1) |

- The residual d_θ then has to learn drag + wind + parameter mismatch, which matches thesis eq. 27.
- Coefficient values are a Run 4 deliverable presented for approval (§13-Q3). They will not be committed silently.

### D6 — PhysX damping: **zero it; model drag explicitly**
- OmniDrones sets `linear_damping = angular_damping = 0.2` via `RigidBodyPropertiesCfg`.
- Run 4 sets both to 0 through the drone's `rigid_props` before spawn. This is supported configuration, not a patch to OmniDrones.
- The value is recorded in metadata, and a test verifies it.

### D7 — IMU: **observation (model input); store clean sensor output; apply noise at load time**
- Remove `accel_imu`/`gyro_imu` from `AUXILIARY_KEYS` and place them under `observations/`.
- Store the clean Isaac `IMUSensor` measurement, which is a real sensor reading of specific force in the body frame.
- A configurable noise/bias model (white noise, per-episode constant bias, bias random walk) is applied in the data loader with a recorded seed. This lets noise levels be studied without regenerating data.
- An optional generation-time noise flag exists for completeness. It is off by default.

### D8 — Zero-wind baselines: **keep them as ID data inside their counterfactual group; also generate a dedicated physics self-consistency set (thesis Case A/B/C)**
- Zero wind lies inside ID [0, 4].
- Each counterfactual group, zero-wind sibling included, goes into exactly one split.
- Cases A/B/C are generated separately (data type `physics_validation`) and are excluded from training splits by default.

### D9 — Physics ranges

| Range | Spread around nominal | Use |
|---|---|---|
| Nominal | config values | — |
| ID | ±10 % | training domains |
| OOD physics-mismatch (H2) | magnitude 10–20 % (outside ±10 %, within ±20 %) | testing |

The 0.5–1.5× ranges are removed. Coupling rules (§6.4):
- **Inertia:** J = J₀ · (m/m₀) · s_J, with s_J ∈ [0.95, 1.05] (shape factor).
- **Torque constant:** KM/KF ratio ±5 %.
- **Feasibility:** T/W ≥ 2.0 enforced.
- **Motor lag:** τ_motor ±20 % in ID.
- **Drag:** coefficients ±10 % ID.

### D10 — Legacy cleanup: **legacy code deprecated to `lagacy/` (done by you); world-model code out of scope until after Run 6**
- `models/`, `training/`, `losses/`, `baselines/`, `evaluation/` are left untouched in Runs 3–6, except that `training/train.py`'s broken import of the deleted `physiworld_uav.physics` is recorded as a known issue.
- Run 3 provides a documented loader (`ModelReadyDataset`) for the new schema. Aligning the model code to it (13-D quaternion state, IMU input, no wind input) is a separate post-Run-6 task.

---

## 0. Executive-summary items

| ID | Audit summary point | How it is resolved | Detailed in | Run |
|---|---|---|---|---|
| E1 | No valid dataset produced | Re-architected pipeline (§4) plus a pilot (Run 5) with a validation gate that blocks marking a dataset complete | §4, §6.5 | 3–5 |
| E2 | Wind never acts on the drone | External wrench merged into OmniDrones' single `base_link` call via an injector, plus a failing-loudly guard and a 1 N lateral-force test | §5-A8/A10 | 4 |
| E3 | ω world-frame, mass 0.68 kg, domain params not applied, hidden damping | ω_B = R_WBᵀω_W at extraction; total articulation mass; domain applied via PhysX setters plus readback; damping set to 0 | §5, §6.4 | 4 |
| E4 | Outer loop unstable for \|ψ\| ≳ 33° | Replace with OmniDrones `LeePositionController` (yaw-aware SO(3)) and `AttitudeController` | §6.3-T1 | 3 |
| E5 | Dataset organisation mismatches the thesis | Model-rate dataset at 50 Hz with future actions, IMU as observation, truth/obs/action separation, splits by design | §6.5 | 3 |
| E6 | Physics models need correction | Geometry, rotor drag, body drag, wake, Dryden, sine refs | §6.1–6.3 | 3–4 |
| E7 | Dead/duplicate code and config | Legacy moved (done); stubs implemented or deleted; configs consolidated | §2, §3 | 3 |
| E8 | 54 steps/s, single-env wrapper | Profile, then vectorise (GPU-resident loop), then multi-env (GridCloner), then multi-GPU sharding | §6.8 | 5–6 |

---

## 1. Environment and versions

| ID | Item | Action / method | Validation | Run |
|---|---|---|---|---|
| V1 | Versions known but not recorded | `core/provenance.py` collects: Isaac Sim (`/isaac-sim/VERSION`), Kit, omni.isaac.core/sensor/physx extension versions, OmniDrones version + git commit, project git commit + dirty flag, torch/numpy/scipy/h5py versions, CUDA, GPU name, driver. Stored in every dataset's `provenance.json` and per-episode metadata (by reference). | Unit test: provenance dict has all keys; the git dirty flag is true when the tree is modified. | 3 |
| V2 | pytest missing | Install `pytest` (and `pytest-timeout`) into `omnidrones_env`. **This needs your OK because it modifies the venv** (§13-Q1). Add `pytest.ini` with markers `cpu` (default) and `isaac` (needs the simulator), with `isaac` tests skipped unless `--isaac` is passed. | `pytest -m cpu` runs in under 1 min. | 3 |
| V3 | `pyproject.toml` has no deps; bad entry point | Declare runtime deps already present in the venv (numpy, scipy, torch, omegaconf, hydra-core, h5py, pandas, psutil, pyyaml); isaac/omni_drones as documented extras (not pip-resolvable). Replace the entry point with `physiworld-generate = physiworld_uav.experiments.generate:main`. Fix the author field formatting. | `pip install -e . --no-deps` succeeds; the CLI entry point resolves. | 3 |
| V4 | Physics settings implicit; no determinism flags | Pass an explicit `sim_params`/physics config from the `physics/` config group: dt, substeps, solver type, position/velocity iterations, `enable_enhanced_determinism`, GPU pipeline flag, CCD. Read the effective values back after `sim.reset()` and record them. Solver iterations (OmniDrones' 4/0) are kept unless Run 4 tests show sensitivity. | Isaac test: recorded settings equal the config; a repeated identical episode is bitwise equal (§6.6). | 4 |
| V5 | SciPy only on path after venv activation | Document in `docs/guides/environment.md`; the generation CLI checks imports at start-up with an actionable error. | Startup check test. | 3 |

---

## 2. Folder structure and module responsibilities

### 2.1 Proposed structure (changes shown against today)

```
config/                               Hydra config groups (see §3)
docs/
  audit/                              run1_repository_audit.md, run2_implementation_plan.md (this), later run reports
  guides/                             NEW: architecture, configuration, environment, generation, validation, benchmarking
  models/                             NEW: one page per physical model (equations, assumptions, validity) → applicability registry source
  datasets/, phase1/, problems/       existing design notes (unchanged)
examples/                             example0 kept as a thin wrapper over the new CLI (or deprecated after Run 3, §13-Q5)
lagacy/                               read-only legacy reference (+ README.md)
src/physiworld_uav/
  config/        NEW  schemas.py (dataclass structured configs), compose.py (Hydra compose helpers), validation.py
  core/               simulation_interface.py (multi-env), state_extractor.py (ω_B, layouts), seeds.py NEW, provenance.py NEW
  drones/        NEW  params.py (DroneParams from yaml, cross-checked with OmniDrones), sim_binding.py (apply params to PhysX + controllers)
  control/       NEW  controllers.py (factory over OmniDrones Lee/Attitude/Rate), references_to_commands.py; replaces core/controller_adapter.py
  forces/             geometry fixed; rotor_drag rewritten (linear D-matrix); body_drag recalibrated; ground_effect.py (was rotor_wake);
                      aerodynamic_torque (ablation only); injection.py NEW (wrench into OmniDrones' base_link call); registry.py NEW
  wind/               Dryden/OU/seeding/temporal-scale fixes; regimes.py NEW (ID/OOD samplers)
  trajectories/       reference fixes; feasibility.py NEW; minimum-jerk primitives; GP jitter fix
  excitation/         signals.py NEW (step/pulse/doublet/multisine/chirp/PRBS); open_loop.py (rotor / collective+rate / wrench);
                      reference_perturbation.py (implemented); metrics.py (was excitation_metrics stub)
  randomization/      coupled samplers, qmc (LHS/Sobol), ID/OOD regions
  experiments/        plan.py NEW (manifest), runner.py NEW (episode execution), counterfactual.py NEW, generate.py NEW (CLI);
                      stubs ablation/identifiability/counterfactual_runner/evaluation_split replaced by the above
  data/               schema.py NEW, recorder.py NEW (GPU buffers), writer.py NEW (HDF5 shards), resampler.py (decimation),
                      dataset_builder.py (model-rate, future actions), splits.py NEW (moved out of builder), dataset_validator.py (extended),
                      metadata.py (extended), dataset.py (loader only), reports.py NEW, paired_dataset_builder.py → analysis/counterfactual.py
  analysis/           identifiability.py (FIM, sensitivity, conditioning), decorrelation.py, coverage.py, physics_consistency.py,
                      counterfactual.py — replacing current stubs
  models/ training/ losses/ baselines/ evaluation/ visualization/   untouched (D10)
tests/
  cpu/                unit + physics-model analytic tests
  isaac/              simulator integration + physics validation
  conftest.py, pytest.ini
```

### 2.2 Items

| ID | Audit point | Action | Run |
|---|---|---|---|
| S1 | Legacy data modules | Done by you (`lagacy/`). Remaining: N1 import fix; `lagacy/README.md`; delete `src/physiworld_uav/data/__pycache__` entries for removed modules. | 3 |
| S2 | Two live drivers (example0 vs hover script); action-length mismatch (N vs N+1) | The hover script is now in `lagacy/`. A single driver remains: `experiments/generate.py`. One schema convention: N+1 states, N actions (§6.5). | 3 |
| S3 | Duplicates: metadata ×2, splits ×2, four force-application paths, three drag definitions, duplicate UAV yaml | Metadata: only `data/metadata.py` (extended). Splits: only `data/splits.py`. Force application: only `forces/injection.py`; `_default_force_fn` and `apply_to_isaac_drone` are removed. Drag: only `forces/` (PhysX damping zeroed, D6). UAV yaml: the project yaml is the source of truth, and a start-up check asserts equality with OmniDrones' `hummingbird.yaml` on the shared keys (mass, inertia, KF, KM, rotor angles, directions, ω_max) and fails on mismatch. | 3–4 |
| S4 | Stubs (`analysis/*`, `experiments/*`, `excitation/{action_excitation,excitation_metrics,reference_perturbation}`, `core/episode_manager.py`) | Implemented where the plan needs them (§6.3, §6.7, §4). `core/episode_manager.py` is deleted (its role is `experiments/runner.py`). `action_excitation.py` merges into `excitation/signals.py` + `open_loop.py`. | 3, 5 |
| S5 | World-model code broken / different schema | D10: untouched; known-issue entry in `docs/guides/architecture.md`. | — |
| S6 | `tests/run_phase2.py` is a shell script with a `.py` name | Delete (or rename to `.sh` if still used; it is not referenced). | 3 |

---

## 3. Configuration architecture

### C1 — No composition or override mechanism → Hydra 1.3 + OmegaConf structured configs (both already installed)

**Reasoning:**
- Hydra gives defaults lists, config groups, experiment-level overrides, command-line overrides and multirun; prompt §3 asks for these.
- Validation uses dataclass structured configs registered in Hydra's `ConfigStore`. OmegaConf type-checks on composition. Cross-field rules (e.g. T/W ≥ 2, ID ranges inside OOD bounds) live in `config/validation.py`.
- Pydantic is installed but not used, to avoid a second schema system.

**Hierarchy:**
```
config/
  defaults.yaml                 # defaults list: drone, physics, forces, wind, domain, controller, excitation, dataset, experiment
  drone/hummingbird.yaml        # A. physical parameters (single source of truth; was uav_hummingbird.yaml)
  physics/isaac_200hz.yaml      # dt, solver, determinism, damping=0, gravity, control_decimation
  forces/
    truth_default.yaml          # D5 set: rotor_drag + body_drag on; ground_effect off; aero_torque off
    none.yaml, body_drag_only.yaml, rotor_drag_only.yaml, full_ablation.yaml
  wind/
    profiles/  none.yaml steady.yaml cosine_gust.yaml step.yaml sinusoid.yaml ou.yaml dryden.yaml random_gusts.yaml gradient.yaml
    regimes/   id.yaml ood_magnitude.yaml ood_direction.yaml ood_temporal.yaml ood_gust.yaml ood_joint.yaml
  domain/      nominal.yaml id_random.yaml ood_physics.yaml
  controller/  lee_position.yaml attitude.yaml rate.yaml
  scenario/    hover.yaml vertical.yaml translation.yaml line.yaml accel_decel.yaml circle.yaml figure_eight.yaml helix.yaml
               sinusoidal.yaml yaw.yaml roll.yaml pitch.yaml attitude_combined.yaml combined.yaml
               open_loop_rotor.yaml open_loop_collective_rate.yaml open_loop_wrench.yaml
  excitation/  none.yaml ref_smooth.yaml multisine.yaml chirp.yaml prbs.yaml
  initial_conditions/ hover_trim.yaml randomized.yaml
  counterfactual/ none.yaml wind_magnitude.yaml wind_direction.yaml mass.yaml inertia.yaml drag.yaml motor_tau.yaml aero_model.yaml
  dataset/     default.yaml     # H. name, version, rates, fields, storage, compression, validation settings
  experiment/  pilot.yaml train_id.yaml ood_magnitude.yaml ood_direction.yaml ood_temporal.yaml ood_gust.yaml
               ood_physics.yaml physics_validation.yaml identifiability_groups.yaml
```

**How an experiment uses templates:**
- An experiment file selects templates via its defaults list and overrides only what differs.
- It also defines a **mix** (fractions of scenario × excitation × wind-profile × domain cells) and counts. Example:

```yaml
# config/experiment/train_id.yaml
defaults:
  - /domain: id_random
  - /wind/regimes: id
  - /forces: truth_default
  - _self_
name: train_id
split_role: train            # train | val_id | test_id | val_ood | test_ood_*
n_domains: 40
mix:                          # fractions normalised by the planner
  - {scenario: hover,        controller: lee_position, excitation: ref_smooth, weight: 0.10}
  - {scenario: translation,  controller: lee_position, excitation: multisine,  weight: 0.25}
  - {scenario: open_loop_rotor, controller: none,      excitation: chirp,      weight: 0.10}
  # ...
counterfactual: {template: wind_magnitude, siblings: 3, fraction: 0.3}
```

- **Scenario overrides** go per mix entry (`overrides: {...}`).
- **Domain overrides** go via `domain:` group selection or inline ranges.
- **Reproducible randomisation:** all sampling happens in the planner with derived seeds (§6.6). Configs contain distributions, never sampled values; the manifest contains sampled values.

### C2 — `getattr`-on-dict bug in `paired_dataset_builder.py:43`
That module moves to `analysis/counterfactual.py` (§6.7). With structured configs, attribute access works for DictConfig, and the `cfg_get` helper (added last session) remains for plain dicts. A unit test checks every config-reading function accepts both `dict` and `DictConfig`. **Run 3.**

### C3 — File statuses

| Current file | Fate |
|---|---|
| `experiments.yaml` | split into the groups above; becomes `experiment/pilot.yaml` + group files |
| `uav_hummingbird.yaml` | → `drone/hummingbird.yaml` |
| `forces.yaml` | → `forces/truth_default.yaml` (+ ablation variants); `mass_kg` key removed (mass comes from the drone/domain only) |
| `wind.yaml` | content reviewed → `wind/profiles/*` + `wind/regimes/*` |
| `domain_randomization.yaml` | → `domain/*.yaml` |
| `trajectory.yaml` | → `scenario/*.yaml` |
| `config.yaml` | → `lagacy/` (world-model training config; mass 1.0 kg conflicts) |
| `controller.yaml`, `counterfactual.yaml`, `excitation.yaml` (0 bytes) | deleted; replaced by groups |

### C4 — Conflicting values: one source of truth per quantity

| Quantity | Single source | Rule |
|---|---|---|
| Mass, inertia, geometry, rotor constants | `drone/hummingbird.yaml` | Every other module reads `DroneParams`. A start-up check against OmniDrones' yaml (S3). |
| dt, control decimation | `physics/*.yaml` | The model rate is derived (`1/(dt·decimation)`); never specified twice. |
| Controller gains | `controller/*.yaml` | Default = OmniDrones' `lee_controller_hummingbird.yaml` values (reused, not copied: the config references the OmniDrones file and allows overrides). |
| Drag coefficients | `forces/*.yaml` | Domain randomisation scales them multiplicatively. |
| Target rate | removed from `experiment`; `dataset.model_rate_hz` is derived and validated against `physics` | |
| Seeds | one `seed` at `experiment` level | All others derived (§6.6). `domain_randomization.seed`/`domain.seed` removed. |

---

## 4. Data-generation pipeline

### New workflow (replaces example0's monolithic loop)

```
            ┌─────────────── Run-time independent ───────────────┐
compose cfg → validate cfg → PLAN (manifest) ──────────────────────┤
            │  domains (qmc) × scenarios × wind interventions     │
            │  × excitation × counterfactual siblings             │
            │  split designation per GROUP, seeds per stream      │
            │  → manifest.parquet-like table (CSV/HDF5 table)     │
            └──────────────────────────────────────────────────────┘
EXECUTE (per worker / GPU / env slot, deterministic assignment from manifest order)
  for each batch of episodes:
    bind domain → PhysX (masses, inertias, KF/KM, motor α, damping 0) + controller params
    reset (pose, velocities in correct frames, motor state = trim or specified)
    loop physics steps @200 Hz; every `decimation` steps: controller/open-loop/replay → rotor command (held)
       wind(t, p) → aero wrench (torch, batched) → injected into OmniDrones base_link call → step
       recorder appends GPU tensors (state, IMU, Ω, commands, wrench components, wind, refs)
    termination checks (safety envelope, ground, divergence) → status
  writer (async thread) → HDF5 shard; per-episode metadata
BUILD   episodes → model-rate arrays (decimate states, held actions, interval-mean Ω/IMU) → windows with future actions
VALIDATE integrity + physics consistency + designation checks + leakage + distributions → validation report
REPORT  per scenario/domain/dataset summaries (JSON + Markdown + plots)
```

| ID | Audit point | Action | Run |
|---|---|---|---|
| P1 | `domains[0]` only; hard-coded `TRAJECTORY_KIND_TO_SPEC`, interventions, excitation | The planner enumerates all domains and draws scenario/wind/excitation **from config groups**. There are no hard-coded specs in code. example0's tables become `scenario/*.yaml`. | 3 |
| P2 | Wind/IMU vs wind_force/thrust offset by one step | Single sampling convention (§6.5 schema). Signals that are *applied over* an interval (commands, aero wrench, wind used for the wrench) are stored with N rows indexed by interval k = [t_k, t_{k+1}). Signals *measured at* an instant (state, IMU, wind at t_k) have N+1 rows. The wind used to compute the wrench for interval k is the wind at (p_k, t_k), stored as `wind_applied_W[k]`. | 3–4 |
| P3 | Index 0 zero-filled IMU/thrust | IMU at k=0 is read after a settle step: before logging starts, the sim steps once with the trim command so the sensor holds a valid frame. Thrust/Ω at k=0 = initial motor state (trim). There are no zero fills; a validity mask per channel records any genuine gaps. | 4 |
| P4 | Forces held over the step (ZOH) | Kept and documented: the wrench is computed from the state at t_k and held for one physics step (5 ms). The error is O(dt·∂F/∂t) and is documented in the force-accounting guide. Recomputing at 200 Hz keeps it small relative to the 50 Hz model step. | 4 |

**Episode termination** (new): ground (z < 0.1 m) · tilt > 80° · |v| > 15 m/s · position outside the workspace · NaN. Status values: `completed`, `terminated_ground`, `terminated_envelope`, `terminated_divergence`, `failed_sim`. Failed episodes are kept on disk (flagged) and excluded from model-ready data by default.

---

## 5. Simulator physics accounting

| ID | Audit point | Action / method | Validation (Isaac test unless noted) | Run |
|---|---|---|---|---|
| A1 | Gravity by PhysX, correctly excluded | Keep. The force system has no gravity term at all (logging gravity added confusion); gravity is documented in force accounting. | Free fall with rotors at cmd = −1 (zero thrust) and damping 0: a_z = −9.81 ± 0.05 m/s² over 0.5 s. | 4 |
| A2 | Thrust t·KF; T_max 24 N | Keep (OmniDrones). Log per-rotor thrust T_i = KF_i·throttle_i² and Ω_i. Make T_max available to the feasibility checker. | Hover trim: Σ T_i = m_total·g within 0.5 %. | 4 |
| A3 | Reaction torque | Keep. Document sign: τ_z = −Σ dir_i·KM_i·throttle_i² (OmniDrones `moments·(−direction)`). Unit-check against our yaw-command sign. | Open-loop yaw doublet: yaw acceleration sign matches the convention; magnitude = τ_z/J_zz within 5 %. | 4 |
| A4 | Motor lag is a per-step blend α = 0.43 (dt-dependent) | Keep OmniDrones' `RotorGroup` (no duplication). Domain randomisation specifies a physical time constant τ_m and converts it to α = 1 − exp(−dt/τ_m) when binding the domain (written to `drone.rotor_params["tau_up"/"tau_down"]`). Record τ_m and α. Nominal τ_m = dt/(−ln(1−0.43)) ≈ 8.9 ms, keeping today's behaviour. | Open-loop rotor step: the measured throttle response matches 1−(1−α)^k exactly. | 4 |
| A5 | OmniDrones drag `+drag_coef·m·v` (anti-drag sign) | Keep `drag_coef = 0` and assert it at bind time. Never enable OmniDrones randomisation of drag (our drag lives in `forces/`). Document the sign issue. | Unit/Isaac assert. | 4 |
| A6 | PhysX damping 0.2 | **D6**: set `rigid_props.linear_damping = angular_damping = 0` before spawn. Read back via `RigidPrimView.get_linear_damping`/`get_angular_damping` (or the USD attribute) and record. | Test: with no wind and zero drag models, a coasting body (gravity disabled via a test-only flag) keeps constant velocity within 0.1 % over 2 s. | 4 |
| A7 | Rotor joint spin as real DOFs (gyroscopic coupling) | Keep (physically real: rotor angular momentum). Document it in force accounting. Measure its effect with an ablation test: OmniDrones sets joint velocities only if `rotor_joint_indices` exists. Compare roll-rate response with spin on vs zeroed. If the coupling is non-negligible, the nominal f_phys either includes the rotor-gyroscopic term or it is documented as part of the residual (§13-Q4). | Measured comparison table. | 4 |
| A8 | Project forces never applied | `forces/injection.py`: `WrenchInjector` wraps `drone.base_link` (composition, no subclassing, no edits to OmniDrones). It exposes `set_external(force_W, torque_W)` and forwards `apply_forces_and_torques_at_pos(forces, torques, …)` with `forces + F_ext`, `torques + τ_ext`, so there is **one call per step**. The injector is installed once after `drone.initialize()` by replacing `drone.base_link` with the proxy. If any expected attribute is missing it **raises** (no `hasattr` silence). External forces are set per step, and the injector clears them after forwarding. | (i) 1 N lateral force at hover → a_y = 1/m_total ± 2 %; (ii) wind on/off sibling episodes differ; (iii) unit test with a mock view that the proxy adds and does not overwrite. | 4 |
| A9 | `sim._mass` = base_link 0.68 kg; composite Izz 0.0130 vs mixer 0.012 | `drones/sim_binding.py` computes **total mass** = Σ link masses (`drone._view.get_body_masses()`), and **composite inertia** about the CoM by the parallel-axis theorem from link masses, inertias and poses (read from PhysX views). These are the *true* physical parameters recorded as ground truth. Controllers are built from the configured mass/inertia policy (§13-Q2). | Test: total mass = 0.716 ± 1e-4; composite inertia equals a hand calculation from USD values. | 4 |
| A10 | External-force API: GPU overwrite vs CPU accumulate | A8 makes the design correct under either semantics. Run 4 still **measures** it with a 2-call experiment (two separate 0.5 N calls vs one 1 N call) and documents the result in `docs/models/force_accounting.md`. | Measured result recorded. | 4 |
| A11 | ω world-frame mislabelled | `core/state_extractor.py`: `angular_velocity_B = quat_rotate_inverse(q, ω_W)`, computed batched on GPU. Both `angular_velocity_B` (model state) and `angular_velocity_W` (diagnostic) are stored. Reset converts a body-frame initial ω to world before `set_velocities`. All consumers (forces, IMU fallback, controllers) take explicitly named arguments (`omega_B`/`omega_W`). | Test: at a 90° yaw attitude spinning about body x, the logged ω_B = [r,0,0]; the gyro (IMU) agrees with ω_B within 1e-3 rad/s. | 4 |
| A12 | IMU validity gate ineffective; no noise | Validity = the frame's `physics_step` advanced since the last read and `time` ≈ sim time. Otherwise mark invalid in the mask, without re-using a stale frame. Noise per D7. Sensor at the base_link origin with identity offset: documented; lever-arm effects are zero by construction. | Test: forced stale read is flagged; the accelerometer at rest = [0,0,+9.81] ± 0.01; free fall ≈ 0. | 4 |
| A13 | No determinism flags; throttle not reset | Reset sequence per episode: `drone._reset_idx(env_ids, train=False)`, which sets throttle to the hover trim and clears `throttle_difference` (no randomisation is configured on the OmniDrones side). Then optionally set a specified initial motor state. Then set pose and velocities. Determinism per V4. | Test: two consecutive identical episodes → bitwise-identical state arrays (same env slot), or a documented tolerance if PhysX GPU is not bitwise-deterministic. | 4 |

**Force-accounting document** (`docs/models/force_accounting.md`, Run 4) contains these tables:

| Table | Content |
|---|---|
| 1 | Forces computed by the PhysX solver (gravity, contacts, articulation constraints, rotor-DOF gyroscopic) |
| 2 | Forces computed and applied by OmniDrones (thrust, reaction torque) |
| 3 | Forces computed by our modules (rotor drag, body drag, optional ground effect / aero torque) |
| 4 | Forces applied to the simulator (2 + 3 via the injector) |
| 5 | Forces recorded (per component + total, frames, sampling) |

It ends with the resulting equations of motion and a statement that nothing is double-counted.

---

## 6.1 Force and torque models

| ID | Audit row | Action | Run |
|---|---|---|---|
| F1 | Application no-op, `reshape(1,3)` | A8 injector, batched over envs (shapes `(N_env,3)`). Delete `apply_to_isaac_drone`. | 4 |
| F2 | Geometry on the wrong plane | `drones/params.py` builds rotor positions from yaml `arm_lengths`/`rotor_angles` in the **body X–Y plane**: r_i = (L_i cos θ_i, L_i sin θ_i, h_rotor), with h_rotor = 0.01 m from the USD. Thrust axis = +Z_B. The `HummingbirdGeometry` +Y convention is removed; geometry becomes generic `DroneParams` (multi-drone ready). Start-up check: computed rotor positions equal the PhysX rotor link poses (from `rotors_view.get_world_poses` at identity attitude) within 1 mm. | 4 |
| F3 | ω frame | A11; force models receive `omega_B` from the extractor. | 4 |
| F4 | Quaternion → R correct | Keep `_common.quat_to_matrix`. Add a property test (random quaternions: R orthonormal, det = 1, matches Isaac `quat_rotate`). | 3 |
| F5 | Body drag (formulation, placeholders) | See F5 details below this table. | 4 |
| F6 | Rotor drag (wrong physics) | See F6 details below this table. | 4 |
| F7 | Rotor wake / ground effect | See F7 details below this table. | 4 |
| F8 | Aerodynamic torque | Off by default (D5). Retained for ablation (Group E) with a frame-invariant form: τ_B = −K₁ω_B − k₂‖ω_B‖ω_B (scalar-magnitude quadratic). Input is ω_B only. The torque is passed to the injector in world frame via R_WB **once**; the double rotation is removed. | 4 |
| F9 | `wind_force_N` logs body drag only | Record every component (`F_rotor_drag_W`, `F_body_drag_W`, `F_ground_effect_W`, `τ_*_B`) plus `F_ext_total_W` and `τ_ext_total_B`. These are exactly what the injector applied (read from the injector buffer, not recomputed). | 4 |
| F10 | Domain `drag_coef` sampled but unused | Domain drag scales (`k_rd_scale`, `cd_scale`) are applied in the force system per env. Metadata records the effective coefficients. | 4 |
| F11 | **Applicability registry** (`requirement.md` #3, prompt §7F) | See F11 details below this table. | 4 |

**F5 — Body drag.**
- **Formulation:** F_B = −½ρ · C_D · A_proj(v̂_B) · ‖v_rel,B‖ · v_rel,B. The projected area for flow direction v̂ is A_proj(v̂) = A_x|v̂_x| + A_y|v̂_y| + A_z|v̂_z|, where A_x, A_y, A_z are the frontal, side and top projected areas. This replaces the component-wise |v_i|v_i form. The new form is direction-consistent (force anti-parallel to the relative velocity) and orientation-dependent through the projected area.
- **Parameters:** A_x, A_y, A_z are computed from the Hummingbird mesh (`sim/stl_parts`, `scripts/usd_to_stl.py`) by projecting the convex hull. C_D ≈ 1.0–1.2 (bluff body, Re ~ 10⁴–10⁵), cited and recorded.
- **Moments:** the CoP offset is computed from the projected-area centroid. If it is below 5 mm it is documented as zero.
- **Applicability:** quasi-steady, no separation hysteresis; airspeed ≤ 12 m/s; attitude-dependent through A_proj. Off-diagonal cross-axis aerodynamic terms (lift/side force) are not modelled, which is valid for a symmetric "+" frame; documented in the registry.

**F6 — Rotor drag (wrong physics).**
- **Replace with the linear rotor-drag model.** For each rotor i, with local relative air velocity v_rel,i,B = R_WBᵀ(v_W − w(p_i)) + ω_B × r_i, the drag is F_i,B = −k_rd · Ω_i · P_⊥ v_rel,i,B, where P_⊥ = diag(1,1,0) keeps only the in-plane component. The torque is τ_B = Σ r_i × F_i,B.
- Total force: F_B = Σ F_i,B. At hover attitude in uniform wind this gives F ≈ −4k_rdΩ_hover·v_⊥, i.e. the Faessler D-matrix form (d_x = d_y = 4k_rdΩ_h/m, d_z = 0).
- **Wind** is evaluated at each rotor position (the planner can enable spatial gradients and turbulence fields). Ω_i comes from the logged motor state (D3).
- **k_rd** is calibrated so that m·d_x falls in the published range for small quadrotors, scaled to the Hummingbird. The value comes with cited sources and is presented for approval (§13-Q3).
- **Applicability:** low to moderate advance ratio μ = V_⊥/(Ω R) ≲ 0.1–0.15. Beyond that, the registry flags episodes.

**F7 — Rotor wake / ground effect.**
- Rename to `forces/ground_effect.py`, keeping only Cheeseman–Bennett: T_IGE/T_OGE = 1/(1 − (R/4z_i)²), clamped for z_i/R ≥ 0.5 per rotor.
- It is applied as an additional force ΔT_i·ẑ_B at rotor i, injected, because thrust itself belongs to OmniDrones.
- It assumes constant power. Under fixed throttle this is documented as an approximation.
- Off by default; enabled only for `near_ground` scenarios. With R = 0.08 m the factor is 1.0016 at z = 0.5 m, so it is negligible for normal flight.
- **Deleted:** `λ_wind` (Glauert induced-velocity ratio misused as a thrust ratio) and `λ_DL` (sign-blind). Wind-on-thrust is documented as an unmodelled effect (D5).

**F11 — Applicability registry.**
- `forces/registry.py` holds one `ModelSpec` per model. Each spec lists:
  - the equations (link to `docs/models/<model>.md`);
  - assumptions, parameters with units and frames;
  - computational cost;
  - its validity envelope (airspeed range, advance ratio, tilt, height above ground, angular rate);
  - applicable and inapplicable scenarios;
  - a validation reference.
- **Runtime:** the recorder evaluates each enabled model's envelope per step and stores the fraction of time outside it per episode (`applicability_violation_frac` in metadata).
- **Validator:** if the fraction exceeds a configured threshold (default 5 %), the episode is flagged and excluded from model-ready data.
- **Planner:** a scenario template can declare required models, and the planner refuses incompatible combinations (e.g. ground effect enabled for altitude > 2 m is a no-op warning).
- The entries are filled from each model's equations, not assigned arbitrarily, as `requirement.md` asks.

**Isolated analytic tests** (CPU, `tests/cpu/test_forces_*.py`):
- **Body drag:** sign; anti-parallel to v_rel; v² scaling; rotation equivariance (R·F(v) = F(R·v) after rotating the body); terminal velocity v_t = √(2mg/(ρC_DA_z)) in a falling-body integration.
- **Rotor drag:** linear in v; zero axial component; ∝ Ω; zero torque in uniform wind at level attitude (Σr_i = 0); damping torque sign for rotation.
- **Ground effect:** limit → 1 at large z; value against a hand calculation.
- **Injector:** additive behaviour with a mock view.
- **Batching:** results for N envs equal N single evaluations.

---

## 6.2 Wind models

| ID | Audit row | Action | Run |
|---|---|---|---|
| W1 | 1-cos/step/sine/gradient correct | Keep. Add a property test per shape (peak value, timing, continuity class). The step gets an optional finite rise time (default 0 = ideal step) to support the temporal-OOD "sharp steps" class with controllable sharpness. | 3 |
| W2 | OU: starts at 0; isotropic | Initialise from the stationary distribution: x₀ ~ N(0, σ²). Anisotropic σ = (σ_h, σ_h, σ_v), configurable, default σ_v = 0.5σ_h. Exact AR(1) update kept. | 4 |
| W3 | Dryden | See W3 details below this table. | 4 |
| W4 | Shared seeding | See W4 details below this table. | 3 |
| W5 | `temporal_scale` skipped for stateful components | Implement it properly: temporal scale s multiplies the time axis. Stateless components evaluate at t·s. OU uses τ/s; Dryden uses L/s (equivalently V·s); random events use rate·s; sinusoid uses f·s. Test: the PSD corner frequency scales by s. | 4 |
| W6 | `direction_override` rectifies noise | Remove the override at the output stage. Direction is a **spec-level** parameter: steady and gust directions are sampled by the regime sampler, and turbulence is generated in a mean-wind-aligned frame (u along the mean wind, v lateral, w vertical) and rotated to world. Directional OOD is then a property of the sampled spec, not a post-hoc rotation. | 4 |
| W7 | Wind sampling in example0 violates the split | `wind/regimes.py` provides per-regime samplers implementing D2. Steady speed, direction (azimuth, elevation), gust type/peak/timing and turbulence σ are drawn from the regime config. The **ID sampler enforces peak ≤ 4 m/s by construction** (steady U + gust peak + 3σ_turb ≤ 4), with rejection on the realised peak after generating the wind time series at plan time (cheap: wind is computed offline from its seed). The realised statistics (mean, peak, RMS, elevation) are written to the manifest. | 3 |
| W8 | `config/wind.yaml` not consumed | Its useful content is migrated into `wind/profiles/*`, and the file is deleted. | 3 |
| W9 | Counterfactual siblings don't share realisations | See W9 details below this table. | 3 |

**W3 — Dryden.**
- **Variance:** replace the fixed-point Lyapunov iteration with the **exact stationary variance** (closed form for these first- and second-order filters), cross-checked against `scipy.linalg.solve_discrete_lyapunov`.
- **Discretisation:** use the exact ZOH (`scipy.linalg.expm` of the augmented matrix, i.e. Van Loan's method for the noise covariance) for all three axes.
- **Initialisation:** from the stationary distribution.
- **Low-altitude parameters (MIL-F-8785C / MIL-HDBK-1797):**
  - L_w = h;
  - L_u = L_v = h/(0.177 + 0.000823h)^1.2, in feet internally, converted to SI;
  - σ_w = 0.1·W₂₀, σ_u/σ_w = σ_v/σ_w = 1/(0.177 + 0.000823h)^0.4.
  - h is clamped to ≥ 10 ft (3.05 m) as the specification requires. Using it at 1–2 m altitude is documented as an extrapolation.
- **Frozen-turbulence speed:** V = max(‖w_mean − v_vehicle‖, V_min = 1 m/s), so it stays valid near hover. It is recomputed per step from the realised relative airspeed, which makes the filter time-varying. Its coefficients are updated per step at a cost of 3 small `expm`s, or approximated with a cached table over V; the choice is benchmarked.
- **Tests:**
  - ensemble σ within 5 % of the target for T ∈ {0.5, 2, 40} s;
  - PSD matches the analytic Dryden spectrum within ±15 % over 0.01–10 Hz;
  - stationarity from t = 0.

**W4 — Shared seeding.**
- `WindField.reset_all` takes a `SeedSequence` and **spawns one child per component**.
- Per-component seeds are recorded in the manifest.
- Test: two OU components in one spec have correlation ≈ 0 (|ρ| < 0.05 over 10⁴ samples), and the same seed reproduces bitwise.

**W9 — Counterfactual siblings.**
The counterfactual module defines sibling types:
- **Wind magnitude:** same realisation seed and direction, amplitude scaled. Every component is multiplied by the factor, keeping the temporal structure.
- **Wind direction:** same realisation rotated about z.
- **Wind type:** different spec (current behaviour, kept as a separate type).
- **Wind on/off:** zero-wind sibling.

The type and the exact parameter difference are stored per sibling (§6.3-T12).

---

## 6.3 Trajectories, controller and excitation

| ID | Audit row | Action | Run |
|---|---|---|---|
| T1 | Outer loop ignores yaw | See T1 details below this table. | 3 |
| T2 | Thrust law (no tilt compensation, z ref ignored) | Resolved by T1: Lee computes thrust = −m·a_cmd·(R e₃) with full 3-D position tracking (z_ref used). For attitude-mode scenarios the collective thrust is m(g + a_z,cmd)/(cosφ cosθ), clamped to [0.1, 0.9]·T_max. | 3 |
| T3 | Euler-error → body-rate mapping | Superseded by OmniDrones controllers (SO(3) attitude error). Where body rates are commanded directly (open-loop collective+rate), the command is already in body rates. | 3 |
| T4 | `yaw_sine`/`attitude_sine` resampled every step | Sample amplitude, frequency and phase **once per phase** and store them in `pstate`, with a seed drawn from the scenario's trajectory stream. Test: the output is an exact sinusoid; parameters are constant over the phase. | 3 |
| T5 | Anchoring/continuity | See T5 details below this table. | 3 |
| T6 | Attitude/yaw/thrust references drift (no horizontal hold) | Attitude scenarios fly on `AttitudeController` with a slow outer position loop (configurable `hold_position: true`, default true for closed-loop attitude scenarios). Free-drift variants are kept only as explicit open-loop scenarios with the envelope terminator. | 3 |
| T7 | GP sampler numerics | See T7 details below this table. | 3 |
| T8 | Parametric `trajectory_generator` key mismatch | `trajectory_generator.py`, `trajectory_types.py` and `trajectory_validator.py` are superseded by `ReferenceManager` + `feasibility.py`. Useful generators (helix, sinusoidal) are ported into `position_references.py`; the old modules are deleted after their tests are ported. | 3 |
| T9 | `random_action` crashes (299/300) | Removed. Replaced by **(a)** stabilised excitation (T10) and **(b)** true open-loop experiments (T11) that start from hover trim, are short (2–5 s), and stop at the safety envelope. | 3 |
| T10 | Closed-loop excitation missing | See T10 details below this table. | 3 |
| T11 | True open-loop excitation missing | See T11 details below this table. | 3–4 |
| T12 | Counterfactual design (output.md §10) | See T12 details below this table. | 3 |
| T13 | No feasibility checks | See T13 details below this table. | 3 |
| T14 | Basic/complex/attitude scenario coverage (prompt §5) | See T14 details below this table. | 3 |

**T1 — Outer loop ignores yaw.**
- Replace `core/controller_adapter.py` with `control/controllers.py`, a factory over **OmniDrones controllers** (no reimplementation):
  - `LeePositionController` for position/velocity/acceleration/yaw tracking (SO(3), yaw-aware, tilt-projected thrust, acceleration feed-forward; Hummingbird gains in `cfg/lee_controller_hummingbird.yaml`);
  - `AttitudeController` for attitude scenarios;
  - `RateController` for collective + body-rate commands.
- Controllers are constructed per domain from `DroneParams` per the mass/inertia policy (§13-Q2) and batched over envs.
- Validation:
  - **CPU:** the controller maps a level hover state to the trim command.
  - **Isaac:** hover-hold at ψ ∈ {0°, 45°, 90°, 180°} with a 1 m step reference converges with < 5 % overshoot difference across yaws.

**T5 — Anchoring and continuity.**
- All position references are expressed relative to the episode's initial state and start **at** the initial state with zero velocity and acceleration error. Circle and figure-eight start at their curve point nearest the initial position, with a minimum-jerk blend-in of configurable duration (default 2 s).
- `line` takes a configurable direction (azimuth/elevation sampled from the trajectory stream).
- `accel_decel` is fixed with a trapezoidal velocity profile smoothed by a minimum-jerk ramp (accelerate → cruise → decelerate → hold).
- Step-type references (attitude/yaw steps) remain intentional discontinuities but go through a 2nd-order reference prefilter (configurable bandwidth) so commanded rates stay feasible. Raw steps are a documented option.

**T7 — GP sampler numerics.**
- Relative jitter ε·mean(diag K) with ε = 1e-6, escalating ×10 up to 1e-3 on failure. Fall back to eigendecomposition sampling (clip negative eigenvalues), recording which path was used.
- Fix the prior σ (each of the 3 components gets variance σ²/3, so the total is σ²).
- Anchor the mean to the initial position.
- Compute only the needed Cholesky.
- Tests: the existing 4 failing tests pass, plus a conditioning test.

**T10 — Closed-loop excitation (prompt §11.2; output.md §11.2–11.3, experiments A–F).**
`excitation/signals.py` implements bounded signals:
- step, pulse, doublet;
- multisine: Schroeder phases, configurable band, low crest factor;
- linear and log chirp;
- PRBS with a configurable clock;
- smooth random: band-limited noise via a filtered OU.

`excitation/reference_perturbation.py` adds these to position, velocity or yaw references within configured amplitude limits. The excitation experiment groups A–F from output.md §11.3 are encoded as `excitation` + `mix` entries, and each episode records `excitation_group`.

**T11 — True open-loop excitation (prompt §6B; identifiability.md D).**
`excitation/open_loop.py` provides three **distinct abstractions**, recorded as `open_loop_kind`:
1. **`rotor_command`:** per-rotor or mixer-space (collective, roll, pitch, yaw) perturbations around the hover trim, at the actuator command level. There is no feedback of any kind. Signals: constant, step, pulse, doublet, chirp.
2. **`collective_rate`:** collective thrust + body-rate commands through OmniDrones' `RateController`. The inner **rate loop is closed**, so this is labelled "rate-stabilised open-loop", not pure open-loop. Signals: steps, pulses, chirps on rate and collective.
3. **`body_wrench`:** rotors held at trim and an external wrench (force/torque) applied through the injector. This is **not a physical UAV input**. It is for identification-oriented experiments only (m, J, drag separation), off by default, and excluded from model-ready data unless explicitly selected.
- All start from hover equilibrium (trim from `DroneParams`) unless overridden. Durations are short (2–5 s), and the envelope terminator applies.
- These experiments carry `data_type = open_loop` (`requirement.md` #2B).

**T12 — Counterfactual design.**
- `experiments/counterfactual.py` defines **Mode A** (closed loop: same IC, same reference, same excitation seed, same initial motor state; one variable changed) and **Mode B** (fixed action: replay the baseline's **applied, clamped rotor commands** at the control rate, from the identical initial state and motor state).
- **Variable types:** wind magnitude (scaled realisation), wind direction (rotated), wind on/off, mass, inertia, drag scale, motor τ, aerodynamic model set.
- Exactly one variable class changes per group (enforced by the planner, which diffs the sibling specs). The diff is stored as `counterfactual_diff: {path: [baseline, sibling]}`.
- **Identical random disturbances:** the turbulence seed and excitation seed are shared across siblings.
- All siblings share `counterfactual_group_id` and are assigned to one split.
- Mode B baselines run first (planner phase ordering), so replays have their action sequences.

**T13 — Feasibility checks.**
`trajectories/feasibility.py`, run at plan time on every reference (sampled at 200 Hz), checks:
- required thrust from differential flatness: T = m‖a + g e₃‖ ≤ 0.85·T_max and ≥ 0.15·T_max;
- tilt ≤ max_tilt;
- body rate and angular acceleration within limits (from jerk and snap via flatness);
- velocity, acceleration and jerk within limits;
- workspace bounds.

Infeasible references are resampled (logged). Plan-time feasibility uses the nominal parameters plus margin. Run-time saturation statistics are recorded per episode (`saturation_frac`).

**T14 — Scenario coverage (prompt §5).**
Scenario templates cover:
- **Basic:** hover; vertical ascent/descent; forward/backward/lateral translation; straight line; constant velocity; accel/decel; stationary attitude change.
- **Complex:** circle, figure-eight, sinusoidal, helix, combined translation + yaw, simultaneous position/orientation/velocity change.
- **Attitude:** pure roll/pitch/yaw, combined, transitions, angular-acceleration profiles.
- **Initial/terminal conditions:** start/end position and orientation, initial linear and angular velocity, terminal velocity/acceleration, desired angular velocity are all configurable per template via `initial_conditions/` and the template's terminal block.
- identifiability.md's six modes map 1:1 onto template groups, with the mode stored as `mode` in metadata.

---

## 6.4 Domain randomization

| ID | Audit row | Action | Run |
|---|---|---|---|
| R1 | Not applied to sim | See R1 details below this table. | 4 |
| R2 | No coupling | D9 coupling rules in `randomization/parameter_sampler.py`: sample m; J = J₀(m/m₀)s_J; KF; KM = KF·(KM₀/KF₀)·s_ratio; τ_m; drag scales. Reject if T/W < 2.0 or any feasibility rule fails. All rules live in config, not code. | 3 |
| R3 | Uniform only; no LHS/Sobol; no ID/OOD separation; one domain | See R3 details below this table. | 3 |
| R4 | Metadata records parameters not simulated | Metadata stores both `domain_sampled` (planner) and `domain_realised` (PhysX readback + bound controller params). The validator fails on a mismatch beyond tolerance. | 4 |

**R1 — Applying domains to the simulator.**
`drones/sim_binding.py::bind_domain(drone, controllers, domain, env_ids)` reuses OmniDrones' own mechanisms:
- **Mass:** `base_link.set_masses(m_base)` with m_base = m_total − Σm_rotor. Rotor masses are kept fixed; this is documented.
- **Inertia:** `base_link.set_inertias(J_base)`, solved so that the **composite** inertia equals the sampled J via the parallel-axis theorem.
- **Rotor constants:** KF and KM written into `drone.rotor_params` (as OmniDrones' `_randomize` does).
- **Motor lag:** α written into `drone.rotor_params` (from τ_m, A4).
- **Drag and controller:** drag scales go to the force system; controller parameters are updated per §13-Q2.
- **Readback:** after binding, masses and inertias are read back from PhysX and the composite values recomputed; these realised values are what metadata records.

**R3 — Sampling.**
- `scipy.stats.qmc`: scrambled Sobol (default) or LHS over the unit hypercube, mapped through each parameter's distribution (uniform, normal, log-normal, truncated normal, custom via inverse CDF).
- ID and OOD regions are defined in config:
  - **ID:** within ±10 %.
  - **OOD:** the shell 10–20 %, with sign sampling, plus an "unseen combinations" region: each parameter inside its ID range, but the joint point outside the ID convex hull in a held-out corner (e.g. heavy + low-KF). This tests combinational OOD.
- OOD domains are generated with a separate seed branch before any training domain is drawn.
- Coverage metric: minimum pairwise distance and discrepancy (`qmc.discrepancy`) reported.

---

## 6.5 Dataset schema, splits and validation

### Schema (`data/schema.py`, single source; versioned `schema_version = 2.0`)

Requirement: separate ground truth, observations and actions (`requirement.md` #4).

**Episode storage:**
- One HDF5 group per episode inside per-worker **shard files** (h5py available). The format is final only after the Run 5 benchmark (DS-STORE).
- Compression: gzip-4 or lzf, chunked by time.

**Sampling semantics:**
- k = 0…N indexes physics steps at 200 Hz; t_k = k·dt.
- "inst" means sampled at t_k after the step k−1 → k: N+1 rows.
- "intv" means held or applied over [t_k, t_{k+1}): N rows.

| Group / field | Shape | dtype | Unit | Frame | Sem. | Notes |
|---|---|---|---|---|---|---|
| `time/t` | (N+1,) | f64 | s | — | inst | |
| `ground_truth/position_W` | (N+1,3) | f32 | m | world ENU | inst | base_link origin (CoM offset ≈ 0.5 mm, documented) |
| `ground_truth/orientation_wxyz` | (N+1,4) | f32 | — | body→world | inst | unit-norm, sign-continuous |
| `ground_truth/velocity_W` | (N+1,3) | f32 | m/s | world | inst | |
| `ground_truth/angular_velocity_B` | (N+1,3) | f32 | rad/s | body FLU | inst | **thesis state ω** |
| `ground_truth/angular_velocity_W` | (N+1,3) | f32 | rad/s | world | inst | diagnostic |
| `ground_truth/linear_acceleration_W` | (N+1,3) | f32 | m/s² | world | inst | central difference of PhysX velocity (derived; flagged) |
| `ground_truth/angular_acceleration_B` | (N+1,3) | f32 | rad/s² | body | inst | derived |
| `ground_truth/rotor_speed` | (N+1,4) | f32 | rad/s | rotor | inst | Ω_i = throttle_i·Ω_max (post-lag) |
| `ground_truth/rotor_thrust` | (N+1,4) | f32 | N | +Z_B | inst | KF·throttle² |
| `ground_truth/wrench/<component>_force_W` | (N,3) | f32 | N | world | intv | rotor_drag, body_drag, ground_effect, aero_torque(0) |
| `ground_truth/wrench/<component>_torque_B` | (N,3) | f32 | N·m | body | intv | |
| `ground_truth/wrench/external_force_W`, `external_torque_B` | (N,3) | f32 | N, N·m | | intv | exactly what was injected |
| `ground_truth/wrench/rotor_force_B`, `rotor_torque_B` | (N,3) | f32 | | body | intv | total thrust and reaction torque (OmniDrones) |
| `observations/imu_accel_B` | (N+1,3) | f32 | m/s² | body (sensor = body) | inst | specific force, clean |
| `observations/imu_gyro_B` | (N+1,3) | f32 | rad/s | body | inst | clean |
| `observations/imu_valid` | (N+1,) | bool | | | inst | |
| `actions/rotor_speed_cmd`… see note | | | | | | |
| `actions/rotor_cmd` | (N,4) | f32 | [−1,1] | — | intv | applied (clamped) command, held over the control interval |
| `actions/rotor_cmd_raw` | (N,4) | f32 | — | — | intv | pre-clamp controller output |
| `actions/saturated` | (N,4) | bool | | | intv | |
| `actions/collective_thrust_cmd` | (N,) | f32 | N | +Z_B | intv | |
| `actions/body_rate_cmd` / `body_torque_cmd` | (N,3) | f32 | rad/s / N·m | body | intv | when applicable |
| `control/reference/{position_W,velocity_W,acceleration_W,yaw}` | (N,·) | f32 | | world | intv | closed loop |
| `control/tracking_error/{position,velocity,attitude}` | (N,·) | f32 | | | intv | |
| `environment/wind_W` | (N+1,3) | f32 | m/s | world | inst | wind at CoM |
| `environment/wind_applied_W` | (N,3) | f32 | m/s | world | intv | wind used for the wrench |
| `environment/wind_rotor_W` | (N,4,3) | f32 | m/s | world | intv | optional (spatial fields) |
| `environment/wind_components/<name>` | (N+1,3) | f32 | m/s | world | inst | steady/gust_i/turbulence |
| `environment/air_density` | scalar attr | | kg/m³ | | | |
| `validity/{applicability_violation, envelope}` | (N,·) | bool | | | intv | |
| attrs `metadata` | JSON | | | | | §6.5-DS10 |

Note on actions: rotor speed is a *state of the motor*, so the action the model consumes is defined at build time as `rotor_speed` (D3) derived from `ground_truth/rotor_speed`. The raw actuator command fields are `actions/*`. Run 2 confirms whether a separate `actions/rotor_speed_cmd` (the steady-state Ω target of the command, √((cmd+1)/2)·Ω_max) is also wanted (§13-Q6).

### Model-ready dataset (built at 50 Hz; D4)

- **Rate:** model step j corresponds to physics step k = 4j.
  - States are **decimated** (taken at k = 4j; no filtering).
  - Actions are taken at k = 4j; they are exactly constant over the interval by construction.
  - `rotor_speed` at k = 4j and the interval mean are both available.
  - IMU at k = 4j and the interval mean are both available.
- **Primary sample** (`ModelReadyDataset.__getitem__`):

  | Field | Shape |
  |---|---|
  | `state_history` | (H, 13), layout **p, v, q, ω_B** |
  | `obs_history` | (H, 6), IMU |
  | `action_history` | (H, 4) |
  | `future_actions` | (P, 4) |
  | `future_states` | (P, 13) |
  | `dt` | scalar |

- **Layout record:** the layout, H, P, rate and the field list are stored in the dataset `index.json`.
- **Windowing:** windows are built in the loader from contiguous episode arrays (no materialised copies), with stride configurable. This avoids the current H×P duplication of memory.
- **Auxiliary access:** `AuxiliaryView` gives wind, wrench and parameters for evaluation only, keyed by (episode_id, step).

| ID | Audit row | Action | Run |
|---|---|---|---|
| DS1 | Built at 200 Hz | Model-rate build at 50 Hz (above). | 3 |
| DS2 | Resampler: actions unresampled; filtfilt on states | `resampler.py` becomes `decimate_episode` (states decimated, actions held, interval means for Ω/IMU). The Butterworth/filtfilt path is retained only as an optional `observations` low-pass for sensor studies, never applied to ground-truth states. | 3 |
| DS3 | IMU auxiliary | D7: moved to `observations`. `AUXILIARY_KEYS` redefined as wind + wrench + parameters + counterfactual ids. The primary/aux separation test is updated accordingly. | 3 |
| DS4 | Future actions missing | `future_actions` (P,4) in every sample. | 3 |
| DS5 | `auxiliary.npz` unmappable | Removed. Auxiliary data stays in the episode files and is accessed via `AuxiliaryView` with explicit keys. | 3 |
| DS6 | State layout undocumented/inconsistent | The canonical model layout p\|v\|q\|ω_B is defined once in `data/schema.py`; the simulator layout p\|q\|v\|ω_W is used only inside `core/`. The dataset index records the layout; a test asserts consistency. | 3 |
| DS7 | Action semantics | D3 fields above. | 3–4 |
| DS8 | Splits broken | See DS8 details below this table. | 3 |
| DS9 | Episode validation disabled | See DS9 details below this table. | 3–5 |
| DS10 | Metadata incomplete | See DS10 details below this table. | 3 |
| DS11 | World-model code mismatch | D10 (out of scope); the loader documents the schema the model code must adopt. | — |
| DS-STORE | Storage format (prompt §12) | See DS-STORE details below this table. | 5 |
| DS-SUM | Experiment summaries (prompt §12) | See DS-SUM details below this table. | 5 |
| DS-DIST | Distribution validation (prompt §13D) | Plots and statistics of state, action, parameter, wind and trajectory distributions per split. KS / energy-distance between train and each OOD split, verifying that the intended ID/OOD differences exist and are confined to the intended factor. For example, OOD-magnitude should differ in wind magnitude but not significantly in trajectory statistics. | 5 |

**DS8 — Splits (design-time).**
- Splits are assigned **at plan time** to whole groups. A group is a counterfactual group, or for unpaired episodes a (domain, scenario) unit. Holdouts are generated from separate parameter regions (D2, D9), before training generation, with disjoint seed branches.
- Split roles:
  - `train`, `val_id`, `test_id`, drawn from the ID regime;
  - `val_ood_<cat>`, `test_ood_<cat>` for cat ∈ {magnitude, direction, temporal, gust, physics, trajectory, joint};
  - `physics_validation` (D8).
- `data/splits.py` keeps only *post-hoc* functions for the sample-efficiency subsets (10/25/50/100 % of train, nested and group-aware).
- The fixed bugs `_episode_peak_wind` and `unseen_wind`/`unseen_gust` are subsumed: designation is metadata, and the realised wind is verified by the validator.
- Empty-split guards: the planner computes per-split group counts and refuses plans that would leave any requested split empty.

**DS9 — Episode validation.**
`dataset_validator.py` is rewritten into checks run per episode and per dataset.
- **Integrity:**
  - shapes, dtypes and NaN/Inf;
  - quaternion norm |‖q‖−1| < 1e-4 and sign continuity;
  - timestamps strictly increasing with Δt = dt ± 1e-9;
  - duplicates (hash of the state array);
  - label validity: enums, split role, data type;
  - parameter validity (positive mass, positive-definite J satisfying the triangle inequality, T/W ≥ 2).
- **Physics consistency:**
  - ‖(p_{k+1}−p_k)/dt − ½(v_k+v_{k+1})‖ below a tolerance derived from the integrator;
  - the same for velocity vs acceleration;
  - quaternion kinematics q̇ = ½ q ⊗ [0, ω_B];
  - IMU specific force ≈ R_WBᵀ(a_W + g e₃) and gyro ≈ ω_B;
  - Newton check: m·a_W ≈ F_rotor_W + F_ext_W + m g (residual statistics per episode; this directly verifies force accounting);
  - actuator bounds;
  - envelope and applicability flags.
- **Designation checks:** realised wind statistics match the split role (D2); realised domain parameters lie within the declared region.
- **Leakage:**
  - no episode, seed, scenario instance or counterfactual group shared across splits;
  - no OOD parameter or wind value inside the training regions;
  - near-duplicate trajectories across splits (cosine similarity of downsampled state trajectories > threshold);
  - overlapping segments impossible by construction (whole episodes).
- **Gate:** a dataset is marked `complete` in `index.json` **only** if all hard checks pass. Soft checks are reported.
- Tolerances are calibrated in Run 4 from clean physics runs and recorded in the validation config.

**DS10 — Metadata.**
`EpisodeMetadata` is extended. Per prompt §11 and §12 it carries:
- dataset id/version and experiment id;
- trajectory (episode) id, scenario id, domain id;
- physical-config id, wind-config id, force-model-config id, controller id;
- initial and terminal conditions;
- sample rate, dt, duration;
- all stream seeds;
- realised and sampled parameter values;
- split role, ID/OOD designation and category;
- `data_type` ∈ {closed_loop, open_loop, counterfactual, physics_validation};
- `experiment_group` (output.md §11.3 A–F and §12.2 A–F), `mode` (identifiability mode), `open_loop_kind`, counterfactual mode/diff;
- applicability and saturation stats;
- status;
- a provenance reference (config hash, git commit, versions, timestamp).

**DS-STORE — Storage format.**
Benchmark in the pilot:
- (a) HDF5 shard per worker with an episode group (h5py available);
- (b) NPZ per episode (current);
- (c) Zarr, **only if** HDF5 fails a criterion, since it needs installation.

Criteria: sequential episode read, random window sampling throughput (DataLoader, 8 workers), parallel write from N processes, compressed size, write overhead per episode. The decision is recorded with numbers. The manifest/index is a CSV + JSON file (pyarrow/Parquet is not available, and the index is small).

**DS-SUM — Summaries.**
`data/reports.py` writes, per scenario, per domain and per dataset, `summary.json` + `summary.md` containing:
- success/failure counts by status;
- simulated duration;
- parameter, wind, scenario and initial-condition distributions;
- state and action ranges;
- min/max acceleration, max angular rate;
- saturation statistics;
- split counts;
- validation results;
- generation time, and CPU/GPU utilisation (psutil + `nvidia-smi` query).

---

## 6.6 Reproducibility

| ID | Audit row | Action | Run |
|---|---|---|---|
| RP1 | Seed derivation ad hoc (+1, +991), shared streams | See RP1 details below this table. | 3 |
| RP2 | No determinism flags; throttle carry-over | V4 + A13. Determinism test: regenerate 3 episodes and compare (bitwise or a documented tolerance). | 4 |
| RP3 | No resolved config/hash/commit | See RP3 details below this table. | 3 |

**RP1 — Seed derivation.**
`core/seeds.py` builds the tree with `numpy.random.SeedSequence`:

```
global
  └ experiment
      └ split_role
          └ domain
              └ scenario
                  └ episode
                      └ streams:
                          initial_conditions, trajectory, wind (→ one child per wind component),
                          excitation, physics_noise, sensor_noise
```

- Children are spawned by **stable keys** (hash of the entity id string → spawn key), not by position in a list. Adding a scenario type or an initial-condition field does not shift other seeds.
- All seed integers are stored in the manifest.
- Counterfactual siblings explicitly share the trajectory, excitation and turbulence streams (T12).

**RP3 — Provenance.**
- `provenance.json` per dataset:
  - fully resolved config (YAML) and its SHA-256 (`config_hash`);
  - global seed;
  - project git commit and dirty flag; OmniDrones commit;
  - versions (V1) and physics settings (V4);
  - generation start/end timestamps;
  - host and GPU list.
- Each episode's metadata stores `config_hash` and its manifest row.
- **Regeneration:** `physiworld-generate regenerate --dataset <dir> --episode <id>` re-runs one manifest row and compares it with the stored episode.

---

## 6.7 Identifiability

| ID | Audit row | Action | Run |
|---|---|---|---|
| I1 | Free-flight/excitation modes missing; hover-dominated | T10 + T11 + T14. The default `train_id` mix follows identifiability.md's recipe, with hover ≤ 10–20 % of data. Its consolidated volumes (hover 10, translation 25, accel 18, yaw 12, attitude 15, combined 20 per 100) are used as the default mix weights, plus open-loop segments. | 3 |
| I2 | Wind step/transient | Gust onset sampled across the episode (not only 2–8 s), plus wind steps on and off mid-flight. Effective once A8 lands. | 3–4 |
| I3 | Counterfactual pairs incomplete | T12 + W9 + A13 (motor reset) + Mode B replays the clamped commands. Mass/drag counterfactuals become possible via R1. | 3–4 |
| I4 | Decorrelation and residual checks missing | See I4 details below this table. | 5 |
| I5 | FIM/sensitivity/conditioning missing | See I5 details below this table. | 5 |
| I6 | Coupled parameters (m with KF; linear drag vs damping; J with the mixer) | See I6 details below this table. | 4–5 |
| I7 | Identifiability groups A–F (output.md §12.2) | Encoded as experiment templates (`experiment/identifiability_groups.yaml`), each with `experiment_group` metadata. Group E (aerodynamic model uncertainty) uses `forces/*` ablation variants. | 3 |
| I8 | Optional wind-inference benchmark (output.md §12.5) | Labels (current wind, short-term mean, magnitude/direction, gust onset, wind force) are all derivable from the stored auxiliary fields. Only a label-extraction helper is provided now; the benchmark itself is deferred. | 5 |

**I4 — Decorrelation and residual checks.**
`analysis/decorrelation.py` computes the output.md §11.4 metrics:
- Pearson and Spearman correlation;
- lagged cross-correlation over ±2 s;
- action vs wind, action vs wind force, reference vs wind;
- action spectrum, action covariance, and the singular values and condition number of the action Hankel matrix (persistent excitation);
- reported per dataset, wind profile, domain, trajectory type and counterfactual group.

`analysis/physics_consistency.py` computes the residual check Δ_t = s_{t+1} − RK4(f_phys(s_t, u_t), Δt) at the model rate, with nominal f_phys and no drag. It plots ‖Δ‖ against wind magnitude and against airspeed; identifiability.md's test requires Δ to grow with wind. It reports `requirement.md`'s sanity table: state coverage std > 0, residual-vs-wind slope, ρ(u, w) ∈ [−0.3, 0.3], transient content per mode, counterfactual pairing fraction ≥ 20 %.

**I5 — FIM, sensitivity and conditioning.**
`analysis/identifiability.py`:
- **Parameter vector:** θ = [m, J_xx, J_yy, J_zz, KF, KM, τ_m, k_rd, C_D·A, w_x, w_y, w_z].
- **Sensitivities:** S = ∂y/∂θ, with y = short-horizon state trajectories (0.5–2 s) at the model rate, by **simulator finite differences**. The counterfactual machinery runs ±δθ siblings (central differences, δ = 1–2 %) with identical seeds, Mode B (fixed actions), so the sensitivity is with respect to the true simulator, not a surrogate. A CPU surrogate (nominal f_phys + the same aero models) is provided for fast design iteration and validated against the simulator FD on a subset.
- **FIM:** F = Σ_t S_tᵀ Σ⁻¹ S_t, with Σ = the measurement-noise covariance from the D7 noise model.
- **Reported:** eigenvalues, condition number, parameter correlation from F⁻¹ (Cramér–Rao bound), and per-experiment-type information contribution (which experiments identify which parameters).
- Assumptions (local linearisation, noise model) are stated in the report, with no claim of global identifiability.
- Deliverable: a table mapping experiment types to parameters with a "well-conditioned" or "confounded" verdict. It feeds back into the default mix weights.

**I6 — Coupled parameters.**
- **(a) m with KF.** Translational response depends on KF/m; rotational response on KF·l/J. With J ∝ m (D9), only the ratios are identifiable from kinematics. This is documented as a structural non-identifiability that the FIM will show. It is harmless for a world model that conditions on observed history, and is stated in the limitations.
- **(b) Linear drag vs PhysX damping.** Removed by D6.
- **(c) J and the controller mixer.** Open-loop rotor-command experiments break this coupling (no controller in the loop).

---

## 6.8 Computational efficiency

| ID | Audit row | Action | Run |
|---|---|---|---|
| PF1 | 54 steps/s, unprofiled | See PF1 details below this table. | 5 |
| PF2 | Volume estimate (8.6 days single env) | Recomputed for the actual plan in Run 5 from the manifest's total simulated seconds and the measured throughput after vectorisation. Target: full dataset in ≤ 24 h on the 4 A6000s. If the projection misses that target, the plan is presented with options (fewer OOD replicates, shorter episodes) before Run 6. | 5 |
| PF3 | Single-drone wrapper (≥ 12 places) | See PF3 details below this table. | 6 |
| PF4 | Multi-GPU and multi-process | See PF4 details below this table. | 6 |
| PF5 | Benchmarks (prompt §15) | See PF5 details below this table. | 6 |
| PF6 | Replicator | Not used for physics trajectories. It is a visual synthetic-data tool; physics batching is done by vectorised environments. Recorded as a decision in the architecture guide. | — |

**PF1 — Profiling (step 1: measure).**
- Profile with `cProfile` and `torch.profiler` on a 2000-step episode; report the per-step time breakdown (PhysX step, state read, controller, wind, forces, IMU read, logging).
- Expected [inferred]: Python and host-sync overhead dominates.
- **Step 2, vectorise the loop:** a GPU-resident recorder (preallocated tensors, no per-step `.cpu().numpy()`), torch-batched wind evaluation (wind time series precomputed per episode at plan time where possible: steady, gusts and OU are all seed-determined and independent of the state; only spatial or airspeed-dependent terms are evaluated online), torch-batched force models, and IMU read batched over the env view. One device→host copy per episode chunk.

**PF3 — Single-drone wrapper (step 3: multi-env).**
- `SimulationInterface` becomes N-env:
  - spawn via OmniDrones' `GridCloner` pattern (`/World/envs/env_*`, `replicate_physics`, collision filtering), one drone per env (no inter-drone downwash or collisions), env spacing ≥ 10 m;
  - all getters return `(N_env, ·)`; reset and binding take `env_ids`;
  - one IMU per env.
- **Episodes in a batch** have independent scenarios and domains. When an env finishes, it is reset individually and takes the next manifest row (no lock-step waste). Mode B replays are scheduled after their baselines.
- **Physics invariance test:** the same episode run in env slot 0 of N=1 and in slot k of N=64 must match within the determinism tolerance.
- **Domain constraint:** per-env mass, inertia, KF and KM are supported by OmniDrones' per-env tensors.

**PF4 — Multi-GPU (step 4).**
- One process per GPU (`device=cuda:i`). Manifest rows are sharded round-robin by a stable hash, so assignment is deterministic and independent of the GPU count.
- Each process writes its own HDF5 shard. An async writer thread per process uses a bounded queue.

**PF5 — Benchmarks.**
- **Cases:** 1 env; 16/64/256/1024 envs on one GPU; 1–4 GPUs; writer batch sizes.
- **Metrics:** trajectories/s, physics steps/s, CPU and GPU utilisation, GPU and CPU memory, write throughput, total time.
- Results are recorded in `docs/guides/benchmarking.md`.
- Parallel complexity is added only after profiling shows where the bottleneck is (prompt §15).

---

## 7. Tests

| ID | Audit row | Action | Run |
|---|---|---|---|
| TS1 | pytest missing | V2. | 3 |
| TS2 | GP tests failing (code bug) | T7. | 3 |
| TS3 | `test_data_pipeline` failing (test bug) | Rewrite the fixture: distinct domain and scenario per group; N+1 states / N actions; uses `data/schema.py`. | 3 |
| TS4 | 6 empty test files | Filled by the new tests below (counterfactual, dataset_builder, excitation, identifiability, resampler → decimation, splits). | 3–5 |
| TS5 | Missing categories | See TS5 details below this table. | 3–6 |

**TS5 — Missing test categories.**

| Kind | Tests |
|---|---|
| CPU unit | config composition and validation; seeds (stable-key independence, reproducibility); quaternion/frame ops; each force model (§6.1 list); each wind model (§6.2 list); domain sampler (coupling, T/W, qmc coverage, ID/OOD disjointness); feasibility; references (anchoring, continuity, sine correctness); excitation signals (spectra, bounds); counterfactual planner (single-variable diff, shared seeds); manifest determinism; schema round-trip (write → read); metadata completeness; validator (synthetic good/bad episodes); decimation semantics; splits (group integrity, nested subsets) |
| CPU physics | analytic: free fall, terminal velocity, drag equivariance, rotor-drag linearity, motor-lag discrete response, wind-induced acceleration a = F_drag/m for a held body, force→torque transforms |
| Isaac integration (`-m isaac`) | sim setup; injector 1 N test; mass and inertia readback; damping = 0; ω_B vs gyro; IMU at rest and in free fall; hover trim; open-loop rotor step; yaw doublet sign; Lee hover at 4 yaws; closed-loop at 50 Hz control; counterfactual Mode A/B identical initial state; determinism repeat; multi-env invariance; end-to-end 3-episode generation → build → validate → load |
| Regression | Golden small episodes (3 scenarios × 2 winds, stored hashes + tolerance) re-run after each physics change; differences must be explained in the commit message |
| Performance | Steps/s and memory for 1 and N envs, compared with recorded baselines (warn on > 20 % regression) |

Documented commands: `pytest -m cpu`, `pytest -m isaac --isaac` (inside the container).

---

## 8. Missing vs redundant functionality

| Audit "missing" item | Resolved by |
|---|---|
| Working force application | A8 / F1 |
| Physics-validation suite | §7 TS5 |
| Force-accounting doc matching code | §5 force accounting |
| Open-loop excitation library | T11 |
| Identifiability metrics | I4, I5 |
| Yaw-aware controller | T1 (Lee reuse) |
| ID/OOD domain sampler | R3, W7 |
| Realised-statistics splits | DS8, D2 |
| Future actions | DS4 |
| Rotor speed/throttle logging | D3, schema |
| Total wrench logging | F9 |
| Provenance | RP3 |
| Summaries | DS-SUM |
| Multi-env | PF3 |
| pytest | V2 |
| Hierarchical config | C1 |

| Audit "redundant" item | Fate |
|---|---|
| Legacy `data/{generate_dataset,control_library,utils,wind}.py` | Moved to `lagacy/` (done); N1 import fix |
| `dataset_generation.yaml` | Moved to `lagacy/` (done) |
| `config.yaml` | → `lagacy/` (C3) |
| 0-byte configs | Deleted (C3) |
| Duplicate hover driver | Moved to `lagacy/` (done) |
| Four force paths | → injector only (S3) |
| Duplicate UAV yaml | Project yaml is the source + start-up equality check (S3) |

---

## 10. File plan

**Create:**
- `src/physiworld_uav/config/{__init__,schemas,compose,validation}.py`
- `src/physiworld_uav/core/{seeds,provenance}.py`
- `src/physiworld_uav/drones/{__init__,params,sim_binding}.py`
- `src/physiworld_uav/control/{__init__,controllers,references_to_commands}.py`
- `src/physiworld_uav/forces/{injection,registry,ground_effect}.py`
- `src/physiworld_uav/wind/regimes.py`
- `src/physiworld_uav/trajectories/feasibility.py`
- `src/physiworld_uav/excitation/{signals,open_loop,metrics}.py`
- `src/physiworld_uav/experiments/{plan,runner,counterfactual,generate}.py`
- `src/physiworld_uav/data/{schema,recorder,writer,splits,reports}.py`
- `src/physiworld_uav/analysis/{identifiability,decorrelation,coverage,physics_consistency,counterfactual}.py` (replacing stubs)
- `config/**` group files (§3)
- `tests/cpu/**`, `tests/isaac/**`, `tests/conftest.py`, `pytest.ini`
- `lagacy/README.md`
- `docs/guides/*.md`, `docs/models/*.md`

**Modify:**
- `core/simulation_interface.py`: multi-env, explicit physics settings, reset sequence, IMU validity.
- `core/state_extractor.py`: ω_B, layouts.
- `forces/{forces,body_drag,rotor_drag,aerodynamic_torque,_common,__init__}.py`
- `wind/{turbulence,gust_generator,wind_field,wind_manager,base}.py`
- `trajectories/{position_references,attitude_references,yaw_references,reference_manager,gp_sampler,gp_kernel,_shared}.py`
- `excitation/{direct_action,reference_perturbation}.py`
- `randomization/{parameter_sampler,domain_generator}.py`
- `data/{dataset_builder,dataset_validator,metadata,resampler,dataset}.py`
- `pyproject.toml`
- `examples/example0_generate_data.py`: thin wrapper or deprecated (§13-Q5).
- `docs/datasets/problems/identifiability.md`: erratum note only (D1).

**Retain unchanged:**
- `models/`, `training/`, `losses/`, `baselines/`, `evaluation/`, `visualization/` (D10)
- `docs/requirement.md`, `docs/prompt.md`, `docs/phase1/*`, `docs/datasets/*` (reference)
- `OmniDrones/`, `IsaacLab/` (no edits)

**Deprecate / delete:**
- `forces/rotor_wake.py` (→ `ground_effect.py`); `forces/hummingbird.py` (→ `drones/params.py`)
- `core/controller_adapter.py` (→ `control/`); `core/episode_manager.py` (empty)
- `trajectories/{trajectory_generator,trajectory_types,trajectory_validator}.py` (after porting)
- `excitation/action_excitation.py`, `excitation_metrics.py` (stubs → merged)
- `experiments/{ablation_runner,counterfactual_runner,identifiability_runner,evaluation_split}.py` (stubs → new modules)
- `analysis/{action_wind_correlation,wind_identifiability,diagnostics,counterfactual_analysis}.py` (stubs → new modules)
- `data/paired_dataset_builder.py` (→ `analysis/counterfactual.py`)
- `config/{controller,counterfactual,excitation}.yaml` (0 bytes); `config/config.yaml` (→ `lagacy/`)
- `tests/run_phase2.py`

---

## 11. Documentation deliverables

Prompt §18 lists these, with the run in which each lands:

| # | Document | Path | Run |
|---|---|---|---|
| 1 | Architecture overview (incl. known issues, Replicator decision) | `docs/guides/architecture.md` | 3 |
| 2 | Configuration guide | `docs/guides/configuration.md` | 3 |
| 3 | Drone model documentation | `docs/models/drone_hummingbird.md` | 4 |
| 4 | Force and torque models (one page each + force accounting) | `docs/models/*.md`, `force_accounting.md` | 4 |
| 5 | Wind models | `docs/models/wind.md` | 4 |
| 6 | Domain randomization guide | `docs/guides/domain_randomization.md` | 3–4 |
| 7 | Scenario-generation guide | `docs/guides/scenarios.md` | 3 |
| 8 | Open-loop / closed-loop / counterfactual guide | `docs/guides/experiment_types.md` | 3 |
| 9 | Dataset schema | `docs/guides/dataset_schema.md` (generated from `data/schema.py` + prose) | 3 |
| 10 | Reproducibility guide | `docs/guides/reproducibility.md` | 3 |
| 11 | Dataset validation guide | `docs/guides/validation.md` | 5 |
| 12 | Performance benchmarking guide | `docs/guides/benchmarking.md` | 6 |
| 13 | Scientific assumptions and limitations | `docs/guides/assumptions_limitations.md` | 4–6 |
| — | Environment guide (container, venv, pytest) | `docs/guides/environment.md` | 3 |
| — | Run reports | `docs/audit/run{3..6}_report.md` | each run |

---

## 12. Traceability matrix

Every audit point and where it is handled. Status after implementation will be tracked in each run report.

| Audit § | IDs → plan section |
|---|---|
| §0 | E1–E8 → §0 table |
| §1 | V1–V5 → §1 |
| §2 | S1–S6 → §2.2; N1, N2 → §B |
| §3 | C1–C4 → §3 |
| §4 | P1–P4 + termination → §4 |
| §5 | A1–A13 + force-accounting doc → §5 |
| §6.1 | F1 (application), F2 (geometry), F3 (ω frame), F4 (quaternion), F5 (body drag), F6 (rotor drag), F7 (wake/ground effect), F8 (aero torque), F9 (logging), F10 (domain drag), F11 (registry) → §6.1 |
| §6.2 | W1 (shapes), W2 (OU), W3 (Dryden), W4 (seeding), W5 (temporal scale), W6 (direction override), W7 (regime sampling), W8 (wind.yaml), W9 (sibling realisations) → §6.2 |
| §6.3 | T1 (yaw), T2 (thrust), T3 (Euler→rate), T4 (sine), T5 (anchoring), T6 (drift), T7 (GP), T8 (generator), T9 (random_action), T10 (closed-loop excitation), T11 (open-loop), T12 (counterfactual), T13 (feasibility), T14 (scenario coverage) → §6.3 |
| §6.4 | R1 (application), R2 (coupling), R3 (sampling/regions), R4 (metadata) → §6.4 |
| §6.5 | DS1 (rate), DS2 (resampler), DS3 (IMU), DS4 (future actions), DS5 (aux npz), DS6 (layout), DS7 (actions), DS8 (splits), DS9 (validation), DS10 (metadata), DS11 (model code), DS-STORE, DS-SUM, DS-DIST → §6.5 |
| §6.6 | RP1–RP3 → §6.6 |
| §6.7 | I1–I8 (+ coupled params I6) → §6.7 |
| §6.8 | PF1–PF6 → §6.8 |
| §7 | TS1–TS5 → §7 |
| §8 | missing/redundant lists → §8 |
| §9 | D1–D10 → §9 (applied throughout) |
| new | N1–N5 → §B; Q1–Q6 → §13 |

---

## 13. New decisions raised by this plan

The plan picks a default for each. Please confirm or change them in Run 2.

| # | Question | Default chosen in this plan | Alternative |
|---|---|---|---|
| Q1 | Install `pytest` (+ `pytest-timeout`) into `omnidrones_env` | Install (dev-only, small, required by prompt §17) | Separate test venv |
| Q2 | Controller knowledge of domain parameters (mass/inertia used by Lee and the mixer) | **True domain mass/inertia** (a well-tuned controller per vehicle) for ID data. Nominal parameters for an ablation subset, recorded as `controller_param_policy`. | Always nominal (more mismatch, risk of instability at ±20 %) |
| Q3 | Drag coefficients (k_rd, C_D, projected areas) | Literature-derived values computed in Run 4 and presented with sources before use | Calibrate against a reference dataset if you have one |
| Q4 | Rotor gyroscopic coupling (rotor DOFs) | Measure in Run 4. If significant, document it as part of the residual (keep f_phys = rigid body + thrust + gravity, as in thesis eqs. 1–4). | Add the rotor-gyroscopic term to f_phys |
| Q5 | Fate of `examples/example0_generate_data.py` | Rewrite as a ~30-line example calling the new CLI API after Run 3 | Keep the old file frozen in `lagacy/` |
| Q6 | Extra action field `rotor_speed_cmd` (steady-state Ω target of the command) | Store it (cheap; useful if the model should consume commanded rather than actual rotor speed) | Omit |
| Q7 | Numeric OOD boundaries for the temporal/gust categories (D2 defaults: f > 0.5 Hz, OU τ < 1 s, rise < 0.2 s, turbulence σ above the ID limit) | As listed; the ID turbulence σ limit is set after Run 4 measures the realised peak/σ relation under the peak ≤ 4 m/s constraint | Your values |
| Q8 | OOD-trajectory category (prompt §4 "unseen trajectory families") | Include `test_ood_trajectory` holding out `helix` and `figure_eight` families (small set) | Omit (wind OOD is the thesis focus) |
