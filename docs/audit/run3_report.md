# Run 3 report — configuration and experiment-generation architecture

Date: 2026-11-16 · Plan: [run2_implementation_plan.md](run2_implementation_plan.md) · Decisions: D1–D10 (recommended options); Q1 = install pytest; Q2 = controllers use the **true** vehicle mass/inertia; Q3–Q8 = plan defaults.

## Exit criteria

| Criterion (plan §A) | Result |
|---|---|
| CPU test suite passes | **112 passed**, 5 skipped (Isaac tests) — `python -m pytest`, 76 s |
| Pilot manifest generated deterministically | **Yes** — 147 episodes in ≈14 s; two plans byte-identical; changing one part leaves the others unchanged (tested) |
| 1-episode Isaac run writes the new schema with complete metadata | **Yes** — Isaac tests: **4 passed, 1 xfailed**. The xfail is the dataset-`complete` flag, blocked by the IMU defect below. |

## What was built

| Area | Implementation | Plan IDs |
|---|---|---|
| Config | Hydra root `config/defaults.yaml` + groups (drone, physics, limits, controller, dataset, experiment) + named templates (scenario ×18, excitation ×8, wind regimes ×6, domain ×2, counterfactual ×6, initial conditions ×2, forces ×2) | C1–C4 |
| Seeds and provenance | `core/seeds.py` (stable-key SeedSequence tree), `core/provenance.py` (config hash, git, versions, GPUs) | RP1, RP3, V1 |
| Drone model | `drones/params.py` — generic `DroneParams`, rotors in body X–Y plane, thrust +Z_B, total mass, hover trim, motor τ↔α, OmniDrones cross-check | F2, S3, C4 |
| Domains | `randomization/` — Sobol/LHS, ID ±h, OOD shell, held-out corners, coupling J ∝ m·shape, k_M ∝ k_F·ratio, T/W ≥ 2 | R2, R3, D9 |
| Wind | `wind/regimes.py` — ID/OOD samplers with realised-statistics guarantees; per-component seeds; step rise time; whole-field rotation; `direction_override` removed | W1, W4, W6, W7, W8, W9, D1, D2 |
| References | `trajectories/library.py` — pure-time families anchored at rest (min-jerk), C³, fixed sine parameters, heading-relative directions; `feasibility.py` (flatness-based thrust/tilt/rate/kinematic/workspace limits); GP fix | T4, T5, T7, T8, T13, T14 |
| Excitation | `excitation/signals.py` (step, pulse, doublet, Schroeder multisine, chirp, PRBS, smooth random), `reference_perturbation.py` (must start at rest), `open_loop.py` (rotor_command, collective_rate, body_wrench) | T9, T10, T11 |
| Control | `control/controllers.py` — OmniDrones Lee / Attitude / Rate controllers built from the domain's true parameters (Q2); yaw-aware attitude hold; tilt-compensated collective | T1, T2, T3, T6 |
| Planning | `experiments/plan.py` — parts → groups → episodes; roles assigned per group; domains partitioned between roles; feasibility resampling; `counterfactual.py` with a single-variable diff check (Mode A/B) | P1, T12, DS8, I1 |
| Execution | `experiments/runner.py` — control every D = 4 steps with the command held; true-mass hover-trim motor reset; IMU settle step; termination envelope; Run-3 guards | P2, P3, A13 (part), D4 |
| Data | `data/schema.py` (v2: ground_truth / observations / actions / control / environment), `recorder.py`, `writer.py` (HDF5 shards), `resampler.py` (decimation), `dataset_builder.py` + `dataset.py` (50 Hz, p\|v\|q\|ω_B, future actions, IMU as observation), `dataset_validator.py` (integrity, IMU validity, designation, leakage, counterfactual rules), `splits.py` | DS1–DS8, DS10, DS9 (part) |
| CLI | `physiworld-generate plan \| run \| validate \| regenerate` | §4, RP3 |
| Docs | `docs/guides/{architecture, configuration, environment, dataset_schema (generated), scenarios_and_experiment_types, reproducibility_and_domains}.md`; `lagacy/README.md`; D1 erratum on `identifiability.md` | §11 |
| Cleanup | Superseded code moved to `lagacy/src_superseded/` (it was untracked, so it was kept rather than deleted); docstring-only stubs deleted; old flat configs moved to `lagacy/config_*.yaml`; 0-byte configs deleted; `pyproject.toml` fixed | N1, N2, S1–S6, V3 |

## Done earlier than planned

* **A11 (partly):** ω_B is now logged and fed to controllers in the right frame. The schema needs it, and logging a mislabelled field would have repeated the audit's finding.
* **A12 (partly):** IMU freshness gate (a frame is valid only if `physics_step` advanced).
* **W6:** `direction_override` removed, because its rectifying behaviour would leak into regime sampling.

## Deferred or partial (with reason)

| Item | Status |
|---|---|
| V5 start-up import check | Documented in `environment.md`; no runtime check yet |
| I7 identifiability groups A–F as an experiment file | Group labels are carried per mix entry (`experiment_group`). The dedicated experiment comes with the Run 5 diagnostics. |
| S3 single force path, F1, F3, F5–F11 | Run 4 (physics). The runner refuses every force config except `none`. |
| DS9 physics-consistency checks, DS-STORE, DS-SUM | Run 5 |

## New findings in Run 3

1. **The Isaac IMU sensor never updates under the current GPU pipeline.**
   - Every frame has `physics_step = 0`, `time = 0` and zero readings (probe script plus smoke run).
   - The old pipeline therefore recorded all-zero "sensor" IMU data as valid.
   - Now: IMU samples are flagged invalid, the validator fails the episode (hard), and the smoke dataset is correctly `complete = false`.
   - **Needs your decision in Run 4.** Options:
     - (a) Run physics with CPU readback enabled (keep GPU dynamics, disable direct-GPU API / `suppressReadback`) and check whether `IMUSensor` updates.
     - (b) CPU physics pipeline (sensor known to work; slower; affects Run 6 scaling).
     - (c) A GPU-side accelerometer from PhysX rigid-body accelerations (`f_B = R_WBᵀ(a_W − g)`), clearly labelled *simulated from rigid-body acceleration*, not *Isaac IMUSensor*.
   - Recommendation: investigate (a) first, since it keeps the real sensor.
2. **OmniDrones `_reset_idx` computes the hover-trim throttle from the base-link mass (0.68 kg),** not the vehicle total (0.716 kg). The runner overwrites the throttle with the true-mass trim; the first logged rotor speed is 453.2 rad/s (verified).
3. **OmniDrones `AttitudeController` interprets the state's angular velocity as a body rate,** while Lee and Rate convert from world frame. The control layer feeds each one the frame it expects (tested).
4. **OmniDrones Rate and Attitude controllers leave autograd enabled.** Disabled in the control layer.
5. **PhysX logs `PxArticulationLink::setGlobalPose … DIRECT_GPU_API` at reset,** yet the logged initial state equals the planned one to < 1 mm and the yaw matches. To be re-checked in Run 4 together with item 1.
6. **Regeneration is bit-exact** for the smoke episode in a fresh single-env launch (max |Δ| = 0 for position and commands).

## Test inventory

| File | Covers |
|---|---|
| `tests/cpu/test_seeds_and_params.py` | seed tree, drone params, OmniDrones cross-check, domain sampler (ID/OOD/corners/coupling/determinism) |
| `tests/cpu/test_wind_regimes.py` | regime limits for all 5 regimes, ID vs OOD disjointness, offline = runner series, scaled/rotated siblings, seed independence, step rise |
| `tests/cpu/test_trajectories_excitation.py` | anchoring at rest (13 families), feasibility + round trip of all 16 closed-loop templates, accel/decel, heading, true sinusoid, signals bounded/deterministic, PRBS, perturbation guard, open-loop allocation vs OmniDrones, GP |
| `tests/cpu/test_controllers.py` | Lee hover = true-mass trim, mass policy, yaw-independent lateral response (5 yaws), yaw-aware attitude hold, rate mode |
| `tests/cpu/test_frames.py` | quaternion/rotation helpers vs matrices and OmniDrones, ω_B, IMU fallback |
| `tests/cpu/test_data_layer.py` | schema, HDF5 round trip, decimation semantics, model-ready sample semantics, validator episode + leakage + counterfactual rules, nested subsets |
| `tests/cpu/test_planner_runner.py` | allocation, plan determinism, part isolation, role disjointness, role ↔ wind/physics, counterfactual single-variable diff, guards, runner with a mock simulator |
| `tests/cpu/test_wind.py`, `test_gp_kernel.py` | pre-existing; updated for W6 and T7 |
| `tests/isaac/test_smoke_episode.py` | outputs + provenance, schema + metadata, hover physics sanity, regeneration; `complete` xfail (IMU) |

## How to reproduce

```bash
docker exec -it isaac-sim bash
cd /workspace/PhysiWorldUAV && source omnidrones_env/bin/activate && export PYTHONPATH=$PWD/src:$PYTHONPATH
python -m pytest                                       # CPU
python -m pytest --isaac tests/isaac                   # Isaac
python -m physiworld_uav.experiments.generate plan experiment=pilot
python -m physiworld_uav.experiments.generate run  experiment=smoke
```

## Inputs needed before Run 4

1. **IMU strategy:** option (a), (b) or (c) above.
2. **Q3 drag coefficients:** I will present the literature-derived values and sources at the start of Run 4, before using them.
