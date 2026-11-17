# PhysiWorld UAV — Run 1: Repository Audit and Scientific Review

Date: 2026-11-12 · Scope: `docs/prompt.md` §19 Phase 1 (repository audit) and Phase 2 (scientific review).
**No code was modified in this run.** The working tree already contains uncommitted fixes from the previous session (listed in §0). Two kinds of claims appear below: **[verified]** means confirmed by running code or reading source; **[inferred]** means it follows from source but was not executed.

Classification legend (Phase 2): **C** Correct · **CL** Correct with limitations · **NI** Needs improvement · **X** Incorrect · **?** Requires clarification.

---

## 0. Executive summary

1. **No valid dataset has been produced so far.**
   - `data/exp_001` holds 18 episodes of length 1, all `aborted_other`.
   - Since the previous session's fixes, a 3-scenario smoke run completes end to end (`/tmp/exp_smoke` in the container).
   - The physics in that output is still wrong. See points 2–5.
2. **Wind never acts on the drone.** [verified]
   - Forces are passed to `drone._view`, an `ArticulationView` with no `apply_forces`/`apply_torques`. `hasattr` guards turn the call into a silent no-op (`forces/forces.py:254-257`).
   - Every "windy" trajectory is a zero-wind trajectory labelled with wind. In the smoke run, the wind and zero-wind siblings have identical altitude ranges.
3. **Frames and parameters are mislabelled.** [verified]
   - The logged `angular_velocity` is **world-frame** but labelled body-frame.
   - `sim._mass` is the base_link mass (0.68 kg), not the total (0.716 kg).
   - Domain mass/inertia/drag are written to metadata but never applied to PhysX.
   - OmniDrones adds hidden PhysX linear/angular damping of 0.2 to every body.
4. **The closed-loop outer controller is unstable for |yaw| ≳ 33°.** [inferred, linear analysis]
   - Initial yaw is sampled uniformly over ±π, so about 80% of episodes are affected.
   - Thrust is not tilt-compensated, and the z reference is ignored.
5. **The dataset is not organised the way the thesis needs.** [verified]
   - Samples are windowed at 200 Hz (a 20-step horizon is 0.1 s).
   - IMU is classified as auxiliary although the thesis uses it as a model input.
   - Future actions are not stored.
   - The OOD-magnitude split is always empty (peak wind is read as 0).
   - Training wind is sampled from 2–16 m/s instead of the thesis's [0, 4] m/s.
6. **The physics models need correction before validation.**
   - Rotor geometry is on the wrong plane (thrust along +Y_B).
   - "Rotor drag" is a second isotropic quadratic drag rather than rotor-induced drag.
   - Rotor-wake thrust factors are physically inverted and not wired in.
   - The Dryden variance is wrong by about 2× on v/w.
   - Sine references are re-randomised every step.
7. **There is a lot of dead or duplicate code and configuration.**
   - Two generation pipelines; one legacy pipeline that cannot import.
   - 7 empty config files and 3 orphaned config files.
   - The model/training code cannot import, and uses a different (12-D) state schema.
8. **Throughput is low.** [measured] A single environment runs at about 54 physics steps/s (0.27× real time) after about 30 s of startup. The wrapper hard-codes one drone. The container has 4× RTX A6000 GPUs.

Ten scientific decisions need your input before Run 3 (§9).

---

## 1. Environment and versions [verified]

| Component | Version / state |
|---|---|
| Isaac Sim | 4.1.0-rc.7 (Kit 106.0.1, omni.isaac.core 3.18.1, omni.isaac.sensor 12.7.1, omni.physx 106.0.20) |
| OmniDrones | 0.1.1, commit 9ce7c202; library code unmodified |
| Isaac Lab | omni-isaac-lab 0.18.3 (v1.0.0, ffec353e). **Not used** by the project path |
| PyTorch / torchrl / tensordict | 2.2.2+cu121 / 0.3.1 / 0.3.2 (torchaudio cu118 mismatch, harmless) |
| SciPy | 1.10.1 (Isaac prebundle; available once the venv is activated) |
| Hardware | 4× RTX A6000 visible in the `isaac-sim` container |
| Venv | `/workspace/PhysiWorldUAV/omnidrones_env` (activated by `entrypoint.sh`). **No pytest installed.** |
| Packaging | `pyproject.toml` declares no dependencies, and its entry point `physiworld_uav:main` does not exist |

**Physics settings in use** (`SimulationContext(sim_params=None)`):
- GPU pipeline; TGS solver; CCD on.
- dt = 0.005 s with 1 substep; gravity −9.81 along Z.
- OmniDrones overrides solver iterations to 4 position / 0 velocity.
- No determinism flags (`enableEnhancedDeterminism` unset).

---

## 2. Folder structure and module responsibilities

```
config/            YAML configs (see §3)
docs/              thesis.pdf, requirement.md, prompt.md, datasets/* design notes, problems/1.md (earlier audit)
examples/          example0_generate_data.py  ← LIVE generation driver
scripts/           collect_data_wind_hover.py ← second live driver (hover only); many smoke/test scripts; 2 empty files
src/physiworld_uav/
  core/            simulation_interface (Isaac/OmniDrones wrapper), controller_adapter (outer loop),
                   state_extractor (state slicing, quaternions, IMU fallback), config_loader; episode_manager.py EMPTY
  forces/          ModularForceSystem: body_drag, rotor_drag, rotor_wake (unwired), aerodynamic_torque, hummingbird geometry
  wind/            WindManager → WindField of steady / gusts (cosine, step, sine, OU, random events) / Dryden / gradient
  trajectories/    ReferenceManager + position/attitude/yaw/thrust references; GP sampler; parametric generator
  excitation/      direct_action (random_action); action_excitation, excitation_metrics, reference_perturbation = STUBS
  randomization/   parameter_sampler, domain_generator (uniform sampling; output not applied to sim)
  data/            dataset_builder (windows + splits), dataset_validator, metadata, resampler, dataset (torch Dataset),
                   paired_dataset_builder (never called);
                   LEGACY: generate_dataset.py (cannot import), control_library.py, utils.py, wind.py
  analysis/, experiments/         STUBS (0–2 lines)
  models/, training/, losses/, baselines/, evaluation/, visualization/
                   world-model side; 12-D state, wind as input; training/train.py imports deleted physics/ → broken
tests/             13 files; 6 contain only a docstring; run_phase2.py is a shell script
OmniDrones/, IsaacLab/, omnidrones_env/   vendored frameworks + venv (git-ignored)
```

**Live generation path:**
- `examples/example0_generate_data.py` drives the run.
- It uses `core/*`, `trajectories/reference_manager`, `excitation/direct_action`, `wind/WindManager`, `forces/*`, `randomization/*` and `data/{dataset_builder, dataset_validator, metadata, resampler}`.

**Second live path:** `scripts/collect_data_wind_hover.py` (Hydra, hover only).
- It logs N+1 actions, where example0 logs N, and both claim `state_then_action`.

**Dead or duplicate code:**

| Item | Duplicated by / status |
|---|---|
| `data/generate_dataset.py` (1687 lines) | Imports deleted `wind_library` |
| `data/control_library.py` | Used only by `data/generate_dataset.py` |
| `data/utils.py` | Duplicated by `trajectories/_shared.py` |
| `data/wind.py` | Duplicated by `wind/`; unseeded global `random` |
| `core/simulation_interface._default_force_fn` | A fourth force-application path |
| Metadata schemas | Two implementations |
| Split mechanisms | Two implementations |
| Drag definitions | Three: `forces/`, PhysX damping, legacy USD damping scaling |

---

## 3. Configuration architecture

**Current system:**
- Plain `yaml.safe_load` via `core/config_loader.load_yaml`. One experiment file inlines most settings.
- Subsystem files are loaded ad hoc by path, relative to the repo root.
- Hydra is used only by the hover script. There is no composition, inheritance or override mechanism.
- Config sections were silently ignored when read with `getattr` on a dict. This is fixed in the builder and resampler, but `paired_dataset_builder.py:43` still does it.

| File | Consumer | Status |
|---|---|---|
| experiments.yaml | example0 | live |
| uav_hummingbird.yaml | domains, force geometry | live (duplicates OmniDrones' own hummingbird.yaml) |
| forces.yaml | `load_force_config` | live (`mass_kg` overrides domain mass) |
| collect_data_wind_hover.yaml | hover script (Hydra) | live, separate schema |
| domain_randomization.yaml, trajectory.yaml | tests only | effectively orphaned |
| wind.yaml | nothing | orphaned (example0 hard-codes wind) |
| dataset_generation.yaml (2924 lines) | dead `data/generate_dataset.py` | orphaned |
| config.yaml | `main.py` looks in `configs/` (wrong dir) | orphaned; conflicting mass 1.0 kg and dt 0.02 |
| controller, counterfactual, dataset, excitation, identifiability, ood_wind, train_wind | — | **0-byte files** |

**Conflicting values across files:**
- **Mass:** 0.716 vs 1.0 kg.
- **Inertia:** 0.007/0.012 vs 0.005/0.009.
- **dt:** 0.005 / 0.01 / 0.02.
- **Controller gains:** two different gain sets.
- **Drag area:** `cd_area` z is 0.10 vs 0.15.
- **Target rate:** defined twice (`experiment.target_hz` vs `dataset.target_hz`).
- **Domain seed:** `domain.seed` is used and `domain_randomization.seed` is ignored.

---

## 4. Current data-generation pipeline (example0)

```
load experiments.yaml → generate_domains (only domains[0] used) → SimulationInterface (1 drone)
→ _apply_domain (no-op on PhysX) → scenarios (hard-coded TRAJECTORY_KIND_TO_SPEC)
→ per scenario: _generate_interventions (hard-coded winds) → run_episode
     loop @200 Hz: ref → ControllerAdapter → OmniDrones RateController → rotor cmd
                   ModularForceSystem.evaluate → apply_to_isaac_drone (NO-OP) → sim.step → log
→ save .hf.npz (200 Hz) + resampled .npz (20 Hz, actions not resampled) + meta.json
→ build_samples on the 200 Hz arrays → assert_valid → dataset.npz, auxiliary.npz → split → splits.npz
```

**Timing semantics:**
- State k is logged before the step; `action[k]` is applied during step k → k+1.
- Wind and IMU at index k+1 are measured after the step.
- `wind_force_N[k+1]` and `thrust_N[k+1]` were applied *during* step k, so these channels are offset by one step.
- Index 0 of IMU and thrust is zero-filled.
- Forces are held over the step (zero-order hold).

---

## 5. Simulator physics accounting [verified from source]

| Effect | Applied by | Frame | Notes |
|---|---|---|---|
| Gravity | PhysX scene | world −Z | Project correctly excludes it from the applied subset. **C** |
| Rotor thrust `t·KF` | OmniDrones `rotors_view.apply_forces_and_torques_at_pos` (local) | rotor = body +Z | KF·ω_max² ≈ 6.0 N/rotor; T_max = 24.0 N; T/W = 3.42 |
| Rotor reaction torque | OmniDrones base_link call | world | |
| Motor dynamics | OmniDrones `RotorGroup` | — | See note 1 below this table. |
| OmniDrones drag `+drag_coef·m·v` | base_link call | world | Inactive (drag_coef = 0). **Sign is +** (anti-drag) if ever randomised. |
| **PhysX linear/angular damping 0.2** | set by OmniDrones on every rigid body at spawn | body | See note 2 below this table. **?** |
| Rotor joint spin | PhysX articulation DOFs | — | See note 3 below this table. |
| Project wind / drag / aero torque | **never applied** | — | **X** |

1. **Motor dynamics.** OmniDrones applies `throttle += 0.43·(target − throttle)` once per physics step. This is a blend factor, so the effective τ ≈ 1.78·dt (8.9 ms at 200 Hz) and changes with dt. Commands are clamped to [0, 1]. The documented `time_up`/`time_down` values are unused.
2. **PhysX damping.** This is a hidden, ground-relative, wind-unaware drag that appears in no metadata.
3. **Rotor joint spin.** Velocities are set to throttle·ω_max each step; the joints are real DOFs. Gyroscopic and rotor-inertia coupling act on the base. [inferred]

**Mass and inertia.**
- The USD has base_link 0.68 kg plus 4 × 0.009 kg rotors, for a total of 0.716 kg.
- `MASS_0`, and therefore `sim._mass`, is 0.68 kg (logged in the smoke run).
- The composite Izz ≈ 0.0130, but the RateController mixer uses 0.012.

**External-force API.**
- The correct target is `drone.base_link` (a `RigidPrimView`).
- OmniDrones already issues one `apply_forces_and_torques_at_pos` call on it per step.
- On the GPU pipeline a second call likely **overwrites** the first [inferred from the compiled backend; verify]. Injecting forces safely therefore means merging them into OmniDrones' single base_link call.

**Frames.**
- `get_state()` returns `(n_env, n_drone, 23)`: p, q (wxyz), v_W, **ω_W**, heading, up, throttle·2−1.
- The RateController converts ω_W to body rates itself.
- The project slices `[:13]`, which is correct, but labels ω as body. **X**

**IMU.**
- Uses `omni.isaac.sensor` `IMUSensor`, a child of base_link.
- With `read_gravity=True` it reports specific force (+9.81 on z at rest) in the body frame. The project fallback uses the same convention. **C**
- The frame dict has no validity key, so the validity gate is ineffective. **NI**
- There is no noise or bias model (only averaging filters).
- The smoke run confirmed `imu_source=sensor` for all episodes.

**Determinism.**
- No determinism flags are set.
- Throttle state is not reset between episodes (`_reset_idx` is never called), so counterfactual siblings start from different motor states. **X**

---

## 6. Phase 2 — scientific review by subsystem

### 6.1 Force and torque models (`forces/`)

| Item | Class | Issue / mathematics |
|---|---|---|
| Force application | **X** | No-op (§0). It also hard-codes `reshape(1,3)` and fails for more than one env. The working fallback in the hover script is unreachable. |
| Geometry (`hummingbird.py:39-48`) | **X** | See details below the table. |
| ω frame fed to models | **X** | World-frame ω used as `omega_B`. Needs ω_B = R_WBᵀ ω_W. |
| Quaternion → R (`_common.py`) | **C** | Hamilton wxyz, body→world, used consistently. |
| Body drag | **CL / NI** | See details below the table. |
| Rotor drag | **X** | See details below the table. |
| Rotor wake / ground effect | **X / CL / ?** | See details below the table. |
| Aerodynamic torque | **NI** | See details below the table. |
| Logged `wind_force_N` | **X** | Body drag only, about 23% of the external force missing. Total external force/torque not logged. |
| Domain `drag_coef` | **X** | Sampled but never applied. Metadata records yaml 0.2, which is in effect nowhere. |

**Geometry (`hummingbird.py:39-48`).**
- Thrust is placed along +Y_B, with rotors in the X–Z plane.
- The USD, OmniDrones and `uav_hummingbird.yaml` all use FLU: rotors in X–Y at (±0.17, 0, 0.01) and (0, ±0.17, 0.01), thrust along +Z_B.
- Logged rotor thrust comes out horizontal (7.02 N along world y at hover) [verified].
- Rotor-drag moment arms are wrong.

**Body drag.**
- Formulation: F_B = −½ρ·diag(C_dA)·|v_rel,B|∘v_rel,B, with v_rel = v − w. Frames and sign are correct.
- C_dA = [0.08, 0.12, 0.10] m² are **placeholders**. They are about 4–8× a plausible frontal area for a 0.34 m frame, and anisotropic on a symmetric frame.
- With these values, at 8 m/s the total drag is 4.08 N = 0.58·W (equilibrium tilt 30°). That approaches the 3.84 N lateral authority at the 0.5 rad tilt limit.
- Quadratic drag suits a bluff body (Re ≈ 1.6×10⁵ at 8 m/s). For small quadrotors at low speed, however, the dominant horizontal drag is rotor-induced and ∝ v (see rotor drag). So pure-quadratic drag under-predicts below about 2–3 m/s.
- The diagonal approximation is adequate only for near-level attitude. It fails for large tilt, sideslip and cross-axis coupling.

**Rotor drag.**
- Implemented as F_i = −½ρC_dπR²|v_rel,i|v_rel,i: isotropic and quadratic, independent of rotor speed.
- Published rotor drag (blade flapping plus induced drag) is F = −Ω̄·R_WB D R_WBᵀ v_rel. It is linear in v, acts in the rotor plane (d_z ≈ 0) and scales with rotor speed (Faessler et al. 2018; Kai et al. 2017; NeuroBEM, Bauersfeld 2021).
- As implemented it double-counts parasitic drag with body drag, and its moment arms come from the wrong geometry.

**Rotor wake / ground effect.**
- `λ_wind = v_h/√(V⊥² + v_h²)` is Glauert's *induced-velocity* ratio misused as a *thrust* ratio, giving 40% thrust loss at 8 m/s. At fixed rotor speed, edgewise flow increases thrust (translational lift). **X**
- The download term is sign-blind (climb and descent give the same factor). **X**
- The Cheeseman–Bennett ground effect is acceptable within its assumptions. **CL**
- The model is never called: `evaluate()` hard-codes zero, and the yaml references a function that doesn't exist. **?**

**Aerodynamic torque.**
- τ = −k₁ω − k₂|ω|∘ω with placeholder coefficients.
- The element-wise quadratic term is not frame-invariant.
- It receives world-frame ω and is then rotated again.
- It physically overlaps with the ω×r rotor-drag damping.

### 6.2 Wind models (`wind/`)

| Item | Class | Issue |
|---|---|---|
| 1-cos gust, step, sinusoid, spatial gradient | **C** | Standard shapes; the gradient is disabled everywhere. |
| OU gust | **CL** | See details below the table. |
| Dryden turbulence | **X** | See details below the table. |
| Seeding | **X** | See details below the table. |
| `temporal_scale` (temporal OOD) | **NI** | Skipped for stateful components (OU, Dryden, random events): output is identical at scale 1 and 4. |
| `direction_override` | **NI** | Rotates the *total* vector onto a fixed axis, which rectifies zero-mean noise (mean 0.68 m/s). This changes the statistics, not just the direction. |
| Wind sampling in example0 | **X** | See details below the table. |
| `config/wind.yaml` | **?** | Not consumed by anything. |

**OU gust.**
- Uses the exact AR(1) discretisation; measured σ is within 3% [verified].
- Starts from 0 rather than the stationary distribution.
- Isotropic in 3D: vertical σ equals horizontal σ.

**Dryden turbulence.**
- The transfer-function shapes are correct.
- The fixed-point Lyapunov solve (500 iterations from P = I) does not converge for T = L/V = 40 s, where the eigenvalue is 0.999875. As a result v/w σ is **2.07× too large**; measured std is 1.66/1.79 against 1 [verified].
- It uses Euler (not the ZOH the docstring claims) for v/w.
- It starts from 0.
- It is time-based with a fixed V = 5 m/s (invalid near hover).
- L = 200 m is a high-altitude value. At h ≈ 1.5 m, MIL-F-8785C gives L_w ≈ h and L_u ≈ 12 m, with the spec valid only for h ≥ 3 m.
- It is disabled in example0.

**Seeding.**
- `reset_all(seed)` reseeds every component with the **same** seed, overriding the per-component seeds.
- Two OU gusts in one spec come out perfectly correlated: `gust_1 = 0.5·gust_0` [verified].

**Wind sampling in example0.**
- Steady wind is U(2, 8) m/s plus gusts up to 5 m/s, including up/down directions.
- Measured over 800 interventions: peak |w| is ≤ 4 m/s in only 1.2% of episodes; the median is 8.5 m/s and the maximum 16 m/s.
- Counterfactual siblings do not share a disturbance realisation: each has its own seed, direction and gust type.

### 6.3 Trajectories, controller and excitation

| Item | Class | Issue |
|---|---|---|
| Outer loop ignores yaw (`controller_adapter.py:128`) | **X** | See details below the table. |
| Thrust law | **X** | See details below the table. |
| Euler-error → body-rate | **CL** | Valid at small angles (r = cosφcosθ·ψ̇ ≈ 0.77ψ̇ at 0.5 rad). |
| `yaw_sine`, `attitude_sine` | **X** | Amplitude and frequency are re-sampled **every 5 ms step** (`yaw_references.py:47`, `attitude_references.py:88`). The result is band-limited noise, not a sine [verified]. |
| Reference anchoring and continuity | **NI** | See details below the table. |
| Attitude / yaw / thrust references | **NI** | No horizontal hold, so the vehicle drifts with the wind. |
| GP sampler | **NI** | See details below the table. |
| Parametric `trajectory_generator` | **X** | Config keys don't match the function arguments, so it raises `TypeError` [verified]. Not used by example0. |
| `random_action` excitation | **X** as excitation | See details below the table. |
| True open-loop excitation | **missing** | See details below the table. |
| Feasibility checks | **missing** | No actuator-feasibility (T_max, tilt, rate) or jerk validation in the live path. |

**Outer loop ignores yaw.**
- Computes pitch = aₓ/g and roll = −a_y/g with no R_z(−ψ) rotation.
- A linearised closed loop becomes unstable for |ψ| ≥ 33.5°.
- With initial ψ ~ U(−π, π), about 81% of position-tracking episodes are affected [inferred].
- Acceleration feed-forward is ignored, and `kd_attitude` is unused.

**Thrust law.**
- `T = m(g + kp(z₀−z) − kd·v_z)`: no division by cos(tilt), no T_max clamp.
- The z reference is ignored: it holds the initial z, so helix, `line.z_change` and waypoint dz have no effect.
- Sag at 0.5 rad tilt is 12% of thrust, about 0.48 m.

**Reference anchoring and continuity.**
- Circle: 1.5 m position step at t = 0.
- Figure-eight: velocity step at t = 0.
- `line` always runs along +x.
- `accel_decel` never decelerates.
- Acceleration steps give infinite jerk.

**GP sampler.**
- The "Cholesky failed" errors come from absolute jitter of 1e-8 on a matrix of numerical rank 18/500. They are not a sign bug; 1e-6 works.
- Prior position σ is √3 rather than σ.
- Very low velocity and acceleration excitation (0.18 m/s, 0.07 m/s²).
- The mean is absolute, not anchored to the initial state.

**`random_action` excitation.**
- Open-loop rates of ±0.5 rad/s with mean thrust below hover.
- In a point-mass Monte Carlo, **299/300 episodes hit the ground** (median 2.3 s) [inferred model]. The excitation produces crashes, not useful dynamics.

**True open-loop excitation.**
- `action_excitation`, `excitation_metrics` and `reference_perturbation` are stubs.
- No steps, pulses, chirps, PRBS or multisines exist at any abstraction (rotor command, collective + rate, wrench).
- No Fisher-information or identifiability metrics are implemented.

**Existing controllers:** OmniDrones provides `LeePositionController` (SO(3), yaw-aware, tilt-projected thrust, acceleration feed-forward, Hummingbird gains available) and `AttitudeController`. Both output rotor commands. Reusing them follows `prompt.md` §6A. Caveat: their mass, inertia and mixer must be updated for randomised domains.

### 6.4 Domain randomization

| Item | Class | Issue |
|---|---|---|
| Application to sim | **X** | See details below the table. |
| Coupling | **NI** | See details below the table. |
| Sampling | **NI** | Uniform only; no LHS or Sobol; no ID/OOD range separation; only one domain used. |
| Metadata | **X** | Records parameters that were not simulated. |

**Application to sim.**
- `_apply_domain` writes the Python attributes `MASS_0` and `INERTIA`. `INERTIA` doesn't exist, and PhysX is unchanged.
- `thrust_coef`, `torque_coef`, `motor_time_constant` and `drag_coef` are never applied.
- OmniDrones has a working mechanism (`setup_randomization`/`_randomize`: `set_masses`, `set_inertias`, KF, KM, tau, com). It changes base_link only, and the controller mixer must be updated too.

**Coupling.** Mass, inertia and thrust are sampled independently. Inertia does not scale with mass, and there is no thrust-to-weight constraint: the worst case is T/W ≈ 1.14.

### 6.5 Dataset schema, splits and validation

| Item | Class | Issue |
|---|---|---|
| Sample rate | **X** | See details below the table. |
| Resampler | **X** (partially fixed) | See details below the table. |
| IMU role | **X** | `accel_imu`/`gyro_imu` are in `AUXILIARY_KEYS`. The thesis (eq. 24) uses IMU as encoder input. |
| Future actions | **X** | Only u up to t is stored; multi-step rollout needs u_{t..t+P−1}. |
| `auxiliary.npz` | **X** | Saved without offsets or episode ids, so it can't be mapped back to samples. |
| State layout | **NI** | See details below the table. |
| Action semantics | **?** | See details below the table. |
| Splits | **X / NI** | See details below the table. |
| Episode validation | **X** | Disabled (`if False`), and would reject every live episode (action length N vs t length N+1). No physics-consistency checks (ṗ ≈ v, quaternion kinematics, IMU vs finite differences). |
| Metadata | **NI** | See details below the table. |
| World-model code | **X** | `models/`, `training/` and friends expect 12-D states and wind as input, and don't read `dataset.npz`. `training/train.py` can't import. |

**Sample rate.** `build_samples` receives the 200 Hz arrays, so H = 10 is 50 ms and P = 20 is 0.1 s.

**Resampler.**
- Filtered along the channel axis; fixed.
- `action` (N rows) is passed through unresampled.
- `filtfilt` low-passes *states*, which breaks ṗ = v consistency. States should be decimated and inputs held (§9 D3).

**State layout.** The simulator gives p|q|v|ω, the builder uses p|v|q|ω, and the models use 12-D. The layout is not stored in `dataset.npz`.

**Action semantics.**
- `action` is the RateController output 2T_i/T_max − 1, *before* the [0, 1] clamp and the motor lag.
- Actual throttle and rotor speed are not logged.
- The thesis eq. 22 defines u as rotor speeds Ω.

**Splits.**
- `ood_wind_magnitude` reads top-level wind keys but the spec is nested, so peak = 0 and the split is always empty [verified].
- It has no upper bound and masks per sample, which splits counterfactual groups across train and test.
- `unseen_wind`/`unseen_gust` read keys that are never written.
- `random_by_group` can produce an empty test set.
- Domain leakage was always flagged for fixed physics; fixed in example0.

**Metadata.**
- Missing: dataset version, config hash, resolved config, git commit, software versions, timestamp, split and ID/OOD designation.
- Open-loop episodes are labelled `closed_loop`.
- `domain_seed` and `initial_condition_seed` are null.

### 6.6 Reproducibility

| Item | Class | Issue |
|---|---|---|
| Seed derivation | **CL** | See details below the table. |
| Simulator determinism | **NI** | No determinism flags. Throttle carries over between episodes. Pairs are not guaranteed bit-identical. |
| Config provenance | **X** | No resolved config, hash or commit stored. |

**Seed derivation.**
- Deterministic given the config: earlier seeds are stable when lists grow [verified].
- Initial conditions and seeds share one stream.
- The offsets +1 and +991 are ad hoc; use `SeedSequence.spawn`.
- Wind components share one seed (§6.2).

### 6.7 Identifiability (against `docs/datasets/problems/identifiability.md`)

| Requirement from doc | Current state | Class |
|---|---|---|
| Free-flight / excitation modes, not hover-dominated | Closed loop on simple references only. `random_action` crashes. No stabilised excitation. | **X** |
| Wind step/transient mid-flight | Step and cosine gusts exist at 2–8 s. | **CL** (wind not applied, so no effect today) |
| Counterfactual pairs | Exist (closed-loop A, action-replay B). See notes below the table. | **NI** |
| Decorrelation of u and wind; residual-vs-wind sanity checks | Not implemented (`analysis/*` stubs). | **missing** |
| FIM / sensitivity / condition number | Not implemented. | **missing** |

**Counterfactual-pair notes.**
- Wind realisations are not shared between siblings.
- Motor state is not reset.
- Mode B replays the commands *before* the clamp.
- Mass/drag counterfactuals are impossible because domains aren't applied.

**Coupled parameters the design must break.** Increasing m together with KF leaves T/m unchanged. Linear drag and the PhysX damping 0.2 are both ∝ v and indistinguishable. J and the controller mixer change together under closed loop.

### 6.8 Computational efficiency [measured / inferred]

**Measured:** about 54 physics steps/s with one env (0.27× real time) and about 30 s of startup.

**At this rate,** `requirement.md`'s volume (about 2×10⁵ s of flight across training and 5 OOD regimes = 4×10⁷ steps at 200 Hz) would take about 8.6 days on one env.

**Likely bottleneck [inferred, not profiled].** Per-step Python overhead in the loop:
- GPU→numpy→GPU round trips;
- per-step torch allocations;
- `rotor_drag` and the wind manager in numpy;
- list appends.

The bottleneck is probably not PhysX.

**What scaling would need.** OmniDrones' `GridCloner` multi-env and one process per GPU are available. The wrapper hard-codes a single drone in at least 12 places (`simulation_interface.py:122-125, 157, 237-248, 499-502, 519-524, 554-556, 578, 641-648, 701, 815`; `forces.py:250-253`; example0 `reshape(-1)`).

---

## 7. Tests [verified: run without pytest, via in-memory runner]

| Module | Result |
|---|---|
| test_uav_config, test_domain_randomization, test_gp_kernel, test_wind, test_trajectories (hover, circle) | PASS (26 total) |
| test_gp_sampler ×2, test_trajectories gp ×2 | FAIL. Code bug: absolute GP jitter too small. |
| test_data_pipeline::end_to_end_build_and_split | FAIL. Test bug: all mock episodes share a domain and scenario. The fixture also uses N, not N−1, action rows. |
| test_counterfactual, test_dataset_builder, test_excitation, test_identifiability, test_resampler, test_splits | **Empty** (docstring only) |

**Missing:**
- per-force analytic tests;
- frame and quaternion tests;
- physics-validation tests (free fall, hover equilibrium, drag-induced acceleration, motor step);
- Isaac integration tests;
- a split between CPU-only and Isaac test markers;
- pytest itself.

---

## 8. Missing vs redundant functionality (summary)

**Missing:**

| Area | What is missing |
|---|---|
| Force application | Working force application |
| Physics validation | Physics-validation suite; force-accounting document matching code |
| Excitation | Open-loop excitation library (rotor / collective + rate / wrench); identifiability metrics (FIM, sensitivity) |
| Controllers | Yaw-aware controller (or reuse of Lee) |
| Domains and splits | ID/OOD domain sampler with held-out ranges; split schemes driven by realised wind statistics |
| Dataset contents | Future-action storage; rotor speed/throttle logging; total external wrench logging |
| Provenance and reporting | Provenance metadata; summary reports |
| Scale and tooling | Multi-env generation; pytest; hierarchical config |

**Redundant:**
- Legacy `data/{generate_dataset, control_library, utils, wind}.py`.
- `dataset_generation.yaml`, `config.yaml` and the 0-byte configs.
- Duplicate hover driver.
- Four force-application paths.
- Duplicate UAV yaml (project vs OmniDrones).

---

## 9. Decisions requiring your input (to be settled in Run 2)

| # | Question | Why it matters | Options |
|---|---|---|---|
| D1 | **Wind units for the OOD split** | The thesis and `requirement.md` use **m/s** (\|w\| ∈ [0,4] vs (4,8]). `identifiability.md` uses **N** ("≤ 4 N", "5–8 N"). | m/s of wind velocity (recommended: it is the controllable quantity, and the force depends on the drag model) or N of disturbance force |
| D2 | **Which wind statistic defines ID/OOD** | Gusts and turbulence stack on the steady wind. | Mean steady \|w\| · peak \|w\| including gusts · RMS. Are vertical components allowed in training? |
| D3 | **Control input u** | Thesis eq. 22 says rotor speeds Ω; currently the pre-clamp normalised command is logged. | (a) Rotor ω (post-lag; most physical, needed by f_phys). (b) Clamped command. (c) Collective thrust + body rates. Log all; choose the model input. |
| D4 | **Model / logging rate** | Sim 200 Hz; the experiment says 20 Hz; `identifiability.md` says 100 Hz. | Recommend physics at 200 Hz, control held over k sub-steps, states decimated (not filtered). Need the target Δt. |
| D5 | **Ground-truth aerodynamics in the sim** | f_phys must exclude what d_θ learns. The choice of "true" drag model (quadratic body drag, linear rotor drag, both) defines the residual. | Pick a minimal calibrated set (e.g. rotor-drag D-matrix + parasitic quadratic), with literature-based coefficients. |
| D6 | **PhysX damping 0.2** | Hidden, ground-relative and wind-unaware. | Zero it and model drag explicitly (recommended), or keep and document it as an unmodelled effect. |
| D7 | **IMU role** | Currently auxiliary; the thesis uses it as encoder input. | Make IMU an observation (recommended). Add sensor noise and bias? |
| D8 | **Zero-wind counterfactual baselines in training** | One third of episodes are zero-wind. | Keep as ID data, or reserve for physics self-consistency (thesis Case A). |
| D9 | **Physics-mismatch range (H2)** | `requirement.md` says ±10–20% (OOD). `experiments.yaml` uses 0.5–1.5× (destabilising). | Define nominal, ID and OOD ranges, and the coupling rules (J ∝ m·L², T/W ≥ 2). |
| D10 | **Legacy cleanup scope** | The deleted `physics/` breaks `training/train.py`. The world-model code expects a 12-D state with wind as input. | Deprecate legacy data code. Is the world-model code in scope for schema alignment? |
