# PhysiWorldUAV — Deep Module Explanations

This is organized by **dependency order**, which is also the natural reading order. Each module is explained at three levels: **Why** (the problem it solves), **What** (the responsibility boundary — what it owns and what it must not touch), and **How** (the mechanism, key algorithms, and invariants).

---

## Part 1 — Foundation layer (no dependencies on physics or simulation)

### `core/config_loader.py`

**Why it exists.**
The research depends on three orthogonal sources of truth — the UAV's nominal physical parameters (task §3), the force-model configuration (task §8), and the experimental setup (§13). Any silent divergence between these three is catastrophic: if the YAML says `mass=0.716` but the simulator spawns a `0.68` kg drone, the physics is wrong, the dataset is wrong, and no amount of model capacity can compensate. This module is the **single place where a config file is read**, so every downstream module consumes the same numbers.

**What it does.**
- Loads YAML into plain Python dicts (no OmegaConf in the hot path).
- Resolves paths relative to the repo root so imports work from any working directory.
- For `uav_hummingbird.yaml`, applies **hard assertions** that encode the task's non-negotiable facts: `mass == 0.716`, `l == 0.34`, `l / 2 == 0.17 == arm_lengths[0]`, `num_rotors == 4`.

**How it works.**
`load_hummingbird()` reads the YAML, then immediately checks the physical invariants and raises before any simulator is touched. This prevents the entire class of "wrong drone spawned" bugs — they fail at load time, not after 100 episodes of wasted compute. The `arm_length_diagonal == l` and `arm_length_individual == l/2` fields in the YAML are **redundant assertions baked into the file itself** so a human editing the YAML can't drift without a test catching it.

---

### `core/state_extractor.py`

**Why it exists.**
Every module needs to know how to read the drone's state, what frame it's in, and how to convert between conventions. Without a single source of truth, one module will use `[x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]` and another will use `[p, q, v, ω]` — and the two will silently corrupt each other's data. The research also needs **SLERP** (for resampling) and **specific-force IMU derivation** (for fallback), and those must match the convention the physics engine uses.

**What it does.**
- Declares the canonical 13-float layout: `POS(3) | QUAT(4) | VEL(3) | ANGVEL(3)`.
- Provides quaternion ↔ rotation matrix in **both** torch and numpy.
- Provides quaternion continuity (sign flips don't change orientation) and SLERP.
- Provides `compute_imu_from_state`, which returns **specific force** `R^T(a - g)` — the same convention a real accelerometer uses.

**How it works.**
The key invariant is that **all quaternions are `[w, x, y, z]`, Hamilton convention, and map body vectors to world vectors**: `v_world = R(q) @ v_body`. Every other module obeys this. The SLERP implementation handles the antipodal case (`dot < 0 → flip q1`) so it always takes the shortest arc; this is what the resampler uses to interpolate orientations without "spinning the long way around". `compute_imu_from_state` deliberately returns *proper* acceleration (with gravity included), because that's what a stationary IMU reads (`+9.81` on the z-axis when level) — a naive finite-difference would return `0`, and downstream sensor-noise experiments would be wrong.

Crucially, **this module has no import of `omni_drones` or `omni.isaac`**. That's not laziness — it's what makes the entire data pipeline testable in CI without launching a simulator.

---

## Part 2 — Environment layer (how the world exerts forces on the drone)

### `wind/` — the complete subsystem

**Why it exists.**
The thesis studies UAV dynamics under hidden wind disturbances. Wind must be:
1. **Velocity, not force.** A drag force depends on the vehicle's own CdA; if you generate force directly, you have smuggled the answer into the input. Velocity is the exogenous physical quantity (measurable by anemometer, independent of vehicle). The whole "hidden disturbance" story only works if wind is a velocity field.
2. **Composable.** The design doc requires `u = u₀ + G(p - p₀) + u_gust(t) + u_turb(p, t)`. Each term is physically different and must be individually toggleable (for ablations) and individually reportable (for diagnostics).
3. **Seeded per episode**, so counterfactual pairs share identical wind parameters.

**What the subsystem owns.**
- The **velocity field function** `u(p, t) → ℝ³`.
- Its own RNG stream, seeded per intervention.
- Per-episode parameter sampling (a range like `[2, 8]` m/s is sampled **once** at episode start, then fixed).

**What it does NOT own.**
- Force computation (that's `forces/`).
- Force application (that's `simulation_interface`).
- Drag coefficients (those are physics parameters, sampled by `randomization/`).

**How the pieces fit together:**

| File | Responsibility | Key mechanism |
|---|---|---|
| `base.py` | Protocol + `WindSample` dataclass | `WindComponent.contribution(pos, t) → ℝ³`. `WindSample` carries `total_W` AND `by_component` so every caller can inspect which term dominated at each timestep. |
| `steady_wind.py` | Constant `u₀` | Trivial: `speed · direction_unit`. |
| `spatial_gradient.py` | `G(p - p₀)` | Matrix-vector product. Why this is a separate file: it's the only component that actually depends on position, so its interface forces every caller to pass position down. |
| `gust_generator.py` | All non-Gaussian gusts | `CosineGust` implements FAR-25 1-cosine (rise time = half duration). `OUGust` implements **exact discrete** Ornstein–Uhlenbeck: `x[k+1] = a·x[k] + σ√(1-a²)·ε`, `a = exp(-dt/τ)` — this is not the Euler approximation, it's the closed-form discretization, so the stationary variance is exactly `σ²` regardless of `dt`. That's why the test in `test_wind.py` can verify `std ≈ σ` and autocorrelation at lag `τ` ≈ `e⁻¹` within tight bounds. |
| `turbulence.py` | Dryden | Implements the MIL-F-8785C state-space form with **explicit variance scaling**: the discrete filter's stationary output variance is solved via Lyapunov iteration and then scaled so the output RMS matches `intensities_mps` exactly. Without this, the reported turbulence intensity in metadata would not match the actual signal RMS, and the OOD split would be wrong. |
| `wind_field.py` | Composition | Sums components, applies `wind_scale` and `temporal_scale`. Stateful components (`is_stateful`) are evaluated on real time; deterministic ones on scaled time so `temporal_scale` warps gusts but doesn't corrupt OU state evolution. |
| `wind_manager.py` | Episode lifecycle + spec → field | `configure(seed)` builds the field and calls `reset_all(seed)` on stateful components. Each `[lo, hi]` range is **sampled once per episode**. `ood.direction_override` is applied at output time, so OOD directional tests don't need a different field — they just change which direction wins. |
| `adapter.py` | Legacy compat | Thin wrappers so the old `wind_library` call sites work unchanged. |

**The critical invariant** every part upholds: **velocity in, velocity out**. No component returns a force, ever. That's what makes §12's identifiability experiments meaningful — the model sees wind only via its effect on the drone, never as a pre-computed label.

---

### `forces/` — the aerodynamic model

**Why it exists.**
The simulator handles gravity and rotor thrust natively. Everything else — body drag, rotor drag, rotor wake, aerodynamic torque — must be **explicitly computed and explicitly applied** because:
1. It's the physical phenomenon the study models (wind-induced disturbance).
2. It must be individually toggleable for the ablations in §17.
3. It must be separately logged per-model so §11.4's correlation analyses can attribute force components.

**What the subsystem owns.**
- Per-model force and torque computation.
- The physical model of how wind produces force (via relative velocity, not via wind speed directly).

**What it does NOT own.**
- Application to the simulator (that's `apply_to_isaac_drone`).
- Gravity and rotor thrust (the simulator owns those).
- Drag coefficients (those come from `randomization/`).

**Why it's split into so many files:**

| File | Why separate | Physical model |
|---|---|---|
| `_common.py` | `quat_to_matrix` is needed by every model but must not create circular imports | Type alias `WindFn`, `quat_to_matrix` |
| `hummingbird.py` | Geometry MUST match the spawned drone | Derives rotor positions `(r·cosθ, 0, r·sinθ)` from YAML. Asserts `l == 2·arm_lengths[0]`. |
| `body_drag.py` | The dominant drag term | `F_B = -0.5 ρ diag(CdA) · \|v_rel\| v_rel`, `v_rel = v_UAV - u_wind`. Optional CoP offset produces `τ = r_CoP × F`. |
| `rotor_drag.py` | Physically distinct from body drag (H-force, per-rotor, produces torque from offsets) | Per-rotor `F_i = -0.5 ρ Cd_rot A_rot \|v_rel,i\| v_rel,i`, with `v_rel,i = v_UAV + ω×r_i - u_wind`. Sum torque via `Σ r_i × F_i`. |
| `rotor_wake.py` | Exposes a **thrust multiplier**, not a force | Cheeseman–Bennett `1/(1 - (R/4z)²)` ground effect + momentum-theory inflow. `thrust_correction_factor()` returns a multiplier; `.compute()` returns zeros — deliberately, because the simulator owns thrust. |
| `aerodynamic_torque.py` | Lumped damping, not per-element | `τ = -k1·ω - k2·\|ω\|·ω`. Body frame. |
| `forces.py` | The orchestrator | Evaluates every enabled model, returns `ForceBreakdown` with **per-model** forces and torques. Exposes `.external_force_W()` / `.external_torque_B()` that **exclude** gravity and rotors — this is the double-application guard. `apply_to_isaac_drone()` rotates body torque to world and calls `apply_forces`/`apply_torques` with `is_global=True`. |

**The key "why" behind the orchestrator's design:** `ForceBreakdown` contains **all** forces, including gravity and rotor thrust, so that logging and ablation analysis see the full force budget. But when *applying* to the simulator, only the **external** subset is applied — gravity and rotors are already handled by Isaac Sim. If you applied the full breakdown, gravity would be doubled (drone falls twice as fast) and rotor thrust would be applied twice (drone shoots upward). The two accessor methods make this distinction explicit and testable.

**What "per-model" enables:** §11.4 requires "correlation between wind-induced force and total acceleration" — that requires isolating body drag from rotor drag from aero torque. Without per-model breakdown, this correlation would be on the total force, which mixes a wind-driven term with a rotor-driven term, destroying the diagnostic.

---

### `randomization/` — physics variation across domains

**Why it exists.**
The identifiability study (§12) requires answering: "when the model sees a state drift, is it due to wind or to a physics parameter change?" To answer that, physics parameters must be sampled at the **domain** level (fixed across episodes in a domain) so they're distinguishable from **wind**, which is sampled at the **episode** level (varies within a domain). If both varied independently per episode, the model would never see the same physics twice and would have no way to attribute a persistent error to physics versus wind.

**What it owns.**
- The `Domain` dataclass: one fixed physical configuration.
- Reproducible sampling from the ranges in `domain_randomization.yaml`.
- **Physical validation**: mass > 0, inertia positive-definite, triangle inequality, motor τ > 0, drag ≥ 0, thrust > 0.

**How it works.**
`parameter_sampler.sample_domain()` takes a fresh RNG, samples each parameter (relative or absolute per the YAML spec), and calls `validate_domain()` before returning. `domain_generator.generate_domains()` supports four modes:
- `fixed_physics` — one nominal domain (for wind-only studies).
- `randomized_physics` — N random domains, wind held constant per domain (for physics-only studies).
- `randomized_physics_joint` — N random domains, wind also random per episode (for joint studies).
- `nominal_physics_heldout` — one nominal domain with split-level wind holdout (for OOD studies).

These four modes are the *structural* separation §12.2 requires. Without distinct modes, you can't run Group A (wind-only) vs Group D (joint) cleanly.

---

## Part 3 — Reference and control layer

### `trajectories/` — reference signals

**Why it exists.**
Closed-loop control requires a **reference trajectory** the drone tries to track. The thesis (§6, §11) requires:
1. **Diverse** references (hover, line, circle, figure-eight, GP...) so the action distribution is rich.
2. **Smooth, differentiable** references so the controller's error signals are well-behaved.
3. **Isolated channels** (position, attitude, yaw, thrust) so §11 excitation can perturb one without corrupting the others.

The whole subsystem deliberately **contains no controller math**. A `ReferenceSample` is a *setpoint*, not a *command*.

**Files and their role:**

| File | Role |
|---|---|
| `reference_types.py` | `ReferenceSample` — the *only* type crossing the reference→control boundary. Every field is `Optional` because "no reference for this channel" is a meaningful state. |
| `_shared.py` | Small math (sampling, direction resolution, quat→euler) used across generators. **Never** controller math. |
| `position_references.py` | 11 position families, each stateful (samples parameters on first call, caches per-phase state). Returns `(position, velocity, acceleration)`. This is where trajectory **diversity** comes from — random directions, random speeds, random waypoints. |
| `attitude_references.py` | 7 attitude families. Returns `(roll, pitch)` targets, and for the sine variant, direct `roll_rate`/`pitch_rate` in metadata (because a rate is more physical than integrating a target angle). |
| `yaw_references.py` | Yaw step/sine/random. Returns `yaw` target or `yaw_rate`. |
| `thrust_references.py` | Collective thrust ramps and sinusoids. Returns `thrust_N`. |
| `reference_manager.py` | Dispatches on `spec.type` (accepting the *legacy* `ControllerLibrary` spec format verbatim) and **merges** parallel channels into one sample. This is why it's a manager, not a generator — it must support `[position_spec, yaw_spec]` as a single combined reference. |
| `trajectory_generator.py` | Standalone (stateful-free) generator used for *offline* evaluation and plotting. Different from `ReferenceManager` because the offline version doesn't need per-phase state — you generate the whole trajectory at once. |
| `trajectory_types.py` | Closed-form stateless constructors (hover, line, circle, figure-eight, accel/decel, yaw, attitude). Used by the offline generator; also gives us the **ground-truth** trajectory for diagnostics. |
| `gp_kernel.py` | Periodic + composite kernels with **analytic derivatives** up to second order. The analytic derivatives are essential: sampling `(p, v, a)` jointly requires the cross-covariance blocks `k_fv`, `k_fa`, `k_vv`, etc., and those come from `g'(τ)`, `g''(τ)`, `g'''(τ)`, `g''''(τ)`. Computing those by finite difference would be numerically noisy. |
| `gp_sampler.py` | Builds the 3N×3N joint covariance and Cholesky-samples `(p, v, a)` jointly. Joint sampling (not per-channel sequential) is what guarantees the sampled trajectory is a *consistent* GP realization in all three derivatives — a per-channel approach would produce `p` and `v` that don't satisfy `v = ṗ`. |
| `trajectory_validator.py` | Checks workspace / max velocity / max acceleration. Runs before a trajectory is used. |

**Why GP is a separate module from parametric:** GP trajectories have fundamentally different properties — smooth but with random spectral content, no closed-form velocity, and specific statistical guarantees. They exercise the drone's control loop very differently from parametric trajectories (a GP can request a direction change the parametric set can't produce), so they're essential for action diversity (§11.2).

---

### `core/controller_adapter.py`

**Why it exists.**
The controller must produce **two things**: `target_rate` (3-vector) and `target_thrust` (scalar). OmniDrones' `RateController` is the *inner loop* — it takes those two and produces rotor speeds. We must **never re-implement** the RateController (that's a well-tested embedded component). But we must implement the *outer loop* — position error → desired acceleration → tilt → rate — because that's not in OmniDrones, and it's what makes closed-loop tracking work.

**What it owns.**
- The outer-loop math: `position_error → v_desired → a_desired → roll_desired/pitch_desired → rate_desired`.
- Conversion of a `ReferenceSample` into `(target_rate, target_thrust_N)`.
- The single call into OmniDrones' `RateController`.

**What it does NOT own.**
- Attitude dynamics (OmniDrones).
- Rotor mixing (OmniDrones).
- Motor lag (OmniDrones' `RateController` handles it).

**How it works.**
`ControllerAdapter.compute(state, ref, hover_altitude)`:
1. If `ref.roll`/`ref.pitch` are set → use those as attitude targets.
2. Else if `ref.position` is set → compute `a = kp·(p_ref - p) + kd·(v_ref - v)`, then `roll = -a_y/g`, `pitch = a_x/g`, clamped to `max_tilt_rad`.
3. `rate = kp_attitude · (attitude_target - attitude_current)`, clamped to `max_rate_radps`. If the reference supplied a direct `roll_rate`/`pitch_rate` (sine variant), use that directly.
4. Yaw: `rate = kp_yaw · (yaw_ref - yaw_current)` wrapped to ±π, clamped.
5. Thrust: `mass · (g + kp_z·(z_ref - z) - kd_z·v_z)`, or `ref.thrust_N` if provided.

The math is preserved **bit-for-bit** from the tested `collect_data_wind_hover.position_hold_commands`. The wrapper is not an "improvement" — it's an extraction that keeps the tested behavior while making it usable from any caller.

---

### `excitation/` — deliberate perturbation of the control loop

**Why it exists.**
A major risk (§11.1): the controller's actions become correlated with the wind. In a strong wind, the controller always pushes back; the model learns "wind from X → action pushes -X" instead of "wind from X → residual dynamics". Excitation breaks this correlation by injecting actions that are **independent of both the reference and the wind**.

**What it owns.**
- `DirectActionGenerator` — bypasses the reference/controller pipeline entirely. `random_action` samples `(target_rate, target_thrust)` at fixed intervals with its own RNG. `open_loop` returns raw motor commands.
- `action_excitation.py` — bounded, frequency-controlled perturbations applied to the *reference* (not motor commands). Sinusoidal, multisine, chirp, PRBS. Bounded amplitude so the drone stays in its envelope.
- `reference_perturbation.py` — bounded position/velocity/attitude perturbations at the reference level.
- `excitation_metrics.py` — action covariance, singular values, condition number, spectrum. The §11.4 diagnostics.

**How it's used.**
Three independent RNG streams are maintained: `trajectory_seed`, `wind_seed`, `excitation_seed`. These are logged in `EpisodeMetadata`. A model that's learned to correlate actions with wind *cannot* be blamed on the RNG streams being coupled, because they're provably independent.

The `random_action` mode (§11.3 Experiment E) is precisely `DirectActionGenerator` with a fixed seed, replayed across multiple wind conditions — this is the fixed-action counterfactual baseline.

---

## Part 4 — Simulation orchestration

### `core/simulation_interface.py`

**Why it exists.**
The simulator lifecycle is: `launch → enable extensions → spawn → reset → read → apply → step → close`. Every step of this involves version-sensitive API calls (OmniDrones 4.1 vs newer, `omni.isaac.core` vs `isaacsim.sensors.physics`, `drone.set_velocities` vs `view.set_velocities`). Without a single boundary, every episode script would duplicate this fragile logic and one bad version bump would break everything.

**What it owns.**
- The entire simulator lifecycle.
- **Lazy imports** (so CI can import it without Isaac Sim).
- **Dependency injection**: `state_fn`, `reset_fn`, `force_fn` can be overridden — so if your installed version's API differs from the default, you pass your tested function and nothing else needs to change.
- The extension enablement (`_enable_isaac_extensions`) — a critical fix because `init_simulation_app` launches Kit with a minimal config and doesn't activate `omni.isaac.core`.

**How the lazy import strategy works.**
Module-level imports are just `numpy`, `torch`, and `state_extractor`. Every `omni.isaac.*`/`omni_drones` import is inside a method. This is why `simulation_interface.py` can be imported in a CI environment without Isaac Sim installed — and why `_default_state_fn`, `_default_reset_fn`, `_default_force_fn` can be unit-tested with mock objects. When Isaac Sim is present, `ensure_ready()` triggers `_setup()` which does the real imports and constructs the scene.

**The specific fixes accumulated over time:**
- `sim_app_config()` — `init_simulation_app` expects a dict with `headless`; the dataclass provides it.
- `_enable_isaac_extensions` — `omni.isaac.core` is *not* enabled by default after `init_simulation_app`.
- `_default_reset_fn` — `drone.set_velocities` takes a single `(n, 6)` tensor, not `velocities=`/`angular_velocities=` kwargs.
- `pause()/play()` wrappers — pose/velocity writes must not race the physics step.

Each of these is a version-specific landmine that would otherwise kill the run with a cryptic error.

---

## Part 5 — Data pipeline

### `data/metadata.py`

**Why it exists.**
Every downstream analysis (§10.4, §12, §16, §17) needs to *find* episodes by their counterfactual group, domain, scenario, wind profile, gust class, excitation type. If these identifiers live only in ad-hoc dicts, no query is reliable. The metadata record is the on-disk contract.

**What it owns.**
- The `EpisodeMetadata` dataclass — identifiers, counterfactual grouping, seeds, per-subsystem parameter blocks, `action_alignment`, `episode_status`.
- Constants (`ACTION_ALIGNMENT_*`, `COUNTERFACTUAL_MODE_*`) that prevent stringly-typed comparisons.
- JSON I/O (`save/load/iter/load_registry`).

**How `__post_init__` enforces invariants.**
The constructor itself refuses to build an invalid record: an episode with `counterfactual_mode` set *must* have `counterfactual_group_id` and `counterfactual_episode_id`; `action_alignment` must be one of the recognized values; `episode_status` must be one of the recognized values. This catches mistakes at write time, not at analysis time six months later.

**Why `split_keys()` exists.**
`SplitSpec` accepts a `group_key` and every split scheme needs a stable set of split-unit identifiers. `split_keys()` returns `{episode_id, domain_id, scenario_id, counterfactual_group_id}`, and `validate_no_leakage` checks all four. Without a single source of truth for "which keys must not cross splits," one scheme would forget to check `counterfactual_group_id` and leak a pair.

---

### `data/resampler.py`

**Why it exists.**
The simulator runs at 200 Hz. The model wants 20 Hz (§15). You cannot just decimate — the resampler must:
1. **Anti-alias** before decimation (Butterworth zero-phase), else high-frequency content aliases into the band of interest.
2. **SLERP** quaternions (linear interp of quaternions gives non-unit quaternions).
3. **Refuse to interpolate across episode boundaries** — the single-episode API is the enforcement mechanism.

**What it owns.**
- Per-episode downsampling to a uniform grid.
- Channel-aware interpolation: quaternions via SLERP, angular channels unwrapped, everything else linear (or PCHIP if `interpolate_scalar="cubic"`).

**Why "one episode at a time" is not a limitation but a design.**
If the API took a list of episodes, a careless implementation would interpolate across the boundary and inject fake dynamics. By forcing the caller to iterate, the boundary is structurally impossible to cross. The cost is trivial (a for-loop in the caller); the benefit is that the dataset is never silently wrong.

---

### `data/dataset_builder.py`

**Why it exists.**
The model's learning objective is `ŝ_{t+1} = f_θ(s_{t-H+1:t}, a_{t-H+1:t})`. It needs:
- A window of **H** past states and actions.
- A **target** state at horizon `h` (from `{1, 5, 10, 20}`).
- **Only** primary fields in the input — wind, forces, physics parameters must NOT be in the sample.

The builder is the place where "episode" becomes "training sample," and where §12.1's primary/aux separation is physically enforced.

**What it owns.**
- `BuiltDataset`: `state_history (N, H, 13)`, `action_history (N, H, A)`, `future_states (N, P, 13)` where `P = max(horizon)`.
- `AuxiliaryTable`: everything else, keyed by `(episode_id, time_index)`. **Cannot** be joined by accident because `ModelReadyDataset` returns only the primary fields.
- `align_arrays`: handles both `state_then_action` (canonical) and `action_produces_state` (legacy).
- `split_samples` with 8 schemes, all group-aware.

**How the horizon indexing works.**
If H=10, P=20, and the episode has N=100 resampled steps, then sample `i` (for `i = 0..N-H-P`) is:
- `state_history[i]` = `state[i : i+H]`
- `action_history[i]` = `action[i : i+H]`
- `future_states[i, h-1]` = `state[i+H-1+h]` for `h = 1..P`

This indexing ensures `action[k]` in the history is the action that produced `state[k+1]`, so `ModelReadyDataset` at horizon 1 uses `action_history[-1]` (the most recent action) to predict `future_states[0]` (the next state). That's exactly the thesis's learning objective.

**Why group-aware splits.**
The default `random_by_group` scheme shuffles **groups** (counterfactual groups, or `domain::scenario` when no group exists) and assigns whole groups to train/val/test. If you split by sample, two members of the same counterfactual pair could land in different splits — the model would train on one and evaluate on the other, and the "generalization" it shows is leakage, not learning.

---

### `data/dataset_validator.py`

**Why it exists.**
The builder can produce well-shaped tensors that are silently wrong — an episode boundary crossed, a non-finite value slipped in, a split that leaks a counterfactual pair. The validator is a hard gate before training, catching exactly the classes of error that are invisible in shapes but catastrophic in results.

**What it checks:**
- **Per-episode**: t strictly increasing, all primary channels 2-D, quaternions unit-norm, cf metadata complete, mass > 0.
- **Per-dataset**: history/horizon shapes match config, no NaNs, time_index + P within episode length.
- **Primary/aux separation**: no `AUXILIARY_KEYS` appears in a `ModelReadyDataset` sample.
- **Split leakage**: no `episode_id`, `domain_id`, `scenario_id`, or `counterfactual_group_id` appears in two splits.
- **Metadata consistency**: cf groups have ≥2 members, single domain, single scenario.

**How it works.** `assert_valid(ds)` runs everything; the individual functions are exposed so tests and diagnostics can call subsets. This is why the test suite can be comprehensive without running the simulator — each invariant is a pure function of the dataset.

---

### `data/dataset.py`

**Why it exists.**
Torch's `Dataset` protocol is what `DataLoader` consumes. The research needs two things this module provides:
1. `ModelReadyDataset` — guaranteed to return only primary fields, regardless of what's in `BuiltDataset`.
2. `AuxiliaryView` — opt-in access for evaluation (`wind_velocity_W`, forces, physics parameters). This is a *separate class* specifically so the training pipeline cannot accidentally pull labels.

**How the separation is enforced structurally.**
`ModelReadyDataset.__getitem__` returns a dict with a fixed set of keys: `state_history`, `action_history`, `target_state`, `timestamp`, `horizon`, `episode_id`, `domain_id`, `scenario_id`, `counterfactual_group_id`. It's not "we promise not to include wind" — it's "there is no code path that includes wind." Adding wind to the model's input would require editing `__getitem__`, which would trigger the `validate_primary_aux_separation` test and fail CI.

`SyntheticUAVDataset` is preserved for smoke tests. It does *not* obey the primary/aux separation because it predates it, but its role in the codebase is explicitly labeled "smoke test only."

---

### `data/paired_dataset_builder.py`

**Why it exists.**
§10.4 requires per-pair divergence metrics: `Δp, Δv, Δa, Δatt (deg), Δω, Δaction`. Mode B requires verifying that the replayed actions are bitwise identical across the pair (otherwise the "fixed action" counterfactual is invalid). This module computes all of that from the saved episodes.

**How `_validate_mode_b_actions` prevents the silent Mode B failure.**
If the replayed actions differ by even `1e-9`, the counterfactual is invalid — the difference in trajectories could be due to the action difference, not the wind difference. The validator raises, and `build_paired_metrics` catches nothing — a Mode B violation fails the whole analysis. This is the correct behavior: better to abort than produce a misleading result.

**Why quaternion differences use axis-angle, not component-wise.**
`q1 - q2` is not a meaningful "attitude difference" — two quaternions representing the same rotation can have opposite signs (`q` and `-q`), giving `2q` as the difference. `_quat_delta_axis_angle` computes `q2 ⊗ q1⁻¹`, forces `w ≥ 0` (shortest arc), and returns `angle · axis`. The test suite verifies this against `q` and `-q` giving zero difference.

---

## Part 6 — Experiment layer

### `experiments/counterfactual_runner.py`

**Why it exists.**
§10.2 defines two counterfactual modes:
- **Mode A (closed-loop)**: same scenario, same physics, different wind → controller naturally produces different actions.
- **Mode B (fixed-action)**: same scenario, same physics, same **actions**, different wind → isolates the pure effect of wind from feedback.

Both must produce datasets tagged with `counterfactual_group_id` so `split_samples` keeps pairs together.

**How Mode B works.**
Run a baseline episode (Mode A of the first intervention). Capture the action sequence. For each other intervention, replay those actions exactly, with the wind switched. The physics integrates the same rotor speeds under different wind, producing different states. The `action_replay` argument to `run_episode` skips the controller entirely.

**Why the two modes are not equivalent and must not be conflated.**
In Mode A, the controller's response changes the action distribution — so `Δs` measures the *combined* effect of wind + controller. In Mode B, the actions are fixed — so `Δs` measures the *pure* effect of wind. §10.4 explicitly says "do not interpret closed-loop paired differences as the pure direct effect of wind." The two modes are separately analyzed; the report labels which mode each pair belongs to.

---

### `experiments/identifiability_runner.py`

**Why it exists.**
§12.2 defines 6 controlled experiment groups (A–F). Each differs in **which axis varies** (wind, physics, or both) and **whether it's ID or OOD**. Without an explicit runner, these groups would be an ad-hoc mix of configs and would be impossible to reproduce.

**How it structures the experiments.**
Each group is a `(physics_mode, wind_mode)` pair plus a set of scenarios:
- A: `fixed_physics` + `wind_varied`
- B: `randomized_physics` + `wind_zero`
- C: `fixed_physics` + `wind_varied` (numerically different from A: A uses one trajectory, C uses many)
- D: `randomized_physics_joint` + `wind_varied`
- E: `fixed_physics` + force-model ablations
- F: train-on-subset, eval-on-holdout

Groups are tagged in metadata (`extra["identifiability_group"] = "A"`), so `split_samples` can hold out entire groups for cross-group OOD analysis.

---

### `experiments/ablation_runner.py`

**Why it exists.**
§17 requires comparing model performance across force-model configurations, trajectory strategies, physics randomization, etc. Each ablation produces a dataset piece tagged with `aerodynamics_config` (or equivalent), so the analysis can group results.

**How it works.** It's mostly a matrix of configs — for each `(force_model, trajectory_strategy, physics_mode, excitation_strategy)` tuple, run the counterfactual or identifiability runner with the appropriate config. The "runner" part is orchestration, not new physics.

---

### `experiments/evaluation_split.py`

**Why it exists.**
`split_samples` accepts a `SplitSpec`; `evaluation_split` provides named constructors for the standard schemes (`make_split(ds, "ood_wind_magnitude", threshold=4.0)`) so callers don't have to know `SplitSpec`'s constructor signature. It also provides persistence (`save_split`/`load_split`) so splits are reproducible across sessions — critical because if the split changes, all reported numbers become incomparable.

---

## Part 7 — Analysis layer

### `analysis/action_wind_correlation.py`

**Why it exists.**
§11.4 requires detecting confounding between actions and wind. A single zero-lag Pearson correlation is not enough — wind affects the drone before the controller responds, so the correlation must be checked at **multiple lags** (cross-correlation over `[-τ_max, +τ_max]`). And Pearson only captures *linear* correlation; Spearman (rank correlation) captures monotonic non-linear relationships.

**How it computes the diagnostics:**
- **Pearson**: `E[(a - μ_a)(w - μ_w)] / (σ_a σ_w)`.
- **Spearman**: Pearson on ranks.
- **Cross-correlation at lag ℓ**: `E[(a_t - μ_a)(w_{t+ℓ} - μ_w)] / (σ_a σ_w)` — sweeping ℓ reveals whether actions *precede* or *follow* wind.
- **Action–wind-force correlation**: same diagnostics but using `\|F_wind\|` instead of `\|w\|`. This distinguishes "action predicts wind speed" from "action predicts wind force" — the latter is what actually confounds the dynamics model.
- **Reference–wind correlation**: if wind correlates with the reference trajectory, then wind isn't independent of the task — this is a design bug caught by diagnostics.

**The module's docstring explicitly states** that low correlation does *not* prove causal independence. It's a diagnostic, not a certificate. This honesty is important for the thesis's claims.

---

### `analysis/wind_identifiability.py`

**Why it exists.**
§12.4 requires five diagnostic families:
- **Excitation coverage**: distribution of states/velocities/actions across the dataset.
- **Wind coverage**: magnitude, direction, gust duration.
- **Confounding**: correlations between wind and everything else (actions, physics parameters, reference).
- **Sensitivity**: `J_θ = ∂y / ∂θ` via finite differences on the simulator or on a model. This tells you whether different wind parameters produce distinguishable responses — if `J_θ` has near-identical columns for two wind parameters, they're not identifiable from that response.
- **Rank/conditioning**: singular values of `J_θ`; condition number tells you how close to singular the parameter-to-observation map is.

**How finite-difference sensitivity works.**
Fix a state, a set of actions, a wind configuration `θ`. Perturb `θ_i` by `ε`. Re-run one step. Measure `Δy / ε`. The result is one column of `J_θ`. Repeat for each parameter. The resulting matrix is analyzed for rank and conditioning.

**The critical caveat** (documented in the module): this is *local* sensitivity of a *specific* trajectory under *specific* conditions. Identifiability is a global property that no single trajectory can demonstrate. Condition number near 1 means the parameters *are distinguishable on this response*; a high condition number means they *might not be* — but a well-conditioned matrix doesn't guarantee identifiability across the full distribution.

---

### `analysis/counterfactual_analysis.py`

**Why it exists.**
`paired_dataset_builder` produces per-pair `PairedMetrics`. The analysis layer aggregates them: per-group distributions of divergence, plots of divergence over time, consistency checks (is the predicted divergence from a model close to the actual divergence?). This is the difference between "here are 200 divergence arrays" and "here's what the divergence tells us."

**How it aggregates.** For each counterfactual group, aggregate the pairwise metrics into a group-level summary: `Δp_max`, `Δp_mean`, `Δatt_max`, etc. Then aggregate across groups to characterize the dataset's counterfactual coverage. The consistency check compares the model's predicted divergence (from `world_model.predict`) against the actual divergence from the saved episodes — this is the counterfactual prediction evaluation §17 lists.

---

### `analysis/diagnostics.py`

**Why it exists.**
§18 lists 12 required diagnostic plots. `diagnostics.py` is the single entry point that produces them all from a dataset directory.

**What each plot tells you:**
1. **Trajectories** — reference vs. actual; if actual never approaches reference, control is broken.
2. **Position/velocity** — smoothness check; jumps indicate bad resampling or bad control.
3. **Wind profiles** — is the wind what you asked for? Any unexpected steps?
4. **Per-force breakdown** — is drag ~2× thrust? Is rotor wake ever non-zero when it shouldn't be?
5. **Total force/torque** — sanity against physics expectations.
6. **Tracking errors** — persistent errors signal physics mismatch; growing errors signal instability.
7. **HF vs. resampled** — visual check that resampling preserved the signal.
8. **Counterfactual comparisons** — the plot that lets a human see the effect of wind.
9. **Action–wind correlation heatmap** — the confounding view.
10. **Action spectra** — is excitation broadband (good) or resonant (bad)?
11. **Coverage** — wind magnitude × direction histograms, physics ranges.
12. **Sensitivity matrices** — heatmaps of `J_θ`.

---

## Part 8 — Learning layer

### `models/`

**Why it exists.**
The thesis's core claim is that a **physics-prior latent world model** beats both pure physics and pure neural baselines. The `models/` package contains all three, plus the composition, so ablations are apples-to-apples.

**How the pieces compose:**

| Model | What's in it | Why |
|---|---|---|
| `encoder.py` | `(s_history, a_history) → z` | Compression from high-dim observation window to latent. This is where any "wind info" would be inferred *if it's inferable* — but the encoder only sees history. |
| `latent_dynamics.py` | `(z, a) → z'` | Pure neural dynamics. The "no physics" baseline. |
| `physics_prior.py` | `(s, a) → s'` from rigid-body equations | Analytical prior. No learning. |
| `residual_dynamics.py` | `physics_prior(s, a) + net(s, a)` | Physics + learned residual. The hybrid. |
| `world_model.py` | Encoder + latent dynamics + decoder, optional physics residual | The full model. |

**Why the encoder is separate from the dynamics.**
The latent space is where "hidden state" lives — including, potentially, the wind. If the encoder learns to compress `(s_history, a_history)` into a latent that captures "there's a 3 m/s wind from the north," then the latent dynamics can use that. The dataset must *enable* this — which is why excitation must be diverse (the encoder can't infer wind from a single hover) and why physics variation must be at the domain level (the encoder needs to see the same physical drone consistently to attribute persistent behavior to physics).

**Why physics-prior is a *residual*, not a *replacement*.**
The physics prior alone is biased (it uses calibrated but not exact coefficients). The learned residual corrects the bias, but is small and easier to learn. A pure neural model must relearn the physics from scratch, needing far more data. This is the hypothesis the ablations test.

---

### `losses/losses.py`

**Why it exists.**
Training requires losses that respect the model structure:
- `mse_state_loss` — the primary objective.
- `physics_prior_loss` — penalizes deviation from the analytical prior.
- `latent_regularizer` — prevents latent collapse.

**Why physics-prior loss is not just MSE on the residual.**
The physics prior is an *inductive bias*: if the true dynamics is nearly the physics prior (in ID, small wind), the model should learn to trust the prior. If it's far (in OOD, large wind), the model should learn a large residual. The loss function balances these — the specific form is a design decision that affects whether the model generalizes to OOD wind.

---

### `training/train.py`

**Why it exists.**
The training loop must:
1. Consume `ModelReadyDataset` — never touch aux.
2. Validate on the correct split (from `evaluation_split`).
3. Support multi-horizon objectives (§15's `{1, 5, 10, 20}`).
4. Support curriculum (`curriculum.py`).

**Why validation splits matter here specifically.**
If the validation split shares a counterfactual group with training, the model will "generalize" to a val episode that's a *paired replay* of a training episode under slightly different wind — and the val loss will be artificially low. `split_samples` + `validate_no_leakage` are the guards.

---

### `evaluation/`

**Why it exists.**
The ablation table (§17) reports:
- 1-step and multi-step RMSE
- Error growth with horizon
- OOD performance
- Cross-domain generalization
- Counterfactual consistency
- Reliable horizon

Each of these is a function of predictions vs. ground truth. `metrics.py` implements each; `reliable_horizon.py` is a summary statistic; `rollout.py` implements autoregressive multi-step evaluation.

**Why rollout is separate from metrics.**
Multi-step prediction can be evaluated two ways:
- **Teacher forcing**: feed the *true* history, predict the next state. Used for one-step metrics.
- **Rollout**: feed the *predicted* state back, predict the next, repeat. Used for long-horizon metrics.

A model can be good at teacher-forced prediction and terrible at rollout (small errors compound). The rollout evaluation is the "real" long-horizon test that §17 cares about.

---

### `baselines/`

**Why they exist.**
Every research result must have baselines to compare against. The three baselines are:
- `physics_only` — analytical prior only. Lower bound.
- `pure_neural` — encoder + latent dynamics + decoder, no physics. What you get without the physics-prior idea.
- `residual_model` — physics + learned residual. The full hybrid.

The thesis's claim is that `residual_model` beats both. Without these three, the claim is untestable.

---

### `visualization/plots.py`

**Why it exists.**
Every plot in §18 is generated by *some* code. Keeping plotting separate from computation means:
1. Plots can be regenerated without recomputing metrics.
2. Plot bugs don't contaminate metric code.
3. The plotting style is consistent across all outputs.

**Why it's a thin layer.**
Each function is small — `plot_forces(t, forces_dict, path)` and similar. The complexity is in the data, not the plotting. A thin plotting layer is a *feature*, not a limitation.

---

## Cross-cutting design themes

Three patterns repeat throughout, and understanding them makes every module's design obvious:

**1. Physical invariants live at the boundary of the module that owns them.**
- `config_loader` asserts `l == 2·arm_lengths[0]`.
- `hummingbird.py` asserts the geometry derivation matches.
- `parameter_sampler` validates inertia positive-definite.
- `dataset_validator` asserts quaternions are unit-norm.

Each module checks its own invariants. If a module's output is invalid, the module that *produced* the output raises, not the module that consumes it. This makes errors local and diagnosable.

**2. Optional channels are the mechanism for isolation.**
`ReferenceSample` uses `Optional` fields so channels are independently perturbable. `EpisodeMetadata` uses nullable fields so non-counterfactual episodes don't need to fake a `counterfactual_group_id`. Everywhere the design wants "can vary independently," the type is `Optional`.

**3. RNG streams are always explicit and logged.**
Trajectory seed, wind seed, excitation seed, domain seed, initial-condition seed — every one is a named parameter, logged in metadata. Never "the global `numpy.random` state." This is what makes counterfactual pairs, ablations, and reproducibility possible.

Every module in the codebase answers "why does this exist" with a specific physical or experimental requirement. The design isn't trying to be elegant — it's trying to make the experiments in the thesis *possible*. A module that isn't necessary for an experiment wouldn't be here; a module whose boundary is unclear is a design smell (and every module here has one responsibility).