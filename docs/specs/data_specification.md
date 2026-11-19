# Data Specification — PhysiWorldUAV

Status: **Part A is the concrete specification** of what the pipeline produces today (schema v2.0, pilot `physiworld_pilot_v0.5.0`, measured 2026-11-17). **Part B is the original requirements text** (the thesis proposal's data requirements), kept unchanged for traceability; where the implementation differs, the table in A.9 says so.
Authoritative field list: `docs/guides/dataset_schema.md` (generated from `src/physiworld_uav/data/schema.py`).

Items marked **PROPOSED** are choices I recommend but that need your approval (they affect what the model learns).

---

# Part A — Concrete specification

## A.1 Learning problem

| Item | Specification |
|---|---|
| Task | Multi-step prediction of quadrotor state under wind, given past states/IMU/actions and **future actions only** (no future observations, no wind). |
| Model form | `ds/dt = f_phys(s,u) + d_θ(h,s,u)` (thesis) |
| Input window (history H) | 10 model steps = 0.2 s |
| Prediction horizon (P) | 20 steps = 0.4 s for training; evaluation rollouts up to 100 steps = 2.0 s |
| Model rate | 50 Hz (Δt = 0.02 s); physics rate 200 Hz (dt = 0.005 s), decimation 4 (states are decimated, never filtered) |
| Prediction type | Multi-step, autoregressive rollout from an encoded history |
| Primary metrics | Position error vs horizon, reliable imagination horizon (first-failure, 0.5 m), attitude (quaternion geodesic) error; per split role (see `training_pipeline.md`) |

## A.2 Features, dimensions, types, units, frames

Conventions: world ENU, body FLU, Hamilton quaternion wxyz (body→world). All model-ready arrays are float32.

| Group | Layout | Dim | Unit | Frame | Used as |
|---|---|---|---|---|---|
| State `s` | p_W[0:3] · v_W[3:6] · q_wxyz[6:10] · ω_B[10:13] | 13 | m · m/s · – · rad/s | world / world / body→world / body | Target and recurrent state |
| Observation `o` | IMU specific force[0:3] · IMU gyro[3:6] | 6 | m/s² · rad/s | body | Encoder input (history only) |
| Action `u` | `rotor_speed_mean` (mean rotor angular velocity over the control interval, after motor lag; decision D3) | 4 | rad/s | rotor | Input, including future actions |
| Auxiliary (never a model input) | wind_W, wind_applied_W (3), external_force_W (3), external_torque_B (3) | – | m/s · N · N·m | world / body | Oracle baseline, diagnostics, OOD labels |
| Episode metadata | split role, ood category, data type, domain factors, wind plan, controller, seeds, config hash | – | – | – | Splitting and evaluation, **not** an input |

Observed ranges in the pilot (141 eligible episodes, all finite): position x −19.5…17.6 m, y −12.4…13.6 m, z 0.59…4.7 m; speed components within ±6.3 m/s; ω_B within ±1.8 rad/s; rotor speed 124…672 rad/s; specific-force z 5.3…12.3 m/s². Pilot per-channel standard deviations (for normalisation checks): p (3.5, 2.6, 0.36), v (0.73, 0.64, 0.13), ω (0.20, 0.17, 0.13), rotor speed ≈ 22.5.

Sample (what the dataloader returns, `data/dataset.py`), for current model step t:

| Key | Shape | Content |
|---|---|---|
| `state_history` | (10, 13) | s_{t−9…t} |
| `obs_history` | (10, 6) | o_{t−9…t} |
| `action_history` | (10, 4) | u_{t−10…t−1} |
| `future_actions` | (20, 4) | u_{t…t+19} |
| `future_states` | (20, 13) | s_{t+1…t+20} (target) |
| `dt` | () | 0.02 s |

## A.3 Temporal alignment

* Physics instants t_k = k·dt. States and IMU are *instantaneous* at t_k (N+1 rows); actions are *held* over [t_k, t_k+1) (N rows).
* Action u_j produces s_{j+1}: `future_states[k]` results from `future_actions[0..k]`. Actions are exactly piecewise constant at the model rate.
* The IMU at t_k is the sensor frame at t_k (`imu_valid` flags fresh frames; the validator rejects episodes with too many stale frames).
* A window is valid for t ∈ [H, M − P]; no padding is used, so no window crosses an episode boundary.

## A.4 Missing values and invalid data

* No NaN/Inf is allowed in any field except those declared optional (e.g. reference fields in open-loop episodes, `collective_thrust_cmd` when the mode has none). The pilot has 0 NaNs in the model-ready state, observation and action arrays.
* Episodes with `status != completed` (ground contact, `terminated_ground`) are **excluded** from the model-ready view by default (`dataset.builder.statuses: [completed]`): 141 of 147 pilot episodes are eligible. They remain in the dataset, flagged.
* Episodes whose force model left its applicability envelope are excluded (`applicability_ok`); the pilot has none.
* Open-loop `body_wrench` excitation episodes are excluded from the model-ready view (they are not rotor-driven).

## A.5 Data types (identities kept in metadata, never mixed silently)

| Type | `data_type` | Purpose | Pilot episodes |
|---|---|---|---|
| Closed-loop trajectories | `closed_loop` | Control and dynamics learning | 117 |
| Open-loop excitation | `open_loop` | System identification (rotor / collective-rate doublets) | 30 |
| Counterfactual pairs | group fields `counterfactual_*` (Mode A closed loop, Mode B fixed-action replay) | Causal comparison; same initial state, different wind or physics | 24 of 99 groups have siblings |
| Physics validation | `physics_validation` | Case A (no wind) and Case C (mass +5 %) | 12 |

## A.6 Data distribution

**Wind (world frame, mean horizontal speed at the vehicle's CoM).** Regimes are in `config/wind/regimes/`, and the validator rejects an episode whose logged wind violates its regime.

| Regime (role) | Mean speed | Direction | Temporal / gust structure | Peak limit |
|---|---|---|---|---|
| `id` (train, val_id, test_id) | 0–3 m/s (limit 4) | azimuth 0–360°, elevation ±5° | steady, smooth gusts, steps, slow sinusoid (0.05–0.3 Hz), OU (τ 2–5 s) | 4 m/s |
| `ood_magnitude` | 4.2–7.5 m/s (limit (4, 8]) | as ID | as ID | 10 m/s |
| `ood_direction` | 1.5–3.5 m/s | elevation ±(20–60)° (ID is ±5°) | steady, OU | 4 m/s |
| `ood_temporal` | 0–2.5 m/s | as ID | fast sinusoid 0.6–1.5 Hz, fast steps, short-τ OU (0.2–0.8 s) | 4 m/s |
| `ood_gust` | 0–2 m/s | as ID | sharp gusts, rise 0.05–0.18 s | 4 m/s |
| `none` (physics validation) | 0 | – | – | 0 |

**Vehicle domain (relative factors around the nominal Hummingbird, `config/domain/randomized.yaml`).**

| Factor | ID range | OOD physics shell |
|---|---|---|
| mass | ±10 % | 10–20 % |
| thrust constant (force_scale) | ±10 % | 10–20 % |
| rotor drag scale, body drag scale | ±10 % | 10–20 % |
| motor time constant | ±20 % | 20–40 % |
| inertia shape (x, y, z), km/kf ratio | ±5 % | – |
| Held-out corner | – | mass −50 %… with force_scale −50 % (`ood_combination`) |

Sampling is Sobol; thrust-to-weight ≥ 2 is enforced.

**Trajectories and excitation.** Closed-loop: hover, translation, line, vertical, accel/decel, circle, sinusoidal, waypoints, combined, yaw step/sine, roll/pitch doublets, attitude-combined. Held out for OOD: helix and figure-eight (`ood_trj`). Open loop: rotor doublets and rate doublets. Reference ranges and feasibility limits: `docs/guides/scenarios_and_experiment_types.md`.

**Noise.** The stored IMU is clean. Measurement noise/bias is applied at **load time** (decision D7), so one dataset serves several noise levels. **Implemented** in `data/noise.py` (`ImuNoise`, a `ModelReadyDataset` transform): per window, `o = o_clean + b + n`, `b ~ N(0, σ_bias²)` per channel (constant over the 0.2 s history), `n ~ N(0, σ_white²)` per sample. Presets (PROPOSED values, not truth; report the preset with every result): `none`; `low` (accel σ 0.05 m/s², bias 0.02; gyro σ 0.005 rad/s, bias 0.002); `nominal` (0.10 / 0.05; 0.010 / 0.005); `high` (0.20 / 0.10; 0.020 / 0.010). Evaluation noise is reproducible (seeded by episode id and time index); training noise is redrawn each time. `training.run --set noise=nominal`, `evaluation.run --noise nominal`.

**Aerodynamic truth.** Rotor drag d = 0.30 s⁻¹, body drag with per-axis C_D (1.1, 1.1, 1.3), validated envelope μ ≤ 0.20 (`docs/models/aerodynamics.md`).

## A.7 Splits and held-out conditions

* Roles are assigned **per counterfactual group** at plan time and stored in metadata. Nothing is reassigned afterwards, and the validator fails if a group, domain or scenario is shared across roles (leakage check).
* Roles: `train`, `val_id`, `test_id`, and `val_*` / `test_*` for ood_magnitude, ood_direction, ood_temporal, ood_gust, ood_physics (includes the unseen-combination corner) and `test_ood_trajectory`; `physics_validation` is separate.
* Validation roles come from the same shifted distribution as their tests, but are separate groups. **Use `val_ood_*` only for model selection if you accept that OOD is then no longer fully unseen; PROPOSED: select on `val_id` only and report `val_ood_*` as diagnostics**, so the OOD claim stays clean.
* Sample-efficiency study: nested group subsets of the training pool at 10 / 25 / 50 / 100 % (`data/splits.py`), no extra generation.
* Wind ground truth, wrench and parameters are **never** model inputs (`AuxiliaryView` is evaluation-only, and the dataset raises if auxiliary keys leak into a sample).

Pilot, model-ready (141 episodes, window stride 1, H = 10, P = 20):

| Role | Episodes | Windows |
|---|---|---|
| train | 63 | 24,718 |
| val_id | 11 | 4,520 |
| test_id | 14 | 6,198 |
| val_ood_{magnitude, direction, temporal, gust, physics} | 3, 2, 2, 2, 2 | 1,450 · 1,046 · 1,291 · 1,061 · 811 |
| test_ood_{magnitude, direction, temporal, gust, physics, trajectory} | 7, 4, 4, 4, 7, 4 | 4,055 · 2,249 · 1,929 · 2,505 · 3,878 · 2,636 |
| physics_validation | 12 | 6,765 |
| **Total** | **141** (6 ground-contact episodes excluded) | **65,112** |

Windows at stride 1 overlap almost completely: the effective training size is **63 trajectories**, not 24,718 samples. Pilot episodes are 2.0–15.6 s (median 10.9 s, 101–781 model steps after decimation for completed ones).

## A.8 Data volume and storage

Per episode ≈ 0.49 MB (HDF5, gzip level 4: 72.4 MB for 147 episodes, includes all physics-rate fields). Estimates for the full dataset are in `data_generation_pipeline.md`. Targets from the thesis requirements (Part B §3): 200–500 training trajectories, 50–100 validation, 50–100 per OOD regime. The pilot has 63 / 11 / 2–7 per OOD role. **The size of the final dataset is a decision for you** (it fixes the Run 6 generation time).

## A.9 Differences between Part B and the implementation

| Part B says | Implemented | Why / effect |
|---|---|---|
| `.npz` files per trajectory, `train/ val/ test/<regime>/` directories | HDF5 shards (`shard_XXX.h5`, `/episodes/<id>/<group>/<field>`) plus `manifest.jsonl` and `index.json`; the split is a **role label** in metadata | One file per 200 episodes, atomic index, gzip; the role label avoids copying files |
| Log at 200 Hz, subsample later | Done: physics-rate fields kept, model-ready view decimated 4× | – |
| `a_imu, ω_imu` in the encoder; wind/f_ext as oracle only | Done (`observations/`, `environment/`, `ground_truth/wrench/`) | – |
| Disturbance torque τ_ext | Logged as `external_torque_B` | – |
| Turbulence parameters σ, L | Stored in each episode's wind plan (profile parameters; Dryden σ/L where used) | Read via `AuxiliaryView.wind_plan` |
| Vision | Not generated | Extension only (M4) |
| OOD regimes: magnitude, direction, temporal, gust, physics | All five plus trajectory-family and unseen-combination OOD | – |
| Identifiability experiment design (FIM etc.) | Not implemented; open-loop doublets and counterfactual pairs address it qualitatively (`docs/datasets/problems/identifiability.md`) | Listed in Part B's closing notes; PROPOSED as a later module |
| Model applicability registry | Implemented (`forces/registry.py`); out-of-envelope episodes are excluded | – |
| Separate ground_truth / observations / actions | Done | – |

## A.10 Known data limitations (as of the pilot)

* Four pilot episodes are shorter than 110 model steps and cannot provide a 100-step evaluation window; evaluation must skip windows that do not fit (do not pad).
* Six episodes ended on ground contact (open-loop programs at ≈ 2 m altitude under wind). They are flagged and excluded from training data; the contact itself is outside the physics prior.
* Attitude-excitation scenarios use a deliberately slow position hold, so their position error is large by design (`docs/audit/run5_report.md`).
* IMU noise is applied at load time only when a preset is chosen (A.6); the pilot baselines in `docs/audit/run5_report.md` were trained and evaluated without noise unless stated.
* **IMU vs ground-truth acceleration:** in high-angular-rate phases the IMU specific force deviates from Δv/dt (p99 1.65 m/s², max 4.7 m/s²) while the ground-truth force balance holds (p99 0.03 m/s²). Cause not yet verified (a sensor lever arm is the lead). See the addendum in `docs/audit/run5_report.md`.
* Resumed episodes ran in a second process; cross-process bit-exactness has not been checked.

---

# Part B — Original requirements (unchanged)

## Data Requirements for Training the World Model
Based on your proposal, the world model has several trainable components — a latent encoder, a physical state decoder, a learned residual network, and a latent transition model — each of which requires specific data streams. Below is a complete specification, organized by purpose.

---

## 1. Data Streams Required

The simulator (Isaac Sim / Isaac Lab) must log the following at every timestep \(t\), synchronized across all channels.

### 1.1 Physical State Ground Truth (targets for the state decoder and multi-step loss)

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Position | \(\mathbf p_t\) | \(\mathbb R^3\) | State prediction loss |
| Linear velocity | \(\mathbf v_t\) | \(\mathbb R^3\) | State prediction loss |
| Attitude (quaternion) | \(\mathbf q_t\) | \(\mathbb R^4\) | State prediction loss, \(d_{SO(3)}\) |
| Angular velocity | \(\boldsymbol\omega_t\) | \(\mathbb R^3\) | State prediction loss |

This is the **supervision signal** for \(\mathcal L_{\mathrm{state}}\) and \(\mathcal L_{\mathrm{multi}}\). Without it, the decoder \(D_s(h_t)\) cannot be trained.

### 1.2 Proprioceptive Observations (inputs to the encoder \(E_P\))

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Accelerometer | \(a_t^{\mathrm{IMU}}\) | \(\mathbb R^3\) | Encoder input |
| Gyroscope | \(\omega_t^{\mathrm{IMU}}\) | \(\mathbb R^3\) | Encoder input |

These are the **core evidence** in the minimum viable thesis. If your simulator produces ideal IMU readings, you may want to optionally add realistic noise/bias to force the encoder to learn robust representations.

### 1.3 Control Inputs

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Motor commands / RPM | \(\omega_1, \omega_2, \omega_3, \omega_4\) | \(\mathbb R^4\) | Encoder input and physics model input |
| Or thrust + torque | \(T, \boldsymbol\tau\) | \(\mathbb R^4\) | Equivalent alternative |

The choice of representation should match what \(f_{\mathrm{phys}}(\cdot)\) expects. If \(f_{\mathrm{phys}}\) uses motor RPMs to compute thrust, you need RPMs; if it uses thrust/torque directly, you can log those.

### 1.4 Wind and Disturbance Data (for OOD evaluation and oracle only)

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Wind velocity field | \(\mathbf w_t\) | \(\mathbb R^3\) | OOD taxonomy, oracle baseline |
| Disturbance force | \(\mathbf f_{\mathrm{ext},t}\) | \(\mathbb R^3\) | Oracle baseline, diagnostic \(E_d\) |
| Disturbance torque | \(\boldsymbol\tau_{\mathrm{ext},t}\) | \(\mathbb R^3\) | Oracle baseline |
| Turbulence parameters | \(\sigma_u, \sigma_v, \sigma_w, L_u, L_v, L_w\) | scalars | Reproducibility, OOD definition |

**Important**: The learned residual \(d_\theta\) is **not** trained against \(\mathbf w_t\). Wind ground truth is used only for (a) the oracle upper bound, (b) the diagnostic \(E_d\), and (c) defining the OOD taxonomy. Do **not** add a loss that forces \(d_\theta \approx \mathbf w\).

### 1.5 Visual Observations (extension only)

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| RGB image | \(I_t\) | \(H \times W \times 3\) | Visual encoder \(E_V\) (extension) |
| Depth image | \(D_t\) | \(H \times W\) | Optional auxiliary |

Only needed if you pursue the multimodal extension (M4). The minimum viable thesis does not require vision.

---

## 2. Data Organization

### 2.1 Trajectory Format

Data should be stored as **trajectories**, not independent tuples:

\[
\tau_i = \big\{ (o_t,\ s_t,\ u_t,\ \mathbf w_t,\ \mathbf f_{\mathrm{ext},t}) \big\}_{t=0}^{T_i}
\]

where \(T_i\) is the trajectory length. Multi-step losses require contiguous windows of length \(T_{\mathrm{burn}} + H\), so per-timestep shuffling is not acceptable.

### 2.2 Suggested Dataset Structure

```
dataset/
├── train/
│   ├── trajectory_0000.npz   # t, p, v, q, ω, a_imu, ω_imu, u, w, f_ext
│   ├── trajectory_0001.npz
│   └── ...
├── val/                       # same format, drawn from training wind distribution
│   └── ...
├── test/
│    ├── magnitude_extrap/      # OOD regime 1
│    ├── directional_extrap/    # OOD regime 2
│    ├── temporal_extrap/       # OOD regime 3
│    ├── gust_extrap/           # OOD regime 4
│    └── physics_mismatch/      # OOD regime 5
├── metadata/
└── manifest.json
```

Each `.npz` stores arrays of shape `(T, dim)`.

### 2.3 Sequence Lengths

- **Burn-in window** \(T_{\mathrm{burn}}\): typically 5–10 steps to initialize \(h\).
- **Prediction horizon** \(H\): 5–20 steps for training; longer (up to 100) for evaluation only.

The trajectory length \(T_i\) should be at least \(T_{\mathrm{burn}} + H_{\max}\) plus a margin.

---

## 3. Data Volume

For an M.Tech thesis, a realistic starting point:

| Wind regime | Trajectories | Steps per trajectory | Approx. total steps |
|---|---|---|---|
| Training (moderate wind) | 200–500 | 1000–2000 | \(2\times10^5\) to \(10^6\) |
| Validation | 50–100 | 1000–2000 | \(5\times10^4\) to \(2\times10^5\) |
| OOD test (each of 5 regimes) | 50–100 | 1000–2000 | \(2.5\times10^5\) to \(10^6\) |

For the **sample-efficiency study** (§16.1 in your proposal), you also need the same training pool at \(10\%, 25\%, 50\%, 100\%\) — which is just a subset of the same data, so no extra generation is needed.

If vision is included, storage grows dramatically (roughly 1–10 MB per RGB frame at \(256\times256\)), so consider storing images only for a subset of trajectories in the extension phase.

---

## 4. Data Diversity Requirements

To support the OOD taxonomy, the training data must vary systematically:

| Dimension | Training range | Held out for OOD |
|---|---|---|
| Mean wind magnitude \(|\mathbf w|\) | \(0\)–\(4\) m/s | \(>4\) m/s |
| Wind direction | All horizontal, limited vertical | Excluded directions |
| Turbulence intensity \(\sigma\) | Low–moderate | High |
| Gust structure | Smooth | Sudden transients |
| Physics parameters \(m, J, C_D\) | Nominal | Perturbed (10–20%) |

The simulator must expose these as controllable parameters so that trajectories can be generated under each regime. This is why Isaac Sim / Isaac Lab is preferred over real flight data — the OOD conditions are not just rare, they are *counterfactually controllable*.

---

## 5. What Each Training Component Needs

| Component | Data required |
|---|---|
| Encoder \(E_P\) | \(a_t^{\mathrm{IMU}}, \omega_t^{\mathrm{IMU}}, u_t\) |
| Encoder \(E_V\) (extension) | \(I_t\) |
| Decoder \(D_s\) | Ground-truth \(s_t = [\mathbf p, \mathbf v, \mathbf q, \boldsymbol\omega]\) |
| Physics \(f_{\mathrm{phys}}\) | \(s_t, u_t\) (no training; parameterized analytically) |
| Residual \(d_\theta\) | \(h_t, u_t\) (learned implicitly via state loss) |
| Latent transition \(\latent\) | \((h_t, u_t, \tilde s_{t+1})\) sequence |

The residual and the latent transition are **not trained with direct supervision** — they are trained through the state prediction and multi-step losses, which require ground-truth \(s_{t+k}\).

---

## 6. Data the Physics Model Validation Requires

For §11 (Physics Model Validation), you need three separate datasets:

- **Case A (no wind)**: same trajectories with \(\mathbf w = 0\).
- **Case B (known wind)**: same trajectories with controlled wind and logged \(\mathbf w_t, \mathbf f_{\mathrm{ext},t}\).
- **Case C (perturbed parameters)**: the simulator's physics parameters are changed (e.g., mass \(m_{\mathrm{sim}} = 1.1\,m_{\mathrm{nominal}}\)), and the nominal \(f_{\mathrm{phys}}\) is evaluated against these.

Each case is generated by a distinct simulator configuration and does not require the neural components to be trained.

---

## 7. Practical Recommendations

1. **Log everything at the highest rate the model uses.** Even if you subsample later, having IMU at 200 Hz and state at 200 Hz simplifies alignment.
2. **Store time explicitly.** A `t` array avoids alignment bugs when some channels are logged at different rates.
3. **Use `.npz` or HDF5, not CSV.** Trajectories of \(10^6\) steps with 20+ fields become unmanageable as text.
4. **Separate the OOD generation from training generation.** Generate OOD data with a distinct seed and label it explicitly.
5. **Record simulator configuration metadata.** Each trajectory's file should carry the wind parameters, physics parameters, and initial conditions used to generate it. This is essential for reproducibility and for defining the OOD taxonomy precisely.
6. **Do not include ground-truth wind as a model input.** Wind is only for evaluation and oracle. Including it would leak the disturbance and invalidate the OOD claim.

---

## 8. Summary

To train the world model, you need, at minimum:

\[
\boxed{
\text{IMU} + \text{control} + \text{ground-truth state} + \text{trajectory continuity}
}
\]

and for evaluation:

\[
\boxed{
\text{wind ground truth} + \text{disturbance force} + \text{OOD metadata}
}
\]

Everything else — vision, uncertainty, MPC — is an extension. The minimum viable thesis requires only proprioception, control, ground-truth state, and controlled wind conditions across a well-defined training/OOD split.

1. Add an identifiability-driven experiment design module
Randomizing mass, inertia and drag is not sufficient. Some parameter combinations can produce very similar trajectories.

For example, increasing mass while increasing thrust can preserve approximately the same acceleration.

You should eventually have an experiment-design component that evaluates whether the generated experiments sufficiently distinguish physical parameters.

Consider:
- Fisher Information Matrix (FIM).
- Sensitivity analysis.
- Parameter correlation.
- Condition number of the information matrix.
- Persistent excitation.
This is particularly important for your open-loop experiments.
2. Distinguish three types of data
A. Closed-loop trajectory data
Control and dynamics learning
- Controller tracks a reference trajectory.
- Realistic flight behavior.
- Contains feedback-dependent control inputs.

B. Open-loop excitation data
System identification
- Predefined motor inputs.
- No feedback correction.
- Captures natural system response.
- Useful for identifying dynamics.

C. Counterfactual paired data
Causal comparison
- Same initial conditions.
- Same control sequence.
- Different physical or environmental parameters.
- Isolates the effects of physical changes.

These three datasets serve complementary purposes. I would maintain their identities in the metadata rather than mixing them without distinction.
3. Introduce a model applicability registry
For each aerodynamic model, maintain a record of its validity.

For example:
Model	Hover	Translation	High angular velocity
Lumped quadratic drag	Applicable	Applicable with limitations	Approximation
Component-level drag	Applicable	Applicable	Depends on local velocity
Rotational damping	Applicable	Applicable	Coefficient-dependent
Rotor ground effect	Near ground only	Requires care	Requires validation

The actual applicability must be established from the equations and assumptions of each implementation, not assigned automatically.

This will prevent physically inappropriate force models from being enabled in certain experiments.
4. Separate simulation truth from observations
This is essential for world-model training.

I recommend maintaining three distinct data categories:
- ground_truth: Actual simulator states and physical quantities.
- observations: IMU, camera and other simulated sensor measurements.
- actions: Motor commands or actuator inputs.
This separation will allow you to experiment with different world-model architectures without changing the underlying dataset.