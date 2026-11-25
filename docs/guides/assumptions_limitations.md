# Scientific assumptions and limitations

Updated each run (Run 4). Each entry gives what is assumed, why, and where it breaks.

## Simulator and vehicle

| Assumption | Basis / evidence | Limitation |
|---|---|---|
| PhysX rigid-body articulation at dt = 5 ms (TGS, 4/0 iterations) | OmniDrones defaults; analytic checks pass to < 1 % (force_accounting.md) | Solver-iteration sensitivity not studied |
| Rotor thrust T = k_F Ω², first-order motor lag (τ_m ≈ 8.9 ms) | OmniDrones `RotorGroup`; step response matches to 6e-8 | No motor saturation dynamics beyond the [0, 1] clamp; no battery sag |
| Rotors do not spin in the articulation (joint velocity 0) | Measured | No rotor gyroscopic effects; a small (~0.3 %) articulation coupling remains, absorbed by the residual (Q4) |
| Effective inertia (rotor links as axisymmetric equivalents, no spin inertia about z) | Measured roll/pitch/yaw within 0.4 % | — |
| External forces act at the base-link origin (0.5 mm below the CoM) | Injector design | Tiny parasitic torque (physical, recorded) |

## Aerodynamics (docs/models/aerodynamics.md)

| Assumption | Limitation |
|---|---|
| Linear rotor drag, ∝ Ω, in-plane only | Validated in the literature at low advance ratio. **At 7–8 m/s wind, μ ≈ 0.16–0.18 exceeded the old 0.15 envelope; widened to μ ≤ 0.20 (Run 5), reduced fidelity in 0.15–0.20.** |
| Quadratic body drag with mesh projected areas and per-axis C_D (1.1, 1.1, 1.3), constant in Re and incidence | No lift/side forces; C_D not measured for this airframe |
| No wind-on-thrust (translational lift, axial inflow) | Thrust is wind-independent at fixed rotor speed |
| Coefficients from a different vehicle (Faessler et al. 2018) | Absolute drag magnitude uncertain by roughly ±50 %; relative structure is sound |

## Wind

| Assumption | Limitation |
|---|---|
| Spatially uniform wind except an optional linear gradient | No turbulence spatial correlation across the airframe |
| Dryden MIL-F-8785C, low-altitude parameters clamped at 10 ft | Extrapolated to 1–3 m altitude |
| Frozen-turbulence speed = mean horizontal wind (≥ 1 m/s), fixed per episode | Ignores the vehicle's own airspeed. Chosen so realisations are trajectory-independent (counterfactual pairing). |
| ID/OOD defined by realised wind velocity statistics (D1, D2) | Disturbance *force* also depends on vehicle velocity and orientation |

## Sensors

| Assumption | Limitation |
|---|---|
| Isaac `IMUSensor` at the base-link origin, clean | Noise and bias applied at load time (D7); no vibration, no sensor dynamics |

## Data and control

| Assumption | Limitation |
|---|---|
| Model step 20 ms with actions held for 4 physics steps (D4) | Closed-loop control at 50 Hz (validated in pilot runs: no saturation, tracking error ≤ 0.4 m at ID wind) |
| Controllers built from the realised vehicle parameters (Q2) | Real controllers would carry model mismatch; the "nominal" policy exists for ablations |
| Excitation amplitudes chosen for feasibility | Identifiability adequacy assessed in Run 5 (FIM, decorrelation) |

## World-model design decisions (latent models B2/B3/B4)

* **Observation vs imagination.** `h` is driven by the real IMU through `E` and `F`. In imagination the physical state `s̃` is carried
  forward by physics (RK4); the decoder is never in the loop. `s_0` is the simulator state (`initial_state="given"`); `"decoder"` is an ablation.
  There is no state teacher forcing: context window (10 steps) then a free rollout, with the recorded future actions as input.
  Curriculum H = 1 → 5 → 10 → 20 → 50 (`TrainConfig.horizon_stages`).
* **φ (synthetic IMU).** φ has no learned parameters: accelerometer = R(q̂)ᵀ(v̇ − g), gyro = ω̂_B, passed through the same `E` and `F`
  as real readings. B3: v̇ from its own ṡ = f_phys + d_θ, evaluated at (s̃_{t+1}, u_{t+1}, h_t). B2 has no ṡ and uses Δv/Δt of its prediction
  (documented asymmetry). Assumes the stored IMU is instantaneous (`obs_mode="instant"`). White noise of the same preset as the real IMU is added to φ;
  the per-window bias is **not** modelled in φ (simplification). The `h` after the last rollout step is not computed (it is never used).
* **IMU noise.** A clean accelerometer plus known `u` reveals the aero force almost directly (specific force = thrust + aero), so headline
  results use the noisy preset (`nominal`, a proposed value, not a measurement). Clean runs are sanity/upper-bound only.
* **Encoder / loss / windows.** `E` is IMU-only (`enc_uses_u` ablation); λ_dec default 1.0; λ_roll divided by Σγ; baseline loss weights floored at a physical resolution (proposed 1 cm, 1 cm/s, 0.5°, 0.01 rad/s); training windows per curriculum stage; one fixed validation set (H = 50). Details in `.idea/specs/model_specification.md` §11.
* **Metrics.** RMSE primary, mean Euclidean error secondary; older run records used the mean error under `pos_m`.
* **Residual.** Default `residual_channels="acc_only"` (v̇, ω̇; ṗ and attitude-rate corrections zeroed); `"all"` is the ablation. The v̇ residual is
  added to the **world-frame** acceleration in `QuadrotorPhysics.derivative`. An APHYNITY-style penalty λ_resid·‖d_θ‖² (default 1e-3, untuned) is
  in the loss; set `lambda_resid=0` for the ablation. Smoothness regularisation is not implemented.
* **Privileged information.** Wind and disturbance forces are never model inputs or training targets (thesis statement). They are used only by the
  post-hoc probe (`evaluation/probe.py`, frozen ridge from `h` to the recorded external force, compared with a state+u probe) and for B6 later.
* **Disturbance targets.** The recorded wrench is the **total injected aero wrench**, not wind only (drag acts with zero wind). The probe uses it and a reconstructed wind-attributable part (`data/disturbance.py`, median relative error of the total-force reconstruction 0.16 % on the pilot). B6 uses the total wrench. B5 has two variants (a: learned vector field, b: exact kinematics + learned v̇, ω̇) and B6 four (a/b × current/future); which are reported is an open decision.
* **Not implemented yet:** re-anchored evaluation, stochastic head, `h`-bottleneck. Mode A (closed-loop) recorded `u` leaks wind information
  into imagination; Mode B (fixed-action replay) pairs are the clean evaluation. State this wherever results are reported.

## Parallel generation

* One generator process is **CPU-bound**: measured on an idle GPU, ~3 % GPU utilisation (744 MiB), ~4.5 busy cores on average (including start-up), ~55 physics steps/s (2784 steps in 50.9 s; 61 steps/s in the Run 4 report). The Run 4 report already showed the CPU pipeline (81 steps/s) faster than GPU dynamics with readback (72). Scale by running **several worker processes**, also several per GPU; the limit is free CPU cores and RAM, not the number of GPUs.
* Workers split the manifest by **group** (a counterfactual group never splits: Mode B siblings read their baseline's commands from its shard) with a deterministic longest-processing-time balance on planned steps. Each worker writes its own `shard_wII_NNN.h5` and `index.workerII.json`; `generate merge` scans all shards, validates the **whole** dataset and writes `index.json`. A worker never opens another worker's shard.
* The GPU is chosen with `CUDA_VISIBLE_DEVICES` (config and config hash untouched). Isaac may print renderer allocation errors for a non-zero GPU; nothing is rendered.
* **Reproducibility of the real simulator (measured and fixed, 2026-11-24).** Every array of an episode except the IMU accelerometer is **bitwise reproducible**: across processes, after `kill -9` + `--resume`, across GPUs (GPU 1 vs 2), and regardless of which episodes ran before it in the same process (positions, velocities, rotor commands, wind, forces, controller inputs and outputs: 696 control calls and 2784 physics steps compared). Two causes of order dependence were found and fixed: (1) `realised()` computed the composite inertia and centre of mass from float32 world link poses read **before** the episode reset, so rounding depended on where the previous episode ended (≈ 1e-9, one ulp); the controller is built from them, its mixer differed by an ulp, and one rotor command differed by one ulp at one control step, growing to ≈ 6e-6 m: fixed by putting the articulation at a canonical pose before the parameters are read (`SimulationInterface.reset_to_canonical_pose`); (2) the first IMU sample carried solver state from the previous episode (4.6e-3 m/s² after the settle step): row 0 is now the analytic rest value R(q₀)ᵀ(0,0,g) and row 1 is derived from the episode's own ground truth. **Remaining:** the Isaac IMU accelerometer differs between regenerations of the same episode by ≈ 2e-5 m/s² typically (≤ 2e-3), in almost every row, although the physics is identical: the sensor's own float32 finite-difference output; accepted (3 orders of magnitude below the smallest noise preset). So whole-episode checksums are not reproducible across regenerations, and `generate regenerate` differences in `observations/imu_accel_B` of this size are expected. My first explanation (a cross-GPU effect) was wrong (confounded with episode order).

## Evaluation analyses

* **No-wind sample is small:** the pilot has few no-wind episodes (2 in `val_id` + `test_ood_magnitude`), so wind vs no-wind comparisons have wide intervals; the Run 6 dataset must contain enough.
* **Mode B has no test episodes** in the pilot (only `train` and `val_id`). The shuffled-u test has low power (B1 jump ≈ 1.00): closed-loop siblings' commands barely differ.
* **B6 and the "no-wind gap":** B6 uses the total recorded wrench, so even without wind B6 removes the vehicle-motion drag; the B3–B6 gap without wind is small but not a pure wind-information measure.
* **Re-observation** feeds the real IMU into the latent and optionally the true state; it is an evaluation analysis (receding-horizon use), never used in training.

## Data pipeline checks added

* **Per-episode checksum** (`sha256` over path, dtype, shape and bytes of all arrays) is written with every new episode and verified on resume (a corrupted or unreadable episode is deleted and regenerated), on merge and on `generate validate` (a mismatch is a hard finding). Pilot episodes written before this have none (`generate checksums --dataset D` adds them; it only adds an attribute).
* **Loader benchmark** (`python -m physiworld_uav.data.loader_benchmark`): on the pilot `train` role (63 episodes, 24,718 windows at H = 20, batch 256): episodes load at 57 episodes/s (66 MB/s of file); random-window batches 13,378 windows/s with no worker, 36,585 with 8; the trainer consumed about 878 windows/s (B3 smoke run), so the loader has ≥ 15× headroom and is not the bottleneck: the model step is.
* **Outlier episodes** (`outliers.episodes` finding and `outliers.png`): per-episode p99 of speed, tilt, ‖ω‖, IMU |f| and rotor max, with episodes more than 5 robust deviations from the median listed (info level).
* **Open-loop action–wind correlation** (0.39 over 24 episodes, limit 0.30): the finding now also reports a permutation p-value (wind shuffled between episodes). Pilot: chance gives a 95th percentile of 0.42 and the observed value has p = 0.085 (1000 permutations, independent of episode order; earlier estimates of 0.125–0.127 used 200 permutations and depended on the episode order), so it is **borderline (not significant at 5 %)** with this few episodes; all 6 open-loop groups with siblings share identical `u`, as designed. The fixed 0.30 limit ignores the sample size; changing the rule is left to the user.
* **YAML configuration** for training (`config/training/*.yaml`), model choice (`config/model/*.yaml`) and evaluation (`config/evaluation/default.yaml`); unknown keys are an error and a test keeps the YAML defaults identical to the dataclass defaults.
* **Model sizes are configuration, not code** (`config/training/default.yaml`, block "model sizes"; per-model overrides in `config/model/*.yaml`, e.g. `b3_wide.yaml`): `latent`, `embed`, `hidden`, `enc_hidden`, `dec_hidden` (layer lists), `trans_hidden`, `resid_hidden`, `b4_hidden`, `b5_hidden` (widths; 0 = match B3's parameter count), `trans_depth`, `resid_depth`, `b4_depth`, `field_depth` (hidden layers), `activation`. Not configurable because the data interface fixes them: IMU input 6, action 4, state features 10, state 13, residual/field output 12, and the single GRU layer. The three architectural-bias options of the spec (§3.4) are flags too: `resid_uses_u`, `lambda_zero` (training key), `shared_trunk`. A test asserts that every `ModelConfig` field has a same-named, same-default `TrainConfig` key and that the default parameter counts are unchanged (B3 227,033, B4 226,392, B4 (B3 width) 72,716, B6b 228,569). Checkpoints from before this change load (missing keys take the defaults).

## IMU accelerometer

* **Pilot v0.5.0 (raw sensor):** the stored `observations/imu_accel_B` is **not** the specific force at the centre of mass. It equals the specific force minus ω_B × v_B (R² 0.83 over 280,152 samples; x axis coefficient −0.98, R² 0.98); adding ω_B × v_B back reduces the Δv/dt-vs-IMU residual p99 from 1.81 to 0.057 m/s² (median 0.008 → 0.001). A sensor lever arm was tested and **refuted** (composite CoM offset 0.56 mm; fitted lever arm R² 0.007). The likely cause is that Isaac's IMU differences the velocity expressed in the body frame (transport term); the code path was not inspected.
* **Fix at generation (decided 2026-11-24):** `physics.imu_transport_correction: true` (default in `config/physics/isaac_200hz.yaml`) makes the runner add ω_B × v_B (ground truth of the same physics step, `core/state_extractor.imu_transport_correction_torch`) to every valid IMU frame, so the logged accelerometer is the specific force at the base link, as a real accelerometer measures. The flag is stored in each manifest row (`sim.imu_transport_correction`) and in the episode statistics (`imu_transport_corrected`); the pilot/smoke dataset versions were bumped to 0.6.0 / 0.4.0 so v0.5.0 is never overwritten. The raw reading is `corrected − ω_B × v_B` (recomputable from the stored ground truth), so nothing is lost.
* **Verified on the real simulator** (6 worst pilot episodes regenerated with the fix): Δv/dt-vs-IMU residual p99 5.04 → 0.173 m/s², median 0.024 → 0.003; per episode 4.46 → 0.146, 5.31 → 0.277, 4.00 → 0.136. The remaining large values are ground-impact impulses (an episode ending by ground contact: IMU and finite difference disagree on a contact impulse within a step); those episodes are excluded from model data.
* **Consequence for the models:** the synthetic IMU φ (specific force) now matches the stored IMU. Datasets with the raw sensor (v0.5.0) are flagged: training and evaluation print a warning when a φ-based model meets a dataset whose episodes lack the correction flag (`imu_transport_status`).
* **Pilot v0.6.0 (2026-11-24):** regenerated with the same plan as v0.5.0 (147 / 147 episodes, validator ok, 141 model-ready + 6 ground contact, identical statuses, step counts and wind). IMU-vs-truth residual over all contact-free episodes: p50 0.0083 → 0.0012, p99 1.645 → 0.051 m/s²; episodes with p99 > 1 m/s²: 47 / 141 → 0 / 141. Trajectories differ from v0.5.0 by at most 9e-5 m (median 2.6e-6); physics is bitwise identical in 61 / 147 episodes (v0.5.0 carried the order-dependent one-ulp effects). The `physical.acc_imu` limit is p99 ≤ 0.25 m/s² for datasets with the correction (1.0 for v0.5.0). Quality report: 0 fail, 3 warn (open-loop correlation, domain-factor correlation, groups per role).
* **Wind plan for Run 6 (decided 2026-11-25):** in-distribution wind `id_wide` (episode mean speed 0–5 m/s, realised peak ≤ 6 m/s) and OOD magnitude `ood_magnitude_wide` (mean 6.2–8.5 m/s, in (6, 9]). The mean speed is **stratified**, not i.i.d. uniform: within each split role the groups are spread over equal-width speed bands (`speed_strata`: 5 ID bands of 1 m/s, 3 OOD bands), and the ID bands are weighted towards high speed (`speed_strata_weights: [0.5, 1, 2, 3, 3]`, no empty band when a part has ≥ 5 groups per role). Reason: the counterfactual siblings (wind × 0 and × 0.5) pile up at low speed, so equal group counts still gave 41 % of the training episodes below 1 m/s and a training maximum of 3.7 m/s while the OOD magnitude episodes were at 4.3–8.2 m/s (`docs/results/sweep_fix_v1.md` §6). Plan preview (`config/experiment/pilot_wide_wind.yaml`, pilot-sized): training episodes per 1 m/s band 19 / 9 / 14 / 10 / 14 (before: 27 / 13 / 11 / 8 / 7), base groups per band 2 / 4 / 8 / 12 / 12. The old regimes (`id`, `ood_magnitude`) are unchanged and the pilots v0.5.0 and v0.6.0 keep them. Open: the OOD upper bound (the earlier envelope check of Option C on the simulator found 24 / 26 OOD episodes inside the force-model envelope, so about 1 in 6 episodes above 7.5 m/s would be excluded from model-ready data; `docs/checklist.md`), and the OOD direction / temporal / gust regimes were not widened.
