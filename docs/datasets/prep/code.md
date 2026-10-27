# Complete Configuration File

Below is a complete Hydra config that captures every detail from the six-mode recipe. Save it as `config/dataset_generation.yaml`. The generation script reads it and dispatches to controller and wind model libraries.

```yaml
# ================================================================
# config/dataset_generation.yaml
# World-model dataset generation — 6 flight modes × multiple winds
# ================================================================
#
# Usage examples:
#
#   # Training set (default)
#   python generate_dataset.py
#
#   # Validation set (different seed, same distribution)
#   python generate_dataset.py experiment.seed=99 \
#       experiment.output_dir=/datasets/world_model_v1_val
#
#   # OOD magnitude extrapolation (2× wind)
#   python generate_dataset.py ood.wind_scale=2.0 \
#       experiment.output_dir=/datasets/ood_magnitude
#
#   # OOD directional extrapolation (vertical only)
#   python generate_dataset.py ood.wind_directions_override='[[0,0,1],[0,0,-1]]' \
#       experiment.output_dir=/datasets/ood_directional
#
#   # OOD temporal extrapolation (2× faster sinusoidal)
#   python generate_dataset.py ood.temporal_speed_scale=2.0 \
#       experiment.output_dir=/datasets/ood_temporal
#
#   # OOD physics mismatch (mass +15%, inertia +10%)
#   python generate_dataset.py ood.mass_scale=1.15 ood.inertia_scale=1.10 \
#       experiment.output_dir=/datasets/ood_physics
#
# ================================================================

defaults:
  - _self_

# ─── Experiment metadata ────────────────────────────────────────
experiment:
  name: "world_model_dataset_v1"
  seed: 42
  output_dir: "/datasets/world_model_v1"
  save_metadata: true       # write .json alongside each .npz
  verbose: true             # print per-second status lines

# ─── Simulation (matches working manual_flight config) ──────────
sim:
  dt: 0.01
  substeps: 1
  gravity: [0, 0, -9.81]
  replicate_physics: false
  use_flatcache: true
  use_gpu_pipeline: false
  device: "cpu"
  solver_type: 1
  use_gpu: false
  bounce_threshold_velocity: 0.2
  friction_offset_threshold: 0.04
  friction_correlation_distance: 0.025
  enable_stabilization: true
  gpu_max_rigid_contact_count: 524288
  gpu_max_rigid_patch_count: 163840
  gpu_found_lost_pairs_capacity: 4194304
  gpu_found_lost_aggregate_pairs_capacity: 33554432
  gpu_total_aggregate_pairs_capacity: 4194304
  gpu_max_soft_body_contacts: 1048576
  gpu_max_particle_contacts: 1048576
  gpu_heap_capacity: 67108864
  gpu_temp_buffer_capacity: 16777216
  gpu_max_num_partitions: 8

# ─── Drone model ────────────────────────────────────────────────
drone_model:
  name: "Hummingbird"
  controller: "RateController"

# ─── Camera (off during dataset generation for speed) ──────────
camera:
  enabled: false
  resolution: [640, 480]
  position: [0.2, 0.0, 0.1]

# ─── High-level controller gains ────────────────────────────────
# All modes (hover, line, circle, figure_eight, helix, waypoint_random,
# attitude_*, yaw_*) are position/attitude trackers built on top of the
# low-level RateController. These gains govern the outer loop.
rate_controller_gains:
  kp_pos_xy: 1.5              # [rad/s per m]  P on horizontal position error
  kd_pos_xy: 0.8              # [rad/s per m/s] D on horizontal velocity
  kp_pos_z:  3.0              # [m/s² per m]   P on altitude error
  kd_pos_z:  2.0              # [m/s² per m/s] D on vertical velocity
  kp_yaw:    1.0              # [rad/s per rad] P on yaw error
  max_tilt_rad: 0.5           # ~29° — safety clamp
  max_rate_radps: 2.0         # safety clamp on body rates

# ─── OOD modifiers (1.0 = training distribution) ────────────────
# Override these via Hydra CLI to generate OOD test sets.
ood:
  wind_scale: 1.0                     # multiplies all wind magnitudes
  wind_directions_override: null      # null → use per-segment defaults
                                      # e.g. [[0,0,1]] for vertical-only
  mass_scale: 1.0                     # physics OOD
  inertia_scale: 1.0
  drag_scale: 1.0
  temporal_speed_scale: 1.0           # >1 = faster sinusoidal = temporal OOD

# ─── Dataset composition ────────────────────────────────────────
dataset:
  total_trajectories: 100
  train_val_split: 0.8                # first 80% train, last 20% val
  action_seed_offset: 100000          # RNG offset for action sampling

  # ── Counterfactual pairs ─────────────────────────────────────
  # For each scripted (non-random) trajectory in the listed modes:
  #   1. Run with wind = multiplier[0] × F_w, record all actions
  #   2. Replay the SAME actions with wind = multiplier[i] × F_w
  # Both runs share a pair_id; each stores its own (s, u, F_w) triple.
  # This breaks the action-wind correlation and defeats Problem 2.
  counterfactual:
    enabled: true
    modes: [translation, acceleration, yaw, attitude, combined]
    wind_multipliers: [1.0, 2.0]      # variant 0 = nominal, variant 1 = 2× wind
    replay_mode: "replay_recorded_actions"  # bypass controller, apply recorded u_t

  # ── Mode fractions (must sum to 1.0) ─────────────────────────
  modes:

    # ==========================================================
    # MODE 1: HOVER (10%)
    # Weakest supervision — included only so hover predictions
    # don't diverge under OOD. Not used for OOD validation.
    # ==========================================================
    hover:
      fraction: 0.10
      definition:
        duration_s: 15.0
        altitudes: [0.5, 1.0, 1.5, 2.5]    # one sampled uniformly per trajectory
        segments:
          - t_range: [0.0, 5.0]
            controller: {type: hover}
            wind: {type: none}
          - t_range: [5.0, 10.0]
            controller: {type: hover}
            wind:
              type: constant
              magnitude_N: 2.0
              direction: [1, 0, 0]
          - t_range: [10.0, 12.0]
            controller: {type: hover}
            wind:
              type: step
              step_time_s: 10.0
              from_magnitude_N: 2.0
              to_magnitude_N: 4.0
              direction: [1, 0, 0]
          - t_range: [12.0, 15.0]
            controller: {type: hover}
            wind:
              type: constant
              magnitude_N: 4.0
              direction: [1, 0, 0]
        counterfactual: false

    # ==========================================================
    # MODE 2: TRANSLATION (27.5%)  — the workhorse
    # Residual grows with (v − v_wind)² → clean identifiability.
    # ==========================================================
    translation:
      fraction: 0.275
      definition:
        duration_s: 100.0
        altitudes: [1.0, 1.5, 2.0]
        altitude_ramp: [1.0, 2.0]          # ramp over the trajectory
        segments:
          # (a) Straight-line +X with crosswind
          - t_range: [0.0, 10.0]
            controller:
              type: line
              velocity_mps: [1.0, 0.0, 0.0]
            wind: {type: constant, magnitude_N: 2.0, direction: [0, 1, 0]}

          # (b) Straight-line +Y with crosswind
          - t_range: [10.0, 20.0]
            controller:
              type: line
              velocity_mps: [0.0, 1.0, 0.0]
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # (c) Circle (r = 2 m), yaw tracks tangent
          - t_range: [20.0, 30.0]
            controller:
              type: circle
              radius_m: 2.0
              omega_radps: 0.3
              yaw_follows_tangent: true
              center_xy: [0.0, 0.0]
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # (d) Figure-eight with wind step
          - t_range: [30.0, 40.0]
            controller:
              type: figure_eight
              amplitude_m: [2.0, 1.0]
              frequency_hz: 0.1
              yaw_follows_tangent: false
            wind:
              type: step
              step_time_s: 35.0
              from_magnitude_N: 2.0
              to_magnitude_N: 6.0
              direction: [1, 0, 0]

          # (e) Random actions, constant wind — defeats action-wind shortcut
          - t_range: [40.0, 70.0]
            controller:
              type: random_action
              thrust_range_N: [4.0, 9.0]
              rate_range_radps: [-0.6, 0.6]
              resample_interval_s: 0.3
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # (f) Random actions with 10s wind steps
          - t_range: [70.0, 100.0]
            controller:
              type: random_action
              thrust_range_N: [4.0, 9.0]
              rate_range_radps: [-0.6, 0.6]
              resample_interval_s: 0.3
            wind:
              type: multi_step
              direction: [1, 0, 0]
              steps:
                - {t_s: 70.0, magnitude_N: 0.0}
                - {t_s: 80.0, magnitude_N: 2.0}
                - {t_s: 90.0, magnitude_N: 4.0}
                - {t_s: 100.0, magnitude_N: 6.0}
        counterfactual: true

    # ==========================================================
    # MODE 3: ACCELERATION / DECELERATION (17.5%)
    # Residual = F_w/m directly — cleanest physical signal.
    # ==========================================================
    acceleration:
      fraction: 0.175
      definition:
        duration_s: 100.0
        altitudes: [1.5, 2.0, 3.0]         # headroom for ramps
        segments:
          # z-thrust ramps, no wind
          - t_range: [0.0, 15.0]
            controller:
              type: thrust_ramp
              axis: z
              thrust_range_frac: [0.6, 1.4]   # of mg
              ramp_duration_s: 3.0
              hold_duration_s: 2.0
            wind: {type: none}

          # z-thrust ramps, constant 4N wind
          - t_range: [15.0, 30.0]
            controller:
              type: thrust_ramp
              axis: z
              thrust_range_frac: [0.6, 1.4]
              ramp_duration_s: 3.0
              hold_duration_s: 2.0
            wind: {type: constant, magnitude_N: 4.0, direction: [1, 0, 0]}

          # z-thrust ramp with wind step mid-ramp
          - t_range: [30.0, 45.0]
            controller:
              type: thrust_ramp
              axis: z
              thrust_range_frac: [0.6, 1.4]
              ramp_duration_s: 3.0
              hold_duration_s: 2.0
            wind:
              type: step
              step_time_s: 37.0
              from_magnitude_N: 0.0
              to_magnitude_N: 4.0
              direction: [1, 0, 0]

          # x-acceleration ramps, no wind
          - t_range: [45.0, 60.0]
            controller:
              type: thrust_ramp
              axis: x
              thrust_range_frac: [0.6, 1.4]
              ramp_duration_s: 3.0
              hold_duration_s: 2.0
            wind: {type: none}

          # x-acceleration ramps, headwind
          - t_range: [60.0, 75.0]
            controller:
              type: thrust_ramp
              axis: x
              thrust_range_frac: [0.6, 1.4]
              ramp_duration_s: 3.0
              hold_duration_s: 2.0
            wind: {type: constant, magnitude_N: 4.0, direction: [-1, 0, 0]}

          # Combined ramp + roll (coupled dynamics)
          - t_range: [75.0, 90.0]
            controller:
              type: combined_ramp_roll
              thrust_range_frac: [0.6, 1.4]
              roll_rate_radps: 0.3
              ramp_duration_s: 3.0
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # Thrust sinusoid + wind step
          - t_range: [90.0, 100.0]
            controller:
              type: thrust_sinusoid
              amplitude_frac: 0.30            # of mg
              frequency_hz: 0.2
            wind:
              type: step
              step_time_s: 95.0
              from_magnitude_N: 2.0
              to_magnitude_N: 6.0
              direction: [1, 0, 0]
        counterfactual: true

    # ==========================================================
    # MODE 4: YAW MOTION (12.5%)
    # Forces the model to learn body-frame Rᵀ F_w transformation.
    # ==========================================================
    yaw:
      fraction: 0.125
      definition:
        duration_s: 100.0
        altitudes: [1.0, 1.5, 2.0]
        segments:
          - t_range: [0.0, 10.0]
            controller: {type: hover}
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          - t_range: [10.0, 20.0]
            controller:
              type: yaw_sine
              amplitude_radps: 0.5
              frequency_hz: 0.2
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          - t_range: [20.0, 30.0]
            controller:
              type: yaw_step
              target_rad: 1.5708             # π/2
              hold_s: 5.0
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          - t_range: [30.0, 40.0]
            controller:
              type: yaw_step
              target_rad: 3.1416             # π
              hold_s: 5.0
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # Same actions, wind rotated 90° → forces Rᵀ learning
          - t_range: [40.0, 50.0]
            controller:
              type: yaw_sine
              amplitude_radps: 0.5
              frequency_hz: 0.2
            wind: {type: constant, magnitude_N: 2.0, direction: [0, 1, 0]}

          - t_range: [50.0, 60.0]
            controller:
              type: yaw_step
              target_rad: 1.5708
              hold_s: 5.0
            wind: {type: constant, magnitude_N: 2.0, direction: [0, 1, 0]}

          # Circle with yaw tracking tangent
          - t_range: [60.0, 70.0]
            controller:
              type: circle
              radius_m: 2.0
              omega_radps: 0.3
              yaw_follows_tangent: true
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          - t_range: [70.0, 80.0]
            controller:
              type: circle
              radius_m: 2.0
              omega_radps: 0.3
              yaw_follows_tangent: true
            wind: {type: constant, magnitude_N: 2.0, direction: [0, 1, 0]}

          # Random yaw + small translation + wind step
          - t_range: [80.0, 100.0]
            controller:
              type: yaw_random
              yaw_rate_range_radps: [-0.5, 0.5]
              translation_rate_range_mps: [-0.3, 0.3]
              resample_interval_s: 0.5
            wind:
              type: step
              step_time_s: 90.0
              from_magnitude_N: 2.0
              to_magnitude_N: 6.0
              direction: [1, 0, 0]
        counterfactual: true

    # ==========================================================
    # MODE 5: MODERATE ATTITUDE (15%)
    # Exposes wind torque and vertical-wind response.
    # ==========================================================
    attitude:
      fraction: 0.15
      definition:
        duration_s: 100.0
        altitudes: [1.0, 1.5, 2.0]
        segments:
          # Roll step sequence, horizontal wind
          - t_range: [0.0, 15.0]
            controller:
              type: attitude_step_sequence
              steps:
                - {t_offset_s: 0.0,  roll_deg: 15, hold_s: 5.0}
                - {t_offset_s: 5.0,  roll_deg: 30, hold_s: 5.0}
                - {t_offset_s: 10.0, roll_deg: 0,  hold_s: 5.0}
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # Pitch step sequence, vertical wind (updraft)
          - t_range: [15.0, 30.0]
            controller:
              type: attitude_step_sequence
              steps:
                - {t_offset_s: 0.0,  pitch_deg: 15, hold_s: 5.0}
                - {t_offset_s: 5.0,  pitch_deg: 30, hold_s: 5.0}
                - {t_offset_s: 10.0, pitch_deg: 0,  hold_s: 5.0}
            wind: {type: constant, magnitude_N: 2.0, direction: [0, 0, 1]}

          # Roll sine, horizontal wind
          - t_range: [30.0, 45.0]
            controller:
              type: attitude_sine
              axis: roll
              amplitude_deg: 20
              frequency_hz: 0.3
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # Same roll sine, vertical wind → isolates vertical-wind torque
          - t_range: [45.0, 60.0]
            controller:
              type: attitude_sine
              axis: roll
              amplitude_deg: 20
              frequency_hz: 0.3
            wind: {type: constant, magnitude_N: 2.0, direction: [0, 0, 1]}

          # Combined roll-pitch sine, horizontal wind
          - t_range: [60.0, 75.0]
            controller:
              type: attitude_sine
              axis: roll_pitch
              amplitude_deg: 15
              frequency_hz: 0.3
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # Combined roll-pitch sine, diagonal wind
          - t_range: [75.0, 90.0]
            controller:
              type: attitude_sine
              axis: roll_pitch
              amplitude_deg: 15
              frequency_hz: 0.3
            wind:
              type: constant
              magnitude_N: 2.0
              direction: [0.7071, 0.7071, 0.0]

          # Random attitude + wind step
          - t_range: [90.0, 100.0]
            controller:
              type: attitude_random
              roll_range_deg: [-15, 15]
              pitch_range_deg: [-15, 15]
              resample_interval_s: 0.5
            wind:
              type: step
              step_time_s: 95.0
              from_magnitude_N: 2.0
              to_magnitude_N: 6.0
              direction: [1, 0, 0]
        counterfactual: true

    # ==========================================================
    # MODE 6: COMBINED TRANSLATIONAL + ROTATIONAL (17.5%)
    # The coupled manifold — closest to evaluation conditions.
    # ==========================================================
    combined:
      fraction: 0.175
      definition:
        duration_s: 100.0
        altitudes: [1.0, 2.0, 3.0]
        segments:
          # Figure-eight, constant wind
          - t_range: [0.0, 20.0]
            controller:
              type: figure_eight
              amplitude_m: [2.0, 1.5]
              frequency_hz: 0.1
              altitude_m: 1.5
              yaw_follows_tangent: true
            wind: {type: constant, magnitude_N: 2.0, direction: [1, 0, 0]}

          # Same figure-eight, wind step 2→6 N
          - t_range: [20.0, 40.0]
            controller:
              type: figure_eight
              amplitude_m: [2.0, 1.5]
              frequency_hz: 0.1
              altitude_m: 1.5
              yaw_follows_tangent: true
            wind:
              type: step
              step_time_s: 30.0
              from_magnitude_N: 2.0
              to_magnitude_N: 6.0
              direction: [1, 0, 0]

          # Helix climb, sinusoidal wind
          - t_range: [40.0, 55.0]
            controller:
              type: helix_climb
              radius_m: 2.0
              omega_radps: 0.3
              climb_rate_mps: 0.2
              yaw_follows_tangent: true
            wind:
              type: sinusoidal
              amplitude_N: 3.0
              frequency_hz: 0.2
              direction: [1, 0, 0]

          # Helix descent, gust at midpoint
          - t_range: [55.0, 70.0]
            controller:
              type: helix_descent
              radius_m: 2.0
              omega_radps: 0.3
              descent_rate_mps: 0.2
              yaw_follows_tangent: true
            wind:
              type: gust
              gust_time_s: 62.0
              from_magnitude_N: 2.0
              peak_magnitude_N: 6.0
              duration_s: 0.5
              direction: [1, 0, 0]

          # Random waypoints, Dryden turbulence
          - t_range: [70.0, 85.0]
            controller:
              type: waypoint_random
              sample_radius_m: 3.0
              sample_interval_s: 2.0
            wind:
              type: dryden
              sigma: 1.5
              length_scale_m: 100.0

          # Random waypoints, wind step 6→0
          - t_range: [85.0, 100.0]
            controller:
              type: waypoint_random
              sample_radius_m: 3.0
              sample_interval_s: 2.0
            wind:
              type: step
              step_time_s: 92.0
              from_magnitude_N: 6.0
              to_magnitude_N: 0.0
              direction: [1, 0, 0]
        counterfactual: true

  # ── Expected output summary (informational) ──────────────────
  report:
    expected_trajectories:
      hover:         10      # 10 × 15 s  =   150 s
      translation:   28      # 28 × 100 s =  2800 s
      acceleration:  18      # 18 × 100 s =  1800 s
      yaw:           12      # 12 × 100 s =  1200 s
      attitude:      15      # 15 × 100 s =  1500 s
      combined:      17      # 17 × 100 s =  1700 s
      # counterfactual replays: 90 scripted trajectories × 2 = 180
      # total trajectories written: 100 base + 90 replays = 190
      # total simulation steps: ~ (100 × 100 × 100 Hz) ≈ 10^6 base
      #                        + (90 × 100 × 100 Hz) ≈ 0.9 × 10^6 replay
    storage_estimate_gb: 0.5    # 15 float32 columns × ~2M rows × 4 bytes
```

---

## Supporting Documentation

### Controller library the generation script must implement

| `controller.type` | Produces (per step) | Notes |
|---|---|---|
| `hover` | `target_thrust = mg + m·(kp_z·(z_d − z) + kd_z·(−vz))`, `target_rate = 0` | Uses `rate_controller_gains` |
| `line` | velocity command → position-target moving at `velocity_mps` → P-D tracking | Requires starting position at segment entry |
| `circle` | circular position target with `radius_m`, `omega_radps`; optional yaw tangent | Center defaults to current xy |
| `figure_eight` | Lissajous with amplitudes `amplitude_m` and `frequency_hz` | |
| `helix_climb` / `helix_descent` | circle + `climb_rate_mps` or `descent_rate_mps` in z | |
| `waypoint_random` | random waypoint every `sample_interval_s` within `sample_radius_m` | |
| `attitude_step_sequence` | directly command `target_rate` from step list | Bypasses position loop on roll/pitch axes |
| `attitude_sine` | `target_rate = A·cos(2π·f·t)` on the chosen axis | |
| `attitude_random` | resample `target_rate` from range every `resample_interval_s` | |
| `yaw_step` / `yaw_sine` / `yaw_random` | yaw-only rate command | Other axes track position |
| `thrust_ramp` | `target_thrust = m·(g + a_cmd)` where `a_cmd` ramps over `ramp_duration_s` | `axis ∈ {x, y, z}` |
| `thrust_sinusoid` | `a_cmd = amplitude_frac · g · sin(2π·f·t)` | |
| `combined_ramp_roll` | thrust ramp + simultaneous roll-rate command | Tests coupling |
| `random_action` | sample `(T, ω_roll, ω_pitch, ω_yaw)` from `thrust_range_N` and `rate_range_radps` every `resample_interval_s` | Defeats action-wind shortcut |
| `open_loop` | direct motor commands `thrust_frac × max` on all 4 motors | Pure physics |

### Wind library the generation script must implement

| `wind.type` | Formula | Parameters |
|---|---|---|
| `none` | `F_w = 0` | — |
| `constant` | `F_w = magnitude_N × direction` | `magnitude_N`, `direction` |
| `step` | `F_w = from_magnitude if t < step_time else to_magnitude` (× direction) | `step_time_s`, `from_magnitude_N`, `to_magnitude_N`, `direction` |
| `multi_step` | piecewise `magnitude_N` by `steps[t_s]` | `direction`, `steps` |
| `sinusoidal` | `F_w = amplitude_N · sin(2π · frequency_hz · t) × direction` | `amplitude_N`, `frequency_hz`, `direction` (frequency × `ood.temporal_speed_scale`) |
| `gust` | `F_w = peak_magnitude_N × direction` when `t ∈ [gust_time_s, gust_time_s + duration_s]`, else `from_magnitude_N × direction` | `gust_time_s`, `from_magnitude_N`, `peak_magnitude_N`, `duration_s`, `direction` |
| `dryden` | first-order correlated Gaussian: `F[k] = α·F[k−1] + √(1−α²)·N(0, σ)` with `α = exp(−dt/τ)` and `τ = length_scale_m / 10` | `sigma`, `length_scale_m` |

All `magnitude_N` values are **scaled by `ood.wind_scale`** before being applied. When `ood.wind_directions_override` is set, it replaces every segment's `direction`.

### Physics OOD application

When `ood.mass_scale ≠ 1.0`, `ood.inertia_scale ≠ 1.0`, or `ood.drag_scale ≠ 1.0`:

1. Read the drone's base parameters from `hummingbird.yaml`
2. Multiply mass, inertia tensor, and drag coefficients by the scale
3. Write into the drone's USD prim via `PhysxRigidBodyAPI` before `drone.initialize()`
4. Log the applied values in the trajectory's `.json` metadata

### Counterfactual replay mechanics

For each scripted trajectory in `counterfactual.modes` when `counterfactual.enabled = true`:

```
Variant 0 (nominal):
  wind_force(t) = wind(t) × wind_multipliers[0] × ood.wind_scale
  For each step:
    u_t = controller(state_t, ...)      ← controller active
    record (state_t, u_t, wind_force(t), state_{t+1})
    save actions to buffer

Variant 1..N (replay):
  wind_force(t) = wind(t) × wind_multipliers[i] × ood.wind_scale
  For each step:
    u_t = buffer[t]                      ← bypass controller, use recorded action
    record (state_t, u_t, wind_force(t), state_{t+1})
```

Both variants share `pair_id`, but each is saved to its own `.npz`. A training script can load pairs by `pair_id` and enforce that the same `(s_t, u_t)` produces different `s_{t+1}` under different `F_w` — this is the direct supervision that forces the residual to learn from the IMU, not from \(u_t\).

### OOD regime quick reference

| Regime | Hydra override | Expected effect |
|---|---|---|
| Training | (none) | Wind ≤ 4 N, all directions, moderate turbulence |
| Magnitude OOD | `ood.wind_scale=2.0` | Wind 4–8 N |
| Directional OOD | `ood.wind_directions_override='[[0,0,1],[0,0,-1]]'` | Vertical-only wind |
| Temporal OOD | `ood.temporal_speed_scale=2.0` | 2× faster sinusoidal, faster gusts |
| Gust OOD | `ood.wind_scale=1.5` + combined mode focus | Higher σ Dryden, larger gusts |
| Physics OOD | `ood.mass_scale=1.15 ood.inertia_scale=1.10 ood.drag_scale=1.3` | +10–15% perturbations |

---

## What You Now Have

This single config file:

1. **Encodes all six modes** with exact segment timings, controllers, and wind specifications from the recipe
2. **Matches the working `manual_flight` setup** — same drone, same `RateController`, same CPU physics, same gains structure
3. **Supports all four OOD regimes** through one-line Hydra overrides
4. **Specifies counterfactual replay** at the trajectory level so the generation script knows when to bypass the controller
5. **Documents the required controller and wind libraries** so the generation script's implementation is unambiguous
6. **Includes a `report` block** so you can sanity-check the expected dataset volume before running

The companion `generate_dataset.py` reads this config and produces the trajectories. If you need me to write that script next — the one that implements the controller library, wind library, counterfactual replay, and `.npz` + `.json` output — I can produce it following the same pattern as the working `manual_flight.py`.