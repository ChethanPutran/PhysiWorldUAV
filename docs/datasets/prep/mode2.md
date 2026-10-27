# Revised Translation Mode Configuration

The original block had three defects that the "what to vary" list exposes:

1. **A duplicated segment** — `t_range: [20, 30]` appears twice (once for headwind, once for circle), which is an impossible timeline (overlapping phases).
2. **Almost no directional variety** — every non-random segment moves +X. Backward, sideways, diagonal are absent.
3. **No parameter randomization** — distance, velocity, acceleration profile, altitude change, and initial velocity are all fixed. Two trajectories generated from this config would be nearly identical.

The rewrite below converts the fixed timeline into **10 scenario templates**, each exercising a distinct combination from your list. This eliminates the duplicate-timeline bug by construction and gives every key "important scenario" a dedicated template.

```yaml
# ==========================================================
# MODE 2: TRANSLATION (27.5%)  — the workhorse
# ==========================================================
# Residual grows with (v − v_wind)² → clean identifiability.
# Every scenario exercises a distinct combination of:
#   direction (fwd/back/side/diagonal), distance, velocity,
#   acceleration profile, altitude change, initial velocity,
#   and wind type.
#
# Trajectory count: 28 (27.5% of 100). With guarantee_coverage
# each scenario gets 2–3 samples, so the per-scenario variance
# is preserved while still averaging over trajectories.
# ==========================================================
translation:
  fraction: 0.275

  definition:
    # ── Mode-level defaults (can be overridden per scenario) ─
    duration_range_s: [15.0, 30.0]

    altitudes_m: [1.0, 1.5, 2.0]           # sampled per trajectory
    altitude_ramp_m: [-0.5, +1.0]          # applied over the trajectory
                                           # unless scenario specifies z_change

    initial_conditions:
      position_xy_range_m: [-3.0, 3.0]     # random start xy
      position_z_offset_m: [-0.1, 0.1]
      velocity_magnitude_range_mps: [0.0, 0.0]  # stationary default
      velocity_direction: random_uniform
      roll_offset_deg: 0.0
      pitch_offset_deg: 0.0
      yaw_range_rad: [-3.14159, 3.14159]

    # ── Wind direction vocabulary ────────────────────────
    # `direction: from_list` samples one of these uniformly.
    directions:
      north:   [ 1.0,  0.0, 0.0]
      south:   [-1.0,  0.0, 0.0]
      east:    [ 0.0,  1.0, 0.0]
      west:    [ 0.0, -1.0, 0.0]
      ne:      [ 0.7071,  0.7071, 0.0]
      nw:      [-0.7071,  0.7071, 0.0]
      se:      [ 0.7071, -0.7071, 0.0]
      sw:      [-0.7071, -0.7071, 0.0]
      up:      [ 0.0, 0.0,  1.0]
      down:    [ 0.0, 0.0, -1.0]

    # ── Acceleration profile vocabulary ──────────────────
    # Controls how fast the target velocity is reached at the
    # start of a `line` segment.
    #   gradual    → 1.5 s ramp
    #   moderate   → 0.8 s ramp
    #   aggressive → 0.3 s ramp
    #   piecewise  → step ramps (0.2 s up, 0.5 s hold, 0.2 s down)
    acceleration_profiles:
      gradual: 1.5
      moderate: 0.8
      aggressive: 0.3
      piecewise: {up_s: 0.2, hold_s: 0.5, down_s: 0.2}

    # ── Scenario templates ───────────────────────────────
    scenarios:

      # ─────────────────────────────────────────────────
      # S1: Forward with headwind  [IMPORTANT — user list #1]
      # ─────────────────────────────────────────────────
      - name: forward_headwind
        weight: 0.12
        initial_conditions:
          velocity_magnitude_range_mps: [0.0, 0.0]
        phases:
          - duration_s: [18.0, 25.0]
            controller:
              type: line
              direction: [1, 0, 0]
              distance_range_m: [3.0, 8.0]
              velocity_range_mps: [0.5, 2.0]
              acceleration_profile: gradual
              altitude_m: null                   # use sampled mode-level
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction: [-1, 0, 0]              # headwind

      # ─────────────────────────────────────────────────
      # S2: Backward with tailwind
      # ─────────────────────────────────────────────────
      - name: backward_tailwind
        weight: 0.08
        initial_conditions:
          velocity_magnitude_range_mps: [0.0, 0.0]
        phases:
          - duration_s: [15.0, 22.0]
            controller:
              type: line
              direction: [-1, 0, 0]
              distance_range_m: [3.0, 6.0]
              velocity_range_mps: [0.5, 1.5]
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction: [1, 0, 0]               # pushes with motion

      # ─────────────────────────────────────────────────
      # S3: Sideways with crosswind  [IMPORTANT — user list #2]
      # ─────────────────────────────────────────────────
      - name: sideways_crosswind
        weight: 0.12
        phases:
          - duration_s: [15.0, 22.0]
            controller:
              type: line
              direction_from_list: [east, west]  # +Y or −Y
              distance_range_m: [2.0, 5.0]
              velocity_range_mps: [0.5, 1.5]
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 4.0]
              direction_from_list: [north, south] # perpendicular

      # ─────────────────────────────────────────────────
      # S4: Diagonal with rotating wind  [IMPORTANT — user list #3]
      # ─────────────────────────────────────────────────
      - name: diagonal_rotating_wind
        weight: 0.10
        phases:
          - duration_s: [18.0, 25.0]
            controller:
              type: line
              direction_from_list: [ne, nw, se, sw]
              distance_range_m: [3.0, 7.0]
              velocity_range_mps: [0.5, 1.5]
              acceleration_profile: gradual
            wind:
              type: rotating
              magnitude_N: [1.5, 3.0]
              rotation_rate_radps: [0.2, 0.5]
              initial_direction_rad: [0.0, 1.5708]  # 0 or π/2 start

      # ─────────────────────────────────────────────────
      # S5: Residual velocity start  [IMPORTANT — user list #4]
      # Drone begins already moving; controller doesn't need
      # an acceleration ramp. Tests transient elimination.
      # ─────────────────────────────────────────────────
      - name: residual_velocity_start
        weight: 0.08
        initial_conditions:
          velocity_magnitude_range_mps: [0.4, 1.2]
          velocity_direction: random_uniform
        phases:
          - duration_s: [12.0, 18.0]
            controller:
              type: line
              direction: [1, 0, 0]
              distance_range_m: [3.0, 6.0]
              velocity_range_mps: [0.5, 1.5]
              acceleration_profile: gradual
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [east, west]

      # ─────────────────────────────────────────────────
      # S6: Multi-waypoint with mid-flight wind change  [IMPORTANT — user list #5]
      # Tests target change while moving + wind disturbance
      # ─────────────────────────────────────────────────
      - name: multi_waypoint_wind_change
        weight: 0.12
        phases:
          - duration_s: [22.0, 30.0]
            controller:
              type: waypoint_sequence
              velocity_range_mps: [0.6, 1.2]
              acceleration_profile: moderate
              waypoints:                          # relative to start
                - {xy: [2.0, 0.0], z_offset_m:  0.0, duration_s: 5.0}
                - {xy: [2.0, 2.0], z_offset_m:  0.0, duration_s: 5.0}
                - {xy: [0.0, 2.0], z_offset_m: -0.2, duration_s: 5.0}
                - {xy: [0.0, 0.0], z_offset_m:  0.0, duration_s: 5.0}
            wind:
              type: multi_step
              direction_from_list: [north, east]
              steps:
                - {t_s:  0.0, magnitude_N: 2.0}
                - {t_s: 10.0, magnitude_N: 4.0}
                - {t_s: 15.0, magnitude_N: 1.0}

      # ─────────────────────────────────────────────────
      # S7: Aggressive acceleration (short distance, fast ramp)
      # ─────────────────────────────────────────────────
      - name: aggressive_acceleration
        weight: 0.08
        phases:
          - duration_s: [10.0, 15.0]
            controller:
              type: line
              direction: [1, 0, 0]
              distance_range_m: [4.0, 8.0]
              velocity_range_mps: [1.5, 2.5]
              acceleration_profile: aggressive
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction: [-1, 0, 0]              # headwind during ramp

      # ─────────────────────────────────────────────────
      # S8: Slow, gradual translation + sinusoidal wind
      # Captures the "gentle flight under time-varying load"
      # regime — closer to the OOD temporal extrapolation test.
      # ─────────────────────────────────────────────────
      - name: slow_gradual_sinusoidal
        weight: 0.08
        phases:
          - duration_s: [22.0, 30.0]
            controller:
              type: line
              direction_from_list: [east, west]
              distance_range_m: [2.0, 4.0]
              velocity_range_mps: [0.3, 0.8]
              acceleration_profile: gradual
            wind:
              type: sinusoidal
              amplitude_N: [1.5, 3.0]
              frequency_hz: [0.05, 0.15]
              direction: [1, 0, 0]

      # ─────────────────────────────────────────────────
      # S9: 3D translation with altitude change
      # Translates horizontally while climbing or descending.
      # ─────────────────────────────────────────────────
      - name: translation_with_climb
        weight: 0.08
        phases:
          - duration_s: [18.0, 25.0]
            controller:
              type: line
              direction_from_list: [north, east, ne]
              distance_range_m: [3.0, 6.0]
              z_change_m: [0.5, 1.5]              # sign sampled
              velocity_range_mps: [0.5, 1.0]
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [east, west]  # perpendicular to climb

      # ─────────────────────────────────────────────────
      # S10: Random actions with multi-step wind (decorrelation workhorse)
      # This is the segment that most effectively breaks the
      # action-wind correlation (Problem 2).
      # ─────────────────────────────────────────────────
      - name: random_actions_multi_step
        weight: 0.14
        phases:
          - duration_s: [30.0, 40.0]
            controller:
              type: random_action
              thrust_range_N: [4.0, 9.0]
              rate_range_radps: [-0.6, 0.6]
              resample_interval_s: 0.3
            wind:
              type: multi_step
              direction_from_list: [north, east, ne]
              steps:
                - {t_s:  0.0, magnitude_N: 0.0}
                - {t_s: 10.0, magnitude_N: 2.0}
                - {t_s: 20.0, magnitude_N: 4.0}
                - {t_s: 30.0, magnitude_N: 6.0}

    # ── Sampling policy ──────────────────────────────────
    sampling:
      guarantee_coverage: true
      # 28 translation trajectories across 10 scenarios with
      # coverage guarantee → each scenario sampled 2–3 times.
      # Every important scenario from the user's list appears.

    # ── Expected composition ─────────────────────────────
    report:
      trajectory_count: 28
      scenarios_present: 10
      avg_duration_s: 22.0
      total_steps: ~61600                 # 22 s × 100 Hz × 28
      # per-scenario average samples
      per_scenario_min: 2
      per_scenario_max: 3

      # Coverage matrix vs user's "important scenarios" list
      coverage:
        forward_with_headwind:      S1
        sideways_with_crosswind:    S3
        diagonal_changing_wind:     S4
        residual_velocity_start:    S5
        target_change_while_moving: S6
        # supplemental coverage
        backward_tailwind:          S2
        aggressive_acceleration:    S7
        slow_gradual_sinusoidal:    S8
        three_d_translation:        S9
        random_decorrelation:       S10

    # Note: translation trajectories ARE eligible for the OOD test
    # sets (unlike hover). The counterfactual replay (see mode-level
    # `counterfactual: true`) doubles the scripted scenarios into
    # counterfactual pairs, which is what enables the world model
    # to validate d_θ on OOD magnitude.

  counterfactual: true
```

---

## What Changed vs. the Original

| Defect in original | How the rewrite fixes it |
|---|---|
| Overlapping `t_range` segments (duplicate `[20, 30]`) | Scenario structure has one timeline per scenario; no overlaps possible |
| Only forward motion (+X) | S2 backward, S3 sideways, S4 diagonal, S9 climbs diagonally |
| Distance fixed | `distance_range_m` sampled per trajectory in every scenario |
| Velocity fixed at 1.0 m/s | `velocity_range_mps` sampled in every scenario (slow 0.3 → fast 2.5) |
| No acceleration profile control | `acceleration_profile` param with 4 modes (gradual/moderate/aggressive/piecewise) |
| No altitude change during translation | S9 exercises `z_change_m` in [−1.5, +1.5] |
| No residual-velocity start | S5 samples initial velocity from a non-zero range |
| No target-change-while-moving | S6 uses 4-waypoint sequence |
| Only constant/step wind | S4 uses `rotating`, S8 uses `sinusoidal`, S10 uses `multi_step` |
| Only 2 wind directions | `direction_from_list` samples across 10 directions |
| Duplicated workload (nearly identical trajectories) | Scenarios have distinct physics: (v − v_wind)² drag, ramp profiles, target changes, and initial conditions all differ |

---

## Controller and Wind Library Additions

The generation script needs these new controller parameters and one new controller type:

### New `line` controller parameters

```yaml
controller:
  type: line
  direction: [x, y, z]           OR
  direction_from_list: [n1, n2, ...]
  distance_range_m: [min, max]   # sampled per trajectory
  velocity_range_mps: [min, max] # sampled per trajectory
  acceleration_profile: gradual|moderate|aggressive|piecewise
  z_change_m: [min, max]         # optional; altitude delta over segment
  altitude_m: null               # null → use sampled mode altitude
```

Behaviour: pick a straight-line path along `direction` at the sampled velocity, ramping from current velocity to target in the profile-defined duration, optionally changing altitude by `z_change_m` over the same duration.

### New `waypoint_sequence` controller

```yaml
controller:
  type: waypoint_sequence
  velocity_range_mps: [min, max]
  acceleration_profile: ...
  waypoints:
    - {xy: [dx, dy], z_offset_m: dz, duration_s: T}
    ...
```

Behaviour: move to each waypoint (offset from segment-entry pose) in `duration_s` seconds, then to the next, with velocity constrained by the ramp profile.

### New `rotating` wind type

```yaml
wind:
  type: rotating
  magnitude_N: [min, max]
  rotation_rate_radps: [min, max]
  initial_direction_rad: [min, max]
```

Behaviour:
\[
F_w(t) = m \cdot [\cos(\theta_0 + \omega t),\; \sin(\theta_0 + \omega t),\; 0]
\]
where \(m\), \(\omega\), \(\theta_0\) are sampled uniformly from their respective ranges.

---

## A Verification Command Before Committing the Data

After generation, run this check to ensure translation trajectories carry the identifiability signal that hover lacks:

```python
# For each translation trajectory, compute:
#   - correlation between v_t and F_w(t)
#   - correlation between u_t and F_w(t)
#   - mean ||s_{t+1} - f_phys(s_t, u_t)||
# If action-wind correlation is too high, Problem 2 is not defeated.

import numpy as np

for npz_path in translation_files:
    data = np.load(npz_path)
    v = data["velocity"]                 # (T, 3)
    u = data["action"]                   # (T, 4)
    F = data["wind_force_N"]             # (T, 3)

    # Scalar summaries
    v_mag = np.linalg.norm(v, axis=1)
    F_mag = np.linalg.norm(F, axis=1)
    u_thrust = u[:, 0]                   # thrust component

    # Correlation u_t vs F_w — must be LOW
    corr_thrust = abs(np.corrcoef(u_thrust, F_mag)[0, 1])

    # Correlation v_t vs F_w — must be HIGH for translation
    corr_vel = abs(np.corrcoef(v_mag, F_mag)[0, 1])

    print(f"{npz_path.name}: "
          f"corr(u_thrust, F_w)={corr_thrust:.2f}  "
          f"corr(|v|, F_w)={corr_vel:.2f}")
```

**Acceptance thresholds** (per your thesis's identifiability requirement):

| Check | Pass | Warning | Fail |
|---|---|---|---|
| `corr(|v|, F_w)` | ≥ 0.3 | 0.1–0.3 | < 0.1 (wind not affecting flight state) |
| `corr(u_thrust, F_w)` | ≤ 0.3 | 0.3–0.6 | > 0.6 (action-wind correlation too strong — Problem 2 not defeated) |
| Residual magnitude | grows with `F_mag` | flat | ≈ 0 (Problem 1) |

If `random_actions_multi_step` (S10) trajectories have `corr(u_thrust, F_w) > 0.4`, the decorrelation is insufficient — the wind step timings may be too slow relative to the 0.3 s random-action resample rate. Reduce `resample_interval_s` to `0.15` s and the multi-step wind will have less time to imprint on the actions.

If `forward_headwind` (S1) trajectories have `corr(|v|, F_w) < 0.2`, the wind magnitude is too small to affect the flight state measurably. Raise `magnitude_N` to `[2.5, 5.0]` and the velocity variation will become visible in the state trajectory.