# Revised Hover Mode Configuration

The original hover block captured a single, narrow scenario (settle → wind step). Your "what to vary" list calls for **disturbance-recovery physics**, which is exactly where hover becomes useful as training data — even if it remains the weakest mode overall.

The rewrite below replaces the single timeline with a **scenario-based structure**. Each trajectory samples one scenario; the scenario determines initial conditions, phase structure, and wind profile. This produces the same 10% data volume but distributes it across the 10 scenarios your list identifies.

```yaml
# ==========================================================
# MODE 1: HOVER (10%) — scenario-based disturbance-recovery
# ==========================================================
# Purpose: capture controller transients when the drone is
# disturbed from equilibrium. Every scenario except calm_hover
# contains at least one disturbance event followed by a
# recovery window.
#
# This mode remains the weakest supervision for d_θ because
# wind is largely absorbed by the controller (Problem 1), but
# the *transient* after each disturbance is real state evolution
# that the world model can learn from.
#
# Trajectory count: 10 (10% of 100), with coverage guarantee
# so every scenario appears at least once.
# ==========================================================
hover:
  fraction: 0.10

  definition:
    # ── Trajectory length ─────────────────────────────────
    # Sampled per trajectory; short/medium/long coverage.
    duration_range_s: [10.0, 20.0]

    # ── Hover altitude ────────────────────────────────────
    # One sampled uniformly per trajectory.
    altitudes_m: [0.5, 1.0, 1.5, 2.5]

    # ── Initial conditions ────────────────────────────────
    # Applied before sim.play(), after drone.spawn() and before
    # drone.initialize(). Overrides spawn defaults.
    # Scenario-level `initial_conditions` (if present) override
    # these mode-level defaults.
    initial_conditions:
      position_xy_range_m: [-2.0, 2.0]       # uniform per axis
      position_z_offset_m: [-0.15, 0.15]     # relative to sampled altitude
      velocity_magnitude_range_mps: [0.0, 0.0]  # zero by default
      velocity_direction: random_uniform     # unit sphere, random
      roll_offset_deg: 0.0                   # level by default
      pitch_offset_deg: 0.0
      yaw_range_rad: [-3.14159, 3.14159]     # full circle

    # ── Sensor noise ──────────────────────────────────────
    # Applied POST-HOC to logged IMU data, NOT during simulation.
    # Rationale: injecting noise into the simulator would feed it
    # back to the controller, changing behaviour. Post-hoc
    # application yields the same clean flight recorded once,
    # with noisy variants generated offline during training-time
    # augmentation.
    sensor_noise:
      apply_during_sim: false
      imu_accel_std_mps2: 0.02
      imu_gyro_std_radps: 0.005
      imu_accel_bias_mps2: [0.0, 0.0, 0.0]    # fixed per trajectory, sampled from
                                              #   range below if desired
      imu_gyro_bias_radps: [0.0, 0.0, 0.0]
      noise_seed_offset: 200000               # distinct from action/IC RNG stream

    # ── Wind direction vocabulary ─────────────────────────
    # `direction: from_list` in any scenario samples one of these
    # uniformly. Individual scenarios can pin a specific direction.
    directions:
      headwind:        [ 1.0,  0.0, 0.0]
      tailwind:        [-1.0,  0.0, 0.0]
      crosswind_right: [ 0.0,  1.0, 0.0]
      crosswind_left:  [ 0.0, -1.0, 0.0]
      diagonal_ne:     [ 0.7071,  0.7071, 0.0]
      diagonal_nw:     [-0.7071,  0.7071, 0.0]
      diagonal_se:     [ 0.7071, -0.7071, 0.0]
      diagonal_sw:     [-0.7071, -0.7071, 0.0]
      updraft:         [ 0.0, 0.0,  1.0]
      downdraft:       [ 0.0, 0.0, -1.0]

    # ── Scenario templates ────────────────────────────────
    # One sampled per trajectory (with coverage guarantee below).
    # Each phase has a fixed `duration_s` and a `wind` spec.
    # Trajectory length = sum of phase durations. If this sum
    # differs from a sampled `duration_range_s`, scale phase
    # durations proportionally so total = sampled value.
    scenarios:

      # ─────────────────────────────────────────────────
      # S1: Calm hover (baseline)
      # ─────────────────────────────────────────────────
      - name: calm_hover
        weight: 0.15
        initial_conditions:
          velocity_magnitude_range_mps: [0.0, 0.0]
          roll_offset_deg: 0.0
          pitch_offset_deg: 0.0
        phases:
          - duration_s: 12.0
            wind: {type: none}

      # ─────────────────────────────────────────────────
      # S2: Light constant wind, steady compensation
      # ─────────────────────────────────────────────────
      - name: constant_wind_steady
        weight: 0.10
        phases:
          - duration_s: 3.0
            wind: {type: none}
          - duration_s: 12.0
            wind:
              type: constant
              magnitude_N: [1.0, 3.0]         # sampled
              direction: from_list

      # ─────────────────────────────────────────────────
      # S3: Sudden lateral gust with recovery
      # Controller must reject the impulse and return to hover.
      # ─────────────────────────────────────────────────
      - name: lateral_gust_recovery
        weight: 0.10
        phases:
          - duration_s: 4.0
            wind: {type: none}
          - duration_s: 0.5
            wind:
              type: constant
              magnitude_N: [4.0, 7.0]         # strong gust
              direction: [0.0, 1.0, 0.0]      # pure lateral
          - duration_s: 10.0
            wind: {type: none}                # observe recovery

      # ─────────────────────────────────────────────────
      # S4: Wind direction rotation
      # Forces Rᵀ F_w learning within hover, not just in yaw mode.
      # ─────────────────────────────────────────────────
      - name: wind_direction_rotation
        weight: 0.10
        phases:
          - duration_s: 3.0
            wind: {type: none}
          - duration_s: 12.0
            wind:
              type: rotating
              magnitude_N: [1.0, 3.0]
              rotation_rate_radps: [0.3, 0.8]  # sampled
              initial_direction_rad: 0.0

      # ─────────────────────────────────────────────────
      # S5: Sinusoidal wind (time-varying load)
      # ─────────────────────────────────────────────────
      - name: sinusoidal_wind
        weight: 0.10
        phases:
          - duration_s: 3.0
            wind: {type: none}
          - duration_s: 12.0
            wind:
              type: sinusoidal
              amplitude_N: [1.5, 3.5]
              frequency_hz: [0.05, 0.30]      # sampled
              direction: from_list

      # ─────────────────────────────────────────────────
      # S6: Dryden turbulence
      # ─────────────────────────────────────────────────
      - name: dryden_turbulence
        weight: 0.10
        phases:
          - duration_s: 15.0
            wind:
              type: dryden
              sigma: [0.3, 1.0]               # sampled
              length_scale_m: 100.0

      # ─────────────────────────────────────────────────
      # S7: Wind step up (disturbance appears mid-flight)
      # ─────────────────────────────────────────────────
      - name: wind_step_up
        weight: 0.10
        phases:
          - duration_s: 3.0
            wind: {type: none}
          - duration_s: 12.0
            wind:
              type: constant
              magnitude_N: [2.0, 5.0]
              direction: from_list

      # ─────────────────────────────────────────────────
      # S8: Wind step down (relaxation transient)
      # ─────────────────────────────────────────────────
      - name: wind_step_down
        weight: 0.05
        phases:
          - duration_s: 6.0
            wind:
              type: constant
              magnitude_N: [3.0, 5.0]
              direction: from_list
          - duration_s: 9.0
            wind: {type: none}                # observe drift-back

      # ─────────────────────────────────────────────────
      # S9: Initial attitude disturbance
      # Start tilted; the controller must recover level flight.
      # ─────────────────────────────────────────────────
      - name: initial_attitude_disturbance
        weight: 0.10
        initial_conditions:
          roll_offset_deg: [10.0, 20.0]       # sampled
          pitch_offset_deg: [10.0, 20.0]      # sampled (one or both nonzero)
          velocity_magnitude_range_mps: [0.0, 0.0]
        phases:
          - duration_s: 12.0
            wind: {type: none}

      # ─────────────────────────────────────────────────
      # S10: Physics perturbation + wind
      # Mass/inertia/drag scaled; wind active.
      # ─────────────────────────────────────────────────
      - name: physics_perturbed_with_wind
        weight: 0.10
        physics_modifiers:
          mass_scale: [0.90, 1.15]
          inertia_scale: [0.90, 1.15]
          drag_scale: [0.80, 1.30]
        phases:
          - duration_s: 3.0
            wind: {type: none}
          - duration_s: 12.0
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction: from_list

    # ── Sampling policy ───────────────────────────────────
    sampling:
      guarantee_coverage: true
      # When true, sample each scenario at least once before
      # sampling by weight. With 10 hover trajectories and 10
      # scenarios, each scenario appears exactly once.
      #
      # When false, sample purely by weight — some scenarios may
      # be absent from small datasets, which is fine if the total
      # hover fraction is raised (e.g. 15% → 15 trajectories).

    # ── Expected composition ──────────────────────────────
    report:
      trajectory_count: 10
      scenarios_present: 10                   # one per scenario when
                                              # guarantee_coverage=true
      avg_duration_s: 13.0
      total_steps: 13000                      # ~13 s × 100 Hz × 10
      disturbance_recovery_segments: 5        # S3, S4, S5, S7, S8
      physics_perturbation_segments: 1        # S9 (mass/inertia/drag)
      attitude_disturbance_segments: 1        # S9 (initial attitude)
      gust_segments: 1                        # S3

    # Note: hover trajectories are NOT used to define OOD test sets
    # (they lack the identifiability property required for validating
    # d_θ). They exist purely so the world model does not learn
    # degenerate behaviour on near-equilibrium states during
    # multi-step rollouts under OOD evaluation.
```

---

## What Changed and Why

| Aspect | Original | Revised |
|---|---|---|
| Structure | Single fixed timeline (settle → wind step → settle) | 10 scenario templates, each with its own phase structure |
| Wind types | Constant and step only | Constant, gust, rotating, sinusoidal, Dryden, step-up, step-down |
| Wind directions | Single direction `[1,0,0]` | `from_list` sampling across 10 directions (head/tail/cross/diagonal/vertical) |
| Initial conditions | Uniform (spawn defaults) | Explicit `position_xy`, `velocity`, `roll_offset`, `pitch_offset`, `yaw` |
| Initial disturbances | None | S9 starts tilted; recovery transient captured |
| Physics perturbation | Absent | S10 with mass/inertia/drag scaling |
| Coverage | All 10 trajectories identical | 10 distinct scenarios with `guarantee_coverage: true` |
| Sensor noise | Absent | Post-hoc application spec (with noise seed offset) |
| Transient content | One 2 s step transient | Recovery windows of 9–12 s in S3, S4, S5, S7, S8, S9 |

---

## New Controller and Wind Types Needed

The generation script must implement these additional types (extending the original list):

**Wind types:**
- `rotating` — `F_w(t) = magnitude · [cos(θ₀ + ω·t), sin(θ₀ + ω·t), 0]` with `ω = rotation_rate_radps`

**Initial-condition setters:**
- `drone.set_world_poses(positions, orientations)` — for initial position, yaw, roll/pitch offset
- `drone.set_velocities(vel_6d)` — for initial velocity

**Per-trajectory physics modifiers:**
- `_apply_physics_modifiers(drone, mass_scale, inertia_scale, drag_scale)` — writes to `PhysxRigidBodyAPI` on `base_link` before `drone.initialize()`

---

## Balancing Hover Volume vs. Coverage

With `fraction: 0.10` and `guarantee_coverage: true`, you get exactly 10 trajectories = 1 per scenario. That's enough to seed the dataset with every transient type, but not enough to have statistical robustness per scenario.

If you find during training that hover-related errors dominate OOD evaluation, two escalation options:

| Option | Change | Effect |
|---|---|---|
| **Raise hover fraction** | `fraction: 0.15` | 15 trajectories, ~1–2 per scenario |
| **Keep 10% but reduce scenarios** | Drop S2, S6, S8 (weakest transients) | 7 scenarios × ~1.4 trajectories each |

The first option is preferable because all 10 scenarios produce useful data — none is truly redundant. The second option only makes sense if simulation time is a hard constraint.

---

## Sanity Check Before Committing the Data

For hover specifically, the following diagnostic distinguishes "useful disturbance-recovery data" from "the degenerate all-steady-state data" that Problem 1 warns about:

For each trajectory, compute:

```
transient_score = mean( ||s_{t+1} - s_t|| ) / mean( ||s|| )
```

- **Calm hover trajectory (S1):** `transient_score ≈ 0` — this is expected and desirable for exactly one trajectory
- **Gust/step/rotation trajectories (S3, S4, S5, S7, S8):** `transient_score > 0.05` — the state is actively changing during the disturbance and recovery windows
- **Initial-disturbance trajectory (S9):** `transient_score > 0.1` — the first 2 seconds are dominated by the recovery transient

If **all 10** hover trajectories come back with `transient_score < 0.01`, something is wrong — either the disturbances aren't being applied, or the controller is overpowered. Check that:
1. The wind force is actually being written to the prim (log `F_w` from `GetForceAttr().Get()` after setting it)
2. The drone's mass is what you expect (log `drone.MASS_0`)
3. The controller gains match the config (log the computed `target_thrust` at `t = 4.0 s` for S3)

If those check out but the transient is still small, the controller gains `kd_pos_xy` and `kd_pos_z` may be too high — the drone is stiff and doesn't visibly move. Reduce them to `0.5` and `1.5` respectively, and the disturbance will leave a larger visible footprint, giving the world model more state variation to learn from.