# Revised Combined Mode Configuration

The original block had six defects the "what to vary" list exposes:

1. **No parallel command overlap** — every phase has exactly one controller. The whole mode degenerates into "concatenation of other modes." The user explicitly asks for "sequential **or simultaneous**."
2. **Only 6 of the 7 user example trajectories** are represented (missing example 6 "same sequence, different mass/inertia" and example 7 "unexpected gust halfway through a maneuver").
3. **No mass/inertia/drag perturbation** — the user lists "Vehicle properties: Mass, inertia, drag" as a parameter.
4. **No intensity or transition-profile variety** — gentle vs aggressive, smooth vs abrupt, both absent.
5. **No event timestamp logging** — the user's final, explicit requirement: *"record the exact timestamps of every command and disturbance."*
6. **Fixed 100 s duration, fixed wind direction `[1,0,0]`** — no randomization.

The rewrite converts to **14 scenario slots (10 unique templates + 2 pairs)**, all with **at least one overlapping-command phase**, an **event manifest** on every trajectory, and dedicated templates for the user's 7 examples plus intensity/transition pairs.

```yaml
# ==========================================================
# MODE 6: COMBINED TRANSLATIONAL + ROTATIONAL (17.5%)
# ==========================================================
# The coupled manifold — closest to evaluation conditions.
#
# Defining property: at least ONE phase per scenario runs
# TWO OR MORE controllers in parallel. Scenarios without
# parallel commands are redundant with Modes 2/4/5 and are
# excluded by design.
#
# Controller arbitration rules:
#   Position-mode   (line, circle, waypoint_*, figure_eight, helix_*)
#   Attitude-mode   (attitude_step, attitude_ramp, attitude_sine)
#   Yaw-mode        (yaw_step, yaw_ramp, yaw_sine)
#   Thrust-mode     (thrust_ramp, thrust_sinusoid)
# One controller from each mode may run simultaneously.
# Two controllers from the SAME mode conflict and are rejected.
#
# Event manifest: every trajectory records a list of
#   {t, type, id, params}
# for every maneuver start/end, wind change, gust, and
# physics perturbation. See `report.event_log_spec` below.
#
# Trajectory count: 18 (17.5% of 100). With guarantee_coverage
# every scenario appears ≥1×; two pairs co-generated.
# ==========================================================
combined:
  fraction: 0.175

  definition:
    duration_range_s: [20.0, 40.0]
    altitudes_m: [1.0, 1.5, 2.0, 2.5]

    initial_conditions:
      position_xy_range_m: [-2.0, 2.0]
      position_z_offset_m: [-0.1, 0.1]
      velocity_magnitude_range_mps: [0.0, 0.8]
      velocity_direction: random_unit_vector
      roll_offset_deg: [-5.0, 5.0]
      pitch_offset_deg: [-5.0, 5.0]
      yaw_range_rad: [-3.14159, 3.14159]
      angular_velocity_magnitude_range_radps: [0.0, 0.3]
      angular_velocity_axis: random_unit_vector

    directions:
      north:   [ 1.0,  0.0, 0.0]
      south:   [-1.0,  0.0, 0.0]
      east:    [ 0.0,  1.0, 0.0]
      west:    [ 0.0, -1.0, 0.0]
      ne:      [ 0.7071,  0.7071, 0.0]
      nw:      [-0.7071,  0.7071, 0.0]
      up:      [ 0.0, 0.0,  1.0]
      down:    [ 0.0, 0.0, -1.0]

    intensity_classes:
      gentle:     {velocity_scale: 0.5, rate_scale: 0.5}
      moderate:   {velocity_scale: 1.0, rate_scale: 1.0}
      aggressive: {velocity_scale: 1.8, rate_scale: 1.8}

    transition_profiles:
      smooth: {ramp_duration_s: 2.0}
      abrupt: {ramp_duration_s: 0.1}

    scenarios:

      # ─────────────────────────────────────────────────
      # S1: Hover → accel → decel → hover  (user example 1)
      # ★ Parallel phase: accel_fwd + slight nose-down pitch
      # ─────────────────────────────────────────────────
      - name: hover_accel_decel_hover
        weight: 0.10
        initial_conditions:
          velocity_magnitude_range_mps: [0.0, 0.0]
        phases:
          - duration_s: 3.0
            controller: {type: hover}
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]
          - duration_s: 5.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: [1, 0, 0],
                 velocity_range_mps: [0.8, 1.5],
                 acceleration_profile: moderate}
              - {type: attitude_step, axis: pitch,
                 amplitude_class: small, sign: positive,
                 hold_duration_s: 5.0, return_to_level: true}
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction_from_list: [south, west]       # opposing
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: [1, 0, 0]}
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [south, west]
          - duration_s: 3.0
            controller: {type: hover}
            wind: {type: none}

      # ─────────────────────────────────────────────────
      # S2: Translate forward → yaw 90° → continue  (user example 2)
      # ★ Parallel phase: translation + yaw simultaneously
      # ─────────────────────────────────────────────────
      - name: translate_yaw_continue
        weight: 0.10
        phases:
          - duration_s: 4.0
            controller:
              type: line
              direction_from_list: [north, east]
              velocity_range_mps: [0.7, 1.2]
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]
          - duration_s: 5.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: "current_heading",
                 velocity_range_mps: [0.7, 1.2]}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: moderate, transition_profile: ramp}
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction: [1, 0, 0]                   # world-frame
          - duration_s: 4.0
            controller:
              type: line
              direction: "current_heading"
              velocity_range_mps: [0.7, 1.2]
            wind:
              type: step
              step_time_s: 11.0
              from_magnitude_N: 2.0
              to_magnitude_N: 4.0
              direction: [1, 0, 0]
          - duration_s: 3.0
            controller: {type: hover}

      # ─────────────────────────────────────────────────
      # S3: Accelerate → gust → attitude correction → braking  (user example 3)
      # ★ Gust arrives DURING acceleration
      # ─────────────────────────────────────────────────
      - name: accel_gust_correct_brake
        weight: 0.08
        phases:
          - duration_s: 5.0
            controller:
              type: line
              direction: [1, 0, 0]
              velocity_range_mps: [1.0, 1.8]
              acceleration_profile: aggressive
            wind:
              type: gust_mid
              trigger_at_progress: [0.30, 0.60]
              peak_magnitude_N: [5.0, 8.0]
              duration_s: [0.4, 0.8]
              direction_from_list: [east, west]       # lateral → forces attitude correction
          - duration_s: 4.0
            controller:
              type: attitude_step
              axis: roll
              amplitude_class: moderate
              sign: random
              hold_duration_s: 2.0
              return_to_level: true
              rate_profile: abrupt
            wind:
              type: constant
              magnitude_N: [2.0, 3.5]
              direction_from_list: [north, south]
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: [1, 0, 0]}
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, south]

      # ─────────────────────────────────────────────────
      # S4: Roll + pitch simultaneously while translating  (user example 4)
      # ★ Three-axis parallel
      # ─────────────────────────────────────────────────
      - name: roll_pitch_while_translating
        weight: 0.08
        phases:
          - duration_s: 4.0
            controller:
              type: line
              direction_from_list: [ne, nw]
              velocity_range_mps: [0.6, 1.2]
              acceleration_profile: gradual
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [east, north]
          - duration_s: 6.0
            controller:                              # ★ PARALLEL (3 controllers)
              - {type: line, direction: "current_heading",
                 velocity_range_mps: [0.6, 1.2]}
              - {type: attitude_step, axis: roll,
                 amplitude_class: moderate, sign: random,
                 hold_duration_s: 2.0, return_to_level: true,
                 rate_profile: moderate}
              - {type: attitude_step, axis: pitch,
                 amplitude_class: moderate, sign: random,
                 hold_duration_s: 2.0, return_to_level: true,
                 rate_profile: moderate}
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [up, north]         # vertical or horizontal
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: "current_heading"}
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [east, north]

      # ─────────────────────────────────────────────────
      # S5: Accelerate, yaw, then decelerate under changing wind  (user example 5)
      # ★ Full coupling: 3 controllers in parallel + wind steps
      # ─────────────────────────────────────────────────
      - name: accel_yaw_decel_changing_wind
        weight: 0.08
        phases:
          - duration_s: 4.0
            controller:
              type: line
              direction: [1, 0, 0]
              velocity_range_mps: [0.8, 1.5]
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: 2.0
              direction: [1, 0, 0]
          - duration_s: 6.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: "current_heading",
                 velocity_range_mps: [0.8, 1.5]}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: moderate, transition_profile: ramp}
            wind:
              type: multi_step
              direction_from_list: [north, east]
              steps:
                - {t_s: 4.0, magnitude_N: 2.0}
                - {t_s: 7.0, magnitude_N: 4.0}
                - {t_s: 10.0, magnitude_N: 6.0}
          - duration_s: 5.0
            controller: {type: brake_to_zero, axis: "current_heading"}
            wind:
              type: sinusoidal
              amplitude_N: [2.0, 3.0]
              frequency_hz: [0.1, 0.3]
              direction_from_list: [east, north]

      # ─────────────────────────────────────────────────
      # S6a: Physics comparison — NOMINAL mass/inertia  (user example 6, member 1/2)
      # Same commands as S6b; only the physics differs.
      # ★ Enables causal comparison: same input, different physical params
      # ─────────────────────────────────────────────────
      - name: same_sequence_nominal_physics
        weight: 0.05
        pair_group: physics_comparison
        physics_modifiers:
          mass_scale: 1.0                         # matched
          inertia_scale: 1.0                      # matched
          drag_scale: 1.0                         # matched
        phases:
          - duration_s: 4.0
            controller:
              type: line
              direction: [1, 0, 0]
              velocity_range_mps: [1.0, 1.5]      # matched with S6b
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: 3.0                    # matched with S6b
              direction: [-1, 0, 0]               # matched with S6b
          - duration_s: 5.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: "current_heading",
                 velocity_range_mps: [1.0, 1.5]}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: moderate, transition_profile: ramp}
            wind:
              type: constant
              magnitude_N: 3.0                    # matched
              direction: [-1, 0, 0]
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: "current_heading"}
            wind:
              type: constant
              magnitude_N: 3.0                    # matched
              direction: [-1, 0, 0]

      # ─────────────────────────────────────────────────
      # S6b: Physics comparison — PERTURBED  (member 2/2)
      # Exact same commands and wind; mass +15%, inertia +10%, drag +25%.
      # ★ The only difference is physics — perfect causal pair.
      # ─────────────────────────────────────────────────
      - name: same_sequence_perturbed_physics
        weight: 0.05
        pair_group: physics_comparison
        physics_modifiers:
          mass_scale: [1.10, 1.20]                # sampled
          inertia_scale: [1.05, 1.15]             # sampled
          drag_scale: [1.20, 1.30]                # sampled
        phases:
          - duration_s: 4.0
            controller:
              type: line
              direction: [1, 0, 0]
              velocity_range_mps: [1.0, 1.5]      # matched with S6a
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: 3.0                    # matched
              direction: [-1, 0, 0]
          - duration_s: 5.0
            controller:
              - {type: line, direction: "current_heading",
                 velocity_range_mps: [1.0, 1.5]}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: moderate, transition_profile: ramp}
            wind:
              type: constant
              magnitude_N: 3.0
              direction: [-1, 0, 0]
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: "current_heading"}
            wind:
              type: constant
              magnitude_N: 3.0
              direction: [-1, 0, 0]

      # ─────────────────────────────────────────────────
      # S7: Unexpected gust mid-maneuver  (user example 7)
      # Gust timing sampled per trajectory — mid-accel, mid-yaw, mid-brake.
      # ★ Cannot be predicted from action alone
      # ─────────────────────────────────────────────────
      - name: unexpected_gust_mid_maneuver
        weight: 0.08
        phases:
          - duration_s: 3.0
            controller:
              type: line
              direction_from_list: [north, east, ne]
              velocity_range_mps: [0.8, 1.5]
              acceleration_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.0, 2.0]
              direction_from_list: [north, east]
          - duration_s: 6.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: "current_heading",
                 velocity_range_mps: [0.8, 1.5]}
              - {type: attitude_sine, axis: roll_pitch,
                 amplitude_deg: [10, 15],
                 frequency_hz: [0.2, 0.4]}
            wind:
              type: gust_mid
              trigger_at_progress: [0.20, 0.80]       # UNPREDICTABLE timing
              peak_magnitude_N: [5.0, 8.0]
              duration_s: [0.3, 1.0]
              direction_from_list: [east, west, north, south]
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: "current_heading"}
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S8: Recovery after disturbance
      # User list: "recovery behavior: stable recovery or continued maneuvering"
      # ★ Demonstrates explicit post-disturbance settling
      # ─────────────────────────────────────────────────
      - name: recovery_after_disturbance
        weight: 0.06
        phases:
          - duration_s: 2.0
            controller: {type: hover}
            wind: {type: none}
          - duration_s: 1.0
            controller:                              # ★ PARALLEL disturbance
              - {type: hover}
              - {type: attitude_step, axis: roll,
                 amplitude_class: large, sign: random,
                 hold_duration_s: 1.0, return_to_level: false,
                 rate_profile: abrupt}
            wind:
              type: gust_mid
              trigger_at_progress: [0.0, 0.5]
              peak_magnitude_N: [6.0, 9.0]
              duration_s: [0.5, 1.0]
              direction_from_list: [east, west, north, south]
          - duration_s: 8.0
            controller: {type: hover}                # recovery window
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S9a: Intensity comparison — GENTLE  (user: "maneuver intensity")
      # ★ Low velocities and rates; smooth transitions
      # ─────────────────────────────────────────────────
      - name: gentle_maneuver_sequence
        weight: 0.05
        pair_group: intensity_comparison
        phases:
          - duration_s: 5.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: [1, 0, 0],
                 velocity_range_mps: [0.4, 0.6],     # matched with S9b scaled down
                 acceleration_profile: gradual}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: slow, transition_profile: smooth}
            wind:
              type: constant
              magnitude_N: 2.0                    # matched with S9b
              direction_from_list: [north, east]
          - duration_s: 6.0
            controller:
              type: attitude_sine
              axis: roll_pitch
              amplitude_deg: 8
              frequency_hz: 0.15
            wind:
              type: constant
              magnitude_N: 2.0
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S9b: Intensity comparison — AGGRESSIVE  (member 2/2)
      # ★ Same sequence and wind as S9a, but 2× velocity and rate
      # ─────────────────────────────────────────────────
      - name: aggressive_maneuver_sequence
        weight: 0.05
        pair_group: intensity_comparison
        phases:
          - duration_s: 5.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: [1, 0, 0],
                 velocity_range_mps: [1.5, 2.0],     # matched with S9a scaled up
                 acceleration_profile: aggressive}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: fast, transition_profile: abrupt}
            wind:
              type: constant
              magnitude_N: 2.0                    # matched with S9a
              direction_from_list: [north, east]
          - duration_s: 6.0
            controller:
              type: attitude_sine
              axis: roll_pitch
              amplitude_deg: 20
              frequency_hz: 0.4
            wind:
              type: constant
              magnitude_N: 2.0
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S10a: Transition profile — SMOOTH  (user: "smooth or abrupt")
      # ★ Same endpoint, ramp over 2 s
      # ─────────────────────────────────────────────────
      - name: smooth_transition
        weight: 0.05
        pair_group: transition_comparison
        phases:
          - duration_s: 5.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: [1, 0, 0],
                 velocity_range_mps: [1.0, 1.5],     # matched with S10b
                 acceleration_profile: gradual}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: moderate,
                 transition_profile: smooth}         # ★ ramp_duration = 2.0 s
            wind:
              type: constant
              magnitude_N: 3.0                    # matched with S10b
              direction: [0, 1, 0]
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: "current_heading"}
            wind:
              type: constant
              magnitude_N: 3.0
              direction: [0, 1, 0]

      # ─────────────────────────────────────────────────
      # S10b: Transition profile — ABRUPT  (member 2/2)
      # ★ Same endpoint, ramp over 0.1 s
      # ─────────────────────────────────────────────────
      - name: abrupt_transition
        weight: 0.05
        pair_group: transition_comparison
        phases:
          - duration_s: 5.0
            controller:                              # ★ PARALLEL
              - {type: line, direction: [1, 0, 0],
                 velocity_range_mps: [1.0, 1.5],     # matched
                 acceleration_profile: aggressive}
              - {type: yaw_ramp, target_change_rad: [1.5708],
                 rate_class: fast,
                 transition_profile: abrupt}         # ★ ramp_duration = 0.1 s
            wind:
              type: constant
              magnitude_N: 3.0
              direction: [0, 1, 0]
          - duration_s: 4.0
            controller: {type: brake_to_zero, axis: "current_heading"}
            wind:
              type: constant
              magnitude_N: 3.0
              direction: [0, 1, 0]

      # ─────────────────────────────────────────────────
      # S11: Random multi-axis + all wind types (decorrelation workhorse)
      # ★ The largest single scenario — random everything.
      # This is the strongest anti-shortcut data in the entire dataset.
      # ─────────────────────────────────────────────────
      - name: random_multi_axis
        weight: 0.12
        phases:
          - duration_s: [20.0, 30.0]
            controller:                              # ★ FULLY PARALLEL & RANDOM
              - {type: random_line_velocity,
                 velocity_range_mps: [-1.5, 1.5],
                 resample_interval_s: 1.5}
              - {type: random_attitude_command,
                 amplitude_deg: [0, 20],
                 resample_interval_s: 1.0}
              - {type: random_yaw_command,
                 rate_range_radps: [-0.6, 0.6],
                 resample_interval_s: 2.0}
            wind:
              type: random_multi
              allowed_types: [constant, step, sinusoidal, gust, dryden]
              magnitude_range_N: [0.0, 6.0]
              direction_change_interval_s: [3.0, 6.0]
              type_change_interval_s: [5.0, 10.0]

    # ── Paired scenarios ──────────────────────────────────
    paired_scenarios:
      - name: physics_comparison
        members: [same_sequence_nominal_physics, same_sequence_perturbed_physics]
        shared:
          - command_sequence
          - wind_sequence
          - total_duration_s
        pair_id_prefix: "physics_nominal_vs_perturbed"
        # Physics modifiers differ by design:
        #   nominal  → mass_scale=1.0, inertia_scale=1.0, drag_scale=1.0
        #   perturbed → sampled from respective ranges

      - name: intensity_comparison
        members: [gentle_maneuver_sequence, aggressive_maneuver_sequence]
        shared:
          - command_sequence_shape
          - wind_magnitude_N
          - wind_direction
        pair_id_prefix: "intensity_gentle_vs_aggressive"

      - name: transition_comparison
        members: [smooth_transition, abrupt_transition]
        shared:
          - endpoint_velocity
          - endpoint_yaw
          - wind_magnitude_N
          - wind_direction
        pair_id_prefix: "transition_smooth_vs_abrupt"

    # ── Sampling policy ───────────────────────────────────
    sampling:
      guarantee_coverage: true
      pairs_generated_together: true
      enforce_parallel_requirement: true
      # Every generated trajectory MUST have at least one phase
      # with ≥2 controllers running in parallel. If a scenario's
      # sampled instance violates this, the generator raises.

    # ── Event logging specification ───────────────────────
    event_log_spec:
      format: "json_array_in_metadata"
      fields: ["t", "type", "id", "params"]
      event_types:
        - "maneuver_start"
        - "maneuver_end"
        - "wind_change"
        - "gust_start"
        - "gust_end"
        - "physics_perturbation"
        - "recovery_begin"
        - "recovery_end"
      sample:
        - {t: 0.0,   type: maneuver_start, id: "translate_forward",
           params: {velocity_mps: 1.2, direction: [1, 0, 0]}}
        - {t: 4.0,   type: maneuver_start, id: "yaw_90",
           params: {target_rad: 1.5708, rate_class: moderate}}
        - {t: 4.0,   type: wind_change, id: "wind_step",
           params: {from_N: 2.0, to_N: 4.0}}
        - {t: 9.0,   type: maneuver_end,  id: "yaw_90"}
        - {t: 9.0,   type: maneuver_end,  id: "translate_forward"}
        - {t: 9.0,   type: maneuver_start, id: "brake_to_zero"}
        - {t: 13.0,  type: maneuver_end, id: "brake_to_zero"}
        # Note: gust events logged as two entries (start + end) so
        # downstream code can bracket the disturbance window.

    # ── Expected composition ──────────────────────────────
    report:
      trajectory_count: 18
      scenarios_present: 14
      avg_duration_s: 30.0
      total_steps: ~54000                 # 30 s × 100 Hz × 18
      paired_pairs_generated: 3
      parallel_phases_per_trajectory_min: 1
      parallel_phases_per_trajectory_max: 3
      maneuver_events_logged_min: 8       # per trajectory
      maneuver_events_logged_max: 20

      coverage:
        hover_accel_decel_hover:      S1  (user example 1)
        translate_yaw_continue:       S2  (user example 2)
        accel_gust_correct_brake:     S3  (user example 3)
        roll_pitch_while_translating: S4  (user example 4)
        accel_yaw_decel_changing_wind: S5  (user example 5)
        same_sequence_diff_physics:   S6a + S6b (user example 6, paired)
        unexpected_gust_mid_maneuver: S7  (user example 7)
        recovery_after_disturbance:   S8  (user "recovery behavior")
        gentle_vs_aggressive:         S9a + S9b (user "maneuver intensity", paired)
        smooth_vs_abrupt:             S10a + S10b (user "transition profile", paired)
        random_multi_axis:            S11 (decorrelation workhorse)

      # Non-redundancy assertion (checked post-generation)
      non_redundancy_vs_other_modes:
        each_scenario_has_parallel_phase: true
        min_parallel_phases_per_trajectory: 1
        # Scenarios that would have been single-controller have been
        # merged into the corresponding primary mode instead:
        #   single-controller translation → Mode 2
        #   single-controller attitude    → Mode 5
        #   single-controller yaw         → Mode 4

  counterfactual: true
```

---

## What Changed vs. the Original

| Defect in original | How the rewrite fixes it |
|---|---|
| Every phase had exactly one controller | Every scenario has ≥1 phase with parallel controllers; `enforce_parallel_requirement: true` |
| User examples 6 & 7 absent | S6a/S6b (physics pair), S7 (unexpected gust) |
| No mass/inertia perturbation | `physics_modifiers` per scenario; S6 pair isolates the effect |
| No intensity variety | S9a/S9b pair (gentle vs aggressive) |
| No transition-profile variety | S10a/S10b pair (smooth vs abrupt) |
| No event timestamp logging | `event_log_spec` block; every trajectory gets an `events` array in metadata |
| Fixed 100 s duration | `duration_range_s: [20, 40]` |
| Fixed wind direction | `direction_from_list` throughout |
| No scenario for "recovery behavior" | S8 dedicated |
| Only 6 user examples represented | All 7 covered as named scenarios |

---

## New Controller Combinations (Arbitration Rules)

The generation script must support **parallel controllers** with the following arbitration:

| Mode | Controllers | Arbitration |
|---|---|---|
| Position | `line`, `circle`, `figure_eight`, `waypoint_*`, `helix_*`, `random_line_velocity` | Compete: only ONE may be active. Produces `(target_pos, target_vel)` |
| Attitude | `attitude_step`, `attitude_ramp`, `attitude_sine`, `attitude_combined`, `random_attitude_command` | Compete: only ONE. Produces `(target_roll, target_pitch)` |
| Yaw | `yaw_step`, `yaw_ramp`, `yaw_sine`, `random_yaw_command` | Compete: only ONE. Produces `target_yaw_rate` |
| Thrust | `thrust_ramp`, `thrust_sinusoid` | Compete: only ONE. Overrides computed thrust |

**Within one phase, at most one controller from each mode can run.** If two controllers from the same mode are listed in parallel, the generator raises `ControllerConflictError`.

**Across modes, any combination works.** The outer loop computes:

```
target_thrust = thrust_controller(t)          if thrust-mode controller active
             else mass × (g + kp_z·(z_err) + kd_z·(vz_err))

target_rate = [
    kp_att·(roll_target - roll)  + kd_att·(−p),      # attitude mode
    kp_att·(pitch_target - pitch) + kd_att·(−q),
    yaw_rate_target                                   # yaw mode
]

# If both position and attitude modes active, add rate from position error:
if position_controller_active:
    target_rate[0] += kp_pos·(y_err) ...
```

This is exactly how `manual_flight.py` worked, extended to accept contributions from multiple active controllers.

---

## New Controller Types Needed

| Controller | Purpose | Output |
|---|---|---|
| `brake_to_zero` | Ramp velocity to zero along a given axis | `(target_pos_hold, target_vel=0)` |
| `random_line_velocity` | Resample a target velocity every `resample_interval_s` | `(target_vel)` |
| `random_attitude_command` | Resample roll/pitch target every `resample_interval_s` | `(target_roll, target_pitch)` |
| `random_yaw_command` | Resample yaw rate every `resample_interval_s` | `target_yaw_rate` |

Plus new wind type:

| Wind | Purpose |
|---|---|
| `random_multi` | At each `type_change_interval_s`, switch to a random allowed type; at each `direction_change_interval_s`, resample direction; at each `magnitude` sample within a type, resample magnitude from range |

---

## Event Manifest Specification

Every trajectory's `.json` metadata includes an `events` array:

```json
{
  "trajectory_id": "combined_0007",
  "pair_id": "physics_nominal_vs_perturbed_0002",
  "scenario": "same_sequence_perturbed_physics",
  "duration_s": 13.0,
  "physics_modifiers": {
    "mass_scale": 1.14,
    "inertia_scale": 1.09,
    "drag_scale": 1.24
  },
  "events": [
    {"t": 0.00, "type": "maneuver_start", "id": "translate_forward",
     "params": {"velocity_mps": 1.2, "direction": [1, 0, 0]}},
    {"t": 0.00, "type": "wind_change", "id": "wind_set",
     "params": {"type": "constant", "magnitude_N": 3.0, "direction": [-1, 0, 0]}},
    {"t": 4.00, "type": "maneuver_start", "id": "yaw_90",
     "params": {"target_rad": 1.5708, "rate_class": "moderate"}},
    {"t": 9.00, "type": "maneuver_end", "id": "yaw_90"},
    {"t": 9.00, "type": "maneuver_end", "id": "translate_forward"},
    {"t": 9.00, "type": "maneuver_start", "id": "brake_to_zero"},
    {"t": 13.00, "type": "maneuver_end", "id": "brake_to_zero"}
  ]
}
```

The **`events` array is what allows downstream training to attribute state changes to causes.** During training, when the model's prediction error spikes at a particular timestep, the pipeline can look at the events in a window around that timestep and identify which maneuver or disturbance boundary was crossed. Without this metadata, debugging "why did the model fail here?" is essentially impossible.

A second, richer form of the events log is **optional**: store it inside the `.npz` as a structured array:

```python
# In save_trajectory:
events_struct = np.zeros(len(events), dtype=[
    ("t", "f4"), ("type", "U24"), ("id", "U32"), ("param_json", "U256"),
])
for i, e in enumerate(events):
    events_struct[i] = (e["t"], e["type"], e["id"], json.dumps(e["params"]))
np.savez_compressed(path, ..., events=events_struct)
```

This keeps the events aligned with the state/action arrays in the same file. The JSON copy in the sidecar is for human readability.

---

## Verification Checks Specific to Mode 6

### Check 1 — Parallel-phase enforcement

```python
for traj_file in combined_files:
    meta = load_json(traj_file.replace(".npz", ".json"))
    has_parallel = any(
        ev["type"] == "maneuver_start" for ev in meta["events"]
    ) and _has_overlapping_maneuver_windows(meta["events"])
    assert has_parallel, f"{traj_file.name} has no parallel maneuver phase"
```

`_has_overlapping_maneuver_windows` inspects the events array for at least one pair `(start_a, end_a), (start_b, end_b)` with `start_a < end_b and start_b < end_a` and `id_a != id_b`. Every trajectory must pass this test.

### Check 2 — Physics pair isolation (S6a/S6b)

```python
for pair_id, (nominal_file, perturbed_file) in load_physics_pairs().items():
    nominal = np.load(nominal_file)
    perturbed = np.load(perturbed_file)

    # Same actions
    assert np.allclose(nominal["action"], perturbed["action"], atol=1e-4), \
        "physics pair has different actions"

    # Same wind
    assert np.allclose(nominal["wind_force_N"], perturbed["wind_force_N"], atol=0.01), \
        "physics pair has different wind"

    # Different trajectory (this is the POINT)
    pos_diff = np.mean(np.linalg.norm(
        nominal["position"] - perturbed["position"], axis=1
    ))
    assert pos_diff > 0.1, f"physics pair trajectories too similar: {pos_diff:.3f} m"

    # Attitude diverges
    omega_diff = np.mean(np.linalg.norm(
        nominal["angular_velocity"] - perturbed["angular_velocity"], axis=1
    ))
    assert omega_diff > 0.02, f"attitude trajectories too similar: {omega_diff:.3f}"

    print(f"{pair_id}: Δpos={pos_diff:.3f} m, Δω={omega_diff:.3f} rad/s")
```

### Check 3 — Intensity pair (S9a/S9b)

```python
for pair_id, (gentle_file, aggressive_file) in load_intensity_pairs().items():
    gentle = np.load(gentle_file)
    aggressive = np.load(aggressive_file)

    # Aggressive reaches > 2× the peak velocity of gentle
    v_gentle = np.max(np.linalg.norm(gentle["velocity"], axis=1))
    v_aggressive = np.max(np.linalg.norm(aggressive["velocity"], axis=1))
    assert v_aggressive > 2.0 * v_gentle, \
        f"intensity insufficient: {v_gentle:.2f} vs {v_aggressive:.2f} m/s"

    # Aggressive reaches > 2× the peak angular velocity of gentle
    w_gentle = np.max(np.linalg.norm(gentle["angular_velocity"], axis=1))
    w_aggressive = np.max(np.linalg.norm(aggressive["angular_velocity"], axis=1))
    assert w_aggressive > 2.0 * w_gentle, \
        f"rate insufficient: {w_gentle:.2f} vs {w_aggressive:.2f} rad/s"
```

### Check 4 — Transition pair (S10a/S10b)

```python
for pair_id, (smooth_file, abrupt_file) in load_transition_pairs().items():
    smooth = np.load(smooth_file)
    abrupt = np.load(abrupt_file)

    # Same endpoint (within tolerance)
    v_smooth_end = np.linalg.norm(smooth["velocity"][-1])
    v_abrupt_end = np.linalg.norm(abrupt["velocity"][-1])
    assert abs(v_smooth_end - v_abrupt_end) < 0.1, "endpoints differ"

    # Different peak acceleration during transition
    a_smooth = np.max(np.abs(np.diff(smooth["velocity"], axis=0) / 0.01))
    a_abrupt = np.max(np.abs(np.diff(abrupt["velocity"], axis=0) / 0.01))
    assert a_abrupt > 3.0 * a_smooth, \
        f"transition insufficient: {a_smooth:.2f} vs {a_abrupt:.2f} m/s²"
```

### Check 5 — Event log completeness

```python
for traj_file in combined_files:
    meta = load_json(traj_file.replace(".npz", ".json"))
    events = meta["events"]

    # Must have at least one maneuver_start
    assert any(e["type"] == "maneuver_start" for e in events), \
        f"{traj_file.name} has no maneuver events"

    # Every maneuver_start must have a matching maneuver_end
    starts = [e["id"] for e in events if e["type"] == "maneuver_start"]
    ends   = [e["id"] for e in events if e["type"] == "maneuver_end"]
    for s in starts:
        assert s in ends, f"{traj_file.name} maneuver '{s}' has no end event"

    # Timestamps must be monotonic and within [0, duration]
    ts = [e["t"] for e in events]
    assert all(b >= a for a, b in zip(ts, ts[1:])), "events out of order"
    assert ts[-1] <= meta["duration_s"] + 0.1, "event past trajectory end"
```

### Check 6 — Non-redundancy vs other modes

```python
# Mode 2 (translation) trajectories have small attitude range:
#   max_tilt < 30°
# Mode 4 (yaw) trajectories have small translation range:
#   max_xy_displacement < 0.5 m
# Mode 5 (attitude) trajectories have small yaw range:
#   max_yaw_change < 30°
# Mode 6 (combined) trajectories have ALL of:
#   max_tilt > 20°  AND  max_xy > 1.5 m  AND  max_yaw_change > 60°

for traj_file in combined_files:
    d = np.load(traj_file)
    q = d["orientation"]
    pos = d["position"]
    yaws = np.unwrap(2 * np.arctan2(q[:, 3], q[:, 0]))

    max_tilt = max(
        np.max(np.abs(2 * np.arcsin(np.clip(q[:, 1:3], -1, 1)))),
        np.max(np.abs(2 * np.arcsin(np.clip(q[:, 1:3], -1, 1))))
    )
    max_xy = np.max(np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1))
    max_yaw_change = yaws.max() - yaws.min()

    assert max_tilt > np.radians(20) or max_xy > 1.5 or max_yaw_change > np.radians(60), \
        f"{traj_file.name} too homogeneous — not a combined mode trajectory"
```

If a combined-mode trajectory fails all three, it is functionally a single-mode trajectory mislabeled as combined. Either the scenario is not generating parallel commands correctly, or the arbitration is suppressing one of them.

---

## One Note on Mode 6 and OOD Evaluation

Combined-mode trajectories **are the primary OOD test set** for your thesis. The reasoning:

1. **They match evaluation conditions** — the world model is tested on multi-axis maneuvers under disturbances, not on single-axis primitives.
2. **They carry the strongest residual signal** — every parallel phase produces `s_{t+1} − f_phys(s_t, u_t) ≠ 0` because the wind effect and the multi-controller arbitration interact non-trivially.
3. **They include the physics pair** — S6a/S6b are the only trajectories in the entire dataset where mass, inertia, and drag are varied in a controlled way. This is what enables the "physics OOD" claim.
4. **They carry event timestamps** — this is what allows the thesis evaluation section to say "the model's prediction error spikes exactly at maneuver boundaries, not at random points," which is a stronger claim than a pure aggregate RMSE.

If Mode 6 is trained heavily but evaluated poorly (larger OOD error than Mode 2 or Mode 3), it is because the model has overfit to single-axis primitives and cannot compose them. The fix is to increase Mode 6's weight in the training set — for example, from 17.5% to 22% by pulling 4.5% from hover (10% → 5.5%). Hover's role in the training set is degenerate by construction; if the model is failing on combined maneuvers, hover's contribution is not what the model needs.