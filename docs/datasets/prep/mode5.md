# Revised Attitude Mode Configuration

The original block had six defects the "what to vary" list exposes:

1. **No sign symmetry** — every step goes to a positive angle. The mirror (roll left vs roll right, pitch forward vs back) is absent.
2. **No amplitude variety** — every segment is a fixed 15–30°. Small (5–10°) and large (30–40°) are absent.
3. **No rate comparison** — every step uses the same implicit rate. Gradual vs abrupt is impossible to learn from.
4. **No initial tilt** — every trajectory starts level. Recovery-from-tilt scenarios are missing.
5. **No initial angular velocity** — ω ≠ 0 at spawn is not exercised.
6. **No combined roll + pitch** — multi-axis tilting is absent, which is the mode's most distinctive feature vs yaw.

The rewrite converts to **13 scenario slots (10 unique templates + 3 pairs)** that cover every "important scenario" plus the additional coverage dimensions from the list. A dedicated note explains the non-redundancy vs yaw mode.

```yaml
# ==========================================================
# MODE 5: MODERATE ATTITUDE (15%)
# ==========================================================
# ROLL + PITCH are the primary objectives. YAW appears only
# as a disturbance to test mode-distinct dynamics.
#
# Non-redundancy with Yaw mode:
#   Yaw mode    → yaw is commanded, roll/pitch held near zero
#   Attitude    → roll/pitch commanded, yaw only perturbed
# These are structurally different command profiles; the model
# cannot fold one into the other.
#
# Every scenario exercises a distinct combination of:
#   - roll target             (positive / negative, paired)
#   - pitch target            (forward / backward, paired)
#   - amplitude               (small / moderate / large)
#   - angular rate            (slow / fast, paired)
#   - angular acceleration    (gradual / abrupt, paired)
#   - maneuver axis           (roll / pitch / combined)
#   - initial attitude        (level / tilted / diagonal)
#   - initial angular velocity (zero / nonzero)
#   - disturbance             (gust / yaw perturbation / initial ω)
#   - hold duration           (short transient vs long hold-and-recover)
#
# Trajectory count: 15 (15% of 100). With guarantee_coverage
# each scenario appears once; three pairs are co-generated.
# ==========================================================
attitude:
  fraction: 0.15

  definition:
    duration_range_s: [10.0, 18.0]
    altitudes_m: [1.0, 1.5, 2.0]

    # ── Initial conditions ─────────────────────────────────
    initial_conditions:
      position_xy_range_m: [-2.0, 2.0]
      position_z_offset_m: [-0.1, 0.1]
      velocity_magnitude_range_mps: [0.0, 0.0]
      velocity_direction: random_unit_vector
      # Default level; scenarios override
      roll_offset_deg: 0.0
      pitch_offset_deg: 0.0
      yaw_range_rad: [-3.14159, 3.14159]
      angular_velocity_magnitude_range_radps: [0.0, 0.0]
      angular_velocity_axis: random_unit_vector

    # ── Amplitude classes (sampled magnitudes in degrees) ──
    attitude_amplitudes_deg:
      small:    [5.0, 10.0]
      moderate: [15.0, 25.0]
      large:    [30.0, 40.0]

    # ── Rate profiles ──────────────────────────────────────
    # Time to reach a commanded target from the current attitude.
    # `abrupt` is a near-step input; `gradual` ramps slowly so the
    # drone tracks a moving attitude target.
    rate_profiles:
      gradual:  3.0
      moderate: 1.0
      abrupt:   0.05

    # ── Wind direction vocabulary ──────────────────────────
    directions:
      north: [ 1.0,  0.0, 0.0]
      south: [-1.0,  0.0, 0.0]
      east:  [ 0.0,  1.0, 0.0]
      west:  [ 0.0, -1.0, 0.0]
      ne:    [ 0.7071,  0.7071, 0.0]
      nw:    [-0.7071,  0.7071, 0.0]
      up:    [ 0.0, 0.0,  1.0]
      down:  [ 0.0, 0.0, -1.0]

    # ── Scenario templates ─────────────────────────────────
    scenarios:

      # ─────────────────────────────────────────────────
      # S1a: Roll in one direction (user list #1, member 1/2)
      # ─────────────────────────────────────────────────
      - name: roll_positive
        weight: 0.05
        pair_group: roll_pair
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: attitude_step
              roll_target_deg: [20.0, 30.0]
              pitch_target_deg: 0.0
              hold_duration_s: [3.0, 8.0]
              return_to_level: true
              rate_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S1b: Roll in the opposite direction (member 2/2)
      # Same |target| and wind as S1a → sign-symmetry pair.
      # ─────────────────────────────────────────────────
      - name: roll_negative
        weight: 0.05
        pair_group: roll_pair
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: attitude_step
              roll_target_deg: [-30.0, -20.0]
              pitch_target_deg: 0.0
              hold_duration_s: [3.0, 8.0]
              return_to_level: true
              rate_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]     # matched with S1a
              direction_from_list: [north, east]  # matched

      # ─────────────────────────────────────────────────
      # S2a: Pitch forward (nose down) (user list #2, member 1/2)
      # ─────────────────────────────────────────────────
      - name: pitch_forward
        weight: 0.05
        pair_group: pitch_pair
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: attitude_step
              roll_target_deg: 0.0
              pitch_target_deg: [15.0, 25.0]
              hold_duration_s: [3.0, 8.0]
              return_to_level: true
              rate_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S2b: Pitch backward (nose up) (member 2/2)
      # ─────────────────────────────────────────────────
      - name: pitch_backward
        weight: 0.05
        pair_group: pitch_pair
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: attitude_step
              roll_target_deg: 0.0
              pitch_target_deg: [-25.0, -15.0]
              hold_duration_s: [3.0, 8.0]
              return_to_level: true
              rate_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]     # matched with S2a
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S3: Amplitude sweep (user list #3)
      # Samples small/moderate/large amplitude across trajectories
      # so the model sees the full range, not just one target.
      # ─────────────────────────────────────────────────
      - name: amplitude_sweep
        weight: 0.12
        phases:
          - duration_s: [10.0, 16.0]
            controller:
              type: attitude_step
              axis_from_list: [roll, pitch]
              amplitude_class_from_list: [small, moderate, large]
              sign: random
              hold_duration_s: [2.0, 5.0]
              return_to_level: true
              rate_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S4: Rapid tilt + recovery (user list #4)
      # Short hold (0.3–1 s) so the trajectory is dominated by
      # the up-and-back transients — very different from the
      # long-hold trajectories above.
      # ─────────────────────────────────────────────────
      - name: rapid_tilt_recovery
        weight: 0.10
        phases:
          - duration_s: [8.0, 12.0]
            controller:
              type: attitude_step
              axis_from_list: [roll, pitch]
              amplitude_class: moderate
              sign: random
              hold_duration_s: [0.3, 1.0]     # ★ short
              return_to_level: true
              rate_profile: abrupt           # ★ snappy
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S5a: Gradual ramp to target (paired with S5b)
      # Same target and endpoint as S5b; only the ramp duration
      # differs. Isolates angular-acceleration effect.
      # ─────────────────────────────────────────────────
      - name: gradual_rate_to_target
        weight: 0.05
        pair_group: rate_comparison
        phases:
          - duration_s: [14.0, 18.0]
            controller:
              type: attitude_ramp
              axis: roll                      # matched with S5b
              target_deg: [20.0, 30.0]        # matched with S5b
              ramp_duration_s: [2.5, 3.5]     # ★ slow
              hold_duration_s: [3.0, 5.0]
              return_to_level: true
            wind:
              type: constant
              magnitude_N: [2.0, 3.5]         # matched with S5b
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S5b: Abrupt ramp to same target (paired with S5a)
      # ─────────────────────────────────────────────────
      - name: abrupt_rate_to_target
        weight: 0.05
        pair_group: rate_comparison
        phases:
          - duration_s: [14.0, 18.0]
            controller:
              type: attitude_ramp
              axis: roll                      # matched with S5a
              target_deg: [20.0, 30.0]        # matched with S5a
              ramp_duration_s: [0.05, 0.15]   # ★ near-instant
              hold_duration_s: [3.0, 5.0]
              return_to_level: true
            wind:
              type: constant
              magnitude_N: [2.0, 3.5]         # matched with S5a
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S6: Combined roll + pitch (user list #5)
      # Simultaneous two-axis tilt. The drone's thrust vector
      # points along a diagonal; wind torque couples both axes.
      # ─────────────────────────────────────────────────
      - name: combined_roll_pitch
        weight: 0.12
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: attitude_combined
              roll_target_deg: [15.0, 25.0]
              pitch_target_deg: [15.0, 25.0]
              roll_sign: random
              pitch_sign: random
              hold_duration_s: [3.0, 6.0]
              return_to_level: true
              rate_profile: moderate
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east, ne]

      # ─────────────────────────────────────────────────
      # S7: Gust under attitude hold (user list #6)
      # Long hold (8–12 s) so the gust strikes while the drone
      # is holding a tilted attitude. Controller must reject the
      # disturbance without abandoning the commanded attitude.
      # ─────────────────────────────────────────────────
      - name: gust_under_attitude_hold
        weight: 0.10
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: attitude_step
              axis_from_list: [roll, pitch]
              amplitude_class: moderate
              sign: random
              hold_duration_s: [8.0, 12.0]    # ★ long
              return_to_level: true
              rate_profile: moderate
            wind:
              type: gust_mid
              trigger_at_progress: [0.40, 0.70]  # during the hold
              peak_magnitude_N: [4.0, 7.0]
              duration_s: [0.3, 0.8]
              direction_from_list: [east, west, north, south]

      # ─────────────────────────────────────────────────
      # S8: Initial tilt recovery
      # Starts already tilted. Controller must recover to level
      # from a nonzero initial attitude error (unlike S1/S2,
      # which ramp from level).
      # ─────────────────────────────────────────────────
      - name: initial_tilt_recovery
        weight: 0.10
        initial_conditions:
          tilt_mode_from_list: [roll_only, pitch_only, diagonal]
          tilt_magnitude_deg: [10.0, 25.0]
        phases:
          - duration_s: [10.0, 14.0]
            controller:
              type: hover
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S9: Nonzero initial angular velocity
      # Level attitude, but spinning. Controller must arrest the
      # rotation and stabilise — no attitude command is issued.
      # ─────────────────────────────────────────────────
      - name: initial_angular_velocity
        weight: 0.08
        initial_conditions:
          angular_velocity_magnitude_range_radps: [0.5, 1.5]
          angular_velocity_axis: random_unit_vector
        phases:
          - duration_s: [8.0, 12.0]
            controller:
              type: hover
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S10: Yaw disturbance during attitude hold
      # ★ Distinct from Yaw mode: here YAW is a disturbance.
      # The objective is to hold roll while rejecting an injected
      # yaw rate. In yaw mode, yaw IS the objective and roll is
      # held at zero.
      # ─────────────────────────────────────────────────
      - name: yaw_disturbance_during_attitude_hold
        weight: 0.08
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: attitude_step
              axis: roll
              target_deg: [15.0, 25.0]
              hold_duration_s: [6.0, 10.0]
              return_to_level: true
              rate_profile: moderate
              # Injects a yaw-rate command perturbation mid-hold
              yaw_perturbation_rate_radps: [0.3, 0.8]
              yaw_perturbation_time_s: [3.0, 5.0]
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

    # ── Paired scenarios ──────────────────────────────────
    paired_scenarios:
      - name: roll_pair
        members: [roll_positive, roll_negative]
        shared:
          - abs_roll_target_deg
          - wind_magnitude_N
          - wind_direction
          - hold_duration_s
          - total_duration_s
        pair_id_prefix: "roll_sign"

      - name: pitch_pair
        members: [pitch_forward, pitch_backward]
        shared:
          - abs_pitch_target_deg
          - wind_magnitude_N
          - wind_direction
          - hold_duration_s
          - total_duration_s
        pair_id_prefix: "pitch_sign"

      - name: rate_comparison
        members: [gradual_rate_to_target, abrupt_rate_to_target]
        shared:
          - axis
          - target_deg
          - hold_duration_s
          - wind_magnitude_N
          - wind_direction
          - total_duration_s
        pair_id_prefix: "attitude_rate_profile"

    # ── Sampling policy ───────────────────────────────────
    sampling:
      guarantee_coverage: true
      pairs_generated_together: true

    # ── Expected composition ──────────────────────────────
    report:
      trajectory_count: 15
      scenarios_present: 13
      avg_duration_s: 13.0
      total_steps: ~19500                 # 13 s × 100 Hz × 15
      paired_pairs_generated: 3

      coverage:
        roll_left_right:            S1a + S1b (paired)
        pitch_fwd_back:             S2a + S2b (paired)
        attitude_step_variety:      S3
        rapid_tilt_recovery:        S4
        gradual_vs_abrupt:          S5a + S5b (paired)
        combined_roll_pitch:        S6
        gust_under_hold:            S7
        initial_tilt:               S8
        initial_angular_velocity:   S9
        yaw_during_attitude:        S10 (yaw as disturbance, not objective)

  counterfactual: true
```

---

## What Changed vs. the Original

| Defect in original | How the rewrite fixes it |
|---|---|
| Only positive-angle steps | S1a/S1b pair covers ±roll; S2a/S2b pair covers ±pitch |
| Fixed 15–30° amplitude | S3 samples from `small / moderate / large` = 5–40° |
| No rate comparison | S5a/S5b paired with `ramp_duration_s` in `[2.5, 3.5]` vs `[0.05, 0.15]` |
| No initial tilt | S8 uses `tilt_mode_from_list` (roll_only, pitch_only, diagonal) with magnitude 10–25° |
| No nonzero initial ω | S9 sets `angular_velocity_magnitude_range_radps: [0.5, 1.5]` |
| No combined roll+pitch | S6 uses `attitude_combined` with sampled magnitudes and signs |
| Gust at fixed time | S7 uses `gust_mid` with `trigger_at_progress: [0.4, 0.7]` (relative to hold window) |
| No yaw distinct from yaw mode | S10 injects yaw perturbation mid-hold; yaw is the disturbance, not the objective |
| Every trajectory nearly identical | 13 slots across distinct (axis, amplitude, rate, initial condition, wind) combinations |

---

## Distinguishing This Mode from Yaw Mode

The user explicitly asks for non-redundancy. The command profiles enforce this:

| | Yaw mode | Attitude mode |
|---|---|---|
| **Primary command** | `yaw_rate` (or `yaw_angle`) | `roll_target`, `pitch_target` |
| **Held at zero** | roll, pitch | yaw |
| **Yaw role** | objective | disturbance (S10) or perturbation (S8 diagonal) |
| **Position** | held (hover) | held (hover) |
| **Typical state change** | ψ rotates 90–180° | φ, θ tilt 20–40° and recover |
| **Wind coupling** | RᵀF_w as drone rotates | τ_w = r×F_w as drone tilts |

The datasets do not overlap because:
- Yaw trajectories have near-zero roll/pitch but full ±π range of ψ.
- Attitude trajectories have near-zero ψ change but full ±40° range of φ/θ.
- The only overlap is S10 (yaw disturbance) — which is intentionally non-redundant because the objective there is roll/pitch, not heading.

If after training the model still confuses the two modes (measured by cross-mode prediction error in validation), the fix is to increase S10's fraction from 0.08 to 0.12, which gives the model more negative examples of "yaw is not the target."

---

## New Controller Types Needed

### `attitude_step`

```yaml
controller:
  type: attitude_step
  roll_target_deg: float | [min, max]
  pitch_target_deg: float | [min, max]
  hold_duration_s: [min, max]
  return_to_level: true
  rate_profile: gradual | moderate | abrupt
```

**Behaviour:**
1. At `t = 0`, the attitude target is set to `(roll_target, pitch_target)`.
2. The internal P controller (`kp_attitude` from `rate_controller_gains`) computes `target_rate` from the attitude error, clamped to `max_rate_radps`.
3. After `hold_duration_s`, if `return_to_level`, the target returns to `(0, 0)` and the controller settles.
4. `rate_profile` sets the effective ramp duration for a gradual target (with `abrupt`, the target changes in one simulation step).

### `attitude_ramp`

Same as `attitude_step` but with an explicit `ramp_duration_s` that overrides `rate_profile`. Used for the paired gradual/abrupt comparison.

### `attitude_combined`

```yaml
controller:
  type: attitude_combined
  roll_target_deg: [min, max]
  pitch_target_deg: [min, max]
  roll_sign: random | positive | negative
  pitch_sign: random | positive | negative
  hold_duration_s: [min, max]
  return_to_level: true
  rate_profile: gradual | moderate | abrupt
```

**Behaviour:** sets both roll and pitch targets simultaneously. The P controller produces two rate commands, which the `RateController` combines into motor thrusts.

### `yaw_perturbation` in `attitude_step`

```yaml
controller:
  type: attitude_step
  ...
  yaw_perturbation_rate_radps: [min, max]   # optional
  yaw_perturbation_time_s: [min, max]       # optional
```

**Behaviour:** during the hold phase, at `yaw_perturbation_time_s`, injects a yaw-rate command for a fixed duration (e.g., 0.5 s). The `RateController` receives the yaw command but the outer loop continues to track roll. This tests whether the model learns cross-axis coupling under disturbance.

### `initial_conditions.tilt_mode_from_list`

Resolves per trajectory to one of:

| Mode | `roll_offset_deg` | `pitch_offset_deg` |
|---|---|---|
| `roll_only` | `±mag` | `0` |
| `pitch_only` | `0` | `±mag` |
| `diagonal` | `±mag` | `±mag` |

with `mag` sampled from `tilt_magnitude_deg` and signs sampled uniformly.

---

## Required Addition to `rate_controller_gains`

The base config (from earlier) needs two new gains:

```yaml
rate_controller_gains:
  # ... existing fields ...
  kp_attitude: 5.0    # [rad/s per rad]  outer attitude P loop
  kd_attitude: 0.5    # [rad/s per rad/s]  optional derivative
```

If `kp_attitude` is too small (< 2), the attitude reaches the target too slowly. If too large (> 10), the controller saturates the rate limit on every step-change and the drone overshoots. `5.0` gives a reasonable default that reaches a 25° target in ~0.5 s without overshoot.

---

## Verification Checks Specific to Mode 5

### Check 1 — Sign-symmetry pairs (S1a/S1b, S2a/S2b)

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_roll_pitch(q):
    """q = (w, x, y, z) → (roll, pitch) in radians"""
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    e = r.as_euler('xyz')
    return e[0], e[1]

for pair_id, (pos_file, neg_file) in load_attitude_sign_pairs().items():
    pos = np.load(pos_file)
    neg = np.load(neg_file)

    roll_p, pitch_p = zip(*[quat_to_roll_pitch(q) for q in pos["orientation"]])
    roll_n, pitch_n = zip(*[quat_to_roll_pitch(q) for q in neg["orientation"]])

    # Peak amplitude should match in magnitude, opposite in sign
    roll_p_peak = roll_p[np.argmax(np.abs(roll_p))]
    roll_n_peak = roll_n[np.argmax(np.abs(roll_n))]
    assert np.sign(roll_p_peak) == -np.sign(roll_n_peak), "roll signs not opposite"
    assert abs(abs(roll_p_peak) - abs(roll_n_peak)) / abs(roll_p_peak) < 0.15, "magnitudes differ"

    # Wind identical
    assert np.allclose(pos["wind_force_N"], neg["wind_force_N"], atol=0.01), "wind differs"
    print(f"{pair_id}: roll_p={np.degrees(roll_p_peak):.1f}°, roll_n={np.degrees(roll_n_peak):.1f}°")
```

### Check 2 — Rate-comparison pair (S5a/S5b)

```python
for pair_id, (gradual_file, abrupt_file) in load_attitude_rate_pairs().items():
    g = np.load(gradual_file)
    a = np.load(abrupt_file)

    omega_g = np.max(np.abs(g["angular_velocity"][:, 0]))   # roll rate peak
    omega_a = np.max(np.abs(a["angular_velocity"][:, 0]))

    # Abrupt should reach 2× the peak rate
    assert omega_a > 2 * omega_g, f"Not distinguishable: {omega_g:.2f} vs {omega_a:.2f}"

    # Same endpoint
    roll_g_end = quat_to_roll_pitch(g["orientation"][-1])[0]
    roll_a_end = quat_to_roll_pitch(a["orientation"][-1])[0]
    assert abs(roll_g_end - roll_a_end) < np.radians(5), "endpoints differ"

    print(f"{pair_id}: ω_gradual={omega_g:.2f}, ω_abrupt={omega_a:.2f} rad/s")
```

### Check 3 — Initial tilt recovery (S8)

```python
for traj_file in initial_tilt_files:
    d = np.load(traj_file)
    rolls, pitches = zip(*[quat_to_roll_pitch(q) for q in d["orientation"]])

    # Attitude error should monotonically decrease over first 1 second
    err = np.sqrt(np.array(rolls)**2 + np.array(pitches)**2)
    err_first_1s = err[:100]              # 100 Hz × 1 s
    err_late     = err[200:300]           # 2–3 s window

    assert err_first_1s[0] > 0.15, "trajectory didn't start tilted"
    assert err_late.mean() < 0.05, "controller didn't recover"
    print(f"{traj_file.name}: err_init={np.degrees(err_first_1s[0]):.1f}°, "
          f"err_late={np.degrees(err_late.mean()):.1f}°")
```

### Check 4 — Gust response transient (S7)

```python
for traj_file in gust_hold_files:
    d = np.load(traj_file)
    rolls, pitches = zip(*[quat_to_roll_pitch(q) for q in d["orientation"]])
    F = np.linalg.norm(d["wind_force_N"], axis=1)

    gust_start = np.argmax(F > 4.0)
    roll_gust = rolls[gust_start:gust_start + 200]   # 2 s window after gust

    # Should show a transient deviation then return to held attitude
    roll_before = rolls[max(0, gust_start - 100):gust_start]
    excursion = np.max(np.abs(np.array(roll_gust) - np.mean(roll_before)))
    assert excursion > 0.03, "no visible attitude excursion during gust"
    assert np.abs(roll_gust[-1] - np.mean(roll_before)) < 0.05, "didn't recover from gust"

    print(f"{traj_file.name}: excursion={np.degrees(excursion):.1f}°, recovery_ok")
```

### Check 5 — Mode-distinct from yaw mode

The most important check for non-redundancy — the attitude and yaw datasets should have orthogonal roll/pitch vs yaw distributions:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def yaw_range(files):
    ranges = []
    for f in files:
        d = np.load(f)
        yaws = [R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')[2]
                for q in d["orientation"]]
        yaws = np.unwrap(yaws)
        ranges.append(yaws.max() - yaws.min())
    return np.mean(ranges), np.max(ranges)

def roll_pitch_range(files):
    r_range, p_range = [], []
    for f in files:
        d = np.load(f)
        rps = [R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz') for q in d["orientation"]]
        rolls, pitches = zip(*[(e[0], e[1]) for e in rps])
        r_range.append(max(rolls) - min(rolls))
        p_range.append(max(pitches) - min(pitches))
    return np.mean(r_range), np.mean(p_range)

yaw_mean, yaw_max = yaw_range(yaw_files)
roll_mean, pitch_mean = roll_pitch_range(attitude_files)

print(f"Yaw mode:      Δψ mean = {np.degrees(yaw_mean):.1f}°, max = {np.degrees(yaw_max):.1f}°")
print(f"Attitude mode: Δφ mean = {np.degrees(roll_mean):.1f}°, Δθ mean = {np.degrees(pitch_mean):.1f}°")

# Non-redundancy assertion:
# Yaw mode should have LARGE Δψ but SMALL Δφ, Δθ
# Attitude mode should have LARGE Δφ, Δθ but SMALL Δψ (except S10)
assert np.degrees(yaw_mean) > 90,  "Yaw mode Δψ too small"
assert np.degrees(roll_mean) > 30, "Attitude mode Δφ too small"
assert np.degrees(pitch_mean) > 20, "Attitude mode Δθ too small"
```

If `yaw_mode Δψ > 90°` but `attitude_mode Δψ > 45°` for more than 20% of attitude trajectories, the two modes are leaking into each other — check that S10's `yaw_perturbation_rate_radps` is set to a *bounded* value and that the perturbation is short (single 0.5 s burst, not continuous).

---

## One Note on Mode 5 and OOD Evaluation

Attitude-mode trajectories **are** eligible for OOD test sets. The reasons:

- Attitude is 7 of 13 state dimensions (quaternion + angular velocity). Under OOD wind, the attitude prediction is where the \(d_{SO(3)}\) metric fails first.
- The `attitude_step` controller produces clean, high-amplitude attitude commands that stress-test the model's ability to predict quaternion evolution under wind torque.
- The sign-symmetry pairs (S1, S2) and rate-comparison pair (S5) give the model causally structured data — it can *infer* the sign and rate effects from the trajectory, rather than memorising them.

Mode 5's 15% share reflects this importance. Combined with Mode 4 (yaw, 12.5%), attitude/yaw data constitutes 27.5% of the training set — equal to translation mode's share — which is the correct balance for a state-space world model that must predict all 13 dimensions under OOD disturbances.