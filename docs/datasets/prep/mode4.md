# Revised Yaw Mode Configuration

The original block had six defects the "what to vary" list exposes:

1. **No CW/CCW pairing** — every yaw step targets a *positive* angle (`+π/2`, `+π`). The user explicitly asks to include both directions and the residual must learn the sign symmetry.
2. **No initial-heading variation** — every trajectory starts at `yaw = 0`. Cross-quadrant data is missing.
3. **No initial yaw rate** — the "nonzero initial yaw rate" scenario is absent.
4. **No quaternion boundary crossing** — no trajectory crosses ±180°, which the user flags as a quaternion-consistency concern.
5. **Fixed rate and profile** — every sinusoid is 0.5 rad/s at 0.2 Hz. Slow/moderate/fast and step/ramp/sinusoidal distinctions are absent.
6. **Fixed duration** — all trajectories are 100 s with the same phases. Short rapid-transient scenarios (the "settling" case) are not exercised.

The rewrite converts to **12 scenario templates** (10 unique + 2 pairs) that cover every "important scenario" from the list plus a dedicated **quaternion-boundary-crossing** template.

```yaml
# ==========================================================
# MODE 4: YAW MOTION (12.5%)
# ==========================================================
# Forces the model to learn body-frame Rᵀ F_w transformation
# as the drone rotates.
#
# Every scenario exercises a distinct combination of:
#   - yaw angle        (small, medium, large, near-full)
#   - yaw direction    (CW and CCW, paired)
#   - yaw rate         (slow / moderate / fast)
#   - yaw acceleration (gradual ramp vs abrupt step)
#   - initial heading  (four quadrants + near-π boundary)
#   - initial ωz       (zero vs non-zero)
#   - flight condition (hovering vs translating)
#   - wind             (calm, crosswind, gust)
#   - yaw profile      (step, ramp, sinusoidal)
#
# Trajectory count: 12–13 (12.5% of 100). With guarantee_coverage
# every scenario appears once; two of them (heading_180 rate pair
# and CW/CCW pair) are co-generated as matched pairs.
# ==========================================================
yaw:
  fraction: 0.125

  definition:
    duration_range_s: [10.0, 22.0]
    altitudes_m: [1.0, 1.5, 2.0]

    # ── Initial conditions ─────────────────────────────────
    initial_conditions:
      position_xy_range_m: [-2.0, 2.0]
      position_z_offset_m: [-0.1, 0.1]
      velocity_magnitude_range_mps: [0.0, 0.0]
      velocity_direction: random_unit_vector
      roll_offset_deg: 0.0
      pitch_offset_deg: 0.0
      # Full yaw coverage by default; scenarios can pin a range
      yaw_range_rad: [-3.14159, 3.14159]
      # Initial angular velocity — zero by default, overridden in S6
      angular_velocity_magnitude_range_radps: [0.0, 0.0]
      angular_velocity_axis: [0, 0, 1]      # pure yaw when nonzero

    # ── Rate and profile vocabularies ─────────────────────
    yaw_rates_radps:
      slow:     [0.15, 0.50]
      moderate: [0.50, 1.20]
      fast:     [1.20, 2.50]

    yaw_profiles:
      step:       abrupt target change (single sample)
      ramp:       linear approach to target over ramp_duration
      sinusoidal: continuous sinusoidal command

    # ── Wind direction vocabulary ─────────────────────────
    directions:
      north:   [ 1.0,  0.0, 0.0]
      south:   [-1.0,  0.0, 0.0]
      east:    [ 0.0,  1.0, 0.0]
      west:    [ 0.0, -1.0, 0.0]
      ne:      [ 0.7071,  0.7071, 0.0]
      nw:      [-0.7071,  0.7071, 0.0]
      up:      [ 0.0, 0.0,  1.0]
      down:    [ 0.0, 0.0, -1.0]

    # ── Scenario templates ─────────────────────────────────
    scenarios:

      # ─────────────────────────────────────────────────
      # S1: 90° heading change while hovering (user list #1)
      # ─────────────────────────────────────────────────
      - name: heading_change_90_hover
        weight: 0.10
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: yaw_step
              target_change_rad: [1.5708]       # exactly +90°
              rate_class: moderate
              transition_profile: ramp
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S2a: 180° turn — SLOW (user list #2, member 1/2)
      # Shared target and wind with S2b; differs only in rate.
      # ─────────────────────────────────────────────────
      - name: heading_change_180_slow
        weight: 0.10
        pair_group: heading_180_rate_comparison
        phases:
          - duration_s: [15.0, 20.0]
            controller:
              type: yaw_step
              target_change_rad: [3.1416]       # 180° (matched with S2b)
              rate_class: slow                  # ★ differs
              transition_profile: ramp
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]           # matched with S2b
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S2b: 180° turn — FAST (user list #2, member 2/2)
      # Same endpoint and wind as S2a → different intermediate
      # dynamics. Pairs give controlled comparison.
      # ─────────────────────────────────────────────────
      - name: heading_change_180_fast
        weight: 0.10
        pair_group: heading_180_rate_comparison
        phases:
          - duration_s: [15.0, 20.0]
            controller:
              type: yaw_step
              target_change_rad: [3.1416]       # 180° (matched with S2a)
              rate_class: fast                  # ★ differs
              transition_profile: ramp
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]           # matched with S2a
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S3: Yaw rotation while translating forward (user list #3)
      # Couples yaw with forward velocity — the drone traces a
      # spiral or arc while its heading rotates.
      # ─────────────────────────────────────────────────
      - name: yaw_while_translating
        weight: 0.12
        phases:
          - duration_s: [15.0, 22.0]
            controller:
              type: yaw_while_translating
              forward_velocity_range_mps: [0.5, 1.2]
              yaw_rate_class: moderate
              yaw_total_rad: [1.5708, 3.1416]   # 90° or 180°
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction_from_list: [north, east, ne]

      # ─────────────────────────────────────────────────
      # S4: Rapid yaw command followed by settling (user list #4)
      # Abrupt step profile → dominant transient.
      # ─────────────────────────────────────────────────
      - name: rapid_yaw_settling
        weight: 0.08
        phases:
          - duration_s: [10.0, 14.0]
            controller:
              type: yaw_step
              target_change_rad: [0.7854, 1.5708]  # 45° or 90°
              rate_class: fast
              transition_profile: step          # ★ abrupt
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S5: Yaw rotation under a lateral gust (user list #5)
      # Gust strikes at a random point in the maneuver.
      # ─────────────────────────────────────────────────
      - name: yaw_under_gust
        weight: 0.08
        phases:
          - duration_s: [12.0, 18.0]
            controller:
              type: yaw_ramp
              target_change_rad: [1.5708]
              rate_class: moderate
            wind:
              type: gust_mid
              trigger_at_progress: [0.30, 0.70]
              peak_magnitude_N: [4.0, 7.0]
              duration_s: [0.3, 0.8]
              direction_from_list: [east, west, south, north]

      # ─────────────────────────────────────────────────
      # S6: Nonzero initial yaw rate (user list #6)
      # ω_z ≠ 0 at spawn; controller stabilises AND executes yaw.
      # ─────────────────────────────────────────────────
      - name: initial_yaw_rate
        weight: 0.08
        initial_conditions:
          angular_velocity_magnitude_range_radps: [0.3, 1.2]
          angular_velocity_axis: [0, 0, 1]        # pure yaw
        phases:
          - duration_s: [10.0, 15.0]
            controller:
              type: yaw_step
              target_change_rad: [1.5708, 3.1416]
              rate_class: moderate
              transition_profile: ramp
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S7a: CW rotation 90° (user list: both directions, member 1/2)
      # ─────────────────────────────────────────────────
      - name: yaw_cw_90
        weight: 0.05
        pair_group: yaw_cw_ccw_90
        phases:
          - duration_s: [10.0, 14.0]
            controller:
              type: yaw_step
              target_change_rad: [-1.5708]      # −90° = clockwise
              rate_class: moderate              # matched with S7b
              transition_profile: ramp
            wind:
              type: constant
              magnitude_N: [2.0, 3.5]           # matched with S7b
              direction: [0, 1, 0]              # matched with S7b

      # ─────────────────────────────────────────────────
      # S7b: CCW rotation 90° — exact mirror of S7a (member 2/2)
      # ─────────────────────────────────────────────────
      - name: yaw_ccw_90
        weight: 0.05
        pair_group: yaw_cw_ccw_90
        phases:
          - duration_s: [10.0, 14.0]
            controller:
              type: yaw_step
              target_change_rad: [1.5708]       # +90° = counterclockwise
              rate_class: moderate              # matched with S7a
              transition_profile: ramp
            wind:
              type: constant
              magnitude_N: [2.0, 3.5]           # matched with S7a
              direction: [0, 1, 0]              # matched with S7a

      # ─────────────────────────────────────────────────
      # S8: Quaternion boundary crossing at ±180°
      # Start near +π and command a positive change → crosses
      # +π into -π region. Tests quaternion sign consistency.
      # ─────────────────────────────────────────────────
      - name: boundary_crossing_pi
        weight: 0.08
        initial_conditions:
          yaw_range_rad: [2.80, 3.10]           # +160° to +178°
        phases:
          - duration_s: [10.0, 15.0]
            controller:
              type: yaw_step
              target_change_rad: [0.50, 1.00]   # forces crossing +π
              rate_class: moderate
              transition_profile: ramp
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S9: Initial-heading variety across four quadrants
      # Ensures the model sees yaw at +90, +180, -90, 0 uniformly.
      # ─────────────────────────────────────────────────
      - name: initial_heading_variety
        weight: 0.08
        initial_conditions:
          yaw_range_rad: [-3.14159, 3.14159]    # full circle
        phases:
          - duration_s: [10.0, 15.0]
            controller:
              type: yaw_sine
              amplitude_radps: [0.30, 0.80]
              frequency_hz: [0.15, 0.35]
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S10: Sinusoidal yaw sweep — broad continuous coverage
      # Integrates to ±90–180° excursions, exercises many
      # yaw-rate reversal points.
      # ─────────────────────────────────────────────────
      - name: sinusoidal_sweep
        weight: 0.08
        phases:
          - duration_s: [15.0, 22.0]
            controller:
              type: yaw_sine
              amplitude_radps: [0.40, 1.00]
              frequency_hz: [0.10, 0.25]
            wind:
              type: constant
              magnitude_N: [1.5, 3.0]
              direction_from_list: [north, east, up]   # up exercise vertical component

    # ── Paired scenarios ──────────────────────────────────
    # Two kinds of pairs:
    # 1. Rate comparison (S2a/S2b): same endpoint, different rate
    # 2. Direction mirror (S7a/S7b): same |change|, opposite sign
    paired_scenarios:
      - name: heading_180_rate_comparison
        members: [heading_change_180_slow, heading_change_180_fast]
        shared:
          - target_change_rad
          - wind_magnitude_N
          - wind_direction
          - total_duration_s
        pair_id_prefix: "heading_180_rate"

      - name: yaw_cw_ccw_90
        members: [yaw_cw_90, yaw_ccw_90]
        shared:
          - abs_target_change_rad
          - rate_class
          - wind_magnitude_N
          - wind_direction
          - total_duration_s
        pair_id_prefix: "yaw_cw_ccw"

    # ── Sampling policy ───────────────────────────────────
    sampling:
      guarantee_coverage: true                # every scenario appears ≥1×
      pairs_generated_together: true          # pairs co-generated

    # ── Expected composition ──────────────────────────────
    report:
      trajectory_count: 12
      scenarios_present: 12
      avg_duration_s: 14.0
      total_steps: ~16800                     # 14 s × 100 Hz × 12
      paired_pairs_generated: 2

      coverage:
        heading_90_hover:           S1
        180_slow_vs_fast:           S2a + S2b (paired, rate comparison)
        yaw_while_translating:      S3
        rapid_yaw_with_settling:    S4
        yaw_under_gust:             S5
        nonzero_initial_yaw_rate:   S6
        cw_vs_ccw:                  S7a + S7b (paired, direction mirror)
        quaternion_boundary:        S8
        initial_heading_variety:    S9
        sinusoidal_sweep:           S10

  counterfactual: true
```

---

## What Changed vs. the Original

| Defect in original | How the rewrite fixes it |
|---|---|
| All yaw steps target positive angles | S7a/S7b provide an explicit ±90° pair; the residual learns sign symmetry |
| No initial-heading variation | S8 pins `yaw_range_rad: [2.80, 3.10]`; S9 samples full circle; mode-level default covers `[-π, π]` |
| No nonzero initial ω_z | S6 sets `angular_velocity_magnitude_range_radps: [0.3, 1.2]` |
| No quaternion boundary crossing | S8's initial heading (+160° to +178°) plus positive target change forces crossing +π |
| Fixed 0.5 rad/s sinusoid | `rate_class` (slow/moderate/fast) + `transition_profile` (step/ramp/sinusoidal) in every scenario |
| Fixed 100 s duration | `duration_range_s: [10, 22]`; scenario-level `duration_s` also a range |
| Only constant + step wind | S5 uses `gust_mid`, S10 exercises vertical wind direction (`up`) |
| No rate comparison | S2a/S2b pair with shared endpoint and wind, different rate |
| No direction pairing | S7a/S7b pair with shared magnitude and wind, opposite sign |
| Every trajectory nearly identical | 10 unique scenario templates, each with different (rate, profile, wind, initial condition) |

---

## New Controller and Wind Types Needed

### New `yaw_step` parameters

```yaml
controller:
  type: yaw_step
  target_change_rad: [min, max]   # Δψ from current heading (signed)
  rate_class: slow|moderate|fast  # resolves to a rad/s via yaw_rates_radps
  transition_profile: step|ramp   # step = abrupt; ramp = linear to target
```

**Behaviour:**
- `step`: the target yaw jumps at t = 0; the RateController produces its natural transient, and the drone settles into the target over ~3–5 controller time constants.
- `ramp`: the target yaw advances linearly at the sampled rate until it reaches `target_change_rad + initial_yaw`.

### New `yaw_ramp` controller

Same as `yaw_step` with `transition_profile: ramp`, provided as a distinct type for clarity when the scenario name is `yaw_ramp`.

### New `yaw_while_translating` controller

```yaml
controller:
  type: yaw_while_translating
  forward_velocity_range_mps: [min, max]
  yaw_rate_class: slow|moderate|fast
  yaw_total_rad: [min, max]   # magnitude of total change
```

**Behaviour:** simultaneously commands forward velocity along the *current* heading (so the drone arcs) and yaw rate. Integrates both to produce a spiral path.

### New `gust_mid` wind type (already introduced in Mode 3)

```yaml
wind:
  type: gust_mid
  trigger_at_progress: [low, high]   # fraction of trajectory duration
  peak_magnitude_N: [min, max]
  duration_s: [min, max]
  direction_from_list: [name1, name2, ...]   # NEW: also supports random direction
```

The extension here is that `direction_from_list` is also supported for `gust_mid` (in Mode 3 it was only `direction: [x,y,z]`).

### New `yaw_sine` parameters

```yaml
controller:
  type: yaw_sine
  amplitude_radps: [min, max]     # peak yaw rate (previously fixed at 0.5)
  frequency_hz: [min, max]
```

The `amplitude_radps` field is now a range; Mode 4 samples it per trajectory.

---

## Quaternion Boundary Consistency

The user specifically flags this: **"If your state representation uses quaternions, also make sure heading changes crossing the ±180° boundary are represented consistently."**

There are two independent problems:

### Problem A — The recorded quaternion may jump between q and −q

Pure yaw quaternion: \( q = [\cos(\psi/2), 0, 0, \sin(\psi/2)] \).

- At \(\psi = 179°\): \(q \approx [0.0087, 0, 0, 0.99996]\)
- At \(\psi = 181°\): \(q \approx [-0.0087, 0, 0, 0.99996]\)

Both describe the same physical yaw (181° ≡ −179°). Many downstream systems (USD, Replicator, some Isaac Sim write paths) normalize the quaternion to keep a chosen component positive. If they flip between \(+q\) and \(-q\) arbitrarily, the recorded trajectory has a discontinuity: `[0.0087, ..., 0.99996]` followed by `[+0.0087, ..., -0.99996]` (which is \(-q\) written with flipped \(w\), not the physically correct \(-q\)).

**Fix in the data-collection script:** After recording the trajectory, enforce continuity in the save step:

```python
def enforce_quaternion_continuity(q):
    """Flip signs of consecutive quaternions so dot(q[i-1], q[i]) ≥ 0."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] = -q[i]
    return q

# In the save function:
traj["orientation"] = enforce_quaternion_continuity(np.asarray(traj["orientation"]))
```

This runs once per trajectory and removes any artificial sign flips introduced by the recorder.

### Problem B — The model needs a quaternion-invariant loss

Even with continuous recorded quaternions, the training loss must be sign-aware. Raw MSE:

\[
\mathcal{L}_{\text{naive}} = \| q_{t+1} - \hat{q}_{t+1} \|^2
\]

The model can achieve near-zero loss with \(\hat q = -q\) (same rotation, opposite representation) only if the network implicitly learns to pick the same sign. This is fragile. The standard fix is \(d_{SO(3)}\):

\[
\mathcal{L}_{SO(3)} = 1 - |\langle q, \hat q \rangle|^2
\]

or equivalently:

\[
\mathcal{L}_{SO(3)} = 1 - \langle q, \hat q \rangle^2
\]

which is invariant to sign flips of either argument. This is *already* in your proposal (§1.1: "quaternion \(d_{SO(3)}\)"). Mode 4's boundary-crossing trajectories ensure the model has training data where the sign matters, so the \(d_{SO(3)}\) loss actually gets exercised.

---

## Verification Checks Specific to Mode 4

### Check 1 — Quaternion continuity per trajectory

```python
import numpy as np

for yaw_file in yaw_files:
    d = np.load(yaw_file)
    q = d["orientation"]                 # (T, 4)

    dots = np.sum(q[:-1] * q[1:], axis=1)
    n_negative = int(np.sum(dots < 0))
    if n_negative > 0:
        print(f"FAIL: {yaw_file.name} has {n_negative} quaternion sign flips")
    else:
        print(f"OK:   {yaw_file.name} quaternion sequence continuous")

    # Also check that at least one trajectory crosses the ±π boundary
    yaws = 2 * np.arctan2(q[:, 3], q[:, 0])
    yaws_unwrapped = np.unwrap(yaws)
    span = yaws_unwrapped.max() - yaws_unwrapped.min()
    crosses_pi = (
        (yaws_unwrapped.max() > np.pi) and (yaws_unwrapped.min() < np.pi)
    ) or (
        (yaws_unwrapped.max() > -np.pi) and (yaws_unwrapped.min() < -np.pi)
    )
    print(f"     yaw span = {np.degrees(span):.1f}°, crosses ±π = {crosses_pi}")
```

**Acceptance criteria:**

| Check | Pass | Fail |
|---|---|---|
| Zero negative dot products | All trajectories | Any trajectory |
| At least one boundary-crossing trajectory | `boundary_crossing_pi` trajectory has `crosses_pi = True` | No trajectory crosses ±π |
| Yaw span coverage | Union over all trajectories ≥ 360° | Union < 270° |

### Check 2 — CW/CCW pair symmetry

For each `yaw_cw_ccw` pair:

```python
for pair_id, (cw_file, ccw_file) in load_yaw_cw_ccw_pairs().items():
    cw  = np.load(cw_file)
    ccw = np.load(ccw_file)

    q_cw  = cw["orientation"]
    q_ccw = ccw["orientation"]

    yaw_cw  = np.unwrap(2 * np.arctan2(q_cw[:,  3], q_cw[:,  0]))
    yaw_ccw = np.unwrap(2 * np.arctan2(q_ccw[:, 3], q_ccw[:, 0]))

    delta_cw  = yaw_cw[-1]  - yaw_cw[0]
    delta_ccw = yaw_ccw[-1] - yaw_ccw[0]

    # Should be approximately ±π/2 (opposite signs, same magnitude)
    assert abs(delta_cw) > 1.3, f"CW undershoot: {np.degrees(delta_cw)}°"
    assert abs(delta_ccw) > 1.3, f"CCW undershoot: {np.degrees(delta_ccw)}°"
    assert np.sign(delta_cw) == -np.sign(delta_ccw), "CW and CCW not opposite"
    assert abs(abs(delta_cw) - abs(delta_ccw)) / abs(delta_cw) < 0.15, "Magnitude mismatch"
    print(f"{pair_id}: CW={np.degrees(delta_cw):.1f}°, CCW={np.degrees(delta_ccw):.1f}°")
```

### Check 3 — Rate comparison pair (S2a / S2b)

```python
for pair_id, (slow_file, fast_file) in load_heading_180_pairs().items():
    slow = np.load(slow_file)
    fast = np.load(fast_file)

    # Same endpoint: final yaw should agree within ±10°
    slow_end = np.unwrap(2 * np.arctan2(slow["orientation"][:, 3], slow["orientation"][:, 0]))[-1]
    fast_end = np.unwrap(2 * np.arctan2(fast["orientation"][:, 3], fast["orientation"][:, 0]))[-1]
    assert abs(slow_end - fast_end) < np.radians(10), "Endpoints differ too much"

    # Different intermediate dynamics: at midpoint, yaws differ by >20°
    mid_slow = np.unwrap(2 * np.arctan2(slow["orientation"][:, 3], slow["orientation"][:, 0]))[len(slow)//2]
    mid_fast = np.unwrap(2 * np.arctan2(fast["orientation"][:, 3], fast["orientation"][:, 0]))[len(fast)//2]
    assert abs(mid_slow - mid_fast) > np.radians(20), "Intermediate dynamics too similar"

    # Peak yaw rate comparison
    omega_slow = np.max(np.abs(slow["angular_velocity"][:, 2]))
    omega_fast = np.max(np.abs(fast["angular_velocity"][:, 2]))
    assert omega_fast > 2 * omega_slow, f"Fast isn't 2× slow: {omega_slow:.2f} vs {omega_fast:.2f}"
    print(f"{pair_id}: ω_slow={omega_slow:.2f} rad/s, ω_fast={omega_fast:.2f} rad/s")
```

### Check 4 — Action–wind decorrelation

Because yaw is largely orthogonal to the wind force (the drone rotates but doesn't translate much), the action–wind correlation should already be low. Verify it explicitly:

```python
for yaw_file in yaw_files:
    d = np.load(yaw_file)
    u = d["action"]                 # (T, 4)
    F = d["wind_force_N"]           # (T, 3)

    u_yaw = u[:, 3]                 # yaw-rate channel
    F_mag = np.linalg.norm(F, axis=1)

    corr = abs(np.corrcoef(u_yaw, F_mag)[0, 1]) if F_mag.std() > 1e-6 else 0.0
    if corr > 0.4:
        print(f"WARN: {yaw_file.name} corr(u_yaw, |F_w|) = {corr:.2f} > 0.4")
    else:
        print(f"OK:   {yaw_file.name} corr = {corr:.2f}")
```

If any trajectory exceeds 0.4, either the wind magnitude is too variable within a trajectory (causing the controller to chatter), or the wind is aligned with the yaw axis in a way that couples. Wind directions in Mode 4 should avoid pure `up`/`down` for yaw-only scenarios — those couple to thrust and produce action correlations. The `sinusoidal_sweep` scenario includes `up` as one of the possible wind directions; if it produces `corr > 0.4`, remove `up` from the `direction_from_list`.

---

## One Note on Mode 4 and OOD Evaluation

Yaw-mode trajectories **are** eligible for OOD test sets. The reason:

- Yaw rotates the body-frame; wind is world-frame. Under yaw, the residual \(R^\top F_w\) changes rapidly even when \(F_w\) is constant. This is a pure structural transform the model must learn.
- The paired CW/CCW trajectories provide the sign symmetry the model needs to learn causal transform, not a statistical one.
- The boundary-crossing trajectories are exactly what a directional-OOD test will stress (any test that rotates wind by an arbitrary angle will hit the ±π boundary).

Mode 4's 12.5% share is enough to cover the direction space, but not enough to dominate over translation (Mode 2) or combined (Mode 6), which is correct — yaw is one degree of freedom among thirteen state dimensions. If Mode 4 turns out to be underweighted after initial training (measured by angular velocity prediction error under OOD wind), raise its fraction to 15% by pulling 2.5% from hover or attitude.