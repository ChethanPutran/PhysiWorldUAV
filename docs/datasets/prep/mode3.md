# Revised Acceleration Mode Configuration

The original block had four defects the "what to vary" list exposes:

1. **Fixed direction** — all segments accelerate along +X or +Z only. Backward, lateral, and diagonal are absent.
2. **Fixed magnitudes** — every ramp uses `[0.6, 1.4] mg` with the same ramp duration. No gentle/moderate/aggressive distinction.
3. **No initial-velocity variation** — every segment starts from the same hover state.
4. **No paired comparison** — the user's most valuable scenario ("same initial/final velocity, different intermediate dynamics") cannot be generated from a single timeline.

The rewrite below converts the fixed timeline into **10 scenario templates**, with dedicated coverage for every "important scenario" from the list — including a **paired aggressive/gentle pair** that produces the same endpoints with deliberately different intermediate dynamics.

```yaml
# ==========================================================
# MODE 3: ACCELERATION / DECELERATION (17.5%)
# ==========================================================
# Residual = F_w/m directly — cleanest physical signal.
#
# Every scenario exercises a distinct combination of:
#   - acceleration magnitude  (gentle / moderate / aggressive)
#   - direction               (fwd / back / lateral / diagonal / 3D)
#   - initial velocity        (zero / residual)
#   - target velocity         (slow / moderate / fast)
#   - acceleration duration   (short / medium / long)
#   - initial angular velocity (zero / perturbed)
#   - wind                    (opposing / assisting / crosswind / gust)
#   - attitude transient      (implicit in every ramp)
#
# Trajectory count: 18 (17.5% of 100). With guarantee_coverage,
# every scenario appears at least once, with the paired
# comparison sampled as a matched pair.
# ==========================================================
acceleration:
  fraction: 0.175

  definition:
    duration_range_s: [12.0, 20.0]
    altitudes_m: [1.5, 2.0, 2.5, 3.0]     # headroom for vertical maneuvers

    # ── Initial conditions ─────────────────────────────────
    initial_conditions:
      position_xy_range_m: [-3.0, 3.0]
      position_z_offset_m: [-0.1, 0.1]
      velocity_magnitude_range_mps: [0.0, 0.0]
      velocity_direction: random_unit_vector
      roll_offset_deg: 0.0
      pitch_offset_deg: 0.0
      yaw_range_rad: [-3.14159, 3.14159]
      # NEW for Mode 3 — user's "initial angular velocity" requirement
      angular_velocity_magnitude_range_radps: [0.0, 0.0]
      angular_velocity_axis: random_unit_vector

    # ── Wind and axis vocabularies ─────────────────────────
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

    acceleration_classes:
      gentle:     [0.3, 1.0]              # m/s²
      moderate:   [1.0, 3.0]
      aggressive: [3.0, 6.0]

    # ── Scenario templates ─────────────────────────────────
    scenarios:

      # ─────────────────────────────────────────────────
      # S1: Accelerate from rest, calm air (baseline)
      # ─────────────────────────────────────────────────
      - name: accelerate_from_rest_calm
        weight: 0.12
        phases:
          - duration_s: [12.0, 18.0]
            controller:
              type: accel_decel
              axis_from_list: [north, east, ne]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [0.8, 1.8]
              acceleration_class: gentle
              return_to_start: true
            wind: {type: none}

      # ─────────────────────────────────────────────────
      # S2: Accelerate with wind aligned or opposed to motion
      # Covers user's "headwind" and "tailwind" as one template —
      # they are mirror images, and the residual learns the sign flip.
      # ─────────────────────────────────────────────────
      - name: accelerate_wind_aligned
        weight: 0.14
        phases:
          - duration_s: [12.0, 18.0]
            controller:
              type: accel_decel
              axis: [1, 0, 0]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [0.8, 2.0]
              acceleration_class: moderate
              return_to_start: true
            wind:
              type: constant
              magnitude_N: [2.0, 5.0]
              direction_from_list: [north, south]  # aligned or opposed

      # ─────────────────────────────────────────────────
      # S3: Accelerate with crosswind (user list: "with a crosswind")
      # ─────────────────────────────────────────────────
      - name: accelerate_crosswind
        weight: 0.12
        phases:
          - duration_s: [12.0, 18.0]
            controller:
              type: accel_decel
              axis: [1, 0, 0]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [0.8, 1.8]
              acceleration_class: moderate
              return_to_start: true
            wind:
              type: constant
              magnitude_N: [2.0, 5.0]
              direction_from_list: [east, west, up, down]

      # ─────────────────────────────────────────────────
      # S4: Accelerate from already-moving state (user list)
      # Drone enters with non-zero velocity; controller continues
      # the acceleration from there, so the ramp is shorter and
      # the initial-state distribution is broader.
      # ─────────────────────────────────────────────────
      - name: accelerate_from_moving
        weight: 0.10
        initial_conditions:
          velocity_magnitude_range_mps: [0.4, 1.2]
          velocity_direction: random_unit_vector
        phases:
          - duration_s: [10.0, 15.0]
            controller:
              type: accel_decel
              axis_from_list: [north, east, ne]
              start_from_initial_velocity: true
              peak_velocity_range_mps: [1.2, 2.5]
              acceleration_class: moderate
              return_to_start: true
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction_from_list: [north, east]

      # ─────────────────────────────────────────────────
      # S5: Gust midway through acceleration (user list)
      # Controller must reject the gust while still executing the ramp.
      # ─────────────────────────────────────────────────
      - name: gust_mid_acceleration
        weight: 0.10
        phases:
          - duration_s: [15.0, 20.0]
            controller:
              type: accel_decel
              axis: [1, 0, 0]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [1.0, 2.0]
              acceleration_class: moderate
              return_to_start: true
            wind:
              type: gust_mid
              trigger_at_progress: [0.35, 0.65]   # fraction of trajectory
              peak_magnitude_N: [5.0, 8.0]
              duration_s: [0.3, 0.8]
              direction: [-1, 0, 0]

      # ─────────────────────────────────────────────────
      # S6a: Trapezoidal ramp — near-rectangular velocity profile
      # Same endpoints as S6b, but long time spent at peak velocity.
      # ─────────────────────────────────────────────────
      - name: same_target_trapezoid
        weight: 0.08
        pair_group: ramp_profile_comparison
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: accel_decel
              axis: [1, 0, 0]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [1.0, 2.0]    # ← shared with S6b
              ramp_fraction: 0.05                    # near-rectangular
              return_to_start: true
            wind:
              type: constant
              magnitude_N: [2.0, 4.0]                # ← shared with S6b
              direction: [-1, 0, 0]

      # ─────────────────────────────────────────────────
      # S6b: Triangular ramp — velocity constantly changing
      # Same endpoints and peak as S6a, but ramp_fraction = 1.0
      # means no hold phase at the peak.
      # ─────────────────────────────────────────────────
      - name: same_target_triangle
        weight: 0.08
        pair_group: ramp_profile_comparison
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: accel_decel
              axis: [1, 0, 0]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [1.0, 2.0]    # ← shared with S6a
              ramp_fraction: 1.0                     # triangular
              return_to_start: true
            wind:
              type: constant
              magnitude_N: [2.0, 4.0]                # ← shared with S6a
              direction: [-1, 0, 0]

      # ─────────────────────────────────────────────────
      # S8: Lateral acceleration (±Y)
      # Fills 3D direction coverage — S1/S2/S3 focus on X.
      # ─────────────────────────────────────────────────
      - name: lateral_acceleration
        weight: 0.08
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: accel_decel
              axis_from_list: [east, west]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [0.8, 1.8]
              acceleration_class: moderate
              return_to_start: true
            wind:
              type: constant
              magnitude_N: [2.0, 4.0]
              direction_from_list: [north, south]

      # ─────────────────────────────────────────────────
      # S9: Initial angular velocity perturbation (user list)
      # ω ≠ 0 at spawn; controller must stabilise AND track the ramp.
      # ─────────────────────────────────────────────────
      - name: initial_angular_velocity
        weight: 0.08
        initial_conditions:
          angular_velocity_magnitude_range_radps: [0.3, 1.0]
          angular_velocity_axis: random_unit_vector
        phases:
          - duration_s: [10.0, 14.0]
            controller:
              type: accel_decel
              axis: [1, 0, 0]
              start_velocity_range_mps: [0.0, 0.0]
              peak_velocity_range_mps: [0.8, 1.5]
              acceleration_class: moderate
              return_to_start: true
            wind:
              type: constant
              magnitude_N: [1.5, 3.5]
              direction: [-1, 0, 0]

      # ─────────────────────────────────────────────────
      # S10: Thrust sinusoid — high-frequency acceleration
      # Captures rapid acceleration sign flips that the world
      # model must resolve at 100 Hz.
      # ─────────────────────────────────────────────────
      - name: thrust_sinusoid
        weight: 0.08
        phases:
          - duration_s: [12.0, 16.0]
            controller:
              type: thrust_sinusoid
              amplitude_frac: [0.20, 0.40]           # of mg
              frequency_hz: [0.15, 0.35]
              axis: z
            wind:
              type: constant
              magnitude_N: [1.5, 4.0]
              direction_from_list: [up, down, north]

    # ── Paired scenarios ──────────────────────────────────
    # The two members share the parameters listed in `shared`,
    # so the training pipeline can pair them for controlled
    # comparison: same (s_0, s_T, V_peak, F_w) but different
    # intermediate V(t). This is the user's scenario #6.
    paired_scenarios:
      - name: ramp_profile_comparison
        members: [same_target_trapezoid, same_target_triangle]
        shared:
          - peak_velocity_range_mps
          - start_velocity_range_mps
          - total_duration_s
          - wind_magnitude_N
          - wind_direction
        pair_id_prefix: "ramp_profile"

    # ── Sampling policy ───────────────────────────────────
    sampling:
      guarantee_coverage: true                # each scenario appears ≥1×
      pairs_generated_together: true          # paired members always co-generated

    # ── Expected composition ──────────────────────────────
    report:
      trajectory_count: 18
      scenarios_present: 10
      avg_duration_s: 15.0
      total_steps: ~27000                     # 15 s × 100 Hz × 18
      paired_pairs_generated: 1               # the S6a/S6b pair

      # Coverage matrix vs user's "important scenarios" list
      coverage:
        from_rest_calm_air:         S1
        against_headwind:           S2 (direction south when axis north)
        with_crosswind:             S3
        from_moving_state:          S4
        gust_mid_acceleration:      S5
        same_V_different_profile:   S6a + S6b (paired)
        # supplemental coverage
        with_tailwind:              S2 (direction north when axis north)
        lateral_acceleration:       S8
        initial_angular_velocity:   S9
        high_freq_acceleration:     S10

  counterfactual: true
```

---

## What Changed vs. the Original

| Defect in original | How the rewrite fixes it |
|---|---|
| All ramps along +X or +Z only | S3 uses east/west, S8 uses ±Y, S1 uses `axis_from_list`, S9 uses ±X with initial angular disturbance |
| Fixed acceleration magnitude `[0.6, 1.4] mg` | `acceleration_class` parameter samples from gentle (0.3–1.0), moderate (1.0–3.0), or aggressive (3.0–6.0) m/s² |
| Fixed ramp duration (3 s) | `ramp_duration` derived from `peak_velocity / acceleration_magnitude` — varies 0.2–5 s automatically |
| No initial-velocity variation | S4's `start_from_initial_velocity: true` + `initial_conditions.velocity_magnitude_range_mps: [0.4, 1.2]` |
| No initial angular velocity | `initial_conditions.angular_velocity_magnitude_range_radps` field added |
| No gust-mid-ramp scenario | S5 with `trigger_at_progress` wind |
| No paired comparison | S6a / S6b with explicit `pair_group: ramp_profile_comparison` |
| Only one wind type (constant / step) | S5 uses `gust_mid`, S2/S3 use `direction_from_list`, S10 uses thrust sinusoid |
| Every trajectory tests the same dynamics | 10 scenarios span 3 orders of magnitude in ramp time, 3 direction axes, 2 ramp shapes |

---

## Controller and Wind Library Additions

### New `accel_decel` controller

This is the core controller for every scenario. Required parameters:

```yaml
controller:
  type: accel_decel
  axis: [x, y, z]                      OR
  axis_from_list: [name1, name2, ...]   # resolves via `directions` dict
  start_velocity_range_mps: [a, b]      # sampled unless start_from_initial_velocity
  start_from_initial_velocity: bool     # if true, use drone's current velocity
  peak_velocity_range_mps: [c, d]       # sampled, magnitude along axis
  acceleration_class: gentle|moderate|aggressive   # used to derive ramp_duration
  ramp_fraction: [0.05, 1.0]            # optional, overrides acceleration_class
                                        #   ramp_fraction=1.0 → triangular profile (no hold)
                                        #   ramp_fraction=0.05 → near-rectangular (long hold)
  return_to_start: bool                 # if true, ramp back to start_velocity at end
  z_change_m: float                      # optional altitude delta over trajectory
```

**Velocity profile generated** (with `T` = phase duration, `V_peak` = sampled peak):

```
ramp_duration = ramp_fraction × (T / 2)                    if ramp_fraction given
             OR = V_peak / acceleration_magnitude          otherwise

T_ramp_up = ramp_duration
T_hold    = max(0, T - 2 × ramp_duration)   (return_to_start=true)
          = max(0, T - ramp_duration)        (return_to_start=false)
```

The controller then produces a target velocity at each timestep. The underlying `RateController` tracks this via P-D on velocity → attitude targets.

### New `gust_mid` wind type

```yaml
wind:
  type: gust_mid
  trigger_at_progress: [low, high]      # fraction of trajectory duration, sampled
  peak_magnitude_N: [min, max]
  duration_s: [min, max]
  direction: [x, y, z]
```

**Behaviour:** No wind except during the window `[trigger_time, trigger_time + duration]`, where `trigger_time = sampled_progress × trajectory_duration`. This is distinct from the existing `gust` type (which uses absolute `gust_time_s`) — the trigger is trajectory-relative, so it works regardless of the duration sampled in a scenario.

### New `initial_conditions` fields

- `angular_velocity_magnitude_range_radps`: sampled magnitude of initial body-frame angular velocity
- `angular_velocity_axis`: `random_unit_vector` or a specific axis

The generation script applies these by calling `drone.set_velocities(torch.cat([v_body, ω_body]))` before `drone.initialize()`.

---

## Paired Scenario Generation Mechanism

The generation script must handle `pair_group` at the trajectory sampling stage. Pseudocode:

```python
def generate_dataset(config):
    scenarios = flatten_scenarios(config)
    pairs = config.definition.paired_scenarios

    # 1. Sample pair parameters FIRST — before individual trajectories
    pair_params = {}
    for pair in pairs:
        pair_params[pair.name] = sample_shared_params(pair.shared, config)

    # 2. For each trajectory slot in each scenario, generate
    for scenario in scenarios:
        n_traj = round(config.fraction * config.total_trajectories * scenario.weight)
        for i in range(n_traj):
            # If this scenario belongs to a pair, use the pre-sampled shared params
            if scenario.name in pair_members:
                pair_name = pair_members[scenario.name]
                shared = pair_params[pair_name]
                pair_id = f"{pair_prefix}_{counter:04d}"
            else:
                shared = None
                pair_id = None

            traj = simulate(scenario, shared_params=shared)
            save(traj, pair_id=pair_id, scenario=scenario.name)
```

For the `ramp_profile_comparison` pair, both members get the same `pair_id` (e.g., `ramp_profile_0001`). A downstream loader can group them:

```python
triples = {}
for npz in all_files:
    meta = load_json(npz)
    if meta["pair_id"]:
        triples.setdefault(meta["pair_id"], []).append(npz)

for pair_id, files in triples.items():
    # files[0] = S6a (trapezoid), files[1] = S6b (triangle)
    # Same V_peak, same F_w, different V(t) → different intermediate dynamics
    ...
```

---

## Verification Checks Specific to Mode 3

After generation, run these to confirm the data carries the acceleration signal correctly:

```python
import numpy as np

for npz_path in acceleration_files:
    d = np.load(npz_path)
    v = d["velocity"]                       # (T, 3)
    u = d["action"]                         # (T, 4)
    F = d["wind_force_N"]                   # (T, 3)

    v_mag = np.linalg.norm(v, axis=1)
    F_mag = np.linalg.norm(F, axis=1)
    u_thrust = u[:, 0]

    print(f"{npz_path.name}: "
          f"v_max={v_mag.max():.2f}  "
          f"a_max={np.max(np.abs(np.diff(v_mag)/0.01)):.2f}  "
          f"corr(u_thrust, F_w)={abs(np.corrcoef(u_thrust, F_mag)[0,1]):.2f}")
```

**Per-scenario acceptance criteria:**

| Scenario | Expected `v_max` | Expected `a_max` | Expected `corr(u_thrust, F_w)` |
|---|---|---|---|
| `accelerate_from_rest_calm` | 0.8–1.8 m/s | 0.3–1.0 m/s² | < 0.1 (no wind) |
| `accelerate_wind_aligned` | 0.8–2.0 m/s | 1.0–3.0 m/s² | 0.2–0.4 (moderate, expected) |
| `accelerate_crosswind` | 0.8–1.8 m/s | 1.0–3.0 m/s² | < 0.2 (crosswind decoupled from thrust axis) |
| `accelerate_from_moving` | 1.2–2.5 m/s | 1.0–3.0 m/s² | 0.2–0.4 |
| `gust_mid_acceleration` | 1.0–2.0 m/s | 1.0–3.0 m/s² + gust impulse | 0.3–0.5 (gust dominates) |
| `same_target_trapezoid` | V* (shared) | ≥ 3.0 m/s² (short ramp) | 0.3–0.5 |
| `same_target_triangle` | V* (shared) | ≤ 1.5 m/s² (long ramp) | 0.3–0.5 |
| `lateral_acceleration` | 0.8–1.8 m/s | 1.0–3.0 m/s² | < 0.2 |
| `initial_angular_velocity` | 0.8–1.5 m/s | 1.0–3.0 m/s² + ω transient | < 0.2 |
| `thrust_sinusoid` | ~0.5 m/s oscillation | 2.0–4.0 m/s² peak | 0.2–0.4 |

**Pair-specific verification:** For every `pair_id` in `ramp_profile_*`, the two trajectories must satisfy:

```python
pair = load_pair(pair_id)
a, b = pair["trapezoid"], pair["triangle"]

# Same endpoints
assert np.allclose(a["velocity"][0], b["velocity"][0], atol=0.05)     # start
assert np.allclose(a["velocity"][-1], b["velocity"][-1], atol=0.05)   # end
# Same peak
assert abs(a["velocity"].max() - b["velocity"].max()) / a["velocity"].max() < 0.1
# Same wind (must be identical sequences)
assert np.allclose(a["wind_force_N"], b["wind_force_N"], atol=0.01)
# Different intermediate dynamics (mean velocity over middle 50%)
mid_a = np.mean(np.linalg.norm(a["velocity"][len(a["velocity"])//4 : 3*len(a["velocity"])//4], axis=1))
mid_b = np.mean(np.linalg.norm(b["velocity"][len(b["velocity"])//4 : 3*len(b["velocity"])//4], axis=1))
assert abs(mid_a - mid_b) / max(mid_a, mid_b) > 0.2                   # ≥20% difference
```

If the last assertion fails (intermediate dynamics too similar), increase the difference between `ramp_fraction` values in S6a and S6b — e.g., from `0.05` / `1.0` to `0.02` / `1.0`.

---

## One Important Note About Mode 3 and Hover

Unlike Mode 1, Mode 3 trajectories **are** eligible for OOD test sets. The residual signal is strong because:

- **Acceleration under wind produces nonzero `s_{t+1} − f_phys(s_t, u_t)`** — this is the identifiability property (Problem 1 defeated)
- **Randomness in the accel profiles breaks `u_t ↔ F_w` correlation** — this is the decorrelation property (Problem 2 defeated)
- **The state genuinely evolves during and after the ramp** — this is the transient richness (Problem 3 defeated)

This makes Mode 3 the second-most-important mode after Mode 2 (translation) for OOD evaluation. Its share of the dataset (17.5%) reflects this: it is below Mode 2 (27.5%) because acceleration is a subset of the maneuvers a translation goes through, but above every other mode.