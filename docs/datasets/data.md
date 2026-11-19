
# Dataset Recipe for World-Model Training Across Flight Modes

Before the per-mode tables, three design principles govern every row. Every mode must satisfy all three, or the residual \(d_\theta\) becomes unlearnable (Problems 1–3 from earlier).

| Principle | What it means for a mode |
|---|---|
| **Identifiability** | \(s_{t+1} - f_{\text{phys}}(s_t, u_t)\) must depend on \(F_w\) and be nonzero for some samples. This fails under perfect hover-compensation (Problem 1). |
| **Decorrelation** | \(u_t\) must **not** be a deterministic function of \(F_w\). This fails under all hover-compensation (Problem 2). |
| **Full transient** | The state must evolve during the wind disturbance. This fails under hover (Problem 3). |

**Every mode below is designed to violate at least one of the failure conditions in a controlled way.**

---

## Mode 1 — Hover (Altitude Hold)

### What it teaches
Equilibrium dynamics: the drone can predict "state stays put" when the controller is active, and can account for the compensating thrust. This is the **only mode** where wind is fully absorbed into \(u_t\), so it is the *weakest* supervision. It is included for completeness but must be the smallest segment.

| Dimension | Specification |
|---|---|
| **State variation** | \(\mathbf p \in [\pm 0.1]\) m jitter, \(\mathbf v \approx 0\) (\(|\mathbf v| < 0.05\) m/s), \(\mathbf q \approx \text{identity}\) (tilt \(< 5°\)), \(\boldsymbol\omega \approx 0\) |
| **Action exploration** | Narrow: thrust near hover (\(T \in [0.95\,mg, 1.05\,mg]\)), roll/pitch/yaw rates \(\in [-0.05, 0.05]\) |
| **Altitude** | **Four fixed altitudes**, not one: 0.5, 1.0, 1.5, 2.5 m. De-correlates altitude from state, so the model does not overfit to \(z = 1.5\) |
| **Wind** | Constant at 0, 1, 2, 4 N (four sub-runs). Also a "step 0 → X N" sub-run for the compensating transient |
| **Transient content** | ~0 (that is the point of this mode). Do **not** use hover for OOD validation |
| **Counterfactual pairs** | Optional — hover is degenerate for causal pairing anyway |

**Trajectory structure (per trajectory):**
```
0–5s   wind = 0 N        (establish hover)
5–10s  wind = constant   (compensating steady state)
10–12s wind stepped to +2 N higher  (transient — 2s only)
12–15s wind constant (new level)    (settle)
```
Total 15 s per hover trajectory. Altitude is picked from {0.5, 1.0, 1.5, 2.5} uniformly.

**Data volume:** 10% of total (≈10 trajectories).

**Why 10% and not 0%:** the model needs *some* equilibrium data so that during OOD evaluation, hover predictions do not explode. But if you use 100% hover, the model learns the trivial "\(s_{t+1} = s_t\)" answer (Problem 1).

---

## Mode 2 — Translational Motion

### What it teaches
The workhorse. Wind under translational flight interacts with the state via **aerodynamic drag** — the effective wind force is roughly \(F_{\text{drag}} \propto (\mathbf v - \mathbf v_{\text{wind}})^2\), so the same wind produces different residuals at different speeds. This is the cleanest identifiability signal for \(d_\theta\).

| Dimension | Specification |
|---|---|
| **State variation** | \(\mathbf v\): 0.2–2.0 m/s, all three axes exercised separately then combined. \(\mathbf p\): ±5 m displacement from origin. Tilt: 5–25° (differential thrust provides acceleration, tilt follows). \(\boldsymbol\omega\): small (< 0.5 rad/s) except during direction changes |
| **Action exploration** | **Two sub-types per trajectory:**<br>(a) *Scripted*: straight-line, circle (r = 2 m), figure-eight (40% of time)<br>(b) *Random*: sample \(u_t\) uniformly from \([-0.6, 0.6]^4\) every 0.3 s (60% of time) |
| **Altitude** | Slow altitude ramps 1.0 → 2.0 m over the trajectory, so \(z\) is not fixed |
| **Wind** | Three wind orientations **per trajectory**:<br>- **Crosswind** (perpendicular to motion): drift + roll coupling<br>- **Headwind** (against motion): increased drag, deceleration<br>- **Tailwind** (with motion): reduced effective drag, acceleration |
| **Transient content** | Constant-speed flight gives steady state; direction changes give 200–400 ms transients; random action segments give continuous transients |
| **Counterfactual pairs** | **Required.** For each scripted segment, replay with 0 N and 2× the nominal wind |

**Trajectory structure:**
```
0–10s    scripted: straight line +X, crosswind
10–20s   scripted: straight line +Y, crosswind
20–30s   scripted: circle r=2m, wind from +X
30–40s   scripted: figure-eight, wind stepped at t=35s
40–70s   random actions, constant 2 N wind
70–100s  random actions, wind stepped every 10s among {0, 2, 4, 6} N
```

**Data volume:** 25–30% of total (≈30 trajectories × 100 s).

**Why this is the workhorse:** the residual \(s_{t+1} - f_{\text{phys}}(s_t, u_t)\) becomes a *strong, v-dependent function of \(F_w\)*. Random actions break the action-wind correlation. Both Problems 1 and 2 are defeated here.

---

## Mode 3 — Acceleration and Deceleration

### What it teaches
The **direct signal** \(a = (T + F_w)/m - g\hat z\). Under free acceleration, \(f_{\text{phys}}\) predicts the rigid-body acceleration correctly, but with the wrong \(T\) (because the controller has not yet compensated for the wind). The residual is **exactly** the wind-induced acceleration \(F_w/m\), and its sign flips with thrust direction.

| Dimension | Specification |
|---|---|
| **State variation** | \(a_t\): \(\pm 3\) m/s² in z, \(\pm 2\) m/s² in x/y. \(\mathbf v\): cross-couples with \(a_t\) during ramps. Tilt: 20–40° during aggressive accel (thrust vectoring) |
| **Action exploration** | **Ramp profiles:**<br>- Thrust ramp: \(T\) from 0.6 → 1.4 \(\times mg\) over 3 s, hold 2 s, back down<br>- Thrust step: \(T\) jumps by ±40% \(mg\) in 1 step<br>- Thrust sinusoid: 0.2 Hz, amplitude 30% of hover thrust<br>- Random rate commands + thrust ramp |
| **Altitude** | 1.5–3.0 m — need headroom so ramps do not hit ground |
| **Wind** | Constant at **two magnitudes per trajectory** (0 and 4 N) so ramp direction × wind direction is exercised. Additionally, a step mid-ramp to test transient during acceleration |
| **Transient content** | **Primary.** Ramp duration × \(1/dt\) = 300 samples of clean acceleration transient. This is where the residual carries the most information |
| **Counterfactual pairs** | **Required.** Same ramp, \(F_w = 0\) vs \(F_w = 4\) N. The residual difference is exactly \(F_w/m\) |

**Trajectory structure:**
```
0–15s    thrust ramps in z, wind=0
15–30s   thrust ramps in z, wind=4 N
30–45s   thrust ramps in z, wind=step 0→4 N at t=37s  (transient during ramp!)
45–60s   thrust ramps in x (forward accel), wind=0
60–75s   thrust ramps in x (forward accel), wind=4 N headwind
75–90s   combined ramp: T and roll command simultaneously, wind=2 N
90–100s  thrust sinusoid + wind step
```

**Data volume:** 15–20% of total (≈20 trajectories × 100 s).

**Why this is essential:** acceleration is the *only* mode where \(f_{\text{phys}}\) is provably wrong under wind. Under hover, \(f_{\text{phys}}\) matches the compensated thrust. Under acceleration, \(f_{\text{phys}}\) uses raw \(T\) while the true force is \(T + F_w\). The residual must therefore equal \(F_w/m\), giving the model an unambiguous target.

---

## Mode 4 — Yaw Motion

### What it teaches
Wind direction is defined in the **world frame**, but the drone's aerodynamics are in the **body frame**. As the drone yaws, the effective wind angle changes — even when \(F_w\) is constant. This forces the model to learn the frame transformation \(\mathbf R^\top F_w\). Without yaw data, the model can succeed on any single flight direction but will fail when the drone turns.

| Dimension | Specification |
|---|---|
| **State variation** | \(\psi\) (yaw angle): full \(-\pi\) to \(+\pi\) coverage. \(\omega_z\): \(\pm 0.5\) to \(\pm 1.5\) rad/s during maneuvers, \(\approx 0\) otherwise. \(\mathbf p, \mathbf v\): small (hover-like) for yaw-only segments; nonzero for yaw-while-translating |
| **Action exploration** | **Three sub-modes:**<br>(a) *Yaw-only*: \(\omega_z = \pm 0.5\) rad/s constant for 5 s, then ±1.0, then ±1.5<br>(b) *Yaw sweep*: \(\omega_z = 0.5 \sin(2\pi \cdot 0.2 \cdot t)\) (0.2 Hz sinusoid)<br>(c) *Yaw while translating*: circle with yaw tangent to the circle |
| **Altitude** | 1.0–2.0 m constant (isolate yaw from altitude dynamics) |
| **Wind** | Constant wind at **four directions** relative to body: +X, +Y, −X, −Y. This means 4 sub-runs per yaw trajectory, each with a different wind direction. Critical: the same \(\omega_z\) must be paired with different world-frame \(F_w\) |
| **Transient content** | Yaw rate steps (~200 ms transient), yaw-stops (overshoot), attitude coupling during aggressive yaw |
| **Counterfactual pairs** | **Required.** Same yaw sequence, wind direction rotated 90° between runs |

**Trajectory structure:**
```
0–10s    hover, wind +X
10–20s   yaw sinusoid 0.2 Hz, wind +X
20–30s   yaw step-and-hold at π/2, wind +X
30–40s   yaw step-and-hold at π, wind +X
40–50s   yaw sinusoid, wind +Y   ← same action, wind rotated
50–60s   yaw step-and-hold, wind +Y
60–70s   circle with yaw tangent, wind +X
70–80s   circle with yaw tangent, wind +Y
80–100s  random: yaw + small translation, wind step at t=90s
```

**Data volume:** 10–15% of total (≈12 trajectories × 100 s).

**Why 10–15% and not more:** yaw is one dimension and the mode is narrow. But without it, the world-model cannot generalize across OOD wind directions — and directional OOD is one of your thesis's explicit regimes.

---

## Mode 5 — Moderate Attitude Changes

### What it teaches
Attitude is where **wind torque** matters. Wind that is not aligned with the drone's symmetry axis creates a torque \(\tau_w = r \times F_w\) at the center of pressure, which is offset from the center of gravity. This is a state dimension (quaternion + angular velocity) that has a *different* wind sensitivity than position. Without this mode, the model has no data on how wind rotates the drone.

| Dimension | Specification |
|---|---|
| **State variation** | Roll \(\phi\): \(-30°\) to \(+30°\). Pitch \(\theta\): \(-30°\) to \(+30°\). Yaw \(\psi\): small. Quaternion components span the full \([-1, 1]\). \(\boldsymbol\omega\): \(\pm 0.5\) to \(\pm 1.0\) rad/s during attitude maneuvers |
| **Action exploration** | **Three sub-modes:**<br>(a) *Attitude hold steps*: command roll = +15°, hold, then +30°, hold, then back to 0<br>(b) *Attitude sine*: \(\phi = 20° \sin(2\pi \cdot 0.3 \cdot t)\), separate run for pitch, separate run for combined roll-pitch<br>(c) *Random attitude commands*: sample roll/pitch rates from \([-0.5, 0.5]\) rad/s, hold for 0.5 s, resample |
| **Altitude** | 1.0–2.0 m (attitude maneuvers do not need altitude variation; keep it constant within each sub-mode) |
| **Wind** | **Vertical wind is critical.** Run each sub-mode with:<br>- Horizontal wind (baseline)<br>- Vertical wind (+\(z\))<br>- Vertical wind (−\(z\)) — simulates updraft/downdraft<br>- Diagonal (combined) |
| **Transient content** | Attitude steps give ~300–500 ms transients. Sine gives continuous oscillation. Random steps give irregular transients |
| **Counterfactual pairs** | **Required.** Same roll/pitch commands, wind switched between horizontal and vertical |

**Trajectory structure:**
```
0–15s    attitude step sequence (roll, then pitch), wind horizontal
15–30s   attitude step sequence, wind vertical +Z
30–45s   attitude sine (roll), wind horizontal
45–60s   attitude sine (roll), wind vertical +Z  ← isolates the vertical wind effect
60–75s   combined roll-pitch sine, wind horizontal
75–90s   combined roll-pitch sine, wind diagonal
90–100s  random attitude + wind step at t=95s
```

**Data volume:** 15% of total (≈15 trajectories × 100 s).

**Why 15%:** attitude has 7 of the 13 state dimensions (\(\mathbf q\) + \(\boldsymbol\omega\)). If you under-sample this mode, the model cannot predict \(s_{t+1}\)'s attitude components under OOD wind — which is exactly where the quaternion distance metric \(d_{SO(3)}\) in your proposal will fail first.

---

## Mode 6 — Combined Translational and Rotational

### What it teaches
The **coupled manifold**. In real flight, translation and rotation are never isolated. The world model must handle the nonlinear interaction: translational velocity affects wind drag, which creates torque, which tilts the drone, which changes the thrust vector, which changes acceleration. This is where the model generalizes (or fails to).

| Dimension | Specification |
|---|---|
| **State variation** | **All 13 dimensions vary.** \(\mathbf v\): 0.5–2 m/s. Tilt: 15–40°. \(\boldsymbol\omega\): 0.3–1.5 rad/s. \(z\): ramps through 1–3 m |
| **Action exploration** | **Waypoint-based + aggressive maneuvers:**<br>- Figure-eight in xy-plane with yaw tracking the tangent (30%)<br>- Helix: circle in xy while climbing in z, yaw tangent (30%)<br>- Random waypoints every 2 s, sample from \(\pm 3\) m sphere (40%) |
| **Altitude** | 1.0–3.0 m — full range exercised in helix |
| **Wind** | **All wind types represented:**<br>- Constant (4 magnitudes)<br>- Step (0→4 N and 4→0 N)<br>- Sinusoidal (0.1 Hz, 0.3 Hz)<br>- Gust (2 N → 6 N for 0.5 s → back)<br>- Dryden turbulence (\(\sigma = 0.5, 1.5, 2.5\)) |
| **Transient content** | Continuous — waypoints change every 2 s, and wind is time-varying, so the state is always in motion |
| **Counterfactual pairs** | **Strongly required.** Replay each trajectory under 3× wind magnitude and 0 wind |

**Trajectory structure (100 s single trajectory):**
```
0–20s    figure-eight, wind=2N constant
20–40s   figure-eight, wind=step 2→6N at t=30s
40–55s   helix climb, wind=sinusoidal 0.2Hz amp 3N
55–70s   helix descent, wind=gust at t=62s
70–85s   random waypoints, wind=dryden σ=1.5
85–100s  random waypoints, wind=step 6→0 at t=92s
```

**Data volume:** 15–20% of total (≈18 trajectories × 100 s).

**Why 15–20%:** this is the mode that most resembles evaluation conditions. Under-trained, the model overfits to isolation modes and fails on the OOD test set — which is exactly the failure mode a reviewer will flag.

---

## Wind Configuration Matrix

Every mode must exercise these wind scenarios. The rows are wind types; the columns are trajectory segments (all drawn from the modes above).

| Wind type | Where it appears | Why |
|---|---|---|
| `none` (0 N) | Every mode has a 0-wind baseline | Establishes what "no wind" looks like |
| `constant` 1–2 N | Hover, translation, yaw | Training-distribution wind |
| `constant` 4–6 N | Translation, accel, combined | OOD magnitude test (higher than training) |
| `step` 0 → 4 N | Accel/decel mid-ramp, combined | Shows transient under load |
| `step` 4 → 0 N | Combined, hover | Shows relaxation transient |
| `step` direction change | Yaw, attitude | Frame-rotation test |
| `sinusoidal` 0.1–0.3 Hz | Combined, translation | Time-varying wind |
| `gust` 2→6→2 N over 0.5 s | Combined | High-frequency disturbance |
| `dryden` σ ∈ {0.5, 1.5, 2.5} | Combined | Realistic turbulence |
| **Vertical** +Z / −Z | Attitude, hover | Updraft/downdraft response |

For the **OOD training regimes** (your thesis's five categories), the rule is:

| Regime | Training set contains | Test set adds |
|---|---|---|
| Magnitude OOD | Wind ≤ 4 N | Wind 5–8 N |
| Directional OOD | Horizontal wind | Vertical, diagonal directions |
| Temporal OOD | Constant + slow sinusoidal | Fast sinusoidal, sharp steps |
| Gust OOD | Mild σ | σ > 2.5 |
| Physics OOD | Nominal m, J, C_D | 10–20% perturbations |

---

## The Counterfactual Pair Requirement

This is the single most important addition. **For every scripted segment, replay the same action sequence with different wind.** Specifically:

```
Trajectory A:  scripted actions, wind = 0 N
Trajectory B:  same actions, wind = 2 N
Trajectory C:  same actions, wind = 4 N (or OOD level)
```

This gives you triples \((s_t, u_t, F_{w}^{(1)}), (s_t, u_t, F_w^{(2)}), (s_t, u_t, F_w^{(3)})\) that share \(s_t\) and \(u_t\) but have different wind. **The only way the model can fit all three is to use the IMU to disambiguate.** This is the direct countermeasure to Problem 2.

The counterfactual pairs are stored as separate trajectories with a `pair_id` field in the metadata, so the training loop can enforce them.

---

## Consolidated Data Volume

For **one OOD regime pair** (e.g., training + one OOD test set), 100 training trajectories × 100 s × 100 Hz = **10⁶ samples**:

| Mode | Trajectories | Steps per traj. | Total steps | Purpose |
|---|---|---|---|---|
| Hover | 10 | 1500 (15 s) | 15,000 | Equilibrium baseline |
| Translation | 25 | 10,000 | 250,000 | Workhorse — drag identifiability |
| Accel/decel | 18 | 10,000 | 180,000 | Direct \(F_w/m\) signal |
| Yaw | 12 | 10,000 | 120,000 | Frame transformation |
| Attitude | 15 | 10,000 | 150,000 | Torque + vertical wind |
| Combined | 20 | 10,000 | 200,000 | Coupling |
| **Total** | **100** | — | **~915,000** | |

Rounded: **~10⁶ steps**, matching the "training (moderate wind): \(2\times 10^5\) to \(10^6\)" spec in your data requirements document.

For validation: 20% of the above, held out by trajectory index.

For OOD test: rerun the *same* recipe with the test-regime wind parameters. Each OOD regime gets its own ~200 trajectories.

---

## Storage Format

Each trajectory is a single `.npz` file plus `.json` metadata. **Log at 100 Hz** (the sim rate); subsample later if needed. Sample rate must be **the same as the world-model's \(dt\)** — do not let the model's internal time-scaling differ from the logged \(dt\).

| Field | Shape | Notes |
|---|---|---|
| `t` | \((T,)\) | Sim time in seconds |
| `position` | \((T, 3)\) | World frame |
| `orientation` | \((T, 4)\) | Quaternion \(w, x, y, z\) |
| `velocity` | \((T, 3)\) | World frame |
| `angular_velocity` | \((T, 3)\) | Body frame |
| `accel_imu` | \((T, 3)\) | Body frame — **required**, this is \(E_P\) input |
| `gyro_imu` | \((T, 3)\) | Body frame — **required** |
| `action` | \((T, 4)\) | Motor commands or rate commands |
| `wind_force_N` | \((T, 3)\) | Ground truth — **for OOD label and oracle only**, never as input |
| `mass`, `inertia`, `drag_coeff` | scalar/tuple | Physics parameters of the drone in this trajectory |
| `mode` | string | `hover` / `translation` / `accel` / `yaw` / `attitude` / `combined` |
| `wind_type` | string | `none` / `constant` / `step` / `sinusoidal` / `gust` / `dryden` |
| `wind_magnitude_N` | scalar | Scalar for OOD split |
| `wind_direction` | \((3,)\) | Unit vector for OOD split |
| `pair_id` | string | Groups counterfactual samples together |
| `seed` | int | Reproducibility |

---

## Sanity Checks Before Training

For every mode, before you commit the data to the world model:

| Check | Method | Pass condition |
|---|---|---|
| **State coverage** | Compute per-dimension std dev of \(s\) across all trajectories | Each of 13 state dims has std > 0 (no dimension is frozen) |
| **Residual magnitude** | Compute \(\bar{\|\Delta_t\|}\) where \(\Delta_t = s_{t+1} - f_{\text{phys}}(s_t, u_t)\) | Should grow linearly with \(F_w\) magnitude (not constant, not zero) |
| **Decorrelation** | Correlation \(\rho(u_t, F_w)\) across samples | \(\rho \in [-0.3, 0.3]\) (weak correlation only) |
| **Transient content** | Mean \(\|\dot{s}_t\|\) per mode | Nonzero in every mode (even hover should have a small \(\dot{s}\)) |
| **Counterfactual pairing** | For each `pair_id`, verify same \((s_t, u_t)\) sequence | Present in at least 20% of scripted segments |

If the **residual magnitude check** fails (residual constant or zero), then \(d_\theta\) is not learnable under that mode — go back and check whether the mode is being flown with compensation (Problem 1) or without action variation (Problem 2).

---

## The One-Sentence Summary

**Every mode must either force \(f_{\text{phys}}\) to be wrong in a way attributable to wind (free acceleration, free translation), or break the action-wind correlation (random actions, counterfactual pairs).** Hover is the only mode where wind is fully absorbed into \(u_t\), which is why it is 10% of the data, not 100%. If you generate the six modes at the volumes above, \(d_\theta\) becomes identifiable, decorrelated, and transient-rich — which is exactly what the world-model needs to generalize to OOD wind.