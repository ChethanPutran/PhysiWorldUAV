# Drone model, sensor and wind models (Run 4)

## Hummingbird (`config/drone/hummingbird.yaml`, `drones/params.py`)

| Quantity | Value | Source / check |
|---|---|---|
| Total mass | 0.716 kg = base 0.68 + 4 × 0.009 | USD; PhysX read-back |
| Base-link inertia (OmniDrones semantics) | diag(0.007, 0.007, 0.012) kg m² | yaml = USD |
| Effective composite inertia J_eff | diag(0.007677, 0.007677, 0.013040) kg m² | Parallel axis; rotor links as axisymmetric equivalents, spin inertia excluded about z. Measured in Isaac: 0.00771 / 0.00768 / 0.01304 |
| Rotors | "+" layout, rotor 0 on +x_B, L = 0.17 m, hub height 0.01 m, R = 0.10 m | yaml, USD, mesh |
| k_F, k_M | 8.54858e-6 N/(rad/s)², 1.36777e-7 N m/(rad/s)² | yaml = OmniDrones |
| Ω_max, T_max, T/W | 838 rad/s, 24.01 N, 3.42 | derived |
| Hover | Ω = 453.2 rad/s, throttle 0.541, command −0.415 | derived; Lee and Isaac hover agree |
| Motor lag | OmniDrones blend α = 0.43 at dt 0.005 s ⇔ τ_m = 8.9 ms | domain binding writes α(τ_m, dt) |

**Domain binding** (`drones/sim_binding.py`) writes the following, then reads all of them back and hard-checks them against the plan (`check_against_plan`):
* base-link mass and inertia;
* `KF`, `KM`, and the motor α to `tau_up`/`tau_down`;
* an assertion that OmniDrones' drag_coef is 0.

**Controllers** (policy "true", Q2) use the read-back total mass and J_eff.

## IMU (`core/simulation_interface.py`; decision (a))

* **Sensor:** Isaac `omni.isaac.sensor.IMUSensor` on the base link (sensor frame = body frame). It reports specific force (`read_gravity=True`) and angular rate.
* **Run 4 fixes:**
  1. The sensor prim is now created **before** `sim.reset()`. Created after it, the plugin never registered the body: `is_valid` stayed false on CPU and GPU, and every frame stayed at physics_step 0.
  2. The physics pipeline is now **GPU dynamics + GPU broadphase with CPU readback** (`gpu_dynamics_cpu_readback`). With the default no-readback GPU pipeline the sensor returns gravity-only frames, (0, 0, 9.81) even in free fall. The CLI rejects that pipeline.
* **Validity:** a frame counts only if the physics step advanced since the last read (freshness gate).
* **Verified:** hover 9.810, free fall 0.000; specific force equals R_WBᵀ(Δv/dt + g e₃) from the logged state within 1e-2 m/s² every step; gyro equals ω_B within 1e-7.
* **Noise:** none at generation. Clean measurements are stored and noise is applied at load time (D7).
* **Throughput:** about 72 physics steps/s for one environment (probe), versus 81 for pure CPU.

## Wind (`wind/`)

| Component | Model | Run 4 status |
|---|---|---|
| Steady | constant vector | — |
| 1-cos gust, step (with rise time), sinusoid | deterministic shapes | step rise time added (W1) |
| Ornstein–Uhlenbeck | exact AR(1) | stationary from t = 0; anisotropic σ = (σ_h, σ_h, 0.5 σ_h) (W2) |
| Random gust events | Poisson 1-cos pulses | seeded per component |
| Dryden (MIL-F-8785C) | exact Van Loan discretisation, closed-form stationary variance, stationary initialisation | rewritten (W3). Measured: per-axis σ within 5 % from t = 0 for L = 2.5–200 m; PSD/analytic median 0.98 |
| Spatial gradient | G (p − p₀) | evaluated per rotor hub |

* **Seeds (W4):** each component draws its own child seed from the episode wind seed and its stable component name. Siblings with the same seed therefore share realisations, and components are independent.
* **Temporal scaling (W5):** applies to every component, including OU, Dryden and events. A scale s shrinks every correlation time by s.
* **Dryden low altitude:** L_w = h, L_u = L_v = h/(0.177 + 0.000823 h)^1.2, σ_u/σ_w = 1/(0.177 + 0.000823 h)^0.4, σ_w = 0.1 W₂₀, with h in ft clamped to ≥ 10 ft (an extrapolation at 1–3 m).
* **Frozen-turbulence speed:** V = max(mean horizontal wind, 1 m/s), fixed per episode.
  * This deviates from the plan, which had V tracking the vehicle's relative airspeed.
  * Reason: wind realisations must not depend on the trajectory, or counterfactual pairs and the "logged wind = planned wind" validation would break.
* **Spatial evaluation:** stateful components are spatially uniform and evaluated **once** per instant at the CoM. The gradient term adds G·(p_hub − p_CoM) per rotor.
* **Regimes (D1, D2):** `config/wind/regimes/*` with realised-statistics guarantees (Run 3).
