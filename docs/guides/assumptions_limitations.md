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
