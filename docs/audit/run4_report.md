# Run 4 report — physical models implemented and validated

Date: 2026-11-16 · Plan: [run2_implementation_plan.md](run2_implementation_plan.md) · IMU decision: **(a)** GPU dynamics with CPU readback.

## Exit criterion

| Criterion | Result |
|---|---|
| Every physics-validation test passes inside Isaac Sim with recorded numbers | **Yes.** `tests/isaac`: **17 passed** (12 physics validation + 5 smoke). Numbers in [`docs/models/validation_run4.json`](../models/validation_run4.json). |
| CPU suite | **131 passed** (17 Isaac tests skipped without `--isaac`) |
| Smoke dataset | Now **`complete = true`** (Run 3 blocker resolved) |

## IMU (decision (a))

Two separate defects were found and fixed:

1. **Creation order.** Created after `sim.reset()`, the IMU prim was never registered by the sensor plugin (`is_valid` stayed false) on **every** pipeline, CPU included. It is now created before the reset and initialised after.
2. **Readback.** On Isaac's default GPU pipeline (no readback) the sensor then reports gravity-only frames: (0, 0, 9.81) even in free fall. Option (a) runs PhysX GPU dynamics and GPU broadphase with CPU readback (`physics_pipeline: gpu_dynamics_cpu_readback`). The CLI rejects the `gpu` pipeline.

**Result:**
* hover 9.810 m/s² and free fall 0.000;
* specific force equals R_WBᵀ(Δv/dt + g e₃) from the logged state within 1e-2 every step;
* gyro equals ω_B within 1e-7;
* throughput ≈ 72 steps/s (1 env), against 81 for pure CPU and 73 for the broken GPU pipeline.

## Implemented (plan IDs)

| Area | What | IDs |
|---|---|---|
| Pipeline settings | `SimulationConfig.physics_pipeline`, enhanced determinism, settings read back into every episode's `sim` metadata | V4 |
| Damping | PhysX linear/angular damping 0 on all bodies, set before spawn and read back | A6, D6 |
| Force injection | `forces/injection.py::WrenchInjector` adds the project wrench into OmniDrones' single base-link call. The old `hasattr` no-op path and `apply_external_force` were removed. | A8, F1, S3 |
| Second-call semantics | Measured: **accumulate** in this pipeline (the injector is correct either way) | A10 |
| Domain binding | `drones/sim_binding.py` writes base mass/inertia, KF/KM and motor α; reads them back; hard-checks against the plan; computes the effective composite inertia | R1, R4, A4, A5, A9 |
| Controllers | Built from read-back total mass and effective inertia (Q2) | A9 |
| Reset | Rotor joint positions/velocities reset (determinism), motors at true-mass trim | A13 |
| Frames | ω_B everywhere (done in Run 3); force models take `omega_B`; torques rotated to world once | A11, F3 |
| Force models | New `forces/models.py` + `system.py`:<br>• linear rotor drag ∝ Ω (in-plane, per-rotor wind, ω×r);<br>• body drag with orientation-dependent projected area;<br>• Cheeseman–Bennett ground effect per rotor;<br>• frame-invariant rotational damping (ablation only).<br>Old models (wrong geometry, isotropic quadratic "rotor drag", inverted λ_wind/λ_DL) moved to `lagacy/src_superseded/forces/`. | F2, F4–F8 |
| Logging | Per-component wrench + exactly-applied totals; validator checks total = Σ components | F9 |
| Domain drag scales | `rotor_drag_scale` / `body_drag_scale` applied per episode | F10 |
| Applicability registry | `forces/registry.py`; envelopes evaluated every step; per-episode violation fractions; out-of-envelope episodes excluded from model-ready data | F11, req. #3 |
| Wind | OU stationary + anisotropic (W2); Dryden rewritten with exact discretisation, closed-form variance, stationary init, low-altitude MIL-F-8785C, mean-wind frame (W3); temporal scale for stateful components (W5); per-rotor wind with spatial gradient | W2, W3, W5 |
| Body-wrench open loop | Runs, with rotors at trim and the program wrench injected and logged | T11 |
| Guards | Runner refuses force configs not `validated` (dev flag `--allow-unvalidated-forces`; such datasets never validate) | — |
| Docs | `docs/models/{force_accounting, aerodynamics, drone_and_wind}.md`, `validation_run4.json`; `docs/guides/assumptions_limitations.md`; architecture guide updated | §11 |

## Validation results (Isaac, production pipeline)

| Check | Measured | Expected |
|---|---|---|
| Hover at true-mass trim | a = 1e-6 m/s², IMU 9.810 | 0, 9.81 |
| Free fall | a_z −9.8100, IMU ≈ 0 | −9.81, 0 |
| Coast 2 m/s for 2 s (damping 0) | 2.0 → 2.0 | unchanged |
| 1 N injected | a_y 1.3957 (IMU 1.3964) | 1.3966 |
| Motor step | max error 6e-8 | first-order lag α = 0.43 |
| Yaw / roll / pitch torque 0.01 N m | 0.7669 / 1.2971 / 1.3070 rad/s² | 0.7668 / 1.3025 / 1.3025 |
| Gyro vs ω_B | 1e-7 | 0 |
| Domain binding (m ×1.1, k_F ×0.95, τ ×1.2) | read-back = plan; hovers at new trim | — |
| 3 m/s wind, hover attitude | a_x 0.9614 m/s² | model F/m 0.9618 |
| Repeatability within one process | bit-exact | — |

## Development run with physics (37 diverse pilot episodes, `physiworld_pilot_v0.4.0-dev`)

* **Coverage:** every part (ID closed/open loop, OOD magnitude/direction/temporal/gust/physics/combination/trajectory), 15 scenario templates, Mode A and Mode B counterfactual groups.
* **Outcomes:**
  * all 37 completed;
  * IMU 100 % valid;
  * no actuator saturation, no terminations;
  * max tilt ≤ 36° (OOD 8 m/s wind);
  * ID tracking error ≤ 0.4 m.
  * Mode B replays drift with wind as expected: tilt 3° → 8° → 16° for 0 / 0.5 / 1.0 × wind.
* **Validation:** the only hard finding is the deliberate "unvalidated force config". Attitude-mode reference fields were fixed: the hold point is logged as the position reference, and control diagnostics may be NaN ("not commanded").
* **Throughput with physics:** ≈ 61 physics steps/s per environment (≈ 0.3× real time).

## Deviations from the plan

* **Dryden frozen-turbulence speed:** V = mean horizontal wind (fixed per episode) instead of the vehicle's relative airspeed, so realisations are trajectory-independent (counterfactual pairing; logged wind = planned wind).
* **Inertia semantics:** the domain samples the *base-link* inertia (OmniDrones semantics), and the composite inertia is derived from PhysX read-back. The plan said "solve the base inertia for a sampled composite"; this is simpler, and both are recorded.
* **Rotor gyroscopic coupling (A7):** nothing to model. The rotor joints don't spin in this OmniDrones setup.

## Decisions needed before Run 5

1. **Q3 — approve the drag coefficients.** Details and sources in [aerodynamics.md](../models/aerodynamics.md).
   * **Values:**
     * rotor drag d = 0.30 s⁻¹ (k_rd = 1.18e-4 N s/(m·rad/s));
     * body C_D = 1.1, areas from the mesh (0.0070, 0.0076, 0.0180 m²);
     * rotor radius 0.10 m;
     * ρ = 1.225 kg/m³.
   * **Resulting drag:** 0.94 N at 4 m/s and 2.03 N at 8 m/s relative wind.
   * On approval, `truth_default` becomes `validated: true` and the runner accepts it.
2. **Rotor-drag validity at OOD wind.**
   * **Problem:** at hover rotor speed the linear rotor-drag model is outside its envelope (μ ≤ 0.15) above ≈ 6.8 m/s relative airspeed. In the dev run, 3 of 4 OOD-magnitude episodes (7.4–8.2 m/s) were flagged and would be excluded from model-ready data.
   * **Options:**
     * (a) Widen the envelope to μ ≤ 0.2 (≈ 9 m/s) and document reduced fidelity in 0.15–0.2. Keeps the thesis OOD range (4, 8] m/s.
     * (b) Keep μ ≤ 0.15 and cap OOD magnitude at ≈ 6.5 m/s.
   * **Recommendation:** (a). The simulator is internally consistent either way; what changes is how physically realistic the high-wind ground truth is, and that becomes a stated limitation.
