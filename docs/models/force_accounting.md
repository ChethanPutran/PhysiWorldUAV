# Force accounting (Run 4)

Which effect is computed and applied by whom, in which frame, and how it is recorded.
Every number below was measured in Isaac Sim 4.1 with the production pipeline
(`gpu_dynamics_cpu_readback`) and is stored in [`validation_run4.json`](validation_run4.json)
(`tests/isaac/test_physics_validation.py`).

## 1. Computed by the PhysX solver

| Effect | Notes |
|---|---|
| Gravity | Scene gravity (0, 0, −9.81) m/s², read back each run (`physics_settings().gravity_W`) |
| Rigid-body / articulation integration | TGS solver, dt = 0.005 s, 4 position / 0 velocity iterations (OmniDrones `RobotCfg`) |
| Rotor joints | Revolute joints about body z, free (drive stiffness/damping 0). They do **not** spin (joint velocity 0), so there is no rotor gyroscopic coupling. Their position drifts between episodes and is reset with the pose. |
| Linear / angular damping | **0** on all five bodies (decision D6; OmniDrones default was 0.2 s⁻¹). Verified: free fall a_z = −9.8100 m/s²; a 2 m/s coast is unchanged after 2 s. |
| Contacts | Ground plane only; episodes terminate at z < 0.1 m |

## 2. Computed and applied by OmniDrones (`MultirotorBase.apply_action`)

| Effect | Formula | Frame |
|---|---|---|
| Motor lag | throttle ← throttle + α (√((cmd+1)/2) − throttle), α = 1 − e^(−dt/τ_m) (bound per domain) | — |
| Rotor thrust | T_i = KF_i · throttle_i², KF_i = k_F Ω_max² | rotor link, +z |
| Reaction torque | τ_z = −Σ dir_i KM_i throttle_i² | base link, world |
| OmniDrones drag / downwash | drag_coef asserted **0** at every bind; downwash only with > 1 drone per environment (not used) | — |

Thrust and reaction torque are simulator-native and are **not** part of the project force models.

## 3. Computed by the project force models (`src/physiworld_uav/forces/`)

| Component | Equation | Inputs |
|---|---|---|
| `rotor_drag` | F_i,B = −k_rd Ω_i P⊥(R_WBᵀ(v − w_i) + ω_B × r_i), τ_B = Σ r_i × F_i | Ω at t_k, wind at each rotor hub |
| `body_drag` | F_B = −½ ρ C_D A(û) ‖v_rel‖ v_rel, τ_B = r_cop × F_B | wind at the CoM |
| `ground_effect` | ΔT_i = (k_GE(h_i) − 1) k_F Ω_i² along +z_B at rotor i | hub height |
| `aero_torque` | τ_B = −K₁ ω_B − k₂ ‖ω_B‖ ω_B (ablation only) | ω_B |
| `program` | open-loop `body_wrench` experiments only | program signal (body frame) |

Details, assumptions and validity envelopes: [aerodynamics.md](aerodynamics.md).

## 4. Applied to the simulator

* The sum of the components from table 3 is injected by `forces/injection.py::WrenchInjector`.
* The injector replaces `drone.base_link` and adds the external force and torque (world frame) into OmniDrones' **single** base-link call, `apply_forces_and_torques_at_pos(forces, torques, is_global=True)`.
* The wrench is set every physics step and cleared after use.
* **Application point: the base-link origin.** The composite CoM sits 0.50 mm above it (rotor links at z = +0.01 m), so a lateral force also produces a small torque. This is physical and recorded. At 1 N lateral it costs about 2 % of the lateral acceleration after 0.5 s (via tilt).
* **A10 measurement (two separate calls in one step):** in this pipeline, forces from a second call on the same view **accumulate** — a 1 N call made before OmniDrones' call gives the full 1/m acceleration. The injector does not rely on this; it is correct under overwrite semantics too.

## 5. Recorded

| Field | Content |
|---|---|
| `ground_truth/wrench/<component>_force_W`, `_torque_B` | Per component, interval k |
| `ground_truth/wrench/external_force_W`, `external_torque_B` | Exactly what the injector forwarded (read back from the injector, not recomputed). The validator checks it equals the component sum. |
| `ground_truth/rotor_thrust`, `rotor_speed` | Motor state at each instant |
| `observations/imu_*` | Isaac `IMUSensor`: specific force and gyro in the body frame |

## Resulting equations of motion

With m the total mass (0.716 kg nominal), J_eff the effective composite inertia, and all external project forces at the base-link origin:

  m v̇ = m g + R_WB Σ_i T_i e₃ + R_WB F_aero,B
  J_eff ω̇_B = Σ_i r_i × T_i e₃ + τ_react,B + τ_aero,B − ω_B × J_eff ω_B (+ small residual coupling, below)

Gravity enters once (PhysX). Thrust and reaction torque enter once (OmniDrones). Aerodynamics enter once (injector). PhysX damping and OmniDrones drag are zero.

## Validation (Isaac, Run 4)

| Check | Result |
|---|---|
| Hover at true-mass trim | a = 1e-6 m/s²; IMU 9.810 m/s² |
| Free fall | a_z = −9.8100 m/s²; IMU ≈ 0 (no hidden damping) |
| 1 N injected | a_y 1.3957 vs 1/m = 1.3966 m/s²; IMU 1.3964 |
| Motor step | throttle error vs first-order lag 6e-8 |
| Yaw / roll / pitch torque 0.01 N m | 0.7669 / 1.2971 / 1.3070 rad/s² vs τ/J_eff 0.7668 / 1.3025 / 1.3025 |
| Gyro vs ω_B = R_WBᵀ ω_W | equal to 1e-7 rad/s |
| IMU specific force vs R_WBᵀ(Δv/dt + g e₃) | ≤ 1e-2 m/s² every step (smoke episode) |
| Domain binding (m ×1.1, k_F ×0.95, τ_m ×1.2) | read-back equals plan; new trim hovers (a < 1e-3) |
| 3 m/s wind at hover | a_x 0.9614 vs model F/m 0.9618 m/s² |
| Repeatability within one process | bit-exact (after the rotor-joint reset) |
| Regeneration in a fresh process | bit-exact (Run 3) |

## Known residual (part of what the learned residual must absorb; decision Q4)

The roll-pitch cross-coupling is about 0.4 rad/s² when pitching at 1 rad/s with an 80 rad/s rotor-speed spread (yaw torque command). That is about 0.3 % of roll authority. It is zero with symmetric rotor speeds. It comes from the articulation with free rotor joints and is not included in the nominal f_phys.
