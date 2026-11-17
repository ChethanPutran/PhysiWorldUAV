# Aerodynamic models (Run 4) — formulation, assumptions, validity

Implementation: `src/physiworld_uav/forces/models.py` (equations), `system.py`
(assembly, domain scales), `registry.py` (validity envelopes, evaluated every
physics step). Configs: `config/forces/*.yaml`.

**Frames.** World ENU, body FLU, quaternion wxyz body→world. Forces are returned in the body frame and applied at the base-link origin. Rotor hubs sit at r_i = (L cos θ_i, L sin θ_i, 0.01 m).

**Relative air velocity.** Always v − w, never ground-relative. Body drag uses the wind at the CoM. Rotor drag uses the wind at each hub plus the rotation term ω_B × r_i.

Decision D5: the simulator's "true" aerodynamics are `rotor_drag` + `body_drag`. `ground_effect` and `aero_torque` exist for ablations. The nominal physics prior f_phys contains none of them.

---

## 1. Rotor-induced drag (`rotor_drag`)

| Item | Content |
|---|---|
| Equation | F_i,B = −k_rd · Ω_i · P⊥ v_rel,i,B, with v_rel,i,B = R_WBᵀ(v_W − w(p_i)) + ω_B × r_i and P⊥ = diag(1, 1, 0); τ_B = Σ r_i × F_i |
| Physics | Blade flapping + induced drag of stiff propellers: a horizontal force ∝ rotor speed × in-plane airspeed. At hover it reduces to the D-matrix model F = −m D v⊥ with d = k_rd ΣΩ_hover / m (Kai et al. 2017; Faessler et al. 2018) |
| Parameters | k_rd [N s/(m·rad/s)], set from the mass-normalised hover coefficient `d_hover_per_s`; domain factor `rotor_drag_scale` |
| Cost | O(R) per environment per step |
| Envelope | advance ratio μ = ‖v⊥‖/(Ω R) ≤ 0.20 (widened from 0.15 in Run 5; fidelity reduced for 0.15–0.20); Ω ≥ 0.2 Ω_max |
| Applicable | hover, translation ≤ ~5 m/s, moderate attitudes, wind ≤ 8 m/s |
| Fails | high advance ratio (translational lift, nonlinear); windmilling; strong axial flow (vortex ring); no axial drag (d_z = 0, as identified by Faessler et al.) |
| Validation | CPU: linear in v and Ω, zero axial force, uniform-flow torque = hub height × force, yaw damping sign, d at hover reproduced exactly. Isaac: 3 m/s wind acceleration equals model F/m (0.1 %). |

## 2. Body parasitic drag (`body_drag`)

| Item | Content |
|---|---|
| Equation | F_B = −½ ρ (C_D A)(û) ‖v_rel‖ v_rel, with (C_D A)(û) = C_D,x A_x‖û_x‖ + C_D,y A_y‖û_y‖ + C_D,z A_z‖û_z‖ and v_rel = R_WBᵀ(v_W − w); τ_B = r_cop × F_B |
| Physics | Quasi-steady bluff-body drag, anti-parallel to the relative flow, with an orientation-dependent projected area. Replaces the old component-wise ‖v_i‖v_i form, which was not direction-consistent. |
| Parameters | per-axis C_D (scalar also accepted), projected areas (A_x, A_y, A_z), ρ, CoP offset; domain factor `body_drag_scale` |
| Cost | O(1) |
| Envelope | airspeed ≤ 12 m/s |
| Fails | lift/side forces from asymmetric geometry; separation hysteresis; very large incidence changes |
| Validation | CPU: anti-parallel, ∝ v², rotation equivariance, exact value, terminal velocity of a falling body. |
| Diagonal vs full matrix | A symmetric "+" frame with no lift/side force needs no cross-axis terms. The orientation dependence enters through A(û). A full aerodynamic matrix would be needed only for asymmetric bodies or lifting surfaces. |

## 3. Ground effect (`ground_effect`)

| Item | Content |
|---|---|
| Equation | ΔT_i = (k_GE − 1) k_F Ω_i² along +z_B at rotor i, with k_GE = 1/(1 − (R/4h_i)²) |
| Assumptions | Cheeseman–Bennett image rotor (constant-power derivation, used at fixed rotor speed); flat ground at z = 0 |
| Envelope | h/R ≥ 0.5 (clamped below) |
| Note | At normal flight heights (≥ 0.5 m) the factor is < 1.003, so it is off in `truth_default` |

## 4. Rotational damping (`aero_torque`, ablation only)

* **Equation:** τ_B = −K₁ ω_B − k₂ ‖ω_B‖ ω_B. The scalar k₂ keeps the quadratic term frame-invariant.
* **Status:** it double counts the damping that rotor drag already produces through ω × r, so it is used only in `full_ablation`.

## 5. Not modelled (documented limitations)

* Translational lift and axial-inflow changes of thrust (wind-on-thrust). The previous λ_wind/λ_DL factors were physically inverted (Glauert's induced-velocity ratio used as a thrust ratio; sign-blind climb/descent) and were removed.
* Unsteady aerodynamics, rotor-rotor interference, vortex-ring state, added mass, blade flexibility.

---

## Proposed coefficients (decision Q3 — awaiting approval)

| Coefficient | Proposed | Basis |
|---|---|---|
| Rotor radius R | 0.10 m | Hummingbird propeller mesh (`sim/stl_parts`), bounds ±0.1001 m; the old config said 0.08 m |
| Body projected areas A_x, A_y, A_z | 0.0070, 0.0076, 0.0180 m² | Rasterised union projection of the base-link visual mesh (frame, body, motors, landing gear), 0.5 mm grid. The old placeholders 0.08–0.12 m² were ~10× too large. |
| Body C_D (per axis x, y, z) | 1.1, 1.1, 1.3 | Bluff-body range 1.0–1.2 (Hoerner, *Fluid-Dynamic Drag*, 1965); z higher for the flat-plate-like face seen from above/below. **z value is an engineering assumption, not measured** |
| Rotor drag d (mass-normalised, at hover) | 0.30 s⁻¹ in x and y; 0 in z | See the derivation below |
| Air density ρ | 1.225 kg/m³ | ISA sea level, 15 °C |

**Rotor-drag coefficient d (0.30 s⁻¹, so k_rd = 1.18e-4 N s/(m·rad/s)):**
* **Source:** Faessler, Franchi & Scaramuzza (RA-L 2018) identified d_x = 0.425 s⁻¹ and d_y = 0.256 s⁻¹ at 2.8 m/s, on a 0.61 kg quadrotor with stiff 6-inch propellers. The values ranged 0.25–0.55 s⁻¹ during identification, and they found d_z adds nothing.
* **Choice:** 0.30 s⁻¹ is mid-range and isotropic, which suits the symmetric "+" frame.
* **Uncertainty:** different vehicle and propellers; their values also lump body drag in.

**Resulting magnitudes at hover attitude:**

| Relative wind | Rotor drag | Body drag | Total | Equilibrium tilt |
|---|---|---|---|---|
| 3 m/s | 0.644 N | 0.044 N | 0.69 N | 5.6° |
| 4 m/s (ID limit) | 0.86 N | 0.079 N | 0.94 N | 7.6° |
| 8 m/s (OOD limit) | 1.72 N | 0.31 N | 2.03 N | 16.1° |

The tilt limit is 35° (feasibility) and the lateral authority at 0.5 rad is 3.8 N, so the OOD regime stays flyable.

**Sources:**
* [Faessler et al., RA-L 2018, arXiv:1712.02402](https://arxiv.org/abs/1712.02402)
* Kai, Allibert, Hua & Hamel, "Nonlinear feedback control of quadrotors exploiting first-order drag effects", IFAC 2017
* Cheeseman & Bennett, ARC R&M 3021 (1955)
* Hoerner, *Fluid-Dynamic Drag* (1965)
