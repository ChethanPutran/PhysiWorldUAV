# Complete List of Forces and Torques to Implement in Your UAV Simulator

For your Hummingbird quadrotor, I recommend organizing the physics into translational forces, rotational torques, rotor aerodynamic effects, and environmental effects.

Below are the mathematical formulations, variables and implementation priorities.

## 1. Translational forces

These forces directly affect the drone's linear acceleration.

| No. | Force                           | Mathematical formula                                                                  | Priority    |
| --- | ------------------------------- | ------------------------------------------------------------------------------------- | ----------- |
| 1   | Rotor thrust                    | \\(\displaystyle \mathbf F_T=R\_{WB}\sum_i T_i\mathbf e_T\\)                          | Essential   |
| 2   | Gravity                         | \\(\displaystyle \mathbf F_g=m\mathbf g\\)                                            | Essential   |
| 3   | Aerodynamic drag                | \\(\displaystyle \mathbf F_D=-\frac12\rho C_DA\|\mathbf v_r\|\mathbf v_r\\)           | Essential   |
| 4   | Surface pressure                | \\(\displaystyle \mathbf F_p=-\int_S(p-p\_\infty)\mathbf n\\,dA\\)                    | Important   |
| 5   | Surface shear                   | \\(\displaystyle \mathbf F_s=\int_S\boldsymbol\tau_w\\,dA\\)                          | Important   |
| 6   | Added mass                      | \\(\displaystyle \mathbf F_A=-\mathbf M_A\dot{\mathbf v}\_r\\)                        | Advanced    |
| 7   | Unsteady aerodynamic force      | \\(\displaystyle \mathbf F_U=\mathbf F\_{\text{steady}}+\mathbf F\_{\text{memory}}\\) | Advanced    |
| 8   | Ground-effect thrust correction | \\(\displaystyle \mathbf F\_{GE}=(\lambda\_{GE}-1)\mathbf F_T\\)                      | Conditional |

Here:

\\[ \mathbf v_r=\mathbf v\_{\text{drone}}-\mathbf u\_{\text{wind}} \\]

is the relative velocity of the drone with respect to the surrounding air.

### Additional details

1\. Rotor thrust

For each rotor:

\\[ T_i=k_T\Omega_i^2 \\]

For a basic model, this is sufficient. For better aerodynamic realism, replace it with a rotor inflow model.

2\. Aerodynamic drag

For body-frame directional drag:

\\[ \mathbf F\_{D,B}= -\frac12\rho \begin{bmatrix} C\_{Dx}A_x|v\_{r,x}|v\_{r,x}\\\ C\_{Dy}A_y|v\_{r,y}|v\_{r,y}\\\ C\_{Dz}A_z|v\_{r,z}|v\_{r,z} \end{bmatrix} \\]

Then transform into world coordinates:

\\[ \mathbf F\_{D,W}=R\_{WB}\mathbf F\_{D,B} \\]

3\. Surface pressure

For each mesh triangle:

\\[ \mathbf F\_{p,j}=-\Delta p_j A_j\mathbf n_j \\]

where:

\\[ \Delta p_j=C\_{p,j}\frac12\rho|\mathbf v\_{r,j}|^2 \\]

Unlike a windward-only model, use signed pressure coefficients to represent both positive pressure and leeward suction. The coefficients should be calibrated or obtained from aerodynamic data.

4\. Surface shear

\\[ \mathbf v\_{t,j}= \mathbf v\_{r,j}- (\mathbf v\_{r,j}\cdot\mathbf n_j)\mathbf n_j \\]

A simplified implementation:

\\[ \mathbf F\_{s,j}= -\frac12\rho C\_{f,j}A_j |\mathbf v\_{t,j}|\mathbf v\_{t,j} \\]

This assumes the relative velocity is drone velocity minus air velocity.

5\. Added mass

\\[ \mathbf F_A=-\mathbf M_A\dot{\mathbf v}\_r \\]

For a spatially varying wind field:

\\[ \dot{\mathbf v}\_r= \dot{\mathbf v}\_{d} - \left( \frac{\partial\mathbf u}{\partial t} + (\mathbf v_d\cdot\nabla)\mathbf u \right) \\]

This is the relative acceleration along the drone trajectory. In an actual coupled added-mass model, additional terms can arise from changing geometry and fluid coupling.

## 2. Rotational torques

These forces generate moments around the drone's center of gravity.

| No. | Torque                           | Mathematical formula                                                                   | Priority    |
| --- | -------------------------------- | -------------------------------------------------------------------------------------- | ----------- |
| 1   | Rotor differential thrust torque | \\(\displaystyle \boldsymbol\tau_T=\sum_i\mathbf r_i\times\mathbf F_i\\)               | Essential   |
| 2   | Rotor reaction torque            | \\(\displaystyle \boldsymbol\tau_Q=-\sum_i s_iQ_i\mathbf e_T\\)                        | Essential   |
| 3   | Aerodynamic pressure torque      | \\(\displaystyle \boldsymbol\tau_p=\sum_j\mathbf r_j\times\mathbf F\_{p,j}\\)          | Important   |
| 4   | Aerodynamic shear torque         | \\(\displaystyle \boldsymbol\tau_s=\sum_j\mathbf r_j\times\mathbf F\_{s,j}\\)          | Important   |
| 5   | Added rotational inertia torque  | \\(\displaystyle \boldsymbol\tau_A=-\mathbf J_A\dot{\boldsymbol\omega}\\)              | Advanced    |
| 6   | Unsteady aerodynamic torque      | \\(\displaystyle \boldsymbol\tau_U=\boldsymbol\tau\_{\text{memory}}\\)                 | Advanced    |
| 7   | Rotor gyroscopic torque          | \\(\displaystyle \boldsymbol\tau\_{\text{gyro}}=-\boldsymbol\omega\times\mathbf H_r\\) | Advanced    |
| 8   | Ground-effect rotor moment       | Model-dependent                                                                        | Conditional |

### Additional details

1\. Rotor differential thrust torque

\\[ \boldsymbol\tau_T=\sum\_{i=1}^{4}\mathbf r_i\times(T_i\mathbf e_T) \\]

This generates roll and pitch.

2\. Rotor reaction torque

\\[ Q_i=k_Q\Omega_i^2 \\]

\\[ \boldsymbol\tau_Q=-\sum_i s_iQ_i\mathbf e_T \\]

This generates yaw.

3\. Aerodynamic torque

For surface forces:

\\[ \boxed{ \boldsymbol\tau\_{\text{aero}}= \sum_j (\mathbf r_j-\mathbf r\_{CG}) \times (\mathbf F\_{p,j}+\mathbf F\_{s,j}) } \\]

This is essential if you want wind to affect the drone's attitude, not just its position.

4\. Added rotational inertia

\\[ \boldsymbol\tau_A=-\mathbf J_A\dot{\boldsymbol\omega} \\]

This accounts for the surrounding air's resistance to angular acceleration. It requires a geometry-dependent added-inertia model.

5\. Rotor gyroscopic torque

For rotor angular momentum:

\\[ \mathbf H_r=\sum_i s_i I\_{r,i}\Omega_i\mathbf e_T \\]

The reaction on the airframe from changing rotor angular momentum is:

\\[ \boldsymbol\tau\_{\text{gyro}}=-\boldsymbol\omega\times\mathbf H_r \\]

This is distinct from the rigid-body gyroscopic coupling already included in your Newton–Euler equation.

## 3. Rotor aerodynamic models to implement

These are not necessarily additional external forces. They are mechanisms that improve the calculation of rotor thrust and torque.

| Model                | Formula / relationship                                                                 | Purpose                             |
| -------------------- | -------------------------------------------------------------------------------------- | ----------------------------------- |
| Momentum theory      | \\(T=2\rho A_rv_i^2\\) in hover                                                        | Calculate induced velocity          |
| Dynamic inflow       | \\(\dot v_i=(v\_{i,\text{eq}}-v_i)/\tau_i\\)                                           | Capture inflow lag                  |
| Wind-rotor coupling  | \\(T=f(\Omega,V\_{\text{axial}},V\_{\perp})\\)                                         | Account for wind-dependent thrust   |
| Blade Element Theory | \\(dL=\frac12\rho V\_{\text{local}}^2cC_Ldr\\)                                         | Calculate blade aerodynamic loading |
| Rotor interference   | \\(\mathbf u_i=\mathbf u\_{\text{ambient}}+\sum\_{j\ne i}\mathbf u\_{\text{wake},j}\\) | Account for neighboring rotor wakes |
| Blade flapping       | \\(\beta(\psi)=\beta_0+\beta\_{1c}\cos\psi+\beta\_{1s}\sin\psi\\)                      | Account for rotor-induced moments   |

I would initially use momentum theory with dynamic inflow and wind-dependent rotor loading rather than implementing full BEMT immediately.

## 4. Complete Newton–Euler equations for your simulator

Once the force and torque models are implemented, your governing equations become:

Translational dynamics

\\[ \boxed{ m\dot{\mathbf v}\_W= \mathbf F_T+ \mathbf F_g+ \mathbf F_D+ \mathbf F_p+ \mathbf F_s+ \mathbf F_A+ \mathbf F_U } \\]

Rotational dynamics

\\[ \boxed{ \mathbf J\dot{\boldsymbol\omega}\_B= \boldsymbol\tau_T+ \boldsymbol\tau_Q+ \boldsymbol\tau_p+ \boldsymbol\tau_s+ \boldsymbol\tau_A+ \boldsymbol\tau_U+ \boldsymbol\tau\_{\text{gyro}} - \boldsymbol\omega_B\times (\mathbf J\boldsymbol\omega_B) } \\]

These equations assume that the individual terms have been defined consistently and that no aerodynamic effect is counted twice.

## 5. My recommended implementation order

For your UAV world-model thesis, I would not implement all these effects simultaneously.

## Implementation roadmap

1. Phase 1 — Fundamental dynamics

   Must implement
   - Gravity
   - Rotor thrust
   - Rotor differential thrust torque
   - Rotor reaction torque
   - Relative-wind quadratic drag
   - Rigid-body gyroscopic coupling
   - Accurate numerical integration
2. Phase 2 — Body aerodynamics

   High priority
   - Triangle-based pressure forces
   - Leeward suction
   - Surface shear
   - Aerodynamic moments about CG
3. Phase 3 — Rotor aerodynamics

   Research improvement
   - Wind-dependent rotor thrust
   - Dynamic inflow
   - Rotor wake interaction
   - Rotor gyroscopic effects
4. Phase 4 — Transient effects

   Advanced
   - Added mass
   - Unsteady aerodynamic response
   - Ground effect
   - Flow separation
5. Phase 5 — Validation
   - Compare force against analytical calculations.
   - Compare rotor thrust against experimental or manufacturer data.
   - Validate wind response at different orientations.
   - Check numerical convergence.
   - Evaluate hover, translation and trajectory-following modes.

## 6. Important implementation warning

Do not add every item in the tables as an independent force.

For example:

- Lumped drag already approximates pressure and shear effects.
- Surface pressure and shear replace that approximation if you use the mesh-based model.
- Rotor inflow modifies rotor thrust; it is not an independent body force.
- Ground effect modifies rotor performance.
- Gyroscopic coupling is already part of rigid-body rotational dynamics.
- Added mass must be incorporated consistently into the effective inertia equations.

My recommendation: Use a modular force architecture. Each physical model should return its force and torque separately, allowing you to enable, disable and validate each contribution independently.

That will also allow you to generate ablation datasets and determine which physical mechanisms actually improve your world model's OOD generalization.