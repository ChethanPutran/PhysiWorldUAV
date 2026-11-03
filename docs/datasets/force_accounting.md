# Complete Mathematical Explanation of Forces Acting on Your UAV

Your simulator models a quadrotor flying through a wind field. To understand whether its physics is realistic, you need to understand where every force comes from, how it is calculated, in which direction it acts, and how it changes the drone's motion.

I will explain the forces in your uploaded simulator audit, starting from fundamental Newton–Euler dynamics, then moving to rotor aerodynamics, body aerodynamics, and the physical effects currently missing from your simulator.

One important distinction:

- Forces change the drone's translational motion.
- Torques (moments) change its rotational motion.
- Aerodynamic effects can influence both translation and rotation.

# 1. Fundamental equation governing your drone

A quadrotor is a rigid body moving through three-dimensional space.

According to Newton's second law:

\\[ \boxed{m\dot{\mathbf v}=\sum_i\mathbf F_i} \\]

Where:

| Symbol                | Meaning                   | Unit |
| --------------------- | ------------------------- | ---- |
| \\(m\\)               | Drone mass                | kg   |
| \\(\mathbf v\\)       | Linear velocity           | m/s  |
| \\(\dot{\mathbf v}\\) | Linear acceleration       | m/s² |
| \\(\mathbf F_i\\)     | Individual external force | N    |

For your drone, the translational equation is:

\\[ \boxed{ m\dot{\mathbf v}\_W = \mathbf F\_{\text{thrust}} +\mathbf F\_{\text{gravity}} +\mathbf F\_{\text{aero}} } \\]

Here, \\(W\\) denotes the world coordinate frame.

The aerodynamic force can contain several contributions:

\\[ \mathbf F\_{\text{aero}} = \mathbf F\_{\text{drag}} +\mathbf F\_{\text{pressure}} +\mathbf F\_{\text{shear}} +\mathbf F\_{\text{added mass}} +\cdots \\]

These terms should not automatically be added together: some are alternative aerodynamic approximations. For example, your simulator uses either lumped drag or the surface pressure/shear model.

The rotational equation is:

\\[ \boxed{ \mathbf J\dot{\boldsymbol\omega}\_B = \boldsymbol\tau_B- \boldsymbol\omega_B\times (\mathbf J\boldsymbol\omega_B) } \\]

Where:

- \\(\mathbf J\\): moment of inertia matrix, kg·m².
- \\(\boldsymbol\omega_B\\): angular velocity in body coordinates, rad/s.
- \\(\boldsymbol\tau_B\\): total torque, N·m.
- \\(\boldsymbol\omega\times(\mathbf J\boldsymbol\omega)\\): gyroscopic coupling.

This equation tells us that forces determine translation, while torques determine rotation.

# 2. Rotor thrust force — the force that keeps the drone flying

[006 드론 비행 원리 그림 - 바이브 코딩(Vibe Coding): 모의 드론 레이싱 게임 개발](https://images.openai.com/static-rsc-4/GkLvcKq6Ql-3JSCM1bEWlNERXNu5dxve3oqfvcPjdWYQHI5WhTYwoDh3AfCmZQqOcHfUfHoJPxsXFL6tNYKZui8GxNtpAWIToOFPgz_qV6DBe3F4SUoAGTJglA8wqwAHMmQgPeA_5YtXKgm91eNYB_6SSGoHHVEJLsG_Y50xbSs?purpose=inline)

[wikidocs.net](https://wikidocs.net/280531?utm_source=chatgpt.com)

## 2.1 Where does thrust come from?

A propeller consists of rotating blades that accelerate air downward.

By Newton's third law:

- The propeller pushes air downward.
- The air exerts an equal and opposite force upward on the propeller and drone.

This upward force is called thrust.

For a quadrotor with four rotors:

\\[ \boxed{ \mathbf F_T=\sum\_{i=1}^{4}\mathbf F\_{T_i} } \\]

The thrust produced by an individual rotor is commonly approximated by:

\\[ \boxed{T_i=k_T\Omega_i^2} \\]

Where:

| Symbol         | Meaning                   | Unit      |
| -------------- | ------------------------- | --------- |
| \\(T_i\\)      | Thrust from rotor \\(i\\) | N         |
| \\(k_T\\)      | Rotor thrust coefficient  | N·s²/rad² |
| \\(\Omega_i\\) | Rotor angular speed       | rad/s     |

### Why is angular velocity squared?

The propeller's aerodynamic force depends approximately on dynamic pressure:

\\[ q=\frac12\rho V^2 \\]

The characteristic blade velocity is proportional to rotational speed:

\\[ V\_{\text{blade}}\propto\Omega \\]

Therefore:

\\[ T\propto \rho\Omega^2 \\]

This is why the simplified rotor model uses a quadratic relationship.

## 2.2 Significance of thrust

Thrust is responsible for:

1. Counteracting gravity during hovering.
2. Producing upward acceleration.
3. Producing horizontal acceleration when the drone tilts.
4. Controlling altitude and trajectory.

For hovering:

\\[ T\_{\text{total}}=mg \\]

For your 0.68 kg drone:

\\[ mg=0.68(9.81)=6.67\text{ N} \\]

Therefore, each rotor produces approximately:

\\[ T_i=\frac{6.67}{4}=1.67\text{ N} \\]

assuming symmetric hover.

## 2.3 What happens when wind is present?

This is especially important for your thesis.

The simplified equation:

\\[ T_i=k_T\Omega_i^2 \\]

does not explicitly account for wind-induced changes in rotor inflow.

The rotor actually operates in air with a velocity relative to its blades. Consequently, the aerodynamic loading can change with:

- Axial inflow.
- Cross-flow.
- Rotor-induced velocity.
- Rotor orientation.
- Blade geometry.

A more physical rotor model uses Blade Element Momentum Theory (BEMT), or an appropriately validated momentum-theory inflow model.

Thesis significance: If your rotor thrust remains unchanged regardless of wind, your simulator cannot reproduce all the aerodynamic coupling between wind and rotor performance.

# 3. Gravity force — the downward force

[Autonomous Landing of Quadrotor Unmanned Aerial Vehicles Based on Multi-Level Marker and Linear Active Disturbance Reject Control - PMC](https://images.openai.com/static-rsc-4/1r9cd6QpemE26apWcF8x77w7dFP6XuwiaWtxMUaQT7jgA-l5iDThQHoqnjXzSzGZkaE6kw1VVN_PlOoQy-Y5ytFugOJL6DW-UR9D2Vis_WX5w-5SSaYjCUubLteX0ON4VBGwJejDjx8JRauIYKN_nXyY4YIvpM6kqbqs6ummYdY?purpose=inline)

[pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC10935343/?utm_source=chatgpt.com)

Gravity is the force exerted by Earth on the drone.

Mathematically:

\\[ \boxed{\mathbf F_g=m\mathbf g} \\]

Where:

\\[ \mathbf g= \begin{bmatrix} 0\\\0\\\\-9.81 \end{bmatrix} \\]

if using the conventional world frame with Z pointing upward.

For your drone:

\\[ \mathbf F_g= \begin{bmatrix} 0\\\0\\\\-6.67 \end{bmatrix}\text{ N} \\]

(Your simulator's uploaded code uses Y as its vertical axis, so the corresponding vector is along negative Y.)

Physical significance:

- Gravity constantly pulls the drone downward.
- Rotor thrust must overcome gravity to maintain altitude.
- If thrust is smaller than weight, the drone accelerates downward.
- If thrust is greater than weight, it accelerates upward, assuming no other forces.

For vertical motion:

\\[ m\ddot z=T-mg \\]

This is the basic equation for altitude dynamics in a level drone.

# 4. Aerodynamic drag force — resistance from moving through air

[Aerodynamics, Weight and Balance - RePL Study Guide | National Drones](https://images.openai.com/static-rsc-4/560hHs3Crvcj9E3K3-9ZCw82taJvTKR5N8jw_R5HUzVEwifV0od-w5YJxNYuVSuqmhnz-Dc5CnN5frGNB9p4BQdQWhho_YZZlag9-Mq9NA7P5Mj5uCnrK4yB1uWbCfRMeMj7P3YvAXDrPFX9CEC2S1YGEPkrSxwjnMRXl3QDy_k?purpose=inline)

[nationaldrones.com.au](https://www.nationaldrones.com.au/training/repl-study-guide/basic-aviation-knowledge/aerodynamics-weight-and-balance/?utm_source=chatgpt.com)

This is one of the most important forces for your wind simulation.

When a drone moves through air, it experiences aerodynamic resistance.

However, there is one fundamental concept:

Drag depends on the velocity of the drone relative to the surrounding air, not simply its velocity relative to the ground.

## 4.1 Relative air velocity

Define:

\\[ \boxed{ \mathbf v\_{\text{rel}}= \mathbf v\_{\text{drone}}- \mathbf u\_{\text{wind}} } \\]

Where:

- \\(\mathbf v\_{\text{drone}}\\): drone velocity in the world frame.
- \\(\mathbf u\_{\text{wind}}\\): local air velocity in the world frame.

For example:

- Drone moves forward at 5 m/s.
- Wind moves in the same direction at 3 m/s.

Then:

\\[ v\_{\text{rel}}=5-3=2\text{ m/s} \\]

If the wind moves against the drone at 3 m/s:

\\[ v\_{\text{rel}}=5-(-3)=8\text{ m/s} \\]

The second case produces significantly more drag.

## 4.2 Quadratic drag equation

For a simplified body:

\\[ \boxed{ F_D=-\frac12\rho C_D A |\mathbf v\_{\text{rel}}|\mathbf v\_{\text{rel}} } \\]

Where:

| Symbol                        | Meaning                    |
| ----------------------------- | -------------------------- |
| \\(\rho\\)                    | Air density, kg/m³         |
| \\(C_D\\)                     | Drag coefficient           |
| \\(A\\)                       | Reference area, m²         |
| \\(\mathbf v\_{\text{rel}}\\) | Relative air velocity, m/s |

Why quadratic?

Dynamic pressure is:

\\[ q=\frac12\rho V^2 \\]

Drag is dynamic pressure multiplied by drag coefficient and reference area:

\\[ F_D=C_DAq \\]

The negative sign indicates that drag opposes relative motion.

## 4.3 What happens physically?

Suppose a drone is hovering in still air.

\\[ \mathbf v\_{\text{drone}}=0,\qquad \mathbf u\_{\text{wind}}=0 \\]

Then:

\\[ F_D=0 \\]

Now a 5 m/s wind hits the stationary drone.

\\[ |\mathbf v\_{\text{rel}}|=5 \\]

The drone experiences a horizontal aerodynamic force.

This force accelerates the drone in the wind direction.

Notice that drag opposes the relative motion of the body through air. It does not necessarily oppose the wind's direction.

Significance for your thesis: This is the primary mechanism through which ambient wind pushes your drone when you use the lumped drag model.

## 4.4 Why use body-frame drag coefficients?

Your simulator uses:

\\[ \mathbf F\_{D,B}= -\frac12\rho \begin{bmatrix} C_xA_x|v_x|v_x\\\ C_yA_y|v_y|v_y\\\ C_zA_z|v_z|v_z \end{bmatrix} \\]

The reason is that a drone does not have the same aerodynamic resistance in every direction.

For example:

- Front-facing area may differ from side-facing area.
- Vertical projected area may differ from horizontal projected area.
- The drag coefficients may differ with body orientation and flow conditions.

The body-frame force is then transformed into world coordinates:

\\[ \boxed{ \mathbf F\_{D,W}=R\_{WB}\mathbf F\_{D,B} } \\]

Here \\(R\_{WB}\\) is the rotation matrix from body coordinates to world coordinates.

# 5. Surface pressure force — wind pushing against the drone's body

[FREE WEBINAR MARCH 16: Harmonic CFD Simulation of Drones (UAVs) Using Fidelity - Digital Engineering 24/7](https://images.openai.com/static-rsc-4/5CcdyGcUfYN2fbY1LsspqXBihFJ3T_LV0zfWdbdON1A7Ye2gkUWEenfB7v8mwGee0TZBGi0JAIlFBIChNg6qJIdNGVIc7o5HH_sSMpqTnLEiyIsrUV_pKzI8Hxr5gPkzMC4RaI_sh4mY9CzLEBGyfULAsbRqi0I61xTM9kmoveQ?purpose=inline)

[digitalengineering247.com](https://www.digitalengineering247.com/article/free-webinar-march-16-harmonic-cfd-simulation-of-drones-uavs-using-fidelity?utm_source=chatgpt.com)

Drag is a simplified way of representing aerodynamic resistance. But where does that resistance actually come from?

When air flows around a drone's body, it creates a pressure distribution over the body surface.

Some surfaces experience high pressure, while others experience low pressure.

The resulting pressure difference produces a net aerodynamic force.

## 5.1 What is pressure?

Pressure is force per unit area:

\\[ p=\frac{F}{A} \\]

Its unit is Pascal (Pa), equivalent to N/m².

The pressure associated with moving air is represented by dynamic pressure:

\\[ \boxed{q=\frac12\rho V\_{\text{rel}}^2} \\]

This represents the kinetic energy per unit volume of the moving air.

For example, if:

- Air density = 1.225 kg/m³.
- Relative velocity = 5 m/s.

Then:

\\[ q=\frac12(1.225)(5^2) =15.31\text{ Pa} \\]

This is the characteristic dynamic pressure available to generate aerodynamic loading.

## 5.2 Pressure coefficient

Pressure distribution over a surface is commonly represented using a dimensionless pressure coefficient:

\\[ \boxed{ C_p=\frac{p-p\_\infty}{q} } \\]

Therefore:

\\[ \boxed{ p=p\_\infty+C_pq } \\]

Where:

- \\(p\\): local surface pressure.
- \\(p\_\infty\\): free-stream atmospheric pressure.
- \\(C_p\\): pressure coefficient.
- \\(q\\): dynamic pressure.

The pressure coefficient tells us how much the local pressure differs from atmospheric free-stream pressure.

Typical interpretation:

| Pressure coefficient | Physical meaning                 |
| -------------------- | -------------------------------- |
| \\(C_p>0\\)          | Higher pressure than free stream |
| \\(C_p=0\\)          | Free-stream reference pressure   |
| \\(C_p<0\\)          | Lower pressure than free stream  |

For example:

- At a stagnation region, air slows down and pressure increases.
- Around accelerated flow regions, pressure can decrease.
- Behind the body, separated flow can produce a low-pressure wake.

## 5.3 How does pressure produce force?

Consider a small surface element with area \\(dA\\).

Its pressure force is:

\\[ \boxed{ d\mathbf F_p=-p\mathbf n\\,dA } \\]

Where:

- \\(p\\): local pressure.
- \\(\mathbf n\\): outward surface normal.
- \\(dA\\): small surface area.

The negative sign appears because pressure pushes inward, opposite to the outward normal.

For the entire drone:

\\[ \boxed{ \mathbf F_p=-\int_S p\mathbf n\\,dA } \\]

This is the actual pressure-force formulation.

### What does this mean physically?

Imagine a drone facing a horizontal wind.

1. Air approaches the front of the drone.
2. Air slows near the front surface.
3. Pressure increases on the front.
4. Air flows around the body.
5. Pressure changes over the remaining surfaces.
6. The pressure distribution produces a net force.

This is why wind can push a stationary hovering drone sideways.

## 5.4 What does your simulator calculate?

Your simulator uses a simplified pressure approximation for each triangular mesh element:

\\[ \boxed{ p_j=C_pq \left[\max(0,-\hat{\mathbf h}\cdot\hat{\mathbf n})\right]^2 } \\]

Where:

- \\(\hat{\mathbf h}\\): incoming wind direction.
- \\(\hat{\mathbf n}\\): triangle surface normal.
- \\(q\\): dynamic pressure.
- \\(C_p\\): empirical pressure coefficient.

The dot product:

\\[ \hat{\mathbf h}\cdot\hat{\mathbf n} \\]

measures the alignment between wind direction and surface orientation.

The maximum operation ensures that only windward surfaces receive pressure.

The force on one triangle is:

\\[ \boxed{ \mathbf F\_{p,j}=-p_jA_j\hat{\mathbf n}\_j } \\]

The total force is:

\\[ \boxed{ \mathbf F_p=\sum\_{j=1}^{N}\mathbf F\_{p,j} } \\]

### Significance

This approach allows your simulator to account for:

- Drone body geometry.
- Surface orientation.
- Wind direction.
- Local pressure loading.
- Different forces on different parts of the body.

However, it is an empirical surface-pressure approximation, not a solution of the Navier–Stokes pressure field.

Important limitation: Your current implementation clips the pressure to windward faces. It does not reproduce leeward suction, which can make a substantial contribution to the actual aerodynamic force.

# 6. Aerodynamic shear force — friction between air and the drone surface

[011 (PPT) Boundary Layer Flow.pdf          . | PDF](https://images.openai.com/static-rsc-4/d1lwasOqT6Kyeb2ti1PrAXq3RrYUGjLcN5X6pMV7SRRBuLp8wVB_qnA93-dDmnfgIWl8PL6L5QhBv-8-vqnP0t8KxrEOyu2SwaVdvfqmmq_PNuK3BOjd-i_4ldZqHdXLw1MZUrvBWnzjHz_UuscL5k2lfXaxsoC0PX4BccBYbE0?purpose=inline)

[slideshare.net](https://www.slideshare.net/slideshow/011-ppt-boundary-layer-flowpdf/266781359?utm_source=chatgpt.com)

Pressure is not the only source of aerodynamic force.

Air also creates frictional forces along the drone's surface.

This is called viscous shear force.

## 6.1 Why does shear occur?

Air has viscosity.

Viscosity is the property that describes resistance to relative motion between adjacent fluid layers.

When air moves over a drone:

- Air touching the surface has approximately the same velocity as the surface (no-slip condition).
- Air farther away moves faster.
- A velocity gradient develops near the surface.
- This gradient generates viscous shear stress.

Mathematically, for a Newtonian fluid:

\\[ \boxed{ \tau_w=\mu\frac{\partial u_t}{\partial n} } \\]

Where:

- \\(\tau_w\\): wall shear stress, Pa.
- \\(\mu\\): dynamic viscosity, Pa·s.
- \\(u_t\\): tangential air velocity.
- \\(n\\): direction normal to the surface.

The larger the tangential velocity gradient, the greater the shear stress.

## 6.2 Shear force

For a surface element:

\\[ dF_s=\tau_wdA \\]

A common aerodynamic approximation is:

\\[ \boxed{ \mathbf F_s= \frac12\rho C_fA |\mathbf v_t|\mathbf v_t } \\]

Where:

\\[ \boxed{ \mathbf v_t= \mathbf v\_{\text{rel}} -(\mathbf v\_{\text{rel}}\cdot\mathbf n)\mathbf n } \\]

This removes the velocity component perpendicular to the surface, leaving only the tangential component.

Here:

- \\(C_f\\): skin-friction coefficient.
- \\(A\\): surface area.
- \\(\mathbf v_t\\): tangential relative velocity.

Your simulator calculates this force for each triangle and sums the results.

\\[ \mathbf F_s=\sum_j\mathbf F\_{s,j} \\]

### Significance

Shear force represents:

- Frictional resistance along the drone's surface.
- Viscous aerodynamic loading.
- Energy dissipation due to air movement near the body.

For a compact quadrotor, pressure drag is often more significant than skin-friction drag, but shear is still a physically meaningful contribution.

Important: The constant \\(C_f\\) used in your simulator is an approximation. Real skin friction depends on Reynolds number, surface roughness, and boundary-layer conditions.

# 7. Added mass force — accelerating the surrounding air

[Swimming, flying, and diving behaviors from a unified 2D potential model | Scientific Reports](https://images.openai.com/static-rsc-4/xvVkkO4-mukoqMS42BRYIfP-40Ehzgf-x1r3FiIXeejq2HAd5GMoJl4dJlnkUTdCqbYSbzufHkf3G4iY9vAxqlwDjowcESXHL-g5git7__XABZM3My9ElVY9Q06LZO1W2m3iTgskc-BOmqAEZLOY4JSuvIJMYXjTm7zNaG4Q-sc?purpose=inline)

[nature.com](https://www.nature.com/articles/s41598-021-94829-7?utm_source=chatgpt.com)

This is a more subtle aerodynamic effect.

Consider a drone suddenly accelerating forward.

It does not only accelerate its own mass. It also has to accelerate some of the surrounding air.

This creates an additional inertial effect called added mass.

## 7.1 Basic concept

Suppose the drone has:

\\[ m=0.68\text{ kg} \\]

But due to the surrounding fluid, its effective inertial mass becomes:

\\[ m\_{\text{effective}}=m+m\_{\text{added}} \\]

Then:

\\[ \boxed{ F=(m+m\_{\text{added}})a } \\]

This is a simplified interpretation for a particular direction and geometry.

The added mass is not additional physical material attached to the drone. It is the inertia of the fluid that must move with the accelerating body.

## 7.2 Mathematical formulation

For a stationary fluid and a simplified body:

\\[ \boxed{ \mathbf F\_{\text{added}} =-\mathbf M_A\dot{\mathbf v}\_{\text{rel}} } \\]

Where:

- \\(\mathbf M_A\\): added-mass matrix, kg.
- \\(\dot{\mathbf v}\_{\text{rel}}\\): acceleration relative to the surrounding air.

A simplified isotropic approximation is:

\\[ \mathbf F\_{\text{added}} =-m_A\dot{\mathbf v}\_{\text{rel}} \\]

For a moving, spatially varying wind field, a complete formulation can also contain terms involving fluid acceleration and added-mass matrix variation.

### Physical significance

Added mass becomes relevant during:

- Sudden gust onset.
- Rapid acceleration.
- Rapid changes in flight direction.
- High-frequency disturbances.

It changes how quickly the drone responds to a transient aerodynamic disturbance.

Your audit estimates vertical added mass at approximately 5–15% of dry mass, but this is an illustrative estimate, not a universal quadrotor constant. It needs geometry-based estimation or validation.

# 8. Aerodynamic torque — wind can rotate the drone

[Quadrotor Path Planning and Polynomial Trajectory Generation Using Quadratic Programming for Indoor Environments | MDPI](https://images.openai.com/static-rsc-4/JHodV6O4ECo7pg2OwxLUMLjA3wKW-eL1-1zAAOmX3aUv_YbWDDRmstRL0yhGder1QR-qFSp_YSr-uAqhnT8CNMDp_Kttb78IK54zKCgx8xLtqZekl6IKpEKP9LoCuqJTEfN_5FghwBSVg_CDXtw911dwWtb_BEqytW9VzSLnEq4?purpose=inline)

[mdpi.com](https://www.mdpi.com/2504-446X/7/2/122?utm_source=chatgpt.com)

So far, we have discussed forces that translate the drone.

But aerodynamic forces can also produce rotation.

Imagine a horizontal wind pushing the front of the drone more strongly than the back.

The resultant force may not pass through the center of gravity.

Therefore, it produces a torque.

## 8.1 Mathematical definition

Torque is:

\\[ \boxed{ \boldsymbol\tau=\mathbf r\times\mathbf F } \\]

Where:

- \\(\mathbf r\\): vector from center of gravity to the point where the force acts.
- \\(\mathbf F\\): applied force.

Its magnitude is:

\\[ \tau=rF\sin\theta \\]

For example:

\\[ r=0.1\text{ m},\quad F=2\text{ N} \\]

If the force is perpendicular to the moment arm:

\\[ \tau=0.1(2)=0.2\text{ N·m} \\]

This torque can cause pitch, roll, or yaw depending on its direction.

## 8.2 Torque from surface forces

For your triangle-based surface model:

\\[ \boxed{ \boldsymbol\tau\_{\text{aero}} = \sum\_{j=1}^{N} (\mathbf r_j-\mathbf r\_{\text{CG}}) \times\mathbf F_j } \\]

This is extremely useful because every triangle can generate a force at a different position.

The drone can therefore experience aerodynamic moments even when the total aerodynamic force is relatively small.

### Significance

Aerodynamic torque affects:

- Attitude stability.
- Roll and pitch disturbances.
- Yaw stability.
- Controller effort.
- Trajectory tracking.

For your thesis, this matters because a wind disturbance can alter the drone's attitude, which subsequently changes the direction of rotor thrust.

That creates a coupled translational-rotational response.

# 9. Rotor reaction torque — why rotating propellers create yaw

[Optimized Adaptive PID Controller Design for Trajectory Tracking of a Quadcopter | IIETA](https://images.openai.com/static-rsc-4/UvL6Cz31MPQmZlNy4lHc58mdfwt6XeJwO9ON9-EgtC-nITnTALKAHvKjpar36bcOH2b_Rt1AyHhWjw0mpBtLysIj9XHmpaXKa-HwS4UEYQ6_K1VxAaoI-CGfIy8gwCStUGm3nHWu9ukfF3TRmUfZS41pMFHk3fUoqgZkIxLN00s?purpose=inline)

[iieta.org](https://iieta.org/journals/mmep/paper/10.18280/mmep.090607?utm_source=chatgpt.com)

Each propeller requires a motor torque to maintain rotation.

By Newton's third law, the propeller produces an opposing reaction torque on the drone.

The magnitude is approximated by:

\\[ \boxed{ Q_i=k_Q\Omega_i^2 } \\]

Where:

- \\(Q_i\\): rotor reaction torque.
- \\(k_Q\\): rotor torque coefficient.
- \\(\Omega_i\\): rotor speed.

Quadrotor rotors rotate in alternating directions.

Therefore, their reaction torques can cancel each other during hover.

To produce yaw:

- Increase the speed of one rotor pair.
- Decrease the speed of the opposite pair.
- Create a net reaction torque.

Mathematically:

\\[ \boxed{ \tau\_{\text{yaw}}=-\sum_i s_iQ_i } \\]

Where \\(s_i=+1\\) or \\(-1\\) represents the rotor rotation direction.

### Significance

Reaction torque controls yaw.

It is distinct from thrust torque, which comes from the moment arm of each rotor.

# 10. Differential thrust torque — how the drone rolls and pitches

Each rotor produces thrust at a location away from the center of gravity.

This generates torque:

\\[ \boxed{ \boldsymbol\tau_i=\mathbf r_i\times\mathbf F_i } \\]

Total rotor moment:

\\[ \boxed{ \boldsymbol\tau\_{\text{rotor}} = \sum_i\mathbf r_i\times\mathbf F_i - \sum_i s_iQ_i\hat{\mathbf y} } \\]

(Your simulator uses body Y as the rotor thrust axis.)

By changing individual rotor speeds, the drone creates:

- Roll torque.
- Pitch torque.
- Yaw torque.

This is how the flight controller changes the drone's orientation.

The distinction is important:

| Mechanism           | Physical effect        |
| ------------------- | ---------------------- |
| Collective thrust   | Changes altitude       |
| Differential thrust | Changes roll and pitch |
| Reaction torque     | Changes yaw            |

# 11. Gyroscopic coupling — rotational dynamics are interconnected

Gyroscopic coupling is not an additional external force. It is an inertial effect that appears in the rotational equations.

The equation is:

\\[ \boxed{ \mathbf J\dot{\boldsymbol\omega} = \boldsymbol\tau- \boldsymbol\omega\times(\mathbf J\boldsymbol\omega) } \\]

Why does this happen?

A rotating rigid body has angular momentum:

\\[ \mathbf H=\mathbf J\boldsymbol\omega \\]

When the drone rotates about one axis, its angular momentum changes direction.

Consequently, rotation about one axis can influence the dynamics of another axis.

For a diagonal inertia matrix:

\\[ \mathbf J= \begin{bmatrix} J_x&0&0\\\ 0&J_y&0\\\ 0&0&J_z \end{bmatrix} \\]

The rotational equations become:

\\[ J_x\dot p=\tau_x+(J_y-J_z)qr \\]

\\[ J_y\dot q=\tau_y+(J_z-J_x)rp \\]

\\[ J_z\dot r=\tau_z+(J_x-J_y)pq \\]

Here \\(p,q,r\\) are body angular velocities.

Significance: This is necessary for realistic rigid-body rotational dynamics, particularly during aggressive maneuvers.

Your simulator already includes this term.

# 12. Unsteady aerodynamic force — why aerodynamic forces do not respond instantaneously

[Controlling upwards and downwards gust loads on aerofoils by pitching | Experiments in Fluids | Springer Nature Link](https://images.openai.com/static-rsc-4/1ztRdSGU-OOnnEmNBsdbmgU9tVW1zKUZzens6jXOrUin05q6K1LXiz1u-pAhP9xWZJ3dkuIfgBWbRrdHeSotQ53rd1lsJBYNocY7aekjfRdVNtr0e_7dsmT87xrRFZ4_Sd7xxs08tmm6caRbxOTGVU8o8M9-xHw_9ZtyLi7kxpk?purpose=inline)

[link.springer.com](https://link.springer.com/article/10.1007/s00348-023-03670-3?utm_source=chatgpt.com)

Consider a drone flying forward when a sudden gust hits it.

The airflow around the drone changes immediately, but the aerodynamic loading does not necessarily reach its new steady-state value immediately.

This is called unsteady aerodynamics.

For an airfoil, the Wagner function is commonly used to describe the development of lift following a sudden change in angle of attack.

A simplified representation is:

\\[ \boxed{ L(t)=L\_{\text{steady}}\phi_W(t) } \\]

Where:

- \\(L\_{\text{steady}}\\): eventual steady aerodynamic lift.
- \\(\phi_W(t)\\): time-dependent response function.

Its important properties are:

\\[ \phi_W(0)=0.5 \\]

\\[ \phi_W(\infty)=1 \\]

This means the aerodynamic response develops over time rather than reaching its final value instantly.

### Significance for your UAV

During sudden gusts:

1. Wind velocity changes.
2. Relative airflow around the drone changes.
3. Surface pressure and aerodynamic loading evolve.
4. The drone experiences a transient force.
5. Eventually, the aerodynamic response approaches its steady-state value.

Important distinction: The Wagner and Theodorsen models were developed for unsteady airfoil aerodynamics. You should not directly apply the airfoil lift equation to the entire quadrotor body. For your application, an appropriate unsteady body/rotor aerodynamic model would need to be developed or validated.

Your current surface model calculates forces from instantaneous local velocity, without aerodynamic memory.

# 13. Rotor inflow dynamics — air entering the propeller

[Figure 3 - from Aerodynamic power control for multirotor](https://images.openai.com/static-rsc-4/VeA3ewS3s36DVdatP-wmBvK9e3M214zBb0ZkMyN_CNXVLSMkAWSFjgTjIvQ1peL9_qTabYKO0AH9TQuDfL06WoI8pMKn-1RX4jte0dVFrE52556PtXg-sRqnM-CKdHLrayIuXYK2kXX9kgFQ1yGrHaTc0jJwXQzVCKKnKtdSKJQ?purpose=inline)

[academia.edu](https://www.academia.edu/figures/24921174/figure-3-momentum-theory-control-volume-we-represent-the?utm_source=chatgpt.com)

This is another important effect.

When a propeller rotates, it accelerates air downward. This induced airflow is called rotor inflow or induced velocity.

Let:

\\[ v_i=\text{induced velocity} \\]

For an ideal rotor in hover, momentum theory gives:

\\[ \boxed{ T=2\rho A_r v_i^2 } \\]

Therefore:

\\[ \boxed{ v_i=\sqrt{\frac{T}{2\rho A_r}} } \\]

Where:

- \\(T\\): rotor thrust.
- \\(\rho\\): air density.
- \\(A_r\\): rotor disk area.
- \\(v_i\\): induced velocity at the rotor disk.

The air velocity below the ideal actuator disk is approximately twice the induced velocity:

\\[ v\_{\text{far wake}}\approx 2v_i \\]

This is the ideal hover momentum-theory result.

## Why does inflow matter?

When the drone moves forward, the rotor experiences a different incoming airflow.

When wind enters the rotor disk, the induced velocity distribution changes.

Consequently, rotor thrust and torque can change.

Real rotor inflow has its own dynamics and does not instantaneously reach equilibrium.

A simplified first-order representation is:

\\[ \boxed{ \dot v_i= \frac{v\_{i,\text{equilibrium}}-v_i}{\tau_i} } \\]

Where \\(\tau_i\\) is the aerodynamic inflow time constant.

This is a conceptual approximation; a more complete model can use Pitt–Peters dynamic inflow equations.

### Significance

Rotor inflow dynamics affect:

- Thrust generation.
- Hover performance.
- Forward-flight performance.
- Response to sudden wind.
- Rotor-wind interaction.

Your simulator calculates induced velocity for visualization, but does not feed that velocity back into its rotor thrust calculation.

# 14. Rotor-rotor interference — one rotor affects another

[Aeronautics Research](https://images.openai.com/static-rsc-4/q902sZPhFR_k93rkDa1kiRwumQH8cmgGoEpzjv3_O8zjHTmCkZkhEyBQ7EUuIjPi2mLk0Haf-QUwmuPRBWdPwIWhtsezKj7jm0H9XcUuZtMb8ZmmiQYHGbN-we15yxGAviQcKH3OjnguQHJecAk7WTlWsGswWKobiBbRKW9Fq5s?purpose=inline)

[nas.nasa.gov](https://www.nas.nasa.gov/areas/aero_research.html?utm_source=chatgpt.com)

A quadrotor has four propellers operating close to each other.

Each rotor creates its own airflow field.

However, these airflow fields can interact.

For rotor \\(i\\):

\\[ \boxed{ \mathbf u_i= \mathbf u\_{\text{ambient}} + \sum\_{j\ne i}\mathbf u\_{\text{wake},j} } \\]

Where:

- \\(\mathbf u\_{\text{ambient}}\\): environmental wind.
- \\(\mathbf u\_{\text{wake},j}\\): airflow generated by another rotor.

This means the airflow entering one rotor can be influenced by the other rotors.

### Significance

Rotor interference can affect:

- Individual rotor thrust.
- Thrust symmetry.
- Pitch and roll moments.
- Performance during aggressive manoeuvres.

Your current model calculates each rotor independently using:

\\[ T_i=k_T\Omega_i^2 \\]

It does not account for the aerodynamic influence of neighboring rotors.

# 15. Ground effect — why flying close to the ground changes thrust

[Model, Control, and Realistic Visual 3D Simulation of VTOL Fixed-Wing Transition Flight Considering Ground Effect](https://images.openai.com/static-rsc-4/kX72HLdt1FmNmSzMzDXzhvhrL7PYbMhc39gbxlCsavgm-Z0qPdQ1S56DlkQGU5sXuo0pyUOubcypaFcfWG6kuXXp7Ka2IDyFkArLaKCsVPGMSBnOQ_lVLy-C3AXzJ_kZdoAB7M9QI1wPCdVkH2g7_WrnVPNAM33Mjrj52nCRKRI?purpose=inline)

[mdpi.com](https://www.mdpi.com/2504-446X/7/5/330?utm_source=chatgpt.com)

When a drone hovers close to the ground, its downward airflow encounters the ground.

The ground restricts the development of the rotor wake.

This changes the induced velocity and rotor performance.

An approximate ground-effect relationship from your audit is:

\\[ \boxed{ \frac{T\_{\text{IGE}}}{T\_{\text{OGE}}} \approx \frac{1}{1-(R/4z)^2} } \\]

Where:

- IGE: in-ground effect.
- OGE: out-of-ground effect.
- \\(R\\): rotor radius.
- \\(z\\): rotor height above ground.

For example, using:

\\[ R=0.17\text{ m},\qquad z=0.5\text{ m} \\]

The approximate thrust ratio is:

\\[ \frac{T\_{\text{IGE}}}{T\_{\text{OGE}}} \approx 1.029 \\]

That corresponds to approximately 3% higher thrust in this approximation.

Significance: Ground effect becomes relevant when studying takeoff, landing, low-altitude hovering, and near-ground disturbances.

Your simulator currently uses a ground-contact constraint but does not calculate ground-effect aerodynamics.

# 16. Flow disturbance — the drone itself changes the surrounding airflow

[Scientific Visualization Gallery](https://images.openai.com/static-rsc-4/a101PDxhgq8JfHXkr_b0idPHIQmvntNfSTMdGj-3uvHnG0yERFGmUjvKVwaRSUFa5dqapN7ip6S7LXEuoZ-CaFeSIpC662yRWhD_80chdjoSW3jLxFTLsrcpR8Fqrsw_YSuEMIJ80JlFCr0I9cgSVOqHPz0SiOlK4RT-8WUI3IE?purpose=inline)

[nas.nasa.gov](https://www.nas.nasa.gov/pubs/gallery.html?utm_source=chatgpt.com)

This is particularly interesting for your wind-field simulator.

So far, we have considered environmental wind affecting the drone.

But a real drone also disturbs the airflow around itself.

For example:

- Incoming wind encounters the drone body.
- Air accelerates around its surfaces.
- Flow separates behind the body.
- A wake develops downstream.
- The wake contains velocity deficits and vortical structures.

Your simulator approximates this using two components.

## 16.1 Potential-flow dipole

Your audit describes a dipole model:

\\[ \boxed{ \mathbf u\_{\text{dipole}}(\mathbf r) = \frac{UR^3}{2r^3} \left( 2\cos\theta\\,\hat{\mathbf e}\_r -\sin\theta\\,\hat{\mathbf e}\_\theta \right) } \\]

This is a potential-flow approximation that represents the disturbance around an idealized body.

Where:

- \\(U\\): reference flow velocity.
- \\(R\\): characteristic body radius.
- \\(r\\): distance from the dipole centre.
- \\(\theta\\): angle relative to the flow direction.

The \\(1/r^3\\) term means that the disturbance decreases rapidly with distance.

## 16.2 Wake deficit

The wake behind the drone is represented by a reduction in flow velocity.

A simplified expression is:

\\[ \boxed{ \mathbf u\_{\text{wake}} = -\Delta U\\,\hat{\mathbf e}\_x } \\]

Where \\(\Delta U\\) depends on the wake strength and spatial position.

Your implementation uses Gaussian spatial factors to represent the wake.

### Significance

These effects help represent:

- Disturbed incoming flow.
- Velocity reduction behind the drone.
- Spatial variation of the wind field.
- Interaction between the body and the surrounding environment.

However, your simulator uses a prescribed disturbance rather than solving the complete fluid-flow equations.

The drone's actual aerodynamic loading does not dynamically determine the wake strength.

# 17. Turbulence and gust forces — time-varying environmental disturbances

[Turbulence - MeteoSwiss](https://images.openai.com/static-rsc-4/_pwOyxTDwKusTq5YWq0hyZndFSd95hCsF9dR2QIWXu3i6zGVrxOCP5WPQbY_RgYZasJofiyPWvQp5Zq69g389oUJFuXUKCTy8-mhpraYktK9j6QAwIVqhnHDRFzwP-8mAT0pg64VqyBjRwLN7MH0ozFUqK8wo6lV61t2qOGwj4g?purpose=inline)

[meteoswiss.admin.ch](https://www.meteoswiss.admin.ch/weather/weather-and-climate-from-a-to-z/turbulence.html?utm_source=chatgpt.com)

Wind is not always steady.

A drone can encounter:

- Sudden gusts.
- Random turbulent fluctuations.
- Spatially varying wind.
- Vertical updrafts.
- Directional changes.

These are not separate mechanical forces by themselves. They change the local air velocity, which changes the aerodynamic forces.

A useful representation is:

\\[ \boxed{ \mathbf u\_{\text{wind}}(\mathbf p,t) = \mathbf u\_{\text{steady}} + \mathbf u\_{\text{gradient}} + \mathbf u\_{\text{gust}} + \mathbf u\_{\text{turbulence}} } \\]

The resulting relative velocity is:

\\[ \mathbf v\_{\text{rel}}= \mathbf v\_{\text{drone}}- \mathbf u\_{\text{wind}}(\mathbf p,t) \\]

This relative velocity enters the aerodynamic force equations.

## Dryden turbulence

The Dryden model represents turbulence statistically using spatial length scales and velocity variances.

For the longitudinal component, your audit gives:

\\[ \boxed{ \Phi\_{uu}(\omega)= \sigma_u^2 \frac{2L_u}{\pi V} \frac{1}{1+(L_u\omega/V)^2} } \\]

Where:

- \\(\sigma_u\\): turbulence standard deviation.
- \\(L_u\\): longitudinal turbulence length scale.
- \\(V\\): reference flight speed.
- \\(\omega\\): temporal angular frequency.

The power spectral density describes how turbulence energy is distributed across frequencies.

### Significance

This is important for training your world model because two wind fields can have the same average velocity but different temporal characteristics.

For example:

- Smooth wind produces gradual disturbances.
- High-frequency turbulence produces rapid fluctuations.
- Sudden gusts create transient disturbances.

A robust model should learn the relationship between the wind characteristics, drone state, and resulting dynamics.

Your audit notes that the implemented first-order Gauss–Markov process does not reproduce the complete standard Dryden spectra for all three axes.

# 18. Other physical effects: should you model them?

Your uploaded audit also identifies several additional phenomena.

| Phenomenon               | Mathematical significance                                                        | Physical effect                                              |
| ------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Blade flapping           | \\(\beta(\psi)=\beta_0+\beta\_{1c}\cos\psi+\beta\_{1s}\sin\psi\\)                | Blade deformation changes rotor thrust direction and moments |
| Blade-vortex interaction | Interaction between blade and previously generated vortices                      | Periodic thrust and aerodynamic fluctuations                 |
| Boundary layer           | Velocity gradient near the surface                                               | Produces viscous shear                                       |
| Flow separation          | Adverse pressure gradient causes boundary-layer detachment                       | Changes pressure distribution and drag                       |
| Compressibility          | Depends on Mach number \\(M=V/a\\)                                               | Density changes in high-speed flow                           |
| Buoyancy                 | \\(F_b\approx \rho Vg\beta\Delta T\\)                                            | Temperature differences create vertical forces               |
| Coriolis                 | \\(F_c=-2m\boldsymbol\Omega_E\times\mathbf v\\), in a rotating-frame formulation | Earth-rotation effect                                        |
| Added rotational inertia | \\(\boldsymbol\tau_A=-\mathbf J_A\dot{\boldsymbol\omega}\\)                      | Fluid inertia resists angular acceleration                   |

For your low-speed UAV application, compressibility, thermal buoyancy and Earth-rotation effects are generally lower priorities than rotor inflow, body pressure and transient aerodynamic effects.

# 19. Putting everything together: complete force and torque accounting

Now let's combine the concepts.

[Figure 65 - from Multi-agent systems design for aerospace](https://images.openai.com/static-rsc-4/8kblMdxdIWdGvqOLdgrufyoyYYeF1s6Fl5YVVRzTCrz5Veemcc3GSVQvJWg-6oOJUNICWwxepJJ7fcIb1qxJ2KfUXuHcVvUDktaJXuel4xqv6-fl_YZyl2hLsk6fptf7x0gDk9K3j7MLJX25xj3nlSlHR2PkApnFxotFxhwFX2s?purpose=inline)

[academia.edu](https://www.academia.edu/figures/38368071/figure-65-free-body-diagram-of-quadrotor-uav-the-rotor-plane?utm_source=chatgpt.com)

## Translational dynamics

A more comprehensive conceptual model is:

\\[ \boxed{ m\dot{\mathbf v}\_W = \mathbf F_T+ \mathbf F_g+ \mathbf F_p+ \mathbf F_s+ \mathbf F_A+ \mathbf F\_{\text{other}} } \\]

Where:

| Force                     | Mathematical representation              | Significance                                     |
| ------------------------- | ---------------------------------------- | ------------------------------------------------ |
| Rotor thrust              | \\(\sum_i T_i\mathbf d_i\\)              | Generates flight force                           |
| Gravity                   | \\(m\mathbf g\\)                         | Pulls drone downward                             |
| Pressure                  | \\(-\int_S p\mathbf n\\,dA\\)            | Aerodynamic loading due to pressure distribution |
| Shear                     | \\(\int_S\boldsymbol\tau_w\\,dA\\)       | Viscous surface friction                         |
| Added mass                | \\(-\mathbf M_A\dot{\mathbf v}\_{rel}\\) | Fluid inertia during acceleration                |
| Other aerodynamic effects | Model-dependent                          | Unsteady and rotor-flow corrections              |

Important: If you use the lumped drag model, pressure and shear are already represented approximately by the combined drag force. Do not add the lumped drag and full surface forces together without accounting for overlap.

## Rotational dynamics

\\[ \boxed{ \mathbf J\dot{\boldsymbol\omega} = \boldsymbol\tau\_{\text{rotor}} + \boldsymbol\tau\_{\text{aero}} + \boldsymbol\tau\_{\text{other}} - \boldsymbol\omega\times\mathbf J\boldsymbol\omega } \\]

Where:

\\[ \boldsymbol\tau\_{\text{rotor}} = \sum_i\mathbf r_i\times\mathbf F_i - \sum_i s_iQ_i\hat{\mathbf y} \\]

And:

\\[ \boldsymbol\tau\_{\text{aero}} = \int_S (\mathbf r-\mathbf r\_{\text{CG}}) \times d\mathbf F\_{\text{aero}} \\]

# 20. What actually happens when wind hits your drone?

Let's take one practical example.

[(PDF) Battery-Aware Energy Model of Drone Delivery Tasks](https://images.openai.com/static-rsc-4/2gRiOB5FNpefYH5HcyGtwaakq_THENuESpygfn5NzFOj0sB01OicS0HJ2GcQ4pl8mcqHl1_9-JuqD4MrvrXxja8i3JPLNZJDviPtIDGFUVF_TmOPI95osC0rJlnEpTcfQU_sld67aFtIvKzZq8cWsraSlrLSWIvKv_0isAee8Y8?purpose=inline)

[academia.edu](https://www.academia.edu/38796079/Battery_Aware_Energy_Model_of_Drone_Delivery_Tasks?utm_source=chatgpt.com)

Assume:

- Drone mass = 0.68 kg.
- Drone initially hovering.
- Horizontal wind = 5 m/s.
- Initial drone velocity = 0 m/s.

The sequence is:

1. Wind enters the drone's surroundings

   The local air velocity changes from zero to 5 m/s.
2. Relative air velocity develops

   \\[ \mathbf v\_{\text{rel}}=\mathbf v_d-\mathbf u_w \\]

   Initially its magnitude is 5 m/s.
3. Aerodynamic pressure develops

   \\[ q=\frac12\rho(5)^2=15.31\text{ Pa} \\]

   Different surfaces experience different pressures.
4. Aerodynamic forces act on the body

   Pressure and shear generate a resultant force and possibly a moment.
5. Drone accelerates

   \\[ m\dot{\mathbf v}=\mathbf F_T+\mathbf F_g+\mathbf F\_{\text{aero}} \\]
6. The controller reacts

   The flight controller changes individual rotor speeds to counteract displacement and maintain the target position.
7. Rotor thrust changes the drone's attitude and trajectory

   The drone tilts to generate the required corrective acceleration.

This is the complete physical chain you want your world model to understand.

# 21. Which effects should you prioritize for your thesis?

Your thesis focuses on reliable UAV prediction under out-of-distribution wind disturbances. You do not necessarily need a complete CFD solver.

You need a simulator that captures the dominant physical mechanisms accurately enough for your research question.

Based on the audit, I would prioritize the following:

| Priority | Physical effect                     | Why it matters                                         |
| -------- | ----------------------------------- | ------------------------------------------------------ |
| 1        | Correct relative-wind drag          | Fundamental wind-to-body force                         |
| 2        | Rotor inflow and wind coupling      | Wind can alter rotor thrust                            |
| 3        | Surface pressure, including suction | Better directional aerodynamic loading                 |
| 4        | Accurate numerical integration      | Avoid confusing numerical errors with physical effects |
| 5        | Rotor aerodynamic lag               | Captures transient response                            |
| 6        | Added mass                          | Improves acceleration response                         |
| 7        | Aerodynamic torque                  | Captures wind-induced attitude disturbances            |
| 8        | Rotor interference                  | Improves rotor interaction modelling                   |
| 9        | Ground effect                       | Relevant for low-altitude flight                       |

The ranking is a practical research recommendation, not a universal ranking for every UAV.

Pasted text.txt

## One important correction to check in your implementation

Your audit defines:

\\[ \mathbf v\_{\text{rel}}=\mathbf u\_{\text{wind}}-\mathbf v\_{\text{drone}} \\]

but also writes:

\\[ \mathbf F_D=-\frac12\rho C_DA |\mathbf v\_{\text{rel}}|\mathbf v\_{\text{rel}} \\]

If \\(\mathbf F_D\\) represents the force acting on the drone, these two sign conventions are inconsistent.

You should use either:

\\[ \boxed{ \mathbf F_D= \frac12\rho C_DA |\mathbf u\_{\text{wind}}-\mathbf v\_{\text{drone}}| (\mathbf u\_{\text{wind}}-\mathbf v\_{\text{drone}}) } \\]

or:

\\[ \boxed{ \mathbf F_D= -\frac12\rho C_DA |\mathbf v\_{\text{drone}}-\mathbf u\_{\text{wind}}| (\mathbf v\_{\text{drone}}-\mathbf u\_{\text{wind}}) } \\]

Both represent the same physical force. This is worth checking in your actual code before generating your dataset.

# 22. The most important conceptual understanding

I want you to remember these five relationships:

| Physical phenomenon                           | Mathematical relationship                             |
| --------------------------------------------- | ----------------------------------------------------- |
| Rotor speed produces thrust                   | \\(T=k_T\Omega^2\\)                                   |
| Wind and drone motion create relative airflow | \\(\mathbf v\_{\text{rel}}=\mathbf v_d-\mathbf u_w\\) |
| Relative airflow creates aerodynamic loading  | \\(F_D\propto\rho C_DA v\_{\text{rel}}^2\\)           |
| Forces acting away from the CG create torque  | \\(\boldsymbol\tau=\mathbf r\times\mathbf F\\)        |
| Forces and torques determine UAV dynamics     | Newton–Euler equations                                |

My recommendation: Before modifying your simulator, understand the distinction between three things:

1. Environmental airflow: What velocity field does the drone experience?
2. Aerodynamic interaction: How does that airflow generate forces and moments on the drone and rotors?
3. Rigid-body dynamics: How do those forces and moments change the drone's position, velocity, and orientation?

These are three separate modelling layers. Keeping them separate will make your simulator easier to validate, debug, and improve.

The next useful step would be to derive the complete mathematical model specifically for your Hummingbird quadrotor, including the relative-wind transformation, rotor inflow, body pressure, shear, aerodynamic torque and Newton–Euler integration, so that you can translate each equation directly into your simulator.\




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


Body drag is a good starting point, but it is not a complete quadrotor aerodynamic model.



A. Rotor thrust changes due to airflow

Wind can alter rotor inflow and thrust. A simple body-drag model does not capture this. A more advanced model accounts for the air velocity relative to each rotor and modifies thrust and rotor moments accordingly.



B. Aerodynamic moments

Airflow acting away from the center of mass can produce torque. A simple approximation is to model body-frame angular damping, but geometry-based aerodynamic moments require additional parameters.


C. Rotor and blade effects

Blade flapping, rotor drag, and induced airflow can matter at higher airspeeds. These are more involved and are best added only if your simulator or vehicle model supports them reliably.



For your thesis, I would implement wind velocity plus body-frame aerodynamic drag first. That gives you a meaningful upgrade over direct force injection without requiring a full CFD model.