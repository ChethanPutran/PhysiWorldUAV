# Task: Implement a Research-Grade UAV Dynamics Data-Generation Architecture Using Python and Isaac Sim

## 1. Research Context

I am developing a research project titled:

**Physics-Prior Latent World Models for Reliable Long-Horizon UAV Prediction under OOD Wind Disturbances.**

The objective is to generate physically meaningful quadrotor flight datasets under diverse aerodynamic and wind conditions and use these datasets to train a learned dynamics/world model.

The world model will learn UAV dynamics from numerical observations and control actions.

The primary learning objective is:

\[
\hat{s}_{t+1}=f_\theta(s_{t-H+1:t},a_{t-H+1:t})
\]

where:

- \(s_t\): Observable UAV state.
- \(a_t\): Control action.
- \(H\): Historical observation horizon.
- \(\hat{s}_{t+1}\): Predicted future state.

The primary model must not receive simulator ground-truth wind, aerodynamic forces, or randomized physical parameters as input.

However, these variables must be recorded as auxiliary ground truth for evaluation, debugging, and research.

The dataset must support:

1. Nominal UAV dynamics learning.
2. Learning under varying wind conditions.
3. Generalization to unseen wind profiles.
4. Long-horizon state prediction.
5. Aerodynamic force-model ablations.
6. Latent wind and physical-parameter inference.
7. Counterfactual evaluation of different wind conditions.
8. Investigation of action–disturbance confounding.
9. Analysis of wind identifiability from state-action histories.

**Central research requirement:** The dataset must enable us to distinguish the effects of control actions, wind disturbances, and persistent physical-parameter variations.

---

# 2. First Task: Inspect the Existing Repository

Before implementing anything, inspect the existing repository and understand its architecture.

The project uses:

- Python.
- NVIDIA Isaac Sim 4.1.0.
- Isaac Lab.
- Hummingbird quadrotor configuration.
- Existing Lee Position Controller or another compatible controller.
- Existing modular force and torque computation.
- Existing wind-field components.

Identify:

1. Simulator initialization and UAV spawning.
2. Existing controller implementation.
3. Existing force and torque computation.
4. Existing wind-field implementation.
5. State and actuator access.
6. External force and torque application.
7. Configuration and logging utilities.
8. Python environment and dependencies.

Do not blindly replace existing implementations.

Reuse existing components wherever possible. Extend them only when required.

Do not invent Isaac Sim or Isaac Lab APIs. Verify compatibility with the installed version and the existing repository.

Before coding, provide a repository architecture assessment and implementation plan.

---

# 3. Hummingbird UAV Configuration

Use the following supplied configuration as the nominal UAV configuration.

```yaml
name: hummingbird

inertia:
  xx: 0.007
  xy: 0.0
  xz: 0.0
  yy: 0.007
  yz: 0.0
  zz: 0.012

mass: 0.716
l: 0.34
drag_coef: 0.2

rotor_configuration:
  arm_lengths:
    - 0.17
    - 0.17
    - 0.17
    - 0.17

  directions:
    - -1.0
    - 1.0
    - -1.0
    - 1.0

  force_constants:
    - 8.54858e-06
    - 8.54858e-06
    - 8.54858e-06
    - 8.54858e-06

  max_rotation_velocities:
    - 838
    - 838
    - 838
    - 838

  moment_constants:
    - 1.3677728816219314e-07
    - 1.3677728816219314e-07
    - 1.3677728816219314e-07
    - 1.3677728816219314e-07

  num_rotors: 4

  rotor_angles:
    - 0
    - 1.57079632679
    - 3.14159265359
    - -1.57079632679
```

Important:

- Treat this configuration as the nominal source of truth.
- Do not silently change the supplied values.
- Verify the units and interpretation of maximum rotor velocity against the existing simulator configuration.
- Verify how `l`, `arm_lengths`, and `rotor_angles` are used in the existing implementation.
- Avoid interpreting `l = 0.34` as the individual rotor arm radius: the supplied individual arm lengths are 0.17.
- Preserve the supplied inertia tensor, including off-diagonal terms.
- Check consistency between the physical configuration and the actual spawned UAV.

Use configurable domain randomization around these nominal values.

---

# 4. Overall Architecture

Implement a modular architecture integrated into the existing repository.

Suggested logical components:

```text
uav_data_generation/
│
├── configs/
│   ├── simulation.yaml
│   ├── uav.yaml
│   ├── domain_randomization.yaml
│   ├── wind.yaml
│   ├── trajectory.yaml
│   ├── controller.yaml
│   ├── counterfactual.yaml
│   ├── identifiability.yaml
│   ├── dataset.yaml
│   └── experiments.yaml
│
├── core/
│   ├── simulation_manager.py
│   ├── episode_manager.py
│   ├── state_extractor.py
│   └── simulation_interface.py
│
├── randomization/
│   ├── domain_generator.py
│   └── parameter_sampler.py
│
├── trajectories/
│   ├── gp_kernel.py
│   ├── gp_sampler.py
│   ├── trajectory_generator.py
│   ├── trajectory_types.py
│   └── trajectory_validator.py
│
├── wind/
│   ├── wind_field.py
│   ├── steady_wind.py
│   ├── gust_generator.py
│   ├── turbulence.py
│   └── wind_manager.py
│
├── aerodynamics/
│   ├── force_manager.py
│   ├── body_drag.py
│   ├── rotor_drag.py
│   ├── rotor_wake.py
│   ├── aerodynamic_torque.py
│   └── force_models.py
│
├── excitation/
│   ├── action_excitation.py
│   ├── reference_perturbation.py
│   └── excitation_metrics.py
│
├── experiments/
│   ├── experiment_runner.py
│   ├── counterfactual_runner.py
│   ├── identifiability_runner.py
│   ├── ablation_runner.py
│   └── evaluation_split.py
│
├── data/
│   ├── data_logger.py
│   ├── resampler.py
│   ├── dataset_builder.py
│   ├── dataset_validator.py
│   ├── metadata.py
│   └── paired_dataset_builder.py
│
├── analysis/
│   ├── action_wind_correlation.py
│   ├── wind_identifiability.py
│   ├── counterfactual_analysis.py
│   └── diagnostics.py
│
├── tests/
│
└── generate_dataset.py
```

Adapt this architecture to the existing repository. Do not create duplicate functionality unnecessarily.

---

# 5. Domain Randomization

Implement domain-level randomization of UAV physical parameters.

A domain represents one fixed physical configuration.

Randomize configurable parameters such as:

| Parameter | Suggested randomization |
|---|---|
| Mass | ±50% of nominal |
| Moment of inertia | ±30% of nominal |
| Motor time constant | Uniform(0.01, 0.1) s |
| Drag coefficient | Uniform(0.1, 0.5) |
| Thrust coefficient | ±50% |
| Torque coefficient | ±50% |

These ranges are configurable experimental defaults, not mandatory physical ranges.

Requirements:

1. Generate a unique domain ID.
2. Sample parameters when creating a domain.
3. Keep physical parameters fixed across episodes in that domain.
4. Support reproducible random seeds.
5. Save all parameters in metadata.
6. Apply sampled parameters to the actual simulator physics.
7. Validate physical parameters before application.

**Important for identifiability experiments:**

Do not randomize physical parameters independently in every episode without recording their relationship to wind conditions.

Wind and physical-parameter variations must be experimentally separable.

Support the following modes:

- Fixed nominal physical parameters.
- Randomized physical parameters with wind held constant.
- Fixed physical parameters with wind randomized.
- Joint physical-parameter and wind randomization.

This separation is necessary to study whether the model confuses persistent physical differences with transient wind disturbances.

---

# 6. Trajectory Generation

Implement smooth and diverse 3D reference trajectories.

Use Gaussian Process trajectory generation with the periodic kernel:

\[
k(\tau)=\sigma^2
\exp\left(
-\frac{2\sin^2(\pi|\tau|/p)}{\ell^2}
\right)
\]

Use a composite kernel:

\[
k(\tau)=k_1(\tau)+k_2(\tau)+k_3(\tau)
\]

Suggested length scales:

| Component | Length scale |
|---|---:|
| Fast | 1.3 |
| Medium | 3.0 |
| Slow | 4.0 |

Axis-specific periods:

```python
periods = {
    "x": [37, 61, 13],
    "y": [17, 23, 52],
    "z": [19, 29, 53],
}
```

Implement:

- Periodic kernel.
- Composite kernel.
- Covariance computation.
- Numerical jitter.
- Cholesky-based sampling.
- Reproducible NumPy random generators.
- Configurable mean function.
- Position, velocity, and acceleration extraction.

Support trajectory types:

1. Hover.
2. Straight-line motion.
3. Circular trajectories.
4. Figure-eight trajectories.
5. Constant-velocity trajectories.
6. Acceleration and deceleration.
7. Yaw maneuvers.
8. Attitude-changing maneuvers.
9. Combined translational and rotational motion.
10. GP trajectories.

Enforce configurable workspace, altitude, velocity, and acceleration constraints.

The generated reference is not the actual UAV trajectory. The UAV must track it using the existing closed-loop controller.

---

# 7. Wind Generation Architecture

Implement a modular wind field:

\[
\mathbf u(\mathbf p,t)=
\mathbf u_0+
G(\mathbf p-\mathbf p_0)+
\mathbf u_{\mathrm{gust}}(t)+
\mathbf u_{\mathrm{turbulence}}(\mathbf p,t)
\]

Support:

### Steady wind

- Magnitude.
- Direction.
- World-frame components.
- Uniform and spatially varying wind.
- Configurable orientation.

Suggested default range: 0–10 m/s.

### Spatial gradients

\[
\mathbf u_{\mathrm{gradient}}(\mathbf p)
=G(\mathbf p-\mathbf p_0)
\]

where \(G\) is a configurable 3×3 matrix.

### Gusts

Implement:

- 1-cosine gust.
- Step gust.
- Sinusoidal gust.
- Random gust events.
- Ornstein–Uhlenbeck stochastic gusts.

### Turbulence

Support configurable turbulence models.

If implementing Dryden turbulence, document the assumptions, equations, filter structure, and parameter interpretation.

Do not label arbitrary Gaussian noise as physically validated atmospheric turbulence.

### Wind manager

Provide:

```python
wind_state = wind_manager.get_wind_state(
    position=position_world,
    time=simulation_time,
)
```

Return total wind and individual contributions.

Wind must be independently configurable from UAV physical parameters and reference trajectory generation.

---

# 8. Aerodynamic Force and Torque Integration

Integrate the existing modular force architecture.

Every force model must return its individual force and torque contributions.

Support independent toggles for:

- Gravity.
- Motor thrust.
- Body aerodynamic drag.
- Rotor aerodynamic drag.
- Rotor wake interaction.
- Wind-induced aerodynamic forces.
- Aerodynamic moments.

Avoid double-applying forces already handled natively by Isaac Sim.

## Relative air velocity

Aerodynamic forces must depend on air-relative velocity:

\[
\mathbf v_{\mathrm{rel}}^W=
\mathbf v_{\mathrm{UAV}}^W-\mathbf u_{\mathrm{wind}}^W
\]

Transform into body coordinates:

\[
\mathbf v_{\mathrm{rel}}^B=
R_{WB}^{T}\mathbf v_{\mathrm{rel}}^W
\]

A possible quadratic body-drag model is:

\[
\mathbf F_D^B=
-\frac{1}{2}\rho
\begin{bmatrix}
C_{Dx}A_x&0&0\\
0&C_{Dy}A_y&0\\
0&0&C_{Dz}A_z
\end{bmatrix}
\begin{bmatrix}
|v_x|v_x\\
|v_y|v_y\\
|v_z|v_z
\end{bmatrix}
\]

Clearly document all coordinate frames and approximations.

---

# 9. Closed-Loop Simulation

Use the existing controller.

For each physics step:

1. Retrieve actual UAV state.
2. Retrieve reference trajectory values.
3. Compute tracking errors.
4. Execute controller.
5. Obtain actuator commands.
6. Apply actuator commands.
7. Compute additional aerodynamic forces and torques.
8. Apply external forces and torques.
9. Step the physics simulation.
10. Retrieve actual resulting UAV state.
11. Log observations and ground truth.

Do not directly force the UAV to follow reference positions.

Record reference state and actual state separately.

Handle controller and physics update frequencies correctly.

---

# 10. NEW: Counterfactual Wind Experiment Architecture

This is a critical addition.

The dataset must contain controlled counterfactual experiments in which the same underlying UAV flight scenario is evaluated under different wind conditions.

The objective is to estimate:

**What would have happened to the UAV if the wind had been different, while the other experimental conditions remained unchanged?**

## 10.1 Counterfactual experiment definition

Define a scenario using:

```python
scenario = {
    "scenario_id": "...",
    "domain_id": "...",
    "initial_state": {},
    "reference_trajectory": {},
    "controller_configuration": {},
    "initial_actuator_state": {},
    "random_seed": 42,
}
```

For each scenario, generate multiple paired episodes:

```text
Scenario 001
│
├── Episode A: No wind
├── Episode B: Constant wind +X
├── Episode C: Constant wind -X
├── Episode D: Constant wind +Y
├── Episode E: Constant wind +Z
├── Episode F: Time-varying gust
└── Episode G: Turbulent wind
```

The following must remain identical within a counterfactual group:

- UAV physical parameters.
- Initial UAV state.
- Initial actuator state.
- Reference trajectory.
- Controller configuration.
- Simulation configuration.
- Relevant random seeds and stochastic disturbances unrelated to wind.

Only the intended wind intervention should change.

Use a unique `counterfactual_group_id` to connect the paired episodes.

## 10.2 Two complementary counterfactual modes

Implement two distinct modes. Do not treat them as equivalent.

### Mode A: Closed-loop counterfactual experiments

Use the same initial condition, reference trajectory, controller, and physical parameters.

Run the controller independently in each wind condition.

The controller will naturally produce different actuator commands because the UAV states diverge.

This experiment measures the combined effect of wind and feedback-controller response.

Record:

- Wind condition.
- Actual state.
- Controller output.
- Motor commands.
- Tracking error.
- Aerodynamic forces.
- Resulting state differences.

### Mode B: Fixed-action counterfactual experiments

Replay the same actuator-action sequence under multiple wind conditions.

Keep initial state and physical parameters identical.

The action sequence must be generated from a baseline trajectory or a separately defined excitation sequence.

Do not allow the controller to recompute actions independently during this replay experiment.

This isolates the effect of changing wind under identical control inputs.

If the simulator cannot replay actuator commands accurately, identify the limitation and implement the smallest compatible solution.

## 10.3 Counterfactual wind interventions

Support interventions such as:

| Baseline | Counterfactual |
|---|---|
| Zero wind | Constant wind |
| +X wind | -X wind |
| +Y wind | +X wind |
| Constant wind | Time-varying gust |
| No gust | 1-cosine gust |
| Mild gust | Stronger gust |
| Uniform wind | Spatial gradient |
| Smooth wind | Turbulent wind |

Support configurable wind intervention magnitudes and directions.

The exact wind parameters must be saved in episode metadata.

## 10.4 Counterfactual comparison metrics

Calculate paired differences:

\[
\Delta s_t^{(i,j)}
=s_t^{(i)}-s_t^{(j)}
\]

\[
\Delta v_t^{(i,j)}
=v_t^{(i)}-v_t^{(j)}
\]

\[
\Delta a_t^{(i,j)}
=a_t^{(i)}-a_t^{(j)}
\]

Calculate:

- Position divergence.
- Velocity divergence.
- Attitude divergence.
- Angular-velocity divergence.
- Tracking-error difference.
- Controller-action difference.
- Force difference.
- Prediction-error difference.

For fixed-action experiments, compare state differences under identical actions.

For closed-loop experiments, separately analyze state differences and controller-action differences.

Do not interpret closed-loop paired differences as the pure direct effect of wind because feedback actions also change.

## 10.5 Counterfactual dataset schema

Include:

```python
{
    "scenario_id": "...",
    "counterfactual_group_id": "...",
    "counterfactual_episode_id": "...",
    "wind_intervention_id": "...",
    "wind_intervention_type": "...",
    "counterfactual_mode": "closed_loop",
    "paired_episode_id": "...",
}
```

The dataset builder must be able to retrieve all episodes belonging to one counterfactual group.

---

# 11. NEW: Action Decorrelation and Excitation Architecture

A major experimental risk is that the controller's actions and the wind disturbances become correlated.

For example, when a strong wind pushes the UAV in one direction, the controller may consistently command the opposite direction.

A model might learn a misleading association between control actions and resulting dynamics rather than learning how hidden wind disturbances affect the UAV.

The dataset must explicitly investigate and reduce this problem.

## 11.1 Separate reference motion from wind generation

The trajectory generator and wind generator must use independent random streams.

Do not generate wind based on the reference trajectory or controller output.

Do not use the same random seed stream for both without explicit justification.

Record independent seeds for:

- Trajectory generation.
- Wind generation.
- Physical parameter randomization.
- Initial conditions.
- Action excitation.
- Other stochastic processes.

The wind profile must be generated independently of the planned reference trajectory.

## 11.2 Action excitation

Implement configurable excitation strategies to provide diverse control inputs.

Support:

1. Reference trajectory diversity.
2. Bounded position-reference perturbations.
3. Bounded velocity-reference perturbations.
4. Smooth sinusoidal excitation.
5. Multisine excitation.
6. Chirp excitation.
7. Pseudorandom binary sequences (PRBS), where compatible with the controller and safety limits.
8. Short acceleration and deceleration maneuvers.
9. Independent yaw excitation.
10. Combined translation and attitude excitation.

The objective is not to inject arbitrary noise into motor commands.

Prefer physically meaningful, bounded excitation through reference commands or a dedicated action interface.

All excitation must respect configured actuator, workspace, velocity, acceleration, and attitude limits.

## 11.3 Independent excitation experiments

Create separate experiment groups:

**Experiment A: No additional excitation**

Only the nominal controller tracks the selected trajectory.

**Experiment B: Diverse reference trajectories**

Use varied reference motion to generate different control actions.

**Experiment C: Smooth reference excitation**

Add bounded smooth perturbations to references.

**Experiment D: Multisine/chirp excitation**

Use controlled frequency variation to excite different dynamic responses.

**Experiment E: Wind-independent action excitation**

Generate a fixed action sequence independently of wind and replay it across multiple wind conditions.

**Experiment F: Combined excitation**

Combine varied trajectories and bounded excitation with multiple wind conditions.

Do not assume that more action noise automatically improves identifiability.

## 11.4 Action decorrelation diagnostics

Log and analyze:

- Motor commands.
- Total thrust command.
- Body torque commands.
- Reference position and velocity.
- Actual position and velocity.
- Wind velocity.
- Wind acceleration or temporal derivative, where defined.
- Wind-induced aerodynamic force.
- Controller tracking errors.

Compute:

1. Pearson correlation between action components and wind components.
2. Spearman rank correlation.
3. Cross-correlation over multiple time lags.
4. Correlation between action and wind-induced force.
5. Correlation between reference trajectory and wind.
6. Action distribution and frequency spectrum.
7. Action covariance matrix.
8. Singular values and condition number of the action excitation matrix.

Use lagged correlations because wind can affect the UAV before the controller responds.

Do not rely on a single zero-lag correlation value.

Report metrics separately for:

- Entire dataset.
- Individual wind profiles.
- Individual domains.
- Trajectory types.
- Counterfactual experiment groups.

Important: Low correlation alone does not prove causal independence. These diagnostics are intended to reveal potential confounding, not certify it away.

## 11.5 Dataset balancing

Avoid allowing a small number of trajectories, wind directions, or controller responses to dominate the dataset.

Use configurable balancing across:

- Wind magnitude.
- Wind direction.
- Gust type.
- Gust onset.
- Trajectory type.
- Velocity range.
- Acceleration range.
- UAV domain.
- Action excitation type.

Save a dataset coverage report.

---

# 12. NEW: Wind Identifiability and Observability Experiments

The objective is to determine whether hidden wind disturbances can be inferred from the observable state-action history.

This is not guaranteed merely because wind ground truth is available in simulation.

Wind can be confounded with:

- Mass variation.
- Inertia variation.
- Drag-coefficient variation.
- Thrust-coefficient variation.
- Motor dynamics.
- Controller behavior.
- Unmodeled aerodynamic forces.
- State-estimation errors.

Design the dataset to investigate these ambiguities explicitly.

## 12.1 Define the inference problem

The primary world-model observation should contain only intended observable variables.

For example:

\[
o_t =
[
p_t,\,
v_t,\,
q_t,\,
\omega_t,\,
a_t
]
\]

where the exact state representation must be consistent with the existing project.

The action history is:

\[
a_{t-H+1:t}
\]

The hidden disturbance includes:

\[
d_t =
[
u_{\mathrm{wind},t},\,
F_{\mathrm{aero},t},\,
\tau_{\mathrm{aero},t}
]
\]

The world model should not receive \(d_t\) as an input in the primary experiment.

Save \(d_t\) as auxiliary ground truth.

Investigate whether a model can infer useful latent disturbance information from a history of observations and actions.

## 12.2 Controlled identifiability experiment matrix

Implement the following experiment groups.

### Group A: Wind identifiability with known physics

- Fix mass and inertia.
- Fix motor and thrust parameters.
- Fix aerodynamic coefficients.
- Vary only wind.
- Use multiple trajectories and action excitation patterns.

This measures the best-case wind inference setting.

### Group B: Physical parameter uncertainty without wind

- Use zero wind.
- Randomize mass, inertia, drag, and motor parameters.
- Test whether the model incorrectly attributes physical variation to wind.

### Group C: Wind variation with fixed physics

- Keep all physical parameters fixed.
- Vary wind magnitude, direction, and temporal structure.
- Measure whether wind is distinguishable from nominal dynamics.

### Group D: Joint wind and physical variation

- Randomize wind and physical parameters.
- Evaluate whether the model can distinguish transient disturbance effects from persistent domain changes.

### Group E: Wind and aerodynamic model uncertainty

- Change the enabled aerodynamic force models.
- Compare wind inference under different model assumptions.
- Determine whether unmodeled force contributions create misleading wind estimates.

### Group F: Unseen wind profiles

- Train on selected wind profiles.
- Evaluate on unseen directions, magnitudes, gradients, and gust characteristics.

Keep these experiment groups separately identifiable in metadata.

## 12.3 Wind identifiability conditions

Ensure that the dataset includes sufficient dynamic excitation.

Examples:

- Hover under different wind directions.
- Forward translation under headwind and tailwind.
- Lateral translation under crosswind.
- Acceleration and deceleration.
- Yaw rotations under wind.
- Changing attitude under steady wind.
- Time-varying gust exposure.
- Different motion directions under the same wind.
- The same motion under different wind conditions.

The same wind should be experienced under different trajectories, and similar trajectories should be experienced under different wind conditions.

This is essential to reduce confounding between motion and wind.

## 12.4 Identifiability diagnostics

Implement an analysis module that reports:

**A. Excitation coverage**

- State distribution.
- Velocity distribution.
- Acceleration distribution.
- Angular-velocity distribution.
- Action distribution.
- Action covariance and singular values.

**B. Wind coverage**

- Wind magnitude distribution.
- Wind direction distribution.
- Gust duration distribution.
- Wind temporal-frequency characteristics.
- Spatial gradient coverage.

**C. Confounding diagnostics**

- Wind versus action correlation.
- Wind versus acceleration correlation.
- Wind versus reference trajectory correlation.
- Wind versus domain parameter correlation.
- Wind-induced force versus total acceleration correlation.

**D. Local sensitivity analysis**

Where feasible, estimate how observable state transitions change when wind parameters change while other conditions are held fixed.

For a selected parameter vector \(\theta\), estimate a local sensitivity matrix:

\[
J_\theta =
\frac{\partial y}{\partial \theta}
\]

using finite differences or other suitable numerical methods.

Here, \(y\) may represent the next-state prediction or a short-horizon state trajectory.

Use this to investigate whether different wind parameters produce distinguishable state responses.

**E. Rank and conditioning**

Analyze the singular values and condition number of suitable sensitivity or regression matrices.

A poorly conditioned matrix can indicate that different parameters have similar effects on observations.

Do not claim full physical identifiability solely from these numerical diagnostics. Clearly state their assumptions and limitations.

## 12.5 Optional wind inference benchmark

Create an auxiliary benchmark for estimating hidden wind from history.

Possible targets:

- Current wind vector.
- Short-term average wind.
- Wind magnitude and direction.
- Gust onset or gust intensity.
- Wind-induced force.

Use separate labels and metrics.

Evaluate:

- Wind-vector RMSE.
- Wind-direction error.
- Wind-magnitude error.
- Gust detection precision and recall.
- Error under unseen wind conditions.

This is an auxiliary research benchmark. It must not leak wind ground truth into the primary dynamics-model input.

---

# 13. Episode and Dataset Generation Strategy

Use hierarchical generation:

```text
Experiment
│
├── Physical Domain
│   │
│   ├── Scenario
│   │   ├── Wind condition A
│   │   ├── Wind condition B
│   │   └── Wind condition C
│   │
│   └── Scenario ...
│
└── Other Physical Domains
```

A domain represents a fixed UAV physical configuration.

A scenario defines a shared experimental setup, such as:

- Initial state.
- Reference trajectory.
- Controller configuration.
- Action excitation.
- Random seeds.

An episode represents one simulation realization with a selected wind intervention.

Use configurable experiment counts and episode duration.

Ensure that counterfactual groups remain intact during dataset construction and splitting.

---

# 14. Data Logging Architecture

Log high-frequency simulation data.

### State

- Timestamp.
- Position.
- Linear velocity.
- Quaternion orientation.
- Angular velocity.
- Linear acceleration, if available.
- Angular acceleration, if available.
- Motor angular velocities.
- Motor commands.

### Controller

- Desired position.
- Desired velocity.
- Desired acceleration.
- Tracking errors.
- Controller outputs.
- Motor thrust commands.

### Wind

- Total wind in world coordinates.
- Wind in body coordinates.
- Steady wind.
- Spatial gradient.
- Gust contribution.
- Turbulence contribution.
- Wind magnitude and direction.

### Forces and torques

- Gravity, where accessible.
- Motor thrust.
- Body drag.
- Rotor drag.
- Wind-related force.
- Rotor wake contribution.
- Aerodynamic torques.
- Total external force.
- Total external torque.

### Experimental metadata

Include:

```python
{
    "experiment_id": "...",
    "domain_id": "...",
    "scenario_id": "...",
    "episode_id": "...",
    "counterfactual_group_id": "...",
    "counterfactual_mode": "...",
    "wind_intervention_id": "...",
    "trajectory_type": "...",
    "action_excitation_type": "...",
    "random_seed": 42,
    "uav_parameters": {},
    "wind_parameters": {},
    "controller_parameters": {},
    "simulation_parameters": {},
    "trajectory_parameters": {},
    "start_time": 0.0,
    "duration": 10.0,
    "episode_status": "completed"
}
```

Use Parquet for tabular data and JSON/YAML for metadata. Support optional NPZ output.

Never silently discard invalid samples.

---

# 15. Resampling and Model-Ready Dataset

Preserve original high-frequency logs.

Initially target a configurable 20 Hz dataset:

\[
\Delta t=0.05\text{ s}
\]

Implement:

- Uniform timestamps.
- State/action synchronization.
- Appropriate continuous-variable interpolation.
- Quaternion SLERP.
- Correct handling of angular variables.
- Anti-alias filtering before downsampling when required.
- No interpolation across episode boundaries or invalid gaps.

Build model-ready samples:

```python
{
    "state": ...,
    "action": ...,
    "next_state": ...,
    "timestamp": ...,
    "episode_id": ...,
    "domain_id": ...,
    "scenario_id": ...
}
```

Support historical input sequences and future prediction targets.

Support prediction horizons of:

- 1 step.
- 5 steps.
- 10 steps.
- 20 steps.

At 20 Hz these correspond to 0.05, 0.25, 0.5, and 1.0 seconds.

Keep wind, forces, physical parameters, and counterfactual metadata in auxiliary fields or a separate linked dataset.

Explicitly exclude auxiliary ground truth from the primary model-input pipeline.

---

# 16. Train, Validation, and Test Splits

Do not randomly split adjacent time samples.

Use episode-, scenario-, counterfactual-group-, and domain-aware splits.

Implement:

### Experiment A: Unseen wind

Train on selected profiles and test on unseen profiles.

### Experiment B: OOD wind magnitude

Train on 0–4 m/s and test on greater than 4 m/s up to 8 m/s.

### Experiment C: Unseen gust characteristics

Hold out selected gust amplitudes and durations.

### Experiment D: Physical parameter generalization

Hold out entire UAV domains.

### Experiment E: Counterfactual generalization

Keep all members of a counterfactual group in the same split.

Do not place one paired episode in training and its counterfactual counterpart in testing.

### Experiment F: Action excitation generalization

Hold out selected excitation types or frequency ranges.

### Experiment G: Joint OOD

Hold out combinations of wind profiles, physical parameters, and action excitation patterns.

Save all split configurations and seeds.

---

# 17. Ablation and Evaluation Support

Support independent force-model toggles and comparisons:

- Nominal physical parameters.
- Randomized physical parameters.
- Steady wind.
- Steady wind plus gusts.
- Steady wind plus gusts and turbulence.
- Different aerodynamic force models.
- Different trajectory strategies.
- Different action excitation strategies.

Also compare:

1. Dataset with and without counterfactual pairs.
2. Dataset with and without action excitation.
3. Fixed-action versus closed-loop counterfactual experiments.
4. Fixed physics versus randomized physics.
5. Wind-only variation versus joint wind and physics variation.
6. Diverse action trajectories versus narrowly distributed actions.

For each experiment, report:

- One-step prediction error.
- Multi-step prediction error.
- Error growth with prediction horizon.
- Position and velocity RMSE.
- Attitude error.
- Performance under OOD wind.
- Performance across physical domains.
- Counterfactual prediction consistency.
- Auxiliary wind-inference performance, if implemented.

Keep the evaluation protocol consistent across experiments.

---

# 18. Validation and Testing

Implement unit and integration tests.

### UAV configuration tests

- Exact nominal parameter loading.
- Inertia tensor validity.
- Rotor count and configuration consistency.
- Correct interpretation of rotor parameters.
- Correct application to simulator physics.

### Wind tests

- Wind magnitude and direction.
- Gradient calculation.
- Gust timing.
- OU behavior.
- Reproducibility.

### Counterfactual tests

- Paired episodes have identical initial conditions.
- Paired episodes have identical physical parameters.
- Reference trajectories are identical.
- Wind interventions are different as configured.
- Fixed-action mode replays identical actions.
- Closed-loop mode records controller-action differences.
- Group IDs and paired episode IDs are correct.

### Action excitation tests

- Independent random streams.
- Bounded excitation.
- Correct frequency content.
- Reproducibility.
- No actuator or workspace constraint violations.

### Identifiability tests

- Correct calculation of correlations.
- Correct lagged cross-correlation.
- Correct action covariance and singular values.
- Correct sensitivity calculations.
- Detection of deliberately confounded experimental configurations.

### Dataset tests

- Correct timestamps.
- Correct state/action alignment.
- No episode-boundary crossing.
- No counterfactual-group leakage.
- Correct auxiliary and primary input separation.
- Correct metadata.

Run a small end-to-end experiment before scaling.

Generate diagnostic plots for:

1. 3D reference and actual trajectories.
2. Position and velocity.
3. Wind profiles.
4. Individual aerodynamic forces.
5. Total force and torque.
6. Tracking errors.
7. High-frequency and resampled signals.
8. Counterfactual trajectory comparisons.
9. Action and wind correlations.
10. Action frequency spectra.
11. Wind and physical-parameter coverage.
12. Identifiability sensitivity matrices.

---

# 19. Performance and Engineering Requirements

Use:

- Python type hints.
- Dataclasses or structured configurations.
- YAML configuration.
- NumPy for numerical computations.
- PyTorch only where useful.
- Buffered logging.
- Progress reporting.
- Checkpointing.
- Recovery from failed episodes.
- Reproducible random generators.
- Explicit SI units and coordinate frames.

Do not use unsafe concurrent access to one Isaac Sim instance.

Parallelize only across independent simulation instances where supported.

Avoid loading the complete dataset into memory.

Do not introduce mock simulation results into the actual dataset.

---

# 20. Expected Deliverables

Provide:

1. Complete modular Python implementation integrated into the existing repository.
2. YAML configuration files.
3. Main dataset-generation script.
4. Small validated sample dataset.
5. Counterfactual experiment runner.
6. Action excitation and decorrelation analysis.
7. Wind identifiability experiment runner.
8. Dataset diagnostics and visualizations.
9. Unit and integration tests.
10. README with architecture, mathematical models, execution, dataset schema, and troubleshooting.

Example execution:

```bash
python generate_dataset.py --config configs/experiments.yaml
```

Adapt to the actual Isaac Sim launch requirements.

---

# 21. Incremental Implementation Strategy

Do not generate the entire architecture as one unverified code dump.

Follow these phases:

**Phase 1: Repository analysis**

Inspect the repository and identify reusable components.

**Phase 2: Independent modules**

Implement and test domain randomization, GP trajectories, wind generation, and logging.

**Phase 3: Physics integration**

Integrate aerodynamic forces and wind into the simulator.

**Phase 4: Closed-loop simulation**

Integrate trajectory generation with the existing controller.

**Phase 5: Counterfactual experiments**

Implement paired closed-loop and fixed-action wind interventions.

**Phase 6: Action excitation**

Implement independent action excitation and correlation diagnostics.

**Phase 7: Identifiability experiments**

Implement controlled experiment groups and sensitivity/conditioning analysis.

**Phase 8: Dataset generation**

Implement logging, resampling, dataset construction, and leakage-safe splits.

**Phase 9: Evaluation**

Implement OOD, counterfactual, action-decorrelation, and identifiability evaluations.

**Phase 10: Documentation**

Provide complete usage instructions and example experiments.

After each phase:

- List created and modified files.
- Explain implementation decisions.
- Run relevant tests.
- Report failures.
- Fix failures before moving on.

Do not claim a component works unless it has actually been tested.

---

# 22. Final Research Requirements

The implementation must preserve the central research objective:

**Learn UAV dynamics under hidden, time-varying wind disturbances.**

The dataset must allow investigation of:

1. How much wind diversity is necessary for OOD generalization?
2. How do aerodynamic models affect prediction accuracy?
3. Can hidden wind disturbances be inferred from state-action histories?
4. How do persistent physical parameter variations differ from transient wind disturbances?
5. How does prediction error grow with prediction horizon?
6. Does the model generalize to unseen gusts and wind directions?
7. What would the same flight have looked like under a different wind condition?
8. Does action–wind correlation create misleading learned dynamics?
9. Which motion patterns make wind easier or harder to identify?
10. When are wind and physical parameters fundamentally difficult to distinguish using the available observations?

**Most important implementation principles:**

- Physical correctness over dataset size.
- Controlled experiments over uncontrolled randomization.
- Counterfactual pairing over unrelated episode comparisons.
- Independent action and wind generation.
- Explicit investigation of confounding and identifiability.
- Strict separation of observable model inputs and simulator ground truth.
- Reproducibility and data integrity.
- Reuse of the existing simulator and repository architecture.

First understand the existing repository, then implement this architecture incrementally. Do not replace working simulation, controller, or force-model components without a clear technical reason.