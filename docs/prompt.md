# Project-Wide Engineering Audit and Implementation: PhysiWorld UAV

## 1. Objective

Perform a comprehensive architectural, scientific, mathematical and implementation-level review of the entire PhysiWorld UAV project.

The primary objective is to develop a **modular, physically meaningful, computationally efficient and reproducible UAV data-generation framework** for training world models.

The generated dataset should allow us to train models that learn UAV dynamics and predict future states under:

- Different physical configurations and parameter variations.
- Different flight scenarios and trajectories.
- Different environmental and wind conditions.
- In-distribution and out-of-distribution physical conditions.
- Different control strategies and input sequences.

The simulator should generate sufficiently diverse, informative and physically consistent trajectories to support system identification, dynamics learning, counterfactual prediction and long-horizon world-model training.

We are using physics-based approximations instead of running CFD for every experiment because of computational constraints. Therefore, every approximation must be scientifically justified, validated and used only within its applicable operating conditions.

**Do not blindly implement the specifications below. First inspect the existing project, understand its architecture, identify existing functionality and evaluate the scientific assumptions.**

---

# 2. Mandatory Working Instructions

Before modifying anything:

1. Inspect the complete repository structure.
2. Read all relevant documentation, particularly:
   - `docs/requirement.md`
   - `docs/datasets/problems/identifiability.md`
   - Existing physics and aerodynamics documentation.
   - Existing dataset-generation documentation.
3. Inspect all modules under `src/physiworld_uav`.
4. Inspect all configuration files under `config/`.
5. Inspect existing tests, scripts, simulation environments and dataset-generation pipelines.
6. Identify the Isaac Sim, Isaac Lab and OmniDrones versions and the APIs currently being used.
7. Understand the existing implementation before proposing architectural changes.

Do not rewrite existing functionality simply because another implementation is possible.

Prefer existing, tested implementations from Isaac Sim, Isaac Lab, OmniDrones, PyTorch and established scientific libraries.

Do not duplicate functionality already provided by these frameworks.

### Important decision-making rules

- Preserve existing functionality unless there is a scientifically or technically justified reason to change it.
- Do not assume that a physical model is correct simply because it produces stable simulation results.
- Do not introduce unnecessary abstractions, configuration files or dependencies.
- Follow DRY, SOLID, separation of concerns and object-oriented design principles where appropriate.
- Avoid overengineering.
- Document all important assumptions, limitations and design decisions.
- Never silently resolve scientific ambiguities.
- If a decision affects physical correctness, dataset validity or research conclusions, explain the alternatives and ask me before proceeding.
- Do not overwrite existing documentation or requirements without reviewing the proposed changes.
- Do not fabricate validation results or claim that an experiment has passed without actually running it.

Work in phases. First provide an audit and implementation plan. Do not start large-scale implementation until I approve the proposed design and unresolved scientific decisions.

---

# 3. Configuration Architecture

The entire data-generation process must be configurable.

I should be able to define an experiment or dataset using a small number of configuration files by reusing templates and modular configuration components.

Avoid duplicating configurations for every scenario or domain.

Design a hierarchical configuration system that supports:

- Shared default configurations.
- Reusable configuration templates.
- Experiment-specific overrides.
- Domain-specific overrides.
- Scenario-specific overrides.
- Reproducible randomization.

Consider using Hydra/OmegaConf if appropriate, but first check the existing configuration system.

### Proposed configuration responsibilities

Organize configuration responsibilities into logically separated components.

**A. Drone models and physical parameters**

Define drone-specific properties:

- Mass.
- Inertia tensor.
- Center of mass.
- Geometry and dimensions.
- Arm lengths.
- Rotor positions and orientations.
- Motor properties.
- Rotor thrust coefficients.
- Rotor torque coefficients.
- Motor time constants.
- Maximum and minimum RPM.
- Aerodynamic coefficients.
- Other relevant physical parameters.

Support multiple drone configurations without duplicating physics implementations.

**B. Force and torque models**

Independently configure each physical contribution:

- Gravity.
- Rotor thrust.
- Rotor reaction torque.
- Body aerodynamic drag.
- Arm aerodynamic drag.
- Motor aerodynamic drag.
- Translational aerodynamic forces.
- Lift and side forces, where physically justified.
- Aerodynamic moments.
- Rotational aerodynamic damping.
- Rotor aerodynamic effects.
- Ground effect, if implemented.
- Other relevant aerodynamic contributions.

Every force model should be independently configurable and enableable/disablable for ablation studies.

**C. Wind configuration**

Support reusable wind profiles:

- No wind.
- Constant wind.
- Directional wind.
- Spatial wind gradients.
- Linear and nonlinear wind fields.
- Step changes.
- Gusts.
- Impulsive gusts.
- Sinusoidal gusts.
- Periodic gusts.
- Dryden turbulence.
- Stochastic turbulence.
- Spatially varying turbulence.
- Combined wind profiles.

Each profile should support configurable:

- Magnitude.
- Direction.
- Frequency.
- Duration.
- Spatial distribution.
- Temporal distribution.
- Random seed.
- Other model-specific parameters.

**D. Domain randomization**

Randomize relevant physical parameters, including:

- Mass.
- Inertia.
- Drag coefficients.
- Aerodynamic coefficients.
- Motor time constants.
- Thrust coefficients.
- Torque coefficients.
- Rotor parameters.
- Other relevant physical parameters.

Support different distributions:

- Uniform.
- Normal.
- Log-normal.
- Truncated distributions.
- Custom distributions.

Allow parameter correlations where physically necessary.

Do not independently randomize physically coupled parameters without considering their relationships.

Define nominal, ID and OOD parameter ranges.

**E. Scenario configuration**

Define reusable scenario templates and their parameters.

**F. Initial conditions**

Configure initial:

- Position.
- Orientation.
- Linear velocity.
- Angular velocity.
- Motor states.
- Other required states.

**G. Counterfactual experiments**

Support paired experiments where the initial conditions and control inputs are identical but one or more physical or environmental parameters are changed.

**H. Dataset configuration**

Configure:

- Dataset name.
- Version.
- Split strategy.
- Number of trajectories.
- Simulation duration.
- Sampling frequency.
- Output fields.
- Storage format.
- Compression.
- Random seeds.
- Metadata.
- Validation settings.

The configuration system should allow an experiment to reference shared templates and override only the necessary parameters.

---

# 4. Domain Randomization and Dataset Domains

Design a systematic domain-generation strategy.

Do not simply generate random parameter combinations without considering identifiability, coverage and physical plausibility.

Define separate domains for:

1. Training.
2. In-distribution validation.
3. In-distribution testing.
4. Out-of-distribution validation.
5. Out-of-distribution testing.

OOD conditions should be explicitly held out before training data generation.

Consider multiple OOD categories:

- Unseen physical parameter ranges.
- Unseen combinations of known parameters.
- Unseen wind magnitudes.
- Unseen wind directions.
- Unseen wind profiles.
- Unseen turbulence characteristics.
- Unseen trajectory families.
- Unseen initial conditions.
- Combinations of multiple OOD factors.

Prevent accidental OOD leakage into training data.

Maintain experiment-level and domain-level separation rather than relying on random splitting of individual time samples.

Investigate whether factorial designs, Latin Hypercube Sampling, Sobol sampling or other space-filling techniques would improve parameter coverage.

Use appropriate sampling methods based on the physical parameter distributions and experimental objectives.

Store the exact sampled parameter values for every experiment.

---

# 5. Flight Scenarios and Trajectory Generation

Create a modular scenario-generation framework.

At minimum, support the following scenarios.

### A. Basic flight scenarios

- Hover.
- Vertical ascent.
- Vertical descent.
- Forward translation.
- Backward translation.
- Lateral translation.
- Straight-line trajectories.
- Constant-velocity flight.
- Acceleration and deceleration.
- Stationary attitude changes.

### B. Complex trajectories

- Circular trajectories.
- Figure-eight trajectories.
- Sinusoidal trajectories.
- Helical trajectories.
- Combined translation and rotation.
- Trajectories with simultaneous changes in position, orientation and velocity.

### C. Attitude-focused experiments

- Pure yaw.
- Pure roll.
- Pure pitch.
- Combined roll, pitch and yaw.
- Attitude transitions.
- Angular acceleration and deceleration.
- Different angular velocity profiles.

### D. Initial and terminal conditions

Allow configurable:

- Starting position.
- Ending position.
- Starting orientation.
- Ending orientation.
- Initial linear velocity.
- Initial angular velocity.
- Desired terminal velocity.
- Desired terminal acceleration.
- Desired angular velocity.

Support different trajectory durations, frequencies, amplitudes and motion profiles.

Ensure that generated trajectories are physically feasible given the drone's actuator limits and dynamic constraints.

Do not assume that arbitrary position and attitude trajectories are feasible.

Validate trajectory continuity, velocity, acceleration, jerk and actuator feasibility where applicable.

---

# 6. Closed-Loop and Open-Loop Data Generation

This is a critical requirement.

For every relevant physical domain and scenario, generate both closed-loop and open-loop experiments.

## A. Closed-loop experiments

Use appropriate flight controllers to track reference trajectories.

Before implementing any controller, inspect Isaac Lab and OmniDrones for existing implementations.

Prefer existing controllers such as:

- Lee position controller.
- Attitude controller.
- Rate controller.
- Other appropriate controllers.

Do not reimplement an existing controller unless necessary.

Record:

- Reference trajectory.
- Controller type.
- Controller parameters.
- Control errors.
- Motor commands.
- Actual motor states.
- Resulting forces and torques.

The controller should operate on the appropriate state estimates and respect actuator limitations.

## B. Open-loop experiments

Generate experiments where predefined control inputs are applied without feedback from a tracking controller.

Include:

- Constant motor commands.
- Constant collective thrust.
- Constant roll/pitch/yaw control inputs, where applicable.
- Step inputs.
- Pulse inputs.
- Other informative excitation signals.

Record the complete free response.

Clarify and distinguish between:

1. Constant rotor-speed or motor-command inputs.
2. Constant collective and differential control inputs.
3. Constant body-wrench inputs, if supported as a separate experimental abstraction.

Do not treat these as physically equivalent.

Include experiments initialized from appropriate equilibrium conditions, particularly hover, wherever useful.

Evaluate the open-loop excitation design for parameter identifiability.

Refer specifically to:

`docs/datasets/problems/identifiability.md`

Use the problems described in this document to design experiments that distinguish the effects of different physical parameters.

For example, determine which experiments are needed to identify:

- Mass.
- Inertia.
- Translational drag.
- Rotational damping.
- Motor time constants.
- Thrust coefficients.
- Torque coefficients.
- Wind effects.

Avoid datasets where different physical parameters produce indistinguishable responses.

Evaluate excitation richness, observability and parameter identifiability using appropriate mathematical methods.

---

# 7. Aerodynamic and Physical Model Validation

We cannot run CFD for every trajectory.

Therefore, implement computationally efficient aerodynamic approximations while maintaining physically meaningful behavior.

This section requires particular scientific care.

First inspect the existing force and torque architecture and identify which models are already implemented.

Then evaluate each model independently.

## A. Relative wind velocity

Aerodynamic forces must be based on the local relative air velocity.

Account for:

- Drone translational velocity.
- Wind velocity.
- Drone orientation.
- Angular velocity.
- Position-dependent wind.
- Component-specific local velocity where necessary.

Use consistent coordinate transformations.

Do not simply add wind velocity to the drone's state or calculate aerodynamic drag using ground-relative velocity.

## B. Body aerodynamic forces

Evaluate the existing lumped aerodynamic drag model.

Determine whether the current representation is sufficient for the intended scenarios.

Consider:

- Body-frame diagonal drag models.
- Full aerodynamic coefficient matrices.
- Orientation-dependent drag.
- Effective projected areas.
- Drag coefficient variation.
- Cross-axis aerodynamic effects.

Do not assume that diagonal drag coefficients are always valid.

Explain when a diagonal approximation is sufficient and when it becomes inadequate.

## C. Component-level aerodynamic forces

Evaluate the feasibility of separately modeling:

- Main body.
- Arms.
- Motors.
- Other significant exposed components.

Determine whether component-level models provide sufficient benefits relative to their computational costs.

Avoid unnecessarily detailed geometry calculations if they do not meaningfully improve the generated dynamics.

## D. Aerodynamic moments

Evaluate:

- Moments caused by distributed aerodynamic forces.
- Rotational damping.
- Moments caused by asymmetric drag.
- Rotor-related aerodynamic effects.

Ensure that forces applied at different locations generate the correct moments.

## E. Rotor aerodynamics

Evaluate which rotor aerodynamic effects are necessary for our research.

Consider:

- Thrust and reaction torque.
- Rotor drag.
- Axial inflow effects.
- Ground effect.
- Rotor interaction.
- Wind effects on rotor thrust.

Clearly distinguish effects already modeled by the simulator from additional effects that need to be implemented.

## F. Model applicability

For every physical model, document:

1. Mathematical formulation.
2. Physical assumptions.
3. Required parameters.
4. Coordinate frame.
5. Units.
6. Computational complexity.
7. Expected operating range.
8. Applicable scenarios.
9. Scenarios where the approximation may fail.
10. Validation method.
11. Limitations.

Do not use a model outside its valid operating conditions without explicitly documenting the approximation.

For example, a simple quadratic drag model should not automatically be assumed valid for every flow regime, rotor operating condition or orientation.

## G. Validation

Where possible, compare approximations against:

- Analytical solutions.
- Simplified reference models.
- Experimental data.
- Published aerodynamic models.
- CFD reference results.
- Isaac Sim or other simulator reference behavior.

Create isolated tests for individual force contributions before evaluating the complete drone dynamics.

Include ablation experiments to understand the contribution of each model to ID and OOD prediction performance.

---

# 8. Simulator Physics and Double-Counting Prevention

This is a mandatory audit.

Inspect the Isaac Sim and Isaac Lab rigid-body dynamics implementation.

Identify exactly which physical effects are already applied by the simulator.

In particular, investigate:

- Gravity.
- Rigid-body mass.
- Inertia.
- External forces.
- External torques.
- Articulation dynamics.
- Joint constraints.
- Rotor or actuator implementations.
- Motor dynamics.
- Contact forces.
- Damping.
- Other built-in effects.

Do not apply gravity twice.

Do not apply rotor thrust twice.

Do not duplicate existing motor dynamics.

Do not manually integrate dynamics that are already integrated by the physics engine.

Clearly distinguish:

1. Forces internally computed by the physics solver.
2. Forces computed by our custom physics modules.
3. Forces applied to the simulator.
4. Forces and torques recorded for dataset generation.

Create a documented force-accounting architecture.

Verify that the resultant physical dynamics are consistent with the intended equations of motion.

---

# 9. Coordinate Frames, Transformations and Units

Audit every module for coordinate-frame and unit consistency.

Explicitly document:

- World frame.
- Body frame.
- Inertial frame, if used.
- Sensor frame.
- Rotor frames.
- Quaternion convention.
- Quaternion component ordering.
- Rotation direction.
- Angular velocity frame.
- Force and torque frames.

Check the transformations between frames.

Pay particular attention to:

- World-to-body transformations.
- Body-to-world transformations.
- Quaternion multiplication conventions.
- Rotation matrices.
- Angular velocity transformations.
- Cross-product calculations.
- Force application points.
- Torque calculations.

Verify the conventions used by Isaac Sim, Isaac Lab and OmniDrones rather than assuming they are identical.

Document all physical units.

Use SI units consistently wherever possible.

Check for errors involving:

- Degrees versus radians.
- RPM versus rad/s.
- Grams versus kilograms.
- Millimetres versus metres.
- Inertia units.
- Force and torque units.
- Time units.
- Quaternion normalization.

Add assertions and tests wherever practical.

---

# 10. Sensor Data Generation

If IMU or camera measurements are required, use actual simulator sensor implementations wherever possible.

Do not simply derive sensor measurements from the state representation and label them as physical sensor measurements.

For IMU:

- Inspect the available Isaac Sim/Isaac Lab IMU sensor implementation.
- Configure the sensor appropriately.
- Collect actual accelerometer and gyroscope measurements.
- Account for sensor-frame transformations.
- Investigate gravity and specific-force conventions.
- Consider configurable noise, bias, drift and sampling frequency.
- Store raw measurements and relevant sensor metadata.

Distinguish between:

1. Ground-truth physical state.
2. Ground-truth acceleration and angular velocity.
3. Sensor measurements.
4. Derived or estimated measurements.

If an actual sensor is unavailable or unsuitable, explicitly document the alternative implementation and its limitations.

Apply the same principle to cameras and other sensors.

---

# 11. Dataset Requirements and Data Collection

Read:

`docs/requirement.md`

Treat it as the starting specification, not an unquestionable final specification.

Evaluate whether the existing requirements are sufficient for training world models and system identification.

Propose improvements where necessary.

At minimum, investigate collecting the following data.

### Ground-truth state

- `p_t`: Position.
- `v_t`: Linear velocity.
- `q_t`: Orientation quaternion.
- `w_t`: Angular velocity.

### Ground-truth dynamics

- Linear acceleration.
- Angular acceleration.
- Applied forces.
- Applied torques.
- Individual force contributions.
- Individual torque contributions.

### IMU

- Accelerometer measurements.
- Gyroscope measurements.

### Actuator data

- Individual rotor angular velocities.
- Individual motor commands.
- Motor states.
- Collective thrust.
- Total thrust.
- Total torque.
- Other relevant actuator information.

### Environmental information

- Wind velocity.
- Local wind direction.
- Wind magnitude.
- Wind acceleration or temporal variation where relevant.
- Turbulence information.
- Relevant environmental parameters.

### Experimental metadata

Every trajectory must contain sufficient metadata to identify and reproduce the experiment.

Include:

- Dataset ID.
- Dataset version.
- Experiment ID.
- Trajectory ID.
- Scenario ID.
- Domain ID.
- Physical configuration ID.
- Wind configuration ID.
- Force model configuration ID.
- Controller ID.
- Initial conditions.
- Terminal conditions.
- Sample rate.
- Simulation timestep.
- Simulation duration.
- Random seeds.
- Parameter values.
- Train/validation/test designation.
- ID/OOD designation.
- Open-loop/closed-loop designation.
- Simulator and software versions.

Review whether additional variables are required for learning a useful world model.

Pay attention to the distinction between input variables, observable state variables, hidden physical variables and training targets.

Do not silently change the meaning of an existing variable.

Define exact tensor shapes, data types, units, frame conventions and sampling semantics.

---

# 12. Dataset Storage and Reproducibility

Design a clean and efficient storage architecture.

Evaluate appropriate formats such as:

- HDF5.
- Zarr.
- Parquet, where suitable.
- NumPy-based formats for intermediate storage.

Choose based on:

- Sequential trajectory access.
- Random batch sampling.
- Parallel writing.
- Compression.
- Storage overhead.
- GPU training compatibility.
- Dataset scalability.

Avoid selecting a format without benchmarking its suitability.

### Reproducibility

The same experiment configuration and seed should reproduce the same experiment within documented simulator determinism limitations.

Store:

- Full resolved configuration.
- Configuration hash.
- Global seed.
- Per-experiment seed.
- Per-domain seed.
- Per-scenario seed.
- Per-trajectory seed.
- Simulator version.
- Physics engine settings.
- Relevant software versions.
- Source-code commit hash.
- Generation timestamp.

Use deterministic seed derivation rather than relying on uncontrolled global random-number generators.

Support regeneration of individual trajectories without regenerating the entire dataset.

### Experiment summaries

After generating each scenario, domain or dataset, create a summary file.

The summary should include:

- Number of successful trajectories.
- Number of failed trajectories.
- Simulation duration.
- Parameter distributions.
- Wind distributions.
- Scenario statistics.
- Initial-condition distributions.
- State and action ranges.
- Maximum and minimum accelerations.
- Maximum angular velocities.
- Actuator saturation statistics.
- Dataset split information.
- Validation results.
- Generation time.
- Computational resource utilization.

Make summaries machine-readable and human-readable where appropriate.

---

# 13. Dataset Quality Assurance and Leakage Detection

Dataset validation is mandatory.

Do not assume that a successfully generated trajectory is a valid training sample.

Implement automated dataset validation.

### A. Data integrity

Check:

- Missing values.
- NaN and Inf values.
- Incorrect tensor dimensions.
- Incorrect data types.
- Invalid quaternions.
- Invalid timestamps.
- Duplicate samples.
- Inconsistent sampling intervals.
- Incorrect labels.
- Invalid physical parameters.

### B. Physical consistency

Check:

- Position and velocity consistency.
- Velocity and acceleration consistency.
- Quaternion normalization.
- Angular velocity consistency.
- Force and torque consistency.
- Actuator constraints.
- Energy and momentum behavior where applicable.
- Physical feasibility of generated trajectories.

Use numerical tolerances appropriate to the simulator.

### C. Dataset leakage

Check for:

- Duplicate trajectories across splits.
- Overlapping trajectory segments.
- Shared experiment seeds.
- Shared physical configurations where prohibited.
- Shared scenario instances.
- Near-duplicate trajectories.
- Counterfactual pairs accidentally split across datasets.
- OOD parameter leakage.
- OOD wind-profile leakage.
- OOD scenario leakage.

Split at the correct experimental grouping level.

Do not randomly split individual time steps from the same trajectory into training and testing.

### D. Distribution validation

Generate statistical summaries and visualizations of:

- State distributions.
- Action distributions.
- Physical parameter distributions.
- Wind distributions.
- Trajectory distributions.
- Training versus validation versus test distributions.

Verify that intended ID and OOD distinctions actually exist.

Create an automated validation report and prevent invalid datasets from being marked as complete.

---

# 14. Counterfactual Data Generation

Develop a dedicated counterfactual generation module.

Generate paired trajectories where controlled experimental variables are changed while other conditions are held fixed.

Examples:

- Same initial conditions and control inputs, different wind directions.
- Same initial conditions and control inputs, different wind magnitudes.
- Same wind and control inputs, different masses.
- Same physical configuration, different inertia.
- Same experiment, different drag coefficients.
- Same conditions, different motor time constants.
- Same conditions, different aerodynamic models.

Ensure that paired experiments use identical random disturbances wherever appropriate.

Record the exact differences between counterfactual pairs.

Evaluate whether counterfactual experiments improve the identification of individual physical mechanisms.

Do not accidentally change multiple unrelated variables in an experiment intended to isolate one physical effect.

---

# 15. Computational Performance and Parallel Data Generation

Investigate how to maximize dataset-generation throughput.

Inspect the current implementation and determine whether it supports:

- Multiple parallel simulation environments.
- GPU-parallel rigid-body simulation.
- Multiple GPUs.
- CPU parallelism.
- Batched tensor operations.
- Asynchronous dataset writing.
- Efficient sensor collection.
- Vectorized force and torque calculations.

Investigate Isaac Lab's environment replication and parallel simulation capabilities.

Evaluate whether Isaac Sim Replicator is relevant to our requirements or whether Isaac Lab's vectorized simulation environments are more appropriate.

Do not assume Replicator is automatically the right solution for physics trajectory generation. Distinguish between visual/synthetic data generation and batched physics simulation.

Investigate:

- `GridCloner` and environment cloning.
- Isaac Lab vectorized environments.
- GPU physics execution.
- CPU worker processes.
- Multiple simulation instances.
- Multi-GPU distribution.
- Batched tensor operations.

Determine whether multiple independent simulation processes or one vectorized simulation instance is more efficient.

Benchmark:

1. Single environment.
2. Multiple environments on one GPU.
3. Multiple CPU workers, where applicable.
4. Multiple GPUs, if supported.
5. Different simulation and data-writing batch sizes.

Measure:

- Trajectories generated per second.
- Simulation steps per second.
- CPU utilization.
- GPU utilization.
- GPU memory usage.
- CPU memory usage.
- Data-writing throughput.
- Overall generation time.

Do not introduce complex parallelization before identifying the actual bottleneck.

Preserve deterministic experiment assignment and seed management when running in parallel.

---

# 16. Software Architecture and Folder Structure

Audit and improve the architecture under:

`src/physiworld_uav`

The codebase should be modular, extensible and maintainable.

Use appropriate abstractions and clear interfaces.

Potential responsibilities include:

- Drone models.
- Physical parameters.
- Force and torque models.
- Wind models.
- Domain randomization.
- Trajectory generation.
- Scenario management.
- Controller integration.
- Sensor management.
- Experiment orchestration.
- Data collection.
- Dataset storage.
- Dataset validation.
- Reproducibility.
- Reporting.

Avoid creating unnecessary classes or wrappers around existing Isaac APIs.

Use dependency injection where it meaningfully improves modularity and testing.

Use type hints, dataclasses or appropriate configuration models.

Keep simulation logic separate from dataset storage and reporting.

Keep physical models independently testable.

Ensure that adding a new drone, force model, wind profile or scenario does not require modifying unrelated modules.

Follow established Python project conventions.

Prefer established libraries over custom implementations when appropriate.

Examples include:

- NumPy.
- PyTorch.
- SciPy.
- Hydra/OmegaConf.
- Pydantic, if appropriate.
- HDF5/Zarr libraries.
- Matplotlib.
- Pytest.

Do not introduce a dependency merely because it is popular.

---

# 17. Testing Strategy

Implement a comprehensive Pytest-based testing framework.

Tests should cover individual components and the complete data-generation pipeline.

### Unit tests

Test:

- Each force model.
- Each torque model.
- Wind models.
- Coordinate transformations.
- Quaternion operations.
- Domain randomization.
- Parameter sampling.
- Trajectory generation.
- Scenario generation.
- Counterfactual generation.
- Data serialization.
- Metadata generation.
- Seed reproducibility.

### Physics validation tests

Test isolated physical scenarios against known analytical behavior.

Examples:

- Free fall.
- Hover equilibrium.
- Constant thrust.
- Translational drag.
- Rotational damping.
- Motor step response.
- Wind-induced acceleration.
- Force-to-torque transformations.

### Integration tests

Test:

- Complete simulation setup.
- Controller integration.
- Open-loop experiments.
- Closed-loop experiments.
- Sensor collection.
- Dataset generation.
- Dataset loading.
- Metadata integrity.
- Dataset validation.

### Regression tests

Ensure that changes to the architecture do not unintentionally alter previously validated physical behavior.

### Performance tests

Monitor generation throughput and memory usage.

Separate tests that require Isaac Sim from ordinary CPU-only unit tests.

Tests should be executable using documented commands.

---

# 18. Documentation

Update documentation to reflect the actual implementation.

At minimum, maintain:

1. Architecture overview.
2. Configuration guide.
3. Drone model documentation.
4. Force and torque model documentation.
5. Wind model documentation.
6. Domain randomization guide.
7. Scenario-generation guide.
8. Open-loop and closed-loop experiment guide.
9. Dataset schema.
10. Reproducibility guide.
11. Dataset validation guide.
12. Performance benchmarking guide.
13. Scientific assumptions and limitations.

Every physical model should have its equations, assumptions, units, applicable conditions and references documented.

Update `docs/requirement.md` only after reviewing and proposing improvements.

Maintain `docs/datasets/problems/identifiability.md` as an important research constraint.

---

# 19. Required Execution Phases

Follow the phases below.

## Phase 1: Repository audit

Inspect the entire relevant project.

Produce:

- Existing folder structure.
- Existing module responsibilities.
- Configuration architecture.
- Current data-generation pipeline.
- Existing physics implementations.
- Existing simulator functionality.
- Existing tests.
- Missing functionality.
- Redundant functionality.
- Scientific and engineering concerns.

Do not modify code yet.

## Phase 2: Scientific and architectural review

Evaluate:

- Physical model assumptions.
- Aerodynamic approximations.
- Wind models.
- Coordinate transformations.
- Units.
- Double-counting risks.
- Identifiability.
- Dataset requirements.
- OOD split strategy.
- Reproducibility.
- Computational efficiency.

For each concern, classify it as:

- Correct.
- Correct with limitations.
- Needs improvement.
- Incorrect.
- Requires clarification.

Provide mathematical explanations for important scientific issues.

## Phase 3: Proposed architecture and implementation plan

Present:

- Proposed folder structure.
- Configuration hierarchy.
- Module responsibilities.
- Data-generation workflow.
- Dataset schema.
- Domain and split strategy.
- Validation architecture.
- Parallel execution strategy.
- Testing strategy.

Provide a clear list of files to create, modify, retain or deprecate.

Identify ambiguities and ask me questions before making decisions that affect research validity.

## Phase 4: Incremental implementation

After approval, implement in small, independently testable stages.

Prioritize:

1. Configuration architecture.
2. Physical model correctness.
3. Wind and domain randomization.
4. Scenario and trajectory generation.
5. Open-loop and closed-loop experiments.
6. Sensor and state collection.
7. Dataset storage.
8. Reproducibility.
9. Dataset validation.
10. Parallel generation and optimization.
11. Documentation.

Do not implement everything in one large, unverified change.

## Phase 5: Testing and benchmarking

Run relevant unit, integration, physics-validation and regression tests.

Run a small end-to-end dataset-generation experiment.

Verify the actual stored data, labels and metadata.

Benchmark performance before and after optimization.

Clearly report any tests that could not be executed and why.

## Phase 6: Final project review

Provide:

- Final architecture.
- Updated folder structure.
- Configuration examples.
- Implemented features.
- Physics models and assumptions.
- Dataset-generation instructions.
- Dataset schema.
- Reproducibility instructions.
- Validation results.
- Test results.
- Performance benchmarks.
- Remaining limitations.
- Outstanding research questions.

---

# 20. Final Engineering Principles

The following principles are non-negotiable:

1. Physical correctness takes priority over generating a large volume of data.
2. Identifiability and experimental diversity take priority over indiscriminate randomization.
3. OOD separation must be scientifically meaningful and verifiable.
4. Reproducibility must be built into the architecture from the beginning.
5. Never double-count forces or dynamics already handled by Isaac Sim.
6. Never mix coordinate frames or physical units without explicit transformations.
7. Use actual simulator sensors when sensor measurements are required.
8. Reuse established simulator and library functionality.
9. Every physical model must be independently testable.
10. Every generated dataset must pass automated validation.
11. Computational optimization must be based on measured bottlenecks.
12. Prefer simple, validated physical approximations over unnecessarily complicated models.
13. Keep the architecture modular without overengineering.
14. Never hide uncertainty or make unsupported claims about physical validity.

**The ultimate goal is not to build a complicated simulator. It is to build a scientifically defensible and computationally efficient data-generation system that produces high-quality, diverse and reproducible datasets for training reliable UAV world models.**

When a scientific or architectural decision is ambiguous, stop, explain the alternatives and ask me rather than silently making an assumption.


