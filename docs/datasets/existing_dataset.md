To collect data for training a UAV world model, you have two primary paths: **simulation-based** and **real-world** collection. The best approach is often a hybrid: use simulation for cheap, large-scale, perfectly labeled data, and real-world flights for the "sim-to-real" fine-tuning that captures true sensor noise and unmodeled dynamics. Here’s a practical guide to both.

### 🏗️ Path 1: Simulation-Based Data Collection

For a world model, simulation is where you can generate massive amounts of synchronized, perfectly labeled data (images, IMU, wind disturbances, and ground-truth states) under diverse and extreme conditions.

**The Recommended Toolchain**
While NVIDIA Isaac Sim is the foundation, it does not natively support UAV wind simulation. You will need to extend it with specialized frameworks and tools:

*   **Pegasus Simulator**: This is a crucial add-on for Isaac Sim that provides the necessary UAV dynamics and control interfaces. It integrates with ROS 2 and supports the automatic generation and publishing of sensor data, including IMU and cameras.
*   **Omniverse Replicator**: This is the engine for synthetic data generation (SDG). You can use it to programmatically randomize scene lighting, textures, object placement, and sensor parameters to create a robust dataset. It can output perfectly labeled data like bounding boxes, segmentation masks, and depth maps alongside your raw images.
*   **Wind Modeling with OpenFOAM**: To get realistic wind data, you can precompute complex airflow fields using a CFD tool like OpenFOAM. Projects like the **Osprey Simulator** (built on Isaac Sim) demonstrate how to integrate these precomputed wind fields into the simulation, allowing your UAV to interact with realistic airflow in real-time.

**What Data You Can Collect in Simulation**
Your pipeline can record synchronized streams at a high frequency (e.g., 1kHz) for:
*   **Images**: From simulated RGB, depth, or event cameras.
*   **IMU Data**: Linear acceleration and angular velocity.
*   **Wind Data**: Disturbance forces or a full 3D wind field vector.
*   **Ground Truth**: Perfect 6-DoF pose, velocity, and control commands.

**Relevant Simulated Datasets to Study**
*   **DroneNav-6D**: A simulated dataset with synchronized RGB observations, 6-DoF trajectories, control commands, and **randomized wind disturbances**.
*   **SkyJEPA**: 5,000 domain-randomized closed-loop quadrotor trajectories generated in simulation specifically for training a latent-dynamics world model.

### ✈️ Path 2: Real-World Data Collection

Simulation cannot capture everything. Real flights are essential for collecting the true sensor noise, vibrations, and aerodynamic effects that your world model must handle.

**Hardware Payload Setup**
You will need a multirotor platform capable of carrying the following sensor suite:

*   **Flight Controller & IMU**: A system like a **Pixhawk** or **CubePilot Cube Orange** provides high-frequency, calibrated IMU data (accelerometer, gyroscope) essential for state estimation.
*   **Camera**: A synchronized RGB camera for first-person-view (FPV) video. Mount it to minimize vibration.
*   **Wind Sensor**: To directly measure wind, you need an anemometer. A **3D ultrasonic anemometer** is the standard for research, as it can measure the full 3D wind vector.
    *   **Critical Mounting Note**: Mount the anemometer on a mast, approximately **85 cm above the UAV body**. This places it above the propeller wash (inflow layer) so it measures the ambient wind, not the downward thrust of the drone itself.
*   **Companion Computer**: An onboard computer (e.g., a Raspberry Pi or NVIDIA Jetson) to log and synchronize all the data streams at high frequency.

**Data Synchronization & Processing**
This is the most critical step. All data streams must be **synchronized and time-stamped** on a common clock.
*   **Sampling Rate**: Sample all sensors at a consistent, high frequency (e.g., **1 kHz**) to capture fast dynamics.
*   **Wind Transformation**: The raw wind measurement from the anemometer is in the UAV's body frame. To be useful, you must transform it into the Earth (static) frame using the UAV's attitude (yaw) and velocity. This is typically done with a Galilean transformation: `u = R(ψ) * u_measured + V_airspeed`.
*   **Filtering**: Apply low-pass or Kalman filters to the IMU and wind data to reduce noise while preserving the true signal.

### 📚 Existing Public Datasets for Pre-training & Benchmarking

Before you build your own collection pipeline, leverage these existing datasets, which are valuable for pre-training or benchmarking.

| Dataset | Key Features | Real/Synthetic | Relevant For |
| :--- | :--- | :--- | :--- |
| **WindSeer (Nature 2024)** | Flight logs with onboard wind sensing, IMU, state estimation, and reconstructed 3D wind fields. | Real | Onboard wind-field estimation and mapping. |
| **Kitepower Flight Data** | Position, orientation, velocities, accelerations, and airflow (pitot tube, wind vanes) from a kite-based UAV. | Real | Airflow measurement and aerodynamic interactions. |
| **s-CAFE Dataset** | 37+ minutes of flight data under horizontal wind speeds up to **8 m/s**, including IMU and aerodynamic force estimates. | Real | Aerodynamic force estimation under wind. |
| **MotionScape** | **30+ hours** of 4K UAV-view video (4.5M+ frames) with semantic annotations for world modeling benchmarks. | Real | Video-based world model training and evaluation. |
| **AirScape Dataset** | **11k video-intention pairs** with over 1,000 hours of motion intention annotations for 6-DoF aerial agents. | Real | Training controllable aerial world models. |
| **DroneWAM / DroneNav-6D** | Simulated navigation dataset with synchronized RGB, 6-DoF trajectories, control commands, and **randomized wind disturbances**. | Simulated | World-action model training for visual navigation. |

### 💎 Summary: A Hybrid Workflow

A robust strategy for training your UAV world model is to combine these paths:

1.  **Pre-train in Simulation**: Use the Isaac Sim + Pegasus + Replicator + OpenFOAM stack to generate a massive, diverse dataset. This teaches your model the fundamental physics of flight and camera motion without the cost and risk of real flights.
2.  **Fine-tune with Real Data**: Conduct real-world flights with your sensor payload. Use this data to fine-tune the model, allowing it to adapt to real-world sensor noise, vibrations, and the unmodeled aerodynamic effects that simulation misses.
3.  **Benchmark with Public Datasets**: Use datasets like MotionScape or WindSeer to evaluate your model's performance against established benchmarks and ensure it generalizes well.

If you have a specific type of UAV (e.g., fixed-wing vs. multirotor) or a particular world model architecture in mind, I can offer more tailored suggestions.


## Part 2: Generating the Data in Isaac Sim / Isaac Lab

You already have everything installed. Here is the concrete path.

### 2.1 Wind and Disturbance Application

Isaac Lab provides the **WrenchComposer** system for applying external forces. Each asset maintains two composers:

| Composer | Lifetime | Use Case |
|---|---|---|
| `instantaneous_wrench_composer` | One simulation step | Time-varying forces (wind, drag, turbulence) |
| `permanent_wrench_composer` | Multiple steps | Constant forces (steady wind) |

The API methods are:

```python
# Overwrite forces (use for one-shot wind kicks)
asset.instantaneous_wrench_composer.set_forces_and_torques(
    forces=wind_force_tensor,  # shape (num_envs, 1, 3)
    torques=wind_torque_tensor,
    is_global=True,  # world-frame wind
)

# Accumulate forces (use for layered disturbances)
asset.instantaneous_wrench_composer.add_forces_and_torques(
    forces=gust_force,
    is_global=True,
)
```

For **AM-Bench** (a benchmark built on Isaac Lab), wind is configured through two fields:

```python
# In the environment config:
cfg.enable_wind_effect = True
cfg.robot.multirotor.aerodynamics.wind_force_w = (x, y, z)  # Newtons, world frame
```

This is the simplest way to apply controlled, constant wind. For time-varying wind (gusts, Dryden turbulence), use the WrenchComposer API directly in your simulation loop.

### 2.2 Dryden Turbulence Model

Multiple papers reference the **Dryden model** for generating realistic turbulent wind. It produces Gaussian wind disturbances with controllable intensity and spatial scale. The implementation is straightforward:

```python
def dryden_wind(dt, sigma_u, sigma_v, sigma_w, L_u, L_v, L_w):
    """Generate Dryden turbulence increments.
    
    sigma_*: RMS turbulence intensity [m/s] per axis
    L_*:     length scale [m] per axis
    """
    # First-order filter (simplified Dryden)
    # Full model uses second-order filters for u and v
    # and first-order for w
    ...
```

You control the OOD regime through `sigma_*` and `L_*`:
- **Training**: moderate `sigma` (e.g., 0.5–1.5 m/s)
- **OOD magnitude**: high `sigma` (>2 m/s)
- **OOD gust**: sudden step changes in mean wind

### 2.3 OmniDrones Integration

You already cloned OmniDrones. It supports a **multimodal wind disturbance pool** where "wind disturbance is enabled in multimodal mode and replayed from a pre-generated disturbance set". The Iris quadrotor model is the default, with a trajectory-tracking task supporting lemniscate, polynomial, zigzag, and figure-eight trajectories.

OmniDrones uses **TensorDict** as its primary data structure for passing observations, actions, and rewards. This maps naturally to your trajectory format.

### 2.4 The Data Logging Pipeline

Your setup already has the container ready. The pipeline you need to write:

```python
# pseudocode — adapt to your OmniDrones environment
import numpy as np

def generate_trajectory(env, wind_config, physics_config, seed):
    env.reset(seed=seed)
    traj = {"t": [], "p": [], "v": [], "q": [], "omega": [],
            "a_imu": [], "omega_imu": [], "u": [],
            "w": [], "f_ext": []}
    
    for t in range(T):
        # Apply wind via WrenchComposer
        env.apply_wind(wind_config.at(t))
        
        # Step simulation
        obs, reward, done, info = env.step(action)
        
        # Log everything
        traj["t"].append(t * dt)
        traj["p"].append(env.drone.position)
        traj["v"].append(env.drone.velocity)
        traj["q"].append(env.drone.quaternion)
        traj["omega"].append(env.drone.angular_velocity)
        traj["a_imu"].append(env.sensors.imu.accel)
        traj["omega_imu"].append(env.sensors.imu.gyro)
        traj["u"].append(action)
        traj["w"].append(wind_config.at(t))
        traj["f_ext"].append(env.drone.wrench_composer.composed_force)
        
        if done: break
    
    return {k: np.array(v) for k, v in traj.items()}
```

Save each trajectory as `.npz`:

```python
np.savez_compressed(f"trajectory_{i:04d}.npz", **traj, 
                    metadata=json.dumps(config))
```

### 2.5 OOD Regime Generation

Your proposal defines five OOD regimes. Here is how to generate each in the simulator:

| OOD Regime | Training Configuration | OOD Configuration | Implementation |
|---|---|---|---|
| Magnitude extrapolation | Wind speed 0–4 m/s | Wind speed 5–8 m/s | Scale `wind_force_w` magnitude |
| Directional extrapolation | Horizontal wind, limited vertical | Excluded vertical/diagonal directions | Set `wind_force_w` to held-out direction |
| Temporal extrapolation | Smooth wind variation | Sudden step changes | Instantaneous wrench composer with step input |
| Gust extrapolation | Low `sigma` turbulence | High `sigma`, short `L` | Dryden model with OOD parameters |
| Physics mismatch | Nominal `m, J, C_D` | Perturbed 10–20% | Modify `physics.json` per trajectory |

### 2.6 Recommended Workflow

Based on your own proposal and the tools available:

1. **Start with state-only trajectories** (no RGB). Generate 100–200 short flights with state, control, wind, and physics metadata. Validate dynamics and OOD labels before adding expensive rendering.

2. **Use OmniDrones as the base.** It already has the Iris quadrotor, trajectory generators, and wind disturbance pool. You only need to add the logging layer.

3. **Log at native rates.** IMU at 200 Hz, state at 200 Hz, control at 100 Hz. Store timestamps explicitly.

4. **Separate OOD generation.** Use a distinct seed and label each OOD regime explicitly in `trajectory.json`.

5. **Do not feed wind to the model.** Wind is for evaluation and oracle only. Including it would leak the disturbance.

6. **Scale up after validation.** Once state-only trajectories look correct, add camera + IMU and scale to thousands.

---

## Part 3: Why Simulation Is the Only Viable Path

The fundamental issue with public datasets is **counterfactual controllability**. Your proposal requires:

- The same trajectory flown under different wind conditions
- Physics parameters perturbed in a controlled way
- Reproducible OOD regimes with known ground-truth disturbance forces

No real-world dataset can provide this. You cannot ask a real drone to fly the exact same trajectory twice, once with 2 m/s wind and once with 7 m/s wind, while logging the ground-truth disturbance force. In simulation, you can — and you can do it millions of times in parallel.

The VID dataset gives you a template for what the data should look like (synchronized imagery, IMU, rotor speed, control inputs, 6-axis force). Use it as a format reference, but generate your own data with the controlled OOD conditions your study requires.

---

## Practical Next Steps

1. **Verify Isaac Lab wind API works** in your container:
   ```python
   # Test that WrenchComposer is accessible
   from omni.isaac.lab.assets import Articulation
   # Apply a test force and verify the drone moves
   ```

2. **Write the logging script** based on the pseudocode above, integrated with OmniDrones' `Track` environment.

3. **Generate a small batch** (10 trajectories) and verify the `.npz` files contain all required fields with correct shapes.

4. **Generate the full training set** (200–500 trajectories) with moderate wind.

5. **Generate OOD test sets** with the five regimes defined above.

6. **Only then** add RGB rendering if you pursue the visual extension.

Your data requirements document is well-structured. The simulator can produce everything in it. The public datasets cannot.


Based on your data requirements document, I searched for public datasets and simulator generation methods. Here is what I found.

## Short Answer

**No public dataset has everything you need simultaneously.** Real-world datasets lack controllable OOD wind conditions and ground-truth disturbance forces. **You must generate the data yourself in Isaac Sim / Isaac Lab** — and the tooling to do so is already installed in your container.

---

## Part 1: Public Datasets — What Exists and What's Missing

Several public UAV datasets partially match your requirements, but each has critical gaps for your OOD wind disturbance study.

### VID (Visual-Inertial-Dynamical) Dataset — Closest Match

This is the most complete public dataset for your purposes. It contains hardware-synchronized imagery and inertial measurements with accurate ground-truth trajectories, plus **rotor speed, motor current, control inputs, and ground-truth 6-axis force data** for external force estimation. It is the first public dataset containing visual-inertial and complete dynamical information for pose and external force evaluation.

**Critical gap:** It does not include controlled wind disturbance experiments. The external forces are from payload/manipulation, not wind. You cannot use it to define an OOD wind taxonomy.

### IDF-DS Telemetry Dataset

A fixed-wing UAV telemetry dataset containing raw IMU at 200–400 Hz, GPS position at 5–10 Hz, and local position/velocity at 10–100 Hz. It includes wind field estimation use cases for mission planning.

**Critical gap:** Fixed-wing, not quadrotor. No control inputs logged in a form usable for your physics model. No controlled wind generation.

### WindSeer Dataset (ETH Zurich, Nature 2024)

Contains real-world flight logs with onboard wind-sensing instrumentation, IMU, state estimation, controller feedback, reconstructed 3D wind fields, and ground-truth wind measurements.

**Critical gap:** It focuses on wind *estimation*, not on providing synchronized state/control/wind tuples for world-model training. The wind fields are reconstructed, not controlled. You cannot define OOD regimes with it.

### Drone Flight Dataset with LSTM Wind Estimation (IEEE Access 2025)

17 synchronized features including IMU, GPS velocity, motor outputs, and orientation across varied wind conditions. 167,417 data points at 50 Hz.

**Critical gap:** Outdoor flights with natural wind, not controllable. No ground-truth disturbance force. Cannot define OOD regimes with reproducibility.

### Summary Table

| Dataset | IMU | GT State | Control | Wind GT | Disturbance Force | OOD Controllable |
|---|---|---|---|---|---|---|
| VID | ✅ | ✅ | ✅ | ❌ | ✅ (6-axis) | ❌ |
| IDF-DS | ✅ | ✅ (GPS) | ❌ | Partial | ❌ | ❌ |
| WindSeer | ✅ | ✅ | ✅ | ✅ (reconstructed) | ❌ | ❌ |
| LSTM Dataset | ✅ | ✅ (GPS) | ✅ | ❌ | ❌ | ❌ |
| **Your requirements** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**None of them satisfy the OOD requirement.** This is why your proposal explicitly states Isaac Sim is preferred over real flight data — OOD conditions are *counterfactually controllable* in simulation.

---


Part 3: Why Simulation Is the Only Viable Path
The fundamental issue with public datasets is counterfactual controllability. Your proposal requires:

The same trajectory flown under different wind conditions

Physics parameters perturbed in a controlled way

Reproducible OOD regimes with known ground-truth disturbance forces

No real-world dataset can provide this. You cannot ask a real drone to fly the exact same trajectory twice, once with 2 m/s wind and once with 7 m/s wind, while logging the ground-truth disturbance force. In simulation, you can — and you can do it millions of times in parallel.

The VID dataset gives you a template for what the data should look like (synchronized imagery, IMU, rotor speed, control inputs, 6-axis force). Use it as a format reference, but generate your own data with the controlled OOD conditions your study requires.