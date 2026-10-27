
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