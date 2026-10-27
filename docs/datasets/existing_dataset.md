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