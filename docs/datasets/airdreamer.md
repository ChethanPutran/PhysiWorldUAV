To generate data similar to the AirDreamer paper, you need to replicate its specific simulation-based data collection pipeline. The paper's approach is highly focused and uses a particular set of tools and randomization techniques to train a world model for drone navigation.

Here is a breakdown of the exact methodology from the paper that you can use as a blueprint.

### 🏗️ Core Simulation Platform
AirDreamer's data is generated entirely in a simulation environment built on **OmniDrones**, which is a drone simulator based on **NVIDIA Isaac Sim 4.1.0**.

*   **Drone Model**: The simulation uses the **Hummingbird** drone model.
*   **Low-Level Controller**: A **LeePositionController (Lee)** is used to control the drone's low-level dynamics.
*   **Simulation Frequencies**: The physical simulation runs at **200 Hz**, while the policy (and thus data collection) is executed at **20 Hz**.

### 📊 Data Collected (Observations)
The world model in AirDreamer is trained on a specific set of observations:

*   **Primary Input**: A **normalized depth image** with a resolution of **48×80 pixels** and a range of **[0.6, 6.0] meters**. Pixels outside this range are replaced with the maximum value (6 m).
*   **State Vector**: A low-dimensional state vector is also provided, which includes:
    *   Unit goal direction in the body frame
    *   Horizontal goal distance and vertical offset
    *   Body-frame linear velocity
    *   Body-frame angular velocity
    *   Orientation (quaternion).

### 🎲 Randomization for Generalization
To ensure the trained policy can transfer to the real world (sim-to-real), AirDreamer applies **domain randomization** to the drone's dynamics parameters at the start of each training episode. This is a critical step for generating diverse data.

The specific parameters randomized and their ranges are detailed in the paper's Table I:

| Parameter | Min | Max |
| :--- | :--- | :--- |
| **Mass scale** | 0.90 | 1.10 |
| **Inertia scale** | 0.85 | 1.15 |
| **Thrust-to-weight scale** | 0.85 | 1.15 |
| **Force-to-moment scale** | 0.85 | 1.15 |
| **Drag coefficient scale** | 0.75 | 1.25 |
| **Motor spin-up gain** | 0.37 | 0.49 |
| **Motor spin-down gain** | 0.37 | 0.49 |

> **Note**: For parameters with multiple components (like the inertia tensor), each component is sampled independently.

Additionally, the paper randomizes the **spawn and goal locations, heights, obstacle dimensions, and initial yaw directions** for each episode.

### 🛠️ Training & Data Volume
The scale of data collection in AirDreamer is substantial:

*   **Training Steps**: The model is trained for **2.5 million environment steps**, which is equivalent to approximately **34.7 hours** of experience.
*   **Replay Buffer**: The replay buffer is limited to **1 million steps**.
*   **Hardware**: Training was performed on a server with **128GB RAM** and **two NVIDIA RTX 4090 D (24GB)** GPUs.

### 💻 How to Replicate This Setup
To generate similar data, you would follow these steps:

1.  **Set up Isaac Sim & OmniDrones**: Install NVIDIA Isaac Sim and the OmniDrones framework.
2.  **Configure the Environment**: Recreate the training environment script, which the paper notes takes inspiration from **NavRL**. This involves setting up the drone model (Hummingbird), controller (Lee), and the observation space (depth image + state vector).
3.  **Implement Domain Randomization**: Code the randomization logic for the parameters listed in Table I above, as well as for spawn/goal locations and obstacle dimensions.
4.  **Run Data Collection**: Execute the simulation for the required number of steps (e.g., 2.5 million) at the specified frequencies (200 Hz physics, 20 Hz policy), logging the depth images and state vectors at each policy step.

The paper states that the **code will be publicly available**, which will be your most direct resource for replicating their exact data generation pipeline. You can monitor the authors' GitHub or the paper's official repository for its release.