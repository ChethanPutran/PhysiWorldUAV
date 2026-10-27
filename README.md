# UAV Physics-Prior Latent World Model

Research prototype implementing:

    ds/dt = f_phys(s,u) + d_theta(h,s,u)

The code currently uses a synthetic differentiable quadrotor simulator
so that the complete pipeline can be tested before connecting Isaac Sim.

## State

    [position(3), velocity(3), rotation_vector(3), angular_velocity(3)]

## Action

    [rotor_1, rotor_2, rotor_3, rotor_4]

## Main baseline

Physics-only:
    ds/dt = f_phys(s,u)

Pure neural:
    s_{t+1} = s_t + NN(s_t,u_t)

Physics-prior:
    ds/dt = f_phys(s,u) + d_theta(h,s,u)

## Run

    pip install -r requirements.txt
    python main.py --device cuda

For CPU:

    python main.py --device cpu

This is an MVP research implementation. The synthetic dynamics should
be replaced by Isaac Sim trajectories for the thesis experiments.


### Installing required packages

```bash
Isaac Sim 4.1.0
pip install isaacsim==4.1.0.0 --extra-index-url https://pypi.nvidia.com
Yes. Your `uv` environment is now correct:

```text
Python 3.10.21
```

For **Isaac Sim 4.1.0**, NVIDIA's archived 4.1 documentation explicitly supports installation through the NVIDIA PyPI index with:

```bash
pip install isaacsim==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```


### 1. First, check GLIBC

Isaac Sim 4.1's pip installation requires GLIBC 2.34+ on Linux. [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/install_python.html?utm_source=chatgpt.com)

Run:

```bash
ldd --version
```

You should see something like:

```text
ldd (Ubuntu GLIBC 2.35-...)
```

### 2. Make sure the `uv` environment has pip

Because `uv venv` doesn't necessarily provide `pip`, NVIDIA/Isaac Lab documentation recommends creating the environment with `--seed` when using uv. [Isaac Sim](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html?utm_source=chatgpt.com)

You already created the environment without `--seed`, so let's simply check:

```bash
python -m pip --version
```

If that says `No module named pip`, run:

```bash
uv pip install pip
```

Then verify:

```bash
python -m pip --version
```

### 3. Install Isaac Sim 4.1.0

Once GLIBC is confirmed, run:

```bash
uv pip install "isaacsim==4.1.0.0" \
    --extra-index-url https://pypi.nvidia.com

    If you get error 
    export UV_HTTP_TIMEOUT=300
    export UV_HTTP_RETRIES=5
```

This is the official Isaac Sim 4.1.0 package/version combination. [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/install_python.html?utm_source=chatgpt.com)

**Do not install PyTorch or OmniDrones yet.**

The installation is fairly large, so let this finish completely.

### 4. Verify installation

After it finishes:

```bash
python -c "import isaacsim; print('Isaac Sim imported successfully')"
```

Then:

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('isaacsim'))"
```

Expected:

```text
4.1.0.0
```

### 5. Check where it was installed

```bash
python -m pip show isaacsim
```

You'll get something similar to:

```text
Name: isaacsim
Version: 4.1.0.0
Location: .../.venv/lib/python3.10/site-packages
```

This is different from the old binary installation model. **We don't need** to manually create `~/.local/share/ov/pkg/isaac-sim-4.1.0` if we're using the pip installation. NVIDIA documents pip installation as an alternative to downloading the Isaac Sim binaries. [Isaac Sim](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html?utm_source=chatgpt.com)

---

### Do this first

Run:

```bash
ldd --version
```

and:

```bash
python -m pip --version
```

**Send me those two outputs before installing Isaac Sim.** I want to make sure your remote IISc machine satisfies the GLIBC requirement before we start downloading the large Isaac Sim package.


 python -m pip show isaacsim
Name: isaacsim
Version: 4.1.0.0
Summary: A metapackage that installs the complete version of Isaac Sim
Home-page: https://developer.nvidia.com/isaac-sim
Author: Isaac Sim team
Author-email: 
License: NVIDIA Proprietary Software
Location: /home/chetan/mtech/PhysiWorldUAV/.venv/lib/python3.10/site-packages
Requires: isaacsim-app, isaacsim-asset, isaacsim-benchmark, isaacsim-code-editor, isaacsim-core, isaacsim-cortex, isaacsim-example, isaacsim-gui, isaacsim-replicator, isaacsim-rl, isaacsim-robot, isaacsim-robot-motion, isaacsim-robot-setup, isaacsim-ros1, isaacsim-ros2, isaacsim-sensor, isaacsim-storage, isaacsim-template, isaacsim-test, isaacsim-utils
Required-by: 



What one trajectory should contain
Camera images (RGB)
What the drone sees
Save timestamped frames from an onboard camera, for example at 30 FPS. Record camera intrinsics and the camera-to-body transform.



IMU measurements
What the drone senses
Record 3-axis accelerometer and gyroscope readings, for example at 200 Hz. Include units, frame convention, noise, and bias.


Ground-truth state
What actually happens in the simulator
Save position, orientation, linear velocity, angular velocity, linear acceleration, and angular acceleration.


Controls and physics
What drives the motion
Record motor commands or thrusts, mass, inertia tensor, drag coefficients, and other relevant model parameters.


How to generate different trajectories
Don't manually fly the drone one trajectory at a time. Build an automated experiment runner that varies initial conditions, flight commands, wind, and physical parameters.
1. Sample an experiment configuration
Initial pose, target path, wind, mass, inertia, drag, sensor seed


2. Simulate the quadrotor
Controller → motor dynamics → UAV physics → camera and IMU


3. Record synchronized data
RGB frames, IMU, states, controls, wind, physical parameters


4. Validate and repeat
Check timestamps, units, trajectory completion and OOD labels


Vary these parameters across runs
Parameter	Example variations
Initial state	Different positions, orientations and velocities
Trajectory	Hover, straight line, circle, figure-eight, waypoint tracking, aggressive turns
Wind	Calm, constant wind, changing direction, gusts and Dryden turbulence
Wind direction	Forward, sideways, vertical and diagonal
Mass	Nominal and perturbed mass
Inertia \(J\)	Nominal inertia and perturbed diagonal entries
Drag	Nominal and increased/decreased drag coefficients
Sensor noise	Different noise and bias seeds
Environment	Different scene layouts and lighting, if visual generalization is being tested
These are experiment-design examples, not mandatory ranges. Choose values consistent with the simulated UAV's dynamics and stability.

You have successfully started the Isaac Sim container in interactive mode on your server. Now you need to access its GUI from your local laptop. The most reliable method, as we discussed, is **noVNC over SSH**, which works entirely over TCP and bypasses the UDP port issues you encountered earlier.

Here is the complete step-by-step procedure.

---

## Overview

You will:
1. Install noVNC components inside the running container.
2. Start a virtual display (Xvfb) and launch Isaac Sim on it.
3. Start x11vnc and websockify to serve the GUI over HTTP.
4. Create an SSH tunnel from your laptop to the server.
5. Open the GUI in your laptop's browser.

---

## Step 1: Install noVNC Inside the Container

You are already inside the container's bash shell (prompt looks like `root@catanzaro:/isaac-sim#`). Run:

```bash
apt-get update
apt-get install -y xvfb x11vnc novnc websockify
```

---

## Step 2: Start the Virtual Display and Isaac Sim

Still inside the container:

```bash
# Start a virtual display on :99
Xvfb :99 -screen 0 1920x1080x24 &

# Point to the virtual display
export DISPLAY=:99

# Launch Isaac Sim (use the headless script or the full GUI script)
./runheadless.native.sh --no-window
# OR for the full GUI experience:
# ./isaac-sim.sh
```

Wait until you see `Isaac Sim Headless WebRTC App is loaded.` or similar in the terminal. The GUI is now rendering to the virtual display.

---

## Step 3: Start x11vnc and websockify

Open a **second terminal inside the container**. From your host server (not inside the container), run:

```bash
docker exec -it isaac-sim bash
```

In this second container terminal, run:

```bash
# Start the VNC server attached to the virtual display
x11vnc -display :99 -forever -nopw -shared -rfbport 5900 &

# Start the noVNC web server on port 6080
websockify --web /usr/share/novnc/ 6080 localhost:5900 &
```

---

## Step 4: Create an SSH Tunnel from Your Laptop

On your **local laptop** (not the server), open a terminal (Command Prompt, PowerShell, or Terminal on macOS/Linux) and run:

```bash
ssh -L 6080:localhost:6080 chetan@10.24.36.121
```

Replace `chetan` with your username and `10.24.36.121` with your server's IP address. Leave this SSH session open — it is your secure tunnel.

---

## Step 5: Access the GUI in Your Browser

On your laptop, open a web browser (Chrome or Firefox recommended) and navigate to:

```
http://localhost:6080/vnc.html
```

You should see the noVNC interface. Click **Connect**, and the full Isaac Sim GUI will appear in your browser window. You can now interact with the simulation using your mouse and keyboard.

---

## Summary of Terminals and Commands

| Location | Terminal | Command / Action |
|---|---|---|
| Container (T1) | Inside `docker run` shell | `apt-get update && apt-get install -y xvfb x11vnc novnc websockify` |
| Container (T1) | Same shell | `Xvfb :99 -screen 0 1920x1080x24 &` |
| Container (T1) | Same shell | `export DISPLAY=:99` |
| Container (T1) | Same shell | `./runheadless.native.sh --no-window` (wait for load) |
| Host server | New terminal | `docker exec -it isaac-sim bash` |
| Container (T2) | Inside `docker exec` shell | `x11vnc -display :99 -forever -nopw -shared -rfbport 5900 &` |
| Container (T2) | Same shell | `websockify --web /usr/share/novnc/ 6080 localhost:5900 &` |
| Local laptop | Terminal | `ssh -L 6080:localhost:6080 chetan@10.24.36.121` |
| Local laptop | Browser | `http://localhost:6080/vnc.html` → Connect |

---

## Important Notes

- **No firewall changes needed:** The SSH tunnel handles the connection securely, so you do not need to open port 6080 on the server's firewall.
- **First launch may take time:** Isaac Sim builds its shader cache on the first run; this can take several minutes.
- **Performance:** noVNC uses TCP, so there may be slight latency, but it is more than sufficient for monitoring dataset generation.
- **Security:** The SSH tunnel encrypts the connection. Do not expose port 6080 directly to the internet.

Once connected, you will see the Isaac Sim GUI in your browser and can begin setting up your UAV trajectory dataset generation pipeline.


You're right to switch to `uv`. It's faster and more modern than conda. The error you hit is simply because `git` isn't installed in the Isaac Sim container by default. We'll fix that first, then walk through the complete `uv`-based installation.

### ⚙️ Step 1: Install `git` Inside the Container

Run this inside your running container to install `git` and other essential build tools:

```bash
apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    wget \
    curl
```


### 📦 Step 2: Install `uv` and Create a Python 3.10 Environment

Isaac Sim 4.x is built against **Python 3.10**, so the environment must match. We'll install `uv` and create a virtual environment with the correct version.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload your shell to make uv available
source ~/.bashrc  # or exec $SHELL

# Create a Python 3.10 environment (the --seed flag ensures pip is available)
uv venv --python 3.10 --seed /workspace/PhysiWorldUAV/omnidrones_env

# Activate it
source /workspace/PhysiWorldUAV/omnidrones_env/bin/activate
```


### 📂 Step 3: Clone the Required Repositories

Now that `git` is installed, clone OmniDrones, Isaac Lab, and the dependencies into your workspace.

```bash
cd /workspace/PhysiWorldUAV

# Clone OmniDrones
git clone https://github.com/btx0424/OmniDrones.git

# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git

# Clone TensorDict and TorchRL (TorchRL dependencies)
git clone https://github.com/btx0424/tensordict.git
git clone https://github.com/pytorch/rl.git
```

### 🛠️ Step 4: Set Up Isaac Sim Environment Variables

Tell the environment where Isaac Sim lives inside the container.

```bash
export ISAACSIM_PATH="/isaac-sim"
echo 'export ISAACSIM_PATH="/isaac-sim"' >> ~/.bashrc
source ~/.bashrc
```

### 🔗 Step 5: Copy the Environment Activation Scripts for OmniDrones

OmniDrones provides scripts that set up `PYTHONPATH` automatically when a conda environment is activated. Since we're using `uv`, we need to adapt this. The simplest approach is to create a small activation script in your environment.

```bash
# Create a custom activation script inside your uv environment
cat > /workspace/PhysiWorldUAV/omnidrones_env/bin/activate_omnidrones << 'EOF'
# Add Isaac Sim and Isaac Lab to PYTHONPATH
export ISAACSIM_PATH="/isaac-sim"
export PYTHONPATH="$ISAACSIM_PATH:$PYTHONPATH"
export PYTHONPATH="/workspace/PhysiWorldUAV/IsaacLab/source:$PYTHONPATH"
export PYTHONPATH="/workspace/PhysiWorldUAV/OmniDrones:$PYTHONPATH"
EOF

chmod +x /workspace/PhysiWorldUAV/omnidrones_env/bin/activate_omnidrones
```

Then, whenever you activate the environment, also source this script:

```bash
source /workspace/PhysiWorldUAV/omnidrones_env/bin/activate
source /workspace/PhysiWorldUAV/omnidrones_env/bin/activate_omnidrones
```

### 🧪 Step 6: Install Isaac Lab

Install Isaac Lab's dependencies and the framework itself using `uv pip`.

```bash
cd /workspace/PhysiWorldUAV/IsaacLab

# Install Isaac Lab's build dependencies
uv pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash

# Run the Isaac Lab installer
./isaaclab.sh --install
```


### 🔄 Step 7: Install TorchRL and TensorDict

Install the specific versions OmniDrones requires from source.

```bash
cd /workspace/PhysiWorldUAV/tensordict
uv pip install -e .

cd /workspace/PhysiWorldUAV/rl
uv pip install -e .
```

### 🚁 Step 8: Install OmniDrones

Finally, install OmniDrones itself in editable mode. This will pull in any remaining dependencies.

```bash
cd /workspace/PhysiWorldUAV/OmniDrones
uv pip install -e .
```


### ✅ Step 9: Verify the Installation

Test that everything is wired up correctly.

```bash
# Activate the environment and the custom script
source /workspace/PhysiWorldUAV/omnidrones_env/bin/activate
source /workspace/PhysiWorldUAV/omnidrones_env/bin/activate_omnidrones

# Check key imports
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')"
python -c "import omni.isaac.orbit; print('Isaac Lab OK')"
python -c "import omni_drones; print('OmniDrones OK')"
```

If all three print successfully, the installation is complete.

### 🔒 Step 10: Persist Your Setup

Because your container was launched with `--rm`, everything you've installed will be lost when the container stops. To keep your work:

**Option A (Recommended): Rebuild a custom image**
Create a `Dockerfile` that includes all these steps so the setup is permanent. This is the most reproducible approach. You can use your existing `Dockerfile` as a base and add these `RUN` commands.

**Option B: Remove `--rm` from your `docker run` command**
This will make the container persist between sessions. You can then restart it with `docker start -ai isaac-sim` and your `uv` environment will still be there. However, if you ever need to recreate the container, you'd have to redo the installation.


The error is expected — you have the latest Isaac Lab (which requires Python 3.12), but OmniDrones is built for **Isaac Sim 4.1.0 + Python 3.10**. We need to use the **compatible version** of Isaac Lab: **v1.0.0**, which still uses the `isaaclab.sh` installer you were originally looking for.

### 🛠️ Corrected Installation Steps

**Step 1: Remove the incompatible Isaac Lab clone**

```bash
cd /workspace/PhysiWorldUAV
rm -rf IsaacLab
```

**Step 2: Clone Isaac Lab and check out the compatible version (v1.0.0)**

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v1.0.0
```

**Step 3: Install Isaac Lab using its intended installer**

Now that you are on the correct version, the `isaaclab.sh` script will be present.

```bash
# Ensure you're in the uv environment
source /workspace/PhysiWorldUAV/omnidrones_env/bin/activate

# Install build dependencies (still needed)
uv pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash

# Run the Isaac Lab installer
./isaaclab.sh --install
```

**Step 4: Verify Isaac Lab**

```bash
python -c "import omni.isaac.lab; print('Isaac Lab OK')"
```

**Step 5: Install OmniDrones**

```bash
cd /workspace/PhysiWorldUAV/OmniDrones
uv pip install -e .
```

**Step 6: Final Verification**

```bash
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')"
python -c "import omni.isaac.lab; print('Isaac Lab OK')"
python -c "import omni_drones; print('OmniDrones OK')"
```

### ⚠️ Important Note About Persistence

Because you are installing inside a running container, this setup will be lost if the container is removed. Since you are using a `Dockerfile` and `entrypoint.sh` approach, the most reliable long-term solution is to **bake these installation steps into your Dockerfile** as `RUN` commands. This way, every new container you launch from your custom image will already have OmniDrones ready to go, and you won't have to repeat this process.

Try the `git checkout v1.0.0` step first, and let me know if the installer runs successfully.