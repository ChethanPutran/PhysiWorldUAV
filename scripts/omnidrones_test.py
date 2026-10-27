from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import torch
from omni_drones.envs import IsaacEnv

# Try to import a known environment
try:
    from omni_drones.envs import Track
    print(">>> Track env class found")
except ImportError as e:
    print(">>> Track env not available:", e)

# List what's registered
import gymnasium as gym
import omni_drones  # triggers registration
envs = [k for k in gym.registry.keys() if "Isaac" in k or "Drone" in k]
print(">>> Registered envs:", envs[:10])

sim_app.close()
print(">>> Test complete")

