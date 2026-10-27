from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import gymnasium as gym
import omni_drones
import omni_drones.envs           

env_ids = sorted([k for k in gym.registry.keys()])
print(f"Total registered envs: {len(env_ids)}")
for e in env_ids:
    print(" ", e)

sim_app.close()