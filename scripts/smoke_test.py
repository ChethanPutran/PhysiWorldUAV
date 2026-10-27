
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

try:
    from omni.isaac.core import World

    world = World(stage_units_in_meters=1.0)
    world.reset()

    for _ in range(10):
        world.step(render=False)

    print("Isaac Sim launch and stepping succeeded.")

finally:
    simulation_app.close()
