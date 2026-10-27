from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

try:
    from omni.isaac.core import World

    world = World(stage_units_in_meters=1.0)
    world.reset()

    for _ in range(100):
        world.step(render=True)

    print("Isaac Sim GUI test completed.")

    # Keep the window alive until you close it.
    while simulation_app.is_running():
        simulation_app.update()

finally:
    simulation_app.close()
