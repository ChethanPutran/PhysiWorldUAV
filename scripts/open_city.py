from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd

stage_path = "/assets/AECO_CityMassingDemoPack/Demos/AEC/TowerDemo/CityMassingDemopack/World_CityMassingDemopack.usd"

omni.usd.get_context().open_stage(stage_path)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
