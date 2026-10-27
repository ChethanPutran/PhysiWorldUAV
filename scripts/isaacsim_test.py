from isaacsim import SimulationApp

# Start Isaac Sim in headless mode
sim_app = SimulationApp({"headless": True})

# If we get here, Kit loaded successfully
print(">>> Isaac Sim started successfully")

# Import OmniDrones AFTER SimulationApp is created
import omni_drones
print(">>> OmniDrones imported inside Kit")

sim_app.close()
print(">>> Clean shutdown")