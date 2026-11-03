# Modular UAV force system

## Files
- `modular_forces.py`: PyTorch force/torque models and aggregation.
- `example_config.yaml`: starter configuration (YAML is illustrative; the Python dataclasses are the executable config API).

## Quick start
```python
import torch
from modular_forces import (
    RigidBodyState, ModularForceSystem, make_example_config
)

state = RigidBodyState(
    position_W=torch.tensor([0., 0., 1.5]),
    velocity_W=torch.tensor([0., 0., 0.]),
    quaternion_WB=torch.tensor([1., 0., 0., 0.]),
    omega_B=torch.tensor([0., 0., 0.]),
    rotor_omega=torch.tensor([500., 500., 500., 500.]),
)

def wind_fn(position_W, t):
    # Replace with evaluateWindVelocity / your Python wind field.
    return torch.zeros_like(position_W)

system = ModularForceSystem(make_example_config(), mass_kg=0.68)
out = system.evaluate(state, wind_fn, time_s=0.0)
print(out.forces_W)
print(out.torques_B)
print(out.total_force_W, out.total_torque_B)
```

## Coordinate and integration contract
- This module preserves the project audit's `+Y_B` rotor thrust axis.
- It assumes `q_WB=[w,x,y,z]` maps body vectors to world vectors.
- Gravity is a world-frame vector.
- Force models return force in world coordinates and torque in body coordinates.
- Integrate translation using `a_W = F_total_W / mass` (unless you formulate added mass as an effective-mass system).
- Integrate rotation using `J_B * omega_dot_B = tau_total_B - omega_B x (J_B omega_B)`.

## Surface mesh
Pass body-frame vertices `(V,3)` and triangle indices `(F,3)`. Normals must be outward.
The surface model calculates triangle-centre wind and body surface velocity, pressure and shear.
The pressure law is a simplified signed pressure-coefficient approximation, not CFD.
Do not combine it with full-body lumped drag unless you partition the aerodynamic areas.

## Added mass warning
The included added-mass module is a deliberately simplified experimental correction. It needs vehicle acceleration and local wind acceleration. It is disabled by default. A rigorous added-mass model requires consistent effective-mass equations and additional terms for nonuniform/unsteady flow; do not treat this implementation as validated physics.

## Important calibration items
1. Confirm rotor order, positions, spin directions and thrust axis from your USD/controller.
2. Calibrate `k_thrust` and `k_torque`; the example `k_torque` is a placeholder.
3. Calibrate `cd_area_B` from wind-tunnel/flight data or a documented aerodynamic estimate.
4. Confirm quaternion ordering and rotation direction against Isaac Sim.
5. Unit-test force signs: with zero wind and positive vehicle velocity, drag must oppose velocity.
6. With all rotor speeds zero, gravity should be the only active force if body aerodynamics are disabled.