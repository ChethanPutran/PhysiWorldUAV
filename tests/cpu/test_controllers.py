"""Control laws over OmniDrones controllers (plan T1–T3, Q2). CPU only."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import omni_drones_available

pytestmark = pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")


def _state(pos=(0, 0, 2.0), yaw=0.0, vel=(0, 0, 0), w_W=(0, 0, 0)):
    q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    return torch.tensor([[*pos, *q, *vel, *w_W]], dtype=torch.float32)


def test_lee_hover_command_equals_true_mass_trim(drone):
    from physiworld_uav.control.controllers import ControlLaw
    from physiworld_uav.trajectories.library import Anchor, Reference
    law = ControlLaw("position", drone)
    ref = Reference("hover", {}, Anchor((0, 0, 2.0), 0.0), 5.0).at(0.0)
    out = law.position(_state(), [ref])
    assert np.allclose(out.cmd_raw.numpy(), drone.hover_command, atol=1e-5)


def test_controller_uses_domain_mass(drone):
    """Q2: a 10 % heavier vehicle gets a 10 % larger hover thrust."""
    from physiworld_uav.control.controllers import ControlLaw
    from physiworld_uav.trajectories.library import Anchor, Reference
    heavy = drone.scaled(mass=drone.mass * 1.1)
    ref = Reference("hover", {}, Anchor((0, 0, 2.0), 0.0), 5.0).at(0.0)
    cmd = ControlLaw("position", heavy).position(_state(), [ref]).cmd_raw.numpy()
    T = (cmd + 1) / 2 * heavy.max_thrust_per_rotor
    assert T.sum() == pytest.approx(heavy.mass * 9.81, rel=1e-5)


@pytest.mark.parametrize("yaw_deg", [0.0, 45.0, 90.0, 180.0, -135.0])
def test_lee_lateral_response_is_yaw_independent(drone, yaw_deg):
    """T1: the old outer loop ignored yaw. Lee must push toward a +x world target at any yaw.
    The commanded body torque rotates the thrust toward +x_W: for small errors the
    roll/pitch torque vector, expressed in world frame, points along +y_W (tilt about y)."""
    from physiworld_uav.control.controllers import ControlLaw
    from physiworld_uav.excitation.open_loop import allocation_matrix
    from physiworld_uav.trajectories.library import Anchor, Reference
    yaw = math.radians(yaw_deg)
    law = ControlLaw("position", drone)
    ref = Reference("hover", {}, Anchor((0.5, 0, 2.0), yaw), 5.0).at(0.0)   # target 0.5 m along +x_W
    cmd = law.position(_state(yaw=yaw), [ref]).cmd_raw.numpy()[0]
    T = (cmd + 1) / 2 * drone.max_thrust_per_rotor
    tau_B = (allocation_matrix(drone) @ T)[:3]
    c, s = math.cos(yaw), math.sin(yaw)
    tau_W = np.array([c * tau_B[0] - s * tau_B[1], s * tau_B[0] + c * tau_B[1]])
    assert tau_W[1] > 0 and abs(tau_W[0]) < 0.05 * abs(tau_W[1])


def test_attitude_hold_is_yaw_aware(drone):
    """Hold point 0.3 m along +x_W with the vehicle yawed 90°: correction must be a roll
    (body −y = world +x), not a pitch."""
    from physiworld_uav.control.controllers import ControlLaw
    from physiworld_uav.trajectories.library import RefPoint
    law = ControlLaw("attitude", drone)
    out = law.attitude(_state(yaw=math.pi / 2), [RefPoint(roll=0.0, pitch=0.0, yaw=0.0, yaw_rate=0.0)],
                       torch.tensor([[0.3, 0.0, 2.0]]))
    assert out.extras["roll_cmd"].item() > 0.01
    assert abs(out.extras["pitch_cmd"].item()) < 1e-6
    assert out.collective_thrust.item() == pytest.approx(drone.mass * 9.81, rel=1e-2)


def test_rate_mode_passthrough(drone):
    from physiworld_uav.control.controllers import ControlLaw
    law = ControlLaw("rate", drone)
    out = law.rate(_state(), torch.tensor([drone.weight]), torch.zeros(1, 3))
    assert np.allclose(out.cmd_raw.numpy(), drone.hover_command, atol=1e-5)
    with pytest.raises(ValueError):
        ControlLaw("position", drone, gains={"bogus": 1})
