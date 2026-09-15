import torch
from .rk4 import rk4_step


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)

    return torch.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def hat_to_matrix(rotvec):
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / theta.clamp_min(1e-8)

    K = skew(axis)
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    I = I.expand(rotvec.shape[:-1] + (3, 3))

    sin_theta = torch.sin(theta)[..., None]
    cos_theta = torch.cos(theta)[..., None]

    return I + sin_theta * K + (1.0 - cos_theta) * (K @ K)


def rotation_matrix_to_rotvec(R):
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    vee = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )

    scale = theta / (2.0 * torch.sin(theta).clamp_min(1e-6))
    return scale.unsqueeze(-1) * vee


class QuadrotorPhysics:
    """
    Simplified rigid-body quadrotor model.

    State:
        [p(3), v(3), rotvec(3), omega(3)]

    Action:
        normalized rotor commands in [0, 1].

    The rotational state is represented internally as a rotation vector
    for a compact neural-network-friendly state representation.
    """

    def __init__(
        self,
        mass=1.0,
        gravity=9.81,
        arm_length=0.17,
        kf=1e-5,
        km=1e-6,
        inertia=(0.005, 0.005, 0.009),
        max_rotor_speed=900.0,
        dt=0.02,
    ):
        self.mass = mass
        self.gravity = gravity
        self.arm_length = arm_length
        self.kf = kf
        self.km = km
        self.max_rotor_speed = max_rotor_speed
        self.dt = dt

        self.J = torch.tensor(inertia, dtype=torch.float32)

    def _parameters(self, device, dtype):
        return self.J.to(device=device, dtype=dtype)

    def rotor_forces(self, action):
        omega = action.clamp(0.0, 1.0) * self.max_rotor_speed
        return self.kf * omega.square()

    def forces_and_torques(self, action):
        thrusts = self.rotor_forces(action)
        total_thrust = thrusts.sum(dim=-1)

        # X configuration approximation.
        tx = self.arm_length * (thrusts[:, 0] + thrusts[:, 1]
                                - thrusts[:, 2] - thrusts[:, 3])
        ty = self.arm_length * (-thrusts[:, 0] + thrusts[:, 1]
                                + thrusts[:, 2] - thrusts[:, 3])

        tz = self.km * (
            -thrusts[:, 0]
            + thrusts[:, 1]
            - thrusts[:, 2]
            + thrusts[:, 3]
        )

        torque = torch.stack([tx, ty, tz], dim=-1)
        return total_thrust, torque

    def derivative(self, state, action, wind=None):
        p = state[..., 0:3]
        v = state[..., 3:6]
        rotvec = state[..., 6:9]
        omega = state[..., 9:12]

        del p

        R = hat_to_matrix(rotvec)
        thrust, torque = self.forces_and_torques(action)

        body_force = torch.zeros_like(v)
        body_force[..., 2] = thrust

        world_force = torch.matmul(R, body_force.unsqueeze(-1)).squeeze(-1)

        gravity = torch.zeros_like(v)
        gravity[..., 2] = -self.mass * self.gravity

        if wind is None:
            wind_force = torch.zeros_like(v)
        else:
            # Simple aerodynamic drag approximation.
            relative_velocity = v - wind
            wind_force = -0.15 * relative_velocity

        acceleration = (world_force + gravity + wind_force) / self.mass

        J = self._parameters(state.device, state.dtype)
        angular_momentum = omega * J
        coriolis = torch.cross(omega, angular_momentum, dim=-1)

        angular_acceleration = (torque - coriolis) / J

        # First-order approximation for rotation-vector derivative.
        rotvec_dot = omega

        return torch.cat(
            [
                v,
                acceleration,
                rotvec_dot,
                angular_acceleration,
            ],
            dim=-1,
        )

    def step(self, state, action, wind=None):
        return rk4_step(
            lambda s: self.derivative(s, action, wind),
            state,
            self.dt,
        )

    def project_state(self, state):
        # Keep rotation vector numerically bounded.
        out = state.clone()
        out[..., 6:9] = torch.clamp(out[..., 6:9], -3.0, 3.0)
        return out