import torch.nn as nn


class PhysicsPrior(nn.Module):
    """
    Wrapper around differentiable nominal quadrotor physics.

    The physical simulator itself is not trainable. The residual model
    learns the discrepancy between nominal physics and the underlying
    dynamics.
    """

    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, state, action, wind=None):
        return self.physics.step(state, action, wind)