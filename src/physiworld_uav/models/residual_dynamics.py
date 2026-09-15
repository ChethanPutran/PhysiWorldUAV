import torch
import torch.nn as nn


class ResidualDynamics(nn.Module):
    """
    Learns the correction to nominal physics.

    Output:
        residual derivative in R^12.
    """

    def __init__(self, latent_dim=128, state_dim=12, action_dim=4, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, h, state, action):
        x = torch.cat([h, state, action], dim=-1)
        return self.net(x)