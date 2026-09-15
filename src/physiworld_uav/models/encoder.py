import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, state_dim=12, action_dim=4, latent_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, state, action):
        return self.net(__import__("torch").cat([state, action], dim=-1))