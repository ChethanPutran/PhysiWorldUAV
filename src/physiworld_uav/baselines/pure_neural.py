
import torch
import torch.nn as nn


class PureNeuralWorldModel(nn.Module):
    """
    Baseline B2.

    Directly predicts the next physical state without explicitly using
    the known quadrotor equations.
    """

    def __init__(self, state_dim=12, action_dim=4, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        delta = self.net(
            torch.cat([state, action], dim=-1)
        )

        return state + delta

    @torch.no_grad()
    def imagine(self, initial_state, actions):
        state = initial_state
        predictions = [state]

        for t in range(actions.shape[1]):
            state = self.forward(
                state,
                actions[:, t],
            )
            predictions.append(state)

        return torch.stack(predictions, dim=1)