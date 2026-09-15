import torch
import torch.nn as nn

from .encoder import StateEncoder
from .residual_dynamics import ResidualDynamics
from .latent_dynamics import LatentDynamics


class PhysicsPriorWorldModel(nn.Module):
    """
    Physics-prior latent world model.

    At every step:
        h_{t+1} = latent dynamics(h_t, u_t)
        ds/dt   = f_phys(s_t, u_t) + residual(h_t, s_t, u_t)

    During imagination, future observations are not required.
    """

    def __init__(
        self,
        physics,
        state_dim=12,
        action_dim=4,
        latent_dim=128,
        hidden_dim=256,
    ):
        super().__init__()

        self.physics = physics

        self.encoder = StateEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
        )

        self.latent_dynamics = LatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.residual = ResidualDynamics(
            latent_dim=latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def initialize_latent(self, state, action):
        return self.encoder(state, action)

    def residual_derivative(self, h, state, action):
        return self.residual(h, state, action)

    def step(self, h, state, action, wind=None):
        h_next = self.latent_dynamics(h, action)

        nominal_derivative = self.physics.derivative(
            state,
            action,
            wind,
        )

        residual_derivative = self.residual(
            h,
            state,
            action,
        )

        state_next = state + self.physics.dt * (
            nominal_derivative + residual_derivative
        )

        return h_next, state_next

    @torch.no_grad()
    def imagine(self, initial_state, actions, wind=None):
        """
        Args:
            initial_state: [B, 12]
            actions: [B, H, 4]
            wind: [B, H, 3] or None

        Returns:
            predicted states: [B, H+1, 12]
        """
        B, H, _ = actions.shape

        dummy_action = actions[:, 0]
        h = self.initialize_latent(initial_state, dummy_action)
        state = initial_state

        predictions = [state]

        for t in range(H):
            current_wind = None if wind is None else wind[:, t]

            h, state = self.step(
                h,
                state,
                actions[:, t],
                current_wind,
            )

            state = self.physics.project_state(state)
            predictions.append(state)

        return torch.stack(predictions, dim=1)

    def forward(self, states, actions, wind=None, horizon=None):
        B, T, _ = states.shape

        if horizon is None:
            horizon = T - 1

        horizon = min(horizon, T - 1)

        h = self.initialize_latent(
            states[:, 0],
            actions[:, 0],
        )

        state = states[:, 0]
        predictions = [state]
        residuals = []

        for t in range(horizon):
            current_wind = None if wind is None else wind[:, t]

            h, state = self.step(
                h,
                state,
                actions[:, t],
                current_wind,
            )

            state = self.physics.project_state(state)
            predictions.append(state)
            residuals.append(
                self.residual(h, state, actions[:, t])
            )

        return (
            torch.stack(predictions, dim=1),
            torch.stack(residuals, dim=1),
        )