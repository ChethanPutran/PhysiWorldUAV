import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=128, action_dim=4, hidden_dim=256):
        super().__init__()

        self.gru = nn.GRUCell(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h, action):
        x = torch.cat([h, action], dim=-1)

        # GRU hidden dimension is allowed to differ from latent dimension.
        if h.shape[-1] != self.gru.hidden_size:
            h_in = torch.zeros(
                *h.shape[:-1],
                self.gru.hidden_size,
                device=h.device,
                dtype=h.dtype,
            )
        else:
            h_in = h

        hidden = self.gru(x, h_in)
        return self.to_latent(hidden)