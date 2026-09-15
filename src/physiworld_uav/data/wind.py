import math
import random
import torch


def sample_constant_wind(batch_size, min_speed, max_speed, device=None):
    speed = torch.empty(batch_size, device=device).uniform_(min_speed, max_speed)
    direction = torch.rand(batch_size, device=device) * (2.0 * math.pi)
    elevation = torch.empty(batch_size, device=device).uniform_(-0.2, 0.2)

    wind = torch.zeros(batch_size, 3, device=device)
    wind[:, 0] = speed * torch.cos(direction) * torch.cos(elevation)
    wind[:, 1] = speed * torch.sin(direction) * torch.cos(elevation)
    wind[:, 2] = speed * torch.sin(elevation)
    return wind


def generate_wind_sequence(
    length,
    min_speed,
    max_speed,
    batch_size=1,
    dt=0.02,
    gust_probability=0.05,
    device=None,
):
    base = sample_constant_wind(batch_size, min_speed, max_speed, device)

    sequence = []
    current = base.clone()

    for _ in range(length):
        if random.random() < gust_probability:
            gust = torch.randn_like(current) * 0.5
            current = current + gust

        sequence.append(current.clone())
        current = 0.995 * current + 0.005 * base

    return torch.stack(sequence, dim=1)