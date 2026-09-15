import torch
from torch.utils.data import Dataset
from .wind import generate_wind_sequence
from physiworld_uav.physics.quadrotor import QuadrotorPhysics


class SyntheticUAVDataset(Dataset):
    """
    Generates trajectories online.

    Each item contains:
        states: [T, 12]
        actions: [T-1, 4]
        wind: [T-1, 3]
    """

    def __init__(
        self,
        num_trajectories,
        sequence_length,
        dt,
        physics,
        wind_min,
        wind_max,
        seed=42,
    ):
        self.num_trajectories = num_trajectories
        self.sequence_length = sequence_length
        self.dt = dt
        self.physics = physics
        self.wind_min = wind_min
        self.wind_max = wind_max
        self.seed = seed

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, index):
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)

        state = torch.zeros(12)
        state[2] = 1.0

        states = [state.clone()]
        actions = []

        wind_seq = generate_wind_sequence(
            self.sequence_length - 1,
            self.wind_min,
            self.wind_max,
            batch_size=1,
            dt=self.dt,
        )[:, :, :].squeeze(0)

        for t in range(self.sequence_length - 1):
            action = 0.45 + 0.10 * torch.randn(4, generator=generator)
            action = action.clamp(0.0, 1.0)

            wind = wind_seq[t].unsqueeze(0)
            next_state = self.physics.step(
                state.unsqueeze(0),
                action.unsqueeze(0),
                wind,
            ).squeeze(0)

            next_state = self.physics.project_state(next_state)
            states.append(next_state.detach().clone())
            actions.append(action)

            state = next_state.detach()

        return {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "wind": wind_seq,
        }


def collate_trajectories(batch):
    states = torch.stack([x["states"] for x in batch])
    actions = torch.stack([x["actions"] for x in batch])
    wind = torch.stack([x["wind"] for x in batch])

    return {
        "states": states,
        "actions": actions,
        "wind": wind,
    }