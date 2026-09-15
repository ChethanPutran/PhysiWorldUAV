import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from physiworld_uav.data.dataset import SyntheticUAVDataset, collate_trajectories
from physiworld_uav.physics.quadrotor import QuadrotorPhysics
from physiworld_uav.models.world_model import PhysicsPriorWorldModel
from physiworld_uav.losses.losses import multistep_loss, residual_smoothness_loss
from physiworld_uav.training.curriculum import curriculum_horizon


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_physics(cfg, device):
    p = cfg["physics"]

    physics = QuadrotorPhysics(
        mass=p["mass"],
        gravity=p["gravity"],
        arm_length=p["arm_length"],
        kf=p["kf"],
        km=p["km"],
        inertia=tuple(p["inertia"]),
        dt=cfg["data"]["dt"],
    )

    physics.J = physics.J.to(device)
    return physics


def train(cfg, device="cuda"):
    device = torch.device(
        device if device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )

    set_seed(cfg["seed"])

    physics = build_physics(cfg, device)

    train_dataset = SyntheticUAVDataset(
        cfg["data"]["train_trajectories"],
        cfg["data"]["sequence_length"],
        cfg["data"]["dt"],
        physics,
        cfg["data"]["train_wind_min"],
        cfg["data"]["train_wind_max"],
        cfg["seed"],
    )

    loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=0,
    )

    model = PhysicsPriorWorldModel(
        physics=physics,
        state_dim=cfg["data"]["state_dim"],
        action_dim=cfg["data"]["action_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    for epoch in range(cfg["training"]["epochs"]):
        model.train()

        horizon = min(
            curriculum_horizon(epoch),
            cfg["data"]["sequence_length"] - 1,
        )

        epoch_loss = 0.0

        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            wind = batch["wind"].to(device)

            predictions, residuals = model(
                states,
                actions,
                wind,
                horizon=horizon,
            )

            target = states[:, :horizon + 1]

            loss = multistep_loss(
                predictions,
                target,
                gamma=cfg["training"]["gamma"],
                lambda_position=cfg["training"]["lambda_position"],
                lambda_velocity=cfg["training"]["lambda_velocity"],
                lambda_rotation=cfg["training"]["lambda_rotation"],
                lambda_angular_velocity=cfg["training"]["lambda_angular_velocity"],
            )

            loss = loss + (
                cfg["training"]["lambda_residual_smooth"]
                * residual_smoothness_loss(residuals)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["grad_clip"],
            )

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"H={horizon:03d} | "
            f"loss={epoch_loss:.6f}"
        )

    return model