import argparse
from pathlib import Path

import torch
import yaml

from physiworld_uav.training.train import train


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Using device: {device}")

    model = train(
        cfg,
        device=device,
    )

    output = Path("checkpoints")
    output.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        output / "physics_prior_world_model.pt",
    )

    print(
        f"Saved model to "
        f"{output / 'physics_prior_world_model.pt'}"
    )


if __name__ == "__main__":
    main()