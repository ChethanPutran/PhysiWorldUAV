import torch


def position_error(predicted, target):
    return torch.linalg.vector_norm(
        predicted[..., 0:3] - target[..., 0:3],
        dim=-1,
    )

def average_position_error(predicted, target):
    return position_error(predicted, target).mean()


def horizon_errors(predicted, target):
    """
    Returns E_p(H) for every H.

    predicted/target:
        [B, T, state_dim]
    """
    errors = position_error(predicted, target)

    return errors[:, 1:].mean(dim=0)


def reliability_curve(predicted, target, threshold):
    errors = position_error(predicted, target)

    return (
        errors[:, 1:] <= threshold
    ).float().mean(dim=0)