import torch

from .metrics import position_error


def reliable_imagination_horizon(
    predicted,
    target,
    threshold=0.5,
):
    """
    First-failure definition.

    A trajectory is reliable at H only if every step up to H has
    position error <= threshold.
    """

    errors = position_error(predicted, target)

    reliable = errors[:, 1:] <= threshold

    cumulative = torch.cumprod(
        reliable.float(),
        dim=1,
    )

    horizons = cumulative.sum(dim=1)

    return horizons


def mean_reliable_horizon(
    predicted,
    target,
    threshold=0.5,
):
    return reliable_imagination_horizon(
        predicted,
        target,
        threshold,
    ).float().mean()