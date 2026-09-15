import torch
import torch.nn.functional as F


def rotation_loss(pred, target):
    return F.mse_loss(pred[..., 6:9], target[..., 6:9])


def state_loss(
    pred,
    target,
    lambda_position=1.0,
    lambda_velocity=1.0,
    lambda_rotation=1.0,
    lambda_angular_velocity=1.0,
):
    lp = F.mse_loss(pred[..., 0:3], target[..., 0:3])
    lv = F.mse_loss(pred[..., 3:6], target[..., 3:6])
    lr = rotation_loss(pred, target)
    lw = F.mse_loss(pred[..., 9:12], target[..., 9:12])

    return (
        lambda_position * lp
        + lambda_velocity * lv
        + lambda_rotation * lr
        + lambda_angular_velocity * lw
    )


def multistep_loss(
    predictions,
    targets,
    gamma=0.99,
    lambda_position=1.0,
    lambda_velocity=1.0,
    lambda_rotation=1.0,
    lambda_angular_velocity=1.0,
):
    H = predictions.shape[1] - 1
    total = predictions.new_tensor(0.0)

    for k in range(1, H + 1):
        weight = gamma ** (k - 1)

        total = total + weight * state_loss(
            predictions[:, k],
            targets[:, k],
            lambda_position,
            lambda_velocity,
            lambda_rotation,
            lambda_angular_velocity,
        )

    return total / H


def residual_smoothness_loss(residuals):
    if residuals.shape[1] < 2:
        return residuals.new_tensor(0.0)

    return torch.mean(
        (residuals[:, 1:] - residuals[:, :-1]).square()
    )