import torch


@torch.no_grad()
def rollout(model, initial_state, actions, wind=None):
    model.eval()

    return model.imagine(
        initial_state,
        actions,
        wind,
    )


def rollout_error(predictions, targets):
    position_error = torch.linalg.vector_norm(
        predictions[..., 0:3] - targets[..., 0:3],
        dim=-1,
    )

    return position_error