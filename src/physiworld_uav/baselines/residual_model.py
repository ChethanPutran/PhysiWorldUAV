import torch


@torch.no_grad()
def rollout_residual_model(model, initial_state, actions, wind=None):
    model.eval()

    return model.imagine(
        initial_state,
        actions,
        wind,
    )