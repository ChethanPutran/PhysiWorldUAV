import torch


@torch.no_grad()
def rollout_physics_only(physics, initial_state, actions, wind=None):
    state = initial_state
    predictions = [state]

    for t in range(actions.shape[1]):
        current_wind = None if wind is None else wind[:, t]

        state = physics.step(
            state,
            actions[:, t],
            current_wind,
        )

        state = physics.project_state(state)
        predictions.append(state)

    return torch.stack(predictions, dim=1)