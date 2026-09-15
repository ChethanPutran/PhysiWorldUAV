import matplotlib.pyplot as plt
import numpy as np


def plot_position_trajectory(
    target,
    prediction,
    title="UAV Position Prediction",
):
    target = np.asarray(target)
    prediction = np.asarray(prediction)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        label="Ground truth",
    )

    ax.plot(
        prediction[:, 0],
        prediction[:, 1],
        prediction[:, 2],
        label="Prediction",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()

    return fig


def plot_horizon_error(errors):
    errors = np.asarray(errors)

    fig = plt.figure()
    plt.plot(
        np.arange(1, len(errors) + 1),
        errors,
    )
    plt.xlabel("Prediction horizon")
    plt.ylabel("Position error")
    plt.title("Long-Horizon Position Error")
    plt.grid(True)

    return fig