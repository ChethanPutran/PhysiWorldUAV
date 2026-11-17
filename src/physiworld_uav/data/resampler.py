"""Physics-rate → model-rate conversion by decimation (plan §6.5-DS2, decision D4).

Ground-truth states are **decimated, never filtered**: filtering states (the
previous Butterworth/filtfilt resampler) breaks the kinematic consistency
ṗ = v that the physics prior relies on.

With control decimation D (model step j ↔ physics step k = D·j):

* ``inst`` fields: value at t_j (k = D·j); for ``ground_truth/rotor_speed``,
  ``ground_truth/rotor_thrust`` and the IMU, the interval mean over the
  instants k = D·j+1 … D·(j+1) is provided as ``<path>__mean``.
* ``intv`` fields that are held by construction (``actions/*``,
  ``control/*``): value at k = D·j; their constancy over the interval is
  verified (a violation means the runner did not hold the command).
* other ``intv`` fields (wrench, applied wind): interval mean.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .schema import BY_PATH

HELD_PREFIXES = ("actions/", "control/")
MEAN_INST = ("ground_truth/rotor_speed", "ground_truth/rotor_thrust",
             "observations/imu_accel_B", "observations/imu_gyro_B")


class DecimationError(ValueError):
    pass


def decimate_episode(arrays: Mapping[str, np.ndarray], decimation: int, *,
                     check_held: bool = True, atol: float = 1e-6) -> dict[str, np.ndarray]:
    D = int(decimation)
    if D < 1:
        raise ValueError("decimation must be >= 1")
    n = int(np.asarray(arrays["time/t"]).shape[0]) - 1
    if n % D != 0:
        raise DecimationError(f"episode has {n} physics steps, not a multiple of decimation {D}")
    m = n // D
    out: dict[str, np.ndarray] = {}
    for path, arr in arrays.items():
        spec = BY_PATH.get(path)
        arr = np.asarray(arr)
        if spec is None:
            continue
        if spec.sem == "inst":
            out[path] = arr[::D]
            if path in MEAN_INST:
                tail = arr[1:].reshape((m, D) + arr.shape[1:]).astype(np.float64)
                out[path + "__mean"] = tail.mean(axis=1).astype(arr.dtype)
        else:
            blocks = arr.reshape((m, D) + arr.shape[1:])
            if path.startswith(HELD_PREFIXES):
                if check_held and arr.dtype != np.bool_:
                    first = blocks[:, :1]
                    same = np.isclose(blocks, first, atol=atol, equal_nan=True)
                    if not same.all():
                        raise DecimationError(f"{path} is not constant within control intervals")
                out[path] = blocks[:, 0]
            elif arr.dtype == np.bool_:
                out[path] = blocks.any(axis=1)
            else:
                out[path] = blocks.astype(np.float64).mean(axis=1).astype(arr.dtype)
    return out
