# 20 Hz controller and model step: tracking check (2026-11-25)

Question: does a controller that runs at 20 Hz (control_decimation 10, model step 0.05 s) track as well as the 50 Hz pilot setting (decimation 4, 0.02 s)? The same plan (`config/experiment/stability_20hz.yaml`: 12 closed-loop scenarios, nominal vehicle, aero truth, 10 s, once without wind and once with `id_wide` wind) was generated twice in the simulator, `physics=isaac_200hz` (A) and `physics=isaac_200hz_m20` (B). Script `scripts/compare_tracking.py`; table copy `outputs/reports/tracking_20hz_vs_50hz.md`. Checked in the stored arrays that the rotor command changes every 4 (A) and every 10 (B) physics steps.

Position tracking error after 1 s, stab50_v0.1.0 (A) vs stab20_v0.1.0 (B), 24 common episodes (24 / 24 found)

| Part | Scenario | n | RMS A [m] | RMS B [m] | RMS B / A | max A [m] | max B [m] |
|---|---|---|---|---|---|---|---|
| calm | accel_decel | 1 | 0.082 | 0.082 | 1.00 | 0.156 | 0.156 |
| calm | attitude_combined | 1 | 0.354 | 0.478 | 1.35 | 0.548 | 0.676 |
| calm | circle | 1 | 0.046 | 0.046 | 1.00 | 0.050 | 0.050 |
| calm | combined | 1 | 0.056 | 0.056 | 1.00 | 0.061 | 0.061 |
| calm | hover | 1 | 0.002 | 0.002 | 0.98 | 0.009 | 0.009 |
| calm | line | 1 | 0.138 | 0.138 | 1.00 | 0.156 | 0.156 |
| calm | roll_doublet | 1 | 0.095 | 0.305 | 3.20 | 0.306 | 0.383 |
| calm | sinusoidal | 1 | 0.017 | 0.017 | 1.00 | 0.027 | 0.027 |
| calm | translation | 1 | 0.137 | 0.137 | 1.00 | 0.154 | 0.154 |
| calm | vertical | 1 | 0.002 | 0.002 | 1.22 | 0.009 | 0.009 |
| calm | waypoints | 1 | 0.030 | 0.030 | 1.00 | 0.072 | 0.071 |
| calm | yaw_step | 1 | 0.002 | 0.002 | 0.98 | 0.009 | 0.009 |
| windy | accel_decel | 1 | 0.463 | 0.463 | 1.00 | 0.527 | 0.527 |
| windy | attitude_combined | 1 | 1.954 | 1.643 | 0.84 | 2.549 | 2.162 |
| windy | circle | 1 | 0.134 | 0.134 | 1.00 | 0.197 | 0.197 |
| windy | combined | 1 | 0.380 | 0.380 | 1.00 | 0.472 | 0.472 |
| windy | hover | 1 | 0.375 | 0.375 | 1.00 | 0.392 | 0.392 |
| windy | line | 1 | 0.259 | 0.259 | 1.00 | 0.321 | 0.321 |
| windy | roll_doublet | 1 | 1.825 | 1.587 | 0.87 | 2.020 | 1.736 |
| windy | sinusoidal | 1 | 0.168 | 0.168 | 1.00 | 0.239 | 0.239 |
| windy | translation | 1 | 0.276 | 0.276 | 1.00 | 0.304 | 0.304 |
| windy | vertical | 1 | 0.289 | 0.289 | 1.00 | 0.376 | 0.376 |
| windy | waypoints | 1 | 0.384 | 0.384 | 1.00 | 0.473 | 0.473 |
| windy | yaw_step | 1 | 0.422 | 0.422 | 1.00 | 0.461 | 0.461 |
| all | all | 24 | 0.329 | 0.320 | 0.97 | 2.549 | 2.162 |

## Reading

- All 48 episodes (24 per rate) completed; none failed or terminated early, so the 20 Hz controller is stable on these 12 scenarios, with and without wind.
- Position-controlled scenarios (hover, line, translation, accel/decel, circle, sinusoidal, combined, waypoints, vertical, yaw step) track identically at both rates (ratio 1.00 to three digits). The position loop is slow compared with 20 Hz.
- The two attitude-mode scenarios are sensitive to the rate: without wind `roll_doublet` has 0.305 m RMS against 0.095 m (3.2 times) and `attitude_combined` 0.478 against 0.354 m (1.35 times); with wind they are not worse (0.84 and 0.87 times; their absolute errors, 1.6 to 2.0 m, are large at both rates). Overall mean RMS 0.320 m (20 Hz) against 0.329 m (50 Hz).
- Limits: one episode per cell, one nominal vehicle, no vehicle randomisation. The differences in the attitude scenarios are not statistically established.
- The rest of the pipeline at 20 Hz was checked on this data: training reads the decimation from the manifest (model step 0.05 s in the checkpoint), evaluation refuses a model trained at another rate, B1 and the oracle B6a run at 0.05 s (oracle 0.024 m at 20 steps = 1 s).
