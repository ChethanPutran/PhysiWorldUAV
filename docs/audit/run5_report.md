# Run 5 report — pilot dataset (`physiworld_pilot_v0.5.0`)

Date: 2026-11-17 · Status: **complete — 147 of 147 episodes written, validator ok (0 hard, 0 soft), `index.json` complete = true.** The first run aborted at 143 (F1); the missing 4 were generated with the fixed generator via `--resume` (see "Changes made after the first run").

Approved at the start of Run 5: Q3 coefficients (d = 0.30 s⁻¹, per-axis C_D = 1.1 / 1.1 / 1.3, areas 0.0070 / 0.0076 / 0.0180 m², R = 0.10 m, ρ = 1.225 kg/m³); rotor-drag envelope widened to μ ≤ 0.20. `truth_default` is `validated: true`. CPU tests 132 passed; Isaac tests 17 passed after the change.

## Result

| Item | Value |
|---|---|
| Planned / written | 147 / 143 |
| Status of written episodes | 139 `completed`, 4 `terminated_ground` |
| Validator (`generate validate`) | ok = True, 0 hard, 0 soft findings (143 episodes) |
| `index.json` | **not written** (the run aborted before the index step) |
| NaNs in state or IMU | 0 |
| IMU valid fraction | 1.0 in every episode |
| Force-envelope violations | 0 of 143 episodes outside an envelope (max rotor-drag violation fraction 0.006, in an open-loop episode; limit 0.05) |
| Throughput | about 24 s per episode of about 2000 physics steps |

## Findings

### F1 — Run aborted at episode 144; the error was swallowed (bug)
Episode 144 is `id_ol_g00010_e1`, a Mode B replay of `id_ol_g00010_e0`. That baseline ended on ground contact after 592 steps (148 control commands), but its replay needs 196 commands, so `runner.py:146` raises `ValueError`. In `generate.py` the loop's `except BaseException` calls `sim.close()` **before** re-raising; `SimulationApp.close()` terminates the process, so no traceback was printed and neither `index.json` nor the final validation ran.
Missing episodes: `id_ol_g00010_e1`, `id_ol_g00010_e2`, `id_ol_g00011_e1`, `id_ol_g00011_e2`.
(The cause is derived from the code path and the replay ordering; I did not re-run episode 144 to capture the exception.)

### F2 — Open-loop programs can crash the vehicle (4 terminations)
| Episode | Role | Steps run / planned |
|---|---|---|
| `id_ol_g00010_e0` | baseline | 592 / 784 |
| `id_ol_g00006_e1`, `_e2` | wind-scale replays | 676 / 784, 488 / 784 |
| `id_ol_g00008_e2` | wind-scale replay | 560 / 576 |

The baseline of group 10 applies a −0.64 collective step at about 1 s from 2.06 m altitude with no feedback. The replays of groups 6 and 8 completed in their baselines but crashed when the wind changed, so these programs are marginal at about 2 m altitude. Open-loop tilt reaches 44.8°. The four terminated episodes end at z = 0.10–0.13 m; the lowest point of any other open-loop episode is 0.60 m.

### F3 — Rotor-drag envelope widening works
All 10 OOD-magnitude episodes (mean wind 4.3–7.7 m/s, peak 8.2 m/s) are inside the μ ≤ 0.20 envelope (violation fraction 0.000). Under the old 0.15 limit, 3 of 4 such episodes in the dev run had been flagged.

### F4 — Closed-loop tracking error is larger than in the dev run
Dev-run ID tracking error was ≤ 0.4 m. Now:

| Part | Max tracking error | Max tilt |
|---|---|---|
| id_cl (64) | 1.55 m | 31.6° |
| ood_mag (10) | 1.05 m | 36.0° |
| all other OOD parts | ≤ 0.38 m | ≤ 29.6° |
| phys_val (12, no wind) | 0.19 m | 24.8° |

The largest ID errors are in the attitude-excitation scenarios (`attitude_combined` 1.55 m, `roll_doublet` 1.21 m, `pitch_doublet` 0.87 m). For the other scenarios the median max-error is ≤ 0.45 m. I did not investigate this further. `ood_mag_g00000_e0` reaches 36.0° tilt, slightly above the 35° feasibility limit, at 7.7 m/s mean wind.

### F5 — Saturation
3 episodes have a saturated-control fraction of 0.5 % (`id_ol_g00006_e0`, `id_ol_g00010_e0`, `id_ol_g00012_e0`), all open loop. No closed-loop episode saturates.

## Decisions taken (by the user, 2026-11-17)

1. **Terminated baselines and their replays:** ground contact is a legitimate outcome (wind plus an unsuited control input), so terminated episodes are **kept**, flagged `status = terminated_ground`, and the planner is unchanged. A replay of a baseline that ended early replays the commands that exist and is flagged `replay_truncated_to_baseline`. Whether terminated episodes enter model-ready training data is a later choice; the pre-impact part is valid, the contact itself is outside the physics prior.
2. **Swallowed exception:** fixed (below).
3. **Attitude-excitation tracking error (F4):** investigated, nothing to fix. The 10 attitude-mode episodes use a deliberately slow position-hold outer loop (`kp_xy` 0.8, `kd_xy` 1.2, max tilt 0.35 rad) whose job is to keep the doublets near their start point, so the roll/pitch excitation displaces the vehicle by design. Position-mode tracking is good: ID max 0.46 m, other OOD max 0.38 m, OOD magnitude max 1.05 m at 7.7 m/s mean wind, 107 position-mode episodes with median max-error 0.19 m. Accepted; attitude-mode errors should not be read as controller quality.

## Changes made after the first run

* `generate.py`: a failing episode is logged with its traceback, recorded under `failures` in `index.json`, and the run **continues**; an abort now logs the traceback and saves the index before closing the simulator.
* **Checkpointing:** `index.json` is rewritten atomically after every episode (`in_progress: true`, validation pending) and finalised at the end. The shard is closed after every episode, so a crash between episodes cannot leave it half-written (only the episode being written at that instant can be lost; an interrupted one is removed on resume).
* **`--resume`:** rebuilds the state from the shards, skips episodes already written, generates the rest. Without `--resume`, running into an existing dataset directory is refused.
* `runner.py`: replay of a short baseline is truncated and flagged instead of failing.
* Stale tests updated after `truth_default` became `validated: true`. CPU suite: 141 tests.
* Live status: `python -m physiworld_uav.visualization.dataset_report --dataset DIR --serve --port 0`.

## Resume result

Generated `id_ol_g00010_e1` (552 steps, ground contact), `id_ol_g00010_e2` (480, ground contact), `id_ol_g00011_e1` and `_e2` (576, completed). Final dataset: 141 completed, 6 `terminated_ground`, 0 failures. The resumed episodes ran in a second process; cross-process bit-exactness has not been checked (only within-process repeatability was validated in Run 4). A copy of the 143-episode state is in `data/datasets/physiworld_pilot_v0.5.0_backup143` (git-ignored).

## Addendum — statistical, physical and coverage quality check (2026-11-17)

Tool: `python -m physiworld_uav.data.statistical_validator --dataset data/datasets/physiworld_pilot_v0.5.0` (new, `src/physiworld_uav/data/statistical_validator.py`; report and figures in `outputs/reports/physiworld_pilot_v0.5.0_quality/`, git-ignored). Result on the pilot: **0 fail, 4 warn, 44 pass, 5 info.**

**Passed.** No frozen state or rotor channel; no sample outside the physical limits (ground penetration, speed, rate, tilt, rotor speed); no non-finite value; no duplicate episode, near-duplicate or repeated seed; realised wind statistics inside their regime limits for all six regimes; ID domain factors inside ±halfwidth; scenario mix within 0.08 total-variation distance of the plan; domain-factor occupancy ≥ 90 % per factor (10 bins) and ≥ 76 % for factor pairs (5×5); azimuth coverage 12/12. Kinematic consistency: Δp/dt vs velocity p99 0.009 m/s; gyro vs true ω_B p99 8e-6 rad/s; quaternion norm and sign continuity exact. **Newton's law on the ground truth holds: Δv/dt vs R·(ΣT/m)ẑ + g + F_ext/m has p50 0.0007 and p99 0.030 m/s² (max 0.47)**, i.e. the labels agree with the logged rotor thrust, attitude and injected force.

**Warnings (open).**
1. **IMU vs ground-truth acceleration (`physical.acc_imu`).** Δv/dt vs R·(IMU specific force) + g: p50 0.008 but p99 1.65 m/s² (max 4.7); 47 of 141 episodes have p99 > 1. Characterisation (measured on the worst episodes): the residual is vertical (mean +2.4…+2.6 m/s² in the affected samples), appears in contiguous high-speed or high-angular-rate phases (accel_decel, rotor doublets), is independent of the control-step phase, of rotor saturation, of the external force and of a one-step sensor lag (shifting the IMU by ±2 physics steps changes nothing), and correlates with ‖ω‖ (ρ ≈ 0.64). The ground-truth force balance above is tight at the same samples. This is consistent with a sensor lever arm (IMU not at the centre of mass) or another angular-rate-dependent term, **but I have not verified the cause**. Follow-up: read the IMU prim position relative to the centre of mass in Isaac Sim and, if offset, either move the sensor to the CoM, or document the lever arm and compensate it in the model/encoder. Until then the IMU is not a clean "specific force at the CoM", which matters for the encoder input.
2. **Open-loop action–wind correlation (`correlation.action_wind.open_loop`).** max |ρ(rotor speed, wind)| 0.39 (within-episode 0.22), ρ(thrust, |wind|) 0.42 over 24 episodes (limit 0.3). Closed loop is fine (0.10 / 0.05 / −0.05). With only 24 open-loop episodes this may be chance, but it is not clean enough to claim decorrelation for the open-loop programs.
3. **Domain-factor correlation (`correlation.domain_factors`).** Largest |ρ| 0.59 (inertia_shape_x × motor_time_constant) over 32 distinct ID domains, Bonferroni p = 0.009. The design has too few domains per part; more domains (or a single joint Sobol sequence across parts) would remove this.
4. **Too few independent groups per evaluation role.** `val_ood_*` have 2–3 groups, `test_ood_direction/temporal/gust/trajectory` and `physics_validation` have 4 (the check asks for 5). Per-role results on the pilot are therefore indicative only.

**Information.** The ID wind distribution is heavily calm-weighted: 50 of 147 episodes have mean wind < 0.5 m/s (12 are no-wind), and only 11 have 3–8.5 m/s of which 10 are OOD magnitude. Wind elevation coverage 75 % of 8 bins (ID is ±5° by design). The residual of nominal physics grows with wind in closed loop (ρ = 0.59; mean ‖Δv‖ 0.57 m/s² calm → 2.7 m/s² at 6–8.5 m/s) and is weaker in open loop (ρ = 0.38); the calm-air floor is parameter mismatch plus unmodelled drag. Test samples of `test_ood_magnitude` lie almost entirely inside the training range of speed, tilt, ‖ω‖, altitude and rotor speed (≤ 0.4 % outside), so the OOD is in the **wind**, not in the visited state space; that is what the thesis intends, but it means an OOD gap must come from the unobserved disturbance.
