# Baseline comparison on the pilot datasets v0.5.0 and v0.6.0

All numbers in this document are copied from the generated CSV tables in `outputs/reports/training_compare_v05_v06/tables/` (and `outputs/eval/*.json` for the cross-evaluation). Figures are in `outputs/reports/training_compare_v05_v06/figures/`; the full auto-generated report is `report.html` / `REPORT.md` in the same folder.

> **Correction (2026-11-25, audit finding F1).** The first version of this document said the checkpoints evaluated for B4, B4 (B3 width) and B5b were their epoch-1 checkpoints and that "this is not a checkpoint bug". That was wrong. The trainer reset its best-metric at every curriculum stage, so `best.pt` was the best epoch of the *last stage*, not of the run. Five of the 16 runs of the evaluated models used a worse checkpoint than their best epoch (table in §2b). The columns "Best epoch" and "Best val RMSE" of the table in §2 are minima from `metrics.jsonl`, **not** the checkpoint that was evaluated. The trainer is fixed (best.pt now follows the lowest validation metric of the whole run); the affected models have **not** been re-evaluated or retrained yet, so the numbers for B2 (v0.6), B4, B4 (B3 width), B5a (v0.6) and B5b below are provisional.

## 1. What was run

- **Datasets.** `physiworld_pilot_v0.5.0` (raw Isaac IMU, which contains an extra −ω×v term) and `physiworld_pilot_v0.6.0` (IMU corrected at generation; wind Option C, extra diversity, same physics). Both: 147 episodes; the same split structure (63 training episodes).
- **Models.** B0 hold, B1 physics only, B2 pure neural latent, B3 physics + residual + GRU memory (the proposed model), B4 no memory (matched total parameters), B4 (B3 width), B5a learned vector field, B5b exact kinematics + learned dynamics, B6a oracle wrench (no learning), B6b B3 + oracle wrench input. "Future" and "current" refer to which interval's wrench the oracle sees. B0, B1 and B6a have no training.
- **Training.** One seed per model, curriculum horizons up to 20 steps (20 ms steps, so 0.4 s), early stopping on one validation split, three jobs per GPU. Training of the 16 runs was done in parallel on 3 GPUs of a shared host.
- **Metric.** Position RMSE in metres at open-loop horizon k steps (k = 1, 5, 10, 20 correspond to 20 ms, 0.1 s, 0.2 s, 0.4 s), given the true initial state. Windows of one episode overlap, so uncertainty intervals are computed over episodes.
- **Important limits.** One seed, a pilot-size training set, several OOD roles with only 2–7 episodes (listed below). Differences of a few percent between models are not reliable.

## 2. Training runs and resources

| Set | Model | Params | Epochs | Stopped by | Best epoch | Best val RMSE @k=20 [m] | Training time | Windows/s | Peak GPU [MiB] | Peak CPU RAM [MiB] | Mean CPU [% of 1 core] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v05 | B2 pure neural | 227,033 | 60 | epoch limit | 59 | 0.0826 | 11 min 58 s | 2715 | 446 | 998 | 111 |
| v05 | B3 physics+residual+memory | 227,033 | 31 | early stopping | 23 | 0.0321 | 15 min 19 s | 1618 | 564 | 1010 | 102 |
| v05 | B4 no memory | 226,392 | 27 | early stopping | 1 | 0.0369 | 12 min 14 s | 1570 | 550 | 996 | 104 |
| v05 | B4 (B3 width) | 72,716 | 25 | early stopping | 1 | 0.0350 | 7 min 52 s | 2395 | 452 | 990 | 106 |
| v05 | B5a learned field | 227,033 | 47 | early stopping | 39 | 0.0875 | 13 min 16 s | 2349 | 540 | 1009 | 105 |
| v05 | B5b learned dynamics | 227,033 | 34 | early stopping | 1 | 0.0743 | 8 min 21 s | 2515 | 540 | 1007 | 101 |
| v05 | B6b oracle (future) | 228,569 | 60 | epoch limit | 55 | 0.0211 | 32 min 24 s | 1451 | 564 | 1020 | 103 |
| v05 | B6b oracle (current) | 228,569 | 60 | epoch limit | 57 | 0.0213 | 32 min 26 s | 1455 | 564 | 1022 | 105 |
| v06 | B2 pure neural | 227,033 | 33 | early stopping | 15 | 0.0995 | 6 min 31 s | 3273 | not recorded | not recorded | not recorded |
| v06 | B3 physics+residual+memory | 227,033 | 38 | early stopping | 30 | 0.0321 | 19 min 52 s | 1524 | 564 | 1009 | 101 |
| v06 | B4 no memory | 226,392 | 27 | early stopping | 1 | 0.0369 | 12 min 13 s | 1572 | 550 | 995 | 104 |
| v06 | B4 (B3 width) | 72,716 | 25 | early stopping | 1 | 0.0350 | 10 min 44 s | 1758 | 452 | 982 | 105 |
| v06 | B5a learned field | 227,033 | 36 | early stopping | 16 | 0.0930 | 12 min 02 s | 2140 | 540 | 998 | 106 |
| v06 | B5b learned dynamics | 227,033 | 30 | early stopping | 1 | 0.0744 | 10 min 02 s | 1923 | 540 | 1003 | 107 |
| v06 | B6b oracle (future) | 228,569 | 60 | epoch limit | 53 | 0.0234 | 33 min 39 s | 1354 | 564 | 1021 | 103 |
| v06 | B6b oracle (current) | 228,569 | 60 | epoch limit | 53 | 0.0244 | 32 min 50 s | 1420 | 564 | 1018 | 104 |

_Training time is the time spent in training epochs. Windows/s is training throughput with up to 9 jobs sharing 3 GPUs, so it is not a stand-alone speed. GPU memory is the process memory seen by nvidia-smi. The v0.6 B2 run has no memory or CPU samples because the sampler was started after it (the report states this). The columns "Best epoch" and "Best val RMSE" are minima from `metrics.jsonl`; they are not the evaluated checkpoint for the runs listed in §2b._

How to read it:

- Training is cheap: 6–34 min per run, about 450–565 MiB of GPU memory and about 1 GB of CPU RAM.
- Every training job used about 100–110 % of one CPU core, i.e. training is CPU-bound at this model size. Three jobs per GPU were therefore possible.
- B6b trains for the full 60 epochs (33 min); the other models stop earlier.
- **B4, B4 (B3 width) and B5b have "best epoch 1"** in both sets (minimum of the validation curve): the validation error was lowest after the first epoch and rose afterwards (§2a). The checkpoint that was evaluated is a different, later epoch (§2b).

## 2a. Validation error rises with training (B4, B4 (B3 width), B5b)

Validation position RMSE (v0.5, 20-step horizon in the last column) over training, from `outputs/experiments/v05_<model>/metrics.jsonl`. Epoch 1 uses half the learning rate (warm-up) and is therefore close to the physics model plus a small residual.

| Model | Epoch | Train horizon | Train loss (total) | Val k=1 [m] | Val k=5 [m] | Val k=20 [m] |
|---|---|---|---|---|---|---|
| B4 | 1 | 1 | 0.0057 | 0.00010 | 0.0022 | **0.0369** |
| B4 | 2 | 1 | 0.0034 | 0.00011 | 0.0025 | 0.0408 |
| B4 | 4 | 1 | 0.0027 | 0.00013 | 0.0031 | 0.0493 |
| B4 | 12 | 10 | 0.0083 | 0.00013 | 0.0031 | 0.0514 |
| B4 | 27 | 20 | 0.0074 | 0.00013 | 0.0031 | 0.0505 |
| B5b | 1 | 1 | 377 | 0.00024 | 0.0050 | **0.0743** |
| B5b | 4 | 1 | 105 | 0.00026 | 0.0061 | 0.0951 |
| B5b | 12 | 5 | 48 | 0.00025 | 0.0062 | 0.0935 |
| B5b | 34 | 20 | 24 | 0.00028 | 0.0061 | 0.0843 |
| B3 (for comparison) | 1 | 1 | 377 | 0.00011 | 0.0024 | 0.0410 |
| B3 | 4 | 1 | 106 | 0.00010 | 0.0020 | 0.0340 |
| B3 | 20 | 10 | 38 | 0.00009 | 0.0019 | 0.0330 |
| B3 | 31 | 20 | 30 | 0.00009 | 0.0020 | 0.0346 |

(B4 has no latent decoder, so its total loss has no decoder term and is not comparable in size with B3 and B5b. B4 (B3 width) behaves like B4: epoch 1 0.0350, epoch 25 0.0559.)

What this shows:

- For B4 and B5b the training loss falls (B4 by half in four epochs) while the validation error at **every** horizon, including k=1, becomes worse. B3 improves on all horizons over the same epochs. For B4 the final 20-step error (0.0505 m) is above the physics-only model B1 (0.0447 m on validation), while the epoch-1 checkpoint (0.0369 m) is below it.
- So the best epoch of B4, B4 (B3 width) and B5b is epoch 1: later epochs made validation worse at every horizon. This is a real generalisation gap on this data. Because of the selection bug (§2b) the evaluated checkpoints are *later, worse* epochs (19, 17, 22 or 26), not epoch 1, so their test numbers are worse than a correct selection would give.
- **Hypothesis (not tested):** models without the GRU context cannot infer the vehicle from history, so their residual fits the average of the training vehicles and does not transfer to the validation vehicles; B3's memory avoids this. A test would be to train B4 with a lower learning rate or stronger weight decay, or compare train and validation error per vehicle; this needs new training and has not been run.

## 2b. Checkpoint selection bug (audit F1)

`best.pt` is the checkpoint the evaluation scripts load. Because the trainer restarted its "best" at each curriculum stage, the first epoch of every new stage overwrote `best.pt`. Checked for all 16 runs (20-step validation position RMSE):

| Run | Epoch of evaluated `best.pt` | Val RMSE of that epoch [m] | True best epoch | Val RMSE of true best [m] | Affected |
|---|---|---|---|---|---|
| v0.5 B4 | 19 | 0.0488 | 1 | 0.0369 | yes |
| v0.5 B4 (B3 width) | 17 | 0.0509 | 1 | 0.0350 | yes |
| v0.5 B5b | 26 | 0.0813 | 1 | 0.0743 | yes |
| v0.6 B4 | 19 | 0.0488 | 1 | 0.0369 | yes |
| v0.6 B4 (B3 width) | 17 | 0.0509 | 1 | 0.0350 | yes |
| v0.6 B5b | 22 | 0.0973 | 1 | 0.0744 | yes |
| v0.6 B2 | 25 | 0.1151 | 15 | 0.0995 | yes |
| v0.6 B5a | 28 | 0.1025 | 16 | 0.0930 | yes |
| v0.5 B2, v0.5 B5a, B3 (both), B6b (all four) | = true best | | | | no |

Consequences:

- The test and validation numbers of the eight affected runs are for the evaluated (worse) checkpoints. B3 and B6b are not affected.
- B3's validation lead over B4 is 34 % as evaluated (0.0321 vs 0.0488 m) but about 13 % against B4's real best epoch (0.0369 m). The "B3 beats B4" statements in §3, §4 and §6 are therefore inflated and provisional.
- The "late rise" of B2 and B5a on v0.6 (problems log C8) is real in the curves, but those runs were also evaluated at a worse epoch than their best.
- Fix: `training/trainer.py` now keeps `best_global` / `best_epoch` for the whole run and saves `best.pt` only when the validation metric improves on it (without a validation set the per-stage rule is kept, because the training loss at different horizons is not comparable). The summary now also stores `best_epoch`. A regression test (`tests/cpu/test_models.py::test_best_checkpoint_is_the_global_validation_minimum_across_curriculum_stages`) fails on the old logic.
- Not done: re-evaluation or retraining of the affected models. Only the last 3 epoch checkpoints are kept, so the true best epoch 1, 15 or 16 is gone; fixing the numbers needs retraining (with approval).

## 3. Main result: open-loop position error on the in-distribution test split

### v0.5.0 (test ID, 14 episodes, 6198 windows)

| Model | k=1 (20 ms) | k=5 (0.1 s) | k=10 (0.2 s) | k=20 (0.4 s) | Attitude error at k=20 [deg] |
|---|---|---|---|---|---|
| B0 hold | 0.0187 | 0.094 | 0.188 | 0.376 | 7.92 |
| B1 physics only | 0.0001 | 0.003 | 0.013 | 0.053 | 2.13 |
| B2 pure neural | 0.0019 | 0.011 | 0.030 | 0.093 | 6.22 |
| **B3 physics+residual+memory** | 0.0001 | 0.002 | 0.009 | 0.035 | 1.32 |
| B4 no memory | 0.0001 | 0.003 | 0.012 | 0.050 | 3.00 |
| B4 (B3 width) | 0.0001 | 0.003 | 0.011 | 0.045 | 2.51 |
| B5a learned field | 0.0020 | 0.011 | 0.030 | 0.096 | 2.09 |
| B5b learned dynamics | 0.0003 | 0.008 | 0.032 | 0.131 | 2.88 |
| B6a oracle wrench, future | 0.0001 | 0.002 | 0.008 | 0.033 | 0.70 |
| B6b B3+oracle, future | 0.0001 | 0.001 | 0.006 | 0.025 | 1.15 |
| B6b B3+oracle, current | 0.0001 | 0.001 | 0.006 | 0.025 | 1.14 |

### v0.6.0 (test ID, 14 episodes, 6198 windows)

| Model | k=1 (20 ms) | k=5 (0.1 s) | k=10 (0.2 s) | k=20 (0.4 s) | Attitude error at k=20 [deg] |
|---|---|---|---|---|---|
| B0 hold | 0.0187 | 0.094 | 0.188 | 0.376 | 7.92 |
| B1 physics only | 0.0001 | 0.003 | 0.013 | 0.053 | 2.13 |
| B2 pure neural | 0.0024 | 0.013 | 0.034 | 0.109 | 5.88 |
| **B3 physics+residual+memory** | 0.0001 | 0.002 | 0.009 | 0.037 | 1.41 |
| B4 no memory | 0.0001 | 0.003 | 0.012 | 0.050 | 3.00 |
| B4 (B3 width) | 0.0001 | 0.003 | 0.011 | 0.045 | 2.51 |
| B5a learned field | 0.0024 | 0.014 | 0.037 | 0.115 | 2.05 |
| B5b learned dynamics | 0.0003 | 0.007 | 0.029 | 0.113 | 2.03 |
| B6a oracle wrench, future | 0.0001 | 0.002 | 0.008 | 0.033 | 0.70 |
| B6b B3+oracle, future | 0.0001 | 0.002 | 0.006 | 0.026 | 1.06 |
| B6b B3+oracle, current | 0.0001 | 0.001 | 0.006 | 0.026 | 1.08 |

Description:

- B0 (hold the last state) grows quickly with horizon (0.376 m at k=20). All learned models and the physics models are much better, so the task is non-trivial.
- **B3 is the best of the non-oracle models on test ID**: 0.035 m on v0.5 and 0.037 m on v0.6, against 0.050 m for B4, 0.045 m for B4 (B3 width), 0.053 m for physics only (B1) and 0.093–0.131 m for B2, B5a and B5b. It also beats B1, so the learned residual improves on the physics-only model.
- **B2 (no physics) is clearly worse** than every model with a physics prior (0.093 m on v0.5, 0.109 m on v0.6), and B5a and B5b (learned dynamics with no explicit physics term) are in the same range. The physics prior is what makes the difference.
- The oracle-wrench models (B6a 0.033, B6b 0.025–0.026 on test ID) show how much better the model could be with knowledge of the wrench. B3 is within 0.002–0.004 m of B6a, and about 0.010 m above B6b.
- Validation (val ID) gives the same order (matrix below).

## 4. All evaluation roles, position RMSE at k=20 [m]

Rows are models, columns are the evaluation roles. Number of episodes per role: val ID 11, test ID 14, OOD magnitude 7, OOD direction 4, OOD temporal 4, OOD gust 4, OOD physics 7, OOD trajectory 4.

### v0.5.0

| Model | val ID | test ID | OOD magnitude | OOD direction | OOD temporal | OOD gust | OOD physics | OOD trajectory |
|---|---|---|---|---|---|---|---|---|
| B0 hold | 0.261 | 0.376 | 0.420 | 0.424 | 0.359 | 0.169 | 0.459 | 0.287 |
| B1 physics only | 0.045 | 0.053 | 0.185 | 0.058 | 0.089 | 0.054 | 0.190 | 0.055 |
| B2 pure neural | 0.083 | 0.093 | 0.127 | 0.054 | 0.040 | 0.036 | 0.090 | 0.046 |
| **B3 physics+residual+memory** | 0.032 | 0.035 | 0.163 | 0.041 | 0.024 | 0.023 | 0.071 | 0.044 |
| B4 no memory | 0.049 | 0.050 | 0.119 | 0.038 | 0.030 | 0.039 | 0.074 | 0.045 |
| B4 (B3 width) | 0.051 | 0.045 | 0.113 | 0.033 | 0.029 | 0.041 | 0.070 | 0.044 |
| B5a learned field | 0.088 | 0.096 | 0.143 | 0.073 | 0.039 | 0.033 | 0.093 | 0.055 |
| B5b learned dynamics | 0.081 | 0.131 | 0.190 | 0.054 | 0.034 | 0.037 | 0.088 | 0.077 |
| B6a oracle wrench, future | 0.028 | 0.033 | 0.018 | 0.034 | 0.074 | 0.047 | 0.180 | 0.016 |
| B6b B3+oracle, future | 0.021 | 0.025 | 0.087 | 0.027 | 0.013 | 0.013 | 0.070 | 0.026 |
| B6b B3+oracle, current | 0.021 | 0.025 | 0.087 | 0.026 | 0.014 | 0.014 | 0.070 | 0.025 |

### v0.6.0

| Model | val ID | test ID | OOD magnitude | OOD direction | OOD temporal | OOD gust | OOD physics | OOD trajectory |
|---|---|---|---|---|---|---|---|---|
| B0 hold | 0.261 | 0.376 | 0.420 | 0.424 | 0.359 | 0.169 | 0.459 | 0.287 |
| B1 physics only | 0.045 | 0.053 | 0.185 | 0.058 | 0.089 | 0.054 | 0.190 | 0.055 |
| B2 pure neural | 0.115 | 0.109 | 0.126 | 0.075 | 0.037 | 0.032 | 0.095 | 0.082 |
| **B3 physics+residual+memory** | 0.032 | 0.037 | 0.163 | 0.040 | 0.026 | 0.020 | 0.078 | 0.046 |
| B4 no memory | 0.049 | 0.050 | 0.119 | 0.038 | 0.030 | 0.039 | 0.074 | 0.045 |
| B4 (B3 width) | 0.051 | 0.045 | 0.113 | 0.033 | 0.029 | 0.041 | 0.070 | 0.044 |
| B5a learned field | 0.103 | 0.115 | 0.124 | 0.049 | 0.036 | 0.030 | 0.081 | 0.049 |
| B5b learned dynamics | 0.097 | 0.113 | 0.140 | 0.056 | 0.030 | 0.035 | 0.111 | 0.094 |
| B6a oracle wrench, future | 0.028 | 0.033 | 0.018 | 0.034 | 0.074 | 0.047 | 0.180 | 0.016 |
| B6b B3+oracle, future | 0.023 | 0.026 | 0.095 | 0.025 | 0.012 | 0.011 | 0.059 | 0.026 |
| B6b B3+oracle, current | 0.024 | 0.026 | 0.097 | 0.024 | 0.013 | 0.010 | 0.059 | 0.026 |

Description:

- **B3 is not the best model on every OOD role.** It is best, or tied for best among non-oracle models, on val/test ID, OOD temporal (0.024 / 0.026 m) and OOD gust (0.023 / 0.020 m).
- **On OOD wind magnitude B3 is poor: 0.163 m, almost as bad as B1 (0.185 m)**, and worse than B4 (0.119 m) and B4 (B3 width) (0.113 m). The oracle-wrench physics model B6a reaches **0.018 m** on the same windows. So the physics model is accurate when the wind-induced force is known, and what fails on strong wind is the learned part: the residual does not extrapolate to wind magnitudes it has never seen. This is the main weakness the pilot shows for B3.
- On OOD physics (different vehicles) all models are between 0.070 and 0.111 m, and B1 and B6a are worst (0.190 and 0.180 m), so the nominal-parameter physics model suffers from the parameter mismatch; the learned models (0.070–0.078 m for B3, B4, B4 (B3 width)) correct part of it.
- On OOD trajectory B3 (0.044–0.046 m) is about equal to B4 and B1.
- On OOD direction B3 (0.040–0.041 m) is slightly better than B1 (0.058 m) and comparable with B4 (0.038 m).
- The OOD roles have 4–7 episodes each, so these rankings among close models are not statistically meaningful; only the large differences (OOD magnitude B3 vs B6a, B0 vs all) are.

## 5. Change from v0.5.0 to v0.6.0

Relative change of the k=20 position RMSE for the same model, trained and evaluated on its own dataset (negative = better on v0.6).

| Model | Reads IMU | val ID | test ID | OOD magnitude | OOD direction | OOD temporal | OOD gust | OOD physics | OOD trajectory |
|---|---|---|---|---|---|---|---|---|---|
| B0 hold | no | +0.0 % | -0.0 % | -0.0 % | -0.0 % | -0.0 % | +0.0 % | +0.0 % | -0.0 % |
| B1 physics only | no | -0.0 % | +0.0 % | -0.0 % | -0.0 % | -0.0 % | -0.0 % | -0.0 % | +0.0 % |
| B2 pure neural | yes | +39.8 % | +16.9 % | -0.9 % | +39.4 % | -5.4 % | -12.2 % | +5.4 % | +79.1 % |
| **B3 physics+residual+memory** | yes | -0.4 % | +4.8 % | +0.3 % | -2.4 % | +9.7 % | -11.8 % | +9.2 % | +5.3 % |
| B4 no memory | no | +0.0 % | +0.0 % | +0.0 % | -0.0 % | -0.0 % | +0.0 % | +0.0 % | +0.0 % |
| B4 (B3 width) | no | +0.0 % | +0.0 % | +0.0 % | -0.0 % | +0.0 % | -0.0 % | +0.0 % | -0.0 % |
| B5a learned field | yes | +17.2 % | +19.9 % | -13.3 % | -32.7 % | -6.3 % | -10.6 % | -12.2 % | -10.2 % |
| B5b learned dynamics | yes | +19.9 % | -14.0 % | -26.2 % | +5.1 % | -12.3 % | -6.1 % | +26.1 % | +23.1 % |
| B6a oracle wrench, future | no | +0.0 % | +0.0 % | -0.0 % | -0.0 % | +0.0 % | -0.0 % | -0.0 % | +0.0 % |
| B6b B3+oracle, future | yes | +11.1 % | +3.2 % | +8.5 % | -8.3 % | -2.8 % | -17.2 % | -15.4 % | -0.4 % |
| B6b B3+oracle, current | yes | +14.2 % | +2.8 % | +10.8 % | -5.3 % | -7.5 % | -29.7 % | -16.3 % | +6.9 % |

Description:

- Models that do not read the IMU (B0, B1, B4, B4 (B3 width), B6a) have **identical** results on both datasets (changes of ±0.0 %). This confirms that v0.5 and v0.6 contain the same physics and the same windows; only the IMU channel differs.
- B3 changes little: −0.4 % on val ID, +4.8 % on test ID, and between −12 % and +10 % on the OOD roles with 4–7 episodes.
- B2, B5a, B5b and B6b change by up to ±80 % in role-wise numbers, with no consistent sign. With one seed per model this is the size of the run-to-run noise, so **no effect of the IMU correction on model accuracy can be claimed from this experiment**.

### Cross-evaluation (test ID, position RMSE at k=20 [m])

Model trained on one dataset and evaluated on the other dataset (from `outputs/eval/v05_on_v06.json` and `v06_on_v05.json`; a name `A_on_B` is a model trained on A and evaluated on B).

| Model | v0.5 → v0.5 | v0.5 → v0.6 | v0.6 → v0.6 | v0.6 → v0.5 |
|---|---|---|---|---|
| B2 pure neural | 0.093 | 0.095 | 0.109 | 0.111 |
| B3 physics+residual+memory | 0.035 | 0.035 | 0.037 | 0.038 |
| B5a learned field | 0.096 | 0.096 | 0.115 | 0.115 |
| B5b learned dynamics | 0.131 | 0.131 | 0.113 | 0.113 |
| B6b B3+oracle, future | 0.025 | 0.025 | 0.026 | 0.026 |

The error depends on the trained model and hardly on the dataset it is evaluated on. In the open-loop rollout the models use the IMU only for the encoder context, so the −ω×v term in the v0.5 IMU does not noticeably change these predictions.

## 6. Wind versus no wind

Position RMSE on windows of the wind episodes and the no-wind baseline episodes (same groups; wind: 20 episodes, 8461 windows; no wind: 5 episodes, 2257 windows).

| Set | Regime | Episodes | Windows | Model | k=1 [m] | k=5 [m] | k=20 [m] |
|---|---|---|---|---|---|---|---|
| v05 | wind | 20 | 8461 | B1 physics only | 0.00013 | 0.0031 | 0.0535 |
| v05 | wind | 20 | 8461 | B6a oracle wrench, future | 0.00009 | 0.0020 | 0.0324 |
| v05 | wind | 20 | 8461 | B2 pure neural | 0.00193 | 0.0110 | 0.0866 |
| v05 | wind | 20 | 8461 | **B3 physics+residual+memory** | 0.00010 | 0.0022 | 0.0361 |
| v05 | wind | 20 | 8461 | B4 no memory | 0.00013 | 0.0030 | 0.0501 |
| v05 | wind | 20 | 8461 | B5b learned dynamics | 0.00031 | 0.0069 | 0.1085 |
| v05 | wind | 20 | 8461 | B6b B3+oracle, future | 0.00007 | 0.0015 | 0.0249 |
| v05 | no wind | 5 | 2257 | B1 physics only | 0.00009 | 0.0020 | 0.0328 |
| v05 | no wind | 5 | 2257 | B6a oracle wrench, future | 0.00008 | 0.0016 | 0.0249 |
| v05 | no wind | 5 | 2257 | B2 pure neural | 0.00201 | 0.0121 | 0.0972 |
| v05 | no wind | 5 | 2257 | **B3 physics+residual+memory** | 0.00008 | 0.0016 | 0.0253 |
| v05 | no wind | 5 | 2257 | B4 no memory | 0.00013 | 0.0029 | 0.0483 |
| v05 | no wind | 5 | 2257 | B5b learned dynamics | 0.00030 | 0.0077 | 0.1293 |
| v05 | no wind | 5 | 2257 | B6b B3+oracle, future | 0.00006 | 0.0011 | 0.0184 |
| v06 | wind | 20 | 8461 | B1 physics only | 0.00013 | 0.0031 | 0.0535 |
| v06 | wind | 20 | 8461 | B6a oracle wrench, future | 0.00009 | 0.0020 | 0.0324 |
| v06 | wind | 20 | 8461 | B2 pure neural | 0.00248 | 0.0139 | 0.1090 |
| v06 | wind | 20 | 8461 | **B3 physics+residual+memory** | 0.00010 | 0.0022 | 0.0373 |
| v06 | wind | 20 | 8461 | B4 no memory | 0.00013 | 0.0030 | 0.0501 |
| v06 | wind | 20 | 8461 | B5b learned dynamics | 0.00030 | 0.0068 | 0.1031 |
| v06 | wind | 20 | 8461 | B6b B3+oracle, future | 0.00007 | 0.0015 | 0.0261 |
| v06 | no wind | 5 | 2257 | B1 physics only | 0.00009 | 0.0020 | 0.0328 |
| v06 | no wind | 5 | 2257 | B6a oracle wrench, future | 0.00008 | 0.0016 | 0.0249 |
| v06 | no wind | 5 | 2257 | B2 pure neural | 0.00263 | 0.0152 | 0.1234 |
| v06 | no wind | 5 | 2257 | **B3 physics+residual+memory** | 0.00007 | 0.0015 | 0.0249 |
| v06 | no wind | 5 | 2257 | B4 no memory | 0.00013 | 0.0029 | 0.0483 |
| v06 | no wind | 5 | 2257 | B5b learned dynamics | 0.00031 | 0.0076 | 0.1190 |
| v06 | no wind | 5 | 2257 | B6b B3+oracle, future | 0.00007 | 0.0012 | 0.0210 |

Paired difference between two models on the same windows (positive = the first model is worse), with a 95 % interval from an episode-level bootstrap:

| Set | Regime | Comparison (first − second) | Episodes | RMSE first [m] | RMSE second [m] | Gap [m] | 95 % interval | Zero inside? |
|---|---|---|---|---|---|---|---|---|
| v05 | wind | B3 − B6a | 20 | 0.0361 | 0.0324 | +0.0037 | [-0.0098, +0.0135] | yes |
| v05 | wind | B3 − B4 | 20 | 0.0361 | 0.0501 | -0.0140 | [-0.0188, -0.0090] | no |
| v05 | wind | B3 − B2 | 20 | 0.0361 | 0.0866 | -0.0505 | [-0.0666, -0.0309] | no |
| v05 | no wind | B3 − B6a | 5 | 0.0253 | 0.0249 | +0.0004 | [-0.0243, +0.0212] | yes |
| v05 | no wind | B3 − B4 | 5 | 0.0253 | 0.0483 | -0.0230 | [-0.0281, -0.0172] | no |
| v05 | no wind | B3 − B2 | 5 | 0.0253 | 0.0972 | -0.0720 | [-0.0913, -0.0340] | no |
| v06 | wind | B3 − B6a | 20 | 0.0373 | 0.0324 | +0.0049 | [-0.0091, +0.0156] | yes |
| v06 | wind | B3 − B4 | 20 | 0.0373 | 0.0501 | -0.0128 | [-0.0183, -0.0076] | no |
| v06 | wind | B3 − B2 | 20 | 0.0373 | 0.1090 | -0.0716 | [-0.0916, -0.0461] | no |
| v06 | no wind | B3 − B6a | 5 | 0.0249 | 0.0249 | +0.0000 | [-0.0248, +0.0210] | yes |
| v06 | no wind | B3 − B4 | 5 | 0.0249 | 0.0483 | -0.0234 | [-0.0261, -0.0199] | no |
| v06 | no wind | B3 − B2 | 5 | 0.0249 | 0.1234 | -0.0985 | [-0.1224, -0.0451] | no |

Description:

- B3 is better than B4 in the wind regime on both datasets (−0.014 m on v0.5, −0.013 m on v0.6, intervals exclude zero) and better than B2 by 0.051 m on v0.5 and 0.072 m on v0.6.
- B3 minus B6a (oracle wrench) is +0.004 m on v0.5 and +0.005 m on v0.6, with intervals that contain zero. With 20 episodes the experiment cannot separate B3 from the oracle-wrench physics model on these windows.
- The no-wind comparison uses only 5 episodes; its intervals are wide.
- B6a and B6b use the total recorded wrench, which includes the drag from the vehicle's own motion, so even without wind their error is not zero.

## 7. Re-observation with the real IMU, and the cost of the synthetic IMU φ

"Re-observe every n" feeds the real IMU into the latent state every n steps during the rollout. "Cost of φ vs real IMU" is RMSE(synthetic φ) − RMSE(real IMU) at the stated horizon: a positive value means using the real IMU is better.

| Set | Model | Open loop k=5 [m] | Open loop k=20 [m] | Re-observe every step, k=20 | every 5 | every 10 | Cost of φ vs real IMU, k=5 [m] | Cost, k=20 [m] |
|---|---|---|---|---|---|---|---|---|
| v05 | B2 pure neural | 0.0111 | 0.0826 | 0.0819 | 0.0797 | 0.0812 | -0.00027 | +0.0007 |
| v05 | **B3 physics+residual+memory** | 0.0019 | 0.0322 | 0.0282 | 0.0310 | 0.0318 | +0.00006 | +0.0040 |
| v05 | B5b learned dynamics | 0.0062 | 0.0811 | 0.1064 | 0.0837 | 0.0818 | -0.00095 | -0.0253 |
| v06 | B2 pure neural | 0.0155 | 0.1154 | 0.1057 | 0.1108 | 0.1136 | +0.00003 | +0.0098 |
| v06 | **B3 physics+residual+memory** | 0.0019 | 0.0320 | 0.0288 | 0.0311 | 0.0317 | +0.00004 | +0.0032 |
| v06 | B5b learned dynamics | 0.0067 | 0.0972 | 0.0972 | 0.0968 | 0.0968 | +0.00012 | +0.0001 |

Description:

- B3: re-observing every step reduces the k=20 error from 0.032 to 0.028–0.029 m (about 10 %); re-observing every 5 or 10 steps gives 0.031–0.032 m. The synthetic φ costs only 6e-5 m (v0.5) and 4e-5 m (v0.6) at k=5, and 0.004 m and 0.003 m at k=20: the φ that the model builds from its own predicted acceleration is nearly as good as the real IMU.
- B2 and B5b are barely helped by re-observation. For B5b on v0.5 re-observation every step makes the error worse (0.106 m versus 0.081 m open loop).

## 8. Does the residual do the work, and does the model use the actions?

**Residual diagnostics (B3).** Size of the learned residual relative to the physics terms at k=1, and the k=20 RMSE ratio when the residual is set to zero:

| Set | Regime | Windows | ‖d_acc‖ / specific force (k=1) | ‖d_ang‖ / ‖ω̇_phys‖ (k=1) | RMSE ratio with residual zeroed (k=20) |
|---|---|---|---|---|---|
| v05 | wind | 8461 | 0.042 | 0.311 | 1.48 |
| v05 | no wind | 2257 | 0.042 | 0.292 | 1.30 |
| v06 | wind | 8461 | 0.044 | 0.290 | 1.43 |
| v06 | no wind | 2257 | 0.045 | 0.253 | 1.31 |

- The translational residual is about 4 % of the specific force; the angular residual is about 25–31 % of the physics angular acceleration.
- Setting the residual to zero makes the k=20 error 1.3–1.5 times larger. The residual is small but is used.

**Shuffled-u test.** The commands of a window are replaced by those of a sibling episode; the ratio is RMSE(sibling commands) / RMSE(original commands). A ratio near 1 means the model's prediction does not depend on the commands any more than B1's (physics only) does:

| Set | Model | Windows | k=1 | k=5 | k=20 |
|---|---|---|---|---|---|
| v05 | B1 physics only | 18043 | 1.002 | 0.999 | 0.970 |
| v05 | B2 pure neural | 18043 | 1.009 | 1.010 | 1.015 |
| v05 | **B3 physics+residual+memory** | 18043 | 1.013 | 1.024 | 1.012 |
| v05 | B4 no memory | 18043 | 1.151 | 1.242 | 1.239 |
| v06 | B1 physics only | 18043 | 1.002 | 0.999 | 0.970 |
| v06 | B2 pure neural | 18043 | 1.004 | 1.005 | 1.008 |
| v06 | **B3 physics+residual+memory** | 18043 | 1.011 | 1.024 | 1.021 |
| v06 | B4 no memory | 18043 | 1.151 | 1.242 | 1.239 |

- B4 reacts strongly (1.15 at k=1, 1.24 at k=5 and k=20), B2 and B3 hardly at all (1.00–1.02), and B1 shows the reference level (about 1.0).
- The report notes that the test has low power, because sibling controllers' commands differ little in this data. It should not be read as proof that B3 ignores the commands.

## 9. Findings and open points

1. **Physics prior helps.** Physics-based models (B1, B3, B4) beat B2, B5a and B5b by about a factor 2 on in-distribution windows; B3 is the best non-oracle model there.
2. **The learned residual is the weakness on strong, unseen wind.** On OOD magnitude B3 is at B1's level (0.163 m versus 0.185 m) while B6a is at 0.018 m. This is the case the thesis question is about, so it should be addressed before a final claim (more seeds, a longer curriculum to horizon 50, an OOD-magnitude validation signal in model selection).
3. **B3's advantage over B4 comes with memory** (about 13 % on test ID, and 0.013 m in the wind regime), but B4 (B3 width) is on par on the OOD roles.
4. **B4, B4 (B3 width) and B5b get worse as they train** (best epoch 1), and because of the checkpoint-selection bug (§2b) the evaluated models are later epochs. Their numbers, and B2 and B5a on v0.6, are provisional until the runs are repeated with the fixed trainer. B3's advantage over B4 (34 % on validation as evaluated) shrinks to about 13 % against B4's real best epoch; it should be repeated with tuned B4 training and several seeds.
5. **The IMU correction has no measurable effect on open-loop accuracy**, within one-seed noise.
6. **Not available from this data:** a statement about OOD roles with 4 episodes, a ranking between models within about 10 % of each other, and any claim over seeds.

## 10. Files

- Tables: `outputs/reports/training_compare_v05_v06/tables/{runs,evaluation_v05,evaluation_v06,v06_vs_v05_change,analysis_regime,analysis_gap,analysis_anchor,analysis_residual,analysis_shuffle}.csv`
- Figures: `outputs/reports/training_compare_v05_v06/figures/`
- Raw evaluation: `outputs/eval/{v05,v06,v05_on_v06,v06_on_v05}.json`, `outputs/eval/analysis_*.json`
- Per-run outputs: `outputs/experiments/{v05,v06}_<model>/`
