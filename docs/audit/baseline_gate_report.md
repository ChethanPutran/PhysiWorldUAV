# Baseline gate report — pilot `physiworld_pilot_v0.5.0` (2026-11-17)

Models B0 (hold last state), B1 (physics only), B2 (pure neural latent), B3 (physics + residual + latent, the proposed model), B4 (physics + residual, no latent), trained on the 63 `train` trajectories (141 eligible episodes in all) and evaluated per split role. B5 and B6 are not implemented, so gates G6 and G7 were **not run**. One seed, no IMU noise, given initial state, history 10, training horizon curriculum 1→5→10→20, early stopping (patience 8) on `val_id`. B2/B3 have 237,209 parameters, B4 237,960. Raw files: `outputs/results/` and `outputs/experiments/<run>/` (git-ignored); commands: `scripts/run_baseline_gates_parallel.sh`.

## Gate outcomes

| Gate | Result | Evidence |
|---|---|---|
| G1 tiny-overfit (8 episodes) | **Pass** for B2, B3, B4 | 20-step position error 1.8 / 0.5 / 0.7 mm (target < 2 cm); loss down 9,172× / 9,124× / 749× |
| G2 curves | **Pass** | training and validation losses fall; early stopping at epochs 29 (B2), 35 (B3), 33 (B4). Train→val gap is large: 20-step position error on train vs `val_id`: B2 0.011 → 0.065 m, B3 0.003 → 0.027, B4 0.005 → 0.050 (63 trajectories only) |
| G3 beats hold-last-state | **Pass** | `val_id` at 1, 5, 10, 20 steps all three trained models beat B0 |
| G4 physics prior helps | **Partly.** B3 < B1 in every role at 20 steps; B3 < B2 on `val_id` (0.027 vs 0.065 m, CIs [0.018, 0.032] vs [0.049, 0.074]) and `test_id`; **not on OOD magnitude, where B2 is better** (0.084 vs 0.145 m, CIs [0.061, 0.102] vs [0.112, 0.176]); other OOD roles are within noise |
| G5 latent memory (B3 vs B4) | **Mixed.** B3 better ID (0.027 vs 0.050 m); B4 better on OOD magnitude (0.115 vs 0.145, CIs overlap) |
| G6 residual-only (B5), G7 oracle (B6) | **Not run** (not implemented) |
| G8 per-role evaluation | **Done** (`pilot_gates.md`) |
| G9 counterfactual consistency, G12 action-shortcut | **Not run** |
| G10 label sanity | **Pass.** B1 on no-wind nominal episodes: 0.0092 m at 20 steps; on the mass +5 % siblings: 0.0438 m (4.8× larger, expected direction) |
| G11 state-estimation diagnostic | **Done.** Decoder start (IMU-only latent): 20-step position error 0.51 m (B2) / 0.52 m (B3) on `val_id` vs 0.065 / 0.027 m with the given state; attitude error ≈ 120° (yaw and absolute attitude are not observable from the IMU) |

## Position error at 20 steps (0.4 s), mean over windows [m]

| Role (episodes) | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| val_id (11) | 0.193 | 0.039 | 0.065 | **0.027** | 0.050 |
| test_id (14) | 0.255 | 0.041 | 0.069 | **0.026** | 0.045 |
| test_ood_magnitude (7) | 0.323 | 0.180 | **0.085** | 0.145 | 0.115 |
| test_ood_direction (4) | 0.271 | 0.057 | **0.038** | 0.043 | **0.035** |
| test_ood_temporal (4) | 0.275 | 0.089 | 0.019 | **0.018** | 0.030 |
| test_ood_gust (4) | 0.046 | 0.053 | **0.033** | 0.034 | 0.040 |
| test_ood_physics (7) | 0.373 | 0.185 | 0.073 | 0.074 | 0.075 |
| test_ood_trajectory (4) | 0.272 | 0.048 | 0.050 | **0.036** | 0.042 |

Roles with 4 episodes (2–4 independent groups) give wide intervals; differences among the learned models there are mostly within noise.

## Long horizon (100 steps = 2 s), models trained with horizon 20

| Role | metric | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| val_id | pos@50 [m] | 0.464 | 0.334 | 0.399 | **0.203** | 0.364 |
| val_id | pos@100 [m] | **0.896** | 2.713 | 1.791 | 1.436 | 2.407 |
| val_id | reliable steps (0.5 m) | 62.1 | 61.2 | 64.2 | **75.0** | 59.9 |
| test_ood_magnitude | pos@50 [m] | **0.829** | 1.607 | 0.590 | 1.058 | 1.072 |
| test_ood_magnitude | pos@100 [m] | **1.582** | 11.87 | 3.19 | 7.00 | 7.84 |
| test_ood_magnitude | attitude at 100 [deg] | **3.7** | 153.8 | 23.9 | 86.5 | 77.7 |
| test_ood_magnitude | reliable steps | **50.9** | 31.5 | 48.8 | 37.2 | 38.8 |

## What this means

1. **The data is learnable and the pipeline works.** Every trained model beats hold-last-state at 20 steps on every ID and OOD role except `test_ood_gust` (B0 0.046 m is close to the learned models there, since that role is nearly calm).
2. **The physics prior helps in distribution** (B3 about 2.4× better than B2 and 1.5× better than B1 at 20 steps) **but the pilot does not show it helping under OOD wind magnitude**: there the pure neural B2 is best at 20 steps and the physics-based rollouts (B1, B3, B4) tumble at longer horizons (attitude error 78–154° at 100 steps), worse than hold-last-state. This is the thesis's central claim, so it is the result to watch with more data; with 7 OOD-magnitude episodes (3 groups) and one seed it is indicative, not conclusive.
3. **No model is useful at 2 s yet** (all worse than hold-last-state at 100 steps on `val_id`); training used horizon 20 only, so the curriculum to 50 and longer-horizon training are untested.
4. **Thesis-exact start state (decoder from IMU only) does not work** (G11): evaluate with the given initial state; this supports decision D1 option (b)/(c).
5. **The train→val gap is 5–11×** with 63 trajectories: volume and diversity of ID data matter. Together with the data-quality findings (calm-weighted wind, few OOD groups, IMU deviation) this feeds the validation-gate decision about what to change in the generator before the final run.

## Caveats
One seed; single run per model; pilot-scale data; the IMU noise transform was not used; B0–B4 only; the IMU deviation from ground-truth acceleration (run5 addendum) is unresolved and affects the encoder input; results are indicative of the pipeline, not of the final thesis numbers.
