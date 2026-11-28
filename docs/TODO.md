# TODO (single list of the remaining work), 2026-11-27

Status marks: **[approval]** needs the user's go-ahead (training or a decision), **[user]** an action only the user can do, **[auto]** can be done without approval.
Evidence and history: `HANDOFF.md`, `docs/problems_log.md` (B23, B24, C6–C14), `docs/results/`, `docs/checklist.md`.

## 1. Evaluation and results of the v7 stage-1 runs (7 runs, seed 0)
| # | Item | Mark |
|---|---|---|
| 1.1 | Evaluation of the 7 runs on all roles with B0, B1, B6a and the constant-velocity reference (`scripts/post_v7_stage1.sh`, running 2026-11-27); results in `outputs/reports/v7_stage1/SUMMARY.md` | auto (running) |
| 1.2 | Write `docs/results/v7_stage1.md` from it: OOD magnitude result (does the wide training wind help B3?), B3 vs small B3, B4 vs pilot, per-role tables | auto |
| 1.3 | Build the v7 training/evaluation report (figures, tables, resources) like `outputs/reports/training_compare_v05_v06` (`evaluation/report.py` is tied to the v05/v06 tags) | auto |
| 1.4 | Wind-estimator probe (`evaluation/probe.py`) on the v7 B3 / B3 small checkpoints, and a non-linear probe (problems log C14) | auto |
| 1.5 | Regime / residual / shuffled-u analyses (`evaluation.analysis`) on the v7 runs | auto |
| 1.6 | Compare with the pilot numbers only per horizon in seconds (pilot 0.4 s, v7 1 s at 20 steps); add evaluation at 8 steps (0.4 s) for a like-for-like comparison | auto |

## 2. Training (all need approval)
| # | Item | Why | Mark |
|---|---|---|---|
| 2.1 | Stage 2: seeds 1 and 2 for B3, B4 and B3 small on v7 | seed spread; B3 vs small B3 is within noise with one seed | approval |
| 2.2 | Learning-rate schedule: cosine tied to the real step count (or constant rate + reduce-on-plateau) and per-stage handling; the current cosine spans 60 epochs x first-stage steps, early-stopped runs stop at 40-60 % of the peak rate (audit F12) | B4, B2, B5 stop at high rate; B3 ends at 1.6e-5 | approval |
| 2.3 | Peak learning-rate test for B3 (3e-4, 1e-3, 3e-3) | rate never varied | approval |
| 2.4 | Longer epoch cap for B3 / B3 small (both hit the 60-epoch cap, best epoch 52-53) | still-falling training loss | approval |
| 2.5 | Stability of B2, B5a, B5b at horizon 20 (validation explodes late): lower rate / stronger clipping / drop from the headline set | optimisation issue | approval |
| 2.6 | Learning curve: B3 and B4 on 25 %, 50 %, 100 % of the training episodes, 3 seeds | answers "more data or bigger model" | approval |
| 2.7 | B3a, B3b, B3c (`resid_uses_u`, `lambda_zero`, `shared_trunk`) on v7 | only run on the pilot; no effect there | approval |
| 2.8 | Longer history (50 steps) for B3 (audit F3: the latent is reset every window and sees 0.5 s) | wind not readable from the latent | approval |
| 2.9 | OOD-aware checkpoint selection (use `val_ood_magnitude`) | C13: ID validation picks near-physics checkpoints | approval |
| 2.10 | Stronger weight decay (0.1 to 1) for B4 / B5b (the 1e-2 test was a no-op, C12) | C12 | approval |
| 2.11 | B6b (oracle wrench input) on v7 and horizon-50 curriculum (2.5 s at 20 Hz) | upper bound, long horizon | approval |

## 3. Data
| # | Item | Mark |
|---|---|---|
| 3.1 | **B24:** at 20 Hz the attitude-mode scenarios (`attitude_combined`, `pitch_doublet`, `roll_doublet`, 318 episodes) run outside the rotor-drag envelope about 77 % of the time and are excluded. Options: accept; controller at 100 Hz with 20 Hz data (needs separate controller and model steps in plan/runner/validator); retune the attitude gains; regenerate | approval |
| 3.2 | **B23 follow-up:** only 341 of the 908 early-ended episodes pass the applicability rule; decide whether training may bypass it for them (+5 % windows now) | approval |
| 3.3 | Quality report of v7 (`outputs/reports/physiworld_pilot_v7_quality`): 1 fail (`physical.force_balance`, p99 0.90 m/s2 against 0.5; probably the ground-contact impacts, not checked), 3 warnings, 835 soft validation findings: inspect, document, fix the validator limits for the 20 Hz step where they are per step | auto |
| 3.4 | Mode B (fixed-action replay): only 47 of 729 completed; if Mode B matters, regenerate the open-loop part with its own 2-4 s durations (the `duration_s: 10.0` override of `pilot_v7.yaml` caused B23); fix the config before any regeneration | approval |
| 3.5 | Vehicle ranges: v7 uses mass +-10 % (OOD 20 %), inertia +-5 %; SkyJEPA used +-50 % / +-30 %; decide before Run 6 | approval |
| 3.6 | Run 6 (the final scale-up): sizing, wind ranges (decided: ID <= 6.2, OOD (6.2, 9]), domains, Mode B in test roles, envelope re-check; not started | approval |
| 3.7 | Optional cleanup of old datasets in `data/datasets` (`physiworld_pilot_v0.4.0-dev`, `..._v0.5.0_backup143`, `smoke_v0.3.0`, `physiworld_pilot_widewind_v0.1.0-preview`, `stab20`, `stab50`) | user |

## 4. Code, tests and repository
| # | Item | Mark |
|---|---|---|
| 4.1 | Push the local commits (last push: `8b7861a`, `git log origin/handoff/run5-start..HEAD` lists the rest) | approval |
| 4.2 | Re-run the full CPU suite (last full run: 279 passed before the dashboards, ntfy and the training-settings changes) and the Isaac suite (`tests/isaac --isaac`, not run since the 20 Hz changes) | auto |
| 4.3 | Decide on the superseded launchers `scripts/start_generation.sh`, `generate_until_complete.sh`, the old gate runners `run_baseline_gates*.sh`, `usd_to_stl.py` | user |
| 4.4 | 20 Hz: other per-step thresholds in the validators (attitude rate, kinematic checks) assume 50 Hz steps; review | auto |
| 4.5 | `evaluation/report.py` and `scripts/summarize_sweep.py` are tied to run-name patterns; generalise | auto |
| 4.6 | Update `docs/guides/assumptions_limitations.md`, `docs/checklist.md`, `.idea/specs/e2e_done.md` with the v7 stage-1 results | auto |

## 5. Operations and user actions
| # | Item | Mark |
|---|---|---|
| 5.1 | Subscribe the phone to the ntfy topic in `logs/ntfy.conf` (shown by `scripts/setup_ntfy.sh`); keep it private | user |
| 5.2 | Open the firewall for the report pages if needed: `sudo ufw allow from 10.0.0.0/8 to any port 60193 proto tcp` and 60194 (see the visualization README); from another network use the SSH tunnel or Tailscale | user |
| 5.3 | Ask IT for a DHCP reservation for your PC (MAC 10:7b:44:7a:ad:97) if its address keeps changing | user |
| 5.4 | The server needs a restart ("System restart required"); the auto-resume (`scripts/resume_on_boot.sh`, container restart policy) is set up but has not been tested with a real reboot | user |
| 5.5 | Isaac Sim crash note: choose GPUs with `CUDA_VISIBLE_DEVICES`, not `physics.device=cuda:N` (problems log B22) | info |


## Added 2026-11-27: visualisations
- [x] `visualization trajectories`: actual vs predicted trajectories, wind speed/azimuth in legend (first run on stage-1 checkpoints: outputs/reports/trajectories_v7).
- [x] `visualization evaluation`: error-vs-horizon, model x role heatmaps, OOD gap from evaluation JSONs. Run on st_v07_c{0,1,2}.json when post_v7_stage1.sh finishes.
- [x] B1 altitude loss explained: vehicle parameter mismatch, see problems_log C15.
- [ ] (C15) Decide: parameter-oracle baseline (B6a/B1 with true mass, force constants) to get a real upper bound on v7; re-read B3 vs B4 results in that light.
- [ ] (C15) Known-vehicle runs: smoke-test the trainer with `known_vehicle`/`vehicle_input` on a real dataset slice, then stage-2 retraining (needs approval). Decided (user, 2026-11-27): f_phys does NOT include drag or any aero (unknown in practice); B1_kv is therefore not exact at zero wind (0.200 m at 1 s), B6a_kv (true wrench) is the upper bound (0.006 m).

## Update 2026-11-28
* Known-vehicle training (9 runs) and all-role evaluation: DONE (`docs/results/v7_known_vehicle.md`, figures `outputs/reports/evaluation_v7_kv/`). Next: seeds 1 and 2 for B2, B3, B4; tests of why the residual does not extrapolate on OOD wind magnitude (longer history, non-linear probe, OOD-aware checkpoint selection, stronger weight decay); reliable-horizon curves with intervals over groups.
* Midterm slides: `report/slides/PhysiWorldUAV_midterm.pptx` (60 slides, rebuilt with the final known-vehicle numbers).
