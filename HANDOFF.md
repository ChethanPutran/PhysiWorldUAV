# HANDOFF

Saved 2026-11-24 (container clock UTC; host clock IST = UTC + 5:30) · Branch `handoff/run5-start` · **Nothing is committed**: all work of the last days is in the working tree (the user also moved or deleted several `docs/` files; do not restore them; the specs now live in `.idea/specs/`).
Read first: `CLAUDE.md` (architecture, commands, gotchas), `docs/problems_log.md` (everything that went wrong and how it was resolved), `docs/checklist.md` and `.idea/specs/e2e_done.md` (status with evidence), `docs/guides/glossary.md` (terms: group, domain, regime, envelope...).

## Sweep `fix_v1`: FINISHED and evaluated (2026-11-25)
* 29 runs, all done (`outputs/sweeps/fix_v1/status.json`), evaluated by `scripts/post_sweep.sh`; results `docs/results/sweep_fix_v1.md`, tables/figure `outputs/reports/sweep_fix_v1/`. The first report folder `outputs/reports/training_compare_v05_v06/` has a `README_STALE.md` (old best.pt).
* Findings: checkpoint fix confirmed; B3a/b/c no help on OOD wind (B3a worse in ID and on OOD physics); a constant-velocity baseline (`scripts/constant_velocity_baseline.py`) beats B3, B4, B1 on OOD magnitude (0.050 vs 0.166/0.183/0.185 m) and direction; B5b = constant velocity at epoch 1; B4 lr 2e-4 no help; weight-decay runs were too weak (C12). Problems log C9, C11-C13.
* Live page and its writer (`scripts/live_monitor.py --html`) and the web server on the port in `logs/http_live.port` may still be running; stop them if not needed (kill by PID).
* Wind probe done on 8 trained models (C14): latent h carries no readable wind force (R² ≤ 0 everywhere; torque control 0.8). `scripts/run_probes.sh`, `outputs/eval/probe_*.json`.
* Wind plan for Run 6 DECIDED (2026-11-25): wide ranges + stratified, upper-band-weighted mean speed (`config/wind/regimes/id_wide.yaml`, `ood_magnitude_wide.yaml`; planner keys `speed_strata`, `speed_strata_weights`; preview `config/experiment/pilot_wide_wind.yaml`, manifest-only dir `data/datasets/physiworld_pilot_widewind_v0.1.0-preview` (can be deleted)). Run 6 itself is NOT started.
* Decisions waiting: which next experiments (longer history; non-linear probe; wind-range augmentation; stronger weight decay; OOD-aware checkpoint selection), Run 6 wind range and sizing.

## Training and report: FINISHED (2026-11-24)
* All 16 runs, the evaluation (own and cross-dataset), the analyses and the report are done and `chown`ed. Report: `outputs/reports/training_compare_v05_v06/report.html` (+ `REPORT.md`, `figures/`, `tables/`). Written results with tables: `docs/results/baseline_comparison_pilot_v05_v06.md`.
* Main findings (one seed, pilot size): B3 is the best non-oracle model in distribution; **B3 is not better than physics-only on OOD wind magnitude** (0.163 vs B1 0.185 m, oracle B6a 0.018 m); B4, B4 (B3 width) and B5b get worse as they train (C6, cause not established); **audit F1 (C9): `best.pt` was not the best checkpoint for 8 of the 16 runs (B4, B4 (B3 width), B5b both versions; B2, B5a on v0.6), so their numbers are provisional; trainer fixed 2026-11-25, runs not yet repeated**; the IMU correction (v0.5 vs v0.6) shows no effect beyond seed noise.
* Still to do after reading: the probe (`evaluation/probe.py`) on trained models; nothing else was left running.

## Done (see the docs for evidence)
* Pilot **v0.6.0** generated (147/147, validated; identical plan to v0.5.0; IMU transport artefact fixed at generation; episode-order dependence fixed). Pilot v0.5.0 got checksums. Quality report 0 fail / 3 warn.
* Models B0-B6 implemented and tested (B5a/b, B6a/b with future/current oracle); evaluation analyses; probe; YAML configs; parallel generation (`scripts/generate_parallel.sh`, ~26 min for the pilot on 3 workers); per-episode checksums; loader benchmark.
* Diversity additions verified in the simulator (perturbed starts, long episodes, Dryden turbulence, composite wind switching, air density, controller gain randomisation) with `config/experiment/diversity_check.yaml`; Option C wind range verified (`config/experiment/wind_envelope_check.yaml`).
* 250 CPU tests pass (`OMP_NUM_THREADS=4`, ~5 min).

## Decisions waiting for the user
1. Interpret the baseline comparison (`docs/results/...`); decide: longer curriculum (horizon 50), more seeds for B2/B3/B4/B5b, and repeating the 8 affected runs with the fixed trainer, and whether to investigate the B4/B5b overfitting (lower learning rate or weight decay) and B3's OOD-magnitude failure - none started, each needs approval.
2. Wind range for Run 6: keep OOD (6, 9] m/s (24/26 inside the force-model envelope, 14/14 up to 7.5 m/s) or narrow to (6, 8] (recommended).
3. Which diversity additions enter the Run 6 plan; future work lists are in `docs/checklist.md` section 5 (ground effect, rotor faults, sensor realism, second airframe, time-windowed wind regimes).
4. Run 6 sizing: ~17 groups per OOD part, >= 100 domains, Mode B episodes in the test roles. **Run 6 has not been started.**

## Known and accepted
* The Isaac IMU accelerometer is not bit-reproducible between regenerations (~2e-5 m/s2 typical); everything else is bitwise reproducible.
* Open-loop action-wind correlation 0.39 (permutation p = 0.085); domain-factor correlation 0.59; few groups in OOD roles; Mode B only in train and val_id.

## Working rules learned (details in the memory notes)
* Ask the user before training baselines or any large job; do not silently resolve scientific ambiguities; report measured numbers and failures plainly.
* Run everything in the `isaac-sim` container; CPU tests with `OMP_NUM_THREADS=4` (shared 96-core host, load ~85); never `pkill -f` a pattern that appears in your own command line (kill by PID); `docker exec` timeouts leave the process running inside the container.
