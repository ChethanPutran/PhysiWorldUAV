# HANDOFF

Date: 2026-11-17 · Branch: `handoff/run5-start` · See `CLAUDE.md` for architecture, commands and conventions.

## Current goal
**Run 5**: regenerate the pilot dataset (`config/experiment/pilot.yaml`, `physiworld_pilot` v0.5.0, 147 episodes) with the approved physics, then inspect its quality (coverage, validity, identifiability, wind/ID-OOD statistics, force-envelope violations, counterfactual pairs). Do **not** start Run 6 (scale-up) until the user asks.

## Status checklist
- [x] Run 1 audit, Run 2 plan (D1–D10 = recommended options)
- [x] Run 3 config + experiment architecture (CPU tests)
- [x] Run 4 physical models validated in Isaac (17 Isaac + 131 CPU tests at the time; IMU option (a))
- [x] Q3 drag coefficients approved by the user (2026-11-17): d = 0.30 s⁻¹, body C_D per axis (1.1, 1.1, 1.3), areas (0.0070, 0.0076, 0.0180) m², R = 0.10 m, ρ = 1.225 → `truth_default` now `validated: true`
- [x] Rotor-drag envelope widened to μ ≤ 0.20 (OOD wind stays (4, 8] m/s)
- [x] Per-axis C_D implemented (`forces/models.py`, `forces/system.py`) + test; CPU suite **132 passed**
- [x] Isaac suite re-run after the force changes: **17 passed** (248 s)
- [ ] Update `docs/models/validation_run4.json` / aerodynamics numbers if the Isaac run shows any change (body drag Isaac check used wind 3 m/s; z-axis C_D changes only vertical-flow drag)
- [ ] Generate pilot: `generate plan experiment=pilot` then `generate run experiment=pilot`, then `generate validate --dataset data/datasets/physiworld_pilot_v0.5.0` (validator must report no hard findings; the old dev dataset `physiworld_pilot_v0.4.0-dev` was unvalidatable by design)
- [ ] Quality inspection and `docs/audit/run5_report.md` (include: fraction of OOD-magnitude episodes inside the widened envelope, tilt/saturation stats, tracking error, throughput ≈ 61 steps/s/env)
- [ ] Update `docs/guides/assumptions_limitations.md`, memory-style notes, and ask the user before Run 6

## Files changed in this uncommitted batch, and why
* `config/forces/truth_default.yaml` — validated, per-axis `cd`, μ ≤ 0.20
* `src/physiworld_uav/forces/{models,system,registry}.py` — per-axis C_D (scalar still accepted), envelope constant/text
* `tests/cpu/test_forces.py` — per-axis test
* `docs/models/aerodynamics.md`, `docs/guides/assumptions_limitations.md` — coefficients/envelope
* Whole Run 3–4 rewrite is also uncommitted (never committed before): new `config/*` groups, `src/physiworld_uav/{core,drones,control,trajectories,excitation,wind,randomization,forces,experiments}`, `data/*` modules, `tests/`, `docs/{audit,guides,models,problems}`; legacy files deleted from the tree (copy lives in ignored `lagacy/`)
* `.gitignore` — added secrets patterns, anchored `/data/`, `/outputs/`
* `README.md`, `scripts/run.sh` — internal server IP/username replaced by placeholders; `pyproject.toml` — author email removed
* `CLAUDE.md`, `HANDOFF.md` — new

## Open issues / next steps (priority order)
1. Generate + validate the pilot; if the validator reports hard findings, fix the cause, don't relax checks.
2. Inspect: split/role balance, hover ≤ 20 % mix, excitation identifiability (FIM, decorrelation), ID/OOD realised statistics, `applicability` violation fractions (OOD 8 m/s episodes are now expected to pass at μ ≤ 0.20 — verify).
3. Known, tracked, not blocking: `training/train.py` imports deleted `physiworld_uav.physics` (out of scope); small (~0.3 %) articulation roll–pitch coupling left to the learned residual (Q4); per-axis C_D z-value 1.3 is an approved assumption, not a measurement.
4. Isaac Sim/OmniDrones cannot be pip-installed — the cloud session can edit code and run only what imports without Isaac. Dataset generation and Isaac tests need the GPU host's `isaac-sim` container.

## Environment (names only)
* `ISAACSIM_PATH` (in local ignored `.env`; container path `/isaac-sim`)
* `PYTHONPATH` (must include `$PWD/src`, appended)
* No API keys or tokens are required by the code.
