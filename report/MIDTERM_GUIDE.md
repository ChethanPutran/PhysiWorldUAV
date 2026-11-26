# Midterm evaluation: what is asked, and how this project answers it

## 1. The instruction, explained

| Instruction | What it means for you |
|---|---|
| Panel assignments shared before Sat Nov 28 | Check mail on Nov 28. Nothing to submit yet. |
| Presentations on **Dec 3 and 4** | One slot each. The panel is told your date and room. |
| Restrict the talk to **work done on the project** | Show what you did and measured. Do not spend time on general background or on plans that have no work behind them. |
| Introduce the **problem statement and the gaps** | Slides 2–4 below: the problem, why it matters, what the literature leaves open. |
| Share **progress, achievements, challenges** | Slides 5–15: the pipeline, the validation, the baselines and results, and the problems met and fixed. |
| **A slide on future work** with the remaining tasks needed to reach the goals | Slide 17: a concrete list tied to the research questions. |
| **25 minutes**, with 10 minutes for Q&A and changeover | About 20 slides at roughly 1 minute each (some 30 s, some 2 min). Practise with a timer; aim for 23 minutes so you do not overrun. |
| Submit the **progress report to the examiner panel by Mon Nov 30, "without fail"** | `report/midterm_report.pdf`. Send it on Fri Nov 27 or Sat Nov 28 as a safety margin. Confirm the recipients (panel members, possibly your guide) once panels are announced. |

Note: the report is not a transcript of the slides. It is the formal record, with the same structure as the template in `report/` (abstract, introduction, related work, methodology and work done, experiments and results, future work, references).

## 2. Messages you want the panel to leave with

1. The problem is well posed: recursive prediction fails under OOD wind, and physics is known for quadrotors.
2. A lot of the work so far has been making the **evidence trustworthy**: a validated simulator, a dataset with quality checks, and baselines evaluated honestly. A first pipeline that did not work was found and replaced.
3. Results are **mixed, and you say so**: physics helps in distribution, not yet on strong OOD wind; a constant-velocity baseline beats the learned models there. You know why this matters and have a diagnosis and plan.
4. The next stage is clear and already running: a much larger, wind-diverse dataset (pilot v7).

Do not claim H1 is confirmed. Do not hide the constant-velocity result: the panel will ask about it, and having found it yourself is a strength.

## 3. Slide plan (25 min)

| # | Slide | Time | Content and source |
|---|---|---|---|
| 1 | Title | 0:20 | Title, name, guide, department |
| 2 | Problem | 1:30 | Compounding error in recursive prediction; OOD wind; one picture of rollout drift |
| 3 | Why UAV and why physics | 1:00 | Known rigid-body dynamics, unknown aero and wind; `ds/dt = f_phys + d_θ` |
| 4 | Gap and research questions | 1:30 | Pure latent vs physics-prior latent, controlled comparison; RQ1–RQ3; H1, H2 |
| 5 | Approach overview | 1:00 | Pipeline: simulate → data → models → evaluate (the box diagram from this guide §4) |
| 6 | How the project evolved | 1:30 | Timeline from `MASTER_README.md` §2; audit found the first pipeline unusable |
| 7 | Simulation platform and data design | 1:30 | Isaac Sim + OmniDrones, 200 Hz physics, counterfactual groups, ID/OOD roles |
| 8 | Physics and wind models | 1:30 | Rotor drag, body drag, wind regimes, envelopes; assumptions |
| 9 | Physics validation | 1:00 | Table of measured vs expected (hover, free fall, 1 N, wind, motor lag) |
| 10 | Pilot dataset and quality checks | 1:30 | 147 episodes, 0 hard findings; the checks; findings (IMU artefact, coverage) |
| 11 | Models and baselines | 1:30 | B0–B6 table; B2 vs B3 is the central comparison |
| 12 | Training and evaluation protocol | 1:00 | Context + rollout, curriculum, per-role evaluation, 20-step = 0.4 s |
| 13 | Results: in distribution | 1:30 | B3 0.036 m vs B2 0.105 m vs B1 0.053 m |
| 14 | Results: OOD wind | 2:00 | Table row for OOD magnitude; constant velocity 0.050 m; oracle 0.018 m; `report/figures/eval_by_role.png` |
| 15 | Diagnostics | 1:30 | Variants within noise; latent probe finds no wind; checkpoint-selection issue |
| 16 | Challenges and how they were handled | 2:00 | Pick 5: IMU artefact, `best.pt` bug, wind no-op in old pipeline, reproducibility, thin OOD roles |
| 17 | Next stage of data: pilot v7 | 1:00 | 4,530 episodes, 20 Hz, wide wind (`report/figures/v7_wind.png`), 29.6 % generated at 12:59 on Nov 26 (update the figure on the day) |
| 18 | Future work | 1:30 | See §5 |
| 19 | Timeline to the thesis | 0:45 | Phases from `docs/thesis.tex`, with what is done |
| 20 | Summary | 0:45 | Three bullets and one honest limitation |
| | Total | about 24:00 | Q&A follows |

Update slide 17 with the progress on the day. Check `HANDOFF.md` first.

## 4. One-picture overview (for slide 5)

```
config (Hydra) -> plan (manifest) -> Isaac Sim + OmniDrones -> HDF5 shards -> validate
        -> model-ready windows (20/50 Hz) -> B0..B6 training (context + rollout)
        -> per-role evaluation (ID, OOD magnitude/direction/temporal/gust/physics/trajectory)
```

## 5. Future-work slide: concrete list

1. Finish and validate pilot v7 (4,530 episodes, 906k model steps).
2. Retrain B1–B6 on v7 with at least 3 seeds and a horizon curriculum to 50 steps.
3. Test why B3's residual does not extrapolate: wind-range augmentation, OOD-aware checkpoint selection, longer history, stronger weight decay, non-linear probe, disturbance-supervision head.
4. Primary experiment: B2 vs B3 error growth and reliable imagination horizon under OOD wind magnitude, with intervals over groups (RQ1, RQ2).
5. Ablations: physics mismatch (RQ3), training-data size, with and without memory.
6. Optional: uncertainty head (RQ4).
7. Thesis writing and updating the proposal text for design changes.

## 6. Likely questions and honest answers

* **Is your hypothesis supported?** Partly. In distribution and for temporal/gust OOD, yes (B3 beats B2 and B1). On strong OOD wind magnitude, not yet: B3 is about at physics-only level. This is pilot scale, one dataset, 3 seeds for B3, 4–7 episodes per OOD role.
* **Why does constant velocity win on OOD magnitude?** The cause is not established. Known facts: B1 lacks the wind force (0.185 m), and B3's residual does not extrapolate beyond the training wind (training peak 3.7 m/s, OOD peaks 4.3 to 8.2 m/s). Say that the reference exists so that wins are not overstated, and that the long-horizon comparison is a next-stage item.
* **Is the simulator realistic?** It is a validated physics-based approximation, not CFD. Each approximation is bounded by an applicability envelope (rotor-drag μ ≤ 0.20, with reduced fidelity between 0.15 and 0.20), and episodes outside it are excluded. See `docs/guides/assumptions_limitations.md`.
* **Why a simulator and not real data?** Ground-truth state, wind and counterfactual pairs are needed to separate wind effects from controller reactions; real flights cannot provide that. Real-flight transfer is out of scope.
* **Leakage through the controller?** Closed-loop `u` correlates with wind. Mitigations: counterfactual Mode B replay, excitation, open-loop segments, and a permutation test. The open-loop correlation was 0.39 (p = 0.085, borderline), and Mode B exists only in train and val roles so far; the v7 plan adds it to test roles.
* **Why start from the true state?** An IMU-only decoder start gave about 120° attitude error (yaw is unobservable), so evaluation uses the given initial state, as decided.
* **Why 20 steps only?** The pilot curriculum stopped at 20 steps (0.4 s). v7 uses 50 ms steps, so 20 steps are 1 s, and the curriculum to 50 steps is planned.
* **What about the bugs you found?** Say so directly: the `best.pt` selection bug affected 8 of 16 runs, was found by audit, fixed, and the runs repeated. The first-generation pipeline was replaced after audit.
* **Novelty?** The components are not new. The contribution is the controlled comparison, the OOD wind benchmark and the reliable-horizon analysis.

## 7. Practical checklist

* [ ] Panel assignment (Nov 28): note date and room.
* [ ] Build and read `report/midterm_report.pdf` end to end; fix anything you disagree with (it was drafted from the repo docs).
* [ ] Send the report to the panel by Nov 30 (earlier is safer).
* [ ] Build the slides from §3 (figures in `report/figures/`; more in `outputs/reports/`).
* [ ] Refresh slide 17 with the generation progress; check `HANDOFF.md`.
* [ ] Rehearse twice with a timer.
