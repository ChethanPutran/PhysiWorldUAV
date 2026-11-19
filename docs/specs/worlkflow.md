## **PhysiWorldUAV development lifecycle**

Status as of 2026-11-17. Current position: **Phase 2 (baseline development)**, with a pilot dataset already generated (Phase 3 done, ahead of the baseline). The Phase 4 gate has not been run. Full-scale generation (Phase 5 = Run 6) must not start until the gate passes.

| Phase | Content | Status | Specification / evidence |
|---|---|---|---|
| 1 — Problem specification | Input/output • distributions • labels • metrics | **Drafted, awaiting your approval** of the PROPOSED items | `data_specification.md`, `model_specification.md` §5, `training_pipeline.md` §6 |
| 2 — Baseline development | Model • dataloader • loss • evaluation pipeline | **Not started** (dataloader done; models, losses, trainer, evaluator to write) | `model_specification.md`, `training_pipeline.md` |
| 3 — Pilot generation | Small, representative dataset | **Done** (`physiworld_pilot_v0.5.0`, 147 episodes, validated) | `docs/audit/run5_report.md` |
| 4 — Validation gate | Quality • learnability • coverage • reproducibility • performance | **Open.** Quality partly done; learnability, coverage, reproducibility and performance checks pending | `model_specification.md` §5 (G1–G8), `data_generation_pipeline.md` §2–7, `checklist.md` |
| — Failure | Debug, fix, regenerate pilot (v0.6.0…) | – | – |
| — Pass | Freeze configuration | – | – |
| 5 — Full-scale generation | Sharded generation • monitoring • quality checks | Infrastructure ready (resume, live status, notifier); parallel scale-up not built | `info.md`, Run 6 |
| 6 — Final dataset validation | Manifest • statistics • integrity • versioning | Validator and index exist | `data_generation_pipeline.md` |
| 7 — Full model training | Experiments • ablations • evaluation • checkpoints | Not started | `training_pipeline.md` §8 |

### Order of work for the gate

1. You approve (or change) the PROPOSED choices in `model_specification.md` §9 and `training_pipeline.md` §11.
2. Implement `f_phys`, the encoder/latent/decoder, B1 and B2, the loss, trainer with checkpoint/resume, evaluator, and their unit tests.
3. Run G1 (tiny overfit) and G2–G3 on the pilot; then B3–B6 (G4–G7), per-role evaluation (G8), counterfactual, label and state-estimation checks (G9–G11).
4. In parallel, close the data-side gaps in `data_generation_pipeline.md` §9 (coverage/correlation report, noise transform, v↔dp/dt check, cross-process check).
5. Write the gate report in `docs/audit/`. Pass → freeze the config and start Run 6. Fail → change the generator, regenerate the pilot as the next version, repeat.

---

## **A few advanced considerations**

### **A. Data generation and model development should be iterative**

Generate an initial dataset → train the baseline → find scenarios where it performs poorly → decide whether the cause is insufficient data, wrong labels or model limits → generate targeted data → retrain. (*Active / adaptive data generation.*)

### **B. Don't freeze the architecture too early**

Keep the data specification stable; let the architecture evolve (longer dependencies, unneeded inputs, horizon too ambitious, different representation, multiple heads).

### **C. Maintain dataset versions**

Never silently overwrite a dataset.

| Version | Purpose |
|---|---|
| v0.3.0 | Run 3 smoke dataset (stale; superseded) |
| v0.5.0 | Pilot (Run 5): per-axis drag, μ ≤ 0.20 envelope |
| v0.6.0+ | Pilot revisions if the gate sends us back |
| v1.0 | Validated research dataset (Run 6) |
| v1.1 | Additional difficult scenarios |
