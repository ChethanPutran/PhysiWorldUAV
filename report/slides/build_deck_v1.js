// Midterm deck builder. Run: node build_deck.js  (needs pptxgenjs; theme applied afterwards)
const pptxgen = require("pptxgenjs");
const path = require("path");
const SK = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SK, "scripts/apply_theme.js"));
const FIG = path.join(__dirname, "fig") + "/";

const THEME = {
  name: "PhysiWorld", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "16202B", lt1: "FFFFFF", dk2: "0B1F33", lt2: "F1F4F7", accent1: "1F6F8B", accent2: "E0A030",
            accent3: "B0413E", accent4: "2E8B57", accent5: "7B4EA3", accent6: "5B6B7A", hlink: "1F6F8B", folHlink: "5B6B7A" },
};
const NAVY = "0B1F33", TEAL = "1F6F8B", AMBER = "E0A030", RED = "B0413E", GREEN = "2E8B57", PURPLE = "7B4EA3",
      LIGHT = "F1F4F7", INK = "16202B", MUTED = "5B6B7A", WHITE = "FFFFFF", PALE = "DCE8EE";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
pres.title = "PhysiWorldUAV: midterm evaluation";
pres.author = "Chethan";

pres.defineSlideMaster({
  title: "CONTENT", background: { color: WHITE },
  objects: [
    { text: { text: "PhysiWorldUAV  ·  Midterm evaluation", options: { x: 0.6, y: 7.08, w: 6, h: 0.3, fontSize: 10, color: MUTED, fontFace: "Calibri", margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.08, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, fontFace: "Calibri", align: "right" },
  placeholders: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 28, bold: true, color: NAVY, fontFace: "Cambria", valign: "middle", margin: 0 }, text: "" } }],
});
pres.defineSlideMaster({ title: "DARK", background: { color: NAVY } });

let sectionNow = null;
function section(t) { pres.addSection({ title: t }); sectionNow = t; }
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sectionNow });
  s.addText(title, { placeholder: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 28, bold: true, color: NAVY, fontFace: "Cambria", valign: "middle", margin: 0 });
  if (notes) s.addNotes(notes);
  return s;
}
const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: "Calibri", color: INK, margin: 0, valign: "top" }, o));
function bullets(s, items, x, y, w, h, size) {
  const runs = items.map((it, i) => {
    const o = { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: 8 };
    if (typeof it === "string") return { text: it, options: o };
    return { text: it.t, options: Object.assign(o, it.o || {}) };
  });
  T(s, runs, { x, y, w, h, fontSize: size || 16 });
}
function card(s, x, y, w, h, head, body, color, size) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.22, h: 0.22, fill: { color: color || TEAL }, line: { color: color || TEAL } });
  T(s, head, { x: x + 0.55, y: y + 0.14, w: w - 0.75, h: 0.36, fontSize: 16, bold: true, color: NAVY, valign: "middle" });
  T(s, body, { x: x + 0.25, y: y + 0.62, w: w - 0.5, h: h - 0.75, fontSize: size || 14 });
}
function stat(s, x, y, w, num, label, color) {
  T(s, num, { x, y, w, h: 0.75, fontSize: 32, bold: true, color: color || TEAL, fontFace: "Cambria", valign: "middle" });
  T(s, label, { x, y: y + 0.78, w, h: 0.55, fontSize: 13, color: MUTED });
}
function img(s, file, x, y, w, h) { s.addImage({ path: FIG + file, x, y, w, h, sizing: { type: "contain", w, h } }); }
function caption(s, text, x, y, w) { T(s, text, { x, y, w, h: 0.5, fontSize: 11, italic: true, color: MUTED }); }
function table(s, rows, x, y, w, colW, opt) {
  opt = opt || {};
  const fs = opt.fs || 12, rh = opt.rh || 0.32;
  const data = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" && c !== null && c.text !== undefined ? c : { text: String(c) };
    const base = { fontSize: fs, fontFace: "Calibri", color: INK, valign: "middle", align: j === 0 ? "left" : (opt.alignLeft ? "left" : "center"), margin: [0.02, 0.08, 0.02, 0.08] };
    if (i === 0) Object.assign(base, { bold: true, color: WHITE, fill: { color: NAVY } });
    else Object.assign(base, { fill: { color: i % 2 ? WHITE : LIGHT } });
    cell.options = Object.assign(base, cell.options || {});
    return cell;
  }));
  s.addTable(data, { x, y, w, colW, rowH: rh, border: { type: "solid", color: "D5DDE3", pt: 0.5 } });
}
const best = (t) => ({ text: t, options: { bold: true, color: GREEN } });
const bad = (t) => ({ text: t, options: { bold: true, color: RED } });

// ---------------------------------------------------------------- 1 Title
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addNotes("0:20. Introduce yourself, guide, department. One sentence on what the talk covers: what was built, what was measured, what worked and what did not.");
  T(s, "M.Tech thesis  ·  Midterm evaluation", { x: 0.9, y: 0.9, w: 10, h: 0.4, fontSize: 16, color: AMBER, bold: true });
  T(s, "Physics-Prior Latent World Models for Reliable Long-Horizon UAV Prediction under Out-of-Distribution Wind", { x: 0.9, y: 1.6, w: 9.4, h: 2.6, fontSize: 38, bold: true, color: WHITE, fontFace: "Cambria" });
  T(s, "Chethan  ·  CDS, Indian Institute of Science, Bangalore", { x: 0.9, y: 5.0, w: 10.5, h: 0.4, fontSize: 15, color: "CADCFC" });
  T(s, "Guide: [name]  ·  Panel date: Dec 3–4, 2026", { x: 0.9, y: 5.6, w: 10, h: 0.4, fontSize: 14, color: "9FB3C8" });
  // motif: three nested rings
  [[10.0, 1.3, 2.8, "1F6F8B"], [10.5, 1.8, 1.8, "E0A030"], [11.0, 2.3, 0.8, "FFFFFF"]].forEach(([x, y, d, c]) =>
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: NAVY }, line: { color: c, width: 3 } }));
}

// ---------------------------------------------------------------- Section A
section("Problem and approach");
{
  const s = content("Recursive prediction drifts, and wind makes it worse",
    "1:30 + 1:00. Compounding error: each predicted step is fed back. The quadrotor is a good test case because rigid-body dynamics are known (thrust, gravity, kinematics) while drag, wind and actuator effects are not. The model is ds/dt = f_phys + d_theta.");
  card(s, 0.6, 1.55, 4.0, 2.55, "The problem", "A world model imagines the future by feeding its own outputs back in. One-step errors compound, and the error grows when conditions differ from training, for example stronger wind than ever seen.", RED, 14);
  card(s, 0.6, 4.3, 4.0, 2.55, "Why a UAV", "A quadrotor is a rigid body: thrust, gravity and kinematics are known and stay valid in any wind. Aerodynamic drag, wind and actuator lag are unknown.", TEAL, 14);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.9, y: 1.55, w: 7.8, h: 2.55, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "The model", { x: 5.2, y: 1.7, w: 3, h: 0.35, fontSize: 14, color: AMBER, bold: true });
  T(s, "ds/dt = f_phys(s, u) + d_θ(h, s, u)", { x: 5.2, y: 2.15, w: 7.3, h: 0.7, fontSize: 26, bold: true, color: WHITE, fontFace: "Cambria", valign: "middle" });
  T(s, [{ text: "f_phys: known rigid-body dynamics, RK4", options: { breakLine: true } }, { text: "d_θ: learned residual on force and torque", options: { breakLine: true } }, { text: "h: recurrent latent memory from IMU history" }],
    { x: 5.2, y: 3.0, w: 7.3, h: 1.0, fontSize: 14, color: "CADCFC" });
  card(s, 4.9, 4.3, 7.8, 2.55, "What would show the idea works", "Physics-prior latent model (B3) loses accuracy more slowly than a pure learned latent model (B2) as the rollout gets longer and the wind gets stronger than anything in training. Measured per OOD axis: wind magnitude, direction, temporal structure, gusts, vehicle physics, trajectory.", AMBER, 14);
}
{
  const s = content("Gap: a controlled comparison of pure latent against physics-prior latent",
    "1:30. Each ingredient exists in the literature (PINNs for quadrotors, NeuroBEM/Neural-Fly residuals, Dreamer latent models). The contribution is the controlled comparison and the OOD wind benchmark, not a new component. State RQ1-RQ3 and H1, H2. A negative or conditional answer is acceptable if the conditions of failure are identified.");
  T(s, "What the literature leaves open", { x: 0.6, y: 1.5, w: 6, h: 0.4, fontSize: 20, bold: true, color: NAVY, fontFace: "Cambria" });
  bullets(s, ["Physics-informed networks put physics in the loss, not in the transition used for imagination (Gu et al. 2024)",
    "Hybrid residual models (NeuroBEM, Neural-Fly) are evaluated one step or on tracking, not as long imagined rollouts",
    "Dreamer-style latents learn the whole state; here the latent is memory attached to an explicit physical state",
    "Little controlled evidence that compares both under recursive prediction with controlled OOD wind"], 0.6, 2.0, 6.1, 4.6, 15);
  const rq = [["RQ1", "Does the physics prior make long-horizon prediction under OOD wind more reliable?", TEAL],
    ["RQ2", "Under which OOD conditions is the benefit largest?", AMBER], ["RQ3", "How sensitive is it to nominal-model error and data size?", PURPLE]];
  rq.forEach((r, i) => {
    const y = 1.55 + i * 1.2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.1, y, w: 5.6, h: 1.05, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: 7.3, y: y + 0.25, w: 0.55, h: 0.55, fill: { color: r[2] }, line: { color: r[2] } });
    T(s, r[0].replace("RQ", ""), { x: 7.3, y: y + 0.25, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, [{ text: r[0] + "  ", options: { bold: true, color: NAVY } }, { text: r[1] }], { x: 8.05, y: y + 0.1, w: 4.5, h: 0.85, fontSize: 14, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.1, y: 5.2, w: 5.6, h: 1.5, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, [{ text: "H1  ", options: { bold: true, color: AMBER } }, { text: "B3's error grows more slowly than B2's.", options: { breakLine: true, color: WHITE } },
    { text: "H2  ", options: { bold: true, color: AMBER } }, { text: "The benefit shrinks as the nominal physics gets worse.", options: { color: WHITE } }], { x: 7.35, y: 5.35, w: 5.2, h: 1.2, fontSize: 15, valign: "middle", paraSpaceAfter: 6 });
}
{
  const s = content("Approach: simulate, validate, build the data, then compare models",
    "1:00. Walk left to right through the pipeline. Stress that most of the effort so far went into the first four boxes so that the last two can be trusted.");
  const steps = [["Config + plan", "Hydra configs, seed tree, manifest: one row = one episode", TEAL], ["Isaac Sim + OmniDrones", "200 Hz physics, Hummingbird, project aero via one wrench injector", TEAL],
    ["HDF5 shards", "Schema v2, checksums, crash-safe resume", AMBER], ["Validate", "Physics checks and statistical quality report", AMBER],
    ["Model-ready windows", "20 / 50 Hz, history + horizon, IMU noise at load time", PURPLE], ["Train B0 to B6", "Context + free rollout, horizon curriculum", PURPLE], ["Per-role evaluation", "ID plus six OOD axes, reliable horizon", GREEN]];
  const w = 1.62, gap = 0.12;
  steps.forEach((st, i) => {
    const x = 0.6 + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.8, w, h: 2.9, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: x + 0.12, y: 1.95, w: 0.5, h: 0.5, fill: { color: st[2] }, line: { color: st[2] } });
    T(s, String(i + 1), { x: x + 0.12, y: 1.95, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, st[0], { x: x + 0.12, y: 2.6, w: w - 0.24, h: 0.6, fontSize: 14, bold: true, color: NAVY });
    T(s, st[1], { x: x + 0.12, y: 3.25, w: w - 0.24, h: 1.4, fontSize: 12 });
  });
  T(s, "Data side: build and validate the evidence", { x: 0.6, y: 4.95, w: 6.5, h: 0.35, fontSize: 15, bold: true, color: TEAL });
  T(s, "Model side: the controlled comparison", { x: 7.1, y: 4.95, w: 5.6, h: 0.35, fontSize: 15, bold: true, color: PURPLE });
  bullets(s, ["Every random choice is made at planning time, so an episode is a pure function of its manifest row", "Counterfactual siblings: same wind with the controller re-run (Mode A) or the action replayed (Mode B)"], 0.6, 5.4, 6.3, 1.5, 14);
  bullets(s, ["Same data, same budget, B2 against B3 as the central comparison", "Evaluation starts from the true state; error is open-loop position RMSE"], 7.1, 5.4, 5.6, 1.5, 14);
}
{
  const s = content("How the project evolved: the first pipeline had to be replaced",
    "1:30. Timeline from MASTER_README section 2. The audit on 12 Nov found the first-generation pipeline had never produced a usable dataset. The project then changed from training a model to building a physically trustworthy data generator. Mention that v7 and the stage-1 training happened on 26-27 Nov.");
  const ev = [["Sep–Oct", "First scaffold; study of AirDreamer; Isaac Sim 4.1 + OmniDrones container", MUTED], ["Nov 12", "Run 1 audit: no usable data, wind and drag were no-ops", RED], ["Nov 13–16", "Plan (D1–D10), Hydra architecture, physics validated in Isaac", TEAL],
    ["Nov 17", "Pilot v0.5.0: 147 episodes, validated", TEAL], ["Nov 20–24", "Models B0–B6; IMU artefact fixed (v0.6.0); 16 baselines", AMBER], ["Nov 25", "Checkpoint bug found; 29-run repeat sweep; constant-velocity baseline", AMBER],
    ["Nov 26", "Pilot v7: 4,530 episodes at 20 Hz; stage-1 training", PURPLE], ["Nov 27", "Vehicle-parameter diagnosis; known-vehicle training (9 runs)", GREEN]];
  const x0 = 0.9, x1 = 12.4, y0 = 3.25;
  s.addShape(pres.shapes.LINE, { x: x0, y: y0, w: x1 - x0, h: 0, line: { color: "AEBBC6", width: 2.5 } });
  ev.forEach((e, i) => {
    const cx = x0 + 0.5 + i * ((x1 - x0 - 1) / 7);
    s.addShape(pres.shapes.OVAL, { x: cx - 0.14, y: y0 - 0.14, w: 0.28, h: 0.28, fill: { color: e[2] }, line: { color: WHITE, width: 1.5 } });
    const up = i % 2 === 0;
    T(s, e[0], { x: cx - 0.8, y: up ? y0 - 0.5 : y0 + 0.25, w: 1.6, h: 0.3, fontSize: 13, bold: true, color: e[2], align: "center" });
    T(s, e[1], { x: cx - 0.95, y: up ? y0 - 1.95 : y0 + 0.6, w: 1.9, h: 1.4, fontSize: 12, align: "center", valign: up ? "bottom" : "top" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.3, w: 12.1, h: 1.4, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  T(s, [{ text: "Turning point.  ", options: { bold: true, color: RED } }, { text: "The thesis moved from “train a model” to “build evidence that can be trusted”: a validated simulator, a quality-checked dataset, and baselines evaluated honestly." }],
    { x: 0.9, y: 5.45, w: 11.5, h: 1.1, fontSize: 16, valign: "middle" });
}
{
  const s = content("What did not work: the first-generation pipeline",
    "1:30. Run 1 audit findings. Each defect on its own would have invalidated the results. Say what replaced each.");
  table(s, [["Defect found in the audit", "Consequence", "Replacement"],
    ["Only 18 episodes of length 1 ever produced", "No usable dataset", "Planned manifest + validated writer, 147-episode pilot"],
    ["Wind and drag were silent no-ops", "The OOD factor never acted on the drone", "One WrenchInjector call; tests that wind changes the motion"],
    ["ω labelled body-frame while world-frame", "Wrong state for every model", "Frame convention fixed and tested (ENU, FLU, wxyz)"],
    ["Mass randomisation never reached PhysX", "Nominal vehicle only, whatever the label", "Domain binding with PhysX read-back"],
    ["Controller ignored yaw; windowing at 200 Hz", "Poor tracking, unusable windows", "Yaw-aware controller; resampler to 20 / 50 Hz"],
    ["Training wind 2–16 m/s, not the planned 0–4 m/s", "OOD axis not controlled", "ID / OOD wind shells with realised-statistics checks"]],
    0.6, 1.6, 12.1, [4.2, 3.4, 4.5], { fs: 13, rh: 0.62, alignLeft: true });
  caption(s, "Source: docs/audit/run1_repository_audit.md; docs/audit/run2_implementation_plan.md", 0.6, 6.55, 12);
}

section("Infrastructure and data generation");
{
  const s = content("Infrastructure: one shared 96-core host, four GPUs, everything in one container",
    "1:00. Isaac Sim 4.1 and OmniDrones are not pip-installable, so everything runs in a docker container. Generation is CPU-bound, so several workers per GPU. Training models are small (60k to 227k parameters) and run three per GPU. GPU 3 is shared with another user and is not used.");
  const hw = [["CPU", "AMD EPYC 8434P, 48 cores / 96 threads"], ["Memory", "251 GB RAM (shared)"], ["GPUs", "4 × NVIDIA RTX A6000, 48 GB (3 used)"], ["Container", "Docker: Isaac Sim 4.1, Isaac Lab v1.0, OmniDrones"], ["Software", "Python 3.10, PyTorch, Hydra, h5py"], ["Simulator", "PhysX, 200 Hz physics, Hummingbird quadrotor"]];
  hw.forEach((h, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    card(s, 0.6 + col * 3.9, 1.55 + row * 1.75, 3.75, 1.6, h[0], h[1], [TEAL, AMBER, PURPLE][row], 14);
  });
  stat(s, 8.6, 1.6, 2.0, "267", "tests: 250 CPU + 17 Isaac", TEAL);
  stat(s, 10.7, 1.6, 2.0, "9", "parallel generation workers (v7)", AMBER);
  stat(s, 8.6, 3.3, 2.0, "6–16", "GB RAM per generation worker", PURPLE);
  stat(s, 10.7, 3.3, 2.0, "< 0.3", "GB GPU memory per training run", GREEN);
  bullets(s, ["Generation is CPU-bound: several workers per GPU, GPU chosen with CUDA_VISIBLE_DEVICES", "Training: three small models per GPU at the same time; the host load stays near 85", "Resilience: crash-safe resume, per-episode checksums, auto-resume after a server restart, live status page"], 8.6, 4.95, 4.1, 2.0, 13);
}
{
  const s = content("Simulation platform and data design: counterfactual groups and ID / OOD roles",
    "1:30. 200 Hz physics (5 ms), model steps 20 ms in the pilot and 50 ms in v7. Counterfactual siblings let wind effects be separated from controller reactions. Splits are assigned per group so that siblings never straddle train and test. OOD roles: wind magnitude (primary), direction, temporal, gust, physics (vehicle parameters), trajectory.");
  card(s, 0.6, 1.55, 3.9, 2.5, "Isaac Sim + OmniDrones", "Hummingbird quadrotor, 200 Hz physics, Lee / attitude / rate controllers, domain randomisation of mass, thrust, inertia, motor lag, air density, controller gains.", TEAL, 14);
  card(s, 0.6, 4.25, 3.9, 2.55, "Counterfactual groups", "A baseline episode and siblings with other wind: Mode A re-runs the controller, Mode B replays the same action. Splits are per group, never per episode.", AMBER, 14);
  T(s, "Splits (roles)", { x: 4.9, y: 1.5, w: 4, h: 0.35, fontSize: 17, bold: true, color: NAVY, fontFace: "Cambria" });
  table(s, [["Role", "Axis that is out of distribution"], ["train / val_id / test_id", "none (in distribution)"], ["test_ood_magnitude", "mean wind 6.2–8.5 m/s (primary)"], ["test_ood_direction", "unseen azimuth sectors"], ["test_ood_temporal", "unseen wind time structure"], ["test_ood_gust", "unseen gust shapes"], ["test_ood_physics", "vehicle parameters outside ID range"], ["test_ood_trajectory", "unseen reference trajectories"]],
    4.9, 1.95, 7.8, [3.0, 4.8], { fs: 13, rh: 0.5, alignLeft: true });
  caption(s, "Roles are assigned per group; OOD roles also have a validation twin (val_ood_*) for model selection.", 4.9, 6.05, 7.8);
}
{
  const s = content("Physics and wind models: every approximation has an envelope",
    "1:30. Rotor drag linear in rotor speed, body drag with per-axis drag coefficients, ground effect. Wind: steady, OU, Dryden turbulence, gusts, steps, sinusoids with realised-statistics guarantees. Envelope: rotor-drag mu <= 0.20; episodes outside are excluded from model-ready data. Honest limitation: not CFD.");
  card(s, 0.6, 1.55, 3.9, 2.9, "Aerodynamics (truth)", "Rotor drag d = 0.30 s⁻¹ (linear in Ω). Body drag with C_D = (1.1, 1.1, 1.3) and projected areas (0.0070, 0.0076, 0.0180) m². R = 0.10 m, ρ = 1.225 kg/m³. Ground effect, rotational damping.", TEAL, 14);
  card(s, 4.7, 1.55, 3.9, 2.9, "Wind", "Steady, OU, Dryden (MIL-F-8785C), cosine gusts, random gusts, steps, sinusoids. Realised statistics are checked against each regime’s limits.", AMBER, 14);
  card(s, 8.8, 1.55, 3.9, 2.9, "Applicability envelope", "Rotor-drag advance ratio μ ≤ 0.20 (widened from 0.15, fidelity reduced between 0.15 and 0.20). Episodes outside any envelope are excluded.", PURPLE, 14);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.7, w: 12.1, h: 2.0, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "Assumptions stated openly", { x: 0.9, y: 4.85, w: 6, h: 0.35, fontSize: 16, bold: true, color: AMBER });
  T(s, [{ text: "A physics-based approximation, not CFD", options: { bullet: { indent: 16 }, breakLine: true } }, { text: "Controllers use the true vehicle mass and effective inertia", options: { bullet: { indent: 16 }, breakLine: true } }, { text: "IMU noise is applied at load time; bias is not modelled", options: { bullet: { indent: 16 } } }],
    { x: 0.9, y: 5.3, w: 5.7, h: 1.3, fontSize: 14, color: WHITE, paraSpaceAfter: 6 });
  T(s, [{ text: "Why a simulator and not real flights: ground-truth state and wind, and counterfactual pairs, are needed to separate wind effects from controller reactions.", options: { color: "CADCFC" } }], { x: 7.0, y: 5.3, w: 5.4, h: 1.3, fontSize: 14 });
}
{
  const s = content("Physics validation: the simulator matches analytic expectations",
    "1:00. All checks ran inside Isaac Sim. Read two rows out loud: 1 N force gives 1.3957 against 1.3966; 3 m/s wind gives 0.9614 against 0.9618. Within-process runs are bitwise repeatable. Cross-process reproducibility had a bug that was found and fixed (next slides).");
  table(s, [["Check", "Measured", "Expected", "Error"], ["Hover acceleration", "9.810 m/s²", "9.810", "~0"], ["Free fall", "0", "0", "0"], ["1 N force on the body", "1.3957 m/s²", "1.3966", "0.06 %"], ["3 m/s wind at hover attitude", "0.9614 m/s²", "0.9618", "0.04 %"], ["Motor first-order step", "", "", "6 × 10⁻⁸"], ["Torque responses", "", "", "≈ 0.5 %"], ["Repeat the same episode (same process)", "identical", "identical", "bitwise"]],
    0.6, 1.6, 6.4, [2.9, 1.3, 1.1, 1.1], { fs: 13, rh: 0.5 });
  card(s, 7.4, 1.55, 5.3, 1.9, "Why this matters", "Every later claim rests on the simulator. These checks fix thrust, gravity, wind force and actuator response against hand calculations.", TEAL, 13);
  card(s, 7.4, 3.6, 5.3, 1.6, "Repeatability", "Bitwise within a process. Across processes a 6×10⁻⁶ m drift was found and fixed (next slides).", AMBER, 13);
  stat(s, 7.5, 5.4, 2.4, "17", "Isaac validation tests", TEAL);
  stat(s, 10.2, 5.4, 2.4, "250", "CPU tests (296 now)", AMBER);
}
{
  const s = content("Pilot v0.5 / v0.6: 147 episodes, validated, with findings reported",
    "1:30. Pilot v0.5.0 (141 completed, 6 terminated on ground contact kept and flagged). 0 hard and 0 soft findings from the validator. Findings: IMU accelerometer artefact (fixed at generation in v0.6.0), open-loop action-wind correlation 0.39 with permutation p = 0.085, domain factors correlated, OOD roles thin.");
  stat(s, 0.6, 1.55, 2.2, "147", "episodes (63 training)", TEAL);
  stat(s, 3.0, 1.55, 2.2, "0 / 0", "hard / soft validator findings", GREEN);
  stat(s, 5.4, 1.55, 2.2, "0.030", "Newton-law residual p99 [m/s²]", AMBER);
  stat(s, 7.8, 1.55, 2.2, "26 min", "generation, 3 workers", PURPLE);
  stat(s, 10.2, 1.55, 2.4, "0 NaN", "no sample out of limits", TEAL);
  T(s, "Findings that shaped the next stage", { x: 0.6, y: 3.3, w: 8, h: 0.4, fontSize: 18, bold: true, color: NAVY, fontFace: "Cambria" });
  table(s, [["Finding", "Evidence", "Action"],
    ["IMU accelerometer carried an extra −ω×v term", "p99 1.65 m/s², R² = 0.83 for the term", "Corrected at generation → v0.6.0; lever-arm theory tested and refuted"],
    ["Training wind never above 3.7 m/s; OOD peaks 4.3–8.2", "Residual has nothing to extrapolate from", "Wide-wind stratified plan (pilot v7)"],
    ["Open-loop action–wind correlation 0.39", "Permutation p = 0.085 (borderline)", "Reported; counterfactual Mode B and excitation in v7 test roles"],
    ["Only 2–7 independent groups per OOD role", "Wide intervals on every OOD number", "v7: 136–245 episodes per OOD role, 14–58 vehicle domains each, 473 vehicles"]],
    0.6, 3.8, 12.1, [4.3, 3.7, 4.1], { fs: 12, rh: 0.62, alignLeft: true });
}
{
  const s = content("Problems found while building the data, and how each was solved",
    "2:00. Pick five for the talk: IMU artefact, reproducibility, silent failures, best.pt bug (next section), wind no-op in the old pipeline. The reproducibility story: after five wrong hypotheses, a bit-level trace of controller inputs located the cause (realised inertia from float32 world poses before reset).");
  table(s, [["Problem", "Cause", "Fix"],
    ["Episode result depended on earlier episodes (≈ 6×10⁻⁶ m)", "Realised inertia computed from float32 world poses before reset; found after five wrong hypotheses by a bit-level trace", "Reset to canonical pose first; bitwise repeatable"],
    ["IMU gave gravity-only frames / never updated", "Sensor created after sim.reset(); wrong pipeline", "Create sensor before reset; gpu_dynamics_cpu_readback"],
    ["Dryden turbulence silently produced no wind", "Profile unit error", "Caught by a unit test before any generation"],
    ["Run aborted at episode 144, no traceback", "SimulationApp.close() kills the process before the error prints", "Log and continue; checkpoint index; --resume"],
    ["Simulator segfault on physics.device=cuda:N", "Isaac device handling", "Select GPUs only with CUDA_VISIBLE_DEVICES"],
    ["Rotor-drag envelope too narrow for OOD wind", "μ ≤ 0.15 excluded strong-wind episodes", "Widened to 0.20, fidelity cost stated"]],
    0.6, 1.55, 12.1, [4.0, 4.6, 3.5], { fs: 12, rh: 0.7, alignLeft: true });
}

section("Models, protocol and pilot results");
{
  const s = content("Models: B2 against B3 is the central comparison",
    "1:30. B0 and B1 are floors. B2 pure neural latent versus B3 physics plus residual plus memory, matched in parameters (227k). B4 removes memory; B5a/B5b isolate the physics split; B6 gives an oracle wrench as an upper bound. Constant-velocity extrapolation was added later as a no-learning reference.");
  table(s, [["ID", "Model", "Question it answers"],
    ["B0", "Hold last state", "Floor: how far does the vehicle move?"], ["B1", "Physics only (f_phys)", "What the nominal physics gives without learning"],
    ["B2", "Pure neural latent dynamics", "The thesis’s “pure learned” side"], [{ text: "B3", options: { bold: true, color: TEAL } }, { text: "Physics + residual + latent memory", options: { bold: true, color: TEAL } }, { text: "Proposed model; matched to B2 in parameters", options: { bold: true, color: TEAL } }],
    ["B4", "Physics + residual, no memory", "Is the memory what helps?"], ["B5a / B5b", "Learned field / exact kinematics + learned dynamics", "Is the physics split what helps?"],
    ["B6a / B6b", "Oracle wrench (no training / trained)", "Upper bound: if the disturbance were known"], ["CV", "Constant velocity (added in audit)", "No-learning reference so wins are not overstated"]],
    0.6, 1.55, 7.6, [1.1, 3.6, 2.9], { fs: 13, rh: 0.5, alignLeft: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.5, y: 1.55, w: 4.2, h: 4.5, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "B3 in one picture", { x: 8.8, y: 1.7, w: 3.6, h: 0.35, fontSize: 15, bold: true, color: AMBER });
  T(s, [{ text: "IMU history → GRU latent h", options: { breakLine: true } }, { text: "h, s, u → residual d_θ on v̇ and ω̇ only", options: { breakLine: true } }, { text: "f_phys + d_θ integrated with RK4", options: { breakLine: true } }, { text: "Physical state carried forward in imagination", options: { breakLine: true } }, { text: "Decoder only initialises the state" }],
    { x: 8.8, y: 2.2, w: 3.7, h: 3.2, fontSize: 14, color: WHITE, paraSpaceAfter: 10 });
  caption(s, "Parameter counts: B2/B3/B5 227k; B4 (B3 width) 73k; B3 small 60k.", 8.5, 6.15, 4.2);
}
{
  const s = content("Training and evaluation protocol",
    "1:00. Context window with teacher forcing, then a free rollout with a horizon curriculum 1, 5, 10, 20 steps. Geodesic attitude loss. Evaluation starts from the true state, because an IMU-only start gave about 120 degrees attitude error (yaw is unobservable). The pilot horizon was 20 steps = 0.4 s; v7 uses 50 ms steps so 20 steps = 1 s.");
  img(s, "curriculum.png", 0.6, 1.5, 6.0, 3.8);
  bullets(s, ["Context window (history 10) with teacher forcing, then a free-running rollout", "Horizon curriculum 1 → 5 → 10 → 20 steps, advancing when validation stops improving", "Loss: position, velocity, geodesic attitude, ω, plus small residual penalty", "Best checkpoint = best validation epoch over the whole run (after the audit fix)", "Evaluation: open-loop rollout from the true state, position / attitude / velocity RMSE per role, reliable imagination horizon"], 6.9, 1.5, 5.8, 4.6, 14);
  card(s, 0.6, 5.5, 6.0, 1.35, "Metric", "Pilot: 20 steps = 0.4 s. v7: 20 steps = 1.0 s at 50 ms steps.", AMBER, 14);
}
{
  const s = content("Pilot results: physics helps in distribution, not on strong OOD wind",
    "2:00. Position RMSE at 20 steps (0.4 s), pilot v0.6.0. B3 0.036 m against B2 0.105 and B1 0.053 in distribution, also best on temporal and gust OOD. On OOD wind magnitude B3 0.166 is about at B1's 0.185 and constant velocity 0.050 is better; the oracle 0.018 shows the information exists. Do not claim H1 is confirmed.");
  s.addChart(pres.charts.BAR, [
    { name: "B1 physics only", labels: ["Test ID", "OOD magnitude", "OOD direction", "OOD temporal", "OOD gust"], values: [0.053, 0.185, 0.058, 0.089, 0.054] },
    { name: "B2 pure neural", labels: ["Test ID", "OOD magnitude", "OOD direction", "OOD temporal", "OOD gust"], values: [0.105, 0.122, 0.056, 0.032, 0.043] },
    { name: "B3 proposed", labels: ["Test ID", "OOD magnitude", "OOD direction", "OOD temporal", "OOD gust"], values: [0.036, 0.166, 0.039, 0.026, 0.022] },
    { name: "Constant velocity", labels: ["Test ID", "OOD magnitude", "OOD direction", "OOD temporal", "OOD gust"], values: [0.072, 0.050, 0.027, 0.032, 0.036] },
    { name: "B6a oracle", labels: ["Test ID", "OOD magnitude", "OOD direction", "OOD temporal", "OOD gust"], values: [0.033, 0.018, 0.034, 0.074, 0.047] }],
    { x: 0.5, y: 1.4, w: 8.0, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [RED, AMBER, TEAL, "8A98A5", PURPLE], showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt",
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valAxisLabelColor: MUTED, catAxisLabelColor: MUTED, valGridLine: { color: "E1E7EC", size: 0.5 }, catGridLine: { style: "none" },
      showTitle: true, title: "Position RMSE at 20 steps (0.4 s) [m]", titleFontSize: 13, titleColor: NAVY, titleFontFace: "+mn-lt", valAxisMaxVal: 0.2, valAxisLabelFormatCode: "0.00" });
  card(s, 8.8, 1.5, 3.9, 1.7, "Worked", "B3 is best in distribution (3× better than B2) and on OOD temporal and gust wind.", GREEN, 13);
  card(s, 8.8, 3.35, 3.9, 1.7, "Did not work", "On OOD magnitude B3 (0.166 m) ≈ physics only (0.185 m); constant velocity (0.050) wins.", RED, 13);
  card(s, 8.8, 5.2, 3.9, 1.6, "Limits", "63 training episodes; 4–7 episodes per OOD role; 3 seeds for B3.", AMBER, 13);
}
{
  const s = content("Diagnostics: variants within noise, a probe finds no wind, a checkpoint bug removed",
    "1:30. B3 variants within seed noise. Ridge probe on the latent finds no linear wind-force information (R2 <= 0) while the torque control reaches 0.8. Audit found that best.pt was not the best checkpoint for 8 of 16 runs because the best metric reset at each curriculum stage; fixed with a regression test and all runs repeated (29-run sweep fix_v1).");
  img(s, "pilot_curves.png", 0.6, 1.4, 6.4, 4.1);
  caption(s, "Pilot validation curves: B4 and B5b get worse as they train. B3 variants (no action in residual, zero-residual penalty, shared trunk) are within seed noise of B3.", 0.6, 5.6, 6.4);
  card(s, 7.3, 1.5, 5.4, 1.5, "Checkpoint bug (C9)", "Best-metric reset at each curriculum stage → best.pt wrong for 8 of 16 runs. Fixed, regression test, runs repeated.", RED, 13);
  card(s, 7.3, 3.15, 5.4, 1.5, "Constant-velocity baseline (C11)", "Missing in the first comparison; it showed B5b’s good OOD numbers were an almost untrained epoch-1 model.", AMBER, 13);
  card(s, 7.3, 4.8, 5.4, 1.5, "Latent probe (C14)", "Ridge probe: no linearly readable wind force in the latent (R² ≤ 0); torque control gives 0.8.", PURPLE, 13);
  T(s, "Also: B4 and B5b get worse as they train; lower learning rate did not help; the weight-decay test was too weak (AdamW decay = lr × wd = 2e-6) to count.", { x: 0.6, y: 6.2, w: 12, h: 0.6, fontSize: 13, color: MUTED });
}

section("Pilot v7 and stage-1 results");
{
  const s = content("Pilot v7: wide, stratified wind so the residual has something to extrapolate from",
    "1:30. The pilot training wind never exceeded 3.7 m/s while OOD peaks were 4.3 to 8.2. v7 widens ID wind to a peak of 6.2 m/s with strata weighted to the upper band, and OOD magnitude to a mean above 6.2 (peak up to 9). Azimuth is uniform; elevation near horizontal by design.");
  img(s, "v7_wind.png", 0.6, 1.4, 8.1, 4.05);
  bullets(s, ["ID wind: mean 0–5.7 m/s, peak ≤ 6.07 m/s", "OOD magnitude: mean 6.25–8.5 m/s, peak ≤ 8.92 m/s", "Seven wind profile types, azimuth uniform over 360°", "Quality gate: each wind distribution is checked against its limits (all pass)"], 8.95, 1.45, 3.8, 3.6, 14);
  card(s, 0.6, 5.65, 12.1, 1.2, "What changed from the pilot", "Training wind peak 3.7 → 6.2 m/s; OOD magnitude from 4.3–8.2 m/s peaks to a mean above 6.2 m/s; 20 Hz model step; 10 s episodes.", TEAL, 13);
}
{
  const s = content("Pilot v7 dataset: 4,530 episodes, 0.91 M model steps, 473 vehicles",
    "1:30. Sizes per role. Train 2,211 episodes; ID tests 455; OOD roles 136 to 245 episodes with 14 to 58 vehicle domains each. 908 open-loop episodes ended early (B23) and are used for training only. Attitude-mode scenarios leave the rotor-drag envelope at 20 Hz (B24) and stay as an open decision.");
  img(s, "v7_sizes.png", 0.6, 1.4, 12.1, 3.3);
  stat(s, 0.6, 4.9, 2.4, "4,530", "episodes, 0 failures", TEAL);
  stat(s, 3.2, 4.9, 2.4, "0.91 M", "model steps at 20 Hz", AMBER);
  stat(s, 5.8, 4.9, 2.4, "3,238", "independent groups", PURPLE);
  stat(s, 8.4, 4.9, 2.1, "473", "vehicle domains", GREEN);
  stat(s, 10.6, 4.9, 2.1, "2.1 GB", "HDF5 on disk", TEAL);
}
{
  const s = content("Data generation cost: 4,530 episodes in 5.8 hours on 9 workers",
    "1:00. generation_wall_s = 20,819 s (5.8 h). 9 parallel workers, CPU-bound, 6 to 16 GB each. The pilot (147 episodes) took 26 minutes on 3 workers. Auto-resume after a server restart is set up. Planned cost was about 42 worker-hours.");
  stat(s, 0.6, 1.5, 2.6, "5.8 h", "wall time, 20,819 s", TEAL);
  stat(s, 3.4, 1.5, 2.6, "9", "parallel workers, GPUs 0–2", AMBER);
  stat(s, 6.2, 1.5, 2.6, "784", "episodes per hour", PURPLE);
  stat(s, 9.0, 1.5, 3.7, "2.1 GB", "dataset size (4,530 episodes)", GREEN);
  table(s, [["", "Pilot v0.5 / v0.6", "Pilot v7"], ["Episodes", "147", "4,530"], ["Model step", "20 ms (50 Hz)", "50 ms (20 Hz)"], ["Workers", "3", "9"], ["Wall time", "26 min", "5.8 h"], ["Disk", "72 MB", "2.1 GB"]],
    0.6, 3.2, 7.2, [2.4, 2.4, 2.4], { fs: 13, rh: 0.46 });
  card(s, 8.1, 3.2, 4.6, 1.65, "Cost driver", "CPU-bound simulation and recording; 6–16 GB RAM per worker, so the worker count is bounded by memory.", AMBER, 13);
  card(s, 8.1, 5.0, 4.6, 1.75, "Resilience", "Per-episode checksums, checkpointed index, --resume, auto-resume after reboot. 0 failures.", GREEN, 13);
}
{
  const s = content("v7 quality report: 44 checks pass, one fails, two findings need decisions",
    "1:30. 1 fail, 3 warn, 44 pass, 6 info. The fail is force_balance (p99 0.90 m/s2 against a limit of 0.5, max 5.0), traced to the attitude-mode scenarios at 20 Hz. B24: attitude-mode scenarios leave the rotor-drag envelope at 20 Hz. B23: 908 open-loop episodes ended early. Be direct that these are open decisions.");
  stat(s, 0.6, 1.5, 1.9, "44", "checks passed", GREEN);
  stat(s, 2.6, 1.5, 1.9, "3", "warnings", AMBER);
  stat(s, 4.6, 1.5, 2.3, "1", "failure (force balance)", RED);
  img(s, "v7_consistency.png", 0.6, 3.0, 6.3, 3.8);
  bullets(s, [{ t: "Fail: Newton check p99 0.90 m/s² (limit 0.5), max 5.0 — from attitude-mode scenarios at 20 Hz (B24, 318 episodes)", o: { color: RED } },
    "B23: 908 open-loop episodes ended early (my duration override); used for training only",
    "Open-loop action–wind correlation: permutation p = 0.033, reported as a warning",
    "Closed-loop correlation 0.03 (p = 0.31); domain-factor correlation ≤ 0.09",
    "20 Hz controller check: stable on 12 scenarios; position scenarios identical to 50 Hz; one attitude case 3× worse (n = 1)"], 7.2, 1.45, 5.5, 5.4, 13);
}
{
  const s = content("Training curves on v7: stage 1 (nominal) and known-vehicle runs",
    "1:30. Stage 1 used nominal physics. Curves: B3 improves steadily over 60 epochs; B2, B5a and B5b become unstable after the curriculum reaches horizon 20 (best checkpoint is kept). B3 and B3 small take about 9 h; the no-memory models stop early after 2 to 3 h. Learning-rate schedule fix is on the TODO.");
  img(s, "train_curves.png", 0.6, 1.4, 12.1, 3.6);
  table(s, [["Run", "Params", "Epochs", "Best epoch", "Best val [m]", "Wall time"], ["B2 pure neural", "227k", "37", "19", "0.218", "1.5 h"], ["B3 physics + residual + memory", "227k", "60", "53", "0.177", "8.9 h"], ["B3 small", "60k", "60", "52", "0.178", "9.0 h"], ["B4 no memory (B3 width)", "73k", "33", "25", "0.178", "2.8 h"], ["B5b learned dynamics", "227k", "37", "15", "0.222", "2.7 h"]],
    0.6, 5.1, 12.1, [4.1, 1.3, 1.3, 1.6, 1.9, 1.9], { fs: 12, rh: 0.29 });
}
{
  const s = content("Stage-1 results on v7: the models learn, B1 and the “oracle” look wrong",
    "2:00. Position RMSE at 20 steps (1 s). B3 0.18 m in distribution, B2 0.22. On OOD magnitude no learned model beats constant velocity by much, and B4 (0.33) beats B3 (0.50). B1 0.94 m and B6a 0.41 m are worse than constant velocity: this is what triggered the diagnosis on the next slide.");
  table(s, [["Model", "Test ID", "OOD magn.", "OOD dir.", "OOD temp.", "OOD gust", "OOD phys.", "OOD traj."],
    ["B0 hold", "0.928", "0.991", "0.944", "0.916", "0.723", "1.130", "0.791"],
    [{ text: "B1 physics only", options: { color: RED, bold: true } }, bad("0.939"), bad("2.165"), "0.740", "0.605", "0.556", bad("1.493"), "1.139"],
    ["Constant velocity", "0.361", best("0.282"), "0.124", "0.123", "0.260", best("0.322"), "0.308"],
    ["B2 pure neural", "0.222", "0.474", "0.130", "0.134", "0.145", "0.410", "0.205"],
    [{ text: "B3 (227k)", options: { bold: true, color: TEAL } }, best("0.180"), "0.502", "0.108", "0.120", "0.113", "0.458", best("0.143")],
    ["B4 (B3 width, 73k)", "0.195", "0.331", best("0.076"), best("0.089"), best("0.111"), "0.513", "0.158"],
    ["B5b learned dyn.", "0.225", "0.467", "0.112", "0.119", "0.143", best("0.378"), "0.227"],
    [{ text: "B6a oracle wrench", options: { color: RED, bold: true } }, bad("0.405"), "0.343", "0.382", "0.363", "0.412", "0.929", "0.397"]],
    0.6, 1.55, 12.1, [2.9, 1.3, 1.4, 1.3, 1.4, 1.3, 1.3, 1.2], { fs: 13, rh: 0.46 });
  caption(s, "Position RMSE at 20 steps (1.0 s), seed 0, best validation epoch. Episodes per role: test ID 266, OOD magnitude 225, direction 143, temporal 143, gust 143, physics 245, trajectory 136.", 0.6, 5.8, 12.1);
  T(s, [{ text: "Reading: ", options: { bold: true, color: NAVY } }, { text: "learned models beat B0 and B1 everywhere. B3 is best in distribution. On OOD wind magnitude the no-memory B4 is better than B3. B1 and B6a are worse than a no-learning baseline, which cannot be right for an “oracle”." }], { x: 0.6, y: 6.25, w: 12.1, h: 0.7, fontSize: 14 });
}
{
  const s = content("Open-loop rollouts: the error grows with the wind speed",
    "1:30. Left: a test OOD-magnitude episode, actual against predicted rollouts every second; the legend carries wind speed and azimuth. B1 drifts away, B3 and B4 stay close. Right: final-step error against wind speed. B1 rises roughly linearly; B3 and B4 are flat in distribution and rise above about 6 m/s, which is the OOD boundary.");
  img(s, "traj_ood_mag.png", 0.6, 1.4, 12.1, 3.56);
  caption(s, "OOD wind magnitude episode (wind 8.1 m/s, azimuth 172°): black = actual path, coloured = open-loop 1 s rollouts of B1, B3 and B4 started every second. Legend carries wind speed and azimuth.", 0.6, 5.0, 12.1);
  stat(s, 0.6, 5.55, 3.6, "2.3 m", "B1 error after 1 s at 8.1 m/s wind", RED);
  stat(s, 4.6, 5.55, 3.6, "0.54 m", "B3 at the same episode", TEAL);
  stat(s, 8.6, 5.55, 3.6, "0.47 m", "B4 at the same episode", GREEN);
}

section("Diagnosis, known vehicle, lessons");
{
  const s = content("Why B1 and the oracle failed on v7: vehicle-parameter mismatch, not a 20 Hz bug",
    "2:00. The oracle B6a knows the wrench but not the vehicle. Vehicle factors range plus or minus 10 percent for mass and force scale, so the thrust-to-weight ratio is off by up to about 20 percent and the acceleration bias up to about 2 m/s2. At a 1 s horizon this dominates. The per-episode B6a error correlates with |force_scale/mass - 1| at r = 0.98.");
  s.addChart(pres.charts.BAR, [{ name: "B6a RMS error at 1 s", labels: ["< 3 %", "3–6 %", "6–10 %", "> 10 %"], values: [0.097, 0.219, 0.397, 0.630] }],
    { x: 0.5, y: 1.4, w: 6.4, h: 4.6, barDir: "col", chartColors: [RED], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 12, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
      showLegend: false, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, valGridLine: { color: "E1E7EC", size: 0.5 }, catGridLine: { style: "none" },
      showTitle: true, title: "Oracle (B6a) error against thrust-to-weight mismatch [m]", titleFontSize: 13, titleColor: NAVY, titleFontFace: "+mn-lt", showCatAxisTitle: true, catAxisTitle: "|force_scale / mass − 1|", catAxisTitleFontSize: 11, catAxisTitleColor: MUTED });
  stat(s, 7.3, 1.5, 2.6, "r = 0.98", "error vs thrust-to-weight mismatch (266 episodes)", RED);
  stat(s, 10.1, 1.5, 2.6, "±10 %", "mass and force-scale randomisation", AMBER);
  bullets(s, ["B1 and B6a use nominal parameters; a 10 % thrust bias gives about 1 m/s² acceleration error, about 0.5 m after 1 s", "Nominal vehicle, no wind: B6a 0.020 m, so the 20 Hz physics core is fine", "With +5 % mass: 0.246 m ≈ 0.5·(0.05·g)·1² as predicted", "B3 vs B4 in stage 1 partly measured how well parameters are identified from history"], 7.3, 3.2, 5.4, 3.4, 14);
}
{
  const s = content("Known-vehicle physics prior: the fix, and an honest remainder",
    "2:00. Decision: in practice the vehicle parameters are known, and the model must predict dynamics under unknown wind. Randomisation tests generalisation across vehicles, not parameter learning. The prior uses each episode's true mass, kF, kM and inertia; drag scales, motor time constant and wind stay unknown. B1 stays drag-free by decision. B1_kv still shows 0.200 m with no wind: that is the vehicle's own aerodynamic drag, which B6a with the recorded wrench removes (0.006 m).");
  table(s, [["Position RMSE at 1 s [m]", "Nominal", "True parameters"], ["B6a oracle, test ID", bad("0.412"), best("0.022")], ["B6a oracle, OOD magnitude", "0.351", best("0.054")], ["B6a oracle, OOD physics", "0.808", best("0.026")], ["B6a, no wind, +5 % mass", "0.246", best("0.007")], ["B6a, no wind, nominal vehicle", "0.020", best("0.006")], ["B1 physics, test ID", "0.878", "0.761"], ["B1 physics, no wind, nominal", "0.207", "0.200"]],
    0.6, 1.55, 6.6, [3.4, 1.5, 1.7], { fs: 13, rh: 0.5 });
  card(s, 7.5, 1.55, 5.2, 2.0, "What changed", "f_phys uses each episode’s true mass, k_F, k_M and inertia (PhysX read-back). Residual nets do not see the vehicle vector; B2 and B5 get it as input.", TEAL, 13);
  card(s, 7.5, 3.7, 5.2, 1.5, "Still unknown", "Drag scales, motor time constant, wind. The physics prior stays drag-free by decision.", AMBER, 13);
  card(s, 7.5, 5.35, 5.2, 1.4, "Why B1 is not exact", "Self-induced drag: B6a with the recorded wrench reaches 0.006 m.", PURPLE, 13);
  caption(s, "Source: outputs/eval/kv_check_v7.json (60 episodes per role); problems log C15.", 0.6, 5.7, 6.6);
}
{
  const s = content("Known-vehicle training (running): B6b is far better; learned models improve slightly",
    "1:30. Best validation position RMSE at 20 steps (1 s). Nine runs launched at 07:42 on 27 Nov; four have finished (B2, B4, B4 B3 width, B5b), the others are still training at the time of writing. B6b with oracle wrench reaches 0.05 to 0.065 m. B3 is 0.175 against 0.177 in stage 1 at epoch 30 of 60. Re-update on the day.");
  s.addChart(pres.charts.BAR, [
    { name: "Stage 1 (nominal physics)", labels: ["B2", "B3", "B4 (B3 width)", "B5a", "B5b"], values: [0.218, 0.177, 0.178, 0.239, 0.222] },
    { name: "Known vehicle (so far)", labels: ["B2", "B3", "B4 (B3 width)", "B5a", "B5b"], values: [0.213, 0.175, 0.169, 0.233, 0.214] }],
    { x: 0.5, y: 1.4, w: 8.2, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: ["8A98A5", TEAL], showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 10, dataLabelPosition: "outEnd", dataLabelFontFace: "+mn-lt", dataLabelColor: INK,
      showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
      valGridLine: { color: "E1E7EC", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "Best val ID position RMSE at 1 s [m]", titleFontSize: 13, titleColor: NAVY, titleFontFace: "+mn-lt", valAxisMaxVal: 0.3, valAxisMinVal: 0 });
  bullets(s, ["9 runs, 3 per GPU, launched 27 Nov 07:42", "Done: B2, B4, B4 (B3 width), B5b. Still running: B3, B3 small, B5a, both B6b", "B6b (oracle wrench input): 0.050 m with future wrench, 0.065 m with current", "B2, B5a, B5b become unstable after epoch 20; best checkpoint kept", "Next: evaluate on all roles; this is where known-vehicle physics should show"], 9.0, 1.5, 3.8, 5.2, 14);
}
{
  const s = content("What worked, what did not, and why",
    "2:00. Summarise the evidence honestly: the physics prior helps where the training distribution covers the test condition and when the parameters are right; it does not extrapolate on strong wind magnitude yet. Say that each negative result has a diagnosis, and say which diagnosis is established and which is only a hypothesis.");
  const col = (x, head, color, rows) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.5, w: 5.95, h: 0.55, fill: { color }, line: { color }, rectRadius: 0.08 });
    T(s, head, { x: x + 0.25, y: 1.5, w: 5.5, h: 0.55, fontSize: 18, bold: true, color: WHITE, valign: "middle" });
    bullets(s, rows, x + 0.1, 2.25, 5.75, 4.6, 14);
  };
  col(0.6, "Worked, and why", GREEN, ["Physics prior in distribution: B3 0.036 m vs B2 0.105 m (pilot); gravity, thrust and kinematics are exact so the network learns only the remainder", "Validated simulator and quality gates: trustworthy evidence; bugs were caught by tests and audit", "Counterfactual groups and per-group splits: no leakage between siblings", "Known-vehicle prior: removes the parameter mismatch (oracle 0.41 → 0.02 m)", "Audit discipline: constant-velocity baseline and checkpoint fix corrected earlier claims"]);
  col(6.75, "Did not work, and why", RED, ["Physics prior on OOD wind magnitude: B3 ≈ B1; training wind never exceeded 3.7 m/s (fact); residual does not extrapolate (hypothesis: bounded training range, short history)", "Constant velocity beats learned models on OOD magnitude: reference was missing earlier (fact)", "B4 and B5b get worse as they train: cause not established", "Latent carries no wind information: probe R² ≤ 0 (fact); non-linear probe still to do", "Nominal-parameter physics at 1 s: mismatch up to 20 % (fact, r = 0.98)"]);
}
{
  const s = content("Limitations and threats to validity",
    "1:30. Be upfront. These are the points the panel will raise: pilot scale, thin OOD roles, one seed in stage 1 on v7, simulator realism, leakage through the controller, true initial state.");
  const items = [["Scale", "v7 stage 1: one seed per model; OOD roles have 136–245 episodes. Pilot: 63 training episodes."], ["Simulator realism", "Physics-based approximation inside stated envelopes, not CFD; one airframe."], ["Leakage", "Closed-loop action correlates with wind; mitigated by Mode B, excitation, open-loop segments and a permutation test (p = 0.033 open loop)."],
    ["Initial state", "Evaluation starts from the true state; IMU-only start gave about 120° attitude error."], ["Open data issues", "B23 (908 early-ended episodes, training only) and B24 (attitude scenarios at 20 Hz leave the envelope)."], ["Not shown yet", "Reliable-horizon curves under OOD wind with intervals; B2 vs B3 with several seeds; the effect of the known-vehicle prior on OOD."]];
  items.forEach((it, i) => card(s, 0.6 + (i % 2) * 6.1, 1.55 + Math.floor(i / 2) * 1.75, 6.0, 1.6, it[0], it[1], [TEAL, AMBER, PURPLE, GREEN, RED, MUTED][i], 13));
}
{
  const s = content("Future work: seven tasks tied to the research questions",
    "1:30. Concrete list from the guide, updated for v7 being done. Primary experiment: B2 against B3 reliability curves and reliable imagination horizon under OOD wind magnitude with intervals over groups.");
  const tasks = [["Finish known-vehicle training; evaluate on all roles", "RQ1", GREEN], ["Retrain B1–B6 on v7 with ≥ 3 seeds; horizon curriculum to 50 steps", "RQ1", TEAL], ["Test why the residual does not extrapolate: wind-range augmentation, OOD-aware checkpoint selection, longer history, stronger weight decay (0.1–1), non-linear probe, disturbance-supervision head", "RQ1, RQ2", AMBER],
    ["Primary experiment: B2 vs B3 error growth and reliable imagination horizon under OOD wind, intervals over groups", "RQ1, RQ2", PURPLE], ["Ablations: physics mismatch, training-data size, with and without memory", "RQ3", TEAL], ["Decide B24 / B23 data issues; optional uncertainty head", "RQ4", MUTED], ["Thesis writing; update proposal text for design changes", "–", MUTED]];
  tasks.forEach((t, i) => {
    const y = 1.5 + i * 0.77;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 0.68, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + 0.12, w: 0.44, h: 0.44, fill: { color: t[2] }, line: { color: t[2] } });
    T(s, String(i + 1), { x: 0.8, y: y + 0.12, w: 0.44, h: 0.44, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, t[0], { x: 1.45, y: y + 0.04, w: 9.6, h: 0.6, fontSize: 13, valign: "middle" });
    T(s, t[1], { x: 11.2, y: y + 0.04, w: 1.4, h: 0.6, fontSize: 13, bold: true, color: t[2] === MUTED ? INK : t[2], align: "right", valign: "middle" });
  });
}
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addNotes("0:45 + timeline. Three bullets and one honest limitation. Do not claim H1 is confirmed. Thank the panel and invite questions.");
  T(s, "Summary", { x: 0.9, y: 0.7, w: 8, h: 0.7, fontSize: 36, bold: true, color: WHITE, fontFace: "Cambria" });
  const pts = [["1", "Trustworthy evidence", "A validated simulator, a 4,530-episode dataset with 44 quality checks passed, and 9 baselines evaluated per OOD role. A first pipeline was found unusable and replaced."],
    ["2", "Mixed results, stated plainly", "The physics prior helps in distribution (B3 0.036 m vs B2 0.105 m, pilot), not yet on strong OOD wind, where a constant-velocity baseline wins. Causes: vehicle mismatch (established), residual extrapolation (hypothesis)."],
    ["3", "Next stage is running", "Known-vehicle training of 9 models on v7 is in progress; evaluation on all roles, seeds and the reliable-horizon experiment follow."]];
  pts.forEach((p, i) => {
    const y = 1.7 + i * 1.45;
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: y + 0.1, w: 0.7, h: 0.7, fill: { color: AMBER }, line: { color: AMBER } });
    T(s, p[0], { x: 0.9, y: y + 0.1, w: 0.7, h: 0.7, fontSize: 22, bold: true, color: NAVY, align: "center", valign: "middle" });
    T(s, p[1], { x: 1.9, y, w: 9.5, h: 0.4, fontSize: 20, bold: true, color: WHITE });
    T(s, p[2], { x: 1.9, y: y + 0.45, w: 10.4, h: 0.95, fontSize: 15, color: "CADCFC" });
  });
  T(s, [{ text: "Honest limitation: ", options: { bold: true, color: AMBER } }, { text: "pilot-scale statistics, one seed per model on v7, and H1 is not yet confirmed.", options: { color: WHITE } }], { x: 0.9, y: 6.2, w: 11.5, h: 0.5, fontSize: 16 });
}

(async () => {
  const out = path.join(__dirname, "PhysiWorldUAV_midterm.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
