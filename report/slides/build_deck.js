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


function arrow(s, x1, y1, x2, y2, color) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1, line: { color: color || INK, width: 2, endArrowType: "triangle" } });
}

section("Opening");
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
{
  const s = content("Outline of the talk",
    "0:30. Six parts. The talk is about work done on the project: what was built, what was measured, what worked and what did not.");
  const parts = [["1", "Problem and literature", "Statement, why it is hard, related work, gap, research questions", TEAL], ["2", "Approach and model", "Architecture, physics prior, residual, training, the baselines", AMBER],
    ["3", "Infrastructure and data", "Hardware, simulator, physics, wind, scenarios, validation", PURPLE], ["4", "Pilot experiments", "Protocol, results, diagnostics, what each result means", GREEN],
    ["5", "Pilot v7 and stage-1 results", "4,530 episodes at 20 Hz, quality, training, evaluation on all roles", RED], ["6", "Diagnosis, lessons, future", "Parameter mismatch, known-vehicle fix, limits, remaining work", MUTED]];
  parts.forEach((p, i) => {
    const x = 0.6 + (i % 2) * 6.1, y = 1.55 + Math.floor(i / 2) * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 6.0, h: 1.55, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.42, w: 0.7, h: 0.7, fill: { color: p[3] }, line: { color: p[3] } });
    T(s, p[0], { x: x + 0.25, y: y + 0.42, w: 0.7, h: 0.7, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, p[1], { x: x + 1.2, y: y + 0.25, w: 4.6, h: 0.45, fontSize: 18, bold: true, color: NAVY, fontFace: "Cambria" });
    T(s, p[2], { x: x + 1.2, y: y + 0.78, w: 4.6, h: 0.7, fontSize: 13, color: MUTED });
  });
}

section("Problem and literature");
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
  const s = content("Problem statement",
    "1:30. Read the research problem aloud. Then the scope: this is a prediction problem, not navigation or reinforcement learning. The observation is proprioceptive: IMU and motor commands. Data come from a simulator because ground-truth state, wind and counterfactuals are needed.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 12.1, h: 1.9, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "Research problem", { x: 0.9, y: 1.62, w: 6, h: 0.35, fontSize: 14, bold: true, color: AMBER });
  T(s, "Can known quadrotor physical dynamics, incorporated as a prior in a latent world model, improve the reliability of recursive long-horizon prediction under out-of-distribution wind disturbances?",
    { x: 0.9, y: 2.0, w: 11.5, h: 1.3, fontSize: 21, color: WHITE, fontFace: "Cambria", italic: true, valign: "middle" });
  card(s, 0.6, 3.65, 3.9, 3.1, "What is predicted", "The physical state of a quadrotor (position, velocity, attitude, body rate) over a long open-loop horizon, given the starting state and the future motor commands.", TEAL, 14);
  card(s, 4.7, 3.65, 3.9, 3.1, "What is measured", "How fast prediction error grows with rollout length, and the reliable imagination horizon: the longest horizon with mean position error below a tolerance.", AMBER, 14);
  card(s, 8.8, 3.65, 3.9, 3.1, "Scope", "A prediction problem, not navigation or RL. Observations: IMU and motor commands. Simulated flights with controlled wind. Vision, uncertainty and MPC are extensions only.", PURPLE, 14);
}

{
  const s = content("Background: a quadrotor is a known rigid body with unknown aerodynamics",
    "1:30. The equations are standard rigid-body dynamics. The point is the list on the right: drag, wind, actuator lag, parameter error and ground effects are not in the nominal model. The residual in this thesis is not a wind estimate; it is the net effect of everything the nominal model leaves out.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 6.4, h: 4.2, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "Nominal rigid-body model", { x: 0.9, y: 1.65, w: 5.8, h: 0.35, fontSize: 14, bold: true, color: AMBER });
  T(s, [{ text: "ṗ = v", options: { breakLine: true } }, { text: "m v̇ = m g + R f_T + f_ext", options: { breakLine: true } }, { text: "Ṙ = R [ω]×", options: { breakLine: true } }, { text: "J ω̇ = τ − ω × (J ω) + τ_ext", options: { breakLine: true } }, { text: "f_T = Σᵢ k_f Ωᵢ²  (four rotors)" }],
    { x: 0.9, y: 2.15, w: 5.9, h: 2.5, fontSize: 20, color: WHITE, fontFace: "Cambria", paraSpaceAfter: 8 });
  T(s, "R ∈ SO(3), ω in the body frame, J body inertia, m mass; f_ext and τ_ext carry everything not modelled.", { x: 0.9, y: 4.65, w: 5.9, h: 0.8, fontSize: 12, color: "CADCFC" });
  T(s, "Not in the nominal model", { x: 7.4, y: 1.5, w: 5.3, h: 0.4, fontSize: 18, bold: true, color: NAVY, fontFace: "Cambria" });
  bullets(s, ["Aerodynamic drag (rotor and body)", "Wind and turbulence", "Actuator dynamics and delay", "Mass and inertia uncertainty", "Rotor and ground effects", "Aerodynamic coupling and parameter error"], 7.4, 2.0, 5.3, 3.0, 15);
  card(s, 7.4, 4.95, 5.3, 1.8, "Residual, not wind estimate", "d_θ represents the net effect of wind, aerodynamic mismatch, actuator mismatch and parameter error. It is never trained against the wind.", TEAL, 13);
  card(s, 0.6, 5.85, 6.4, 0.95, "Idea", "Handle the known part analytically; learn only the remainder.", AMBER, 13);
}

{
  const s = content("Why it is hard (1): recursive prediction compounds error",
    "1:30. A world model feeds its prediction back as the next input, so a small one-step error affects every later step. Our own numbers show it: B3 has 0.0005 m error after one step and 0.18 m after twenty (1 s). One-step accuracy does not predict long-horizon reliability, so the evaluation has to look at error against horizon.");
  const xs = [0.7, 2.7, 4.7, 6.7];
  ["ŝ_t", "ŝ_t+1", "ŝ_t+2", "…  ŝ_t+H"].forEach((t, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xs[i], y: 1.6, w: 1.5, h: 0.7, fill: { color: i === 0 ? TEAL : (i === 3 ? RED : AMBER) }, line: { color: WHITE }, rectRadius: 0.08 });
    T(s, t, { x: xs[i], y: 1.6, w: 1.5, h: 0.7, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
    if (i < 3) arrow(s, xs[i] + 1.5, 1.95, xs[i + 1], 1.95, MUTED);
  });
  T(s, "The prediction at t+1 is the input for t+2, and so on to the horizon H.", { x: 0.7, y: 2.45, w: 7.6, h: 0.6, fontSize: 14, color: MUTED });
  bullets(s, ["A model that is accurate for one step can be unreliable over many", "Error must be reported as a function of horizon E(H), not as one-step RMSE", "Position, velocity and attitude have different units and are evaluated separately"], 0.7, 3.3, 7.6, 2.4, 15);
  s.addChart(pres.charts.LINE, [
    { name: "B1 physics only", labels: ["1", "5", "10", "20"], values: [0.0016, 0.0418, 0.1801, 0.9388] },
    { name: "B2 pure neural", labels: ["1", "5", "10", "20"], values: [0.0086, 0.0372, 0.0774, 0.2220] },
    { name: "B3 physics + residual", labels: ["1", "5", "10", "20"], values: [0.0005, 0.0104, 0.0410, 0.1800] }],
    { x: 8.5, y: 1.4, w: 4.4, h: 4.4, chartColors: [RED, AMBER, TEAL], lineSize: 3, lineDataSymbolSize: 7, valAxisLogScaleBase: 10, showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt",
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, valGridLine: { color: "E1E7EC", size: 0.5 }, catGridLine: { style: "none" },
      showTitle: true, title: "Position RMSE, test ID, v7 [m] (log)", titleFontSize: 12, titleColor: NAVY, titleFontFace: "+mn-lt", showCatAxisTitle: true, catAxisTitle: "rollout length [steps of 50 ms]", catAxisTitleFontSize: 11, catAxisTitleColor: MUTED });
  stat(s, 0.7, 5.75, 3.5, "0.0005 m", "B3 error after 1 step", TEAL);
  stat(s, 4.4, 5.75, 3.5, "0.18 m", "B3 error after 20 steps (1 s)", RED);
}

{
  const s = content("Why it is hard (2): the test wind differs from the training wind",
    "1:30. The model is trained with wind drawn from P_train and tested on P_test that differs. A pure learned model must extrapolate the whole dynamics. Gravity, thrust and kinematics stay valid, which is the motivation for the physics prior. The picture shows how the ranges changed between the pilot and v7: the pilot training wind never exceeded 3.7 m/s while the OOD peaks were 4.3 to 8.2 m/s.");
  T(s, "P_train ≠ P_test", { x: 0.6, y: 1.5, w: 5, h: 0.5, fontSize: 24, bold: true, color: NAVY, fontFace: "Cambria" });
  bullets(s, ["Stronger wind than any training sample (primary axis)", "Unseen wind directions", "Different temporal structure and sudden gusts", "Moderate changes in vehicle parameters"], 0.6, 2.1, 5.6, 2.4, 15);
  const rows = [["Pilot v0.6", 0, 3.7, 4.3, 8.2], ["Pilot v7", 0, 6.07, 6.25, 8.92]];
  const x0 = 6.7, wpm = 0.62;
  T(s, "Wind speed [m/s]: training range and OOD magnitude test range", { x: x0, y: 1.5, w: 6, h: 0.35, fontSize: 13, color: MUTED });
  rows.forEach((r, i) => {
    const y = 2.1 + i * 1.4;
    T(s, r[0], { x: x0, y, w: 3, h: 0.35, fontSize: 14, bold: true, color: NAVY });
    s.addShape(pres.shapes.RECTANGLE, { x: x0, y: y + 0.4, w: (r[2] - r[1]) * wpm, h: 0.45, fill: { color: TEAL }, line: { color: TEAL } });
    s.addShape(pres.shapes.RECTANGLE, { x: x0 + r[3] * wpm, y: y + 0.4, w: (r[4] - r[3]) * wpm, h: 0.45, fill: { color: RED }, line: { color: RED } });
    T(s, "train 0–" + r[2], { x: x0, y: y + 0.9, w: 2.2, h: 0.3, fontSize: 11, color: TEAL, bold: true });
    T(s, "OOD " + r[3] + "–" + r[4], { x: x0 + r[3] * wpm, y: y + 0.9, w: 2.4, h: 0.3, fontSize: 11, color: RED, bold: true });
  });
  card(s, 0.6, 4.7, 5.6, 2.1, "Pure learned model", "Must extrapolate the complete dynamics (gravity, thrust and aerodynamics) beyond the training wind.", RED, 14);
  card(s, 6.7, 4.7, 6.0, 2.1, "Physics-prior model", "Gravity, thrust, kinematics and rotation stay valid when wind changes; only the remainder is learned. Whether that remainder extrapolates is the open question.", GREEN, 14);
}

{
  const s = content("Why it is hard (3): the wind is not measured and the data can mislead",
    "2:00. Six difficulties that shaped the data design. Under a good controller the trajectory barely changes with wind, so the residual is weakly determined. In closed loop the action correlates with wind, so a network can learn wind from the action instead of the IMU. IMU and action alone do not give absolute position or yaw. Each point has a mitigation in the data or evaluation design.");
  const c = [["Wind is not an input", "Only IMU and motor commands are available; wind and disturbance force are logged for analysis and for the oracle baseline only.", TEAL],
    ["Weak identifiability", "A good controller cancels wind, so the state barely moves and the residual is weakly determined. Fix: bounded excitation, open-loop segments.", AMBER],
    ["Action shortcut", "In closed loop the rotor command correlates with wind, so d_θ can learn wind from u. Fix: counterfactual Mode B replay, permutation test.", RED],
    ["Observability", "IMU plus commands do not give absolute position or yaw (IMU-only start gave about 120° attitude error). Fix: start from the true state.", PURPLE],
    ["Parameter mismatch", "Mass, thrust, inertia differ per vehicle (±10 %); this error looks like a disturbance to the residual (d_param).", GREEN],
    ["Thin OOD evidence", "Few independent groups per OOD role make intervals wide. Fix: v7 with 136–245 episodes per OOD role.", MUTED]];
  c.forEach((k, i) => card(s, 0.6 + (i % 3) * 4.1, 1.55 + Math.floor(i / 3) * 2.7, 3.95, 2.55, k[0], k[1], k[2], 13));
}

{
  const s = content("Literature review (1): physics, hybrid models and wind",
    "1:30. Four strands. Rigid-body models are interpretable but approximate. Physics-informed networks (Gu 2024) put the physics in the loss. Hybrid residual models (NeuroBEM 2021, Islam 2026) add a learned correction to a first-principles model. PI-WAN (2025) studies wind-adaptive prediction. None of them measures how the hybrid behaves inside a recurrent latent model that is rolled forward open loop for a long horizon.");
  table(s, [["Strand", "Representative work", "What it shows", "What it leaves open"],
    ["First-principles quadrotor models", "Rigid-body mechanics", "Interpretable, used in simulation and control", "Approximate: aero, wind, actuators, parameters"],
    ["Physics-informed networks", "Gu, Primatesta, Rizzo 2024", "Physics in the loss improves accuracy and consistency under aero effects and periodic wind", "The learned transition itself stays a neural approximation"],
    ["Hybrid residual dynamics", "NeuroBEM (Bauersfeld et al. 2021); Islam et al. 2026", "First-principles + learned residual beats analytical and pure learned models", "Not studied inside a recurrent latent rollout over long open-loop horizons"],
    ["Wind and distribution shift", "PI-WAN (Wang et al. 2025)", "Physics-informed networks predict and track better in unseen wind, payload, perturbations", "Aims at adaptation and control, not at a reliable imagination horizon"]],
    0.6, 1.55, 12.1, [2.4, 2.9, 3.6, 3.2], { fs: 12, rh: 0.9, alignLeft: true });
  caption(s, "Full references in the backup slides.", 0.6, 6.5, 12);
}

{
  const s = content("Literature review (2): latent world models and UAV world models",
    "1:30. World Models, PlaNet, Dreamer and DreamerV3 established latent imagination. AirDreamer, WorldFly and SkyJEPA show UAV world models are an active field, so the thesis does not claim UAV world modelling is new. The survey by Zidan et al. and the thesis argument: one-step accuracy does not predict long-horizon reliability.");
  card(s, 0.6, 1.5, 3.95, 3.6, "Latent world models", "World Models (Ha 2018), PlaNet (Hafner 2019), Dreamer (2020), DreamerV3 (2023): e_t = E(o_t), h_t+1 = F(h_t, e_t, u_t), rolled forward without observations. Limitation relevant here: recursive error accumulation.", TEAL, 13);
  card(s, 4.7, 1.5, 3.95, 3.6, "World models for UAVs", "AirDreamer (Liu 2026) and WorldFly (Zheng 2026) for navigation; SkyJEPA (Rao 2026) for long-horizon quadrotor world models and sim-to-real control. UAV world modelling is an active direction, not new.", AMBER, 13);
  card(s, 8.8, 1.5, 3.9, 3.6, "Long-horizon reliability", "Survey by Zidan et al. 2026: one-step accuracy does not predict long-horizon reliability. Evaluation should report E(H) = mean of e(t+k) over k = 1…H, per quantity.", PURPLE, 13);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.35, w: 12.1, h: 1.4, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, [{ text: "Used in this project:  ", options: { bold: true, color: AMBER } }, { text: "data setup follows AirDreamer (OmniDrones, Hummingbird, 200 Hz physics / 20 Hz policy, domain randomisation); the latent follows the Dreamer family, but is memory attached to an explicit physical state, not the state itself.", options: { color: WHITE } }],
    { x: 0.9, y: 5.45, w: 11.5, h: 1.2, fontSize: 15, valign: "middle" });
}

{
  const s = content("Research gap and contribution",
    "1:30. Five facts are established separately in the literature. What is missing is a controlled comparison of the pure latent and the physics-prior latent under recursive prediction with controlled OOD wind. The contribution is the comparison, the benchmark and the reliable-horizon analysis, not a new component.");
  T(s, "Established separately", { x: 0.6, y: 1.5, w: 6, h: 0.4, fontSize: 18, bold: true, color: NAVY, fontFace: "Cambria" });
  const facts = ["Known physical structure improves learned UAV dynamics", "Residual learning captures the discrepancy between model and observation", "Physics-informed models are more robust to wind and environment changes", "Latent world models support recursive imagined rollouts", "UAV world models support long-horizon reasoning"];
  facts.forEach((f, i) => {
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: 2.1 + i * 0.82, w: 0.4, h: 0.4, fill: { color: GREEN }, line: { color: GREEN } });
    T(s, "✓", { x: 0.6, y: 2.1 + i * 0.82, w: 0.4, h: 0.4, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, f, { x: 1.2, y: 2.05 + i * 0.82, w: 5.0, h: 0.6, fontSize: 14, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.6, y: 1.5, w: 6.1, h: 2.2, fill: { color: LIGHT }, line: { color: RED, width: 2 }, rectRadius: 0.08 });
  T(s, "Missing", { x: 6.9, y: 1.62, w: 5, h: 0.35, fontSize: 16, bold: true, color: RED });
  T(s, "A controlled comparison of pure learned latent dynamics against physics-prior latent dynamics, under recursive long-horizon prediction and controlled OOD wind, with matched data and capacity.", { x: 6.9, y: 2.05, w: 5.5, h: 1.6, fontSize: 15 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.6, y: 3.9, w: 6.1, h: 2.85, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "Contribution", { x: 6.9, y: 4.02, w: 5, h: 0.35, fontSize: 16, bold: true, color: AMBER });
  T(s, [{ text: "A controlled B2 against B3 comparison", options: { bullet: { indent: 16 }, breakLine: true } }, { text: "An OOD wind benchmark with six OOD axes", options: { bullet: { indent: 16 }, breakLine: true } }, { text: "A reliable-imagination-horizon analysis", options: { bullet: { indent: 16 }, breakLine: true } }, { text: "Not a new component: each ingredient exists", options: { bullet: { indent: 16 } } }],
    { x: 6.9, y: 4.5, w: 5.5, h: 2.1, fontSize: 15, color: WHITE, paraSpaceAfter: 8 });
}

{
  const s = content("Research questions and hypotheses",
    "1:30. RQ1 is primary. H1: the physics-prior error grows more slowly. H2 is conditional: the benefit shrinks as the nominal model gets worse. A negative or conditional result is acceptable if the conditions of failure are identified. Reliable imagination horizon is defined on the right.");
  const rq = [["RQ1", "Does a known-physics prior make long-horizon prediction under OOD wind more reliable?", TEAL], ["RQ2", "Under which OOD wind conditions is the benefit largest?", AMBER], ["RQ3", "How sensitive is the benefit to nominal-model accuracy and data amount?", PURPLE], ["RQ4", "Extension: can predictive uncertainty flag the horizon where imagination becomes unreliable?", MUTED]];
  rq.forEach((r, i) => {
    const y = 1.5 + i * 1.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 6.3, h: 0.9, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + 0.18, w: 0.55, h: 0.55, fill: { color: r[2] }, line: { color: r[2] } });
    T(s, r[0].replace("RQ", ""), { x: 0.8, y: y + 0.18, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, r[1], { x: 1.55, y: y + 0.05, w: 5.2, h: 0.8, fontSize: 14, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 1.5, w: 5.5, h: 3.85, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "Hypotheses", { x: 7.5, y: 1.62, w: 4, h: 0.35, fontSize: 14, bold: true, color: AMBER });
  T(s, [{ text: "H1:  E_phys(H) < E_pure(H)", options: { breakLine: true, bold: true, fontSize: 20 } }, { text: "error grows more slowly with physics, under OOD wind", options: { breakLine: true, color: "CADCFC", fontSize: 13 } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "H2:  E_phys(H; δ) ≥ E_pure(H; δ)  for large δ", options: { breakLine: true, bold: true, fontSize: 18 } }, { text: "the benefit shrinks as the nominal model mismatch δ grows", options: { color: "CADCFC", fontSize: 13 } }],
    { x: 7.5, y: 2.1, w: 5.0, h: 3.1, fontSize: 18, color: WHITE, fontFace: "Cambria", paraSpaceAfter: 6 });
  card(s, 0.6, 5.55, 12.1, 1.25, "Reliable imagination horizon", "The longest horizon for which mean position error stays below a tolerance. A negative or conditional answer is acceptable if the conditions where the prior fails are identified.", TEAL, 13);
}

section("Approach and model");
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
  const s = content("Model interface: state, action, observation",
    "1:30. State is 13-dimensional: position, velocity, quaternion, body rate. Action is the mean rotor speed over the interval. Observation is the IMU: specific force and gyro. IMU noise is applied at load time, never stored. The model step is 20 ms in the pilot and 50 ms in v7. Wind and disturbance force are never model inputs; only B6 uses them.");
  table(s, [["Symbol", "Meaning", "Dim", "Frame / unit"],
    ["p", "position", "3", "world ENU, m"], ["v", "velocity", "3", "world, m/s"], ["q", "attitude, Hamilton unit quaternion", "4", "body → world, wxyz"], ["ω", "angular velocity", "3", "body FLU, rad/s"],
    [{ text: "s = (p, v, q, ω)", options: { bold: true } }, { text: "physical state", options: { bold: true } }, { text: "13", options: { bold: true } }, ""],
    ["u", "rotor speed, mean over the interval", "4", "rad/s"], ["o", "IMU: specific force + gyro (noise added at load time)", "6", "body, m/s² and rad/s"]],
    0.6, 1.55, 7.8, [2.3, 3.2, 0.7, 1.6], { fs: 13, rh: 0.5, alignLeft: true });
  card(s, 8.7, 1.55, 4.0, 1.9, "Time step", "Physics 200 Hz (5 ms). Model step 20 ms in the pilot, 50 ms in v7. History window 10 steps.", TEAL, 13);
  card(s, 8.7, 3.6, 4.0, 1.7, "Never inputs", "Wind, disturbance force and torque are logged but not given to B2–B5. Only B6 (oracle) uses them.", RED, 13);
  card(s, 8.7, 5.45, 4.0, 1.35, "Frames", "World ENU, body FLU, quaternion wxyz body→world.", AMBER, 13);
}

{
  const s = content("Model architecture (B3): physics-prior latent world model",
    "2:00. Top row: the latent path. The IMU and motor command go through an MLP encoder to e_t, a GRU updates the latent h_t, and a decoder gives a state estimate used to start or to supervise. Bottom row: the physical transition. f_phys and the learned residual d_theta are summed and integrated with RK4. The predicted state is turned into a synthetic IMU, phi, and fed back into the same encoder and GRU, so the latent keeps its memory during open-loop imagination.");
  const box = (x, y, w, h, head, sub, fill, fg) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
    T(s, [{ text: head, options: { bold: true, fontSize: 14, breakLine: true } }, { text: sub, options: { fontSize: 11 } }], { x: x + 0.08, y, w: w - 0.16, h, color: fg || WHITE, align: "center", valign: "middle" });
  };
  box(0.6, 1.55, 1.9, 1.05, "IMU + u", "6 + 4 inputs", MUTED);
  box(2.95, 1.55, 2.2, 1.05, "Encoder E", "MLP → e_t (64)", TEAL);
  box(5.6, 1.55, 2.2, 1.05, "Latent F", "GRU, h_t (128)", TEAL);
  box(8.25, 1.55, 2.0, 1.05, "Decoder D", "128→256→13, ŝ_t", AMBER);
  box(10.7, 1.55, 2.0, 1.05, "State ŝ_t", "start / supervision", MUTED);
  arrow(s, 2.5, 2.07, 2.95, 2.07, INK); arrow(s, 5.15, 2.07, 5.6, 2.07, INK); arrow(s, 7.8, 2.07, 8.25, 2.07, INK); arrow(s, 10.25, 2.07, 10.7, 2.07, INK);
  box(3.6, 3.4, 3.4, 0.95, "Residual d_θ(h, ŝ, u)", "MLP 2×256 SiLU → 12", PURPLE);
  box(3.6, 4.45, 3.4, 0.85, "f_phys(ŝ, u)", "known rigid body, no drag", NAVY);
  box(7.5, 3.75, 1.7, 1.15, "Σ → RK4", "Δt step", NAVY);
  box(9.7, 3.75, 3.0, 1.15, "s̃_t+1", "next physical state", GREEN);
  arrow(s, 7.0, 3.87, 7.5, 4.15, INK); arrow(s, 7.0, 4.87, 7.5, 4.5, INK); arrow(s, 9.2, 4.32, 9.7, 4.32, INK);
  arrow(s, 6.7, 2.6, 6.7, 3.4, PURPLE);
  box(0.6, 5.55, 3.9, 1.1, "φ: synthetic IMU", "R(q̂)ᵀ(v̇ − g), ω̂ → same E, F", AMBER);
  s.addShape(pres.shapes.LINE, { x: 11.2, y: 4.9, w: 0, h: 1.2, line: { color: INK, width: 2 } });
  arrow(s, 11.2, 6.1, 4.5, 6.1, INK);
  arrow(s, 3.2, 5.55, 3.2, 2.6, AMBER);
  T(s, [{ text: "Observation phase: ", options: { bold: true, color: TEAL } }, { text: "real IMU drives E and F.   ", options: {} }, { text: "Imagination: ", options: { bold: true, color: AMBER } }, { text: "φ(s̃) replaces the IMU; only future actions are used." }], { x: 4.8, y: 6.3, w: 7.9, h: 0.65, fontSize: 13 });
}

{
  const s = content("The physics prior f_phys: exact rigid body, no aerodynamics",
    "1:30. f_phys is differentiable and contains thrust from rotor speed, gravity, kinematics and rotation. It deliberately contains no drag, wind, ground effect or motor lag; these are the unknown part. The action is already the interval-mean rotor speed, so motor lag is not needed. The known-vehicle option feeds each episode's true mass, kF, kM and inertia.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 7.0, h: 4.25, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "State derivative (13-dim state)", { x: 0.9, y: 1.62, w: 6, h: 0.35, fontSize: 14, bold: true, color: AMBER });
  T(s, [{ text: "ṗ = v", options: { breakLine: true } }, { text: "v̇ = g + R(q) (Σ Tᵢ / m) ẑ_B", options: { breakLine: true } }, { text: "q̇ = ½ q ⊗ [0, ω_B]", options: { breakLine: true } }, { text: "J ω̇ = τ − ω × J ω", options: { breakLine: true } }, { text: "Tᵢ = k_F Ωᵢ²,  yaw reaction k_M Ωᵢ²", options: {} }],
    { x: 0.9, y: 2.1, w: 6.5, h: 2.6, fontSize: 20, color: WHITE, fontFace: "Cambria", paraSpaceAfter: 8 });
  T(s, "Torque from thrust differences (arm geometry) and the alternating yaw reaction. Integrated jointly with the residual by RK4; quaternion renormalised.", { x: 0.9, y: 4.7, w: 6.5, h: 0.95, fontSize: 12, color: "CADCFC" });
  card(s, 7.9, 1.5, 4.8, 1.9, "Contains", "Gravity, thrust from rotor speed, kinematics, rigid-body rotation. Differentiable in PyTorch.", GREEN, 13);
  card(s, 7.9, 3.55, 4.8, 1.9, "Does not contain", "Wind, drag (rotor and body), ground effect, motor lag, any aerodynamics. Decision: a drag-free prior; drag is unknown.", RED, 13);
  card(s, 0.6, 5.9, 12.1, 0.9, "Parameters", "Nominal Hummingbird by default; known-vehicle option uses each episode’s true mass, k_F, k_M and effective inertia.", TEAL, 13);
}

{
  const s = content("The learned residual d_θ and the latent memory",
    "1:30. The residual is an MLP on the latent, the state estimate and the action. It outputs a tangent-space correction, so the quaternion stays on the manifold. Default acc_only: only velocity and angular acceleration are corrected because the kinematics are exact. The last layer starts near zero, so training starts from pure physics. The latent is memory of what physics cannot explain; it is not a copy of the state.");
  card(s, 0.6, 1.5, 3.9, 2.55, "Input and output", "Input [h_t, ŝ_t, u_t]. Output 12 values in the tangent space: position-rate, v̇, attitude-rate, ω̇ corrections. 2 hidden layers of 256, SiLU.", TEAL, 13);
  card(s, 4.7, 1.5, 3.9, 2.55, "acc_only (default)", "Only v̇ and ω̇ are corrected; ṗ and attitude-rate stay exact, since kinematics are exact. The last layer is initialised near zero: training starts as pure physics.", AMBER, 13);
  card(s, 8.8, 1.5, 3.9, 2.55, "Interpretation", "d_θ ≈ d_wind + d_aero + d_param + d_other. This split is for analysis only; no loss ties d_θ to wind.", PURPLE, 13);
  card(s, 0.6, 4.25, 6.0, 2.55, "Latent memory h", "GRU (128) driven by the real IMU in the observation phase and by the synthetic IMU φ in imagination. Memory of what physics cannot explain, not a second copy of the state.", GREEN, 13);
  card(s, 6.7, 4.25, 6.0, 2.55, "Variants against the action shortcut", "B3a: residual without the action input. B3b: soft penalty on the residual at zero latent. B3c: residual shares the decoder trunk. All within seed noise of B3 on pilot data.", RED, 13);
}

{
  const s = content("Open-loop imagination: the object that is evaluated",
    "1:30. During observation the latent is driven by real IMU. During imagination, future observations are unavailable: the model uses only its own predictions and the future motor commands. The rollout h to s to h to s is what the thesis evaluates.");
  T(s, "Observation phase (history window)", { x: 0.6, y: 1.5, w: 6, h: 0.35, fontSize: 15, bold: true, color: TEAL });
  ["(o_j, u_j)", "E", "F", "h_j", "ŝ_j = D(h_j)"].forEach((t, i) => {
    const x = 0.6 + i * 2.45, w = 2.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.95, w, h: 0.8, fill: { color: i === 3 ? TEAL : LIGHT }, line: { color: TEAL }, rectRadius: 0.08 });
    T(s, t, { x, y: 1.95, w, h: 0.8, fontSize: 14, bold: true, color: i === 3 ? WHITE : NAVY, align: "center", valign: "middle" });
    if (i < 4) arrow(s, x + w, 2.35, x + 2.45, 2.35, MUTED);
  });
  T(s, "Imagination (no observations, only future actions u)", { x: 0.6, y: 3.2, w: 8, h: 0.35, fontSize: 15, bold: true, color: AMBER });
  ["h_t", "s̃_t+1", "h_t+1", "s̃_t+2", "h_t+2", "…", "s̃_t+H"].forEach((t, i) => {
    const x = 0.6 + i * 1.75, w = 1.35;
    const isS = t.startsWith("s");
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.75, w, h: 0.8, fill: { color: isS ? GREEN : (t === "…" ? WHITE : TEAL) }, line: { color: t === "…" ? WHITE : (isS ? GREEN : TEAL) }, rectRadius: 0.08 });
    T(s, t, { x, y: 3.75, w, h: 0.8, fontSize: 15, bold: true, color: t === "…" ? INK : WHITE, align: "center", valign: "middle" });
    if (i < 6) arrow(s, x + w, 4.15, x + 1.75, 4.15, MUTED);
  });
  T(s, "h → f_phys + d_θ → RK4 → s̃ → φ → E, F → h ...", { x: 0.6, y: 4.65, w: 12, h: 0.4, fontSize: 14, color: MUTED });
  bullets(s, ["Evaluation starts from the true state at the end of the window (IMU-only start gives about 120° attitude error)", "Position, velocity and attitude errors are reported per horizon and per role", "Rollouts of 1, 5, 10 and 20 steps (0.05 to 1.0 s in v7)"], 0.6, 5.15, 12, 1.7, 14);
}

{
  const s = content("Training objective, curriculum and settings",
    "1:30. Loss is decomposed by state component with a geodesic attitude distance on the quaternion. The rollout loss is a discounted sum over the horizon. Training uses a horizon curriculum 1, 5, 10, 20, advancing when validation plateaus. Checkpoint is the best validation epoch of the whole run.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 6.6, h: 3.5, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, [{ text: "L_state = λ_p L_p + λ_v L_v + λ_R L_R + λ_ω L_ω", options: { breakLine: true } }, { text: "L_roll = Σ_k γ^(k−1) L_state(ŝ_t+k, s_t+k)", options: { breakLine: true } }, { text: "L = L_dec + λ_roll L_roll + λ_d ‖d_θ‖²", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } }, { text: "L_R = d_q(q̂, q)²,  d_q = 2 atan2(‖Im(q̂ ⊗ q*)‖, |Re(q̂ ⊗ q*)|)", options: { fontSize: 14 } }],
    { x: 0.9, y: 1.7, w: 6.1, h: 3.1, fontSize: 18, color: WHITE, fontFace: "Cambria", paraSpaceAfter: 8 });
  table(s, [["Setting", "Value"], ["Optimiser", "AdamW, lr 1e-3, wd 1e-4"], ["Batch / warm-up", "256 / 200 steps"], ["Gradient clip", "1.0"], ["γ (rollout discount)", "0.99"], ["Curriculum", "1 → 5 → 10 → 20 steps"], ["Stage patience", "3 epochs"], ["Early stop patience", "10 epochs"], ["Checkpoint", "best val ID, whole run"]],
    7.5, 1.5, 5.2, [2.4, 2.8], { fs: 12, rh: 0.39, alignLeft: true });
  card(s, 0.6, 5.2, 12.1, 1.6, "Loss weights", "Inverse of each term’s mean value for the hold-last-state baseline on the training set (so no term dominates), floored at a physical resolution (1 cm, 1 cm/s, 0.5°, 0.01 rad/s). Attitude uses the unit quaternion with the geodesic distance (sign-invariant).", TEAL, 13);
}

{
  const s = content("Baselines: what each model isolates",
    "2:00. B0 and B1 are floors. B2 against B3 is the central comparison: same encoder, latent and decoder, same parameters, but B2 has no physics. B4 removes memory, B5 removes the physics split, B6 gives an oracle disturbance. Constant velocity is a no-learning reference added after the audit. B6a is an upper bound only for the aerodynamic part: it still uses nominal mass.");
  const Y = "✓", N = "–";
  table(s, [["ID", "Model", "f_phys", "Memory h", "Learned", "Purpose"],
    ["B0", "Hold last state", N, N, N, "Floor"],
    ["B1", "Physics only, RK4, nominal parameters", Y, N, N, "What physics alone gives"],
    ["B2", "Pure neural latent: same E, F, D, learned step ŝ′ = F_pure(ŝ, u, h)", N, Y, "all", "Pure-learned side of the thesis"],
    [{ text: "B3", options: { bold: true, color: TEAL } }, { text: "Physics + residual + latent memory", options: { bold: true, color: TEAL } }, Y, Y, "residual", { text: "Proposed model", options: { bold: true, color: TEAL } }],
    ["B4", "Physics + residual d_θ(s, u), no latent", Y, N, "residual", "Is memory what helps?"],
    ["B5a / B5b", "Learned field / exact kinematics + learned dynamics, no f_phys", N, Y, "all", "Is the physics split what helps?"],
    ["B6a", "f_phys + oracle disturbance wrench (no training)", Y, N, "none", "Upper bound if disturbance known"],
    ["B6b", "B3 with the oracle wrench as extra input", Y, Y, "residual", "Learned model given the disturbance"],
    ["CV", "Constant velocity (added in the audit)", N, N, N, "No-learning reference"]],
    0.6, 1.5, 12.1, [1.2, 4.6, 0.9, 1.1, 1.1, 3.2], { fs: 12, rh: 0.5, alignLeft: true });
}

{
  const s = content("Fair comparison: matched data, capacity and protocol",
    "1:30. All trained models use the same windows, loss, curriculum and optimiser. Parameter counts are matched to B3 (about 227 thousand); B4 is also shown at B3 width (73k) and B3 small (60k) to separate capacity from memory. The oracle B6 uses the nominal mass and inertia, so the parameter mismatch is not removed by it (unless the known-vehicle option is on).");
  s.addChart(pres.charts.BAR, [{ name: "Parameters (thousands)", labels: ["B2", "B3", "B3 small", "B4", "B4 (B3 width)", "B5a / B5b"], values: [227, 227, 60, 226, 73, 227] }],
    { x: 0.5, y: 1.4, w: 6.2, h: 4.2, barDir: "col", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelColor: INK, showLegend: false,
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, valGridLine: { color: "E1E7EC", size: 0.5 }, catGridLine: { style: "none" },
      showTitle: true, title: "Trainable parameters [thousands]", titleFontSize: 13, titleColor: NAVY, titleFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 260 });
  bullets(s, ["Same windows, loss, curriculum, optimiser, history and targets for every trained model", "B2’s hidden width is set so its parameter count is within 10 % of B3", "B4 and B3 small test whether capacity or memory matters", "B2 receives the latent through the same coupling: memory does not separate B2 from B3, the physics prior does", "B6 removes only the aerodynamic part of the error; it keeps nominal mass and inertia"], 7.0, 1.5, 5.7, 5.2, 14);
  card(s, 0.6, 5.75, 6.1, 1.05, "Central comparison", "B2 vs B3: identical except for the physics prior.", AMBER, 13);
}

{
  const s = content("Key design decisions (documented, each with its reason)",
    "1:30. These were decided with the guide as the work went on; each one is recorded in the specs and the problems log. The key reasoning: do not hide a scientific choice. The physics prior is drag-free on purpose, evaluation starts from the true state because yaw and position are unobservable from IMU alone.");
  table(s, [["Decision", "Choice", "Reason"],
    ["Rollout start state", "True state at window end", "IMU + commands do not give position or yaw; isolates dynamics from state estimation"],
    ["Residual channels", "acc_only (v̇, ω̇)", "Kinematics are exact; fewer ways to fit the wrong thing"],
    ["f_phys scope", "Rigid body and rotor thrust only", "Drag is unknown by decision; the residual must learn it"],
    ["Attitude loss", "Geodesic on unit quaternion", "Sign-invariant, same angle as SO(3)"],
    ["Checkpoint selection", "val ID only", "Keeps the OOD claim clean"],
    ["Latent feedback φ", "Synthetic IMU through the same E", "No extra learned embedding; memory survives imagination"],
    ["IMU noise", "Applied at load time, nominal preset", "Clean accelerometer plus u would reveal aero force directly"],
    ["Known vehicle", "True mass, k_F, k_M, inertia given to f_phys", "In practice the vehicle is known; wind and aero are what is unknown"]],
    0.6, 1.5, 12.1, [2.6, 3.9, 5.6], { fs: 12, rh: 0.58, alignLeft: true });
}

section("How the project evolved");
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
  const s = content("Data-generation software: one manifest row is one episode",
    "1:00. Plan, simulate, write, validate. Every random choice is made at planning time from a seed tree, so an episode is a pure function of a manifest row and can be regenerated bit for bit. The package table lists what each part does.");
  T(s, "config → plan (manifest) → Isaac runner → HDF5 shards → validator → model-ready windows", { x: 0.6, y: 1.5, w: 12.1, h: 0.45, fontSize: 16, bold: true, color: TEAL });
  table(s, [["Package", "Responsibility"],
    ["config/", "Hydra groups: drone, physics, limits, controller, scenario, excitation, wind, domain, forces"], ["core/, drones/", "Simulator wrapper, frames, seeds, provenance; domain → PhysX binding with read-back"],
    ["control/, trajectories/", "Lee / attitude / rate controllers; 18 scenario templates with feasibility checks"], ["excitation/, wind/", "Bounded excitation signals; wind components and regimes with realised-statistics guarantees"],
    ["randomization/, forces/", "Sobol / LHS domain sampling, ID / OOD shells; aero models and the wrench injector"], ["experiments/, data/", "Planner, counterfactual siblings, runner; schema v2, HDF5 writer, validator, splits"],
    ["models/, training/, evaluation/", "B0–B6, curriculum trainer, per-role evaluation, probes"]],
    0.6, 2.1, 8.0, [2.7, 5.3], { fs: 12, rh: 0.55, alignLeft: true });
  stat(s, 9.0, 2.1, 3.7, "296", "CPU tests pass", TEAL);
  stat(s, 9.0, 3.6, 3.7, "17", "Isaac validation tests", AMBER);
  stat(s, 9.0, 5.1, 3.7, "1 : 1", "manifest row : episode (seed tree, checksums)", PURPLE);
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
  const s = content("The vehicle: Hummingbird quadrotor, domain-randomised",
    "1:30. The Hummingbird: 0.716 kg, effective composite inertia measured in Isaac. Domain binding writes mass, inertia, thrust and moment constants and the motor lag into PhysX and reads them all back, hard-checking against the plan. Controllers use the true vehicle parameters. Domain factors: mass and force scale plus or minus 10 percent in distribution.");
  table(s, [["Quantity", "Value"], ["Total mass", "0.716 kg (base 0.68 + 4 × 0.009)"], ["Effective inertia", "diag(0.00768, 0.00768, 0.01304) kg m²"], ["Rotors", "“+” layout, arm L = 0.17 m, R = 0.10 m"], ["k_F, k_M", "8.55e-6 N/(rad/s)², 1.37e-7 N m/(rad/s)²"], ["Ω_max, T_max, T/W", "838 rad/s, 24.0 N, 3.42"], ["Hover", "Ω = 453 rad/s, throttle 0.54"], ["Motor lag", "τ_m = 8.9 ms (first order)"]],
    0.6, 1.5, 6.6, [2.3, 4.3], { fs: 12, rh: 0.5, alignLeft: true });
  card(s, 7.5, 1.5, 5.2, 2.55, "Domain binding", "Writes mass, inertia, k_F, k_M and motor lag into PhysX, reads them back and hard-checks against the plan. OmniDrones drag_coef is asserted to be 0.", TEAL, 13);
  card(s, 7.5, 4.2, 5.2, 2.6, "Randomisation (v7)", "473 vehicles: mass and force scale ±10 % in distribution, ±20 % in OOD physics; inertia ±5 %, motor time constant ±20 %, drag scales ±10 %.", AMBER, 13);
}

{
  const s = content("Aerodynamics: rotor drag and body drag, each with an envelope",
    "2:00. The simulator truth is rotor drag plus body drag, applied through one wrench injector. Rotor drag is linear in rotor speed and the in-plane relative airspeed; body drag is quadratic with per-axis coefficients and projected areas. Relative air velocity is always v minus w. Each model has an applicability envelope and episodes outside it are excluded. What is not modelled is listed.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 6.0, h: 2.6, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "Rotor-induced drag", { x: 0.9, y: 1.6, w: 5, h: 0.35, fontSize: 14, bold: true, color: AMBER });
  T(s, "F_i = −k_rd Ωᵢ P⊥ v_rel,i,   P⊥ = diag(1, 1, 0)", { x: 0.9, y: 2.05, w: 5.5, h: 0.6, fontSize: 16, color: WHITE, fontFace: "Cambria" });
  T(s, "d = 0.30 s⁻¹ at hover (x, y), none axial. Envelope: advance ratio μ ≤ 0.20 (fidelity reduced 0.15–0.20).", { x: 0.9, y: 2.75, w: 5.5, h: 1.2, fontSize: 12, color: "CADCFC" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.7, y: 1.5, w: 6.0, h: 2.6, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  T(s, "Body parasitic drag", { x: 7.0, y: 1.6, w: 5, h: 0.35, fontSize: 14, bold: true, color: AMBER });
  T(s, "F = −½ ρ (C_D A)(û) ‖v_rel‖ v_rel", { x: 7.0, y: 2.05, w: 5.5, h: 0.6, fontSize: 16, color: WHITE, fontFace: "Cambria" });
  T(s, "C_D = (1.1, 1.1, 1.3), areas (0.0070, 0.0076, 0.0180) m², ρ = 1.225. Envelope: airspeed ≤ 12 m/s.", { x: 7.0, y: 2.75, w: 5.5, h: 1.2, fontSize: 12, color: "CADCFC" });
  card(s, 0.6, 4.3, 4.0, 2.5, "Ground effect", "Image-rotor model; at normal heights (≥ 0.5 m) the factor is < 1.003, so it is off in the truth model.", MUTED, 13);
  card(s, 4.7, 4.3, 4.0, 2.5, "Not modelled", "Translational lift, wind-on-thrust, unsteady aero, rotor-rotor interference, vortex-ring state, blade flexibility.", RED, 13);
  card(s, 8.8, 4.3, 3.9, 2.5, "Rule", "Episodes outside any envelope are excluded from model-ready data; every config must be validated.", GREEN, 13);
}

{
  const s = content("Wind models: steady, gusts, OU, Dryden turbulence, with realised-statistics guarantees",
    "1:30. Eight wind components. Each component draws its own child seed from the episode wind seed and a stable name, so counterfactual siblings share realisations. The Dryden frozen-turbulence speed is the episode mean wind rather than the vehicle airspeed, so wind does not depend on the trajectory. Regimes carry limits that are checked on realised statistics.");
  table(s, [["Component", "Model"], ["Steady", "constant vector"], ["1-cos gust, step, sinusoid", "deterministic shapes, step with rise time"], ["Ornstein–Uhlenbeck", "exact AR(1), stationary from t = 0, anisotropic"], ["Random gust events", "Poisson 1-cos pulses"], ["Dryden (MIL-F-8785C)", "exact discretisation; σ within 5 % from t = 0"], ["Spatial gradient", "G (p − p₀) per rotor hub"]],
    0.6, 1.5, 7.0, [3.0, 4.0], { fs: 12, rh: 0.5, alignLeft: true });
  card(s, 7.9, 1.5, 4.8, 2.0, "Seeds", "Each component has its own child seed (episode seed + stable name): siblings share realisations, components are independent.", TEAL, 13);
  card(s, 7.9, 3.65, 4.8, 1.75, "Frozen turbulence", "Speed = episode mean wind, not vehicle airspeed, so counterfactual pairs share realisations.", AMBER, 13);
  card(s, 0.6, 5.2, 7.0, 1.6, "Regimes", "ID, OOD magnitude, direction, temporal, gust; each has limits on mean and peak speed that the realised wind must satisfy (checked in the quality report).", PURPLE, 13);
  card(s, 7.9, 5.55, 4.8, 1.25, "Profile types in v7", "steady, OU, sinusoid, step, cosine gust, random gusts.", GREEN, 13);
}

{
  const s = content("Scenarios, excitation and data types",
    "1:30. Three data types: closed loop (controller tracks a reference), open loop (rotor programs, free response for system identification) and physics validation (thesis Cases A and C). Eighteen scenario templates, seven excitation signal types. References are feasibility checked. Helix and figure-eight appear only in the trajectory OOD role.");
  table(s, [["Data type", "How", "Purpose"], ["closed_loop", "Lee / attitude controller tracks a reference", "Realistic flight"], ["open_loop", "Predefined rotor programs from hover trim, no tracking", "System identification, free response"], ["physics_validation", "Case A (no wind), Case C (mass +5 %)", "Test the nominal f_phys"]],
    0.6, 1.5, 7.2, [1.9, 3.3, 2.0], { fs: 12, rh: 0.55, alignLeft: true });
  card(s, 8.1, 1.5, 4.6, 2.2, "Scenario templates (18)", "Basic: hover, vertical, translation, line, accel/decel. Complex: circle, helix, figure eight, sinusoidal, waypoints, combined. Attitude: yaw step/sine, roll/pitch doublet, attitude combined.", TEAL, 12);
  card(s, 0.6, 3.9, 6.0, 2.9, "Excitation", "Smooth random, multisine, chirp, PRBS, yaw excitation, rotor doublets and chirps. Bounded, anchored at rest, all parameters sampled once and stored. Purpose: make the wind effect identifiable.", AMBER, 13);
  card(s, 6.7, 3.9, 6.0, 2.9, "Guarantees", "Every reference starts at rest and is C³. Plan-time feasibility check: thrust 15–85 % of max, tilt, tilt rate, yaw rate, speed, acceleration, jerk. Hover is 10 % of the closed-loop mix; helix and figure eight appear only in OOD trajectory.", PURPLE, 13);
}

{
  const s = content("Dataset format and reproducibility",
    "1:30. HDF5 shards with schema v2 plus a manifest, provenance and an index with a completeness flag. A dataset is complete only if every planned episode was written and every hard validation check passed. Every array except the IMU accelerometer is bitwise reproducible across processes, kill and resume, and GPUs; the accelerometer is reproducible to about 2e-5 m/s2.");
  table(s, [["File", "Content"], ["manifest.jsonl", "one fully specified plan row per episode (every sampled value and seed)"], ["shard_XXX.h5", "episodes: ground truth, IMU, rotor commands, wrench, wind, metadata"], ["index.json", "episode rows, split role, group, status, validation summary, complete flag"], ["provenance.json", "config hash, git state, Isaac and library versions, GPUs"], ["validation.json", "hard and soft findings of the validator"]],
    0.6, 1.5, 7.4, [2.2, 5.2], { fs: 12, rh: 0.55, alignLeft: true });
  card(s, 8.3, 1.5, 4.4, 2.35, "Sampling semantics", "Physics rate 200 Hz. State sampled at instants, actions held over intervals; the actions are exactly piecewise constant at the model rate.", TEAL, 13);
  card(s, 8.3, 4.0, 4.4, 2.8, "Reproducibility", "Bitwise across processes, resume and GPUs, except the IMU accelerometer (about 2e-5 m/s²). Per-episode SHA-256 checksums verified on resume and merge.", GREEN, 13);
  card(s, 0.6, 5.0, 7.4, 1.8, "Windows", "Model-ready view: decimate to the model rate, windows of history 10 plus horizon, future actions; IMU noise at load time; splits per group.", AMBER, 13);
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
section("Pilot experiments");
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
{
  const s = content("The constant-velocity result: what it means and what it does not",
    "1:30. This is the result the panel will ask about. Constant velocity needs no learning and is better than every physics-based model on OOD magnitude and direction in the pilot. Known: B1 lacks the wind force, and B3's residual does not extrapolate beyond the training wind. Unknown: why exactly the residual fails to extrapolate. The reference exists so wins are not overstated.");
  stat(s, 0.6, 1.5, 3.8, "0.050 m", "constant velocity, OOD magnitude, 0.4 s (pilot)", GREEN);
  stat(s, 4.6, 1.5, 3.8, "0.166 m", "B3 on the same role", RED);
  stat(s, 8.6, 1.5, 4.0, "0.185 m", "B1 physics only", MUTED);
  card(s, 0.6, 3.1, 6.0, 3.7, "What is established", "B1 has no wind force (0.185 m). B3’s residual does not extrapolate beyond the training wind: the pilot training wind peaked at 3.7 m/s, the OOD peaks were 4.3 to 8.2 m/s. A ridge probe finds no linearly readable wind force in the latent (R² ≤ 0).", TEAL, 16);
  card(s, 6.7, 3.1, 6.0, 3.7, "What is not established", "Why the residual fails to extrapolate: bounded training range, short history with latent reset per window, a linear probe, or something else. Tests planned: wider training wind (done in the v7 data), longer history, non-linear probe, OOD-aware checkpoint selection, stronger weight decay.", AMBER, 16);
}

section("Pilot v7");
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
  const s = content("Pilot v7 randomisation: 473 vehicles, ID and OOD ranges",
    "1:30. Vehicle factors relative to nominal. The ID range is plus or minus 10 percent in mass and force scale; the OOD physics role uses plus or minus 20 percent. Other factors are plus or minus 5 percent for inertia shape, 20 percent for the motor time constant. The factors are nearly uncorrelated (largest absolute correlation 0.09), which was a problem in the pilot (0.59).");
  img(s, "v7_vehicle.png", 0.6, 1.4, 8.2, 5.4);
  bullets(s, ["473 distinct vehicles", "ID: mass and force scale ±10 %; OOD physics ±20 %", "Domain factors nearly uncorrelated: largest |ρ| = 0.09 (pilot 0.59)", "Colours: train, val ID, test ID, OOD magnitude, other OOD"], 9.1, 1.5, 3.6, 4.5, 14);
}

{
  const s = content("Pilot v7 coverage: wind bins, vehicle bins and scenario mix",
    "1:00. Left: episodes per mean-wind bin; calm is still the largest bin but the strata cover 0 to 8.5 m/s. Middle: ID domains per vehicle-factor bin, all populated. Right: scenario by role; closed loop dominates training, helix and figure eight appear only in the trajectory OOD role.");
  img(s, "v7_coverage.png", 0.6, 1.5, 12.1, 4.0);
  bullets(s, ["Mean wind is stratified and weighted to the upper band; calm episodes remain the largest bin", "Every vehicle-factor bin of the ID range holds 30–46 domains", "Helix and figure-eight trajectories appear only in test_ood_trajectory"], 0.6, 5.6, 12.1, 1.4, 14);
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
section("Stage-1 results");
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
  const s = content("Stage-1 evaluation on all roles: position RMSE at 1 s",
    "1:30. Heat map of position RMSE at the longest horizon (20 steps, 1 s) across 13 roles and all models. Darker is worse. B1 is the darkest on OOD magnitude and physics. The B6a oracle row is not an upper bound here because of the parameter mismatch, which is the diagnosis on the next slides.");
  img(s, "heatmap_v7.png", 0.6, 1.4, 12.1, 4.7);
  caption(s, "Log colour scale; numbers are metres. B6a and B1 use nominal vehicle parameters in this evaluation.", 0.6, 6.2, 12.1);
}

{
  const s = content("Error growth with rollout length: selected models on three roles",
    "1:30. Position RMSE against rollout length in seconds on test ID, OOD wind magnitude and OOD wind direction. Learned models are 4 to 5 times better than physics alone at 1 s in distribution. B3 and B4 are close in distribution; on OOD magnitude, B4 is better than B3 and all models are far from the in-distribution level.");
  img(s, "err_horizon_sel.png", 0.6, 1.4, 12.1, 3.85);
  bullets(s, ["In distribution B3 is best at every horizon; B1 grows fastest", "OOD direction errors are lower than in distribution; OOD magnitude is hard for all models", "On OOD magnitude the error at 1 s is 1.7 to 2.8 times the in-distribution value"], 0.6, 5.4, 12.1, 1.5, 14);
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
section("Lessons and next steps");
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
  const s = content("Status against the thesis plan",
    "1:00. Done, running and still to do. The next stage is already running: known-vehicle training. Then evaluation on all roles, seeds, the OOD-failure tests and the primary B2 against B3 experiment.");
  const rows = [["Audit, plan, configuration architecture", "done", GREEN], ["Simulator physics validated in Isaac", "done", GREEN], ["Pilot v0.5 / v0.6 and 16 baseline runs, repeat sweep (29 runs)", "done", GREEN], ["Pilot v7 generated and quality-checked", "done, B23 / B24 decisions open", GREEN],
    ["v7 stage-1 training (7 models) and evaluation", "done", GREEN], ["Known-vehicle training (9 runs)", "running", AMBER], ["Evaluation on all roles, seeds, reliable-horizon curves", "next", PURPLE], ["Tests of why the residual does not extrapolate", "next", PURPLE], ["Primary experiment B2 vs B3 and ablations (RQ1 to RQ3)", "next", PURPLE], ["Thesis writing", "continuing", MUTED]];
  rows.forEach((r, i) => {
    const y = 1.45 + i * 0.54;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 0.46, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.06 });
    s.addShape(pres.shapes.OVAL, { x: 0.75, y: y + 0.1, w: 0.26, h: 0.26, fill: { color: r[2] }, line: { color: r[2] } });
    T(s, r[0], { x: 1.2, y, w: 7.4, h: 0.46, fontSize: 14, valign: "middle" });
    T(s, r[1], { x: 8.8, y, w: 3.8, h: 0.46, fontSize: 14, bold: true, color: r[2] === MUTED ? INK : r[2], align: "right", valign: "middle" });
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
section("Backup");
{
  const s = content("Backup: likely questions and honest answers",
    "Backup slide. Use it when asked.");
  table(s, [["Question", "Answer"],
    ["Is H1 supported?", "Partly. In distribution and for temporal and gust OOD, yes (pilot). On strong OOD wind magnitude, not yet: B3 ≈ physics only; constant velocity is better."],
    ["Why a simulator?", "Ground-truth state, wind and counterfactual pairs are needed to separate wind effects from controller reactions. Real-flight transfer is out of scope."],
    ["Leakage through the controller?", "Closed-loop u correlates with wind. Mitigations: Mode B replay, excitation, open-loop segments, permutation test (open loop p = 0.033 in v7)."],
    ["Why start from the true state?", "IMU-only start gave about 120° attitude error (yaw unobservable)."],
    ["Is the simulator realistic?", "A validated approximation inside stated envelopes, not CFD."],
    ["What about the bugs?", "The best.pt selection bug affected 8 of 16 runs; found by audit, fixed, runs repeated. The first pipeline was replaced after audit."],
    ["Novelty?", "Components are not new. The contribution is the controlled comparison, the OOD benchmark and the reliable-horizon analysis."]],
    0.6, 1.4, 12.1, [3.0, 9.1], { fs: 12, rh: 0.66, alignLeft: true });
}

{
  const s = content("Backup: references",
    "Backup slide.");
  const refs = ["D. Ha, J. Schmidhuber, “World Models,” arXiv:1803.10122, 2018.", "D. Hafner et al., “Learning Latent Dynamics for Planning from Pixels,” ICML, 2019.", "D. Hafner et al., “Dream to Control: Learning Behaviors by Latent Imagination,” ICLR, 2020.", "D. Hafner et al., “Mastering Diverse Domains through World Models,” arXiv:2301.04104, 2023.",
    "W. Gu, S. Primatesta, A. Rizzo, “Physics-informed neural network for quadrotor dynamical modeling,” Robotics and Autonomous Systems 171, 2024.", "L. Bauersfeld et al., “NeuroBEM: Hybrid Aerodynamic Quadrotor Model,” RSS, 2021.", "M. Wang et al., “PI-WAN: A Physics-Informed Wind-Adaptive Network for Quadrotor Dynamics Prediction in Unknown Environments,” IROS, 2025.",
    "M. A. Islam et al., “Physics-guided residual learning for phase-aware UAV trajectory prediction in urban environments,” Scientific Reports 16, 2026.", "P. Rao et al., “SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors,” arXiv:2606.23444, 2026.", "Z. Liu et al., “AirDreamer: Generalist Drone Navigation with World Models,” arXiv:2606.03252, 2026.",
    "S. Zheng et al., “WorldFly: A World-Model-Based Vision-Language-Action Model for UAV Navigation,” arXiv:2606.06147, 2026.", "A. H. Zidan et al., “World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications,” arXiv:2606.00133, 2026."];
  T(s, refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 5 } })), { x: 0.6, y: 1.4, w: 12.1, h: 5.5, fontSize: 12 });
}


(async () => {
  const out = path.join(__dirname, "PhysiWorldUAV_midterm.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
