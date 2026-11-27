//@@outline
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
//@@problem
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
//@@dynamics
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
//@@ch1
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
//@@ch2
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
//@@ch3
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
//@@lit1
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
//@@lit2
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
//@@gap
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
//@@rq
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
//@@state
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
//@@arch
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
//@@fphys
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
//@@resid
{
  const s = content("The learned residual d_θ and the latent memory",
    "1:30. The residual is an MLP on the latent, the state estimate and the action. It outputs a tangent-space correction, so the quaternion stays on the manifold. Default acc_only: only velocity and angular acceleration are corrected because the kinematics are exact. The last layer starts near zero, so training starts from pure physics. The latent is memory of what physics cannot explain; it is not a copy of the state.");
  card(s, 0.6, 1.5, 3.9, 2.55, "Input and output", "Input [h_t, ŝ_t, u_t]. Output 12 values in the tangent space: position-rate, v̇, attitude-rate, ω̇ corrections. 2 hidden layers of 256, SiLU.", TEAL, 13);
  card(s, 4.7, 1.5, 3.9, 2.55, "acc_only (default)", "Only v̇ and ω̇ are corrected; ṗ and attitude-rate stay exact, since kinematics are exact. The last layer is initialised near zero: training starts as pure physics.", AMBER, 13);
  card(s, 8.8, 1.5, 3.9, 2.55, "Interpretation", "d_θ ≈ d_wind + d_aero + d_param + d_other. This split is for analysis only; no loss ties d_θ to wind.", PURPLE, 13);
  card(s, 0.6, 4.25, 6.0, 2.55, "Latent memory h", "GRU (128) driven by the real IMU in the observation phase and by the synthetic IMU φ in imagination. Memory of what physics cannot explain, not a second copy of the state.", GREEN, 13);
  card(s, 6.7, 4.25, 6.0, 2.55, "Variants against the action shortcut", "B3a: residual without the action input. B3b: soft penalty on the residual at zero latent. B3c: residual shares the decoder trunk. All within seed noise of B3 on pilot data.", RED, 13);
}
//@@imag
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
//@@loss
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
//@@baselines
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
//@@baselines2
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
//@@decisions
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
//@@software
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
//@@drone
{
  const s = content("The vehicle: Hummingbird quadrotor, domain-randomised",
    "1:30. The Hummingbird: 0.716 kg, effective composite inertia measured in Isaac. Domain binding writes mass, inertia, thrust and moment constants and the motor lag into PhysX and reads them all back, hard-checking against the plan. Controllers use the true vehicle parameters. Domain factors: mass and force scale plus or minus 10 percent in distribution.");
  table(s, [["Quantity", "Value"], ["Total mass", "0.716 kg (base 0.68 + 4 × 0.009)"], ["Effective inertia", "diag(0.00768, 0.00768, 0.01304) kg m²"], ["Rotors", "“+” layout, arm L = 0.17 m, R = 0.10 m"], ["k_F, k_M", "8.55e-6 N/(rad/s)², 1.37e-7 N m/(rad/s)²"], ["Ω_max, T_max, T/W", "838 rad/s, 24.0 N, 3.42"], ["Hover", "Ω = 453 rad/s, throttle 0.54"], ["Motor lag", "τ_m = 8.9 ms (first order)"]],
    0.6, 1.5, 6.6, [2.3, 4.3], { fs: 12, rh: 0.5, alignLeft: true });
  card(s, 7.5, 1.5, 5.2, 2.55, "Domain binding", "Writes mass, inertia, k_F, k_M and motor lag into PhysX, reads them back and hard-checks against the plan. OmniDrones drag_coef is asserted to be 0.", TEAL, 13);
  card(s, 7.5, 4.2, 5.2, 2.6, "Randomisation (v7)", "473 vehicles: mass and force scale ±10 % in distribution, ±20 % in OOD physics; inertia ±5 %, motor time constant ±20 %, drag scales ±10 %.", AMBER, 13);
}
//@@aero
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
//@@wind
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
//@@scen
{
  const s = content("Scenarios, excitation and data types",
    "1:30. Three data types: closed loop (controller tracks a reference), open loop (rotor programs, free response for system identification) and physics validation (thesis Cases A and C). Eighteen scenario templates, seven excitation signal types. References are feasibility checked. Helix and figure-eight appear only in the trajectory OOD role.");
  table(s, [["Data type", "How", "Purpose"], ["closed_loop", "Lee / attitude controller tracks a reference", "Realistic flight"], ["open_loop", "Predefined rotor programs from hover trim, no tracking", "System identification, free response"], ["physics_validation", "Case A (no wind), Case C (mass +5 %)", "Test the nominal f_phys"]],
    0.6, 1.5, 7.2, [1.9, 3.3, 2.0], { fs: 12, rh: 0.55, alignLeft: true });
  card(s, 8.1, 1.5, 4.6, 2.2, "Scenario templates (18)", "Basic: hover, vertical, translation, line, accel/decel. Complex: circle, helix, figure eight, sinusoidal, waypoints, combined. Attitude: yaw step/sine, roll/pitch doublet, attitude combined.", TEAL, 12);
  card(s, 0.6, 3.9, 6.0, 2.9, "Excitation", "Smooth random, multisine, chirp, PRBS, yaw excitation, rotor doublets and chirps. Bounded, anchored at rest, all parameters sampled once and stored. Purpose: make the wind effect identifiable.", AMBER, 13);
  card(s, 6.7, 3.9, 6.0, 2.9, "Guarantees", "Every reference starts at rest and is C³. Plan-time feasibility check: thrust 15–85 % of max, tilt, tilt rate, yaw rate, speed, acceleration, jerk. Hover is 10 % of the closed-loop mix; helix and figure eight appear only in OOD trajectory.", PURPLE, 13);
}
//@@schema
{
  const s = content("Dataset format and reproducibility",
    "1:30. HDF5 shards with schema v2 plus a manifest, provenance and an index with a completeness flag. A dataset is complete only if every planned episode was written and every hard validation check passed. Every array except the IMU accelerometer is bitwise reproducible across processes, kill and resume, and GPUs; the accelerometer is reproducible to about 2e-5 m/s2.");
  table(s, [["File", "Content"], ["manifest.jsonl", "one fully specified plan row per episode (every sampled value and seed)"], ["shard_XXX.h5", "episodes: ground truth, IMU, rotor commands, wrench, wind, metadata"], ["index.json", "episode rows, split role, group, status, validation summary, complete flag"], ["provenance.json", "config hash, git state, Isaac and library versions, GPUs"], ["validation.json", "hard and soft findings of the validator"]],
    0.6, 1.5, 7.4, [2.2, 5.2], { fs: 12, rh: 0.55, alignLeft: true });
  card(s, 8.3, 1.5, 4.4, 2.35, "Sampling semantics", "Physics rate 200 Hz. State sampled at instants, actions held over intervals; the actions are exactly piecewise constant at the model rate.", TEAL, 13);
  card(s, 8.3, 4.0, 4.4, 2.8, "Reproducibility", "Bitwise across processes, resume and GPUs, except the IMU accelerometer (about 2e-5 m/s²). Per-episode SHA-256 checksums verified on resume and merge.", GREEN, 13);
  card(s, 0.6, 5.0, 7.4, 1.8, "Windows", "Model-ready view: decimate to the model rate, windows of history 10 plus horizon, future actions; IMU noise at load time; splits per group.", AMBER, 13);
}
//@@cv
{
  const s = content("The constant-velocity result: what it means and what it does not",
    "1:30. This is the result the panel will ask about. Constant velocity needs no learning and is better than every physics-based model on OOD magnitude and direction in the pilot. Known: B1 lacks the wind force, and B3's residual does not extrapolate beyond the training wind. Unknown: why exactly the residual fails to extrapolate. The reference exists so wins are not overstated.");
  stat(s, 0.6, 1.5, 3.8, "0.050 m", "constant velocity, OOD magnitude, 0.4 s (pilot)", GREEN);
  stat(s, 4.6, 1.5, 3.8, "0.166 m", "B3 on the same role", RED);
  stat(s, 8.6, 1.5, 4.0, "0.185 m", "B1 physics only", MUTED);
  card(s, 0.6, 3.1, 6.0, 3.7, "What is established", "B1 has no wind force (0.185 m). B3’s residual does not extrapolate beyond the training wind: the pilot training wind peaked at 3.7 m/s, the OOD peaks were 4.3 to 8.2 m/s. A ridge probe finds no linearly readable wind force in the latent (R² ≤ 0).", TEAL, 16);
  card(s, 6.7, 3.1, 6.0, 3.7, "What is not established", "Why the residual fails to extrapolate: bounded training range, short history with latent reset per window, a linear probe, or something else. Tests planned: wind-range augmentation (done in v7 data), longer history, non-linear probe, OOD-aware checkpoint selection, stronger weight decay.", AMBER, 16);
}
//@@v7veh
{
  const s = content("Pilot v7 randomisation: 473 vehicles, ID and OOD ranges",
    "1:30. Vehicle factors relative to nominal. The ID range is plus or minus 10 percent in mass and force scale; the OOD physics role uses plus or minus 20 percent. Other factors are plus or minus 5 percent for inertia shape, 20 percent for the motor time constant. The factors are nearly uncorrelated (largest absolute correlation 0.09), which was a problem in the pilot (0.59).");
  img(s, "v7_vehicle.png", 0.6, 1.4, 8.2, 5.4);
  bullets(s, ["473 distinct vehicles", "ID: mass and force scale ±10 %; OOD physics ±20 %", "Domain factors nearly uncorrelated: largest |ρ| = 0.09 (pilot 0.59)", "Colours: train, val ID, test ID, OOD magnitude, other OOD"], 9.1, 1.5, 3.6, 4.5, 14);
}
//@@v7cov
{
  const s = content("Pilot v7 coverage: wind bins, vehicle bins and scenario mix",
    "1:00. Left: episodes per mean-wind bin; calm is still the largest bin but the strata cover 0 to 8.5 m/s. Middle: ID domains per vehicle-factor bin, all populated. Right: scenario by role; closed loop dominates training, helix and figure eight appear only in the trajectory OOD role.");
  img(s, "v7_coverage.png", 0.6, 1.5, 12.1, 4.0);
  bullets(s, ["Mean wind is stratified and weighted to the upper band; calm episodes remain the largest bin", "Every vehicle-factor bin of the ID range holds 30–46 domains", "Helix and figure-eight trajectories appear only in test_ood_trajectory"], 0.6, 5.6, 12.1, 1.4, 14);
}
//@@heat
{
  const s = content("Stage-1 evaluation on all roles: position RMSE at 1 s",
    "1:30. Heat map of position RMSE at the longest horizon (20 steps, 1 s) across 13 roles and all models. Darker is worse. B1 is the darkest on OOD magnitude and physics. The B6a oracle row is not an upper bound here because of the parameter mismatch, which is the diagnosis on the next slides.");
  img(s, "heatmap_v7.png", 0.6, 1.4, 12.1, 4.7);
  caption(s, "Log colour scale; numbers are metres. B6a and B1 use nominal vehicle parameters in this evaluation.", 0.6, 6.2, 12.1);
}
//@@horizon
{
  const s = content("Error growth with rollout length: selected models on three roles",
    "1:30. Position RMSE against rollout length in seconds on test ID, OOD wind magnitude and OOD wind direction. Learned models are 5 to 10 times better than physics alone at 1 s. B3 and B4 are close in distribution; on OOD magnitude, B4 is better than B3 and all models are far from the in-distribution level.");
  img(s, "err_horizon_sel.png", 0.6, 1.4, 12.1, 3.85);
  bullets(s, ["In distribution B3 is best at every horizon; B1 grows fastest", "OOD direction is easy for every learned model; OOD magnitude is hard for all of them", "On OOD magnitude the error at 1 s is about 2.5 to 3 times the in-distribution value"], 0.6, 5.4, 12.1, 1.5, 14);
}
//@@status
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
//@@qa
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
//@@refs
{
  const s = content("Backup: references",
    "Backup slide.");
  const refs = ["D. Ha, J. Schmidhuber, “World Models,” arXiv:1803.10122, 2018.", "D. Hafner et al., “Learning Latent Dynamics for Planning from Pixels,” ICML, 2019.", "D. Hafner et al., “Dream to Control: Learning Behaviors by Latent Imagination,” ICLR, 2020.", "D. Hafner et al., “Mastering Diverse Domains through World Models,” arXiv:2301.04104, 2023.",
    "W. Gu, S. Primatesta, A. Rizzo, “Physics-informed neural network for quadrotor dynamical modeling,” Robotics and Autonomous Systems 171, 2024.", "L. Bauersfeld et al., “NeuroBEM: Hybrid Aerodynamic Quadrotor Model,” RSS, 2021.", "M. Wang et al., “PI-WAN: A Physics-Informed Wind-Adaptive Network for Quadrotor Dynamics Prediction in Unknown Environments,” IROS, 2025.",
    "M. A. Islam et al., “Physics-guided residual learning for phase-aware UAV trajectory prediction in urban environments,” Scientific Reports 16, 2026.", "P. Rao et al., “SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors,” arXiv:2606.23444, 2026.", "Z. Liu et al., “AirDreamer: Generalist Drone Navigation with World Models,” arXiv:2606.03252, 2026.",
    "S. Zheng et al., “WorldFly: A World-Model-Based Vision-Language-Action Model for UAV Navigation,” arXiv:2606.06147, 2026.", "A. H. Zidan et al., “World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications,” arXiv:2606.00133, 2026."];
  T(s, refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 5 } })), { x: 0.6, y: 1.4, w: 12.1, h: 5.5, fontSize: 12 });
}
