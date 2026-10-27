# Data Requirements for Training the World Model

Based on your proposal, the world model has several trainable components — a latent encoder, a physical state decoder, a learned residual network, and a latent transition model — each of which requires specific data streams. Below is a complete specification, organized by purpose.

---

## 1. Data Streams Required

The simulator (Isaac Sim / Isaac Lab) must log the following at every timestep \(t\), synchronized across all channels.

### 1.1 Physical State Ground Truth (targets for the state decoder and multi-step loss)

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Position | \(\mathbf p_t\) | \(\mathbb R^3\) | State prediction loss |
| Linear velocity | \(\mathbf v_t\) | \(\mathbb R^3\) | State prediction loss |
| Attitude (quaternion) | \(\mathbf q_t\) | \(\mathbb R^4\) | State prediction loss, \(d_{SO(3)}\) |
| Angular velocity | \(\boldsymbol\omega_t\) | \(\mathbb R^3\) | State prediction loss |

This is the **supervision signal** for \(\mathcal L_{\mathrm{state}}\) and \(\mathcal L_{\mathrm{multi}}\). Without it, the decoder \(D_s(h_t)\) cannot be trained.

### 1.2 Proprioceptive Observations (inputs to the encoder \(E_P\))

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Accelerometer | \(a_t^{\mathrm{IMU}}\) | \(\mathbb R^3\) | Encoder input |
| Gyroscope | \(\omega_t^{\mathrm{IMU}}\) | \(\mathbb R^3\) | Encoder input |

These are the **core evidence** in the minimum viable thesis. If your simulator produces ideal IMU readings, you may want to optionally add realistic noise/bias to force the encoder to learn robust representations.

### 1.3 Control Inputs

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Motor commands / RPM | \(\omega_1, \omega_2, \omega_3, \omega_4\) | \(\mathbb R^4\) | Encoder input and physics model input |
| Or thrust + torque | \(T, \boldsymbol\tau\) | \(\mathbb R^4\) | Equivalent alternative |

The choice of representation should match what \(f_{\mathrm{phys}}(\cdot)\) expects. If \(f_{\mathrm{phys}}\) uses motor RPMs to compute thrust, you need RPMs; if it uses thrust/torque directly, you can log those.

### 1.4 Wind and Disturbance Data (for OOD evaluation and oracle only)

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| Wind velocity field | \(\mathbf w_t\) | \(\mathbb R^3\) | OOD taxonomy, oracle baseline |
| Disturbance force | \(\mathbf f_{\mathrm{ext},t}\) | \(\mathbb R^3\) | Oracle baseline, diagnostic \(E_d\) |
| Disturbance torque | \(\boldsymbol\tau_{\mathrm{ext},t}\) | \(\mathbb R^3\) | Oracle baseline |
| Turbulence parameters | \(\sigma_u, \sigma_v, \sigma_w, L_u, L_v, L_w\) | scalars | Reproducibility, OOD definition |

**Important**: The learned residual \(d_\theta\) is **not** trained against \(\mathbf w_t\). Wind ground truth is used only for (a) the oracle upper bound, (b) the diagnostic \(E_d\), and (c) defining the OOD taxonomy. Do **not** add a loss that forces \(d_\theta \approx \mathbf w\).

### 1.5 Visual Observations (extension only)

| Quantity | Symbol | Dimension | Purpose |
|---|---|---|---|
| RGB image | \(I_t\) | \(H \times W \times 3\) | Visual encoder \(E_V\) (extension) |
| Depth image | \(D_t\) | \(H \times W\) | Optional auxiliary |

Only needed if you pursue the multimodal extension (M4). The minimum viable thesis does not require vision.

---

## 2. Data Organization

### 2.1 Trajectory Format

Data should be stored as **trajectories**, not independent tuples:

\[
\tau_i = \big\{ (o_t,\ s_t,\ u_t,\ \mathbf w_t,\ \mathbf f_{\mathrm{ext},t}) \big\}_{t=0}^{T_i}
\]

where \(T_i\) is the trajectory length. Multi-step losses require contiguous windows of length \(T_{\mathrm{burn}} + H\), so per-timestep shuffling is not acceptable.

### 2.2 Suggested Dataset Structure

```
dataset/
├── train/
│   ├── trajectory_0000.npz   # t, p, v, q, ω, a_imu, ω_imu, u, w, f_ext
│   ├── trajectory_0001.npz
│   └── ...
├── val/                       # same format, drawn from training wind distribution
│   └── ...
└── test/
    ├── magnitude_extrap/      # OOD regime 1
    ├── directional_extrap/    # OOD regime 2
    ├── temporal_extrap/       # OOD regime 3
    ├── gust_extrap/           # OOD regime 4
    └── physics_mismatch/      # OOD regime 5
```

Each `.npz` stores arrays of shape `(T, dim)`.

### 2.3 Sequence Lengths

- **Burn-in window** \(T_{\mathrm{burn}}\): typically 5–10 steps to initialize \(h\).
- **Prediction horizon** \(H\): 5–20 steps for training; longer (up to 100) for evaluation only.

The trajectory length \(T_i\) should be at least \(T_{\mathrm{burn}} + H_{\max}\) plus a margin.

---

## 3. Data Volume

For an M.Tech thesis, a realistic starting point:

| Wind regime | Trajectories | Steps per trajectory | Approx. total steps |
|---|---|---|---|
| Training (moderate wind) | 200–500 | 1000–2000 | \(2\times10^5\) to \(10^6\) |
| Validation | 50–100 | 1000–2000 | \(5\times10^4\) to \(2\times10^5\) |
| OOD test (each of 5 regimes) | 50–100 | 1000–2000 | \(2.5\times10^5\) to \(10^6\) |

For the **sample-efficiency study** (§16.1 in your proposal), you also need the same training pool at \(10\%, 25\%, 50\%, 100\%\) — which is just a subset of the same data, so no extra generation is needed.

If vision is included, storage grows dramatically (roughly 1–10 MB per RGB frame at \(256\times256\)), so consider storing images only for a subset of trajectories in the extension phase.

---

## 4. Data Diversity Requirements

To support the OOD taxonomy, the training data must vary systematically:

| Dimension | Training range | Held out for OOD |
|---|---|---|
| Mean wind magnitude \(|\mathbf w|\) | \(0\)–\(4\) m/s | \(>4\) m/s |
| Wind direction | All horizontal, limited vertical | Excluded directions |
| Turbulence intensity \(\sigma\) | Low–moderate | High |
| Gust structure | Smooth | Sudden transients |
| Physics parameters \(m, J, C_D\) | Nominal | Perturbed (10–20%) |

The simulator must expose these as controllable parameters so that trajectories can be generated under each regime. This is why Isaac Sim / Isaac Lab is preferred over real flight data — the OOD conditions are not just rare, they are *counterfactually controllable*.

---

## 5. What Each Training Component Needs

| Component | Data required |
|---|---|
| Encoder \(E_P\) | \(a_t^{\mathrm{IMU}}, \omega_t^{\mathrm{IMU}}, u_t\) |
| Encoder \(E_V\) (extension) | \(I_t\) |
| Decoder \(D_s\) | Ground-truth \(s_t = [\mathbf p, \mathbf v, \mathbf q, \boldsymbol\omega]\) |
| Physics \(f_{\mathrm{phys}}\) | \(s_t, u_t\) (no training; parameterized analytically) |
| Residual \(d_\theta\) | \(h_t, u_t\) (learned implicitly via state loss) |
| Latent transition \(\latent\) | \((h_t, u_t, \tilde s_{t+1})\) sequence |

The residual and the latent transition are **not trained with direct supervision** — they are trained through the state prediction and multi-step losses, which require ground-truth \(s_{t+k}\).

---

## 6. Data the Physics Model Validation Requires

For §11 (Physics Model Validation), you need three separate datasets:

- **Case A (no wind)**: same trajectories with \(\mathbf w = 0\).
- **Case B (known wind)**: same trajectories with controlled wind and logged \(\mathbf w_t, \mathbf f_{\mathrm{ext},t}\).
- **Case C (perturbed parameters)**: the simulator's physics parameters are changed (e.g., mass \(m_{\mathrm{sim}} = 1.1\,m_{\mathrm{nominal}}\)), and the nominal \(f_{\mathrm{phys}}\) is evaluated against these.

Each case is generated by a distinct simulator configuration and does not require the neural components to be trained.

---

## 7. Practical Recommendations

1. **Log everything at the highest rate the model uses.** Even if you subsample later, having IMU at 200 Hz and state at 200 Hz simplifies alignment.
2. **Store time explicitly.** A `t` array avoids alignment bugs when some channels are logged at different rates.
3. **Use `.npz` or HDF5, not CSV.** Trajectories of \(10^6\) steps with 20+ fields become unmanageable as text.
4. **Separate the OOD generation from training generation.** Generate OOD data with a distinct seed and label it explicitly.
5. **Record simulator configuration metadata.** Each trajectory's file should carry the wind parameters, physics parameters, and initial conditions used to generate it. This is essential for reproducibility and for defining the OOD taxonomy precisely.
6. **Do not include ground-truth wind as a model input.** Wind is only for evaluation and oracle. Including it would leak the disturbance and invalidate the OOD claim.

---

## 8. Summary

To train the world model, you need, at minimum:

\[
\boxed{
\text{IMU} + \text{control} + \text{ground-truth state} + \text{trajectory continuity}
}
\]

and for evaluation:

\[
\boxed{
\text{wind ground truth} + \text{disturbance force} + \text{OOD metadata}
}
\]

Everything else — vision, uncertainty, MPC — is an extension. The minimum viable thesis requires only proprioception, control, ground-truth state, and controlled wind conditions across a well-defined training/OOD split.