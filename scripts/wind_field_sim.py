#!/usr/bin/env python3
"""
wind_field_sim.py — Interactive time-varying wind field visualizer with Dash.

Model:
    u(p, t) = u0 + G·(p − p_ref) + profile(t) + gust_overlay(t)
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, ctx, dcc, html, Input, Output, State


# ═══════════════════════════════════════════════════════════════════════
# 1. Wind Model & Profiles
# ═══════════════════════════════════════════════════════════════════════

def _smooth_noise(x: np.ndarray, seed: int, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.3, 2.5, size=n)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
    amps = rng.normal(0.0, 1.0, size=n)
    denom = np.sqrt(np.sum(amps ** 2) / 2.0)
    if denom > 1e-12:
        amps = amps / denom
    return sum(
        a * np.sin(2.0 * np.pi * f * x + p)
        for a, f, p in zip(amps, freqs, phases)
    )


def _direction(d: Any) -> np.ndarray:
    v = np.asarray(d, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _profile_term(pts: np.ndarray, t: float, P: dict) -> np.ndarray:
    n = len(pts)
    kind = P["type"]
    zero = np.zeros((n, 3))

    if kind == "none":
        return zero

    if kind == "constant":
        d = _direction([P["c_dir_x"], P["c_dir_y"], P["c_dir_z"]])
        return np.tile(d * float(P["c_magnitude"]), (n, 1))

    if kind == "step":
        if t < float(P["s_step_time"]):
            return zero
        d = _direction([P["s_dir_x"], P["s_dir_y"], P["s_dir_z"]])
        return np.tile(d * float(P["s_magnitude"]), (n, 1))

    if kind == "sinusoidal":
        d = _direction([P["sin_dir_x"], P["sin_dir_y"], P["sin_dir_z"]])
        scalar = float(P["sin_offset"]) + float(P["sin_amplitude"]) * np.sin(
            2.0 * np.pi * float(P["sin_freq_hz"]) * t + float(P["sin_phase_rad"])
        )
        return np.tile(d * scalar, (n, 1))

    if kind == "gust":
        start = float(P["g_start"])
        if not (start <= t < start + float(P["g_duration"])):
            return zero
        d = _direction([P["g_dir_x"], P["g_dir_y"], P["g_dir_z"]])
        return np.tile(d * float(P["g_magnitude"]), (n, 1))

    if kind == "dryden":
        tau = max(float(P["d_tau_s"]), 0.05)
        scale = 1.0 / tau
        x = pts[:, 0] * scale + t * scale
        y = pts[:, 1] * scale + t * scale
        z = pts[:, 2] * scale + t * scale
        ux = _smooth_noise(x + 0.3 * y, seed=101)
        uy = _smooth_noise(y + 0.3 * z, seed=202)
        uz = _smooth_noise(z + 0.3 * x, seed=303)
        return float(P["d_sigma"]) * np.stack([ux, uy, uz], axis=1)

    if kind == "random":
        rng = np.random.default_rng(7)
        return float(P["r_sigma"]) * rng.standard_normal((n, 3))

    return zero


def _spatial_term(pts: np.ndarray, P: dict) -> np.ndarray:
    if not P["spatial_enabled"]:
        return np.zeros_like(pts, dtype=np.float64)
    p_ref = np.array([P["ref_x"], P["ref_y"], P["ref_z"]], dtype=np.float64)
    dp = pts - p_ref
    if P["gradient_mode"] == "diagonal":
        g = np.array([P["diag_x"], P["diag_y"], P["diag_z"]], dtype=np.float64)
        return dp * g
    G = np.array(
        [
            [P["M00"], P["M01"], P["M02"]],
            [P["M10"], P["M11"], P["M12"]],
            [P["M20"], P["M21"], P["M22"]],
        ],
        dtype=np.float64,
    )
    return (G @ dp.T).T


def _overlay_gust(pts: np.ndarray, t: float, P: dict) -> np.ndarray:
    n = len(pts)
    if not P["gust_enabled"]:
        return np.zeros((n, 3))
    start = float(P["go_start"])
    if not (start <= t < start + float(P["go_duration"])):
        return np.zeros((n, 3))
    d = _direction([P["go_dir_x"], P["go_dir_y"], P["go_dir_z"]])
    return np.tile(d * float(P["go_amplitude"]), (n, 1))


def _base_wind(pts: np.ndarray, P: dict) -> np.ndarray:
    n = len(pts)
    return np.tile(
        np.array([P["u0_x"], P["u0_y"], P["u0_z"]], dtype=np.float64), (n, 1)
    )


def wind_velocity(pts: np.ndarray, t: float, P: dict) -> np.ndarray:
    u = _base_wind(pts, P)
    u = u + _spatial_term(pts, P)
    u = u + _profile_term(pts, t, P)
    u = u + _overlay_gust(pts, t, P)
    return u


def isolate_component(pts: np.ndarray, t: float, P: dict, mode: str) -> np.ndarray:
    n = len(pts)
    if mode == "base":
        return _base_wind(pts, P)
    if mode == "gradient":
        return _spatial_term(pts, P)
    if mode == "profile":
        return _profile_term(pts, t, P)
    if mode == "gust":
        return _overlay_gust(pts, t, P)
    if mode == "delta":
        return wind_velocity(pts, t, P) - wind_velocity(pts, 0.0, P)
    # default "total"
    return wind_velocity(pts, t, P)


# ═══════════════════════════════════════════════════════════════════════
# 2. Grid & Figure Generation
# ═══════════════════════════════════════════════════════════════════════

def make_grid(nx: int = 11, ny: int = 11, nz: int = 5,
              extent: float = 5.0, z_max: float = 4.0) -> np.ndarray:
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    z = np.linspace(0.0, z_max, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


_VIRIDIS_HEX = ["#440154", "#414487", "#2a788e",
                "#22a884", "#7ad151", "#fde725"]


def build_arrow_traces(pts: np.ndarray, u: np.ndarray, mag: np.ndarray,
                       cmax: float, arrow_scale: float, line_width: int,
                       head_fraction: float, n_buckets: int = 6) -> list:
    """Split the shaft lines into colour buckets + one cone trace for heads."""
    scaled = u * arrow_scale
    shaft_end = pts + (1.0 - head_fraction) * scaled

    mag_norm = np.clip(mag / max(cmax, 1e-6), 0.0, 1.0)
    bucket_idx = np.minimum((mag_norm * n_buckets).astype(int), n_buckets - 1)

    traces: list = []
    for b in range(n_buckets):
        mask = bucket_idx == b
        if not np.any(mask):
            continue
        p_b = pts[mask]
        e_b = shaft_end[mask]
        m = len(p_b)

        sx: list = []
        sy: list = []
        sz: list = []
        for i in range(m):
            sx.extend([float(p_b[i, 0]), float(e_b[i, 0]), None])
            sy.extend([float(p_b[i, 1]), float(e_b[i, 1]), None])
            sz.extend([float(p_b[i, 2]), float(e_b[i, 2]), None])

        traces.append(go.Scatter3d(
            x=sx, y=sy, z=sz,
            mode="lines",
            line=dict(color=_VIRIDIS_HEX[b], width=int(line_width)),
            hoverinfo="skip",
            showlegend=False,
        ))

    traces.append(go.Cone(
        x=shaft_end[:, 0].tolist(),
        y=shaft_end[:, 1].tolist(),
        z=shaft_end[:, 2].tolist(),
        u=scaled[:, 0].tolist(),
        v=scaled[:, 1].tolist(),
        w=scaled[:, 2].tolist(),
        colorscale="Viridis",
        cmin=0.0,
        cmax=float(cmax),
        sizemode="scaled",
        sizeref=float(head_fraction),
        anchor="tail",
        showscale=True,
        colorbar=dict(title="|u| m/s", thickness=14, len=0.7, x=1.02),
        hovertemplate=(
            "p=(%{x:.1f},%{y:.1f},%{z:.1f})<br>"
            "u=(%{u:.2f},%{v:.2f},%{w:.2f}) m/s<extra></extra>"
        ),
    ))
    return traces


def build_figure(P: dict, t: float, view_mode: str = "total") -> go.Figure:
    pts = make_grid()

    grid_markers = go.Scatter3d(
        x=pts[:, 0].tolist(),
        y=pts[:, 1].tolist(),
        z=pts[:, 2].tolist(),
        mode="markers",
        marker=dict(size=2, color="#c8c8c8"),
        hoverinfo="skip",
        showlegend=False,
    )

    u = isolate_component(pts, t, P, view_mode)
    mag = np.linalg.norm(u, axis=1)
    cmax = max(float(np.percentile(mag, 95)), 0.5) if mag.size else 0.5

    arrow_traces = build_arrow_traces(
        pts, u, mag, cmax,
        arrow_scale=float(P["arrow_scale"]),
        line_width=int(P["arrow_line_width"]),
        head_fraction=float(P["arrow_head_fraction"]),
    )

    ground = go.Surface(
        x=[[-5.0, 5.0], [-5.0, 5.0]],
        y=[[-5.0, -5.0], [5.0, 5.0]],
        z=[[0.0, 0.0], [0.0, 0.0]],
        colorscale=[[0, "#e8e8e8"], [1, "#e8e8e8"]],
        showscale=False,
        opacity=0.35,
        hoverinfo="skip",
    )

    ref_marker = go.Scatter3d(
        x=[float(P["ref_x"])],
        y=[float(P["ref_y"])],
        z=[float(P["ref_z"])],
        mode="markers+text",
        marker=dict(size=7, color="#e74c3c", symbol="x"),
        text=["p_ref"],
        textposition="top center",
        textfont=dict(size=10, color="#e74c3c"),
        hoverinfo="skip",
        showlegend=False,
    )

    label_map = {
        "total": "Total wind",
        "base": "Base u₀",
        "gradient": "Gradient G·(p−p₀)",
        "profile": f"Profile '{P['type']}'",
        "delta": "Δu = u(p,t) − u(p,0)",
        "gust": "Gust overlay",
    }
    label = label_map.get(view_mode, "Total wind")
    peak = float(mag.max()) if mag.size else 0.0

    fig = go.Figure(data=[ground, grid_markers, *arrow_traces, ref_marker])
    fig.update_layout(
        title=dict(
            text=f"{label}   |   t = {t:.2f} s   |   max|u| = {peak:.2f} m/s",
            x=0.02,
            xanchor="left",
            font=dict(size=13),
        ),
        scene=dict(
            xaxis=dict(title="X (m)", range=[-6, 6], backgroundcolor="#f8f8f8"),
            yaxis=dict(title="Y (m)", range=[-6, 6], backgroundcolor="#f8f8f8"),
            zaxis=dict(title="Z (m)", range=[-0.5, 4.5], backgroundcolor="#f8f8f8"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="white",
        uirevision="keep-camera",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. Defaults & Presets
# ═══════════════════════════════════════════════════════════════════════

DEFAULTS: dict = {
    # Base wind
    "u0_x": 2.0, "u0_y": 0.0, "u0_z": 0.0,
    # Profile
    "type": "none",
    "c_magnitude": 1.0, "c_dir_x": 0.0, "c_dir_y": 1.0, "c_dir_z": 0.0,
    "s_magnitude": 2.0, "s_dir_x": 0.0, "s_dir_y": 1.0, "s_dir_z": 0.0,
    "s_step_time": 5.0,
    "sin_amplitude": 1.0, "sin_dir_x": 0.0, "sin_dir_y": 1.0, "sin_dir_z": 0.0,
    "sin_freq_hz": 0.2, "sin_offset": 0.0, "sin_phase_rad": 0.0,
    "g_magnitude": 2.0, "g_dir_x": 0.0, "g_dir_y": 1.0, "g_dir_z": 0.0,
    "g_start": 3.0, "g_duration": 1.0,
    "d_sigma": 0.3, "d_tau_s": 1.0,
    "r_sigma": 0.5,
    # Gust overlay
    "gust_enabled": False,
    "go_start": 3.0, "go_duration": 2.0, "go_amplitude": 1.5,
    "go_dir_x": 1.0, "go_dir_y": 0.0, "go_dir_z": 0.0,
    # Spatial gradient
    "spatial_enabled": True,
    "gradient_mode": "matrix",
    "diag_x": 0.5, "diag_y": 0.0, "diag_z": 0.0,
    "M00": 0.5, "M01": 0.0, "M02": 0.0,
    "M10": 0.2, "M11": 0.3, "M12": 0.0,
    "M20": 0.0, "M21": 0.1, "M22": 0.0,
    "ref_x": 0.0, "ref_y": 0.0, "ref_z": 1.5,
    # Time
    "time": 0.0, "max_time": 10.0,
    # View
    "view_mode": "total",
    "arrow_scale": 1.2, "arrow_line_width": 3.0, "arrow_head_fraction": 0.30,
}

PRESETS: dict = {
    "Default (as in config)": DEFAULTS,
    "Calm air": dict(DEFAULTS, u0_x=0.2, u0_y=0.1, spatial_enabled=False),
    "Steady + shear": dict(DEFAULTS, u0_x=2.0),
    "Sinusoidal": dict(DEFAULTS, type="sinusoidal", sin_amplitude=2.0, sin_freq_hz=0.3),
    "Gusty": dict(DEFAULTS, type="gust", g_magnitude=4.0, g_start=2.0, g_duration=1.5),
    "Dryden": dict(DEFAULTS, type="dryden", d_sigma=1.0, d_tau_s=0.8),
    "Combined storm": dict(
        DEFAULTS, u0_x=2.5, u0_y=1.0,
        type="dryden", d_sigma=0.8, d_tau_s=0.6,
        gust_enabled=True, go_amplitude=2.5,
    ),
    "Pure gradient": dict(
        DEFAULTS, u0_x=0.0, u0_y=0.0, u0_z=0.0, view_mode="gradient"
    ),
}

NUMERIC_SLIDER_IDS: list = [
    "u0_x", "u0_y", "u0_z",
    "c_magnitude", "c_dir_x", "c_dir_y", "c_dir_z",
    "s_magnitude", "s_dir_x", "s_dir_y", "s_dir_z", "s_step_time",
    "sin_amplitude", "sin_dir_x", "sin_dir_y", "sin_dir_z",
    "sin_freq_hz", "sin_offset", "sin_phase_rad",
    "g_magnitude", "g_dir_x", "g_dir_y", "g_dir_z", "g_start", "g_duration",
    "d_sigma", "d_tau_s",
    "r_sigma",
    "go_start", "go_duration", "go_amplitude",
    "go_dir_x", "go_dir_y", "go_dir_z",
    "diag_x", "diag_y", "diag_z",
    "M00", "M01", "M02", "M10", "M11", "M12", "M20", "M21", "M22",
    "ref_x", "ref_y", "ref_z",
    "time", "max_time",
    "arrow_scale", "arrow_line_width", "arrow_head_fraction",
]

# These are rendered as dcc.Input (no readout span under them).
NUM_INPUT_IDS: set = {
    "M00", "M01", "M02", "M10", "M11", "M12",
    "M20", "M21", "M22", "ref_x", "ref_y", "ref_z",
}
READOUT_IDS: list = [s for s in NUMERIC_SLIDER_IDS if s not in NUM_INPUT_IDS]

PARAM_SLIDER_IDS: list = [s for s in NUMERIC_SLIDER_IDS
                          if s not in ("time", "max_time")]

PLAY_SPEEDS: dict = {
    "0.25×": 0.0125, "0.5×": 0.025, "1×": 0.05,
    "2×": 0.10, "4×": 0.20,
}

PROFILE_KEYS: list = ["none", "constant", "step", "sinusoidal",
                      "gust", "dryden", "random"]


# ═══════════════════════════════════════════════════════════════════════
# 4. UI Layout Helpers
# ═══════════════════════════════════════════════════════════════════════

def slider(sid: str, label: str, mn: float, mx: float, val: float,
           step: float = 0.05) -> html.Div:
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "11px", "color": "#333"}),
            html.Span(id=f"val-{sid}", style={
                "float": "right", "fontSize": "11px",
                "fontWeight": "bold", "color": "#1f77b4",
            }),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "marginBottom": "1px"}),
        dcc.Slider(id=sid, min=mn, max=mx, value=val, step=step,
                   marks=None, updatemode="drag",
                   tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"marginBottom": "8px"})


def num_input(sid: str, val: float, step: float = 0.05,
              width: str = "58px") -> dcc.Input:
    return dcc.Input(id=sid, type="number", value=val, step=step,
                     debounce=True,
                     style={"width": width, "fontSize": "11px",
                            "padding": "2px 4px", "border": "1px solid #ccc",
                            "borderRadius": "2px"})


def vec3(prefix: str, label: str, defaults: dict) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#555",
                               "marginTop": "4px", "marginBottom": "2px"}),
        slider(f"{prefix}_x", "x", -3, 3, defaults[f"{prefix}_x"], 0.1),
        slider(f"{prefix}_y", "y", -3, 3, defaults[f"{prefix}_y"], 0.1),
        slider(f"{prefix}_z", "z", -3, 3, defaults[f"{prefix}_z"], 0.1),
    ])


def section(title: str, children: list) -> html.Div:
    return html.Div([
        html.H4(title, style={
            "margin": "10px 0 5px 0", "fontSize": "11px",
            "fontWeight": "bold", "color": "#555",
            "textTransform": "uppercase", "letterSpacing": "0.6px",
            "borderBottom": "1px solid #ddd", "paddingBottom": "3px",
        }),
        *children,
    ])


def toggle(sid: str, label: str, value: bool = False) -> dcc.Checklist:
    return dcc.Checklist(
        id=sid,
        options=[{"label": f"  {label}", "value": "on"}],
        value=["on"] if value else [],
        style={"fontSize": "12px", "margin": "4px 0 6px 0"},
        inputStyle={"marginRight": "6px"},
    )


def _panel(children, initial_visible: bool = False, extra_style: dict | None = None):
    """Wrapper that ALWAYS contains style=display plus any extra styling,
    so the visibility toggle doesn't wipe out user styling."""
    style = {"display": "block" if initial_visible else "none"}
    if extra_style:
        style.update(extra_style)
    return html.Div(children, style=style)


# ── Profile sub-panels ──────────────────────────────────────────────────
PROFILE_PANELS: dict = {
    "none": _panel(
        "Steady wind + spatial gradient only.",
        initial_visible=True,
        extra_style={"fontSize": "11px", "color": "#999",
                     "fontStyle": "italic", "padding": "4px 0"},
    ),
    "constant": _panel([
        slider("c_magnitude", "Magnitude (m/s)", 0, 5, DEFAULTS["c_magnitude"]),
        vec3("c_dir", "Direction (normalised)", DEFAULTS),
    ]),
    "step": _panel([
        slider("s_magnitude", "Magnitude (m/s)", 0, 5, DEFAULTS["s_magnitude"]),
        vec3("s_dir", "Direction (normalised)", DEFAULTS),
        slider("s_step_time", "Step time (s)", 0, 20, DEFAULTS["s_step_time"]),
    ]),
    "sinusoidal": _panel([
        slider("sin_amplitude", "Amplitude (m/s)", 0, 5, DEFAULTS["sin_amplitude"]),
        vec3("sin_dir", "Direction (normalised)", DEFAULTS),
        slider("sin_freq_hz", "Frequency (Hz)", 0, 2, DEFAULTS["sin_freq_hz"], 0.02),
        slider("sin_offset", "Offset (m/s)", -2, 2, DEFAULTS["sin_offset"]),
        slider("sin_phase_rad", "Phase (rad)", 0, 6.28, DEFAULTS["sin_phase_rad"], 0.05),
    ]),
    "gust": _panel([
        slider("g_magnitude", "Magnitude (m/s)", 0, 8, DEFAULTS["g_magnitude"]),
        vec3("g_dir", "Direction (normalised)", DEFAULTS),
        slider("g_start", "Start time (s)", 0, 20, DEFAULTS["g_start"], 0.1),
        slider("g_duration", "Duration (s)", 0.1, 5, DEFAULTS["g_duration"], 0.1),
    ]),
    "dryden": _panel([
        slider("d_sigma", "Intensity σ (m/s)", 0, 3, DEFAULTS["d_sigma"], 0.02),
        slider("d_tau_s", "Correlation τ (s)", 0.05, 3, DEFAULTS["d_tau_s"], 0.05),
    ]),
    "random": _panel([
        slider("r_sigma", "Spatial σ (m/s)", 0, 3, DEFAULTS["r_sigma"], 0.02),
    ]),
}
# Attach the required ids to each panel by index order.
for _k in PROFILE_KEYS:
    PROFILE_PANELS[_k].id = f"panel-profile-{_k}"

GUST_OVERLAY_PANEL = _panel([
    slider("go_amplitude", "Amplitude (m/s)", 0, 5, DEFAULTS["go_amplitude"]),
    vec3("go_dir", "Direction (normalised)", DEFAULTS),
    slider("go_start", "Start time (s)", 0, 20, DEFAULTS["go_start"], 0.1),
    slider("go_duration", "Duration (s)", 0.1, 5, DEFAULTS["go_duration"], 0.1),
])
GUST_OVERLAY_PANEL.id = "panel-gust-overlay"

DIAG_PANEL = _panel([
    slider("diag_x", "∂uₓ/∂x (1/s)", -1, 1, DEFAULTS["diag_x"], 0.02),
    slider("diag_y", "∂uᵧ/∂y (1/s)", -1, 1, DEFAULTS["diag_y"], 0.02),
    slider("diag_z", "∂uᵤ/∂z (1/s)", -1, 1, DEFAULTS["diag_z"], 0.02),
])
DIAG_PANEL.id = "panel-spatial-diag"

MATRIX_PANEL = _panel([
    html.Div("gradient_matrix_per_m (rows = wind, cols = position)",
             style={"fontSize": "10px", "color": "#666", "marginBottom": "4px"}),
    *[
        html.Div([
            html.Span(f"∂u{axis}/∂", style={
                "fontSize": "10px", "color": "#666",
                "display": "inline-block", "width": "28px",
            }),
            num_input(f"M{i}0", DEFAULTS[f"M{i}0"]),
            html.Span(" ", style={"display": "inline-block", "width": "3px"}),
            num_input(f"M{i}1", DEFAULTS[f"M{i}1"]),
            html.Span(" ", style={"display": "inline-block", "width": "3px"}),
            num_input(f"M{i}2", DEFAULTS[f"M{i}2"]),
        ], style={"marginBottom": "3px"})
        for i, axis in enumerate(["ₓ", "ᵧ", "ᵤ"])
    ],
])
MATRIX_PANEL.id = "panel-spatial-matrix"

REF_PANEL = _panel([
    html.Div("reference_position_xyz (m)",
             style={"fontSize": "10px", "color": "#666",
                    "marginTop": "6px", "marginBottom": "3px"}),
    html.Div([
        html.Span("x", style={"fontSize": "10px", "color": "#666",
                              "width": "14px", "display": "inline-block"}),
        num_input("ref_x", DEFAULTS["ref_x"]),
        html.Span("  y", style={"fontSize": "10px", "color": "#666",
                                "width": "18px", "display": "inline-block"}),
        num_input("ref_y", DEFAULTS["ref_y"]),
        html.Span("  z", style={"fontSize": "10px", "color": "#666",
                                "width": "18px", "display": "inline-block"}),
        num_input("ref_z", DEFAULTS["ref_z"]),
    ]),
])
REF_PANEL.id = "panel-spatial-ref"

SPATIAL_MODE_PANEL = html.Div([
    html.Label("Gradient mode", style={"fontSize": "10px"}),
    dcc.Dropdown(id="gradient_mode",
                 options=[{"label": "diagonal", "value": "diagonal"},
                          {"label": "matrix", "value": "matrix"}],
                 value=DEFAULTS["gradient_mode"], clearable=False,
                 style={"fontSize": "11px", "marginBottom": "6px"}),
], id="panel-spatial-mode", style={"display": "block"})


# ═══════════════════════════════════════════════════════════════════════
# 5. App Layout
# ═══════════════════════════════════════════════════════════════════════

app = Dash(__name__, title="Wind Field Simulator",
           suppress_callback_exceptions=True)

sidebar = html.Div([
    html.H2("Wind Field Simulator",
            style={"margin": "0 0 2px 0", "fontSize": "15px"}),
    html.Div("u(p,t) = u₀ + G(p−p₀) + profile(t) + gust_overlay(t)",
             style={"fontSize": "10px", "color": "#666",
                    "marginBottom": "8px", "fontStyle": "italic"}),

    html.Div([
        html.Label("Preset", style={"fontSize": "10px"}),
        dcc.Dropdown(id="preset",
                     options=[{"label": k, "value": k} for k in PRESETS],
                     value="Default (as in config)", clearable=False,
                     style={"fontSize": "11px", "marginBottom": "6px"}),
        html.Button("▶ Play", id="play-btn", n_clicks=0, style={
            "width": "100%", "padding": "5px", "border": "1px solid #ccc",
            "backgroundColor": "#f4f4f4", "cursor": "pointer",
            "borderRadius": "3px", "fontSize": "11px",
        }),
        html.Div([
            html.Label("Speed", style={"fontSize": "10px"}),
            dcc.Dropdown(id="play_speed",
                         options=[{"label": k, "value": k} for k in PLAY_SPEEDS],
                         value="1×", clearable=False,
                         style={"fontSize": "11px"}),
        ], style={"marginBottom": "6px"}),
    ]),

    section("View mode", [
        dcc.Dropdown(id="view_mode",
                     options=[{"label": v, "value": v} for v in
                              ("total", "base", "gradient", "profile",
                               "gust", "delta")],
                     value=DEFAULTS["view_mode"], clearable=False,
                     style={"fontSize": "11px"}),
    ]),

    section("Arrow appearance", [
        slider("arrow_scale", "Length scale", 0.1, 5.0, 1.2, 0.1),
        slider("arrow_line_width", "Shaft thickness (px)", 1, 10, 3, 1),
        slider("arrow_head_fraction", "Head fraction of length",
               0.05, 0.60, 0.30, 0.02),
    ]),

    section("Steady wind  u₀ (m/s)", [
        slider("u0_x", "u₀ₓ", -5, 5, DEFAULTS["u0_x"]),
        slider("u0_y", "u₀ᵧ", -5, 5, DEFAULTS["u0_y"]),
        slider("u0_z", "u₀ᵤ", -3, 3, DEFAULTS["u0_z"]),
    ]),

    section("Profile", [
        dcc.Dropdown(id="type",
                     options=[{"label": t, "value": t} for t in PROFILE_KEYS],
                     value=DEFAULTS["type"], clearable=False,
                     style={"fontSize": "11px", "marginBottom": "6px"}),
        *PROFILE_PANELS.values(),
    ]),

    section("Gust overlay", [
        toggle("gust_enabled", "Enable overlay", DEFAULTS["gust_enabled"]),
        GUST_OVERLAY_PANEL,
    ]),

    section("Spatial gradient", [
        toggle("spatial_enabled", "Enable spatial gradient",
               DEFAULTS["spatial_enabled"]),
        SPATIAL_MODE_PANEL,
        DIAG_PANEL,
        MATRIX_PANEL,
        REF_PANEL,
    ]),

    section("Time", [
        slider("max_time", "Total duration (s)", 1, 60,
               DEFAULTS["max_time"], 1.0),
        slider("time", "t (s)", 0, 60, DEFAULTS["time"], 0.05),
    ]),
], style={
    "width": "360px", "height": "100vh", "padding": "10px 12px",
    "backgroundColor": "#fafafa", "position": "fixed",
    "left": 0, "top": 0, "overflowY": "auto",
    "borderRight": "1px solid #ddd", "fontFamily": "Arial",
    "boxSizing": "border-box",
})

main = html.Div([
    dcc.Graph(
        id="wind-graph",
        figure={"data": [], "layout": {}},
        style={"height": "calc(100vh - 44px)"},
        config={"displayModeBar": True, "displaylogo": False},
    ),
    html.Div(
        "Loading wind field…",
        id="loading-overlay",
        style={"position": "absolute", "top": "50%", "left": "50%",
               "transform": "translate(-50%, -50%)",
               "fontSize": "16px", "color": "#888",
               "pointerEvents": "none"},
    ),
], style={"marginLeft": "360px", "marginTop": "44px", "position": "relative"})

status_bar = html.Div(
    [
        html.Span("t = ", style={"fontSize": "14px", "color": "#666"}),
        html.Span("0.00 / 10.00 s", id="time-readout",
                  style={"fontSize": "16px", "fontWeight": "bold",
                         "color": "#1f77b4"}),
        html.Span("   |   ", style={"color": "#ccc"}),
        html.Span("STOPPED", id="play-status",
                  style={"fontSize": "12px", "fontWeight": "bold",
                         "color": "#d62728"}),
        html.Span("   |   ", style={"color": "#ccc"}),
        html.Span("gust: OFF", id="gust-status",
                  style={"fontSize": "12px", "fontWeight": "bold",
                         "color": "#999"}),
        html.Span("   |   ticks: ", style={"color": "#999",
                                           "fontSize": "11px"}),
        html.Span("0", id="tick-count",
                  style={"fontSize": "11px", "color": "#999"}),
    ],
    style={"position": "fixed", "top": "8px", "left": "376px",
           "zIndex": 1000, "padding": "6px 12px",
           "backgroundColor": "rgba(255,255,255,0.95)",
           "borderRadius": "4px", "border": "1px solid #ddd",
           "fontFamily": "Arial", "display": "flex", "alignItems": "center"},
)

app.layout = html.Div([
    sidebar,
    status_bar,
    main,
    dcc.Interval(id="interval", interval=50, disabled=True, n_intervals=0),
], style={"fontFamily": "Arial"})


# ═══════════════════════════════════════════════════════════════════════
# 6. Callbacks
# ═══════════════════════════════════════════════════════════════════════

def _display(visible: bool) -> dict:
    return {"display": "block"} if visible else {"display": "none"}


# ── Panel visibility ────────────────────────────────────────────────────
@app.callback(
    [Output(f"panel-profile-{k}", "style") for k in PROFILE_KEYS],
    Input("type", "value"),
    prevent_initial_call=False,
)
def _toggle_profile_panels(kind: str | None):
    kind = kind or "none"
    out = []
    for k in PROFILE_KEYS:
        # Preserve extra styling on the "none" panel
        if k == "none":
            base = {"fontSize": "11px", "color": "#999",
                    "fontStyle": "italic", "padding": "4px 0"}
        else:
            base = {}
        base.update(_display(k == kind))
        out.append(base)
    return out


@app.callback(
    Output("panel-gust-overlay", "style"),
    Input("gust_enabled", "value"),
)
def _toggle_gust_overlay(val):
    return _display(bool(val))


@app.callback(
    [Output("panel-spatial-mode", "style"),
     Output("panel-spatial-diag", "style"),
     Output("panel-spatial-matrix", "style"),
     Output("panel-spatial-ref", "style")],
    [Input("spatial_enabled", "value"),
     Input("gradient_mode", "value")],
)
def _toggle_spatial_panels(enabled, mode):
    if not enabled:
        return _display(False), _display(False), _display(False), _display(False)
    is_diag = mode == "diagonal"
    is_matrix = mode == "matrix"
    return (_display(True),
            _display(is_diag),
            _display(is_matrix),
            _display(True))


# ── Status readouts ─────────────────────────────────────────────────────
@app.callback(
    [Output("time-readout", "children"),
     Output("play-status", "children"), Output("play-status", "style"),
     Output("gust-status", "children"), Output("gust-status", "style"),
     Output("tick-count", "children")],
    [Input("time", "value"), Input("max_time", "value"),
     Input("interval", "disabled"), Input("interval", "n_intervals"),
     Input("type", "value"), Input("gust_enabled", "value"),
     Input("g_start", "value"), Input("g_duration", "value"),
     Input("go_start", "value"), Input("go_duration", "value")],
)
def _update_status(t_now, max_time, interval_disabled, n_intervals,
                   kind, gust_toggle, g_start, g_dur, go_start, go_dur):
    t = float(t_now or 0.0)
    mt = float(max_time or 10.0)
    time_str = f"{t:.2f} / {mt:.2f} s"

    playing = not interval_disabled
    play_str = "RUNNING" if playing else "STOPPED"
    play_style = ({"fontSize": "12px", "fontWeight": "bold", "color": "#2ca02c"}
                  if playing else
                  {"fontSize": "12px", "fontWeight": "bold", "color": "#d62728"})

    gs = float(g_start or 0.0)
    gd = float(g_dur or 0.0)
    os_ = float(go_start or 0.0)
    od = float(go_dur or 0.0)
    profile_gust = (kind == "gust") and (gs <= t < gs + gd)
    overlay_gust = bool(gust_toggle) and (os_ <= t < os_ + od)

    if profile_gust or overlay_gust:
        which = []
        if profile_gust:
            which.append("profile")
        if overlay_gust:
            which.append("overlay")
        gust_str = f"gust: ON ({'+'.join(which)})"
        gust_style = {"fontSize": "12px", "fontWeight": "bold",
                      "color": "#e67e22"}
    else:
        gust_str = "gust: OFF"
        gust_style = {"fontSize": "12px", "fontWeight": "bold",
                      "color": "#999"}

    return (time_str, play_str, play_style, gust_str, gust_style,
            str(int(n_intervals or 0)))


# ── Figure rebuild ──────────────────────────────────────────────────────
def _coerce_number(v: Any, fallback: float) -> float:
    if v is None or isinstance(v, bool):
        return fallback
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback


@app.callback(
    [Output("wind-graph", "figure"),
     Output("loading-overlay", "style")],
    [Input(sid, "value") for sid in NUMERIC_SLIDER_IDS]
    + [Input("type", "value"), Input("gradient_mode", "value"),
       Input("gust_enabled", "value"), Input("spatial_enabled", "value"),
       Input("view_mode", "value")],
)
def _update_figure(*vals):
    n_sl = len(NUMERIC_SLIDER_IDS)
    slider_vals = vals[:n_sl]
    kind, gradient_mode, gust_toggle, spatial_toggle, view_mode = vals[n_sl:]

    P = dict(DEFAULTS)
    for sid, v in zip(NUMERIC_SLIDER_IDS, slider_vals):
        P[sid] = _coerce_number(v, DEFAULTS[sid])
    P["type"] = kind or DEFAULTS["type"]
    P["gradient_mode"] = gradient_mode or DEFAULTS["gradient_mode"]
    P["gust_enabled"] = bool(gust_toggle)
    P["spatial_enabled"] = bool(spatial_toggle)
    P["view_mode"] = view_mode or DEFAULTS["view_mode"]

    try:
        fig = build_figure(P, P["time"], P["view_mode"])
        return fig, _display(False)
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        err_fig = go.Figure()
        err_fig.add_annotation(
            text=f"<b>Figure build failed</b><br>{type(exc).__name__}: {exc}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="red"),
        )
        return err_fig, _display(False)


# ── Preset application ──────────────────────────────────────────────────
@app.callback(
    [Output(sid, "value") for sid in PARAM_SLIDER_IDS]
    + [Output("type", "value"), Output("gradient_mode", "value"),
       Output("gust_enabled", "value"), Output("spatial_enabled", "value"),
       Output("view_mode", "value")],
    Input("preset", "value"),
    prevent_initial_call=True,
)
def _apply_preset(name: str):
    p = PRESETS.get(name, DEFAULTS)
    slider_vals = [p.get(sid, DEFAULTS[sid]) for sid in PARAM_SLIDER_IDS]
    return (slider_vals
            + [p.get("type", DEFAULTS["type"]),
               p.get("gradient_mode", DEFAULTS["gradient_mode"]),
               ["on"] if p.get("gust_enabled") else [],
               ["on"] if p.get("spatial_enabled", True) else [],
               p.get("view_mode", DEFAULTS["view_mode"])])


# ── Playback control ────────────────────────────────────────────────────
@app.callback(
    Output("play-btn", "children"),
    Output("interval", "disabled"),
    Input("play-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_playback(n_clicks):
    """Flip the button state based on click parity (n_clicks starts at 0
    and only fires when prevent_initial_call is True)."""
    playing = (n_clicks % 2) == 1
    if playing:
        return "⏸ Pause", False
    return "▶ Play", True


@app.callback(
    Output("time", "value"),
    Input("interval", "n_intervals"),
    State("time", "value"),
    State("play_speed", "value"),
    State("max_time", "value"),
    prevent_initial_call=True,
)
def advance_time(n, t_now, speed_label, max_time):
    step = PLAY_SPEEDS.get(speed_label, 0.05)
    max_t = max(float(max_time or 10.0), 0.1)
    t = float(t_now or 0.0)
    new_t = t + step
    if new_t >= max_t:
        new_t = 0.0
    return round(new_t, 4)


# ── Readout labels below every slider ───────────────────────────────────
@app.callback(
    [Output(f"val-{sid}", "children") for sid in READOUT_IDS],
    [Input(sid, "value") for sid in READOUT_IDS],
)
def _update_readouts(*vals):
    out = []
    for v in vals:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(f"{float(v):.2f}")
        else:
            out.append("—")
    return out


# ═══════════════════════════════════════════════════════════════════════
# 7. Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print(" Wind Field Simulator  —  http://0.0.0.0:8050")
    print("=" * 64)
    run_kwargs = dict(host="0.0.0.0", port=8050,
                      debug=False, use_reloader=False)
    try:
        app.run(**run_kwargs)
    except AttributeError:  # older dash
        app.run_server(**run_kwargs)