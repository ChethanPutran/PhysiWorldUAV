"""Model applicability registry (plan §6.1-F11; requirement.md #3).

Each model declares its validity envelope; ``AeroForceSystem`` evaluates the
envelope every physics step and the runner records the fraction of time each
enabled model was used outside it (``stats.applicability_violation_frac``).
The validator flags episodes above ``max_violation_fraction``.

Envelopes follow from each model's equations and assumptions
(docs/models/aerodynamics.md), not from arbitrary labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    equation: str
    assumptions: tuple[str, ...]
    parameters: dict[str, str]
    frame: str
    cost: str
    envelope: dict[str, float]
    fails_when: tuple[str, ...]
    validation: str
    doc: str = "docs/models/aerodynamics.md"
    sources: tuple[str, ...] = field(default=())


REGISTRY: dict[str, ModelSpec] = {
    "rotor_drag": ModelSpec(
        name="rotor_drag",
        equation="F_i,B = −k_rd Ω_i P⊥ (R_WBᵀ(v − w_i) + ω_B × r_i);  τ = Σ r_i × F_i",
        assumptions=("rigid, stiff propellers (blade flapping + induced drag lumped)",
                     "linear in in-plane airspeed (small advance ratio)",
                     "no axial component (d_z = 0, Faessler 2018)"),
        parameters={"k_rd": "N s/(m·rad/s)", "scale": "domain factor"},
        frame="body; wind evaluated at each rotor hub",
        cost="O(R) per env per step",
        envelope={"max_advance_ratio": 0.20, "min_rotor_speed_fraction": 0.2},
        fails_when=("advance ratio μ = |v⊥|/(Ω R) > ~0.2 (nonlinear, translational lift)",
                    "very low rotor speed / windmilling", "large axial inflow (climb/descent > ~2 m/s)"),
        validation="CPU: linearity, zero axial force, ∝ Ω, zero net torque in uniform wind; "
                   "Isaac: wind-induced acceleration equals model force / mass",
        sources=("Kai, Allibert, Hua, Hamel, IFAC 2017", "Faessler, Franchi, Scaramuzza, RA-L 2018"),
    ),
    "body_drag": ModelSpec(
        name="body_drag",
        equation="F_B = −½ ρ C_D (Σ A_j|û_j|) ‖v_rel‖ v_rel;  τ = r_cop × F",
        assumptions=("quasi-steady bluff-body drag, Re ~ 10⁴–10⁵", "no lift/side force",
                     "projected areas from the visual mesh, C_D constant"),
        parameters={"cd": "-", "area_m2": "m² (A_x, A_y, A_z)", "air_density": "kg/m³"},
        frame="body; wind at the CoM",
        cost="O(1) per env per step",
        envelope={"max_airspeed_mps": 12.0},
        fails_when=("airspeed > ~12 m/s", "large sideslip lift/side forces on non-symmetric frames"),
        validation="CPU: anti-parallel to v_rel, ∝ v², rotation equivariance, terminal velocity",
        sources=("Hoerner, Fluid-Dynamic Drag (1965) — bluff-body C_D ≈ 1.0–1.2",),
    ),
    "ground_effect": ModelSpec(
        name="ground_effect",
        equation="ΔT_i = (1/(1 − (R/4h_i)²) − 1) k_F Ω_i²  along +z_B",
        assumptions=("Cheeseman–Bennett image-rotor model (constant power)", "flat ground at z = 0"),
        parameters={"rotor_radius": "m", "min_height_over_radius": "-"},
        frame="body; hub height in world",
        cost="O(R)",
        envelope={"min_height_over_radius": 0.5},
        fails_when=("h/R < 0.5 (clamped)", "partial-ground / slopes", "translational flight near ground"),
        validation="CPU: factor → 1 as h → ∞, value vs hand calculation",
        sources=("Cheeseman & Bennett, ARC R&M 3021 (1955)",),
    ),
    "aero_torque": ModelSpec(
        name="aero_torque",
        equation="τ_B = −K₁ ω_B − k₂ ‖ω_B‖ ω_B",
        assumptions=("lumped rotational damping; overlaps with rotor-drag ω×r (ablation only)",),
        parameters={"k1": "N m s/rad", "k2": "N m s²/rad²"},
        frame="body",
        cost="O(1)",
        envelope={"max_body_rate_radps": 10.0},
        fails_when=("used together with rotor_drag (double counting)",),
        validation="CPU: opposes ω, frame invariance",
    ),
}
