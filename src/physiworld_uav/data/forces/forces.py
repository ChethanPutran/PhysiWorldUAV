"""
Modular quadrotor force/torque architecture.

Conventions
-----------
World frame: right-handed, user supplies gravity_W (normally [0, 0, -9.81]).
Body frame: Hummingbird convention from the project audit: rotor thrust axis is +Y_B.
Quaternion: q_WB = [w, x, y, z], rotating body-frame vectors into world frame.
Angular velocity: omega_B in body frame.
All forces returned in world frame [N]; all torques returned in body frame [N m].

Dependencies: Python >=3.9, PyTorch.
The module deliberately separates force models. Do not enable lumped drag and
triangle-surface aerodynamics simultaneously unless their geometry/areas have
been partitioned to avoid double-counting.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple
import torch

Tensor = torch.Tensor
WindFn = Callable[[Tensor, float], Tensor]


def _vec(x, ref: Tensor) -> Tensor:
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def quat_normalize(q: Tensor) -> Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_to_matrix(q: Tensor) -> Tensor:
    """q=[w,x,y,z], body-to-world rotation matrix; supports batch dimensions."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),
        2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


@dataclass
class ModelConfig:
    enabled: bool = True


@dataclass
class RotorConfig(ModelConfig):
    k_thrust: float = 8.54858e-6
    k_torque: float = 1.0e-7
    thrust_axis_B: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    # Rotor positions in body frame, metres. Replace with exact Hummingbird geometry.
    positions_B: Tuple[Tuple[float, float, float], ...] = (
        ( 0.17, 0.0,  0.17),
        (-0.17, 0.0,  0.17),
        (-0.17, 0.0, -0.17),
        ( 0.17, 0.0, -0.17),
    )
    # +1/-1 signs for rotor spin direction. Verify against your motor ordering.
    spin_directions: Tuple[float, ...] = (1.0, -1.0, 1.0, -1.0)


@dataclass
class GravityConfig(ModelConfig):
    gravity_W: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class LumpedDragConfig(ModelConfig):
    # C_i * A_i [m^2], expressed along body x,y,z axes.
    cd_area_B: Tuple[float, float, float] = (0.08, 0.12, 0.10)
    air_density: float = 1.225


@dataclass
class SurfaceAeroConfig(ModelConfig):
    air_density: float = 1.225
    cp_windward: float = 1.0
    cp_leeward: float = 0.6
    skin_friction_cf: float = 0.02
    # Set to True only if the supplied mesh normals point outward.
    outward_normals: bool = True


@dataclass
class AddedMassConfig(ModelConfig):
    # Diagonal added mass in body axes [kg]. Start at zero; identify/calibrate it.
    added_mass_B: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ForceConfig:
    gravity: GravityConfig = field(default_factory=GravityConfig)
    rotors: RotorConfig = field(default_factory=RotorConfig)
    lumped_drag: LumpedDragConfig = field(default_factory=LumpedDragConfig)
    surface_aero: SurfaceAeroConfig = field(default_factory=lambda: SurfaceAeroConfig(enabled=False))
    added_mass: AddedMassConfig = field(default_factory=lambda: AddedMassConfig(enabled=False))
    # Select exactly one body aerodynamic model: "lumped", "surface", or "none".
    body_aero_mode: str = "lumped"


@dataclass
class RigidBodyState:
    position_W: Tensor       # (...,3), m
    velocity_W: Tensor       # (...,3), m/s
    quaternion_WB: Tensor    # (...,4), [w,x,y,z]
    omega_B: Tensor          # (...,3), rad/s
    rotor_omega: Tensor      # (...,N), rad/s


@dataclass
class ForceBreakdown:
    # Each entry is a separately inspectable contribution.
    forces_W: Dict[str, Tensor]
    torques_B: Dict[str, Tensor]
    total_force_W: Tensor
    total_torque_B: Tensor

    def as_dict(self):
        return {
            "forces_W": self.forces_W,
            "torques_B": self.torques_B,
            "total_force_W": self.total_force_W,
            "total_torque_B": self.total_torque_B,
        }


class ForceModel:
    name = "base"
    def __init__(self, cfg):
        self.cfg = cfg
    def compute(self, state: RigidBodyState, R_WB: Tensor, wind_fn: WindFn,
                time_s: float, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class GravityModel(ForceModel):
    name = "gravity"
    def compute(self, state, R_WB, wind_fn, time_s, mass: float, **kwargs):
        g = _vec(self.cfg.gravity_W, state.position_W)
        return mass * g.expand_as(state.position_W), torch.zeros_like(state.omega_B)


class RotorModel(ForceModel):
    name = "rotors"
    def compute(self, state, R_WB, wind_fn, time_s, **kwargs):
        cfg = self.cfg
        omega = state.rotor_omega
        if omega.shape[-1] != len(cfg.positions_B):
            raise ValueError(f"Expected {len(cfg.positions_B)} rotor speeds, got {omega.shape[-1]}")
        thrust = cfg.k_thrust * omega.square()
        reaction = cfg.k_torque * omega.square()
        axis = _vec(cfg.thrust_axis_B, omega)
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
        f_each_B = thrust.unsqueeze(-1) * axis
        r_B = _vec(cfg.positions_B, omega)
        # (...,N,3), moment from each rotor thrust.
        tau_thrust_each_B = torch.linalg.cross(r_B, f_each_B, dim=-1)
        spin = _vec(cfg.spin_directions, omega)
        tau_reaction_each_B = (-spin * reaction).unsqueeze(-1) * axis
        force_B = f_each_B.sum(dim=-2)
        tau_B = (tau_thrust_each_B + tau_reaction_each_B).sum(dim=-2)
        force_W = torch.matmul(R_WB, force_B.unsqueeze(-1)).squeeze(-1)
        return force_W, tau_B


class LumpedQuadraticDragModel(ForceModel):
    name = "lumped_drag"
    def compute(self, state, R_WB, wind_fn, time_s, **kwargs):
        # v_rel = air velocity relative to vehicle. Force follows v_rel.
        wind_W = wind_fn(state.position_W, time_s)
        v_rel_W = wind_W - state.velocity_W
        v_rel_B = torch.matmul(R_WB.transpose(-1, -2), v_rel_W.unsqueeze(-1)).squeeze(-1)
        ca = _vec(self.cfg.cd_area_B, v_rel_B)
        f_B = 0.5 * self.cfg.air_density * ca * v_rel_B.abs() * v_rel_B
        f_W = torch.matmul(R_WB, f_B.unsqueeze(-1)).squeeze(-1)
        return f_W, torch.zeros_like(state.omega_B)


class TriangleSurfaceAeroModel(ForceModel):
    name = "surface_aero"
    def compute(self, state, R_WB, wind_fn, time_s, mesh_vertices_B: Tensor,
                mesh_faces: Tensor, **kwargs):
        """
        mesh_vertices_B: (V,3), mesh_faces: (F,3), triangle mesh in body coordinates.
        Uses local air velocity at triangle centroids, including rotational surface velocity.
        Pressure uses a simple signed Cp model; Cf is a constant skin-friction proxy.
        """
        cfg = self.cfg
        device, dtype = state.position_W.device, state.position_W.dtype
        verts = _vec(mesh_vertices_B, state.position_W)
        faces = torch.as_tensor(mesh_faces, dtype=torch.long, device=device)
        tri = verts[faces]                       # (F,3,3)
        cent_B = tri.mean(dim=1)
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        cross = torch.linalg.cross(e1, e2, dim=-1)
        area = 0.5 * torch.linalg.vector_norm(cross, dim=-1).clamp_min(1e-12)
        normals_B = cross / (2.0 * area.unsqueeze(-1))
        if not cfg.outward_normals:
            normals_B = -normals_B

        # Broadcast mesh geometry over any leading batch dimensions.
        lead = state.position_W.shape[:-1]
        cent_B_b = cent_B.expand(lead + cent_B.shape)
        normals_B_b = normals_B.expand(lead + normals_B.shape)
        area_b = area.expand(lead + area.shape)
        cent_W = state.position_W.unsqueeze(-2) + torch.matmul(
            R_WB.unsqueeze(-3), cent_B_b.unsqueeze(-1)).squeeze(-1)
        cent_offset_W = torch.matmul(R_WB.unsqueeze(-3), cent_B_b.unsqueeze(-1)).squeeze(-1)
        omega_W = torch.matmul(R_WB, state.omega_B.unsqueeze(-1)).squeeze(-1)
        v_surface_W = state.velocity_W.unsqueeze(-2) + torch.linalg.cross(
            omega_W.unsqueeze(-2).expand_as(cent_offset_W), cent_offset_W, dim=-1)
        wind_W = wind_fn(cent_W, time_s)
        v_rel_W = wind_W - v_surface_W
        speed = torch.linalg.vector_norm(v_rel_W, dim=-1, keepdim=True).clamp_min(1e-9)
        h_W = v_rel_W / speed
        n_W = torch.matmul(R_WB.unsqueeze(-3), normals_B_b.unsqueeze(-1)).squeeze(-1)
        h_dot_n = (h_W * n_W).sum(dim=-1)
        q_dyn = 0.5 * cfg.air_density * speed.squeeze(-1).square()
        # Positive pressure on windward side, negative gauge pressure (suction) leeward.
        delta_p = (cfg.cp_windward * q_dyn * torch.relu(-h_dot_n).square()
                   - cfg.cp_leeward * q_dyn * torch.relu(h_dot_n).square())
        f_pressure_W = -delta_p.unsqueeze(-1) * area_b.unsqueeze(-1) * n_W

        v_t_W = v_rel_W - (v_rel_W * n_W).sum(dim=-1, keepdim=True) * n_W
        vt_speed = torch.linalg.vector_norm(v_t_W, dim=-1, keepdim=True)
        # Shear force on body points in direction of local relative airflow.
        f_shear_W = (0.5 * cfg.air_density * cfg.skin_friction_cf
                     * area_b.unsqueeze(-1) * vt_speed * v_t_W)
        f_tri_W = f_pressure_W + f_shear_W
        f_total_W = f_tri_W.sum(dim=-2)
        tau_W = torch.linalg.cross(cent_offset_W, f_tri_W, dim=-1).sum(dim=-2)
        tau_B = torch.matmul(R_WB.transpose(-1, -2), tau_W.unsqueeze(-1)).squeeze(-1)
        return f_total_W, tau_B


class AddedMassModel(ForceModel):
    name = "added_mass"
    def compute(self, state, R_WB, wind_fn, time_s, acceleration_W: Tensor,
                wind_acceleration_W: Optional[Tensor] = None, **kwargs):
        """
        Simplified diagonal added-mass correction:
        F_added,B = -M_added,B * (a_vehicle,B - a_air,B).
        Requires acceleration estimates. This is an approximation, not a complete
        variable-flow added-mass formulation. Keep disabled until calibrated.
        """
        if wind_acceleration_W is None:
            wind_acceleration_W = torch.zeros_like(acceleration_W)
        a_rel_W = acceleration_W - wind_acceleration_W
        a_rel_B = torch.matmul(R_WB.transpose(-1, -2), a_rel_W.unsqueeze(-1)).squeeze(-1)
        ma = _vec(self.cfg.added_mass_B, a_rel_B)
        f_B = -ma * a_rel_B
        f_W = torch.matmul(R_WB, f_B.unsqueeze(-1)).squeeze(-1)
        return f_W, torch.zeros_like(state.omega_B)


class ModularForceSystem:
    def __init__(self, config: ForceConfig, mass_kg: float,
                 mesh_vertices_B: Optional[Tensor] = None,
                 mesh_faces: Optional[Tensor] = None):
        self.cfg = config
        self.mass = float(mass_kg)
        self.mesh_vertices_B = mesh_vertices_B
        self.mesh_faces = mesh_faces
        if config.body_aero_mode not in ("none", "lumped", "surface"):
            raise ValueError("body_aero_mode must be 'none', 'lumped', or 'surface'")
        if config.body_aero_mode == "lumped" and config.lumped_drag.enabled is False:
            raise ValueError("body_aero_mode='lumped' but lumped_drag is disabled")
        if config.body_aero_mode == "surface" and config.surface_aero.enabled is False:
            raise ValueError("body_aero_mode='surface' but surface_aero is disabled")
        if config.body_aero_mode == "surface" and (mesh_vertices_B is None or mesh_faces is None):
            raise ValueError("Surface mode requires mesh_vertices_B and mesh_faces")
        self.models = {
            "gravity": GravityModel(config.gravity),
            "rotors": RotorModel(config.rotors),
            "lumped_drag": LumpedQuadraticDragModel(config.lumped_drag),
            "surface_aero": TriangleSurfaceAeroModel(config.surface_aero),
            "added_mass": AddedMassModel(config.added_mass),
        }

    def evaluate(self, state: RigidBodyState, wind_fn: WindFn, time_s: float,
                 acceleration_W: Optional[Tensor] = None,
                 wind_acceleration_W: Optional[Tensor] = None) -> ForceBreakdown:
        R = quat_to_matrix(state.quaternion_WB)
        forces, torques = {}, {}
        zero_f = torch.zeros_like(state.position_W)
        zero_t = torch.zeros_like(state.omega_B)

        def add(name, enabled, model, **kwargs):
            if not enabled:
                forces[name], torques[name] = zero_f, zero_t
                return
            f, tau = model.compute(state, R, wind_fn, time_s, **kwargs)
            forces[name], torques[name] = f, tau

        add("gravity", self.cfg.gravity.enabled, self.models["gravity"], mass=self.mass)
        add("rotors", self.cfg.rotors.enabled, self.models["rotors"])

        if self.cfg.body_aero_mode == "lumped":
            add("lumped_drag", self.cfg.lumped_drag.enabled, self.models["lumped_drag"])
            forces["surface_aero"], torques["surface_aero"] = zero_f, zero_t
        elif self.cfg.body_aero_mode == "surface":
            add("surface_aero", self.cfg.surface_aero.enabled, self.models["surface_aero"],
                mesh_vertices_B=self.mesh_vertices_B, mesh_faces=self.mesh_faces)
            forces["lumped_drag"], torques["lumped_drag"] = zero_f, zero_t
        else:
            forces["lumped_drag"], torques["lumped_drag"] = zero_f, zero_t
            forces["surface_aero"], torques["surface_aero"] = zero_f, zero_t

        if self.cfg.added_mass.enabled:
            if acceleration_W is None:
                raise ValueError("acceleration_W is required when added_mass is enabled")
            add("added_mass", True, self.models["added_mass"],
                acceleration_W=acceleration_W,
                wind_acceleration_W=wind_acceleration_W)
        else:
            forces["added_mass"], torques["added_mass"] = zero_f, zero_t

        total_f = torch.stack(list(forces.values()), dim=0).sum(dim=0)
        total_t = torch.stack(list(torques.values()), dim=0).sum(dim=0)
        return ForceBreakdown(forces, torques, total_f, total_t)


def make_example_config() -> ForceConfig:
    """Conservative starter config. Replace rotor positions/kQ/drag areas from calibration."""
    return ForceConfig(
        gravity=GravityConfig(enabled=True),
        rotors=RotorConfig(enabled=True),
        lumped_drag=LumpedDragConfig(enabled=True, cd_area_B=(0.08, 0.12, 0.10)),
        surface_aero=SurfaceAeroConfig(enabled=False),
        added_mass=AddedMassConfig(enabled=False, added_mass_B=(0.0, 0.0, 0.0)),
        body_aero_mode="lumped",
    )


if __name__ == "__main__":
    dtype = torch.float32
    state = RigidBodyState(
        position_W=torch.tensor([0., 0., 1.5], dtype=dtype),
        velocity_W=torch.zeros(3, dtype=dtype),
        quaternion_WB=torch.tensor([1., 0., 0., 0.], dtype=dtype),
        omega_B=torch.zeros(3, dtype=dtype),
        rotor_omega=torch.full((4,), 500.0, dtype=dtype),
    )
    # Replace this with your actual wind-field evaluator.
    def wind_fn(position_W: Tensor, time_s: float) -> Tensor:
        return torch.tensor([3.0, 0.0, 0.0], dtype=position_W.dtype,
                            device=position_W.device).expand_as(position_W)

    system = ModularForceSystem(make_example_config(), mass_kg=0.68)
    result = system.evaluate(state, wind_fn, time_s=0.0)
    print("Individual forces [N]:")
    for key, value in result.forces_W.items():
        print(f"  {key:16s}: {value.detach().cpu().numpy()}")
    print("Individual torques [N m]:")
    for key, value in result.torques_B.items():
        print(f"  {key:16s}: {value.detach().cpu().numpy()}")
    print("TOTAL F_W:", result.total_force_W)
    print("TOTAL tau_B:", result.total_torque_B)