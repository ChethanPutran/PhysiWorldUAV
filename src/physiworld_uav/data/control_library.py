# ═══════════════════════════════════════════════════════════════════════════
# ControllerLibrary
# ═══════════════════════════════════════════════════════════════════════════


import math
import numpy as np
from typing import Any, Dict, List, Union
from .utils import _sample, _resolve_direction, _to_list, _clamp, _unit, _quat_to_roll_pitch, _quat_to_yaw

class ControllerLibrary:
    """
    Factory for all flight controller types described in the config.

    Design: each controller is a method that receives the phase spec, the
    current state dict, elapsed time within the phase, dt, and a per-phase
    state dict (for storing ramp progress, start position, sampled targets).

    The public method `compute()` groups parallel controllers by mode,
    arbitrates between them, and produces the final (target_rate,
    target_thrust_N) pair that goes directly to OmniDrones' RateController.
    """

    # Which controller type produces which output mode
    POSITION_MODES = frozenset({
        "hover", "line", "circle", "figure_eight",
        "helix_climb", "helix_descent",
        "waypoint_random", "waypoint_sequence",
        "brake_to_zero", "random_line_velocity",
        "accel_decel",
    })
    ATTITUDE_MODES = frozenset({
        "attitude_step", "attitude_ramp", "attitude_sine",
        "attitude_combined", "attitude_step_sequence",
        "attitude_random", "random_attitude_command",
    })
    YAW_MODES = frozenset({
        "yaw_step", "yaw_ramp", "yaw_sine", "yaw_random",
        "random_yaw_command", "yaw_while_translating",
    })
    THRUST_MODES = frozenset({
        "thrust_ramp", "thrust_sinusoid", "thrust_ramp_plus_roll",
    })
    DIRECT_MODES = frozenset({
        "random_action", "open_loop", "combined_ramp_roll",
    })

    # ───────────────────────────────────────────────────────────────────
    def __init__(self, cfg, drone, device="cpu", seed: int = 42):
        self.cfg = cfg
        self.drone = drone
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.gains = {
            "kp_pos_xy": float(cfg.rate_controller_gains.kp_pos_xy),
            "kd_pos_xy": float(cfg.rate_controller_gains.kd_pos_xy),
            "kp_pos_z": float(cfg.rate_controller_gains.kp_pos_z),
            "kd_pos_z": float(cfg.rate_controller_gains.kd_pos_z),
            "kp_yaw": float(cfg.rate_controller_gains.kp_yaw),
            "kp_attitude": float(getattr(cfg.rate_controller_gains, "kp_attitude", 5.0)),
            "kd_attitude": float(getattr(cfg.rate_controller_gains, "kd_attitude", 0.5)),
            "max_tilt_rad": float(cfg.rate_controller_gains.max_tilt_rad),
            "max_rate_radps": float(cfg.rate_controller_gains.max_rate_radps),
        }
        self.g = 9.81
        self.mass = self._read_mass()

        # Direction vocabulary (merged across modes; last write wins)
        self.directions = {}
        for mode_name in ("translation", "yaw", "attitude", "combined"):
            try:
                mode_cfg = cfg.dataset.modes[mode_name]
                if "definition" in mode_cfg and "directions" in mode_cfg.definition:
                    self.directions.update(dict(mode_cfg.definition.directions))
            except Exception:
                pass
        if not self.directions:
            # Safe fallback
            self.directions = {
                "north": [1, 0, 0], "south": [-1, 0, 0],
                "east": [0, 1, 0], "west": [0, -1, 0],
                "up": [0, 0, 1], "down": [0, 0, -1],
                "ne": [0.7071, 0.7071, 0], "nw": [-0.7071, 0.7071, 0],
            }

        # Per-phase state containers, keyed by caller's phase_key
        self._phase_state: Dict[Any, Dict[str, Any]] = {}

        # Amplitude and rate vocabularies from configs
        self._amplitude_tables = {}
        self._rate_tables = {}
        self._transition_tables = {}
        for mode in ("attitude", "yaw", "combined"):
            try:
                d = cfg.dataset.modes[mode].definition
                for tbl in ("attitude_amplitudes_deg", "yaw_rates_radps",
                            "rate_profiles", "transition_profiles",
                            "acceleration_classes", "intensity_classes"):
                    if tbl in d:
                        self._amplitude_tables.setdefault(tbl, dict(d[tbl]))
            except Exception:
                pass
    

    def clear_phase_state(self):
        """Clear all per-phase state (for new episode)."""
        self._phase_state.clear()
        
    # ───────────────────────────────────────────────────────────────────
    def _read_mass(self) -> float:
        try:
            return float(
                np.asarray(self.drone.MASS_0).reshape(-1)[0].item()
                if not hasattr(self.drone.MASS_0, "item")
                else self.drone.MASS_0.item()
            )
        except Exception:
            try:
                return float(np.asarray(self.drone.MASS_0).reshape(-1)[0])
            except Exception:
                return 0.68  # Hummingbird default fallback

    # ───────────────────────────────────────────────────────────────────
    def compute(
        self,
        spec: Union[dict, List[dict]],
        state_dict: Dict[str, np.ndarray],
        step_idx: int,
        phase_elapsed: float,
        dt: float,
        phase_key: Any,
    ) -> tuple[np.ndarray, float]:
        """
        Compute (target_rate, target_thrust_N) for the current step.

        Parameters
        ----------
        spec
            A single controller dict OR a list of dicts (parallel controllers).
        state_dict
            Dict with keys: position(3,), orientation(4,) as (w,x,y,z),
            velocity(3,), angular_velocity(3,).
        step_idx
            Index within the current phase.
        phase_elapsed
            Elapsed time within the phase, in seconds.
        dt
            Simulation timestep, in seconds.
        phase_key
            Hashable key identifying the phase — used to persist per-phase
            state (ramp progress, sampled targets, integration buffers).

        Returns
        -------
        target_rate : np.ndarray, shape (3,)   [roll_rate, pitch_rate, yaw_rate]
        target_thrust_N : float
        """
        # Lazily initialise phase state
        if phase_key not in self._phase_state:
            self._phase_state[phase_key] = {}
        pstate = self._phase_state[phase_key]

        # Normalise spec to list
        if isinstance(spec, dict) or (hasattr(spec, "keys") and not hasattr(spec, "__iter__")):
            specs = [spec]
        else:
            try:
                specs = list(spec)
                if len(specs) == 0:
                    specs = [{"type": "hover"}]
            except TypeError:
                specs = [spec]

        # Group parallel controllers by mode
        position_ctrl = None
        attitude_ctrl = None
        yaw_ctrl = None
        thrust_ctrl = None
        direct_ctrl = None

        for s in specs:
            ctype = s.get("type", None) if isinstance(s, dict) else getattr(s, "type", None)
            if ctype is None:
                continue
            if ctype in self.POSITION_MODES:
                position_ctrl = s
            elif ctype in self.ATTITUDE_MODES:
                attitude_ctrl = s
            elif ctype in self.YAW_MODES:
                yaw_ctrl = s
            elif ctype in self.THRUST_MODES:
                thrust_ctrl = s
            elif ctype in self.DIRECT_MODES:
                direct_ctrl = s
            else:
                raise ValueError(f"Unknown controller type: {ctype!r}")

        # ── Direct mode short-circuits everything else ─────────────
        if direct_ctrl is not None:
            return self._compute_direct(direct_ctrl, state_dict, phase_elapsed, dt, pstate)

        # ── Attitude target or rate ────────────────────────────────
        target_roll, target_pitch = 0.0, 0.0
        roll_rate_direct, pitch_rate_direct = None, None

        if attitude_ctrl is not None:
            contrib = self._compute_attitude(
                attitude_ctrl, state_dict, phase_elapsed, dt, pstate
            )
            target_roll = contrib.get("roll_target", 0.0)
            target_pitch = contrib.get("pitch_target", 0.0)
            roll_rate_direct = contrib.get("roll_rate")
            pitch_rate_direct = contrib.get("pitch_rate")
        elif position_ctrl is not None:
            pos_contrib = self._compute_position(
                position_ctrl, state_dict, phase_elapsed, dt, pstate
            )
            # Convert position-velocity error to attitude targets
            target_roll, target_pitch = self._position_to_attitude(pos_contrib, state_dict)

        # ── Convert attitude targets to rate commands ──────────────
        current_roll, current_pitch = _quat_to_roll_pitch(state_dict["orientation"])
        if roll_rate_direct is not None:
            target_roll_rate = roll_rate_direct
        else:
            target_roll_rate = self.gains["kp_attitude"] * (target_roll - current_roll)

        if pitch_rate_direct is not None:
            target_pitch_rate = pitch_rate_direct
        else:
            target_pitch_rate = self.gains["kp_attitude"] * (target_pitch - current_pitch)

        # Clamp rates
        mr = self.gains["max_rate_radps"]
        target_roll_rate = _clamp(target_roll_rate, -mr, mr)
        target_pitch_rate = _clamp(target_pitch_rate, -mr, mr)

        # ── Yaw rate ───────────────────────────────────────────────
        if yaw_ctrl is not None:
            target_yaw_rate = self._compute_yaw(
                yaw_ctrl, state_dict, phase_elapsed, dt, pstate
            )
        else:
            target_yaw_rate = 0.0
        target_yaw_rate = _clamp(target_yaw_rate, -mr, mr)

        # ── Thrust ─────────────────────────────────────────────────
        if thrust_ctrl is not None:
            target_thrust_N = self._compute_thrust(
                thrust_ctrl, state_dict, phase_elapsed, dt, pstate
            )
        else:
            # Default: altitude P-D toward hover
            z = float(state_dict["position"][2])
            vz = float(state_dict["velocity"][2])
            z_des = pstate.get("hover_z", z)
            a_cmd = self.g + self.gains["kp_pos_z"] * (z_des - z) \
                        + self.gains["kd_pos_z"] * (-vz)
            target_thrust_N = max(0.0, self.mass * a_cmd)

        target_rate = np.array(
            [target_roll_rate, target_pitch_rate, target_yaw_rate],
            dtype=np.float32,
        )
        return target_rate, float(target_thrust_N)

    # ────────────────────────────────────────────────────────────────
    # Position-mode controllers
    # ────────────────────────────────────────────────────────────────
    def _compute_position(self, spec, state, phase_elapsed, dt, pstate):
        """Return a dict with target_pos (3,) and target_vel (3,)."""
        ctype = spec["type"]

        # Cache start pos on first call
        if "start_pos" not in pstate:
            pstate["start_pos"] = np.asarray(state["position"], dtype=np.float32).copy()

        method = {
            "hover": self._pos_hover,
            "line": self._pos_line,
            "circle": self._pos_circle,
            "figure_eight": self._pos_figure_eight,
            "helix_climb": self._pos_helix_climb,
            "helix_descent": self._pos_helix_descent,
            "waypoint_random": self._pos_waypoint_random,
            "waypoint_sequence": self._pos_waypoint_sequence,
            "brake_to_zero": self._pos_brake_to_zero,
            "random_line_velocity": self._pos_random_line_velocity,
            "accel_decel": self._pos_accel_decel,
        }.get(ctype)
        if method is None:
            raise ValueError(f"Not a position controller: {ctype}")
        return method(spec, state, phase_elapsed, dt, pstate)

    def _pos_hover(self, spec, state, t, dt, pstate):
        if "hover_pos" not in pstate:
            pstate["hover_pos"] = np.asarray(state["position"], dtype=np.float32).copy()
        z_des = pstate.get("hover_alt", float(state["position"][2]))
        target = pstate["hover_pos"].copy()
        target[2] = z_des
        return {"target_pos": target, "target_vel": np.zeros(3, dtype=np.float32)}

    def _pos_line(self, spec, state, t, dt, pstate):
        # Direction (may be "current_heading" — resolve via current yaw)
        dspec = spec.get("direction", spec.get("direction_from_list", [1, 0, 0]))
        if isinstance(dspec, str) and dspec == "current_heading":
            yaw = _quat_to_yaw(state["orientation"])
            direction = np.array([math.cos(yaw), math.sin(yaw), 0], dtype=np.float32)
        else:
            direction = _resolve_direction(dspec, self.directions, self.rng)
        direction = _unit(direction)

        # Sample velocity and distance on first call
        if "v_target" not in pstate:
            pstate["v_target"] = _sample(self.rng, spec.get("velocity_range_mps", spec.get("velocity_mps", 1.0)))
        if "distance_m" not in pstate:
            pstate["distance_m"] = _sample(self.rng, spec.get("distance_range_m", float("inf")))
        if "z_change_m" not in pstate:
            pstate["z_change_m"] = _sample(self.rng, spec.get("z_change_m", 0.0))
        if "ramp_duration" not in pstate:
            pstate["ramp_duration"] = self._resolve_accel_profile_duration(
                spec.get("acceleration_profile", "moderate")
            )

        # Velocity magnitude ramping from 0 to v_target over ramp_duration
        v_target = pstate["v_target"]
        ramp = pstate["ramp_duration"]
        v_t = v_target * min(1.0, t / max(ramp, 1e-6))

        # Distance covered so far (analytical integration of v_t)
        if t < ramp:
            dist = 0.5 * v_target * (t * t) / max(ramp, 1e-6)
        else:
            dist = 0.5 * v_target * ramp + v_target * (t - ramp)
        dist = min(dist, pstate["distance_m"])

        # Target
        target = pstate["start_pos"].copy() + direction * dist
        if pstate["z_change_m"] != 0.0:
            frac = min(1.0, dist / max(pstate["distance_m"], 1e-6))
            target[2] = pstate["start_pos"][2] + pstate["z_change_m"] * frac
        else:
            target[2] = pstate.get("hover_alt", pstate["start_pos"][2])

        target_vel = direction * v_t
        return {"target_pos": target, "target_vel": target_vel}

    def _pos_circle(self, spec, state, t, dt, pstate):
        if "circle_params" not in pstate:
            center = spec.get("center_xy", None)
            if center is None:
                c = np.asarray(state["position"], dtype=np.float32).copy()
                c[2] = pstate.get("hover_alt", c[2])
            else:
                c = np.array([float(center[0]), float(center[1]), 0.0], dtype=np.float32)
                c[2] = pstate.get("hover_alt", float(state["position"][2]))
            pstate["circle_params"] = {
                "center": c,
                "radius": float(spec.get("radius_m", 2.0)),
                "omega": float(spec.get("omega_radps", 0.3)),
                "start_angle": 0.0,
            }
        p = pstate["circle_params"]
        theta = p["start_angle"] + p["omega"] * t
        target = p["center"].copy()
        target[0] += p["radius"] * math.cos(theta)
        target[1] += p["radius"] * math.sin(theta)
        target_vel = np.array(
            [-p["radius"] * p["omega"] * math.sin(theta),
             p["radius"] * p["omega"] * math.cos(theta), 0.0],
            dtype=np.float32,
        )
        return {"target_pos": target, "target_vel": target_vel}

    def _pos_figure_eight(self, spec, state, t, dt, pstate):
        if "fig8_params" not in pstate:
            c = np.asarray(state["position"], dtype=np.float32).copy()
            c[2] = pstate.get("hover_alt", c[2])
            amps = _to_list(spec.get("amplitude_m", [2.0, 1.5]))
            pstate["fig8_params"] = {
                "center": c,
                "amp_x": float(amps[0]),
                "amp_y": float(amps[1]) if len(amps) > 1 else float(amps[0]),
                "freq": float(spec.get("frequency_hz", 0.1)),
            }
        p = pstate["fig8_params"]
        omega = 2 * math.pi * p["freq"]
        x = p["center"][0] + p["amp_x"] * math.sin(omega * t)
        y = p["center"][1] + p["amp_y"] * math.sin(2 * omega * t)
        z = p["center"][2]
        target = np.array([x, y, z], dtype=np.float32)
        vx = p["amp_x"] * omega * math.cos(omega * t)
        vy = p["amp_y"] * 2 * omega * math.cos(2 * omega * t)
        return {"target_pos": target, "target_vel": np.array([vx, vy, 0.0], dtype=np.float32)}

    def _pos_helix_climb(self, spec, state, t, dt, pstate):
        base = self._pos_circle(spec, state, t, dt, pstate)
        base["target_pos"][2] += float(spec.get("climb_rate_mps", 0.2)) * t
        return base

    def _pos_helix_descent(self, spec, state, t, dt, pstate):
        base = self._pos_circle(spec, state, t, dt, pstate)
        base["target_pos"][2] -= float(spec.get("descent_rate_mps", 0.2)) * t
        return base

    def _pos_waypoint_random(self, spec, state, t, dt, pstate):
        if "wp_target" not in pstate:
            pstate["wp_target"] = np.asarray(state["position"], dtype=np.float32).copy()
            pstate["wp_next_t"] = 0.0
        radius = float(spec.get("sample_radius_m", 3.0))
        interval = float(spec.get("sample_interval_s", 2.0))
        if t >= pstate["wp_next_t"]:
            offset = self.rng.uniform(-radius, radius, size=3).astype(np.float32)
            pstate["wp_target"] = pstate["start_pos"] + offset
            pstate["wp_target"][2] = max(0.5, pstate["wp_target"][2])
            pstate["wp_next_t"] = t + interval
        return {
            "target_pos": pstate["wp_target"],
            "target_vel": np.zeros(3, dtype=np.float32),
        }

    def _pos_waypoint_sequence(self, spec, state, t, dt, pstate):
        if "wps" not in pstate:
            pstate["wps"] = [
                {
                    "xy": np.array([float(w["xy"][0]), float(w["xy"][1])], dtype=np.float32),
                    "dz": float(w.get("z_offset_m", 0.0)),
                    "duration": float(w["duration_s"]),
                }
                for w in spec["waypoints"]
            ]
            pstate["wp_idx"] = 0
            pstate["wp_start_t"] = 0.0
            pstate["wp_origin"] = pstate["start_pos"].copy()

        # Advance to current segment
        while pstate["wp_idx"] < len(pstate["wps"]):
            w = pstate["wps"][pstate["wp_idx"]]
            t_into = t - pstate["wp_start_t"]
            if t_into <= w["duration"]:
                break
            # Move to next
            pstate["wp_start_t"] += w["duration"]
            pstate["wp_idx"] += 1
            prev_xy = pstate["wps"][pstate["wp_idx"] - 1]["xy"]
            pstate["wp_origin"] = np.array(
                [prev_xy[0], prev_xy[1], pstate["wp_origin"][2]], dtype=np.float32
            )

        if pstate["wp_idx"] >= len(pstate["wps"]):
            return {"target_pos": pstate["start_pos"], "target_vel": np.zeros(3, dtype=np.float32)}

        w = pstate["wps"][pstate["wp_idx"]]
        t_into = t - pstate["wp_start_t"]
        frac = min(1.0, t_into / max(w["duration"], 1e-6))
        target_xy = pstate["wp_origin"][:2] + (w["xy"] - pstate["wp_origin"][:2]) * frac
        target_z = pstate["wp_origin"][2] + w["dz"] * frac
        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
        target_vel = (target - pstate["wp_origin"]) / max(w["duration"], 1e-6)
        return {"target_pos": target, "target_vel": target_vel}

    def _pos_brake_to_zero(self, spec, state, t, dt, pstate):
        if "brake_start_pos" not in pstate:
            pstate["brake_start_pos"] = np.asarray(state["position"], dtype=np.float32).copy()
            pstate["brake_start_vel"] = np.asarray(state["velocity"], dtype=np.float32).copy()
        # Target is simply "stay at start position" with zero velocity
        return {
            "target_pos": pstate["brake_start_pos"],
            "target_vel": np.zeros(3, dtype=np.float32),
        }

    def _pos_random_line_velocity(self, spec, state, t, dt, pstate):
        if "rl_v" not in pstate:
            pstate["rl_v"] = np.zeros(3, dtype=np.float32)
            pstate["rl_next_t"] = 0.0
        interval = float(spec.get("resample_interval_s", 1.5))
        if t >= pstate["rl_next_t"]:
            v_range = _to_list(spec.get("velocity_range_mps", [-1.5, 1.5]))
            pstate["rl_v"] = self.rng.uniform(v_range[0], v_range[1], size=3).astype(np.float32)
            pstate["rl_v"][2] = 0.0  # keep altitude P-D in charge of z
            pstate["rl_next_t"] = t + interval
        # Integrate to get target pos
        if "rl_pos" not in pstate:
            pstate["rl_pos"] = np.asarray(state["position"], dtype=np.float32).copy()
        pstate["rl_pos"] += pstate["rl_v"] * dt
        pstate["rl_pos"][2] = pstate.get("hover_alt", pstate["rl_pos"][2])
        return {"target_pos": pstate["rl_pos"], "target_vel": pstate["rl_v"]}

    def _pos_accel_decel(self, spec, state, t, dt, pstate):
        # Sample trajectory-level constants
        if "accel_params" not in pstate:
            v_peak = _sample(self.rng, spec.get("peak_velocity_range_mps", [1.0, 2.0]))
            v_start = (
                float(np.linalg.norm(np.asarray(state["velocity"])))
                if spec.get("start_from_initial_velocity", False)
                else _sample(self.rng, spec.get("start_velocity_range_mps", [0.0, 0.0]))
            )
            accel_class = spec.get("acceleration_class", "moderate")
            class_table = self._amplitude_tables.get("acceleration_classes", {})
            acc_range = class_table.get(accel_class, [1.0, 3.0])
            a_mag = _sample(self.rng, acc_range)
            ramp_dur = v_peak / max(a_mag, 1e-6)
            pstate["accel_params"] = {
                "v_peak": v_peak, "v_start": v_start,
                "ramp": ramp_dur,
                "direction": _unit(_resolve_direction(
                    spec.get("direction",
                             spec.get("direction_from_list", [1, 0, 0])),
                    self.directions, self.rng)),
                "return": bool(spec.get("return_to_start", False)),
                "pos": np.asarray(state["position"], dtype=np.float32).copy(),
            }

        p = pstate["accel_params"]
        v_peak = p["v_peak"]
        ramp = max(p["ramp"], 1e-3)

        # Velocity magnitude profile
        if p["return"]:
            # Assume caller's total phase duration = 2*ramp + hold
            # (Caller must know duration; if it's longer, target velocity becomes 0)
            if t < ramp:
                v_mag = p["v_start"] + (v_peak - p["v_start"]) * (t / ramp)
            else:
                v_mag = v_peak
        else:
            if t < ramp:
                v_mag = p["v_start"] + (v_peak - p["v_start"]) * (t / ramp)
            else:
                v_mag = v_peak

        v_cmd = p["direction"] * v_mag
        p["pos"] += v_cmd * dt
        p["pos"][2] = pstate.get("hover_alt", p["pos"][2])
        return {"target_pos": p["pos"], "target_vel": v_cmd}

    def _resolve_accel_profile_duration(self, profile_name) -> float:
        table = self._amplitude_tables.get("acceleration_profiles", {})
        if profile_name in table:
            val = table[profile_name]
            return float(val) if not isinstance(val, dict) else 1.0
        # Fallbacks
        return {"gradual": 1.5, "moderate": 0.8, "aggressive": 0.3}.get(profile_name, 0.8)

    def _position_to_attitude(self, contrib, state):
        """Convert (target_pos, target_vel) into (roll, pitch) targets."""
        pos_err = contrib["target_pos"] - np.asarray(state["position"], dtype=np.float32)
        vel = np.asarray(state["velocity"], dtype=np.float32)
        vel_err = contrib["target_vel"] - vel

        a_x = self.gains["kp_pos_xy"] * pos_err[0] + self.gains["kd_pos_xy"] * vel_err[0]
        a_y = self.gains["kp_pos_xy"] * pos_err[1] + self.gains["kd_pos_xy"] * vel_err[1]

        # Desired tilt (small-angle): pitch ∝ a_x, roll ∝ -a_y
        pitch = a_x / self.g
        roll = -a_y / self.g

        max_tilt = self.gains["max_tilt_rad"]
        roll = _clamp(roll, -max_tilt, max_tilt)
        pitch = _clamp(pitch, -max_tilt, max_tilt)
        return roll, pitch

    # ────────────────────────────────────────────────────────────────
    # Attitude-mode controllers
    # ────────────────────────────────────────────────────────────────
    def _compute_attitude(self, spec, state, phase_elapsed, dt, pstate):
        ctype = spec["type"]
        method = {
            "attitude_step": self._att_step,
            "attitude_ramp": self._att_ramp,
            "attitude_sine": self._att_sine,
            "attitude_combined": self._att_combined,
            "attitude_step_sequence": self._att_step_sequence,
            "attitude_random": self._att_random,
            "random_attitude_command": self._att_random,
        }.get(ctype)
        if method is None:
            raise ValueError(f"Not an attitude controller: {ctype}")
        return method(spec, state, phase_elapsed, dt, pstate)

    def _att_step(self, spec, state, t, dt, pstate):
        if "att_targets" not in pstate:
            amp_class = spec.get("amplitude_class", "moderate")
            amp_table = self._amplitude_tables.get("attitude_amplitudes_deg", {})
            amp = _sample(self.rng, amp_table.get(amp_class, [15.0, 25.0]))
            axis = spec.get("axis", "roll")
            sign = spec.get("sign", "random")
            sgn = self.rng.choice([-1, 1]) if sign == "random" else (1 if sign == "positive" else -1)

            roll_t = 0.0
            pitch_t = 0.0
            if axis in ("roll", "roll_only"):
                roll_t = sgn * math.radians(amp)
            elif axis in ("pitch", "pitch_only"):
                pitch_t = sgn * math.radians(amp)
            elif axis == "roll_pitch":
                roll_t = sgn * math.radians(amp)
                pitch_t = sgn * math.radians(amp)

            rate_profile = spec.get("rate_profile", "moderate")
            ramp_table = self._amplitude_tables.get("rate_profiles", {})
            ramp_dur = float(ramp_table.get(rate_profile, 1.0)) \
                if not isinstance(ramp_table.get(rate_profile, 1.0), dict) else 1.0

            pstate["att_targets"] = {
                "roll": roll_t, "pitch": pitch_t,
                "hold": float(spec.get("hold_duration_s", 5.0)),
                "return": bool(spec.get("return_to_level", True)),
                "ramp": ramp_dur,
            }

        tgt = pstate["att_targets"]
        ramp = tgt["ramp"]
        hold = tgt["hold"]
        total_active = ramp + hold
        if tgt["return"]:
            # Target rises over ramp, holds, then returns over another ramp
            if t < ramp:
                frac = t / max(ramp, 1e-6)
                scale = frac
            elif t < ramp + hold:
                scale = 1.0
            elif t < 2 * ramp + hold:
                scale = 1.0 - (t - ramp - hold) / max(ramp, 1e-6)
            else:
                scale = 0.0
        else:
            if t < ramp:
                scale = t / max(ramp, 1e-6)
            else:
                scale = 1.0

        return {
            "roll_target": tgt["roll"] * scale,
            "pitch_target": tgt["pitch"] * scale,
        }

    def _att_ramp(self, spec, state, t, dt, pstate):
        if "att_ramp" not in pstate:
            axis = spec.get("axis", "roll")
            sign = spec.get("sign", "positive")
            sgn = self.rng.choice([-1, 1]) if sign == "random" else (1 if sign == "positive" else -1)
            target_deg = _sample(self.rng, spec.get("target_deg", [20.0, 30.0]))
            pstate["att_ramp"] = {
                "roll": sgn * math.radians(target_deg) if axis == "roll" else 0.0,
                "pitch": sgn * math.radians(target_deg) if axis == "pitch" else 0.0,
                "ramp": _sample(self.rng, spec.get("ramp_duration_s", [1.0, 3.0])),
                "hold": _sample(self.rng, spec.get("hold_duration_s", [3.0, 5.0])),
                "return": bool(spec.get("return_to_level", True)),
            }
        p = pstate["att_ramp"]
        ramp, hold = p["ramp"], p["hold"]
        if t < ramp:
            scale = t / max(ramp, 1e-6)
        elif t < ramp + hold:
            scale = 1.0
        elif p["return"] and t < 2 * ramp + hold:
            scale = 1.0 - (t - ramp - hold) / max(ramp, 1e-6)
        else:
            scale = 0.0
        return {"roll_target": p["roll"] * scale, "pitch_target": p["pitch"] * scale}

    def _att_sine(self, spec, state, t, dt, pstate):
        # Produces rate directly (not a target angle)
        axis = spec.get("axis", "roll")
        amp_deg = _sample(self.rng, spec.get("amplitude_deg", [10.0, 20.0]))
        freq = _sample(self.rng, spec.get("frequency_hz", [0.2, 0.4]))
        A = math.radians(amp_deg) * 2 * math.pi * freq  # rate amplitude
        w = 2 * math.pi * freq
        out = {"roll_target": 0.0, "pitch_target": 0.0}
        if axis in ("roll", "roll_pitch"):
            out["roll_rate"] = A * math.cos(w * t)
        if axis in ("pitch", "roll_pitch"):
            out["pitch_rate"] = A * math.cos(w * t + math.pi / 2)
        return out

    def _att_combined(self, spec, state, t, dt, pstate):
        if "att_comb" not in pstate:
            roll_amp = _sample(self.rng, spec.get("roll_target_deg", [15.0, 25.0]))
            pitch_amp = _sample(self.rng, spec.get("pitch_target_deg", [15.0, 25.0]))
            rs = spec.get("roll_sign", "random")
            ps = spec.get("pitch_sign", "random")
            rs = self.rng.choice([-1, 1]) if rs == "random" else (1 if rs == "positive" else -1)
            ps = self.rng.choice([-1, 1]) if ps == "random" else (1 if ps == "positive" else -1)
            rate_profile = spec.get("rate_profile", "moderate")
            ramp_table = self._amplitude_tables.get("rate_profiles", {})
            rp = ramp_table.get(rate_profile, 1.0)
            ramp = float(rp) if not isinstance(rp, dict) else 1.0
            pstate["att_comb"] = {
                "roll": rs * math.radians(roll_amp),
                "pitch": ps * math.radians(pitch_amp),
                "ramp": ramp,
                "hold": _sample(self.rng, spec.get("hold_duration_s", [3.0, 6.0])),
                "return": bool(spec.get("return_to_level", True)),
            }
        p = pstate["att_comb"]
        ramp, hold = p["ramp"], p["hold"]
        if t < ramp:
            scale = t / max(ramp, 1e-6)
        elif t < ramp + hold:
            scale = 1.0
        elif p["return"] and t < 2 * ramp + hold:
            scale = 1.0 - (t - ramp - hold) / max(ramp, 1e-6)
        else:
            scale = 0.0
        return {"roll_target": p["roll"] * scale, "pitch_target": p["pitch"] * scale}

    def _att_step_sequence(self, spec, state, t, dt, pstate):
        steps = spec["steps"]
        # Find the currently active step
        active = None
        for i, s in enumerate(steps):
            t0 = float(s.get("t_offset_s", 0.0))
            hold = float(s.get("hold_s", 5.0))
            if t0 <= t < t0 + hold:
                active = s
                break
        if active is None:
            return {"roll_target": 0.0, "pitch_target": 0.0}
        return {
            "roll_target": math.radians(float(active.get("roll_deg", 0.0))),
            "pitch_target": math.radians(float(active.get("pitch_deg", 0.0))),
        }

    def _att_random(self, spec, state, t, dt, pstate):
        if "att_rand" not in pstate:
            pstate["att_rand"] = {"roll": 0.0, "pitch": 0.0, "next_t": 0.0}
        p = pstate["att_rand"]
        interval = float(spec.get("resample_interval_s", 0.5))
        if t >= p["next_t"]:
            roll_range = _to_list(spec.get("roll_range_deg", [-15.0, 15.0]))
            pitch_range = _to_list(spec.get("pitch_range_deg", [-15.0, 15.0]))
            p["roll"] = math.radians(self.rng.uniform(roll_range[0], roll_range[1]))
            p["pitch"] = math.radians(self.rng.uniform(pitch_range[0], pitch_range[1]))
            p["next_t"] = t + interval
        return {"roll_target": p["roll"], "pitch_target": p["pitch"]}

    # ────────────────────────────────────────────────────────────────
    # Yaw-mode controllers
    # ────────────────────────────────────────────────────────────────
    def _compute_yaw(self, spec, state, phase_elapsed, dt, pstate) -> float:
        ctype = spec["type"]
        method = {
            "yaw_step": self._yaw_step,
            "yaw_ramp": self._yaw_step,
            "yaw_sine": self._yaw_sine,
            "yaw_random": self._yaw_random,
            "random_yaw_command": self._yaw_random,
            "yaw_while_translating": self._yaw_step,
        }.get(ctype)
        if method is None:
            raise ValueError(f"Not a yaw controller: {ctype}")
        return method(spec, state, phase_elapsed, dt, pstate)

    def _yaw_step(self, spec, state, t, dt, pstate):
        if "yaw_step" not in pstate:
            d = _sample(self.rng, spec.get("target_change_rad", [1.5708]))
            rate_class = spec.get("rate_class", "moderate")
            rate_table = self._amplitude_tables.get("yaw_rates_radps", {})
            rate = _sample(self.rng, rate_table.get(rate_class, [0.5, 1.2]))
            pstate["yaw_step"] = {
                "start": _quat_to_yaw(state["orientation"]),
                "delta": d,
                "rate": rate,
            }
        p = pstate["yaw_step"]
        # Commanded yaw rate toward target
        current = _quat_to_yaw(state["orientation"])
        target = p["start"] + p["delta"]
        err = target - current
        # Wrap to [-π, π]
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi
        # P toward target with rate limit
        rate = self.gains["kp_yaw"] * err
        return _clamp(rate, -p["rate"], p["rate"])

    def _yaw_sine(self, spec, state, t, dt, pstate):
        A = _sample(self.rng, spec.get("amplitude_radps", [0.4, 1.0]))
        f = _sample(self.rng, spec.get("frequency_hz", [0.1, 0.25]))
        return A * math.sin(2 * math.pi * f * t)

    def _yaw_random(self, spec, state, t, dt, pstate):
        if "yaw_rand" not in pstate:
            pstate["yaw_rand"] = {"rate": 0.0, "next_t": 0.0}
        p = pstate["yaw_rand"]
        interval = float(spec.get("resample_interval_s", 0.5))
        if t >= p["next_t"]:
            r = _to_list(spec.get("yaw_rate_range_radps", spec.get("rate_range_radps", [-0.5, 0.5])))
            p["rate"] = self.rng.uniform(r[0], r[1])
            p["next_t"] = t + interval
        return p["rate"]

    # ────────────────────────────────────────────────────────────────
    # Thrust-mode controllers
    # ────────────────────────────────────────────────────────────────
    def _compute_thrust(self, spec, state, phase_elapsed, dt, pstate) -> float:
        ctype = spec["type"]
        if ctype == "thrust_ramp":
            return self._thrust_ramp(spec, state, phase_elapsed, dt, pstate)
        if ctype == "thrust_sinusoid":
            return self._thrust_sinusoid(spec, state, phase_elapsed, dt, pstate)
        if ctype == "thrust_ramp_plus_roll":
            return self._thrust_ramp(spec, state, phase_elapsed, dt, pstate)
        raise ValueError(f"Not a thrust controller: {ctype}")

    def _thrust_ramp(self, spec, state, t, dt, pstate) -> float:
        if "thrust_ramp" not in pstate:
            fr = _to_list(spec.get("thrust_range_frac", [0.6, 1.4]))
            pstate["thrust_ramp"] = {
                "lo": float(fr[0]),
                "hi": float(fr[1]),
                "ramp": float(spec.get("ramp_duration_s", 3.0)),
                "hold": float(spec.get("hold_duration_s", 2.0)),
            }
        p = pstate["thrust_ramp"]
        ramp, hold = p["ramp"], p["hold"]
        lo, hi = p["lo"], p["hi"]
        if t < ramp:
            frac = lo + (hi - lo) * (t / max(ramp, 1e-6))
        elif t < ramp + hold:
            frac = hi
        elif t < 2 * ramp + hold:
            frac = hi - (hi - lo) * ((t - ramp - hold) / max(ramp, 1e-6))
        else:
            frac = lo
        # frac is a multiple of mg
        return max(0.0, self.mass * self.g * frac)

    def _thrust_sinusoid(self, spec, state, t, dt, pstate) -> float:
        amp = _sample(self.rng, spec.get("amplitude_frac", [0.20, 0.40]))
        f = _sample(self.rng, spec.get("frequency_hz", [0.15, 0.35]))
        return self.mass * self.g * (1.0 + amp * math.sin(2 * math.pi * f * t))

    # ────────────────────────────────────────────────────────────────
    # Direct-mode controllers (bypass everything else)
    # ────────────────────────────────────────────────────────────────
    def _compute_direct(self, spec, state, phase_elapsed, dt, pstate):
        ctype = spec["type"]
        if ctype == "random_action":
            return self._random_action(spec, state, phase_elapsed, dt, pstate)
        if ctype == "open_loop":
            return self._open_loop(spec, state, phase_elapsed, dt, pstate)
        if ctype == "combined_ramp_roll":
            # Fuse thrust_ramp + roll rate
            thrust_N = self._thrust_ramp(spec, state, phase_elapsed, dt, pstate)
            roll_rate = float(spec.get("roll_rate_radps", 0.3))
            return np.array([roll_rate, 0.0, 0.0], dtype=np.float32), thrust_N
        raise ValueError(f"Unknown direct controller: {ctype}")

    def _random_action(self, spec, state, t, dt, pstate):
        if "ra" not in pstate:
            pstate["ra"] = {"thrust": 0.0, "rates": np.zeros(3, dtype=np.float32), "next_t": 0.0}
        p = pstate["ra"]
        interval = float(spec.get("resample_interval_s", 0.3))
        if t >= p["next_t"]:
            t_range = _to_list(spec.get("thrust_range_N", [4.0, 9.0]))
            r_range = _to_list(spec.get("rate_range_radps", [-0.6, 0.6]))
            p["thrust"] = self.rng.uniform(t_range[0], t_range[1])
            p["rates"] = self.rng.uniform(r_range[0], r_range[1], size=3).astype(np.float32)
            p["next_t"] = t + interval
        return p["rates"].copy(), float(p["thrust"])

    def _open_loop(self, spec, state, t, dt, pstate):
        # Direct motor commands — not routed through RateController
        motor_fracs = _to_list(spec.get("motor_fracs", [0.6, 0.6, 0.6, 0.6]))
        thrust = float(spec.get("thrust_frac", 0.6)) * self.mass * self.g * 2.0
        # Report a zero rate + a thrust; the generation script must handle
        # open_loop differently by calling drone.apply_action with raw motor cmds.
        return np.zeros(3, dtype=np.float32), thrust