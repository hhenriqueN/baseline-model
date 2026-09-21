"""Deterministic post-touchdown ground controller (spec: after confirmed and
stable touchdown, transfer control to a simple deterministic controller that
sets throttle idle, maintains runway alignment with bounded rudder, applies
progressive symmetric braking, and stops the aircraft without leaving the
runway). Not learned, not part of the baseline MLPs (spec section 24: this
belongs to a future ground controller; this is the minimum deterministic
version required to bring the aircraft to a safe stop for logging purposes).
"""
from __future__ import annotations

from controller.config import RolloutConfig


class RolloutController:
    def __init__(self, cfg: RolloutConfig, runway_heading_deg: float):
        self.cfg = cfg
        self.runway_heading_deg = runway_heading_deg
        self._rollout_start_t: float | None = None

    def reset(self, t: float):
        self._rollout_start_t = t

    def compute(self, t: float, heading_deg: float, groundspeed_kt: float) -> dict:
        if self._rollout_start_t is None:
            self._rollout_start_t = t
        elapsed = t - self._rollout_start_t

        heading_error = ((heading_deg - self.runway_heading_deg + 180.0) % 360.0) - 180.0
        rudder = max(-self.cfg.rudder_limit, min(self.cfg.rudder_limit, self.cfg.rudder_kp * heading_error))

        brake_frac = min(1.0, max(0.0, elapsed / self.cfg.brake_ramp_duration_s))
        brake = self.cfg.brake_max * brake_frac
        if groundspeed_kt <= self.cfg.stop_groundspeed_kt:
            brake = self.cfg.brake_max  # full progressive braking once essentially stopped, hold it

        return {
            "aileron": 0.0,
            "elevator": 0.0,
            "rudder": rudder,
            "throttle": self.cfg.throttle_idle,
            "brake_left": brake,
            "brake_right": brake,
        }
