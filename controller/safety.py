"""Safety envelope and abort-condition checks (spec: "If telemetry is lost,
the model fails, the aircraft leaves a configurable safe envelope, or
commands remain saturated, abort safely and record the reason.").
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from controller.config import SafetyEnvelope


@dataclass
class AbortDecision:
    should_abort: bool
    reason: str = ""


class SafetyMonitor:
    def __init__(self, envelope: SafetyEnvelope, runway_heading_deg: float):
        self.env = envelope
        self.runway_heading_deg = runway_heading_deg
        self._consec_saturated: dict[str, int] = {}

    def check_telemetry(self, telemetry, last_received_monotonic: float | None) -> AbortDecision:
        if telemetry is None or last_received_monotonic is None:
            return AbortDecision(True, "no_telemetry_received_yet")
        age = time.monotonic() - last_received_monotonic
        if age > self.env.telemetry_stale_s:
            return AbortDecision(True, f"telemetry_stale_{age:.2f}s")
        if not telemetry.is_finite():
            return AbortDecision(True, "telemetry_contains_nan_or_inf")
        return AbortDecision(False)

    def check_model_output(self, name: str, values: list[float]) -> AbortDecision:
        for v in values:
            if not math.isfinite(v):
                return AbortDecision(True, f"{name}_model_output_non_finite")
        return AbortDecision(False)

    def check_envelope(self, cross_track_m: float, heading_error_deg: float, roll_deg: float,
                         pitch_deg: float, agl_ft: float, airspeed_kt: float) -> AbortDecision:
        if abs(cross_track_m) > self.env.max_cross_track_m:
            return AbortDecision(True, f"cross_track_exceeded_{cross_track_m:.1f}m")
        if abs(heading_error_deg) > self.env.max_heading_error_deg:
            return AbortDecision(True, f"heading_error_exceeded_{heading_error_deg:.1f}deg")
        if abs(roll_deg) > self.env.max_bank_deg:
            return AbortDecision(True, f"bank_angle_exceeded_{roll_deg:.1f}deg")
        if abs(pitch_deg) > self.env.max_pitch_deg:
            return AbortDecision(True, f"pitch_angle_exceeded_{pitch_deg:.1f}deg")
        if agl_ft > self.env.max_agl_ft:
            return AbortDecision(True, f"agl_exceeded_{agl_ft:.0f}ft")
        if airspeed_kt < self.env.min_airspeed_kt:
            return AbortDecision(True, f"airspeed_too_low_{airspeed_kt:.1f}kt")
        if airspeed_kt > self.env.max_airspeed_kt:
            return AbortDecision(True, f"airspeed_too_high_{airspeed_kt:.1f}kt")
        return AbortDecision(False)

    def track_saturation(self, channel: str, saturated: bool) -> AbortDecision:
        if saturated:
            self._consec_saturated[channel] = self._consec_saturated.get(channel, 0) + 1
        else:
            self._consec_saturated[channel] = 0
        n = self._consec_saturated[channel]
        if n >= self.env.max_consecutive_saturated_cycles:
            return AbortDecision(True, f"{channel}_saturated_for_{n}_consecutive_cycles")
        return AbortDecision(False)

    def check_crashed(self, crashed_flag) -> AbortDecision:
        if crashed_flag:
            return AbortDecision(True, "flightgear_reported_crashed")
        return AbortDecision(False)
