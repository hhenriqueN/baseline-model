"""Central configuration for the real-time FlightGear controller.

All numeric defaults here are derived from the project's own data (see
comments) -- nothing is an arbitrary guess. Reuses pipeline.paths for the
project root so this package never hardcodes a duplicate path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline import paths as pipeline_paths

PROJECT_ROOT = pipeline_paths.ROOT
RUNS_DIR = PROJECT_ROOT / "runs"

# Default FlightGear application-support/user directory on macOS, matching
# the local install found during project inspection. Overridable via
# --fg-root on the CLI for portability to other machines.
DEFAULT_FG_ROOT = Path.home() / "Library/Application Support/FlightGear/fgdata_2024_1"
DEFAULT_FG_HOME = Path.home() / "Library/Application Support/FlightGear"

# Candidate locations to auto-detect the fgfs/FlightGear executable, in
# priority order. On macOS the official .app bundle's main binary IS the
# classic fgfs simulator (confirmed by inspecting its call stack: fgMainInit
# / fgOSMainLoop), just named "FlightGear" to satisfy the bundle's
# CFBundleExecutable -- passing real simulation flags (aircraft/airport/etc.)
# makes it boot straight into the sim rather than showing the Qt launcher.
FGFS_CANDIDATES = [
    Path.home() / "Desktop/FlightGear.app/Contents/MacOS/FlightGear",
    Path("/Applications/FlightGear.app/Contents/MacOS/FlightGear"),
    Path("/usr/games/fgfs"),
    Path("/usr/local/bin/fgfs"),
    Path("/opt/homebrew/bin/fgfs"),
]

CONTROL_HZ = 10.0
CONTROL_DT = 1.0 / CONTROL_HZ

TELEMETRY_PORT = 5500   # FlightGear (out) -> controller (in), UDP
COMMAND_PORT = 5510     # controller (out) -> FlightGear (in), UDP
TELNET_PORT = 5401      # secondary props channel, used only for readiness/init verification

PROTOCOL_NAME = "baseline_controller"

# --- Nominal-scenario initial condition -----------------------------------
# Derived empirically from the project's own demonstrations (see
# reports/training_report.md and the controller implementation report for
# the exact numbers): along_track_m ~ -3000 (verified: -2993m in
# voo-condicoes-normais row 0), initial MSL altitude ~800 ft (verified:
# 800.8 ft), target approach speed 65 kt (runway.yaml target_airspeed_kt).
# Flaps/trim/throttle are NOT copied from the raw (unconfigured) t=0 sample
# -- that is exactly the "artificial poorly configured initial condition"
# the project's own transient detector flags and excludes from
# baseline_core (data/manifests/transient_diagnostics.csv). Instead we start
# already in the STABILIZED configuration observed at
# phase==FINAL_APPROACH_ESTABLISHED across the 14 standardized-start
# episodes (median values, computed in the controller implementation
# report): flaps=0.99 (~full flaps), throttle=0.52, elevator-trim=0.031.
@dataclass
class NominalInitCondition:
    along_track_m: float = -3000.0
    cross_track_m: float = 0.0
    initial_msl_altitude_ft: float = 800.0
    airspeed_kt: float = 65.0
    flaps_init: float = 0.99
    throttle_init: float = 0.52
    elevator_trim_fixed: float = 0.031  # held constant for the whole controlled flight; see run.py docstring
    metar: str = "XXXX 010000Z 00000KT 9999 FEW250 15/10 Q1013 NOSIG"  # calm, clear, matches prior project sessions


SCENARIOS = {
    "nominal": NominalInitCondition(),
}


# --- Landing state machine thresholds (reused from data/config/preprocessing.yaml) ---
@dataclass
class FSMConfig:
    touchdown_confirm_s: float = 0.3
    rollout_confirm_s: float = 0.8
    complete_groundspeed_kt: float = 5.0
    complete_duration_s: float = 2.0
    diagnostic_long_candidate_s: float = 2.0
    nose_gear_index: int = 0
    main_gear_indices: tuple = (1, 2)


# --- Safety envelope (abort conditions) -----------------------------------
@dataclass
class SafetyEnvelope:
    max_cross_track_m: float = 150.0
    max_heading_error_deg: float = 45.0
    max_bank_deg: float = 45.0
    max_pitch_deg: float = 25.0
    min_agl_ft_before_flare: float = -20.0  # sanity floor; real ground contact is via WOW, not this
    max_agl_ft: float = 3000.0
    min_airspeed_kt: float = 40.0
    max_airspeed_kt: float = 120.0
    telemetry_stale_s: float = 0.5  # abort if no valid telemetry for this long
    max_consecutive_saturated_cycles: int = 30  # 3s at 10Hz of fully-saturated output on any channel
    max_loop_overrun_s: float = 0.5  # single-cycle compute time considered a control-loop stall


@dataclass
class RateLimits:
    """Max absolute change per 10 Hz cycle, applied to sent commands only
    (never to the raw model predictions that get logged) -- smooths actuator
    motion without altering what the network "wanted" to do, for the
    post-flight log to distinguish raw prediction from actuated command."""
    aileron_per_cycle: float = 0.20
    rudder_per_cycle: float = 0.20
    elevator_per_cycle: float = 0.20
    throttle_per_cycle: float = 0.10


@dataclass
class RolloutConfig:
    """Deterministic post-touchdown ground controller (spec: throttle idle,
    bounded rudder for alignment, progressive symmetric braking)."""
    throttle_idle: float = 0.0
    rudder_kp: float = 0.03       # rudder command per degree of heading error, clipped to [-1,1]
    rudder_limit: float = 0.6
    brake_ramp_duration_s: float = 4.0
    brake_max: float = 0.7
    stop_groundspeed_kt: float = 2.0
