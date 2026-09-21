"""Stage 8: build the fixed-schema 11-input lateral / 12-input longitudinal
feature and label vectors (spec sections 14-15), phase labels, and the
baseline_core / full_approach / initialization_transient masks, on top of the
10 Hz resampled data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import columns as C
from pipeline.runway import RunwayConfig, compute_runway_features

LATERAL_FEATURE_ORDER = [
    "along_track_m", "cross_track_error_m", "heading_error_sin", "heading_error_cos",
    "agl_ft", "roll_deg", "sideslip_rad", "roll_rate_degps", "yaw_rate_degps",
    "airspeed_kt", "lateral_body_speed_fps",
]
LATERAL_LABEL_ORDER = ["aileron_command", "rudder_command"]

LONGITUDINAL_FEATURE_ORDER = [
    "along_track_m", "glideslope_error_deg", "height_above_threshold_ft", "agl_ft",
    "pitch_deg", "alpha_deg", "pitch_rate_degps", "airspeed_kt", "airspeed_error_kt",
    "vertical_speed_fps", "flaps_state_norm", "flare_flag",
]
LONGITUDINAL_LABEL_ORDER = ["elevator_effective", "throttle_command"]

FEATURE_SCHEMA_VERSION = 1

PHASE_ORDER = [
    "INITIALIZATION_TRANSIENT", "APPROACH_MANEUVERING", "FINAL_APPROACH_ESTABLISHED",
    "FLARE", "CONTACT_CANDIDATE", "TOUCHDOWN_CONFIRMED", "BOUNCE", "ROLLOUT", "COMPLETE",
]

# Phases eligible for the two airborne MLPs' training data (spec section 7).
# CONTACT_CANDIDATE is included only because the landing-FSM correction (spec
# section 5 / pipeline/landing_fsm.py) makes it trustworthy: it no longer
# absorbs airborne bounce/intermittent-contact gaps between the first main-gear
# touch and the eventual confirmed touchdown.
BASELINE_CORE_PHASES = {"FINAL_APPROACH_ESTABLISHED", "FLARE", "CONTACT_CANDIDATE"}
FULL_APPROACH_EXTRA_PHASES = {"APPROACH_MANEUVERING"}  # added on top of BASELINE_CORE_PHASES
RECOVERY_ABLATION_EXTRA_PHASES = {"INITIALIZATION_TRANSIENT"}  # added on top of BASELINE_CORE_PHASES


def _apply_glideslope_floor_hold(glideslope_error_deg: np.ndarray, along_track_m: np.ndarray, floor_m: float) -> np.ndarray:
    """Once distance-to-threshold drops below floor_m (including the whole
    post-threshold region), hold the last valid glideslope_error_deg instead
    of letting the atan2-based angle spin toward +-180 deg as the geometric
    triangle degenerates near/at the threshold (see runway.py docstring)."""
    dist_to_threshold_m = -along_track_m
    valid = dist_to_threshold_m >= floor_m
    out = glideslope_error_deg.copy()
    last_valid = None
    for i in range(len(out)):
        if valid[i]:
            last_valid = out[i]
        elif last_valid is not None:
            out[i] = last_valid
        # else: no valid value yet (episode starts already inside the floor) -> leave as-is, flagged in QA
    return out


def build_phase_and_masks(df: pd.DataFrame, boundaries: dict, is_standardized_start: bool, has_am_fa: bool,
                            landing_overlay: pd.Series | None = None) -> pd.DataFrame:
    """boundaries: dict with sim_time keys among transient_end_s,
    approach_maneuvering_s, final_approach_established_s, flare_s (all still
    single-threshold, one-way crossings -- correct as-is).

    landing_overlay: optional array/Series aligned to df.index, holding the
    per-row CONTACT_CANDIDATE/TOUCHDOWN_CONFIRMED/BOUNCE/ROLLOUT/COMPLETE
    label computed by pipeline/landing_fsm.py (None/NaN where the FSM overlay
    does not apply, i.e. still FLARE). This
    OVERRIDES the old approach of painting a single contact_candidate_s..
    touchdown_confirmed_s span, which is what silently absorbed airborne
    bounce/intermittent-contact gaps into CONTACT_CANDIDATE (spec section 4).
    """
    t = df[C.SIM_TIME].to_numpy()
    n = len(t)
    phase = np.full(n, "FINAL_APPROACH_ESTABLISHED", dtype=object)

    if has_am_fa:
        phase[t < boundaries["approach_maneuvering_s"]] = "PRE_APPROACH_CIRCUIT"
        phase[(t >= boundaries["approach_maneuvering_s"]) & (t < boundaries["final_approach_established_s"])] = "APPROACH_MANEUVERING"
        phase[t >= boundaries["final_approach_established_s"]] = "FINAL_APPROACH_ESTABLISHED"

    if is_standardized_start and boundaries.get("transient_end_s") is not None:
        phase[t < boundaries["transient_end_s"]] = "INITIALIZATION_TRANSIENT"

    phase[t >= boundaries["flare_s"]] = "FLARE"

    if landing_overlay is not None:
        overlay = np.asarray(landing_overlay, dtype=object)
        assert len(overlay) == n, f"landing_overlay length {len(overlay)} != {n}"
        has_label = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in overlay])
        phase[has_label] = overlay[has_label]

    initialization_transient = np.zeros(n, dtype=int)
    if is_standardized_start and boundaries.get("transient_end_s") is not None:
        initialization_transient[t < boundaries["transient_end_s"]] = 1

    # Dataset-variant masks (spec section 8): defined by PHASE-SET membership,
    # not by an unbounded "from some start time to end of file" threshold --
    # the latter is what previously let ROLLOUT/COMPLETE (and, at 10Hz
    # resolution, occasionally TOUCHDOWN_CONFIRMED/BOUNCE) rows leak into
    # baseline_core (spec section 7's known bug).
    baseline_core = np.isin(phase, list(BASELINE_CORE_PHASES)).astype(int)
    full_approach = np.isin(phase, list(BASELINE_CORE_PHASES | FULL_APPROACH_EXTRA_PHASES)).astype(int)
    recovery_ablation = np.isin(phase, list(BASELINE_CORE_PHASES | RECOVERY_ABLATION_EXTRA_PHASES)).astype(int)

    return pd.DataFrame({
        "phase": phase,
        "initialization_transient": initialization_transient,
        "baseline_core": baseline_core,
        "full_approach": full_approach,
        "recovery_ablation": recovery_ablation,
    }, index=df.index)


def build_episode_features(df10hz: pd.DataFrame, runway_cfg: RunwayConfig, cfg: dict, boundaries: dict,
                             is_standardized_start: bool, has_am_fa: bool,
                             landing_overlay: pd.Series | None = None) -> pd.DataFrame:
    rfeat = compute_runway_features(
        df10hz, runway_cfg,
        lat_col=C.LAT, lon_col=C.LON, heading_col=C.HEADING_DEG,
        alt_msl_col=C.ALT_MSL_FT, airspeed_col=C.AIRSPEED_KT, agl_col=C.ALT_AGL_FT,
    )
    floor_m = cfg["glideslope_feature"]["floor_distance_to_threshold_m"]
    rfeat["glideslope_error_deg"] = _apply_glideslope_floor_hold(
        rfeat["glideslope_error_deg"].to_numpy(), rfeat["along_track_m"].to_numpy(), floor_m)

    phase_masks = build_phase_and_masks(df10hz, boundaries, is_standardized_start, has_am_fa, landing_overlay)

    lateral = pd.DataFrame({
        "along_track_m": rfeat["along_track_m"],
        "cross_track_error_m": rfeat["cross_track_error_m"],
        "heading_error_sin": rfeat["heading_error_sin"],
        "heading_error_cos": rfeat["heading_error_cos"],
        "agl_ft": df10hz[C.ALT_AGL_FT],
        "roll_deg": df10hz[C.ROLL_DEG],
        "sideslip_rad": df10hz[C.SIDESLIP_RAD],
        "roll_rate_degps": df10hz[C.ROLL_RATE_DEGPS],
        "yaw_rate_degps": df10hz[C.YAW_RATE_DEGPS],
        "airspeed_kt": df10hz[C.AIRSPEED_KT],
        "lateral_body_speed_fps": df10hz[C.VBODY_FPS],
    })
    lateral_labels = pd.DataFrame({
        "aileron_command": df10hz[C.AILERON_CMD],
        "rudder_command": df10hz[C.RUDDER_CMD],
    })

    longitudinal = pd.DataFrame({
        "along_track_m": rfeat["along_track_m"],
        "glideslope_error_deg": rfeat["glideslope_error_deg"],
        "height_above_threshold_ft": rfeat["height_above_threshold_ft"],
        "agl_ft": df10hz[C.ALT_AGL_FT],
        "pitch_deg": df10hz[C.PITCH_DEG],
        "alpha_deg": df10hz[C.ALPHA_DEG],
        "pitch_rate_degps": df10hz[C.PITCH_RATE_DEGPS],
        "airspeed_kt": df10hz[C.AIRSPEED_KT],
        "airspeed_error_kt": rfeat["airspeed_error_kt"],
        "vertical_speed_fps": df10hz[C.VERTICAL_SPEED_FPS],
        "flaps_state_norm": df10hz[C.FLAPS_CMD],
        "flare_flag": rfeat["flare_flag"],
    })
    longitudinal_labels = pd.DataFrame({
        "elevator_effective": df10hz[C.ELEVATOR_CMD] + df10hz[C.ELEVATOR_TRIM],
        "throttle_command": df10hz[C.THROTTLE_ENGINE0],
    })

    meta = df10hz[["episode_id", "session_id", "source_file", "scenario_id", "runway_id", C.SIM_TIME, "frame"]].copy()
    meta = meta.rename(columns={"frame": "frame_10hz"})

    out = pd.concat([
        meta, phase_masks,
        lateral.add_prefix("lat__"), lateral_labels.add_prefix("laty__"),
        longitudinal.add_prefix("lon__"), longitudinal_labels.add_prefix("lony__"),
    ], axis=1)
    out.attrs["lateral_feature_order"] = LATERAL_FEATURE_ORDER
    out.attrs["lateral_label_order"] = LATERAL_LABEL_ORDER
    out.attrs["longitudinal_feature_order"] = LONGITUDINAL_FEATURE_ORDER
    out.attrs["longitudinal_label_order"] = LONGITUDINAL_LABEL_ORDER
    return out
