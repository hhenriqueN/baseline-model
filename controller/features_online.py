"""Online feature engineering for one telemetry sample, built ENTIRELY from
the project's own training-time functions (pipeline.runway,
pipeline.features, pipeline.scalers) so unit conventions, sign conventions,
feature order, and normalization are bit-for-bit the same code path used
offline. This module only adapts a single telemetry dict into the small
one-row DataFrame those functions expect, plus reproduces the one piece of
per-episode CAUSAL state training used (the glideslope floor-hold, see
pipeline/features.py::_apply_glideslope_floor_hold) as an explicit carried
state object instead of a full-trajectory array.
"""
from __future__ import annotations

import pandas as pd

from pipeline import columns as C
from pipeline.runway import RunwayConfig, compute_runway_features
from pipeline.features import LATERAL_FEATURE_ORDER, LONGITUDINAL_FEATURE_ORDER
from pipeline.scalers import apply_scaler


def _telemetry_to_raw_row(t: dict) -> pd.DataFrame:
    """One-row DataFrame using the exact property-path column names
    pipeline.runway.compute_runway_features expects."""
    row = {
        C.LAT: t["lat_deg"], C.LON: t["lon_deg"],
        C.HEADING_DEG: t["heading_deg"],
        C.ALT_MSL_FT: t["alt_msl_ft"], C.ALT_AGL_FT: t["alt_agl_ft"],
        C.AIRSPEED_KT: t["airspeed_kt"],
    }
    return pd.DataFrame([row])


class OnlineFeatureBuilder:
    """Carries the one bit of causal state needed to reproduce training-time
    feature engineering exactly: the glideslope-error floor hold (once
    distance-to-threshold drops below floor_distance_to_threshold_m, hold the
    last valid glideslope_error_deg instead of the atan2 angle spinning
    toward +-180 deg as the geometry degenerates -- see
    pipeline/features.py::_apply_glideslope_floor_hold, same threshold from
    data/config/preprocessing.yaml: glideslope_feature.floor_distance_to_threshold_m).
    """

    def __init__(self, runway_cfg: RunwayConfig, glideslope_floor_m: float):
        self.runway_cfg = runway_cfg
        self.glideslope_floor_m = glideslope_floor_m
        self._last_valid_glideslope_error_deg: float | None = None

    def build_raw_features(self, t: dict) -> dict:
        df = _telemetry_to_raw_row(t)
        rfeat = compute_runway_features(
            df, self.runway_cfg, lat_col=C.LAT, lon_col=C.LON, heading_col=C.HEADING_DEG,
            alt_msl_col=C.ALT_MSL_FT, airspeed_col=C.AIRSPEED_KT, agl_col=C.ALT_AGL_FT,
        )
        along_track_m = float(rfeat["along_track_m"].iloc[0])
        glide_deg = float(rfeat["glideslope_error_deg"].iloc[0])
        dist_to_threshold_m = -along_track_m
        if dist_to_threshold_m >= self.glideslope_floor_m:
            self._last_valid_glideslope_error_deg = glide_deg
        elif self._last_valid_glideslope_error_deg is not None:
            glide_deg = self._last_valid_glideslope_error_deg
        # else: no valid value seen yet (spawned already inside the floor) -- leave as-is, matches training

        raw = {
            "along_track_m": along_track_m,
            "cross_track_error_m": float(rfeat["cross_track_error_m"].iloc[0]),
            "heading_error_sin": float(rfeat["heading_error_sin"].iloc[0]),
            "heading_error_cos": float(rfeat["heading_error_cos"].iloc[0]),
            "agl_ft": float(t["alt_agl_ft"]),
            "roll_deg": float(t["roll_deg"]),
            "sideslip_rad": float(t["sideslip_rad"]),
            "roll_rate_degps": float(t["roll_rate_degps"]),
            "yaw_rate_degps": float(t["yaw_rate_degps"]),
            "airspeed_kt": float(t["airspeed_kt"]),
            "lateral_body_speed_fps": float(t["vbody_fps"]),
            "glideslope_error_deg": glide_deg,
            "height_above_threshold_ft": float(rfeat["height_above_threshold_ft"].iloc[0]),
            "pitch_deg": float(t["pitch_deg"]),
            "alpha_deg": float(t["alpha_deg"]),
            "pitch_rate_degps": float(t["pitch_rate_degps"]),
            "airspeed_error_kt": float(rfeat["airspeed_error_kt"].iloc[0]),
            "vertical_speed_fps": float(t["vertical_speed_fps"]),
            "flaps_state_norm": float(t["flaps_pos"]),
            "flare_flag": float(rfeat["flare_flag"].iloc[0]),
        }
        return raw


def to_normalized_vectors(raw: dict, scaler: dict) -> tuple[list, list]:
    """Applies the SAME z-score/clip scaler used at training time
    (pipeline.scalers.apply_scaler) and returns (lateral_11, longitudinal_12)
    ordered feature vectors, ready for the MLPs."""
    lat_row = {"lat__" + k: raw[k] for k in LATERAL_FEATURE_ORDER}
    lon_row = {"lon__" + k: raw[k] for k in LONGITUDINAL_FEATURE_ORDER}
    df = pd.DataFrame([{**lat_row, **lon_row}])
    df_norm = apply_scaler(df, scaler)
    lateral_vec = [float(df_norm["lat__" + f].iloc[0]) for f in LATERAL_FEATURE_ORDER]
    longitudinal_vec = [float(df_norm["lon__" + f].iloc[0]) for f in LONGITUDINAL_FEATURE_ORDER]
    return lateral_vec, longitudinal_vec
