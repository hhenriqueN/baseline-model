"""Stage 9b: Z-score scaler computation (spec section 16). Computed ONLY on
TRAIN sessions' baseline_core rows, never on validation sessions, never on
the fixed diagnostic holdout or the final_all_valid scaler's own held-out use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ZSCORE_VARIABLES = [
    "along_track_m", "cross_track_error_m", "agl_ft", "roll_deg", "sideslip_rad",
    "roll_rate_degps", "yaw_rate_degps", "airspeed_kt", "lateral_body_speed_fps",
    "glideslope_error_deg", "height_above_threshold_ft", "pitch_deg", "alpha_deg",
    "pitch_rate_degps", "airspeed_error_kt", "vertical_speed_fps",
]
NON_ZSCORE_VARIABLES = ["heading_error_sin", "heading_error_cos", "flaps_state_norm", "flare_flag"]
CLIP_MIN, CLIP_MAX = -5.0, 5.0
STD_NEAR_ZERO_THRESHOLD = 1e-6

SCHEMA_VERSION = 1


def _resolve_column(df: pd.DataFrame, var: str) -> str:
    for prefix in ("lat__", "lon__"):
        cand = prefix + var
        if cand in df.columns:
            return cand
    raise KeyError(f"variable {var} not found under lat__/lon__ prefixes")


def compute_scaler(train_dfs: list[pd.DataFrame], training_session_ids: list[str], purpose: str) -> dict:
    concatenated = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    mean = {}
    std = {}
    degenerate = []
    for var in ZSCORE_VARIABLES:
        col = _resolve_column(concatenated, var)
        vals = concatenated[col].to_numpy(dtype=float)
        m = float(np.mean(vals)) if len(vals) else float("nan")
        s = float(np.std(vals)) if len(vals) else float("nan")
        if not np.isfinite(s) or s < STD_NEAR_ZERO_THRESHOLD:
            degenerate.append({"variable": var, "std": s})
            s = 1.0  # explicit, documented fallback: do NOT silently divide by ~0
        mean[var] = m
        std[var] = s

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": purpose,
        "feature_order_zscored": ZSCORE_VARIABLES,
        "feature_order_not_zscored": NON_ZSCORE_VARIABLES,
        "mean": mean,
        "std": std,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "training_session_ids": sorted(training_session_ids),
        "degenerate_std_variables": degenerate,
    }


def save_scaler(scaler: dict, path: Path):
    with open(path, "w") as f:
        json.dump(scaler, f, indent=2)


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for var in scaler["feature_order_zscored"]:
        for prefix in ("lat__", "lon__"):
            col = prefix + var
            if col in out.columns:
                z = (out[col].to_numpy(dtype=float) - scaler["mean"][var]) / scaler["std"][var]
                out[col] = np.clip(z, scaler["clip_min"], scaler["clip_max"])
    return out
