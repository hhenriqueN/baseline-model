"""Stage 2: column-level and file-level audit of the raw CSVs.

Produces (written by scripts/02_audit.py):
- data/manifests/columns_catalog.csv
- data/manifests/duplicate_columns.csv
- data/manifests/exclusions.csv
- data/reports/qa_summary.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import columns as C

NEAR_CONSTANT_STD_THRESHOLD = 1e-6
NEAR_CONSTANT_RANGE_THRESHOLD = 1e-6

# Known physically-bounded control ranges (normalized commands), used for the
# "physically impossible values" check. Values are inclusive tolerances.
CONTROL_RANGES = {
    C.AILERON_CMD: (-1.0, 1.0),
    C.ELEVATOR_CMD: (-1.0, 1.0),
    C.RUDDER_CMD: (-1.0, 1.0),
    C.AILERON_TRIM: (-1.0, 1.0),
    C.RUDDER_TRIM: (-1.0, 1.0),
    C.ELEVATOR_TRIM: (-1.0, 1.0),
    C.FLAPS_CMD: (0.0, 1.0),
    C.THROTTLE_ENGINE0: (0.0, 1.0),
    C.THROTTLE_ENGINE1: (0.0, 1.0),
    C.SPEEDBRAKE: (0.0, 1.0),
    C.SPOILERS: (0.0, 1.0),
}

# Explicit candidate duplicate pairs to verify (col_a is the one we keep,
# col_b is the candidate to discard), with the conversion function b->a used
# purely to *check* the relation (does not mutate data).
UNIT_DUPLICATE_CANDIDATES = [
    dict(keep=C.ALT_AGL_FT, drop=C.ALT_AGL_M,
         convert=lambda ft: ft * 0.3048, forward=True,
         formula="agl_m = agl_ft * 0.3048", note="ft->m unit conversion"),
    dict(keep=C.VERTICAL_SPEED_FPS, drop=C.SPEED_DOWN_FPS,
         convert=lambda vs: -vs, forward=True,
         formula="speed_down_fps = -vertical_speed_fps", note="sign-flipped NED component"),
    dict(keep=C.AIRSPEED_KT, drop=C.VIAS_KT,
         convert=lambda kt: kt, forward=True,
         formula="vias_kts ~= airspeed_kt (identity check)", note="indicated vs calibrated airspeed"),
    dict(keep=C.SIDESLIP_RAD, drop=C.BETA_DEG,
         convert=lambda rad: np.degrees(rad), forward=True,
         formula="beta_deg = degrees(side_slip_rad)", note="deg vs rad sideslip representation"),
    dict(keep=C.THROTTLE_ENGINE0, drop=C.THROTTLE_ENGINE1,
         convert=lambda x: x, forward=True,
         formula="engine1_throttle ~= engine0_throttle (identity check)", note="C172P is single-engine; engine[1] may be an unused copy"),
]


def load_all_raw(raw_files_manifest: pd.DataFrame, io_read_fn) -> pd.DataFrame:
    frames = []
    for _, row in raw_files_manifest.iterrows():
        df, _fmt = io_read_fn(row["raw_path"])
        df.insert(0, "source_file", row["source_file"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_columns_catalog(all_df: pd.DataFrame, baseline_files_mask) -> pd.DataFrame:
    """One row per raw column with descriptive stats computed across all
    eligible (non pilot-familiarization) rows, plus role/duplicate metadata.
    """
    rows = []
    numeric_df = all_df.select_dtypes(include=[np.number])
    baseline_df = numeric_df[baseline_files_mask]

    for col in all_df.columns:
        if col == "source_file":
            continue
        series_all = all_df[col]
        is_numeric = np.issubdtype(series_all.dtype, np.number)
        series = baseline_df[col] if (is_numeric and col in baseline_df.columns) else series_all[baseline_files_mask]

        missing = int(series.isna().sum())
        inf_count = int(np.isinf(series.to_numpy(dtype=float)).sum()) if is_numeric else 0
        n_unique = int(series.nunique(dropna=True))
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()

        if is_numeric and len(finite) > 0:
            col_min = float(finite.min())
            col_max = float(finite.max())
            col_mean = float(finite.mean())
            col_std = float(finite.std())
        else:
            col_min = col_max = col_mean = col_std = np.nan

        value_range = (col_max - col_min) if is_numeric and len(finite) else np.nan
        is_constant = n_unique <= 1
        is_near_constant = (
            is_numeric and len(finite) > 0 and not is_constant and
            (col_std < NEAR_CONSTANT_STD_THRESHOLD or (not np.isnan(value_range) and value_range < NEAR_CONSTANT_RANGE_THRESHOLD))
        )

        rows.append({
            "column_name": col,
            "dtype": str(series_all.dtype),
            "unit": "",  # filled in below via heuristic
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "std": col_std,
            "n_unique": n_unique,
            "missing_count": missing,
            "inf_count": inf_count,
            "is_constant": is_constant,
            "is_near_constant": is_near_constant,
            "duplicate_of": "",
            "conversion_relation": "",
            "selected_for_baseline": False,
            "role": "",
            "exclusion_reason": "",
        })

    catalog = pd.DataFrame(rows).set_index("column_name", drop=False)
    _assign_units(catalog)
    _assign_roles(catalog)
    return catalog.reset_index(drop=True)


def _assign_units(catalog: pd.DataFrame):
    def unit_for(name: str) -> str:
        if name.endswith("-deg[0]") or "-deg[0]" in name:
            return "deg"
        if "-degps[0]" in name:
            return "deg/s"
        if "-rad[0]" in name:
            return "rad"
        if "-ft[0]" in name:
            return "ft"
        if "-m[0]" in name and "gear" not in name:
            return "m"
        if "-kt[0]" in name:
            return "kt"
        if "-fps[0]" in name or "fps_sec" in name:
            return "ft/s"
        if "-ms[0]" in name:
            return "m/s"
        if "norm[0]" in name:
            return "norm[0..1 or -1..1]"
        if name in (C.FRAME,):
            return "count"
        if name == C.SIM_TIME:
            return "s"
        return ""
    for col in catalog.index:
        catalog.loc[col, "unit"] = unit_for(col)


def _assign_roles(catalog: pd.DataFrame):
    metadata_cols = {C.FRAME, C.SIM_TIME}
    label_cols = {C.AILERON_CMD, C.ELEVATOR_CMD, C.RUDDER_CMD, C.THROTTLE_ENGINE0}
    derived_source_cols = {C.ELEVATOR_TRIM, C.LAT, C.LON}
    flight_manager_cols = {
        C.gear_wow(0), C.gear_wow(1), C.gear_wow(2),
        C.gear_compression(0), C.gear_compression(1), C.gear_compression(2),
        C.gear_rollspeed_ms(0), C.gear_rollspeed_ms(1), C.gear_rollspeed_ms(2),
        C.GROUNDSPEED_KT, "/controls[0]/gear[0]/gear-down[0]",
    }
    neural_input_cols = {
        C.ALT_MSL_FT, C.ALT_AGL_FT, C.HEADING_DEG, C.ROLL_DEG, C.PITCH_DEG,
        C.ALPHA_DEG, C.SIDESLIP_RAD, C.ROLL_RATE_DEGPS, C.PITCH_RATE_DEGPS,
        C.YAW_RATE_DEGPS, C.AIRSPEED_KT, C.VERTICAL_SPEED_FPS, C.VBODY_FPS,
        C.FLAPS_CMD,
    }
    qa_only_cols = {
        C.WIND_SPEED_KT, C.WIND_FROM_HEADING_DEG, C.TURBULENCE_MAGNITUDE,
        C.GROUND_ELEV_M, C.AILERON_TRIM, C.RUDDER_TRIM,
        C.YOKE_AILERON, C.YOKE_ELEVATOR, C.PEDALS_RUDDER, C.FLAPS_LEVER,
        C.SURF_ELEVATOR, C.SURF_FLAP, C.SURF_AILERON_L, C.SURF_AILERON_R, C.SURF_RUDDER,
        "/instrumentation[0]/slip-skid-ball[0]/indicated-slip-skid[0]",
        C.VIAS_KT, C.SPEED_DOWN_FPS, C.ALT_AGL_M, C.THROTTLE_ENGINE1,
        "/environment[0]/temperature-degc[0]", "/environment[0]/pressure-sea-level-inhg[0]",
        "/environment[0]/magnetic-variation-deg[0]", "/environment[0]/visibility-m[0]",
    }
    forbidden_substrings = C.FORBIDDEN_AS_INPUT_SUBSTRINGS

    for col in catalog.index:
        role = None
        reason = ""
        if col in metadata_cols:
            role = "metadata"
        elif col in label_cols:
            role = "label"
        elif col in derived_source_cols:
            role = "derived_feature_source"
        elif col in flight_manager_cols:
            role = "flight_manager_only"
        elif col in qa_only_cols:
            role = "qa_only"
        elif col in neural_input_cols:
            role = "neural_input"
        else:
            role = "excluded"
            if catalog.loc[col, "is_constant"]:
                reason = "constant_or_not_applicable_to_c172p"
            elif any(s in col for s in forbidden_substrings):
                reason = "forbidden_input_family_section10"
            else:
                reason = "not_in_v1_feature_schema_sections14_15"
        catalog.loc[col, "role"] = role
        catalog.loc[col, "selected_for_baseline"] = role in ("neural_input", "label")
        if role == "excluded" and not reason:
            reason = "not_in_v1_feature_schema_sections14_15"
        if role == "excluded":
            catalog.loc[col, "exclusion_reason"] = reason


def _best_lag_residual(a: np.ndarray, b: np.ndarray, convert, max_lag: int = 2):
    """Try small sample shifts between two per-row-aligned series to absorb a
    one-tick property-tree logging stagger (confirmed empirically for
    altitude-agl-ft vs altitude-agl-m: see qa_summary notes). Returns the best
    (max_abs_error, lag) pair, evaluated only where both sides are finite.
    """
    best_err, best_lag = np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[:lag]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        mask = np.isfinite(aa) & np.isfinite(bb)
        if not mask.any():
            continue
        residual = bb[mask] - convert(aa[mask])
        err = float(np.nanmax(np.abs(residual)))
        if err < best_err:
            best_err, best_lag = err, lag
    return best_err, best_lag


def verify_unit_duplicates(all_df: pd.DataFrame) -> pd.DataFrame:
    """Numerically verify the curated candidate duplicate pairs.

    Computed PER SOURCE FILE (never shifting samples across a file boundary,
    since consecutive files are unrelated flights) then aggregated. A pair is
    confirmed a duplicate only if, using one single consistent lag across all
    files (0 unless noted), the worst per-file residual is numerically
    negligible relative to the kept column's own per-file scale. This
    absorbs the one-frame property-tree logging stagger empirically found
    between altitude-agl-ft and altitude-agl-m, without ever comparing
    samples that come from two different flights.
    """
    rows = []
    for cand in UNIT_DUPLICATE_CANDIDATES:
        keep, drop = cand["keep"], cand["drop"]
        if keep not in all_df.columns or drop not in all_df.columns:
            rows.append({
                "keep_column": keep, "drop_column": drop, "confirmed_duplicate": False,
                "formula": cand["formula"], "max_abs_error": np.nan, "lag_samples": np.nan,
                "note": "column missing from header",
            })
            continue

        per_file_best = []  # (source_file, best_lag, best_err, a_range)
        for src, g in all_df.groupby("source_file", sort=False):
            a = g[keep].to_numpy(dtype=float)
            b = g[drop].to_numpy(dtype=float)
            err, lag = _best_lag_residual(a, b, cand["convert"])
            finite_a = a[np.isfinite(a)]
            a_range = float(np.nanmax(finite_a) - np.nanmin(finite_a)) if len(finite_a) else np.nan
            per_file_best.append((src, lag, err, a_range))

        lags_used = sorted(set(l for _, l, e, _ in per_file_best if np.isfinite(e)))
        # pick the lag that is best (or tied) for the largest number of files
        from collections import Counter
        lag_votes = Counter(l for _, l, e, _ in per_file_best if np.isfinite(e))
        consensus_lag = lag_votes.most_common(1)[0][0] if lag_votes else 0

        # re-evaluate every file at the consensus lag to get a fair worst-case error
        worst_err, worst_file, worst_range = -np.inf, None, np.nan
        for src, g in all_df.groupby("source_file", sort=False):
            a = g[keep].to_numpy(dtype=float)
            b = g[drop].to_numpy(dtype=float)
            lag = consensus_lag
            if lag < 0:
                aa, bb = a[-lag:], b[:lag]
            elif lag > 0:
                aa, bb = a[:-lag], b[lag:]
            else:
                aa, bb = a, b
            mask = np.isfinite(aa) & np.isfinite(bb)
            if not mask.any():
                continue
            residual = bb[mask] - cand["convert"](aa[mask])
            err = float(np.nanmax(np.abs(residual)))
            finite_a = a[np.isfinite(a)]
            a_range = float(np.nanmax(finite_a) - np.nanmin(finite_a)) if len(finite_a) else np.nan
            if err > worst_err:
                worst_err, worst_file, worst_range = err, src, a_range

        tolerance = max(1e-3, 1e-3 * (worst_range if np.isfinite(worst_range) and worst_range > 0 else 1.0))
        confirmed = bool(np.isfinite(worst_err) and worst_err < tolerance)
        rows.append({
            "keep_column": keep, "drop_column": drop, "confirmed_duplicate": confirmed,
            "formula": cand["formula"], "max_abs_error": worst_err, "lag_samples": consensus_lag,
            "tolerance_used": tolerance,
            "note": cand["note"] + f" (per-file consensus lag={consensus_lag}; worst file={worst_file}; lags seen={lags_used})",
        })
    return pd.DataFrame(rows)


def find_exact_duplicate_columns(all_df: pd.DataFrame) -> pd.DataFrame:
    """Brute-force search for columns that are byte-for-byte identical
    (after NaN-aware comparison), excluding trivially constant columns
    (those are already captured by is_constant in the catalog).
    """
    numeric = all_df.select_dtypes(include=[np.number])
    cols = [c for c in numeric.columns if numeric[c].nunique(dropna=True) > 1]
    signatures = {}
    for c in cols:
        arr = numeric[c].to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=-999999.0, posinf=1e18, neginf=-1e18)
        signatures.setdefault(arr.round(9).tobytes(), []).append(c)
    rows = []
    for sig, members in signatures.items():
        if len(members) > 1:
            keep = members[0]
            for dup in members[1:]:
                rows.append({"keep_column": keep, "drop_column": dup,
                             "formula": "identity (byte-exact)", "confirmed_duplicate": True,
                             "max_abs_error": 0.0, "note": "exact duplicate values"})
    return pd.DataFrame(rows)


def check_control_ranges(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in CONTROL_RANGES.items():
        if col not in all_df.columns:
            continue
        s = all_df[col].astype(float)
        below = int((s < lo - 1e-6).sum())
        above = int((s > hi + 1e-6).sum())
        rows.append({
            "column_name": col, "expected_min": lo, "expected_max": hi,
            "observed_min": float(s.min()), "observed_max": float(s.max()),
            "n_below_range": below, "n_above_range": above,
        })
    return pd.DataFrame(rows)


def audit_file_timing(df: pd.DataFrame, source_file: str) -> dict:
    t = df[C.SIM_TIME]
    n = len(t)
    is_monotonic = bool(t.is_monotonic_increasing)
    n_dup_ts = int(t.duplicated().sum())
    dt = t.diff().dropna()
    eff_freq_hz = float(1.0 / dt.median()) if len(dt) else np.nan
    gap_threshold = 3 * dt.median() if len(dt) else np.nan
    n_gaps = int((dt > gap_threshold).sum()) if len(dt) else 0
    max_gap_s = float(dt.max()) if len(dt) else np.nan
    numeric = df.select_dtypes(include=[np.number])
    n_nan = int(numeric.isna().sum().sum())
    n_inf = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    return {
        "source_file": source_file,
        "n_rows": n,
        "t_start_s": float(t.iloc[0]) if n else np.nan,
        "t_end_s": float(t.iloc[-1]) if n else np.nan,
        "duration_s": float(t.iloc[-1] - t.iloc[0]) if n else np.nan,
        "is_monotonic": is_monotonic,
        "n_duplicate_timestamps": n_dup_ts,
        "effective_freq_hz_median": eff_freq_hz,
        "dt_mean_s": float(dt.mean()) if len(dt) else np.nan,
        "dt_std_s": float(dt.std()) if len(dt) else np.nan,
        "n_gaps_gt_3x_median_dt": n_gaps,
        "max_gap_s": max_gap_s,
        "n_nan_total": n_nan,
        "n_inf_total": n_inf,
    }
