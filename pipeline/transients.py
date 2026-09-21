"""Stage 6.1 (spec section 6.1): detect where the artificial initial
transient ends for episodes that start in the standardized ~3km/800ft
position (13 straight landings + circuito-completo__landing-01).

The transient is considered over once the aircraft holds, continuously for
`min_stable_duration_s`, ALL of:
  - airspeed within airspeed_band_kt
  - flaps non-null (> flaps_nonzero_min) and stable (rolling std below
    flaps_stable_std_max)
  - inbound trajectory (along_track_m < 0, i.e. still approaching threshold)
  - enough approach time remaining before touchdown
Never uses aileron/elevator/rudder to pick "pretty" actions (per spec).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import columns as C
from pipeline.runway import RunwayConfig, along_cross_track


def detect_transient_end(df: pd.DataFrame, cfg: dict, runway_cfg: RunwayConfig, touchdown_idx: int) -> dict:
    tcfg = cfg["initialization_transient"]
    t = df[C.SIM_TIME].to_numpy()
    fs = 1.0 / np.median(np.diff(t))

    airspeed = df[C.AIRSPEED_KT].to_numpy()
    flaps = df[C.FLAPS_CMD].to_numpy()
    along_m, _cross_m = along_cross_track(df[C.LAT].to_numpy(), df[C.LON].to_numpy(), runway_cfg)

    win = max(1, int(round(1.0 * fs)))  # 1-second rolling window for flap stability
    flaps_roll_std = pd.Series(flaps).rolling(win, min_periods=win).std().to_numpy()

    lo, hi = airspeed_band = tcfg["airspeed_band_kt"]
    cond = (
        (airspeed >= lo) & (airspeed <= hi) &
        (flaps > tcfg["flaps_nonzero_min"]) &
        (flaps_roll_std < tcfg["flaps_stable_std_max"]) &
        (along_m < 0) &
        ((t[touchdown_idx] - t) >= tcfg["min_remaining_approach_s"])
    )

    max_search_idx = np.searchsorted(t, t[0] + tcfg["max_search_s"], side="right")
    max_search_idx = min(max_search_idx, touchdown_idx)

    n = len(t)
    result = {
        "detected": False, "transient_end_idx": None, "reason": "",
        "airspeed_before": float(airspeed[0]), "airspeed_after": None,
        "flaps_before": float(flaps[0]), "flaps_after": None,
        "along_track_m_at_end": None, "agl_ft_at_end": None,
    }
    k = 0
    while k <= max_search_idx:
        if cond[k]:
            run_start = k
            while k <= max_search_idx and cond[k]:
                k += 1
            run_end = k - 1
            if t[run_end] - t[run_start] >= tcfg["min_stable_duration_s"]:
                result["detected"] = True
                result["transient_end_idx"] = int(run_start)
                result["reason"] = "auto_detected"
                result["airspeed_after"] = float(airspeed[run_start])
                result["flaps_after"] = float(flaps[run_start])
                result["along_track_m_at_end"] = float(along_m[run_start])
                result["agl_ft_at_end"] = float(df[C.ALT_AGL_FT].iloc[run_start])
                break
        else:
            k += 1
    if not result["detected"]:
        result["reason"] = "not_detected_within_max_search_window_keep_full_episode_REVIEW_NEEDED"
    return result
