"""Stage 6: causal anti-alias filtering + resampling to a regular 10 Hz grid.

Design choices (documented here, not silently assumed):
  - Anti-alias filter: causal (one-directional) Butterworth low-pass, applied
    with scipy.signal.lfilter (never filtfilt, which needs future samples).
  - Placing filtered values on the 10 Hz output grid: for each output grid
    timestamp t_k, we use the filtered value at the MOST RECENT raw sample
    with t_raw <= t_k ("causal hold" / zero-order hold on the anti-alias
    output). True linear interpolation between (t_i, t_{i+1}) was avoided
    because when t_i <= t_k < t_{i+1}, t_{i+1} is strictly in the future
    relative to t_k -- exactly what the spec prohibits ("sem usar amostras
    futuras"). Because the raw rate (~60 Hz) is ~6x the 10 Hz target grid,
    the extra error from holding instead of interpolating is at most one
    raw sample period (~16.7 ms), an order of magnitude below the 100 ms
    grid spacing.
  - Discrete signals (WOW) are never filtered, only causally held, so real
    ground-contact transitions are preserved exactly (never smeared).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from pipeline import columns as C
from pipeline.audit import CONTROL_RANGES

DISCRETE_COLUMNS = {C.gear_wow(0), C.gear_wow(1), C.gear_wow(2), C.FLAPS_CMD}
# FLAPS_CMD is held (not anti-alias filtered): it moves between a handful of
# fixed notches (0/10/20/30 deg normalized), and applying the causal
# Butterworth filter to those step transitions produced small ringing
# overshoot/undershoot (~+-1.4%) outside the natural [0,1] range -- observed
# empirically as e.g. flaps_state_norm reaching 1.014 or -0.011. Spec section
# 12 already groups flaps with WOW as a transition that must be preserved
# exactly, which supports treating it as a held/discrete-style signal here.


def design_causal_filter(fs_hz: float, cutoff_hz: float, order: int):
    nyq = fs_hz / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(order, wn, btype="low", analog=False)
    return b, a


def estimate_group_delay_ms(b, a, cutoff_hz: float, fs_hz: float) -> float:
    w, gd = signal.group_delay((b, a), w=[2 * np.pi * cutoff_hz / fs_hz], fs=2 * np.pi)
    samples = gd[0]
    return float(samples / fs_hz * 1000.0)


def resample_episode(df: pd.DataFrame, numeric_cols, meta_cols, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Causally filter + resample one episode's validated dataframe.

    Returns (resampled_df, diagnostics_dict).
    """
    rcfg = cfg["resampling"]
    t_raw = df[C.SIM_TIME].to_numpy()
    dt_median = np.median(np.diff(t_raw))
    fs_hz = 1.0 / dt_median
    cutoff_hz = rcfg["cutoff_hz"]
    order = rcfg["filter_order"]
    b, a = design_causal_filter(fs_hz, cutoff_hz, order)
    group_delay_ms = estimate_group_delay_ms(b, a, cutoff_hz, fs_hz)

    dt_target = 1.0 / rcfg["target_hz"]
    t0, t1 = t_raw[0], t_raw[-1]
    n_grid = int(np.floor((t1 - t0) / dt_target)) + 1
    t_grid = t0 + np.arange(n_grid) * dt_target

    hold_idx = np.searchsorted(t_raw, t_grid, side="right") - 1
    hold_idx = np.clip(hold_idx, 0, len(t_raw) - 1)

    out = {C.SIM_TIME: t_grid, "frame": np.arange(n_grid)}
    for col in numeric_cols:
        raw_vals = df[col].to_numpy(dtype=float)
        if col in DISCRETE_COLUMNS:
            filtered = raw_vals  # never filtered
        else:
            filtered = signal.lfilter(b, a, raw_vals)
            if col in CONTROL_RANGES:
                # The causal Butterworth filter can ring a few % past a hard
                # physical bound (e.g. throttle briefly < 0) around sharp
                # step changes. These quantities are physically clamped in
                # the simulator, so re-clip after filtering rather than let
                # an anti-aliasing artifact leak into a bounded label/input.
                lo, hi = CONTROL_RANGES[col]
                filtered = np.clip(filtered, lo, hi)
        out[col] = filtered[hold_idx]

    resampled = pd.DataFrame(out)
    for col in meta_cols:
        resampled[col] = df[col].iloc[0]  # constant per episode

    diagnostics = {
        "n_raw_samples": len(df),
        "n_resampled_samples": n_grid,
        "effective_fs_raw_hz": fs_hz,
        "filter_order": order,
        "cutoff_hz": cutoff_hz,
        "group_delay_ms_at_cutoff": group_delay_ms,
        "target_hz": rcfg["target_hz"],
    }
    return resampled, diagnostics
