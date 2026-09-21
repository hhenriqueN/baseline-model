"""Stage 3: detect the two landings inside circuito-completo.csv.

Implements the state machine from spec section 4.1:
  CONTACT_CANDIDATE -> TOUCHDOWN_CONFIRMED -> (BOUNCE)* -> ROLLOUT -> COMPLETE

and separates the file into two independent search windows using the long
airborne gap between the two landings (climb-out, circuit, re-approach).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import columns as C


def find_airborne_gap(df: pd.DataFrame, main_indices, gap_min_s: float, agl_ft_thresh: float):
    """Find the single longest contiguous stretch where all main gear are off
    the ground AND agl > agl_ft_thresh, lasting at least gap_min_s. This is
    used only to split the file into two independent landing-search windows;
    it is NOT itself a segmentation boundary.
    """
    wow_cols = [C.gear_wow(i) for i in main_indices]
    any_main_contact = (df[wow_cols].sum(axis=1) > 0).to_numpy()
    agl = df[C.ALT_AGL_FT].to_numpy()
    airborne = (~any_main_contact) & (agl > agl_ft_thresh)

    t = df[C.SIM_TIME].to_numpy()
    best = None
    i = 0
    n = len(airborne)
    while i < n:
        if airborne[i]:
            j = i
            while j < n and airborne[j]:
                j += 1
            duration = t[j - 1] - t[i]
            if duration >= gap_min_s and (best is None or duration > best[2]):
                best = (i, j - 1, duration)
            i = j
        else:
            i += 1
    return best  # (start_idx, end_idx, duration_s) or None


def _runs_of_true(mask: np.ndarray):
    """Yield (start_idx, end_idx_inclusive) for contiguous True runs."""
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield (i, j - 1)
            i = j
        else:
            i += 1


def segment_one_landing(df: pd.DataFrame, window: tuple[int, int], cfg: dict, search_start_override: int | None = None):
    """Run the CONTACT_CANDIDATE/TOUCHDOWN_CONFIRMED/BOUNCE/ROLLOUT/COMPLETE
    state machine within df.iloc[window[0]:window[1]+1].

    Returns a dict of detected sample indices (absolute, into df) and an
    events list of (event_name, idx, sim_time) for plotting/diagnostics.
    """
    seg = cfg["landing_segmentation"]
    main_idx = cfg["gear"]["main_indices"]
    nose_idx = cfg["gear"]["nose_index"]
    wow_main = [C.gear_wow(i) for i in main_idx]
    wow_all = [C.gear_wow(nose_idx)] + wow_main

    lo, hi = window
    t = df[C.SIM_TIME].to_numpy()
    gs = df[C.GROUNDSPEED_KT].to_numpy()
    agl = df[C.ALT_AGL_FT].to_numpy()
    thr = df[C.THROTTLE_ENGINE0].to_numpy()
    both_main = (df[wow_main[0]].to_numpy() == 1) & (df[wow_main[1]].to_numpy() == 1)
    any_gear = (df[wow_all].sum(axis=1) > 0).to_numpy()

    events = []
    start = search_start_override if search_start_override is not None else lo

    # --- CONTACT_CANDIDATE: first sample in window with any main gear WOW=1
    main_any = (df[wow_main[0]].to_numpy() == 1) | (df[wow_main[1]].to_numpy() == 1)
    cand_idx = None
    for k in range(start, hi + 1):
        if main_any[k]:
            cand_idx = k
            break
    if cand_idx is None:
        return None, events
    events.append(("CONTACT_CANDIDATE", cand_idx, t[cand_idx]))

    # --- scan forward through candidate/bounce cycles to find TOUCHDOWN_CONFIRMED.
    # Tier 1 (spec default): both main gear in contact continuously for
    # >= touchdown_confirm_s. Tier 2 (fallback for asymmetric/crosswind
    # touchdowns, e.g. wing-low crosswind technique or gusty porpoising,
    # where the two mains rarely settle down together): either single main
    # gear continuously for >= touchdown_confirm_s. Tier 3 (last resort,
    # for landings that porpoise/bounce for several cycles before settling):
    # treat the eventual ROLLOUT-qualifying run (>= rollout_confirm_s of any
    # gear contact) as the touchdown point itself. Every BOUNCE observed
    # along the way (tier-1 sense) is still recorded as an event regardless
    # of which tier ultimately confirms touchdown.
    def _scan_runs(mask, min_duration, from_idx):
        k = from_idx
        while k <= hi:
            if mask[k]:
                run_start = k
                while k <= hi and mask[k]:
                    k += 1
                run_end = k - 1
                duration = t[run_end] - t[run_start]
                if duration >= min_duration:
                    return run_start, run_end
                else:
                    events.append(("BOUNCE", run_end, t[run_end]))
            else:
                k += 1
        return None

    touchdown_idx = None
    touchdown_tier = None
    found = _scan_runs(both_main, seg["touchdown_confirm_s"], cand_idx)
    if found is not None:
        touchdown_idx = found[0]
        touchdown_tier = "tier1_both_main_gear"
    else:
        found = _scan_runs(main_any, seg["touchdown_confirm_s"], cand_idx)
        if found is not None:
            touchdown_idx = found[0]
            touchdown_tier = "tier2_either_main_gear"
        else:
            found = _scan_runs(any_gear, seg["rollout_confirm_s"], cand_idx)
            if found is not None:
                touchdown_idx = found[0]
                touchdown_tier = "tier3_rollout_proxy_bouncy_landing"
    if touchdown_idx is None:
        return None, events
    events.append(("TOUCHDOWN_CONFIRMED", touchdown_idx, t[touchdown_idx]))

    # --- ROLLOUT: first run of continuous any-gear contact (from touchdown
    # onward, allowing intervening BOUNCE) lasting >= rollout_confirm_s
    rollout_idx = None
    k = touchdown_idx
    while k <= hi:
        if any_gear[k]:
            run_start = k
            while k <= hi and any_gear[k]:
                k += 1
            run_end = k - 1
            duration = t[run_end] - t[run_start]
            if duration >= seg["rollout_confirm_s"]:
                rollout_idx = run_start
                break
        else:
            k += 1
    if rollout_idx is None:
        rollout_idx = touchdown_idx  # degenerate but keep going
    events.append(("ROLLOUT", rollout_idx, t[rollout_idx]))

    # --- COMPLETE: groundspeed < threshold for complete_duration_s, OR
    # sustained liftoff (touch-and-go) for liftoff_complete_s, OR end of window
    complete_idx = None
    complete_reason = None
    k = rollout_idx
    while k <= hi:
        if gs[k] < seg["complete_groundspeed_kt"]:
            run_start = k
            while k <= hi and gs[k] < seg["complete_groundspeed_kt"]:
                k += 1
            run_end = k - 1
            if t[run_end] - t[run_start] >= seg["complete_duration_s"]:
                complete_idx = run_start
                complete_reason = "groundspeed_below_threshold"
                break
        elif not any_gear[k]:
            run_start = k
            while k <= hi and not any_gear[k]:
                k += 1
            run_end = k - 1
            # Distinguish a genuine touch-and-go / go-around (power applied,
            # groundspeed increasing, sustained climb) from a mere porpoise
            # bounce during rollout (throttle stays low, groundspeed keeps
            # decaying, AGL rise is only a few feet). Both can transiently
            # satisfy "any_gear false for >= liftoff_complete_s with agl
            # rising"; only the former is a real episode-ending liftoff.
            gs_accelerating = gs[run_end] > gs[run_start] + 2.0
            throttle_applied = thr[run_end] >= 0.6
            real_liftoff = (
                t[run_end] - t[run_start] >= seg["liftoff_complete_s"] and
                agl[run_end] > agl[run_start] and
                (gs_accelerating or throttle_applied)
            )
            if real_liftoff:
                complete_idx = run_start
                complete_reason = "liftoff_after_rollout_touch_and_go"
                break
            k = run_end + 1
        else:
            k += 1
    if complete_idx is None:
        complete_idx = hi
        complete_reason = "end_of_recording_window"
    events.append(("COMPLETE", complete_idx, t[complete_idx]))

    result = {
        "contact_candidate_idx": cand_idx,
        "touchdown_confirmed_idx": touchdown_idx,
        "touchdown_confirmation_tier": touchdown_tier,
        "rollout_idx": rollout_idx,
        "complete_idx": complete_idx,
        "complete_reason": complete_reason,
        "episode_start_idx": lo,
        "episode_end_idx": complete_idx,
    }
    return result, events
