"""Row-level landing state machine (spec section 5).

CONTACT_CANDIDATE -> TOUCHDOWN_CONFIRMED -> (BOUNCE)* -> ROLLOUT -> COMPLETE,
with loss of main-gear contact BEFORE confirmation looping back to FLARE
(recorded as an "unconfirmed/intermittent contact" event, not an episode end),
and loss of main-gear contact AFTER confirmation but before ROLLOUT classified
as BOUNCE (also looping back to FLARE, episode continues).

This replaces the previous single-timestamp-threshold phase painting in
pipeline/features.py::build_phase_and_masks, which is the root cause of the
known bug (spec section 4): because only ONE contact_candidate_s and ONE
touchdown_confirmed_s timestamp were recorded per episode, EVERY row between
the first main-gear touch and the eventual (possibly much later, after
several porpoise/bounce cycles) confirmed touchdown was painted
CONTACT_CANDIDATE -- including the airborne gaps in between where the
aircraft was not in contact with the ground at all.

Runs directly on whatever dataframe/sample rate it is given. It is used on
the 10 Hz processed grid (the same grid the neural-training datasets are
built from), so touchdown/rollout confirmation-duration thresholds are
translated into an EXPLICIT number of consecutive samples at that rate
(ceil(duration_s / dt_s) samples spanning duration_s of *covered* time,
i.e. N consecutive 10 Hz samples represent N * 0.1s -- not the older,
off-by-one-prone "timestamp of last sample minus timestamp of first sample"
convention). At exactly 10 Hz this means touchdown_confirm_s=0.3 requires
exactly 3 consecutive samples, and rollout_confirm_s=0.8 requires exactly 8.

Never uses the nose gear (cfg["gear"]["nose_index"]) to confirm CONTACT_CANDIDATE
or TOUCHDOWN_CONFIRMED -- only for the any_gear signal used once rollout
tracking begins (post-touchdown taxi/rollout naturally involves the nose
wheel too; it is not a touchdown-confirmation criterion).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline import columns as C

# Phases this module can emit as a row-level overlay (on top of the
# FINAL_APPROACH_ESTABLISHED/FLARE scalar-threshold phases computed upstream).
LANDING_PHASES = ["CONTACT_CANDIDATE", "TOUCHDOWN_CONFIRMED", "BOUNCE", "ROLLOUT", "COMPLETE"]


def _min_samples(duration_s: float, dt_s: float) -> int:
    """Number of consecutive samples at spacing dt_s whose covered duration
    (count * dt_s) first reaches duration_s. Explicit, rounds UP (never
    truncates), so e.g. 0.3s at 10Hz (dt=0.1s) -> 3 samples exactly, never 2."""
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    return max(1, math.ceil(round(duration_s / dt_s, 6)))


def _gear_cols(df: pd.DataFrame, main_idx, nose_idx):
    w_main = [(df[C.gear_wow(i)].to_numpy() == 1) for i in main_idx]
    w_nose = df[C.gear_wow(nose_idx)].to_numpy() == 1
    return w_main, w_nose


def run_landing_fsm(df: pd.DataFrame, start_idx: int, end_idx: int, cfg: dict) -> dict:
    """Run the state machine over df.iloc[start_idx:end_idx+1] (absolute,
    inclusive indices into df). `start_idx` should be the row at/after the
    FLARE crossing; rows before it are not this function's concern.

    Returns a dict:
      touchdown_confirmed: bool
      contact_candidate_first_idx: int | None
      touchdown_confirmed_idx: int | None
      rollout_idx: int | None
      complete_idx: int | None
      complete_reason: str | None
      row_phase: np.ndarray[object], length end_idx-start_idx+1, aligned to
        df.iloc[start_idx:end_idx+1]; entries are one of LANDING_PHASES or
        None (None means "still FLARE" -- caller fills that in).
      events: list[dict] -- transition table rows (spec section 6).
      bounce_records: list[dict] -- unconfirmed-contact and bounce episodes,
        with start/end time, duration, and max AGL during the airborne gap
        (spec section 5's BOUNCE record requirements).
      review_flags: list[str] -- non-fatal QA flags requiring human review.
    """
    seg = cfg["landing_segmentation"]
    main_idx = cfg["gear"]["main_indices"]
    nose_idx = cfg["gear"]["nose_index"]

    t = df[C.SIM_TIME].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    gs = df[C.GROUNDSPEED_KT].to_numpy()
    agl = df[C.ALT_AGL_FT].to_numpy()
    thr = df[C.THROTTLE_ENGINE0].to_numpy()
    has_vs = C.VERTICAL_SPEED_FPS in df.columns
    vs = df[C.VERTICAL_SPEED_FPS].to_numpy() if has_vs else None
    has_frame = "frame" in df.columns

    w_main, w_nose = _gear_cols(df, main_idx, nose_idx)
    main_any = w_main[0] | w_main[1]
    both_main = w_main[0] & w_main[1]
    any_gear = w_nose | main_any

    comp_cols = {}
    for label, i in (("main_left", main_idx[0]), ("main_right", main_idx[1]), ("nose", nose_idx)):
        col = C.gear_compression(i)
        comp_cols[label] = df[col].to_numpy() if col in df.columns else None

    touchdown_min_samples = _min_samples(seg["touchdown_confirm_s"], dt)
    rollout_min_samples = _min_samples(seg["rollout_confirm_s"], dt)
    complete_min_samples = _min_samples(seg["complete_duration_s"], dt)
    liftoff_min_samples = _min_samples(seg["liftoff_complete_s"], dt)
    diag_long_candidate_s = seg.get("diagnostic_long_candidate_s", 2.0)

    n = end_idx - start_idx + 1
    row_phase = np.full(n, None, dtype=object)

    def set_phase(a, b, label):
        row_phase[a - start_idx: b - start_idx + 1] = label

    def mk_event(prev_state, new_state, idx, reason):
        return {
            "previous_state": prev_state, "new_state": new_state,
            "frame": int(df["frame"].iloc[idx]) if has_frame else None,
            "sim_time_s": float(t[idx]),
            "agl_ft": float(agl[idx]),
            "groundspeed_kt": float(gs[idx]),
            "vertical_speed_fps": float(vs[idx]) if has_vs else None,
            "wow_main_left": bool(w_main[0][idx]), "wow_main_right": bool(w_main[1][idx]),
            "wow_nose": bool(w_nose[idx]),
            "compression_main_left": float(comp_cols["main_left"][idx]) if comp_cols["main_left"] is not None else None,
            "compression_main_right": float(comp_cols["main_right"][idx]) if comp_cols["main_right"] is not None else None,
            "compression_nose": float(comp_cols["nose"][idx]) if comp_cols["nose"] is not None else None,
            "reason": reason,
        }

    events: list[dict] = []
    review_flags: list[str] = []
    bounce_records: list[dict] = []

    first_candidate_idx = None
    touchdown_idx = None
    rollout_idx = None
    complete_idx = None
    complete_reason = None

    # ---- Phase A: candidate / unconfirmed-contact loop, until touchdown confirmed ----
    k = start_idx
    confirmed = False
    while k <= end_idx and not confirmed:
        if not main_any[k]:
            k += 1
            continue
        run_start = k
        is_first = first_candidate_idx is None
        if is_first:
            first_candidate_idx = run_start
        events.append(mk_event(
            "FLARE", "CONTACT_CANDIDATE", run_start,
            "first_main_gear_contact" if is_first else "renewed_main_gear_contact_after_unconfirmed_loss"))

        # Scan forward WITHIN this main-gear contact attempt (main_any stays
        # true) for a qualifying both-main sub-run, stopping as soon as one
        # is found -- do NOT keep consuming the rest of the contact run first
        # (that portion is TOUCHDOWN_CONFIRMED/ROLLOUT territory, handled by
        # Phase B; greedily pre-consuming it here was a bug that swallowed
        # the entire post-touchdown rollout into a single TOUCHDOWN_CONFIRMED
        # label and starved Phase B of any rows to evaluate for ROLLOUT).
        qualifying = None
        j = run_start
        while j <= end_idx and main_any[j]:
            if both_main[j]:
                bj = j
                while j <= end_idx and both_main[j]:
                    j += 1
                if (j - bj) >= touchdown_min_samples:
                    qualifying = (bj, j - 1)
                    break
            else:
                j += 1
        run_end = j - 1  # end of the contact attempt IF it never confirms (main_any went false at j)

        if qualifying is not None:
            touchdown_idx = qualifying[0]
            if touchdown_idx > run_start:
                set_phase(run_start, touchdown_idx - 1, "CONTACT_CANDIDATE")
            set_phase(touchdown_idx, qualifying[1], "TOUCHDOWN_CONFIRMED")
            events.append(mk_event(
                "CONTACT_CANDIDATE", "TOUCHDOWN_CONFIRMED", touchdown_idx,
                f"both_main_gear_in_contact_for_{touchdown_min_samples}_consecutive_samples"
                f"(>={seg['touchdown_confirm_s']}s_at_dt={dt:.4f}s)"))
            confirmed = True
            # Phase B re-scans any_gear starting exactly at touchdown_idx (the
            # instant continuous contact began, not the later confirmation
            # instant) so "0.8s in total" naturally overlaps/includes the
            # initial 0.3s confirmation window, matching the original
            # segment_circuit.py convention.
            k = touchdown_idx
        else:
            set_phase(run_start, run_end, "CONTACT_CANDIDATE")
            duration_s = (run_end - run_start + 1) * dt
            events.append(mk_event(
                "CONTACT_CANDIDATE", "FLARE", run_end,
                "contact_lost_before_touchdown_confirmation_unconfirmed_intermittent_contact"))
            gscan = run_end + 1
            while gscan <= end_idx and not main_any[gscan]:
                gscan += 1
            gap_end = gscan - 1
            max_agl_gap = float(np.max(agl[run_end + 1:gap_end + 1])) if gap_end >= run_end + 1 else None
            bounce_records.append({
                "kind": "unconfirmed_intermittent_contact",
                "start_s": float(t[run_start]), "end_s": float(t[run_end]),
                "duration_s": float(duration_s), "max_agl_ft_during_gap": max_agl_gap,
            })
            if duration_s >= diag_long_candidate_s:
                review_flags.append(
                    f"long_unconfirmed_CONTACT_CANDIDATE_interval: t=[{t[run_start]:.2f},{t[run_end]:.2f}]s "
                    f"duration={duration_s:.2f}s (>= diagnostic threshold {diag_long_candidate_s}s)")
            k = run_end + 1  # still airborne (FLARE) until next contact or window end.

    if not confirmed:
        if first_candidate_idx is None:
            return {
                "touchdown_confirmed": False, "contact_candidate_first_idx": None,
                "touchdown_confirmed_idx": None, "rollout_idx": None,
                "complete_idx": None, "complete_reason": None,
                "row_phase": row_phase, "events": events,
                "bounce_records": bounce_records,
                "review_flags": ["no_main_gear_contact_found_in_window"],
            }
        # By loop construction, falling through to here always means k has
        # run past end_idx (the only OTHER way to exit Phase A's while loop
        # is confirmed=True) -- every candidate contact run actually seen was
        # already labeled CONTACT_CANDIDATE, and every gap between them is
        # correctly left as the airborne default (FLARE). There is no
        # trailing "unresolved" span left to separately label:
        # TOUCHDOWN_CONFIRMED/ROLLOUT/COMPLETE simply never get set for this
        # episode, which already keeps it out of every rollout/complete-only
        # downstream use; CONTACT_CANDIDATE and FLARE rows remain legitimate
        # training-eligible data (spec section 7).
        review_flags.append(
            "touchdown_never_confirmed: both main gear were never simultaneously in contact for "
            f">={touchdown_min_samples} consecutive samples (>={seg['touchdown_confirm_s']}s) anywhere in the "
            "window. ROLLOUT/COMPLETE could not be determined for this episode (excluded from any "
            "rollout/complete-dependent use); FLARE/CONTACT_CANDIDATE rows remain valid for training. "
            "Manual review required -- do not force a proxy touchdown criterion.")
        return {
            "touchdown_confirmed": False, "contact_candidate_first_idx": first_candidate_idx,
            "touchdown_confirmed_idx": None, "rollout_idx": None,
            "complete_idx": None, "complete_reason": None,
            "row_phase": row_phase, "events": events,
            "bounce_records": bounce_records, "review_flags": review_flags,
        }

    # ---- Phase B: confirmed -- track BOUNCE (confirmed contact subsequently
    # lost) vs sustained contact reaching ROLLOUT ----
    while k <= end_idx and rollout_idx is None:
        if any_gear[k]:
            run_start = k
            while k <= end_idx and any_gear[k]:
                k += 1
            run_end = k - 1
            if (run_end - run_start + 1) >= rollout_min_samples:
                rollout_idx = run_start
                set_phase(run_start, run_end, "ROLLOUT")
                events.append(mk_event(
                    "TOUCHDOWN_CONFIRMED", "ROLLOUT", rollout_idx,
                    f"any_gear_in_contact_for_{rollout_min_samples}_consecutive_samples"
                    f"(>={seg['rollout_confirm_s']}s_at_dt={dt:.4f}s)"))
            else:
                set_phase(run_start, run_end, "TOUCHDOWN_CONFIRMED")
        else:
            run_start = k
            while k <= end_idx and not any_gear[k]:
                k += 1
            run_end = k - 1
            set_phase(run_start, run_end, "BOUNCE")
            max_agl = float(np.max(agl[run_start:run_end + 1]))
            duration_s = (run_end - run_start + 1) * dt
            events.append(mk_event("TOUCHDOWN_CONFIRMED", "BOUNCE", run_start, "confirmed_main_gear_contact_lost"))
            events.append(mk_event("BOUNCE", "FLARE", run_end, "bounce_recoverable_return_to_flare"))
            bounce_records.append({
                "kind": "bounce", "start_s": float(t[run_start]), "end_s": float(t[run_end]),
                "duration_s": float(duration_s), "max_agl_ft_during_gap": max_agl,
            })

    if rollout_idx is None:
        review_flags.append(
            "window_ended_before_rollout_confirmed_after_touchdown (recording likely truncated at/near touchdown)")
        return {
            "touchdown_confirmed": True, "contact_candidate_first_idx": first_candidate_idx,
            "touchdown_confirmed_idx": touchdown_idx, "rollout_idx": None,
            "complete_idx": None, "complete_reason": None,
            "row_phase": row_phase, "events": events,
            "bounce_records": bounce_records, "review_flags": review_flags,
        }

    # ---- Phase C: COMPLETE -- sustained low groundspeed, or sustained
    # liftoff (touch-and-go) after rollout ----
    k = rollout_idx
    while k <= end_idx:
        if gs[k] < seg["complete_groundspeed_kt"]:
            run_start = k
            while k <= end_idx and gs[k] < seg["complete_groundspeed_kt"]:
                k += 1
            run_end = k - 1
            if (run_end - run_start + 1) >= complete_min_samples:
                complete_idx = run_start
                complete_reason = "groundspeed_below_threshold"
                break
        elif not any_gear[k]:
            run_start = k
            while k <= end_idx and not any_gear[k]:
                k += 1
            run_end = k - 1
            gs_accelerating = gs[run_end] > gs[run_start] + 2.0
            throttle_applied = thr[run_end] >= 0.6
            real_liftoff = (
                (run_end - run_start + 1) >= liftoff_min_samples and
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
        complete_idx = end_idx
        complete_reason = "end_of_recording_window"
    set_phase(complete_idx, end_idx, "COMPLETE")
    events.append(mk_event("ROLLOUT", "COMPLETE", complete_idx, complete_reason))

    return {
        "touchdown_confirmed": True, "contact_candidate_first_idx": first_candidate_idx,
        "touchdown_confirmed_idx": touchdown_idx, "rollout_idx": rollout_idx,
        "complete_idx": complete_idx, "complete_reason": complete_reason,
        "row_phase": row_phase, "events": events,
        "bounce_records": bounce_records, "review_flags": review_flags,
    }
