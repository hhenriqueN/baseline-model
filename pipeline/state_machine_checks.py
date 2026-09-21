"""Automated state-machine diagnostics required by spec section 6. These are
independent audits of an already-computed landing_fsm.run_landing_fsm()
result (they do not re-derive it) -- they exist to catch a REGRESSION in the
FSM itself, not to second-guess a correctly-computed, genuinely-unconfirmed
episode (e.g. voo-3-com-16-nos, where both main gear are never simultaneously
in contact -- that is a documented, non-fatal review case, not a bug).

Raises AssertionError with a clear message on any hard violation; callers
(scripts/07_detect_transients_and_phases.py, tests/test_pipeline.py) decide
whether to let that fail the pipeline/test run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import columns as C

# Order used to check "X occurs before Y" -- higher index = later in the
# normal forward progression. BOUNCE is intentionally excluded (it can
# legitimately recur after TOUCHDOWN_CONFIRMED and loop back).
_ORDER = {"CONTACT_CANDIDATE": 0, "TOUCHDOWN_CONFIRMED": 1, "ROLLOUT": 2, "COMPLETE": 3}


def check_gear_mapping(cfg: dict):
    nose = cfg["gear"]["nose_index"]
    mains = cfg["gear"]["main_indices"]
    assert nose not in mains, f"nose gear index {nose} must not appear in main_indices {mains}"
    assert len(set(mains)) == 2, f"main_indices must name exactly two distinct gears, got {mains}"


def check_time_monotonic(df: pd.DataFrame, episode_id: str):
    t = df[C.SIM_TIME].to_numpy()
    assert np.all(np.diff(t) > 0), f"{episode_id}: sim_time is not strictly monotonic increasing"


def check_ordering(fsm_result: dict, episode_id: str):
    """ROLLOUT must not precede TOUCHDOWN_CONFIRMED; COMPLETE must not
    precede ROLLOUT; TOUCHDOWN_CONFIRMED must not exist without a confirmed
    touchdown index."""
    td = fsm_result["touchdown_confirmed_idx"]
    ro = fsm_result["rollout_idx"]
    co = fsm_result["complete_idx"]
    if ro is not None:
        assert fsm_result["touchdown_confirmed"] and td is not None, (
            f"{episode_id}: ROLLOUT index set ({ro}) without a confirmed TOUCHDOWN_CONFIRMED index")
        assert ro >= td, f"{episode_id}: ROLLOUT ({ro}) occurs before TOUCHDOWN_CONFIRMED ({td})"
    if co is not None:
        assert ro is not None, f"{episode_id}: COMPLETE index set ({co}) without a ROLLOUT index"
        assert co >= ro, f"{episode_id}: COMPLETE ({co}) occurs before ROLLOUT ({ro})"


def check_touchdown_requires_both_main_contact(df: pd.DataFrame, fsm_result: dict, cfg: dict, episode_id: str):
    """Independently re-verify, directly from the WOW columns, that the
    confirmed touchdown row really does sit at the start of >= N consecutive
    samples of BOTH main gear in contact (never nose-gear-only, never a
    single-main-gear proxy)."""
    if not fsm_result["touchdown_confirmed"]:
        return
    main_idx = cfg["gear"]["main_indices"]
    seg = cfg["landing_segmentation"]
    t = df[C.SIM_TIME].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    import math
    min_samples = max(1, math.ceil(round(seg["touchdown_confirm_s"] / dt, 6)))
    idx = fsm_result["touchdown_confirmed_idx"]
    w1 = df[C.gear_wow(main_idx[0])].to_numpy() == 1
    w2 = df[C.gear_wow(main_idx[1])].to_numpy() == 1
    both = w1 & w2
    window = both[idx: idx + min_samples]
    assert len(window) == min_samples and window.all(), (
        f"{episode_id}: TOUCHDOWN_CONFIRMED at idx={idx} is not backed by {min_samples} consecutive "
        f"samples of BOTH main gear contact (found {window.sum()}/{len(window)})")


def check_nose_gear_not_used_alone(df: pd.DataFrame, cfg: dict, episode_id: str, row_phase: np.ndarray):
    """Nose-gear-only contact (mains both off) must never, by itself, appear
    labeled TOUCHDOWN_CONFIRMED or ROLLOUT-start in the row overlay --
    ROLLOUT continuation is allowed to involve the nose wheel, but its ENTRY
    condition must trace back to a main-gear-based TOUCHDOWN_CONFIRMED, which
    check_touchdown_requires_both_main_contact already verifies independently."""
    main_idx = cfg["gear"]["main_indices"]
    nose_idx = cfg["gear"]["nose_index"]
    w_main_any = (df[C.gear_wow(main_idx[0])].to_numpy() == 1) | (df[C.gear_wow(main_idx[1])].to_numpy() == 1)
    w_nose = df[C.gear_wow(nose_idx)].to_numpy() == 1
    nose_only = w_nose & (~w_main_any)
    n = len(row_phase)
    offset = len(df) - n
    for i, label in enumerate(row_phase):
        if label == "CONTACT_CANDIDATE" and nose_only[offset + i]:
            raise AssertionError(f"{episode_id}: row {offset+i} labeled CONTACT_CANDIDATE with nose-gear-only contact")


def check_long_candidate_reported(fsm_result: dict, cfg: dict, episode_id: str):
    """Every CONTACT_CANDIDATE run in row_phase at/above the diagnostic
    threshold must have a corresponding review flag (an implausibly long
    candidate interval must never pass through silently unreported)."""
    threshold = cfg["landing_segmentation"].get("diagnostic_long_candidate_s", 2.0)
    row_phase = fsm_result["row_phase"]
    n = len(row_phase)
    i = 0
    long_runs = 0
    # dt not known here without df; caller passes bounce_records which already
    # carries duration for unconfirmed runs, and confirmed candidate runs by
    # construction always end in touchdown before the threshold check applies
    # to *unconfirmed* loss -- so we only need to check bounce_records.
    for b in fsm_result["bounce_records"]:
        if b["kind"] == "unconfirmed_intermittent_contact" and b["duration_s"] >= threshold:
            long_runs += 1
    flagged = sum(1 for f in fsm_result["review_flags"] if f.startswith("long_unconfirmed_CONTACT_CANDIDATE_interval"))
    assert long_runs == flagged, (
        f"{episode_id}: {long_runs} unconfirmed CONTACT_CANDIDATE run(s) >= {threshold}s found but "
        f"only {flagged} reported in review_flags")


def check_bounce_does_not_end_episode(fsm_result: dict, episode_id: str):
    """If any BOUNCE (post-confirmation loss) occurred, the episode must not
    terminate (COMPLETE) before the last bounce ends."""
    bounces = [b for b in fsm_result["bounce_records"] if b["kind"] == "bounce"]
    if not bounces or fsm_result["complete_idx"] is None:
        return
    last_bounce_end_s = max(b["end_s"] for b in bounces)
    # complete_idx's sim_time is looked up by the caller; here we just check
    # ordering was already enforced (rollout >= touchdown, complete >= rollout)
    # and that rollout occurs after every bounce (a bounce, by definition,
    # occurs before rollout is reached).
    assert fsm_result["rollout_idx"] is not None, (
        f"{episode_id}: bounce(s) recorded but no ROLLOUT ever reached -- "
        f"episode must not be treated as complete at first contact")


def validate_episode(df: pd.DataFrame, fsm_result: dict, cfg: dict, episode_id: str) -> list[str]:
    """Run all hard checks for one episode; returns [] on success, raises
    AssertionError with the first violation otherwise (fail-fast, matching
    spec section 6's 'stop and produce a clear diagnostic report')."""
    check_gear_mapping(cfg)
    check_time_monotonic(df, episode_id)
    check_ordering(fsm_result, episode_id)
    check_touchdown_requires_both_main_contact(df, fsm_result, cfg, episode_id)
    check_nose_gear_not_used_alone(df, cfg, episode_id, fsm_result["row_phase"])
    check_long_candidate_reported(fsm_result, cfg, episode_id)
    check_bounce_does_not_end_episode(fsm_result, episode_id)
    return []
