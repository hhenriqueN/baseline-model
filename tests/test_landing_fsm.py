"""Synthetic-sequence unit tests for the corrected landing state machine
(pipeline/landing_fsm.py), required by spec section 21 "Phase annotation".

All synthetic frames are built at a clean 10 Hz grid (dt=0.1s) so touchdown
(0.3s -> 3 samples) / rollout (0.8s -> 8 samples) thresholds are exact
integers, making the tests deterministic and off-by-one-sensitive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import columns as C
from pipeline.landing_fsm import run_landing_fsm
from pipeline.state_machine_checks import check_ordering, check_gear_mapping

CFG = {
    "gear": {"nose_index": 0, "main_indices": [1, 2]},
    "landing_segmentation": {
        "touchdown_confirm_s": 0.3,
        "rollout_confirm_s": 0.8,
        "complete_groundspeed_kt": 5.0,
        "complete_duration_s": 1.0,
        "liftoff_complete_s": 1.0,
        "diagnostic_long_candidate_s": 2.0,
    },
}


def make_frame(n, dt=0.1, agl=30.0, gs=60.0, thr=0.3):
    """All-airborne baseline frame: nothing touching, constant AGL/GS."""
    t = np.arange(n) * dt
    df = pd.DataFrame({
        C.SIM_TIME: t,
        "frame": np.arange(n),
        C.ALT_AGL_FT: np.full(n, agl, dtype=float),
        C.GROUNDSPEED_KT: np.full(n, gs, dtype=float),
        C.THROTTLE_ENGINE0: np.full(n, thr, dtype=float),
        C.VERTICAL_SPEED_FPS: np.zeros(n),
        C.gear_wow(0): np.zeros(n),
        C.gear_wow(1): np.zeros(n),
        C.gear_wow(2): np.zeros(n),
        C.gear_compression(0): np.zeros(n),
        C.gear_compression(1): np.zeros(n),
        C.gear_compression(2): np.zeros(n),
    })
    return df


def set_wow(df, col_idx, lo, hi, value=1.0):
    df.loc[lo:hi, C.gear_wow(col_idx)] = value


def test_gear_mapping_nose_not_in_mains():
    check_gear_mapping(CFG)  # nose=0, mains={1,2} -- must not raise


def test_first_main_gear_contact_enters_contact_candidate():
    df = make_frame(20)
    set_wow(df, 1, 5, 19)  # only left main ever touches -- never confirms
    res = run_landing_fsm(df, 0, 19, CFG)
    assert res["contact_candidate_first_idx"] == 5
    assert res["row_phase"][5] == "CONTACT_CANDIDATE"


def test_nose_gear_alone_never_enters_contact_candidate_or_confirms():
    df = make_frame(20)
    set_wow(df, 0, 5, 19)  # nose gear only
    res = run_landing_fsm(df, 0, 19, CFG)
    assert res["contact_candidate_first_idx"] is None
    assert res["touchdown_confirmed"] is False
    assert "no_main_gear_contact_found_in_window" in res["review_flags"]
    assert not any(label == "CONTACT_CANDIDATE" for label in res["row_phase"])


def test_exactly_3_samples_of_both_main_confirms_touchdown_at_10hz():
    df = make_frame(30, gs=2.0)  # groundspeed already low -> complete resolves quickly
    set_wow(df, 1, 10, 29)
    set_wow(df, 2, 10, 29)
    res = run_landing_fsm(df, 0, 29, CFG)
    assert res["touchdown_confirmed"] is True
    assert res["touchdown_confirmed_idx"] == 10
    check_ordering(res, "synthetic")


def test_2_samples_of_both_main_does_not_confirm_touchdown():
    df = make_frame(30)
    set_wow(df, 1, 10, 11)  # only 2 consecutive samples (0.2s < 0.3s)
    set_wow(df, 2, 10, 11)
    res = run_landing_fsm(df, 0, 29, CFG)
    assert res["touchdown_confirmed"] is False


def test_loss_of_contact_before_confirmation_returns_to_flare_not_episode_end():
    df = make_frame(40)
    # brief 2-sample both-main touch (unconfirmed), fully airborne gap, then a
    # real >=3-sample confirmed touchdown later, sustained to rollout+complete.
    set_wow(df, 1, 5, 6)
    set_wow(df, 2, 5, 6)
    set_wow(df, 1, 20, 39)
    set_wow(df, 2, 20, 39)
    df.loc[20:, C.GROUNDSPEED_KT] = 2.0
    res = run_landing_fsm(df, 0, 39, CFG)
    assert res["touchdown_confirmed"] is True
    assert res["touchdown_confirmed_idx"] == 20
    unconfirmed = [b for b in res["bounce_records"] if b["kind"] == "unconfirmed_intermittent_contact"]
    assert len(unconfirmed) == 1
    assert unconfirmed[0]["start_s"] == pytest.approx(0.5)
    # the airborne gap rows (7..19) must NOT be labeled CONTACT_CANDIDATE
    assert all(res["row_phase"][i] != "CONTACT_CANDIDATE" for i in range(7, 20))


def test_loss_after_confirmation_is_a_bounce_not_episode_end():
    df = make_frame(40, gs=2.0)
    set_wow(df, 1, 5, 39)
    set_wow(df, 2, 5, 39)
    # brief post-touchdown liftoff (ALL gear off, including nose) -- a bounce
    set_wow(df, 1, 12, 13, value=0.0)
    set_wow(df, 2, 12, 13, value=0.0)
    res = run_landing_fsm(df, 0, 39, CFG)
    assert res["touchdown_confirmed"] is True
    assert res["touchdown_confirmed_idx"] == 5
    bounces = [b for b in res["bounce_records"] if b["kind"] == "bounce"]
    assert len(bounces) == 1
    assert bounces[0]["start_s"] == pytest.approx(1.2)
    assert res["rollout_idx"] is not None
    assert res["rollout_idx"] > 13  # rollout only reached after the bounce resolves
    check_ordering(res, "synthetic")


def test_first_wow_does_not_terminate_episode():
    df = make_frame(60, gs=50.0)  # realistic touchdown speed, decaying afterward
    df[C.GROUNDSPEED_KT] = np.clip(50.0 - 2.0 * np.arange(60), 1.0, 50.0)  # crosses 5kt around idx=23
    set_wow(df, 1, 10, 59)
    set_wow(df, 2, 10, 59)
    res = run_landing_fsm(df, 0, 59, CFG)
    assert res["complete_idx"] is not None
    assert res["complete_idx"] > res["touchdown_confirmed_idx"] + 5  # not the very first WOW=1 sample


def test_rollout_never_precedes_touchdown_confirmed():
    df = make_frame(40, gs=2.0)
    set_wow(df, 1, 8, 39)
    set_wow(df, 2, 8, 39)
    res = run_landing_fsm(df, 0, 39, CFG)
    check_ordering(res, "synthetic")
    if res["rollout_idx"] is not None:
        assert res["rollout_idx"] >= res["touchdown_confirmed_idx"]


def test_complete_never_precedes_rollout():
    df = make_frame(50, gs=2.0)
    set_wow(df, 1, 8, 49)
    set_wow(df, 2, 8, 49)
    res = run_landing_fsm(df, 0, 49, CFG)
    check_ordering(res, "synthetic")
    if res["complete_idx"] is not None and res["rollout_idx"] is not None:
        assert res["complete_idx"] >= res["rollout_idx"]


def test_abnormally_long_candidate_interval_is_reported():
    df = make_frame(60)
    # main gear touches (single/alternating, never both simultaneously long
    # enough) continuously for 2.5s (25 samples) then fully releases -- an
    # unconfirmed candidate interval at/above the 2.0s diagnostic threshold.
    set_wow(df, 1, 5, 29)
    res = run_landing_fsm(df, 0, 59, CFG)
    assert res["touchdown_confirmed"] is False
    assert any(f.startswith("long_unconfirmed_CONTACT_CANDIDATE_interval") for f in res["review_flags"])


def test_touchdown_never_confirmed_flags_review_and_excludes_tail():
    df = make_frame(30)
    set_wow(df, 1, 10, 19)  # single main gear only, released, never confirmed
    res = run_landing_fsm(df, 0, 29, CFG)
    assert res["touchdown_confirmed"] is False
    assert any(f.startswith("touchdown_never_confirmed") for f in res["review_flags"])
    # the one candidate attempt is still legitimately CONTACT_CANDIDATE (spec
    # section 7 keeps it training-eligible); no ROLLOUT/COMPLETE ever appear.
    assert res["row_phase"][10] == "CONTACT_CANDIDATE"
    assert res["rollout_idx"] is None and res["complete_idx"] is None
    assert not any(label in ("ROLLOUT", "COMPLETE", "TOUCHDOWN_CONFIRMED") for label in res["row_phase"])
    # the trailing gap after the last unconfirmed attempt is left as the
    # airborne default (None -> caller paints it FLARE), not mislabeled.
    assert res["row_phase"][25] is None
