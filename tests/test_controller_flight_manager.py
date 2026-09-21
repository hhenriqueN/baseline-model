"""Streaming Flight-Manager tests -- mirror tests/test_landing_fsm.py's cases
but exercised one causal sample at a time, the way the real-time loop calls
FlightManager.step()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from controller.config import FSMConfig
from controller.flight_manager import FlightManager

DT = 0.1
FLARE_AGL = 20.0


def make_fm():
    return FlightManager(FSMConfig(touchdown_confirm_s=0.3, rollout_confirm_s=0.8,
                                     complete_groundspeed_kt=5.0, complete_duration_s=1.0), DT, FLARE_AGL)


def test_flare_crossing():
    fm = make_fm()
    fm.step(t=0.0, agl_ft=30, wow_nose=False, wow_left=False, wow_right=False, groundspeed_kt=60)
    assert fm.state.phase == "FINAL_APPROACH_ESTABLISHED"
    fm.step(t=0.1, agl_ft=15, wow_nose=False, wow_left=False, wow_right=False, groundspeed_kt=60)
    assert fm.state.phase == "FLARE"


def test_first_contact_enters_contact_candidate_and_does_not_confirm_immediately():
    fm = make_fm()
    fm.state.phase = "FLARE"
    fm.step(t=0.0, agl_ft=2, wow_nose=False, wow_left=True, wow_right=False, groundspeed_kt=55)
    assert fm.state.phase == "CONTACT_CANDIDATE"


def test_3_consecutive_both_main_samples_confirms_touchdown_at_10hz():
    fm = make_fm()
    fm.state.phase = "FLARE"
    fm.step(t=0.0, agl_ft=1, wow_nose=False, wow_left=True, wow_right=True, groundspeed_kt=55)
    assert fm.state.phase == "CONTACT_CANDIDATE"
    fm.step(t=0.1, agl_ft=0, wow_nose=False, wow_left=True, wow_right=True, groundspeed_kt=55)
    assert fm.state.phase == "CONTACT_CANDIDATE"  # only 2 samples so far
    fm.step(t=0.2, agl_ft=0, wow_nose=False, wow_left=True, wow_right=True, groundspeed_kt=55)
    assert fm.state.phase == "TOUCHDOWN_CONFIRMED"  # 3rd consecutive sample


def test_loss_before_confirmation_returns_to_flare_not_episode_end():
    fm = make_fm()
    fm.state.phase = "FLARE"
    fm.step(t=0.0, agl_ft=1, wow_nose=False, wow_left=True, wow_right=True, groundspeed_kt=55)
    fm.step(t=0.1, agl_ft=0, wow_nose=False, wow_left=True, wow_right=True, groundspeed_kt=55)
    # loses contact before the 3rd confirming sample
    fm.step(t=0.2, agl_ft=3, wow_nose=False, wow_left=False, wow_right=False, groundspeed_kt=55)
    assert fm.state.phase == "FLARE"
    assert fm.state.n_unconfirmed_contacts == 1
    assert fm.state.touchdown_confirmed_t is None


def test_nose_gear_alone_never_confirms_touchdown():
    fm = make_fm()
    fm.state.phase = "FLARE"
    for i in range(10):
        phase = fm.step(t=i * DT, agl_ft=0, wow_nose=True, wow_left=False, wow_right=False, groundspeed_kt=55)
    assert phase == "FLARE"  # never even enters CONTACT_CANDIDATE
    assert fm.state.touchdown_confirmed_t is None


def test_loss_after_confirmation_is_a_bounce_and_episode_continues():
    fm = make_fm()
    fm.state.phase = "FLARE"
    t = 0.0
    for _ in range(3):
        fm.step(t=t, agl_ft=0, wow_nose=False, wow_left=True, wow_right=True, groundspeed_kt=55)
        t += DT
    assert fm.state.phase == "TOUCHDOWN_CONFIRMED"
    # confirmed contact lost -> bounce
    fm.step(t=t, agl_ft=2, wow_nose=False, wow_left=False, wow_right=False, groundspeed_kt=55)
    assert fm.state.phase == "BOUNCE"
    assert fm.state.n_bounces == 1
    t += DT
    fm.step(t=t, agl_ft=0, wow_nose=False, wow_left=True, wow_right=False, groundspeed_kt=55)
    assert fm.state.phase == "CONTACT_CANDIDATE"  # recoverable, back into the landing sequence


def test_8_consecutive_any_gear_samples_confirms_rollout_after_touchdown():
    fm = make_fm()
    fm.state.phase = "TOUCHDOWN_CONFIRMED"
    fm.state.any_gear_consec = 0
    t = 0.0
    phase = None
    for i in range(8):
        phase = fm.step(t=t, agl_ft=0, wow_nose=True, wow_left=True, wow_right=True, groundspeed_kt=40)
        t += DT
    assert phase == "ROLLOUT"


def test_rollout_never_precedes_touchdown_confirmed_and_complete_never_precedes_rollout():
    fm = make_fm()
    fm.state.phase = "FLARE"
    t = 0.0
    for _ in range(3):
        fm.step(t=t, agl_ft=0, wow_nose=False, wow_left=True, wow_right=True, groundspeed_kt=55)
        t += DT
    assert fm.state.phase == "TOUCHDOWN_CONFIRMED"
    assert fm.state.rollout_confirmed_t is None  # not yet
    for _ in range(8):
        fm.step(t=t, agl_ft=0, wow_nose=True, wow_left=True, wow_right=True, groundspeed_kt=3.0)
        t += DT
    assert fm.state.phase in ("ROLLOUT", "COMPLETE")
    assert fm.state.rollout_confirmed_t is not None
    assert fm.state.rollout_confirmed_t >= fm.state.touchdown_confirmed_t


def test_first_wow_does_not_terminate_episode_sustained_low_gs_required_for_complete():
    fm = make_fm()
    fm.state.phase = "ROLLOUT"
    fm.state.low_gs_consec = 0
    phase = fm.step(t=0.0, agl_ft=0, wow_nose=True, wow_left=True, wow_right=True, groundspeed_kt=3.0)
    assert phase == "ROLLOUT"  # 1 sample below threshold is not enough (complete_duration_s=1.0 -> 10 samples)
    for i in range(1, 10):
        phase = fm.step(t=i * DT, agl_ft=0, wow_nose=True, wow_left=True, wow_right=True, groundspeed_kt=3.0)
    assert phase == "COMPLETE"


def test_is_neural_controlled_and_ground_controlled_partition():
    fm = make_fm()
    for phase in ("FINAL_APPROACH_ESTABLISHED", "FLARE", "CONTACT_CANDIDATE", "TOUCHDOWN_CONFIRMED", "BOUNCE"):
        fm.state.phase = phase
        assert fm.is_neural_controlled() and not fm.is_ground_controlled()
    for phase in ("ROLLOUT", "COMPLETE"):
        fm.state.phase = phase
        assert fm.is_ground_controlled() and not fm.is_neural_controlled()
