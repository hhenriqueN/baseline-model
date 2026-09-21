"""Causal (online, real-time) landing state machine.

Reuses the SAME phase set, SAME both-main-gear-only touchdown criterion, SAME
confirmation durations, and SAME gear-index mapping as
pipeline/landing_fsm.py and data/config/preprocessing.yaml, adapted to a
streaming, one-sample-at-a-time form suitable for real-time control (the
offline FSM backdates confirmation to the start of the qualifying run,
using knowledge that the run persisted -- that is exactly the future
information a real-time controller doesn't have; this version confirms at
the instant the Nth consecutive sample is actually observed, never earlier).

Phases and who controls them (per spec):
  FINAL_APPROACH_ESTABLISHED, FLARE, CONTACT_CANDIDATE  -> neural networks
  TOUCHDOWN_CONFIRMED, BOUNCE                            -> neural networks
    (still airborne-ish / not yet a stable rollout; spec: "Touchdown must
    still require both main gears in continuous contact for 0.3 seconds. Do
    not end control at the first WOW signal, since intermittent contact and
    bounce can occur" -- the deterministic controller only takes over once
    ROLLOUT is reached, matching pipeline/landing_fsm.py's own definition of
    ROLLOUT as the point stable contact is confirmed for rollout_confirm_s)
  ROLLOUT, COMPLETE                                      -> deterministic rollout controller
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from controller.config import FSMConfig

NEURAL_PHASES = {"FINAL_APPROACH_ESTABLISHED", "FLARE", "CONTACT_CANDIDATE",
                  "TOUCHDOWN_CONFIRMED", "BOUNCE"}
GROUND_PHASES = {"ROLLOUT", "COMPLETE"}


@dataclass
class Transition:
    t: float
    previous_state: str
    new_state: str
    reason: str


@dataclass
class FlightManagerState:
    phase: str = "FINAL_APPROACH_ESTABLISHED"
    both_main_consec: int = 0
    any_gear_consec: int = 0
    low_gs_consec: int = 0
    touchdown_confirmed_t: float | None = None
    rollout_confirmed_t: float | None = None
    transitions: list = field(default_factory=list)
    bounce_events: list = field(default_factory=list)
    n_unconfirmed_contacts: int = 0
    n_bounces: int = 0
    _candidate_open_t: float | None = None
    _bounce_open_t: float | None = None
    _bounce_max_agl: float = -1e9


class FlightManager:
    def __init__(self, cfg: FSMConfig, dt: float, flare_agl_ft: float):
        self.cfg = cfg
        self.dt = dt
        self.flare_agl_ft = flare_agl_ft
        self.touchdown_min_samples = max(1, math.ceil(round(cfg.touchdown_confirm_s / dt, 6)))
        self.rollout_min_samples = max(1, math.ceil(round(cfg.rollout_confirm_s / dt, 6)))
        self.complete_min_samples = max(1, math.ceil(round(cfg.complete_duration_s / dt, 6)))
        self.state = FlightManagerState()

    def _transition(self, new_phase: str, t: float, reason: str):
        if new_phase == self.state.phase:
            return
        self.state.transitions.append(Transition(t, self.state.phase, new_phase, reason))
        self.state.phase = new_phase

    def step(self, t: float, agl_ft: float, wow_nose: bool, wow_left: bool, wow_right: bool,
              groundspeed_kt: float) -> str:
        """Advance the state machine by one 10 Hz sample. Returns the new phase."""
        s = self.state
        both_main = wow_left and wow_right
        any_gear = wow_nose or wow_left or wow_right

        if s.phase == "FINAL_APPROACH_ESTABLISHED":
            if agl_ft <= self.flare_agl_ft:
                self._transition("FLARE", t, f"agl_ft={agl_ft:.1f}<=flare_agl_ft={self.flare_agl_ft}")

        if s.phase == "FLARE":
            if wow_left or wow_right:
                s._candidate_open_t = t
                s.both_main_consec = 1 if both_main else 0
                self._transition("CONTACT_CANDIDATE", t, "first_main_gear_contact")

        elif s.phase == "CONTACT_CANDIDATE":
            if both_main:
                s.both_main_consec += 1
                if s.both_main_consec >= self.touchdown_min_samples:
                    s.touchdown_confirmed_t = t
                    s.any_gear_consec = s.both_main_consec  # continuity already established
                    self._transition("TOUCHDOWN_CONFIRMED", t,
                                      f"both_main_gear_contact_for_{s.both_main_consec}_consecutive_samples")
            else:
                s.both_main_consec = 0
                if not (wow_left or wow_right):
                    # full main-gear loss before confirmation -- unconfirmed/
                    # intermittent contact, return to FLARE, episode continues.
                    s.n_unconfirmed_contacts += 1
                    self.state.bounce_events.append({
                        "kind": "unconfirmed_intermittent_contact",
                        "start_s": s._candidate_open_t, "end_s": t,
                    })
                    self._transition("FLARE", t, "contact_lost_before_confirmation_unconfirmed_intermittent_contact")

        elif s.phase == "TOUCHDOWN_CONFIRMED":
            if any_gear:
                s.any_gear_consec += 1
                if s.any_gear_consec >= self.rollout_min_samples:
                    s.rollout_confirmed_t = t
                    self._transition("ROLLOUT", t, f"any_gear_contact_for_{s.any_gear_consec}_consecutive_samples")
            else:
                # confirmed contact lost -> bounce (still recoverable, return to FLARE)
                s.n_bounces += 1
                s._bounce_open_t = t
                s._bounce_max_agl = agl_ft
                s.any_gear_consec = 0
                s.both_main_consec = 0
                self._transition("BOUNCE", t, "confirmed_main_gear_contact_lost")

        elif s.phase == "BOUNCE":
            s._bounce_max_agl = max(s._bounce_max_agl, agl_ft)
            if wow_left or wow_right:
                self.state.bounce_events.append({
                    "kind": "bounce", "start_s": s._bounce_open_t, "end_s": t,
                    "max_agl_ft": s._bounce_max_agl,
                })
                s._candidate_open_t = t
                s.both_main_consec = 1 if both_main else 0
                self._transition("CONTACT_CANDIDATE", t, "renewed_main_gear_contact_after_bounce")
                # touchdown was already confirmed once; a fresh CONTACT_CANDIDATE
                # entry here still requires a fresh both-main confirmation
                # before re-attempting ROLLOUT, matching the conservative,
                # never-assume-stable-contact posture the spec asks for.

        elif s.phase == "ROLLOUT":
            if groundspeed_kt < self.cfg.complete_groundspeed_kt:
                s.low_gs_consec += 1
                if s.low_gs_consec >= self.complete_min_samples:
                    self._transition("COMPLETE", t, "groundspeed_below_threshold")
            else:
                s.low_gs_consec = 0

        return s.phase

    def is_neural_controlled(self) -> bool:
        return self.state.phase in NEURAL_PHASES

    def is_ground_controlled(self) -> bool:
        return self.state.phase in GROUND_PHASES
