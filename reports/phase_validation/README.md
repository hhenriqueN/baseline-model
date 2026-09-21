# Phase-validation diagnostics index

This directory documents where the spec section 6 diagnostics live (they are
generated in `data/` by `scripts/07_detect_transients_and_phases.py`, next to
the rest of the pipeline's manifests, rather than duplicated here):

- **Transition table** (previous state, new state, timestamp, frame, AGL,
  groundspeed, vertical speed, left/right main WOW, nose WOW, gear compression,
  reason): `data/manifests/phase_transitions.csv` (100 rows across 15 episodes).
- **Bounce / unconfirmed-contact events** (start, end, duration, max AGL during
  the gap): `data/manifests/bounce_events.csv` (22 rows, all
  `unconfirmed_intermittent_contact` -- zero confirmed post-touchdown bounces
  exist in the demonstration data, so none are invented, per spec section 5).
- **Review flags** (non-fatal QA flags requiring human review):
  `data/manifests/phase_review_flags.csv`.
- **Per-episode summary boundaries** (flare/contact/touchdown/rollout/complete
  timestamps, transient-removal stats): `data/manifests/episode_phase_boundaries.csv`,
  `data/manifests/transient_diagnostics.csv`.
- **Per-row landing-FSM phase overlay** (10 Hz grid):
  `data/manifests/landing_fsm_row_phase.csv`.
- **Touchdown diagnostic plots** (AGL, vertical speed, groundspeed, 3x WOW, 3x
  gear compression, phase, aileron, elevator, rudder, throttle, all on one time
  axis) for every episode: `data/reports/figures/touchdown_diagnostics/<episode_id>.png`.
- **Automated checks** (fail the pipeline on any hard violation):
  `pipeline/state_machine_checks.py`, invoked from
  `scripts/07_detect_transients_and_phases.py` (stops the run) and re-verified
  in `tests/test_pipeline.py::test_state_machine_checks_pass_on_regenerated_boundaries`.
  Synthetic-sequence unit tests for the FSM itself: `tests/test_landing_fsm.py`.

## Known review case

`voo-3-com-16-nos`: both main gear are never simultaneously in contact for the
required 3 consecutive 10 Hz samples (0.3 s) anywhere in the recording (raw
60 Hz data shows a single-sample maximum overlap of the two main-gear WOW
signals). Touchdown is therefore **not auto-confirmed** for this episode;
ROLLOUT/COMPLETE are undefined for it. This does not reduce the baseline
training set, since only FINAL_APPROACH_ESTABLISHED/FLARE/CONTACT_CANDIDATE
rows are ever used for training and this episode still contributes those
normally (1259 baseline_core rows, fold 04). See
`data/manifests/phase_review_flags.csv` and
`data/reports/figures/touchdown_diagnostics/voo-3-com-16-nos.png` for the full
evidence trail. Flagged for human confirmation before relying on this
episode's ROLLOUT/COMPLETE behavior for anything (e.g. a future ground
controller).
