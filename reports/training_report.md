# Baseline training report

## 1. Root cause of the erroneous `CONTACT_CANDIDATE` durations

`pipeline/features.py::build_phase_and_masks` (previous version) recorded only a
**single** `contact_candidate_s` and a **single** `touchdown_confirmed_s` timestamp
per episode (from `pipeline/segment_circuit.py`, which additionally used a 3-tier
touchdown-confirmation fallback including a single-main-gear-only criterion) and
then painted **every** row between those two timestamps as `CONTACT_CANDIDATE` via
`phase[t >= contact_candidate_s] = "CONTACT_CANDIDATE"`. Whenever the eventual
confirmed touchdown followed one or more porpoise/bounce cycles (gear touching,
lifting off again, touching again), the airborne gaps in between were silently
absorbed into `CONTACT_CANDIDATE` instead of being correctly attributed to `FLARE`
(pre-confirmation loss) or `BOUNCE` (post-confirmation loss). This reproduced the
task's exact known anomalies (verified against the corrected implementation):
`voo-normal2` 2.67s, `voo-normal3` 3.02s, `turbulencia05` 3.77s, `voo-com-8-nos`
8.86s, `voo-com-16-nos` 38.53s. `turbulencia08`'s `FLARE`-directly-to-`ROLLOUT`
transition was a related resolution artifact: its candidate-to-touchdown gap was
only 0.02s wide, narrower than one 10 Hz grid cell, so no resampled row ever fell
inside it.

A second, independent bug in the same function: `baseline_core` was defined as
`t >= baseline_core_start_s` with **no upper bound**, so it silently included every
row through `ROLLOUT` and `COMPLETE` (and, at 10 Hz resolution, occasionally
`TOUCHDOWN_CONFIRMED`) -- exactly the `2352`-row leak (`1899` `ROLLOUT` + `453`
`COMPLETE`) documented in the task's known-problem phase counts (`14491` total =
`10314+857+968+1899+453`).

## 2. Exact correction made

- **`pipeline/landing_fsm.py`** (new): a genuine row-level state machine
  (`run_landing_fsm`) that walks the WOW time series sample by sample on the 10 Hz
  processed grid: `CONTACT_CANDIDATE` on first main-gear contact -> scans for a
  qualifying **both-main-gear** sub-run of >= `ceil(0.3s / dt)` = 3 consecutive
  samples (touchdown confirmation is main-gear-only, never nose-gear, and never a
  single-gear or rollout-duration proxy -- both removed); on confirmation, hands off
  to a rollout/bounce scan requiring >= `ceil(0.8s / dt)` = 8 consecutive samples of
  any-gear contact for `ROLLOUT`, classifying any confirmed-then-lost contact as
  `BOUNCE` (returns to `FLARE`, episode continues); pre-confirmation contact loss is
  recorded as `unconfirmed_intermittent_contact` (also returns to `FLARE`, episode
  continues); `COMPLETE` unchanged (sustained low groundspeed or a genuine
  touch-and-go liftoff). Duration thresholds are expressed as an explicit sample
  count (`ceil(duration_s/dt_s)`), not a timestamp subtraction, to avoid the
  off-by-one the spec warns about.
- **`pipeline/state_machine_checks.py`** (new): independent, pipeline-failing
  assertions (ordering, touchdown-requires-both-main-gear, nose-gear-not-used-alone,
  long-candidate-must-be-reported, bounce-does-not-end-episode, gear mapping,
  time monotonicity) -- run for every episode in
  `scripts/07_detect_transients_and_phases.py` and re-verified in
  `tests/test_pipeline.py`.
- **`pipeline/features.py::build_phase_and_masks`**: now takes a `landing_overlay`
  (the FSM's per-row labels) and overlays it on top of the unchanged
  `FINAL_APPROACH_ESTABLISHED`/`FLARE`/transient/approach-maneuvering scalar
  thresholds (those were already correct one-way crossings, not part of the bug).
  `baseline_core`/`full_approach`/`recovery_ablation` are now defined by **phase-set
  membership** (`BASELINE_CORE_PHASES = {FINAL_APPROACH_ESTABLISHED, FLARE,
  CONTACT_CANDIDATE}`, plus `APPROACH_MANEUVERING` for `full_approach` and
  `INITIALIZATION_TRANSIENT` for `recovery_ablation`), not an unbounded time
  threshold -- this is what fixes the `ROLLOUT`/`COMPLETE` leak.
- **`scripts/07_detect_transients_and_phases.py`**: rewritten to run the FSM on the
  10 Hz processed grid (touchdown/rollout thresholds are meaningful in exact sample
  counts there) instead of the validated 60 Hz grid; `FLARE`/transient/approach
  detection logic is untouched (not part of the bug); now also writes
  `phase_transitions.csv`, `bounce_events.csv`, `phase_review_flags.csv`,
  `landing_fsm_row_phase.csv`, and a touchdown diagnostic plot per episode.
- `gear.main_indices=[1,2]` / `gear.nose_index=0` in `data/config/preprocessing.yaml`
  were already correct (confirmed via `steering-norm`, constant for the two mains,
  variable for the nose gear) and were left unchanged.

## 3. Before / after phase counts

| Phase | Before (given) | After (measured) |
|---|---:|---:|
| `FINAL_APPROACH_ESTABLISHED` | 10,314 | 13,014 |
| `FLARE` | 857 | 1,691 |
| `CONTACT_CANDIDATE` | 968 | 803 |
| `TOUCHDOWN_CONFIRMED` | (uncounted, leaked in) | 0 (collapses to <1 sample at 10Hz in this dataset) |
| `BOUNCE` | (uncounted, leaked in) | 0 (no confirmed post-touchdown bounce exists in the 15 demonstrations) |
| `ROLLOUT` | 1,899 | 2,229 (now correctly EXCLUDED from baseline_core) |
| `COMPLETE` | 453 | 468 (now correctly EXCLUDED from baseline_core) |
| **`baseline_core` total** | **14,491** (contaminated) | **15,508** (clean: only FAE+FLARE+CONTACT_CANDIDATE) |

The "after" totals differ from a naive re-labeling of the same 14,491 rows because
(a) the corrected `FLARE` crossing and FSM now correctly attribute rows that were
previously mislabeled `CONTACT_CANDIDATE` back to `FLARE`/legitimate `FINAL_APPROACH_ESTABLISHED`,
and (b) `baseline_core` no longer stops at an arbitrary single boundary per episode
but is recomputed from the actual per-row phase for all 15 eligible episodes
(the earlier count was over "12 complete sessions"; ours covers all 14 eligible
sessions / 15 episodes, since `voo-3-com-16-nos` is now correctly retained for its
valid FAE/FLARE/CONTACT_CANDIDATE rows despite never confirming touchdown).

## 4. Episode requiring manual review

`voo-3-com-16-nos`: touchdown never auto-confirmed (both main gear never
simultaneously in contact for 3 consecutive 10 Hz samples anywhere in the file --
verified down to the raw 60 Hz data, where the maximum overlap is a single 1/60s
sample). See `reports/phase_validation/README.md`. Does not reduce the baseline
training set (its FAE/FLARE/CONTACT_CANDIDATE rows are still included normally).

## 5. Rows and phases used for neural training

`eligible_for_training == True and baseline_core == 1`, i.e.
`phase in {FINAL_APPROACH_ESTABLISHED, FLARE, CONTACT_CANDIDATE}` exclusively --
verified by `tests/test_pipeline.py::test_no_prohibited_phases_in_training_datasets`
and `test_baseline_core_only_contains_allowed_phases` across every
`baseline_core`/`full_approach`/`recovery_ablation`/normalized-fold CSV.
**15,508** total `baseline_core` rows across the 15 eligible episodes (14 sessions).
`voo-inicio-teste` (familiarization flight) contributes 0 rows to any dataset
variant or fold, confirmed by `test_pilot_familiarization_excluded_from_folds`.

## 6. Fold composition (session, scenario) -- no leakage

13 leave-one-session-out folds over 13 CV sessions (`data/splits/cv_folds_summary.csv`):

| fold | val session | scenario | train rows | val rows |
|---|---|---|---:|---:|
| 0 | circuito-completo | circuit | 12,419 | 1,993 |
| 1 | turbulencia05 | turbulence_05 | 13,383 | 1,029 |
| 2 | turbulencia08 | turbulence_08 | 13,414 | 998 |
| 3 | voo-2-com-rajada | gust | 13,330 | 1,082 |
| 4 | voo-3-com-16-nos | wind_16kt | 13,153 | 1,259 |
| 5 | voo-com-16-nos | wind_16kt | 13,106 | 1,306 |
| 6 | voo-com-8-nos | wind_8kt | 13,538 | 874 |
| 7 | voo-com-rajada | gust | 13,380 | 1,032 |
| 8 | voo-com-rajada-vindo-da-direita | gust_direction_variant | 13,323 | 1,089 |
| 9 | voo-condicoes-normais | normal | 13,424 | 988 |
| 10 | voo-normal2 | normal | 13,568 | 844 |
| 11 | voo-normal3 | normal | 13,518 | 894 |
| 12 | voo2-com-8-nos | wind_8kt | 13,388 | 1,024 |

Fixed diagnostic holdout session (never in any fold's train or val):
`voo-2-com-16-nos` (wind_16kt), chosen deterministically by
`sha256(session_id)` ascending among the three wind_16kt sessions. Both
`circuito-completo` episodes (landing-01, landing-02) share one session and
travel together in fold 0, confirmed by
`test_circuit_landings_share_session_and_fold`. No leakage confirmed by
`test_fold_scaler_uses_only_train_sessions`,
`test_every_eligible_non_holdout_session_validated_exactly_once`,
`test_holdout_session_never_in_any_fold` (all passing, 245/245 total tests).

## 7. Paper findings that influenced this implementation

See `reports/paper_analysis.md` for the full breakdown. Confirmed and reused:
10 Hz control frequency (matches spec exactly); "current-minus-desired" inputs are
always aircraft-state tracking errors, never actuator/surface positions (validates
spec section 14's design choice rather than contradicting it). Explicitly **not**
reused: the paper's 13-ANN-cascade decomposition (this baseline uses the 2 shared
MLPs fixed by spec sections 15-16), its plain-backprop-with-momentum optimizer and
fixed-MSE-threshold stopping rule (we use AdamW + held-out early stopping, since our
smaller-per-session but multi-session dataset benefits from an explicit
generalization check the paper's single-flight regime never performs), and its lack
of any reported input normalization (we z-score, since our networks take raw-scale
state variables rather than the paper's pre-differenced error scalars).

## 8. Training configuration

Seed 42, AdamW (lr=1e-3, weight_decay=1e-4), batch_size=256, max_epochs=300,
early-stopping patience=25 on validation MSE, gradient-norm clipping at 5.0, equal
MSE weighting of both outputs per network, no phase weighting/oversampling/
augmentation. Architectures exactly per spec sections 15-16 (no BatchNorm, no
dropout). Config: `training/train_core.py::TRAINING_CONFIG`,
`training/models.py::ARCHITECTURE_CONFIG`.

## 9. Cross-validation results (mean ± std across 13 folds)

**Lateral** (aileron_command, rudder_command):

| output | MSE | RMSE | MAE | R² | Pearson r |
|---|---|---|---|---|---|
| aileron_command | 0.0035 ± 0.0057 | 0.048 ± 0.036 | 0.026 ± 0.015 | -1.10 ± 5.44 | 0.63 ± 0.25 |
| rudder_command | 0.0116 ± 0.0104 | 0.099 ± 0.044 | 0.065 ± 0.030 | 0.15 ± 0.41 | 0.53 ± 0.17 |

best_epoch: 16.5 ± 16.3 (early stopping fires quickly on several folds -- small
per-fold dataset).

**Longitudinal** (elevator_effective, throttle_command):

| output | MSE | RMSE | MAE | R² | Pearson r |
|---|---|---|---|---|---|
| elevator_effective | 0.0020 ± 0.0032 | 0.038 ± 0.026 | 0.023 ± 0.013 | 0.51 ± 0.38 | 0.77 ± 0.21 |
| throttle_command | 0.0185 ± 0.0143 | 0.130 ± 0.043 | 0.104 ± 0.042 | 0.60 ± 0.24 | 0.83 ± 0.12 |

best_epoch: 12.4 ± 9.5.

Full per-fold, per-flight, per-scenario, per-phase tables:
`reports/training/{lateral,longitudinal}/cv_fold_results.csv`,
`metrics_per_flight.csv`, `metrics_per_scenario.csv`, `metrics_per_phase.csv`.
Loss curves, predicted-vs-true scatter, residual histograms:
`reports/training/{lateral,longitudinal}/figures/`. Per-episode expert-vs-predicted
time series (all 4 commands): `reports/training/timeseries/<episode_id>.png`.

The lateral network's `aileron_command` shows a negative mean R² (dominated by a
few high-variance/negative-R² folds, e.g. crosswind episodes where the small
per-fold sample makes the target's own variance a weak baseline) despite a
positive-on-average Pearson correlation (0.63) -- i.e. the model tracks direction
reasonably but its absolute-error calibration is inconsistent fold-to-fold. This is
an honest property of a small-N behavioral-cloning baseline, not a training bug (CV
loss curves converge normally; see figures). Longitudinal outputs generalize more
consistently (positive R² and Pearson > 0.75 on both outputs).

## 10. Final-training results

Final scaler: `data/scalers/final_all_valid/scaler_final_all_valid.json`
(14 eligible sessions, including the fixed diagnostic holdout -- **never** used for
offline validation, only for this final fit per spec section 18).
Final training rows: 15,508 (`baseline_core`, all valid demonstrations).
Epoch count = median CV best_epoch (0-indexed) + 1: **lateral 15 epochs**,
**longitudinal 10 epochs**. No held-out split for this run (architecture/
hyperparameters/epoch count were already frozen by cross-validation); the frozen
checkpoints are the ones a future closed-loop FlightGear controller would load.

## 11. Artifact paths

- Models: `models/{lateral,longitudinal}/folds/fold_XX/{checkpoint.pt,config.json,history.json,metrics.json,val_predictions.csv}`,
  `models/{lateral,longitudinal}/final/{checkpoint.pt,config.json,history.json}`.
- Machine-readable manifest for a future real-time controller: `models/model_manifest.json`.
- Scalers: `data/scalers/folds/fold_XX_<val_session>.json`, `data/scalers/final_all_valid/scaler_final_all_valid.json`.
- Split manifests: `data/splits/cv_folds_summary.csv`, `data/splits/cv_folds_session_assignment.csv`, `data/splits/fixed_diagnostic_holdout.yaml`.
- Phase-validation diagnostics: `reports/phase_validation/README.md` (index into `data/manifests/*` + `data/reports/figures/touchdown_diagnostics/`).
- Training reports/plots: `reports/training/`.
- Paper analysis: `reports/paper_analysis.md`.

## 12. Remaining limitations

- Small-N behavioral cloning baseline: 14 sessions is enough for leave-one-session-out
  model selection, not for strong statistical generalization claims -- CV std across
  folds is large relative to the mean for several metrics (see §9).
- `voo-3-com-16-nos` needs human confirmation of its ground-contact behavior before
  any future component relies on its ROLLOUT/COMPLETE timing.
- The RWY29 finding for `circuito-completo__landing-02` (pre-existing, documented in
  `data/reports/cleaning_report.md`) still awaits human confirmation.
- Offline regression metrics (this report) say nothing about closed-loop stability,
  error accumulation, or actual landing success. **The trained networks are offline
  behavioral-cloning baselines only.** Autonomous-landing capability can only be
  established by the future closed-loop FlightGear evaluation (explicitly out of
  scope here, per spec section 24) -- this report makes no claim that the aircraft
  can land autonomously.
- No phase weighting/recurrent state/temporal context was used (by design, per spec
  section 17); a stateless per-row MLP cannot model command smoothing or actuator
  rate limits, which the future real-time controller must add separately.

## 13. Reproducible commands

```bash
# rebuild the datasets (phase correction + features + folds + scalers)
python3 scripts/01_ingest.py && python3 scripts/02_audit.py && \
python3 scripts/03_segment_circuit.py && python3 scripts/04_build_episodes.py && \
python3 scripts/05_build_validated.py && python3 scripts/06_resample.py && \
python3 scripts/07_detect_transients_and_phases.py && python3 scripts/08_build_features.py && \
python3 scripts/08b_build_processed_variants.py && python3 scripts/09_build_splits_and_scalers.py && \
python3 scripts/10_apply_scalers.py && python3 scripts/11_build_reports.py

# run all tests (phase-machine + dataset + split + model tests)
python3 -m pytest tests/ -v

# train all 13 CV folds for both networks
python3 -m training.run_cv

# produce plots + flattened metric tables
python3 -m training.make_reports

# train the final models on all valid demonstrations
python3 -m training.run_final

# rebuild the machine-readable model manifest
python3 -m training.build_model_manifest
```
