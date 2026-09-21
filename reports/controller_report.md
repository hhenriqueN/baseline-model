# Real-time FlightGear controller: implementation report

## 1. Command to run

```bash
PYTHONUNBUFFERED=1 python3 -m controller.run --launch-flightgear --scenario nominal --mode shadow
PYTHONUNBUFFERED=1 python3 -m controller.run --launch-flightgear --scenario nominal --mode active
```

`Ctrl+C` at any time triggers a graceful emergency stop (neutral commands + full brakes sent once,
FlightGear terminated cleanly, `summary.json` written with `abort_reason: "user_interrupt"`).

## 2. Files created / changed

New package `controller/`: `config.py`, `protocol.py`, `launcher.py`, `telemetry_io.py`,
`command_io.py`, `features_online.py`, `flight_manager.py`, `rollout_controller.py`,
`models_runtime.py`, `safety.py`, `run_logger.py`, `run.py`, `mock_flightgear.py`.
New tests: `tests/test_controller_{protocol,features,flight_manager,models_runtime,safety,launcher}.py`
(41 tests). Extended `pipeline/runway.py` with `point_at_along_cross` (reused, not duplicated,
for computing the spawn position from the project's own runway geometry). No existing pipeline,
training, or model file was modified -- **the two final checkpoints are loaded read-only and
never retrained or altered**.

## 3. Project inspection findings (used directly)

- `models/model_manifest.json`: checkpoint paths, feature/label order, output saturation bounds,
  scaler reference -- consumed as-is by `controller/models_runtime.py::ModelBundle`.
- `data/scalers/final_all_valid/scaler_final_all_valid.json`: the exact z-score mean/std/clip
  used at training time, applied online via the same `pipeline.scalers.apply_scaler` function.
- `data/config/runway.yaml`: BIKF RWY02 threshold lat/lon/heading/elevation, used both for the
  spawn-position calculation and for the online runway-relative features (same function,
  `pipeline.runway.compute_runway_features`, as training).
- `pipeline/landing_fsm.py` / `data/config/preprocessing.yaml`: the corrected landing state
  machine's thresholds (touchdown_confirm_s=0.3, rollout_confirm_s=0.8) and gear-index mapping
  (nose=0, mains=[1,2]) -- reused conceptually by `controller/flight_manager.py`, which
  re-implements the same rules in a **causal/streaming** form (the offline FSM backdates
  confirmation to the start of a qualifying run using knowledge that it persisted; a real-time
  controller does not have that knowledge and must confirm at the instant the Nth consecutive
  sample actually arrives).
- No existing FlightGear communication code was present in the project. The communication
  mechanism (UDP "generic" protocol) and the local FlightGear install's own prior session log
  (`~/Library/Application Support/FlightGear/fgfs.log`) were both inspected before designing
  `controller/protocol.py` -- see section 5.
- c172p's JSBSim flight-control system (`Aircraft/c172p/c172p.xml`, `<flight_control name="FCS: c172">`,
  "Pitch" channel) was inspected directly: `fcs/pitch-trim-sum = clip(gain(rate_limit(elevator-cmd-norm))
  + pitch-trim-cmd-norm, -1, 1)` -- i.e. `/controls/flight/elevator` and `/controls/flight/elevator-trim`
  are summed inside the FDM, confirming the project's own `elevator_effective = elevator_command +
  elevator_trim` label definition and informing the exact online decomposition used (section 6).

## 4. Detected FlightGear executable and version

`/Users/iquenavarro/Desktop/FlightGear.app/Contents/MacOS/FlightGear`, version **2024.1.7**
(confirmed via `Info.plist` and the app's own historical log). This macOS `.app` bundle's main
binary IS the classic `fgfs` simulator (its call stack shows `fgMainInit`/`fgOSMainLoop`, not a
separate launcher process) -- `controller/launcher.py::find_fgfs()` checks this path first, then
`/Applications/FlightGear.app/...`, then common Linux/Homebrew `fgfs` locations, then `$PATH`,
and accepts an explicit `--fgfs-path` override.

## 5. Communication method and ports

FlightGear's **"generic" I/O protocol** over UDP (ASCII, comma-separated), not the telnet
property-tree interface. Chosen because telnet is a request/response-per-property protocol
unsuited to a 10 Hz, ~35-field loop, whereas the generic protocol is a stream FlightGear's own
I/O manager drives at the requested rate with no per-field round trip -- the standard mechanism
for this kind of external control loop, and the same category of interface (`Protocol/*.xml`)
already shipped with the local FlightGear install (`aura-imu.xml`, `avare.xml`, etc.), whose exact
XML schema `controller/protocol.py::build_protocol_xml()` follows.

- Telemetry (FlightGear -> controller): UDP, port **5500** (`--generic=socket,out,10,...`).
- Commands (controller -> FlightGear): UDP, port **5510** (`--generic=socket,in,10,...`).
- A secondary telnet channel on port **5401** is also opened (matching the project's own prior
  interactive session, which used telnet for props access) but is not on the 10 Hz control path;
  it exists only as a spare inspection channel.
- The protocol XML is generated from `controller/protocol.py`'s single field list
  (`OUTPUT_FIELDS`/`INPUT_FIELDS`) and written to `$FG_ROOT/Protocol/baseline_controller.xml` at
  launch time -- there is exactly one place field names/property-paths are defined, so the XML
  and the Python encoder/decoder cannot drift apart.

## 6. Elevator/trim handling (no double-counting)

The longitudinal network predicts a single scalar `elevator_effective`. Every cycle:

```
elevator_command_sent = clip(elevator_effective_predicted - trim_fixed, -1, 1)
elevator_trim_sent    = trim_fixed   # set ONCE at initialization, never touched again
```

so `elevator_command_sent + elevator_trim_sent == elevator_effective_predicted` by construction
(verified by `tests/test_controller_models_runtime.py::test_compose_elevator_command_no_double_counting`).
`trim_fixed = 0.031` is the median `/controls/flight/elevator-trim[0]` observed at
`phase == FINAL_APPROACH_ESTABLISHED` across the 14 standardized-start demonstrations (computed
directly from `data/processed/full_episodes/`, not guessed). Nothing else in a scripted session
writes `/controls/flight/elevator-trim` (no joystick/keyboard trim input is bound), so it cannot
drift and create a second, uncounted trim contribution.

## 7. Exact initialization used (nominal scenario)

Computed from `data/config/runway.yaml` via `pipeline.runway.point_at_along_cross` (added to the
existing runway module, not duplicated elsewhere): along_track_m=-3000, cross_track_m=0 ->
lat=63.93756634, lon=-22.60547415, heading=0.020 (runway true heading), altitude=800 ft MSL
(matches the demonstrations' own t=0 MSL altitude, ~800.8 ft, verified directly against
`data/validated/voo-condicoes-normais.csv`), vc=65 kt. Flaps=0.99, throttle=0.52, elevator-trim=
0.031 are the median values observed at the STABILIZED `FINAL_APPROACH_ESTABLISHED` phase across
the demonstrations -- deliberately **not** copied from the raw, unconfigured t=0 sample (airspeed
101.6 kt, flaps 0.0 there), which is exactly the artificial initialization transient the project's
own pipeline detects and excludes from `baseline_core` (spec's "avoid the artificial poorly
configured initial condition"). Weather: calm METAR (`00000KT`, `FEW250`, matching the format of
the project's own prior sessions), `--disable-real-weather-fetch`, `--timeofday=noon`.
Autopilot is never engaged (default-off state; `/autopilot/locks/passive-mode` also set
defensively). Gear: C172P has fixed (non-retractable) gear, always down; `/controls/gear/gear-down`
is still set explicitly for clarity.

## 8. Test results

`python3 -m pytest tests/ -v` -> **286 passed, 0 failed** (245 from the dataset/training stage +
41 new controller tests: protocol encode/decode round-trips, online feature engineering
cross-checked line-for-line against `pipeline.runway.compute_runway_features` on identical input,
the causal Flight Manager exercised through every phase/transition with synthetic WOW sequences,
elevator/trim composition and saturation, safety-envelope and staleness/NaN/saturation abort
logic, the deterministic rollout controller, and launcher argv/init-position construction).

## 9. Live FlightGear launch attempts and finding

FlightGear **did launch successfully** (process starts, appears as a foreground GUI app, the
project's own runway/scenery/aircraft data are all present and previously used successfully by
this exact install per its historical logs), but across five attempts in this automated session
it never progressed past the earliest native-init stage -- no FlightGear log line was ever
written (confirmed via file mtimes: none of the per-run `fgfs_N.log` files were touched by any
attempt today), and a process sample showed it idling in a Cocoa/Qt event loop
(`mach_msg`/`CFRunLoopRun`) at ~0% CPU rather than doing simulator work. `--disable-sentry` and
clearing a stale `fgfs_exclusive.lock` were both tried and ruled out as the cause. The most
consistent explanation is a **one-time macOS permission dialog** (most likely "Local Network
access", which macOS gates per-app the first time a process opens listening sockets -- exactly
what this controller's UDP telemetry/command ports do) blocking on user interaction that this
automated session cannot grant: it has no Accessibility/screen-recording permission (`osascript`
GUI scripting and `screencapture` both failed with permission errors), and the task explicitly
disallows mouse automation as a workaround.

**This is a one-time, one-click environment setup step, not a code defect.** To clear it: run
`python3 -m controller.run --launch-flightgear --scenario nominal --mode shadow` once from an
interactive Terminal session (not through this automated session) and click "Allow" on any
permission dialog that appears (Local Network access, and/or a Gatekeeper "downloaded app" prompt
if this is the first-ever launch of this copy of the app); after that one-time grant, FlightGear
should launch normally and every subsequent automated run should proceed straight into the 10 Hz
control loop.

## 10. End-to-end validation performed instead (mock FlightGear)

Since live FlightGear could not be driven to completion in this session, the full controller
pipeline was validated against `controller/mock_flightgear.py`: a scripted stand-in that speaks
the identical UDP generic-protocol on the identical ports, replaying a kinematically plausible
3 km/800 ft/65 kt straight-in approach, flare, touchdown, and rollout to a stop -- proving every
code path *except the real 3D simulator itself*:

- `runs/example_mock_closed_loop_full_landing/`: startup sequence completed (readiness check
  confirmed 5 consecutive valid telemetry samples before starting control), ran the full 10 Hz
  loop, correctly progressed `FINAL_APPROACH_ESTABLISHED -> FLARE -> CONTACT_CANDIDATE ->
  TOUCHDOWN_CONFIRMED -> ROLLOUT`, confirmed touchdown at t=119.1s with near-zero cross-track/
  heading error (by construction of the mock's centerline trajectory), handed off correctly to
  the deterministic rollout controller (throttle idle, progressive braking, rudder centerline
  hold), and — **a real bug this test caught and I fixed** — an initial version incorrectly
  aborted during rollout on "airspeed too low" (the min-airspeed envelope check was firing during
  intentional ground deceleration); fixed by scoping the airspeed/bank/pitch envelope check to
  neural-controlled phases only (`controller/run.py`, `flight_manager.is_neural_controlled()`).
- `runs/example_mock_closed_loop_full_landing_v2_safety_fix/`: re-run after the fix -- touchdown
  and rollout again confirmed cleanly, ran through rollout deceleration without a false abort, and
  safely aborted with `telemetry_stale_0.55s` once the mock itself stopped sending telemetry (its
  scripted flight ended) -- i.e. the staleness-abort safety mechanism itself was also exercised
  and confirmed working.
- `runs/example_mock_closed_loop_partial_80s/`: an earlier, shorter run demonstrating shadow-vs-
  active wiring and per-cycle logging before the full-length run above.

Measured control-loop performance (`example_mock_closed_loop_full_landing/cycles.jsonl`,
1252 cycles): mean compute latency **8.9 ms** (p95 14.0 ms, max 33.9 ms) against a 100 ms budget;
mean achieved cycle spacing **97.8 ms** (target 100 ms). Command round-trip was confirmed directly
(the mock echoes back whatever aileron/elevator/rudder/throttle it last received as the next
cycle's telemetry `*_pos` fields, and these are visibly non-zero/non-default in the logs).

**What this does and does not prove**: it proves the controller's telemetry parsing, causal
feature engineering (bit-for-bit cross-checked against the training-time functions), model
inference, elevator/trim decomposition, Flight Manager phase logic, safety envelope, rollout
controller, rate limiting, and logging are all correctly wired end-to-end. It does **not** prove
FlightGear itself launches cleanly in every environment, nor that the trained networks can fly the
*real* nonlinear JSBSim aircraft model -- that requires the live run described in section 9, which
is the necessary next step, not yet completed.

## 11. What must happen before the nominal live-landing attempt

1. Clear the one-time macOS permission dialog per section 9 (interactive, one click).
2. Re-run the validation sequence in order: launch test -> live telemetry/command-interface test
   -> shadow-mode test (verify predictions look sane against real JSBSim telemetry) -> a short
   active-control smoke test (verify control *signs*, e.g. positive cross-track error produces a
   correcting, not aggravating, aileron/rudder response) -> the full nominal landing attempt.
3. Do not skip the shadow-mode step: the mock proves the code paths work, but only real JSBSim
   telemetry can reveal whether the trained networks' predictions are well-behaved outside the
   exact conditions seen in mock replay.

Per the task's explicit scope, this report does not claim the aircraft can land autonomously --
that can only be established by completing the steps above against the real simulator.
