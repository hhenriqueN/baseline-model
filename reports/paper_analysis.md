# Paper analysis: Baomar & Bentley (2021)

**Full citation**: Baomar, H., & Bentley, P. J. (2021). Autonomous flight cycles and
extreme landings of airliners beyond the current limits and capabilities using
artificial neural networks. *Applied Intelligence*. https://doi.org/10.1007/s10489-021-02202-y
(accepted 7 Jan 2021, open access, CC BY 4.0).

Read in full (27 pages) from the local PDF
`Baomar_Baomar-Bentley2021_Article_AutonomousFlightCyclesAndExtre.pdf`. Every claim
below is tagged **[reported]** (stated in the paper, with locator), **[inferred]**
(not stated explicitly but reasonably concluded from context), or **[this project]**
(what our baseline does, stated for contrast — not a claim about the paper).

## 1. Training-data procedure **[reported]**

- Simulator: X-Plane, communicating with the authors' custom "IAS interface" over
  UDP, sending/receiving packets **every 0.1 s** (Section III.A.1, p.4-5).
- A human teacher flies the aircraft using a Logitech/Saitek X52 Pro HOTAS while the
  interface logs flight-data inputs and the pilot's control-surface outputs into a
  SQL Server database as paired input/output vectors (Section III.A.2-4, p.5).
- **Number and duration of demonstrations** **[reported]**: **one** demonstration
  flight, London Heathrow (EGLL) → Birmingham (EGBB), flown by a single captain
  (9000+ flight hours) (Section IV, p.10-11, "the pilot provided one demonstration
  of a short flight from one airport to another"). Duration is not given in
  minutes/seconds; only the route and phase sequence (ground-run → takeoff, climb to
  10,000 ft cruise at 240 kt, descent at the standard 3° glideslope, deceleration to
  150 kt, manual hand-flying below ~1500 ft to touchdown) are described (p.11).
  Aircraft: Boeing 787 Dreamliner model (p.10, "a certified Boeing B787 Dreamliner
  model...rather than a light single-engine model"). No held-out validation split is
  described; each ANN is trained directly on its slice of that single flight's data
  until an MSE threshold is met (see §5 below).

**[this project]**: our baseline instead uses **15 separate landing
demonstrations** (12-14 usable sessions after excluding the familiarization flight)
from a Cessna 172P, each with its own scenario (normal/gust/turbulence/crosswind),
session-grouped leave-one-out cross-validation with held-out validation sessions per
fold — a materially larger and more rigorously validated dataset than the paper's
single-flight regime, though still small by general deep-learning standards.

## 2. Neural-network decomposition **[reported]**

The full IAS has **22 feedforward ANNs** total; Table 1 (p.6) lists the **13** that
are relevant to this paper (the rest — brakes, gear, emergency handling — are
described in the authors' earlier papers [2-5], not this one):

| ANN | Flight phase | Input | Output |
|---|---|---|---|
| Pitch Rate of Change | Takeoff | pitch − desired pitch | desired pitch-rate |
| Elevators | Takeoff | pitch-rate − desired pitch-rate | elevator command |
| Altitude Rate of Change | Cruise | altitude − desired altitude | desired climb/sink rate |
| Elevators Trim | Cruise | rate − desired rate (from above) | elevator-trim command |
| Speed Rate of Change | All | speed − desired speed | desired speed rate |
| Throttle | All | speed-rate − desired speed-rate | throttle command |
| Flaps | Takeoff/approach/final | altitude, flight phase | flaps command |
| Roll | All | roll − desired roll (0°/centerline) | desired roll angle |
| Ailerons | All | roll − desired roll (from above) | aileron command |
| Heading | Landing (ground) | heading − desired heading (0°/centerline) | desired heading |
| Rudder | Final approach, landing | heading − desired heading (from above) | rudder command |
| Glideslope Rate of Change | Approach, final approach | glideslope° − desired glideslope° | desired glideslope rate |
| Glideslope Elevators Trim | Approach, final approach | rate − desired rate (from above) | elevator-trim command |

**Pattern [reported]**: nearly every control surface is handled by a **cascade of
two ANNs** — an outer "rate-of-change" ANN that converts a *state* error (pitch,
altitude, speed, roll, heading, glideslope) into a *desired rate*, and an inner
"actuator" ANN that converts the error between the current and that desired rate
into the actual control-surface command. This is the paper's concrete meaning of
"current vs. desired" inputs (Section III.B.1, p.6-7, Table 1): the errors are
**always aircraft-state tracking errors** (angle, rate, or speed differences),
**never** actuator/control-surface positions fed back as inputs. This directly
confirms — rather than contradicts — the current project's spec section 14
prohibition on using surface positions as neural inputs.

## 3. Architecture, activation, loss, optimizer, stopping criterion **[reported]**

- Architecture: "fully connected single-hidden-layer Artificial Neural Networks"
  (Abstract, p.1; Section II, p.4, "our system uses Supervised Learning by applying
  fully connected single-layer ANNs"). Exact hidden-layer width is **not stated
  numerically** anywhere in the text for any of the 13 ANNs — Fig. 3 (p.7) shows the
  topologies graphically but the extracted text does not give neuron counts, and no
  table lists them. **[inferred]**: given the very small (single-flight) training
  set and 1-3 scalar inputs/outputs per ANN, hidden layers were almost certainly
  narrow (order of a handful of units), but we do not claim a specific number since
  it is not reported.
- Activation: Hyperbolic Tangent, Φ(x) = tanh(x) (Eq. 1, p.7), chosen explicitly
  "because it can handle negative values compared to the Sigmoid function."
- Training algorithm: standard backpropagation with momentum (Eq. 2-3, p.7):
  Δw(t) = −ε·∂E/∂w(t) + α·Δw(t−1), with **learning rate ε = 0.1** and **momentum
  α = 0.9**, "commonly used settings in feed-forward ANNs for Supervised Learning
  problems [36]," applied **uniformly to all 22 ANNs** (p.7).
- Loss function: **Mean Squared Error (MSE)**, stated repeatedly as the metric each
  ANN is "trained until a low MSE value was achieved (below 0.01)" (Section IV.A-G
  training subsections, p.11-13) — e.g., Elevators ANN MSE 0.004, Pitch Rate of
  Change ANN MSE 0.001 (p.13), Elevators Trim/Climb Rate ANNs MSE 0.01/0.0003
  (p.13), etc.
- **Stopping criterion [reported]**: a fixed **MSE threshold (< 0.01)**, not a
  fixed epoch count and not early-stopping against a held-out validation loss. No
  gradient clipping, weight decay, or regularization of any kind is mentioned.
  No train/val split, cross-validation, or generalization-gap discussion appears
  anywhere in the paper.

**[this project]**: AdamW optimizer, lr=1e-3 with weight decay 1e-4, gradient-norm
clipping at 5.0, early stopping on a **held-out validation** MSE with patience=25
over a max of 300 epochs, SiLU activations, seed=42 — a materially more
regularized/validated regime than the paper's, chosen because our dataset, while
larger session-count-wise, is still small and our evaluation standard (leave-one-
session-out CV) is stricter than the paper's (which reports no held-out
generalization metric at all for training).

## 4. Normalization **[reported: not mentioned]**

No normalization, standardization, or input-scaling procedure of any kind is
described anywhere in the paper — not in the training-algorithm description (p.7),
not in any of the 8 experiment subsections. Inputs appear to be used in their raw
physical units (degrees, ft/min, knots) directly into tanh-activated networks.
**[inferred]**: this is plausible because each ANN's inputs are all low-magnitude
*error* values (already differences, not raw magnitudes) fed into a single hidden
layer with tanh saturation — the paper simply does not need or report an explicit
z-scoring step. **[this project]**: we z-score all continuous inputs (train-fold
statistics only, clipped to [-5,5]) per spec section 11, since our lateral/
longitudinal networks take *raw-magnitude* state variables (AGL in feet, airspeed in
knots, along-track distance in meters) directly, not pre-differenced errors, so
normalization is necessary here in a way it may not have been for the paper's
cascaded-error design.

## 5. Control frequency **[reported]**

**0.1 s (10 Hz)**, both for data collection (Section III.A.1, p.4-5, "the simulator
is set up to send and receive packets...every 0.1 s") and for closed-loop
autonomous control (Section III.C.1, p.9, "receives flight data from the flight
simulator every 0.1 s...sends them to the flight simulator as autonomous control
commands...every 0.1 s"). This matches the current project's target 10 Hz baseline
frequency exactly (spec section 9) — the one place the paper's design point and
ours coincide precisely.

## 6. Use of current-vs-desired errors **[reported]**

Confirmed throughout Table 1 and Section III.B.1 (p.6-7): every ANN pair takes a
"current value − desired value" input, but in **every case** the two values being
differenced are **aircraft state variables** (pitch, altitude, climb rate, speed,
roll, heading, glideslope angle) — never a control-surface position, current
actuator command, or trim setting used as an *input*. Trim and elevator commands
only ever appear as **outputs**. This is exactly the reading spec section 14 of the
current project's task insists on, and the paper's own design supports it: we do
**not** interpret "current minus desired" as license to feed actuator positions into
our networks; our analogous errors are cross-track, heading, glideslope, airspeed,
and height-above-threshold — all aircraft-state tracking errors, none of them
control-surface state.

## 7. Simulator evaluation protocol **[reported]**

Section IV, p.10-20: after training, "the aircraft was reset to the runway in the
flight simulator" and the IAS was engaged to fly **closed-loop** (Sections IV.A-G,
"Autonomous Control" subsections). Performance is judged by three comparisons,
selected per experiment: (a) IAS vs. the demonstration pilot, (b) IAS vs. the
aircraft's standard/stock autopilot, (c) current IAS vs. the authors' earlier IAS
version [4]. Statistical comparison method: the **Two One-Sided Test (TOST)**
equivalence test [40] (Schuirmann 1987) on the means of the held parameter (pitch,
altitude, climb rate, speed, glideslope angle, or centerline angle) between IAS and
comparator, α = 0.05 (Tables 2-19, p.12-20). For the crosswind-landing centerline
experiments the metric switches to a **binary success-rate count** — "successful"
if the aircraft stays within ±0.05° of the runway centerline (Tables 20-21, p.17-18).

## 8. Number of evaluation attempts **[reported]**

Varies by experiment, all stated explicitly:
- Takeoff pitch: 10 (Fig. 7 caption says 10, Table 2 legend area says 15 — the
  figure caption and a later cross-reference are inconsistent in the source text;
  we report both numbers as they appear rather than silently picking one) takeoff
  runs (p.13).
- Altitude/speed maintenance: 3 altitudes × 3 speeds combinations, plus dedicated
  speed-transition trials (Section IV.B/D, p.13-19).
- Climb-rate maintenance: 6 climb/sink rates × 6 speeds (Section IV.C, p.15-17).
- Runway-centerline crosswind test (airborne, extreme 90° crosswind case): **10
  attempts**, IAS 10/10 vs. previous IAS version 4/10 (Table 20, p.17-18).
- Runway-centerline crosswind test (airborne + ground, modified 360° wind case):
  **20 airborne attempts** (IAS 20/20 vs. standard autopilot 5/20) and **10 ground
  (post-touchdown) attempts** (IAS 10/10 vs. standard autopilot 2/10) (Table 21,
  p.18).

## 9. Environmental / extreme conditions tested **[reported]**

Two named extreme-weather profiles recur across the final-approach/glideslope and
centerline experiments (Section IV.F-G, p.16-18):
1. 90° crosswind at 50 kt with gusts to 70 kt, wind shear direction 70°, strong
   turbulence — under this profile the standard autopilot "kept disengaging every
   time," so no IAS-vs-autopilot comparison was possible and the authors instead
   compared current-vs-previous IAS versions (p.17-18, Table 20).
2. A relaxed 360° wind at 50 kt, gust to 70 kt, wind-shear direction 70°, lower
   turbulence — used specifically so the standard autopilot could complete the
   approach and a direct IAS-vs-autopilot comparison could be made (Tables 17-18,
   21).
3. A single touchdown illustration (Fig. 33, p.20) under measured 64 kt crosswind
   at ~270°, described as "beyond the current limits and capabilities" of
   conventional autopilots/human pilots.

## 10. Reported limitations **[reported]**

- Elevators Trim ANN shows recurring **oscillations** in both the Altitude
  Maintenance and Climb Rate Maintenance experiments, attributed to that single ANN
  being trained to handle two different tasks (holding altitude and holding
  climb/sink rate) through the same control surface; the authors suggest — as future
  work, not a solved result — that further task-splitting could reduce or eliminate
  it (Section IV, p.15-16).
- The IAS is explicitly called "a proof-of-concept designed to prove the
  possibility of introducing intelligent autonomy to the cockpit, not a fully
  developed mature autopilot" (Section II, p.4, responding to a NASA/Honeywell
  evaluation report [31]).
- Training data is a **single demonstration per task** from a **single pilot**; no
  cross-pilot, cross-aircraft, or statistical generalization analysis is presented —
  robustness claims rest entirely on the closed-loop reset-and-refly evaluation, not
  on offline held-out metrics (no held-out split exists in their pipeline at all).
- The captain's own qualitative feedback (Section 5, p.24-25, an interview-style
  Q&A) states the IAS still "needs to be trained more on scenarios and various
  conditions and malfunctions" to be considered beyond "intermediate" pilot skill.

## 11. What this baseline explicitly does differently, and why

| Aspect | Paper | This project |
|---|---|---|
| Decomposition | 13 task-specific ANNs (rate ANN → actuator ANN cascades per surface) | 2 ANNs (lateral: aileron+rudder; longitudinal: elevator+throttle), spec sections 15-16 |
| Inputs | Pre-differenced current-vs-desired scalars (1 input per ANN in most cases) | 11/12 raw-magnitude state features per network (spec sections 15-16), not pre-differenced |
| Normalization | Not reported/used | Z-score (train-fold-only stats, clipped ±5), spec section 11 |
| Optimizer | Plain backprop + momentum (lr=0.1, momentum=0.9) | AdamW (lr=1e-3, weight decay=1e-4), spec section 17 |
| Stopping | Fixed MSE threshold (<0.01), no held-out split | Early stopping on held-out validation loss (patience=25/300 epochs), spec sections 17-18 |
| Validation | None (single flight, no CV) | Leave-one-session-out CV across 13 sessions + fixed diagnostic holdout, spec section 10 |
| Errors as inputs | Current-vs-desired **state** errors (never actuator positions) | Cross-track/heading/glideslope/airspeed/height errors (never actuator positions) — same principle, spec section 14 |
| Evaluation | Closed-loop reset-and-refly in the simulator, TOST equivalence + success counts | Offline behavioral-cloning regression metrics only, at this stage (spec section 24); closed-loop FlightGear evaluation is explicitly future work |

**Where the paper conflicts with the current baseline spec, the current spec's
decision is preserved** (per the task's instruction), specifically: the paper's
"current-desired" language could be misread as license to include actuator/trim
values as neural inputs, but a close read (Table 1, Section III.B.1) shows the
paper never does this either — so there is no actual conflict here, only a risk of
misreading avoided. The one place we deliberately diverge with justification is
network decomposition granularity (2 shared-trunk MLPs vs. 13 single-output ANNs):
spec sections 15-16 fix this decision for the current baseline and we do not
second-guess it against the paper's finer-grained design, which trades our
implementation simplicity for their traceability-per-surface, and which the paper
itself concedes causes cross-talk (the Elevators Trim ANN's dual-task oscillation)
that a cleaner separation-of-concerns architecture like the paper's is meant to
avoid — a tradeoff worth flagging for future baseline iterations but out of scope
to change now.
