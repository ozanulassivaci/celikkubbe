# Çelikkubbe — project briefing

Dense reference for picking this codebase back up. Not user documentation —
assume familiarity with Python, control systems, and the code itself.

## Project overview

TEKNOFEST air defence competition entry. A pan/tilt turret with a depth
camera detects, tracks, and engages targets. Three competition stages:

- **Stage 1** — manual: operator + gamepad drive selection and firing.
- **Stage 2** — fully autonomous.
- **Stage 3** — fully autonomous with IFF (friend/foe by target colour).

**Responsibility split, fixed by the electronics, not a style choice:**
the PC perceives and decides (which target, what angle); the STM32 moves
(trajectory generation, step pulses, encoder loop). Every core/ design
decision assumes the PC never has a measured turret position — see
"encoders are driver-side" below.

## Architecture

```
core/       decision layer — state machines, gates, priority, health
vision/     perception — frame sources, L2 colour detector
tracking/   Kalman filtering, association, track lifecycle
demos/      headless integration (no camera, no STM32, no GUI)
```

Import direction inside `core/` is strictly one-way:

```
types -> commands -> protocols -> {config, clock, health, gates, priority, strings}
                                -> {cascade, modes, engagement}
```

`vision/` and `tracking/` depend on `core/` (types, config, protocols,
clock) but `core/` never imports from them. `demos/` sits on top of
everything and is the only place all three layers plus a simulated
`TurretLink` are wired together.

**What each module owns:**

- `core/types.py` — enums + dataclasses, zero internal deps.
- `core/commands.py` — `Command` union (`SetMode`, `Goto`, `Jog`,
  `SetVelocity`, `Home`, `Arm`, `Disarm`, `Fire`, `SoftEstop`, `SetParam`).
  No `MotorEnable` — see hardware facts.
- `core/protocols.py` — `Clock`, `FrameSource`, `TurretLink` interfaces.
- `core/config.py` — every threshold. Nothing else may hardcode a number.
- `core/clock.py` — `SystemClock`, `FakeClock`.
- `core/health.py` — sliding-window monitors (`InferenceHealth` uses a
  consecutive-failure counter, not a rolling average).
- `core/gates.py` — `SafetyGate`, `IFFGate`, `RangeGate`, `ConfidenceGate`,
  `AngleGate`, `LimitGate`, `evaluate_all()`.
- `core/priority.py` — `compute_risk_score`, `order_track_ids` (hysteresis),
  `filter_engageable` (permanent FRIENDLY exclusion).
- `core/cascade.py` — L1/L2/L3 layer selection + recovery monitor.
- `core/modes.py` — M1–M4 mode machine.
- `core/engagement.py` — S1–S6 engagement machine, `StepResult`.
- `core/strings.py` — Turkish UI strings keyed by `ReasonCode`. Nothing in
  decision logic imports this.
- `vision/sources.py` — `SyntheticSource`, `VideoFileSource`,
  `WebcamSource`. `RealSenseSource` does not exist yet (no camera in hand).
- `vision/l2_color.py` — `ColorDetector`: the **permanent** L2 fallback,
  not a scaffold. Runs whenever L1/YOLO health degrades.
- `tracking/kalman.py` — `CentroidKalmanFilter` (normalised xy + velocity),
  `RangeKalmanFilter` (separate, idles without a measurement).
- `tracking/association.py` — `associate()`: IoU cost + Hungarian
  (`scipy.optimize.linear_sum_assignment`) + colour gating.
- `tracking/manager.py` — `TrackManager`: owns track lifecycle,
  confirmation, class/IFF voting.
- `demos/pipeline.py` — `python -m celikkubbe.demos.pipeline --source
  {synthetic,video,webcam}`. `SimulatedTurretLink`, `stub_aim_solutions`
  (bearing-only, no lead compensation — real aim solving doesn't exist).

L1/YOLO does not exist yet. `cls` is currently always `None` in the whole
system (L2 is the only detector). `cascade.py`'s L1 health monitoring has
nothing real to watch yet.

## Hard constraints

- No `PyQt6`/`PySide6` imports anywhere in `core/`.
- No `pyserial`, `pyrealsense2`, `cv2` imports in `core/` — those belong to
  `vision/`/I-O layers. `vision/` and `tracking/` *are* allowed `cv2`
  and `numpy`.
- State machines (`modes.step`, `engagement.step`, `cascade.Cascade`) never
  execute anything. They return `Command` objects or mutate nothing;
  an outer control loop executes commands and applies results via
  `dataclasses.replace()`.
- Never call `time.monotonic()` directly. Every time-dependent class takes
  a `Clock`; tests use `FakeClock`, never real sleeps.
- Python 3.10, `from __future__ import annotations`, full type hints,
  `frozen=True` dataclasses for anything crossing a thread boundary.
- All code/identifiers/comments/docstrings in English. Turkish user-facing
  text lives only in `strings.py` as `REASON_CODE_TR`, keyed by
  `ReasonCode`, never inlined.
- No magic numbers — every threshold in `config.py`, even ones added for
  a single test.

## Non-obvious decisions and why

**Normalised coordinates everywhere.** `BoundingBox = tuple[x1,y1,x2,y2]`
(min corner, max corner, each in [0,1]) — never pixels, never
`(cx,cy,w,h)`. Dev happens on 640×480 webcams, competition runs a
1920×1080 D435i; no coordinate may cross a module boundary in pixels or
the two would be silently incompatible.

**Encoders are driver-side, not MCU-side.** Confirmed from the
electronics schematic: motor encoders wire to the stepper drivers, which
close the position loop internally. The MCU only knows how many step
pulses it issued. Consequences:
- `Telemetry.pan_deg`/`tilt_deg` are the **commanded** position (step
  integral), not measured. Documented on the dataclass because it's easy
  to misread and dangerous to misread.
- `following_error_deg` was removed — it was never observable by the MCU.
- `AngleGate` cannot compute an angle delta. It passes on three
  MCU-reportable facts instead: the echoed setpoint matches what was last
  commanded within `SETPOINT_ACK_EPSILON_DEG` (`SETPOINT_NOT_ACKED`
  otherwise — guards a race where telemetry still echoes the *previous*
  Goto for a few ticks), `Telemetry.motion_complete` is true
  (`MOTION_IN_PROGRESS` otherwise), and neither driver alarm is set
  (`DRIVER_ALARM` otherwise).
- `ANGLE_TOLERANCE_DEG` is no longer evaluated directly by `AngleGate`;
  it's meant to configure the MCU's own trajectory generator once
  `SetParam` wiring exists (not built).
- `MotorEnable` command was dropped entirely — ENA inputs aren't wired,
  so software motor disable can't be implemented. Add back if hardware
  changes.

**`aim_solutions: dict[track_id -> (az, el)]`, not a single angle.**
`engagement.step()` chooses which track to engage internally (S3, via
`priority.py` + hysteresis). The outer layer can't precompute one angle
without predicting that choice, and both ways of predicting it are
broken: a stale previous-tick selection lags a frame (measurable angular
error at 30 Hz), and re-running `priority.py` outside duplicates
selection logic and can silently diverge from `step()`'s own hysteresis
state. So the outer layer solves for every confirmed track and hands over
the whole map; `step()` looks up the one it picked. Missing entry for the
selected track → `NO_AIM_SOLUTION`, stays in S4.

**Asymmetric colour-gated association.** `tracking/association.py`
gates track/detection matching by colour, but not symmetrically:
- A track voted `FRIENDLY` is **hard**-gated — a mismatched detection is
  ineligible, period. Fail-safe IFF voting must never un-commit from
  `FRIENDLY` once earned.
- A track voted `HOSTILE`/`UNKNOWN` is **soft**-gated — a mismatch only
  adds `COLOR_MISMATCH_COST` to the association cost. It loses to any
  same-colour candidate but can still win if nothing else is available.

  Reason: a genuine hostile track hit by one bad blue reading (specular
  highlight, motion blur, momentary overlap) must not be permanently
  relabelled and orphaned under a new `track_id` — because
  `engagement_attempts` and `deferred` are keyed by `track_id` in the FSM
  (see below), identity churn silently resets bookkeeping that has
  nothing to do with tracking.

**Exclude (permanent) vs. defer (temporary) in `engagement.py`.** Gate
failures on a *selected* S4 target aren't all the same kind of problem:
- `TARGET_FRIENDLY` never clears (fail-safe voting is permanent), so
  `priority.filter_engageable()` excludes friendly tracks *before*
  scoring — they never enter the candidate list, never get selected. They
  still appear in `SystemState.tracks` (UI shows blue boxes, counts them);
  exclusion is scoped to engagement selection only.
- `RANGE_OUT_OF_BOUNDS`, `RANGE_UNKNOWN`, `LIMIT_EXCEEDED`,
  `NO_AIM_SOLUTION` can all clear on their own (target moves into range,
  crosses back inside limits, aim solver catches up). A track stuck on
  one of these for `GATE_REJECT_TIMEOUT_MS` is recorded in
  `SystemState.deferred: dict[track_id -> (defer_until, ReasonCode)]`,
  cleared from selection, S3 tries the next candidate. Pruned every tick
  regardless of current engagement state, so a track that recovers while
  a different target is selected is available next time S3 looks. All
  candidates excluded/deferred → S1, not spinning in S3.

**`UNKNOWN_CLASS_RANGE` — the Stage 3 silent-fail fix.** `RangeGate`
fails closed on missing class, and L2 never produces a class. So the
moment the cascade falls back to L2 in Stage 3, *every* target became
permanently range-rejected — mute, not conservative, the opposite of the
design intent ("L2 fallback applies conservative IFF"). Fix: when
`cls is None` and `iff is HOSTILE` (colour-confirmed) in Stage 3,
`RangeGate` applies `config.UNKNOWN_CLASS_RANGE` — the intersection of
every class's rule (`max(los), min(his)`), currently `(10.0, 15.0)` —
instead of rejecting outright. Narrowest band valid for *any* possible
class: genuinely conservative. Derived via
`config.derive_unknown_class_range(RANGE_RULES)`, a function, not an
inline constant, so it tracks future `RANGE_RULES` changes and is
testable against a different table. Scoped to Stage 3 only — Stage 1/2
already allow an unknown class through unconditionally and must not get
*stricter*. `range_m is None` still rejects regardless — no range at all
is a different problem from no class.

**`engagement_attempts` lives in the FSM, keyed by `track_id`, never on
`Track`.** `Track` is frozen and rebuilt fresh every frame by
`tracking.py`; a counter stored on it would reset every frame. It lives
in `SystemState.attempts: dict[track_id -> int]`, owned by
`engagement.py`, threaded forward via `StepResult` and
`dataclasses.replace()` exactly like `deferred` and `gate_fail_since`.

**Frozen dataclasses everywhere state crosses a boundary.**
`SystemState`, `Track`, `Detection`, `Telemetry`, every `Command` variant.
`SystemState` is the immutable snapshot a GUI thread reads while the
control worker computes the next one; a mutable shared object would be a
cross-thread mutation hazard. (`Track` was mutable in an earlier draft —
changed after realising the same hazard applied to the track list a GUI
would render.)

**IFF from colour, class from nowhere.** Competition targets are
colour-coded (hostile red, friendly blue), so L2 sets `Detection.iff`
directly from which `ColorClass` matched, even though it can never set
`cls`. `IFFGate` takes only `iff` (rejects `FRIENDLY` or `UNKNOWN`) — it
used to also reject on missing class, but that constraint correctly
belongs to `RangeGate` alone (single place for it).

**Range estimation prefers depth, falls back to size, never a single
pixel.** Depth: median of valid pixels in a small window at the centroid
(D435i drops out and is noisy). Size fallback: `fx * size_m / pixel_size`
using `minAreaRect`'s larger dimension (not the enclosing circle's
diameter, which overstates size for non-circular aircraft silhouettes),
assuming the *largest* of the three known sizes — conservative, since
overestimating range is the safe direction of error for a gate bounding
engagement distance. Deliberately **not** gated on
`CameraIntrinsics.is_reliable` (that flag governs crosshair-placement
precision, a different concern) — confirmed necessary because
`SyntheticSource` always reports `"estimated"` quality and must still
exercise size-based ranging end to end. `range_source: Literal["depth",
"size", "none"]` on both `Detection` and `Track` records which.

**L2 confidence is bounding-box fill ratio, not circularity.** Aircraft
aren't circular; gating on circularity would reject legitimate detections
of non-circular shapes. Circularity is still computed and reported in
`DebugMasks` for the tuning panel; `require_circularity` (default
`False`) makes gating on it opt-in.

**Stage 1 shares one state machine with Stage 2/3.** Not a separate
manual path — `step()` runs S1–S6 identically; only the S4→S5 trigger
differs (autonomous: gates pass; Stage 1: gates pass **and**
`operator.fire_requested` **and** `operator.arm_held`, a continuous
dead-man switch — releasing it anytime before the shot aborts to S3).
Keeps "never fire without passing every gate" true in exactly one place.

**M4 SAFE has no automatic exit.** Deliberate safety decision. Only
`operator_ack_fault=True` moves M4 → M1 for a fresh self-test.

## Confirmed hardware facts

- MCU: **Nucleo-G431KB** (STM32G431).
- Pan motor: **JK-HSD86** stepper, 5:1 gear reduction.
- Tilt motor: **JK-HSD57** stepper, 2.5:1 belt reduction.
- PC↔MCU link: **RS422** via **MAX490** transceiver.
- Firing trigger: **PC817** optocoupler.
- Motor driver **ENA inputs are not wired** — no software motor disable
  possible (`MotorEnable` command removed accordingly).
- Encoders wire to the **stepper drivers**, not the MCU — see "encoders
  are driver-side" above.
- Camera: **Intel RealSense D435i**, mounted on the chassis (does not
  rotate with the turret). Assumed 69° horizontal FOV for all
  `"estimated"`-quality intrinsics (`vision.sources.estimated_intrinsics`).
- Firing unit: HPA airsoft.
- Target colours (confirmed from competition spec):
  hostile red `#F50A0A` → OpenCV HSV H0 S245 V245;
  friendly blue `#00A3E0` → H98 S255 V224 (azure, **not** true cyan —
  friendly hue range is `(90, 108)`, not centred on H90).
- Target sizes: 30 cm (drone), 40 cm (missile length), 50 cm
  (helicopter/F-16) — `known_sizes_m` in `ColorDetectorConfig`.
- Course range: 15 m. `RANGE_RULES` (metres): F16 (10,15), HELICOPTER
  (5,15), MISSILE (5,15), UAV (0,15), BALLOON (0,15 — leftover from a
  pre-confirmation draft, not in the actual target list, kept because
  removing it wasn't asked for and nothing currently depends on it being
  gone).

## Open questions / not yet built

- `RealSenseSource` — no camera in hand yet.
- L1/YOLO detection layer — doesn't exist; `cls` is always `None`
  everywhere in the system today. `cascade.py`'s L1 health path is
  unexercised until this lands.
- Ballistic lead-angle computation — `demos/pipeline.py` uses a
  bearing-only placeholder (`stub_aim_solutions`) with no lead
  compensation and no ballistics.
- RS422 serial protocol encode/decode — not implemented.
- GUI — out of scope for `core/` by hard constraint, not started.
- `SetParam` wiring to push `ANGLE_TOLERANCE_DEG` (and similar) to the
  MCU's trajectory generator — constant exists, plumbing doesn't.
- Empirical placeholders needing retuning against real (non-synthetic)
  detections: `AIM_MAX_VEL_DPS`/`AIM_MAX_ACCEL_DPS2`
  (`engagement.py`'s Goto profile), Kalman process/measurement noise
  constants (`tracking/kalman.py`), `MAX_ASSOCIATION_DISPLACEMENT`,
  `COLOR_MISMATCH_COST = 0.5` (`tracking/association.py`).
- `GATE_REJECT_TIMEOUT_MS` (1000) / `DEFER_COOLDOWN_MS` (3000) are first-
  pass values, not measured against a real multi-target engagement.

## Testing conventions

- pytest, everything driven by `FakeClock` — no real sleeps, no wall-clock
  flakiness.
- Coverage gate: `fail_under = 90` in `pyproject.toml`; actual is ~97%
  across `core`/`vision`/`tracking`/`demos`.
- ruff: line length 100, `target-version = "py310"`,
  `select = ["E", "F", "I", "UP", "B"]`. Run `ruff format` too — several
  past commits needed a follow-up formatting pass after edits shortened
  lines below the wrap threshold.
- `tests/factories.py` — shared `make_track`/`make_telemetry`/`make_state`
  builders with sane defaults, not collected by pytest (no `test_`
  prefix). Same pattern per-package: `tests/tracking/helpers.py`.
- Vision/tracking tests use `SyntheticSource` + NumPy-generated frames
  exclusively — nothing depends on a file that might not exist.
  `VideoFileSource`/`WebcamSource` tests mock `cv2.VideoCapture` directly.
- Conventional Commits, one logical change per commit, commit body
  explains *why* (root cause / trade-off), not what the diff already
  shows.
