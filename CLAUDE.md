# Çelikkubbe — project briefing

Dense reference for picking this codebase back up. Not user documentation —
assume familiarity with Python, control systems, and the code itself.

## Status

All layers built: `core/`, `vision/`, `tracking/`, `io/`, `geometry/`,
`ui/`. All committed, tested, pushed. 671 tests passing (gate is 90% in
`pyproject.toml`; actual fluctuates ~94-96% run to run — see Testing
conventions for why) across every package including `ui/`. The GUI was
built in four parts — shell/threading/theme/canvas; left/right panels
plus manual control; the HSV tuning window; keyboard/gamepad/click-to-aim
convergence — see "What `ui/` owns" and "ui/ decisions and why" below.
Nothing has run against real turret hardware or a physical gamepad yet;
everything is verified against `SimTurretLink`, synthetic sources, and
(for the gamepad) a hand-written fake `evdev` device.

Two field-testing fixes landed after the four GUI parts: a `--dev` mode
that bypasses the self-test/homing/estop gates that otherwise make it
impossible to run the GUI without a physical turret (see "`--dev` mode"
below), and four structural noise filters on the L2 colour detector —
computed `min_area_px`, a per-class detection cap, a solidity filter and
an aspect-ratio filter — added after a webcam in an ordinary room
produced dozens of spurious red/blue detections tight HSV thresholds
alone could not fix (see "vision/ decisions and why").

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

## Protocol

`docs/protocol.md` is signed off by the electronics team and is
authoritative for the PC↔MCU wire format. Both firmware and `io/` are
implemented from it; where code and the document disagree, the document
wins — flag the divergence, don't silently resolve it in code's favour.

**Corrected pin assignment** (differs from the original KiCad schematic):
**PB4 is the E-STOP input**; **PA5 drives the airsoft trigger** — the two
are swapped from the schematic's original assignment. Reason: PB4 is
NJTRST on the STM32G431 and comes out of reset in JTAG mode with an
internal pull-up already active, so wiring the trigger to it would
energise the PC817 LED on every power-up, reset and firmware flash —
unacceptable for a fire-control output that must default to no-fire. The
same pull-up is exactly what an E-STOP input wants instead (NO contact
pulls to GND, pull-up holds it high otherwise, no external resistor
needed). Consequent hardware requirement: fit a 10kΩ pull-down from the
PC817 LED anode to GND, so the trigger stays inactive even while the MCU
pin is high-impedance (e.g. before firmware has configured it).

## Architecture

```
core/       decision layer — state machines, gates, priority, health
vision/     perception — frame sources, L2 colour detector
tracking/   Kalman filtering, association, track lifecycle
io/         PC<->MCU link — wire codec, simulated + real TurretLink, worker
geometry/   pixel<->angle, parallax, ballistics, aim solving, calibration
ui/         PyQt6 GUI — shell, panels, tuning window, operator input
demos/      headless integration (no camera, no STM32, no GUI)
```

Non-package top-level dirs: `docs/protocol.md` (the spec above),
`docs/screenshots/` (GUI reference screenshots, the one deliberate
exception to the `*.png` gitignore rule — see below for what each one
shows), `config/` (calibration JSON and HSV tuning presets, created on
first save — empty and untracked until then; the whole `config/`
directory is gitignored), `assets/fonts/` (bundled monospace font for
`ui/theme.py`'s `load_monospace_font()` — empty today, see open
questions), `tools/` (dev-environment verification scripts, not part of
the installed package).

`docs/screenshots/` contents, in order captured: `01_startup_selftest`
(M1, self-test mid-run, stm32_link still failing) through
`06_crosshair_offscreen` (a parked, off-boresight crosshair with its
direction arrow) are the shell/canvas set, captured **before** Part 0's
UI-polish pass — they still show the English chrome (`SELF-TEST`,
`ACKNOWLEDGE`) and layout bugs (oversized event log, duplicated mode
text) that have since been fixed, so do not treat them as current UI
reference, only as shell/canvas-layer history. `07_populated_panels`,
`08_tuning_window`, `09_stage1_manual_control` are current: the left
and right panels both populated with a real mixed hostile/friendly
scene, the HSV tuning window's live source preview, and Stage 1's manual
control pad with its three mode cards. `10_manual_class_assignment`
(operator-assigned `F-16 [EL]` visible simultaneously on the canvas
overlay, the left panel's threat card and target list, and the right
panel's locked-target readout — the `[EL]` provenance tag proving all
four render sites agree) and `11_eyedropper_calibration` (the tuning
window's eyedropper row with NEGATİF ÖRNEK active and its result panel
reporting both classes reject a background sample) are captured against
`SyntheticSource`, not printed models — no physical models or camera
exist in this environment; see "Physical measurements still outstanding"
for the HSV-threshold caveat this implies.

Import direction inside `core/` is strictly one-way:

```
types -> commands -> protocols -> {config, clock, health, gates, priority, strings}
                                -> {cascade, modes, engagement}
```

`vision/`, `tracking/`, `io/` and `geometry/` all depend on `core/` but
`core/` never imports from any of them. `geometry/` additionally depends
on `vision/` (for `CameraIntrinsics`) — the only cross-dependency between
sibling layers; `io/` does not depend on `vision/` or `geometry/`, nor
they on it. `demos/` sits on top of everything and is the only place all
layers plus a `TurretLink` (simulated or, via `--link sim`, the richer
`SimTurretLink`) are wired together.

`ui/` sits at that same top level, alongside `demos/` — depends on every
layer below it, nothing below it imports `ui/` — and is the only place
`PyQt6` (or `evdev`) appears in this codebase, see Hard constraints.
`ui/pipeline_worker.PipelineWorker` is the GUI's equivalent of
`demos/pipeline.py`'s own loop: it is what actually wires a
`FrameSource`, the L2 detector, `TrackManager`, `AimSolver` and a
`TurretLink` together and runs them one tick at a time, just on a `QThread`
behind a live window instead of headlessly.

**What each module owns:**

- `core/types.py` — enums + dataclasses, zero internal deps. `McuMode`
  lives here (not `io/codec.py`) since `core/` can't import `io/` but
  `Telemetry.mcu_mode` needs the type.
- `core/commands.py` — `Command` union: `SetMode`, `Goto`, `Jog`, `Stop`,
  `Home`, `Zero`, `Arm`, `Disarm`, `Fire`, `SoftEstop`, `SetParam`. No
  `MotorEnable` — see hardware facts. No `SetVelocity` — removed, see
  io/ decisions. `Zero(axis, value_deg)` and `Stop()` were added after
  the initial core/ build — see io/ decisions for why.
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
- `core/modes.py` — M1–M4 mode machine. M2→M3 now additionally gated on
  `Telemetry.homed_pan`/`homed_tilt` — see io/ decisions. `step()` also
  takes `dev_mode`/`operator_skip_self_test`, a field-testing escape
  hatch — see "`--dev` mode" below.
- `core/engagement.py` — S1–S6 engagement machine, `StepResult`.
- `core/strings.py` — Turkish UI strings keyed by `ReasonCode` (now
  including `NOT_HOMED`). Nothing in decision logic imports this.
- `vision/sources.py` — `SyntheticSource`, `VideoFileSource`,
  `WebcamSource`. `RealSenseSource` does not exist yet (no camera in hand).
- `vision/l2_color.py` — `ColorDetector`: the **permanent** L2 fallback,
  not a scaffold. Runs whenever L1/YOLO health degrades.
  `ColorDetectorConfig.min_area_px` defaults to `None` (computed per
  frame from optics via `compute_min_area_px`, not a hardcoded pixel
  count) and three more structural filters — `solidity_min`,
  `aspect_ratio_range`, `max_detections_per_class` — gate every contour
  before it can become a `Detection`, on top of the HSV threshold itself.
  See "vision/ decisions and why" below.
- `tracking/kalman.py` — `CentroidKalmanFilter` (normalised xy + velocity),
  `RangeKalmanFilter` (separate, idles without a measurement).
- `tracking/association.py` — `associate()`: IoU cost + Hungarian
  (`scipy.optimize.linear_sum_assignment`) + colour gating.
- `tracking/manager.py` — `TrackManager`: owns track lifecycle,
  confirmation, class/IFF voting.
- `io/codec.py` — `encode_command`/`encode_heartbeat` (PC→MCU JSON+CRC),
  `FrameParser` (MCU→PC streaming binary decode), `TelemetryFrame`/
  `AckFrame`/`EventFrame`/`LogFrame`, `AckResult`/`EventId`.
- `io/sim_link.py` — `SimTurretLink`: real physics, not a stub. Fault
  injection API (`inject_estop`, `inject_driver_alarm`,
  `inject_link_dropout`, `inject_crc_errors`, `set_latency`).
- `io/serial_link.py` — `SerialTurretLink`: real RS422 over `pyserial`,
  auto-detects the Waveshare converter by USB VID/PID, reconnects with
  backoff. Never exercised against real MCU firmware — no hardware yet.
- `io/link_worker.py` — `LinkWorker`: owns a `TurretLink` in a background
  thread, heartbeats, ACK/retransmit tracking, link-health staleness.
  Not yet wired into `demos/pipeline.py` — see open questions.
- `geometry/frames.py` — `TurretGeometry`, `rotation_matrix`,
  `barrel_direction`, `muzzle_position`, `camera_to_turret`/
  `turret_to_camera`. `DEFAULT_TURRET_GEOMETRY` is an unmeasured
  placeholder.
- `geometry/projection.py` — `pixel_to_ray`, `pixel_to_turret_point`,
  `point_to_angles`, `crosshair_pixel`/`crosshair_with_indicator`.
- `geometry/ballistics.py` — `BallisticTable` (slant-range-indexed),
  `angular_velocity_dps` (undoes the Kalman filter's tan-nonlinearity),
  `lead_angle`, `gravity_only_drop_deg`.
- `geometry/solver.py` — `AimSolver.solve()`/`solve_all()`, `AimSolution`.
- `geometry/calibration.py` — `BoresightCorrection`/`BoresightTable`,
  JSON load/save for both calibrations, `refine_geometry()` (least
  squares, real but never run against a physical rig).
- `demos/pipeline.py` — `python -m celikkubbe.demos.pipeline --source
  {synthetic,video,webcam} --link {stub,sim} --inject NAME[:VALUE]@TIMEs`.
  `SimulatedTurretLink` is the lightweight `--link stub` default; `--link
  sim` uses the real `io.sim_link.SimTurretLink` with no `LinkWorker` in
  between (the demo loop calls `send_heartbeat()`/`poll()` itself). Aim
  solving is the real `geometry.solver.AimSolver` — `stub_aim_solutions`
  (bearing-only, no lead, no ballistics, no parallax) was removed once it
  existed.

L1/YOLO does not exist yet. `cls` is currently always `None` in the whole
system (L2 is the only detector). `cascade.py`'s L1 health monitoring has
nothing real to watch yet.

**What `ui/` owns:**

- `ui/theme.py` — colour tokens, `build_qss()`, and the custom-paint
  widgets QSS alone can't express (`StatusBadge`, `ToggleSwitch`,
  `RiskBar`, `AngleGauge`). `load_monospace_font()` falls back to an
  installed system family, then the generic `"monospace"`, when
  `assets/fonts/` has no bundled file — true today, see open questions.
- `ui/snapshot.py` — `UiSnapshot`/`HealthSnapshot`/`PipelineTimings`,
  the one frozen object `PipelineWorker` hands the GUI thread per tick.
  `ordered_track_ids` and `engagement_fallback_reason` exist purely so
  the GUI never has to recompute priority ordering or guess why S4_AIM
  is stalled — see the aim_solutions rationale above for the identical
  anti-duplication concern.
- `ui/pipeline_worker.py` — `PipelineWorker`: the `QThread` owning
  perception/tracking/decision, one tick at a time. `tick()` is a plain
  method, not hidden in `run()`, so tests call it directly against a
  `FakeClock` — same pattern as `io/link_worker.py`. Runs the six-item
  self-test (M1) and — a real bug this caught — skips `engagement.step()`
  entirely while self-test is in progress, since engagement runs
  regardless of Mode by design and would otherwise fight the self-test's
  own pan/tilt verification move for the link. `detector` and
  `aim_solver` are exposed as properties so the tuning window and
  click-to-aim reuse the exact live instances instead of constructing
  second ones that could silently drift.
- `ui/video_canvas.py` — the centre widget: frame, boxes, crosshair,
  lock banner, header/telemetry strips, all `QPainter`-painted over the
  displayed pixmap, never burned into the frame with OpenCV.
  `compute_letterbox`/`frame_to_pixmap`/`place_label`/`inset_point`/
  `place_badge`/`class_label`/`lock_banner_rect`/`indicator_row_top` are
  pure functions, extracted specifically so paint-adjacent placement
  math is unit-testable without pixel-sampling.
- `ui/left_panel.py` — threat level, `RiskBar`, the AI recommendation
  (`build_recommendation()`, surfacing `engagement.StepResult`'s own
  `fallback_reason` through `strings.describe()`), classification
  counts, the target list. Renders in `UiSnapshot.ordered_track_ids`
  order rather than recomputing one, and throttles to
  `config.TRACK_LIST_UPDATE_HZ`.
- `ui/right_panel.py` — stage/layer selection, the manual control pad,
  speed slider, homing row, and the pinned-bottom safety region (ACİL
  STOP / EMNİYET KİLİDİ / ATIŞ / warning line) that must never scroll
  out of reach. `compute_fire_enabled()` is the pure ATIŞ-enablement
  check; the disabled button's tooltip and the warning line both render
  its own returned reason, so "why can't I fire" can never drift from
  the real gate. `jog_speed_dps` is exposed so keyboard jog reuses the
  operator's own chosen speed instead of a second default.
- `ui/tuning_window.py` — live HSV tuning. Sliders write straight to
  the running `PipelineWorker.detector.config` as whole-object swaps
  (never individual field mutation — see its own docstring on why); the
  preview re-runs `detect(debug=True)` on fresh snapshots at a throttled
  10Hz, and only while the dialog is actually visible. Preset
  persistence (`config_to_dict`/`config_from_dict`/`save_hsv_preset`/
  `load_hsv_preset`/`list_hsv_presets`) lives in `vision/l2_color.py`,
  not here — the type being persisted belongs to that module, the same
  reasoning `geometry/calibration.py` owns `BoresightTable`'s own JSON
  persistence.
- `ui/operator_input.py` — `OperatorInputBuilder`: the single point
  keyboard, gamepad, and GUI-button fire/arm intent converge before
  reaching `PipelineWorker.set_operator_input()`. Each source reports
  its own held/not-held state under its own name; `build()` ORs every
  source together, so one source releasing (a dropped gamepad) can
  never clear what another is still holding.
- `ui/gamepad.py` — `GamepadWorker`: a Logitech F310 in XInput mode via
  `evdev`, on its own thread (`read_loop()` blocks). No device present
  is not an error — this hardware is optional and most development has
  none plugged in. A dropped or never-found device forces its own
  arm/fire state to `False` (Stage 1's RT-released treatment) without
  touching any other source's state.
- `ui/main_window.py` — the shell: title bar, three-column splitter,
  `StatusStrip`, and the self-test/SAFE/help overlays (the tuning window
  is a separate non-modal `QDialog`, not an overlay). Converges
  keyboard, an optional passed-in `GamepadWorker`, and canvas clicks
  into the one `OperatorInputBuilder`. Clicking empty canvas in Stage 1
  solves a synthetic zero-area `Track` through `PipelineWorker.aim_solver`
  for a direct `Goto`, so the click still gets ballistic drop and
  boresight correction rather than a raw bearing.
- `ui/overlays/selftest.py`, `safe.py`, `help.py` — full-window
  `QWidget` overlays painted with a semi-transparent backdrop over
  `centralWidget().rect()`. self-test/SAFE are shown or hidden purely by
  `Mode`; `help.py` is static content toggled by F1, the only overlay
  with no `UiSnapshot` involved at all.
- `ui/app.py` — `python -m celikkubbe.ui.app`. `build()` does
  everything except run the Qt event loop (so a test can call it
  without blocking on `app.exec()`); `main()` is the thin wrapper
  `__main__` calls.

## Hard constraints

- No `PyQt6`/`PySide6` imports anywhere in `core/` or `io/`.
  `io/link_worker.py`'s own docstring states this explicitly — telemetry,
  events and command outcomes reach the owner through plain callbacks so
  it stays usable headless.
- `PyQt6` appears **only** in `ui/` — every other package, including
  `demos/`, stays importable headless. `ui/gamepad.py`'s `evdev` import
  is not an exception to "hardware access lives in `io/`": a gamepad is
  operator input, not PC↔MCU communication, so it belongs in `ui/` for
  the same reason keyboard/mouse handling does, just through `evdev`
  instead of Qt's own event system.
- No `pyserial`, `pyrealsense2`, `cv2` imports in `core/` — those belong
  to `vision/`/I-O layers. `vision/` and `tracking/` *are* allowed `cv2`
  and `numpy`; `io/serial_link.py` is allowed `pyserial`; `geometry/` is
  allowed `numpy` and `scipy` (for `calibration.refine_geometry`'s
  least-squares fit) but nothing hardware- or Qt-related.
- State machines (`modes.step`, `engagement.step`, `cascade.Cascade`) never
  execute anything. They return `Command` objects or mutate nothing;
  an outer control loop executes commands and applies results via
  `dataclasses.replace()`.
- Never call `time.monotonic()` directly. Every time-dependent class takes
  a `Clock`; tests use `FakeClock`, never real sleeps. The one exception:
  `io/link_worker.py`'s background thread loop has a single real
  `time.sleep()` between ticks, but it's a pure CPU-yield that plays no
  part in any timing *decision* — every threshold check inside `tick()`
  compares against `clock.now()`, and tests call `tick()` directly
  against a `FakeClock` without ever starting the real thread.
- Python 3.10, `from __future__ import annotations`, full type hints,
  `frozen=True` dataclasses for anything crossing a thread boundary.
- All code/identifiers/comments/docstrings in English. Turkish user-facing
  text lives only in `strings.py` as `REASON_CODE_TR`, keyed by
  `ReasonCode`, never inlined.
- No magic numbers — every threshold in `config.py`, even ones added for
  a single test. (`io/`'s own link-timing constants —
  `DEFAULT_ACK_TIMEOUT_MS`, reconnect backoff — and `geometry/`'s
  `_BACKLASH_DEG`-style physical constants live as module-level constants
  in their own files instead, since they're not `core/` decision
  thresholds; still no bare literals inline.)

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
`geometry.AimSolver.solve_all()` now produces exactly this map for real,
not a stub.

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
- A real consequence of `MAX_ENGAGEMENT_ATTEMPTS` (3) worth knowing: once
  a track exhausts its attempts, it's permanently excluded from
  `_pick_target` too (not just deferred), and if the same physical target
  keeps re-confirming under the same `track_id` (e.g. a stationary
  synthetic target), the engagement machine will cycle S1→S2→S3→S1
  indefinitely without ever reaching S4 again. Seen directly in
  `tests/demos/test_acceptance.py`'s full-loop test, which had to move
  from a fixed-tick snapshot to a tick-until-condition loop once the real
  `AimSolver` started converging fast enough to exhaust the 3 attempts
  within what used to be a safe tick budget.

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
"size", "none"]` on both `Detection` and `Track` records which. (This is
`core.types.RangeSource`; `geometry.AimSolution.range_source` is a
separate, wider Literal that adds `"assumed"` — see geometry/ decisions.)

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

## vision/ decisions and why

**Structural filters, not tighter HSV, fixed L2's false-positive problem.**
A webcam pointed at an ordinary room produced dozens of red/blue
detections — tightening hue/saturation thresholds only ever trades false
positives for false negatives against real targets, because the actual
gap was that the detector had no constraint at all on a contour's shape
or count, only its colour. Four independent structural filters were
added instead, each gating a property no genuine competition target
violates but noise commonly does:

- **`min_area_px`, computed, not hardcoded.** The old fixed `12` was
  roughly 60x too permissive — a 50cm model at 15m is ~31px across at
  720p, ~750px of area. `compute_min_area_px()` derives the floor from
  the smallest known target (`0.30m`, the drone) at
  `config.MAX_ENGAGEMENT_RANGE_M` (reusing the same derived constant
  `priority.py` already uses, rather than inventing a second "how far
  can this course engage" number): `expected_px = fx * 0.30 /
  max_range_m`, `min_area_px = expected_px**2 * 0.3` — squared for area,
  scaled by `0.3` (not `1.0`) so a non-square aircraft silhouette isn't
  penalised for not filling its own bounding box. `ColorDetectorConfig.
  min_area_px` defaults to `None` ("auto" — computed every frame from
  the frame's own intrinsics) rather than removing the field; setting an
  int overrides it, e.g. a competition-day value tuned by eye against
  the real venue. `"estimated"`-quality intrinsics (a webcam) have a
  guessed `fx`, so computing from it would compound one guess with
  another; the auto path instead falls back to a fixed fraction of frame
  area (`_MIN_AREA_ESTIMATED_FRAME_FRACTION`), which scales with
  resolution the same way the reliable-intrinsics formula does (both fx
  and frame area scale with width² at a fixed FOV and aspect ratio).
  `ColorDetector` logs the computed floor once per distinct value
  (cached, not once per frame) so it stays visible without spamming the
  log at 30Hz. In `ui/tuning_window.py`'s min-area slider, `0` means
  "auto" (`None`); any value above `0` is an explicit override — the
  slider's own range had to grow from `1..200` to `0..2000`, since a
  real D435i's larger `fx` can genuinely compute a floor above the old
  slider's maximum.
- **`max_detections_per_class` (default 3).** The course has three lanes
  and at most three models of one colour; a fourth same-colour detection
  in one frame is noise that happened to pass every other filter, not a
  real target. Applied *after* every other structural filter (area,
  solidity, aspect ratio), sorted by area descending, keeping only the
  top N — so on a frame with more survivors than the cap, the *largest*
  contours win, not whichever the contour-scan order happened to reach
  first.
- **`solidity_min` (default 0.75), on by default** — unlike
  `require_circularity`. Solidity (contour area over its own convex hull
  area) doesn't assume anything about the target's silhouette the way
  circularity does, so it doesn't share circularity's problem of
  rejecting a legitimately non-circular aircraft. A solid printed model
  scores close to `1.0` regardless of shape; scattered shadow patches,
  specular streaks and fragmented blobs — genuinely concave or
  disconnected-looking noise — score well below.
- **`aspect_ratio_range` (default `(0.2, 5.0)`), via `cv2.minAreaRect`.**
  Rejects long thin colour bands (a strip of wall trim, a doorframe
  edge) no competition target's silhouette resembles.

All four report their specific rejection reason on `Contour.
reject_reason` (`"area"`, `"solidity"`, `"aspect_ratio"`,
`"class_cap"`, plus the pre-existing `"circularity"`) so
`ui/tuning_window.py`'s contour preview can show *why* a box is red, not
just that it is. That label is drawn with `cv2.putText` directly on the
preview image using the raw English slug, not
`REASON_CODE_TR`/`UI_LABEL_TR`'s Turkish text: OpenCV's Hershey fonts
cannot render Turkish diacritics (İ/Ş/Ğ/Ü/Ö/Ç come out missing or wrong),
and a corrupted label would be worse for a tuning tool than an English
one. The per-class-cap discard count is also reported separately in the
tuning window's counts label (`SINIF LİMİTİ: N`) from the general
accepted/rejected counts — a persistently high value there means real
targets are being discarded by the cap itself, a different problem from
a too-tight HSV/area/solidity threshold.

## io/ decisions and why

**Asymmetric protocol, deliberately.** PC→MCU is newline-delimited JSON
with a CRC16 trailer (`codec.encode_command`); MCU→PC is fixed binary
(`codec.FrameParser`). JSON PC→MCU is low-rate (≤50Hz) and human-readable,
so a serial terminal can drive the MCU directly during bring-up — worth
a lot during integration week. Binary MCU→PC is 100Hz telemetry; JSON at
that rate wouldn't fit the link's bandwidth budget as comfortably.

**`FrameParser` is a streaming resync parser, not a line reader.** Frames
split across reads reassemble; several frames in one read all parse; a
corrupted frame resyncs one byte at a time to the next `0xAA 0x55` rather
than discarding the whole buffer (a real `AA 55` inside payload data must
not swallow the next genuine frame either); buffer growth is capped so a
claimed frame that never completes — or a stream with no sync byte at
all — can't grow it unbounded.

**`send_tracked()`/`drain_acks()` exist on the concrete link classes, not
on `TurretLink`.** `TurretLink.send(cmd) -> None` gives no way to learn
which seq a command went out under, so `LinkWorker` can't correlate a
later ACK against it. Both `SimTurretLink` and `SerialTurretLink` add
`send_tracked(cmd, seq=None) -> int` (returns the seq used; passing one
explicitly reuses it, for retransmission) and `drain_acks() ->
list[(seq, AckResult)]` as duck-typed extras beyond the core `TurretLink`
Protocol — `core/protocols.py` stays untouched. `LinkWorker` degrades to
fire-and-forget `send()` against any link that doesn't provide these.

**Duplicate seq is discarded, so a retransmitted `Fire` cannot
double-fire.** docs/protocol.md section 3.1's stated behaviour.
`LinkWorker` retransmits an unacknowledged command once under the *same*
seq (not a fresh one) before giving up — if the original `Fire` actually
landed but its ACK was merely lost, a fresh seq would make the MCU (or,
in simulation, `SimTurretLink`) execute it twice. `SimTurretLink`
implements the dedup itself (`_last_applied_seq`) so this is exercised in
simulation, not just documented as a firmware responsibility.

**`Telemetry.t` is the PC-side receive timestamp; `mcu_ms` is a separate,
unsynchronised clock.** Same principle as `Clock`/`FakeClock` elsewhere —
never trust or compare the MCU's own uptime counter against `now()`.
`mcu_ms` lives on `codec.TelemetryFrame`, not `core.types.Telemetry`,
since nothing in decision logic needs it.

**`homed_pan`, `homed_tilt` and `mcu_mode` live on `core.types.Telemetry`
itself, not `codec.TelemetryFrame`.** They are decision inputs, not
diagnostics: `modes.py` refuses M2_STANDBY → M3_OPERATIONAL until both
homing bits are set (docs/protocol.md section 5 — no limit switches
exist, `zero` is the only homing mechanism), and `mcu_mode` exists for
detecting PC/MCU mode disagreement (the enum and field exist; the actual
disagreement check is not yet implemented — see open questions). `McuMode`
lives in `core/types.py` too, not `io/codec.py`, since `core/` cannot
import from `io/`. Every other wire-only field (`mcu_ms`, `ammo_fired`,
`watchdog_tripped`, `limit_pan`/`limit_tilt`) stays on
`codec.TelemetryFrame` as a diagnostic with no decision-logic consumer.
`SimTurretLink` computes its own `mcu_mode` independently of what the PC
last commanded via `SetMode` — reports `SAFE` immediately on
estop/watchdog/driver-alarm regardless, the same independent-authority
principle already applied to fire/aim rejection.

**`SimTurretLink` is not a stub.** Real trapezoidal velocity profiles
(closed-form, not incrementally integrated — exact under a `FakeClock`
jump of any size, however large), pan backlash modelled as a slack
dead-zone with the firmware's overshoot-and-return compensation
(`BACKOFF_DEG` > the 0.1° gap guarantees the final leg always fully
retakes the slack), gravity droop on the tilt axis when E-stop cuts
power, and its own independent 200ms watchdog and fire/aim safety
authority — a PC-side logic bug shows up in simulation exactly as it
would against real firmware, rather than being silently absorbed by a
lenient stub. Gotcha for anyone writing a test against it: it enforces
its watchdog for real, so advancing a `FakeClock` by more than 200ms
without calling `send_heartbeat()` trips it and freezes motion —
several tests during development had to switch from a single big
`clock.advance()` to a heartbeat-pumping loop for exactly this reason.

**`SetVelocity` removed, `Stop` added.** `SetVelocity` had no wire
equivalent (protocol only has single-axis `jog`, no simultaneous pan+tilt
velocity command) and nothing in `core/` ever emitted it — a command
that throws when called is a trap, so it was deleted rather than kept
undocumented-broken. `Stop` (`{"cmd":"stop"}`) was added because manual
jog needs a way to end motion when the operator releases a direction
button; heartbeats alone never do that, they only keep the watchdog
satisfied. `Command` is now `SetMode | Goto | Jog | Stop | Home | Zero |
Arm | Disarm | Fire | SoftEstop | SetParam` — `Zero(axis, value_deg)` was
also added (the actual homing mechanism; `Home` maps to the protocol's
`home` command, reserved until limit switches exist).

## geometry/ decisions and why

**Conventions, fixed — see frames.py's module docstring for the full
statement.** Camera frame: X right, Y **down**, Z forward (OpenCV
convention). Turret frame: origin at the pan/tilt axis intersection, but
— unlike a body-fixed barrel frame — does not rotate with pan/tilt; it's
a fixed frame whose axes coincide with the camera's at pan=0, tilt=0
(physically correct: the D435i is chassis-mounted, not turret-mounted,
so the camera↔turret relationship is one fixed rigid transform,
independent of the current pan/tilt). Azimuth positive right (viewed from
above — the same right-handed sense as aeronautical yaw about a
down-pointing axis: North→East is clockwise-positive from above).
Elevation positive up. The Y-down-to-elevation-up sign inversion
(`el = atan2(-y, hypot(x,z))`) is the single most likely place to
introduce a sign error in this layer — called out in every docstring
that touches it.

**Crosshair moves RIGHT as pan increases.** Not a judgement call about
"which way feels right" — forced by `point_to_angles`'s own
`az = atan2(x, z)` plus the round-trip identity
`point_to_angles(barrel_direction(pan, tilt)) == (pan, tilt)`, which the
whole aim-solving pipeline depends on (if it didn't hold, commanding the
solved angle would not point the barrel where the solver thought it
would). An early prompt draft asserted the opposite ("moves left");
resolved by deriving the direction from the stated azimuth convention and
the existing `point_to_angles` formula rather than trusting either
party's intuition, and confirmed by test.

**`range_m` is Z-depth**, matching both the D435i's native depth
convention and the size-based estimate — not straight-line distance from
the camera. `projection.pixel_to_turret_point` scales the pixel ray so
its Z component equals `range_m`, algebraically identical to the standard
pinhole back-projection `X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy, Z=Z`.

**Ballistics uses slant range, not Z-depth.** `range_m` staying Z-depth
is correct for placing the target's turret-frame point, but
`ballistics.BallisticTable.time_of_flight()`/`interpolate_drop_deg()`
need the actual muzzle-to-target distance the BB travels:
`slant_range_m = |target_point - muzzle_position|` in turret frame
(`frames.muzzle_position` accounts for `muzzle_offset_z_m`). The two
values agree only on-axis and diverge as `1/cos(theta)` off-axis, and
because the camera is chassis-fixed, targets are genuinely engaged well
off-axis, not just near boresight: at 15m Z-depth and 34.5° off-axis
(the edge of a 69° FOV), slant range is 18.2m, and confusing the two
costs ~0.09° of drop error — nearly the entire 0.10° aim tolerance. Both
values are on `AimSolution` (`range_m` and `slant_range_m`) so the
distinction stays visible rather than implied.
`ballistics.BallisticTable.ranges_m` is indexed by slant range throughout
— live fire naturally produces it (you fire at a target and measure its
actual distance). Boresight correction lookup is deliberately still on
`range_m` (Z-depth), out of scope for the slant-range fix, which was
specifically about ballistics.

**`AimSolution` reports components separately**
(`lead_az_deg`/`lead_el_deg`, `drop_deg`, `range_m`, `slant_range_m`,
`confidence`, `warnings`) rather than just the final `az_deg`/`el_deg`.
Lets a GUI show the operator *why* the turret is pointing where it is,
and keeps debugging a bad solve tractable instead of a black box. `ui/`
does not actually surface this breakdown yet — `left_panel.py`/
`right_panel.py` show class/confidence/range, not the lead/drop
components separately — so the capability this was built for is still
unused, not wrong.
Pipeline (bbox centre → ray → point at range → angles → lead → drop →
boresight → clamp) is additive — each correction summed independently,
not re-derived from a corrected aim point — because every correction is
sub-degree at these ranges, so the small-angle interactions between them
are negligible.

**A missing range yields a low-confidence solution, never `None`.**
`AimSolver.solve()` returns `None` only for a track that isn't yet
CONFIRMED. For any CONFIRMED track — even with no range at all — it
substitutes `config.DEFAULT_RANGE_M` and marks confidence `"low"` rather
than dropping the target: `engagement.step()` already treats a missing
`solve_all()` entry as `NO_AIM_SOLUTION`, and silently dropping a target
the operator can see on screen would be worse than a low-confidence
solution they can evaluate. Confidence rules: `"high"` needs both
depth-derived range and reliable (non-`"estimated"`) intrinsics —
estimated-quality intrinsics downgrade even depth-derived range to
`"low"`, not just size-derived, since the crosshair-placement imprecision
that implies affects every range source equally; `"medium"` is
size-derived range with reliable intrinsics; everything else is `"low"`.

## ui/ decisions and why

**`OperatorInputBuilder` ORs sources instead of last-write-wins.**
Keyboard, gamepad and the right panel's own ATIŞ button can all
independently want fire/arm held, and none of them know about the
others. A naive design where each source's handler called
`PipelineWorker.set_operator_input()` directly would let whichever call
landed last silently win — releasing a gamepad trigger could un-arm a
shot the keyboard's space bar is still physically holding down. Instead
every source reports its own held/not-held state under its own name
(`"keyboard"`/`"gamepad"`/`"gui"`) to one shared builder; `build()` ORs
them all. ATIŞ and space bar both set fire *and* arm together (neither
has a separate hold-to-arm control), but the gamepad's RT and A stay
independent, since on real hardware they are two different physical
controls — see `MainWindow._set_fire_and_arm_source` vs
`_set_fire_source`/`_set_arm_source`.

**EMNİYET KİLİDİ toggles `Telemetry.armed`, not `OperatorInput.arm_held`.**
These are two different safety layers, both genuinely required in Stage
1 (`engagement.py` checks both `telemetry.armed`, via `SafetyGate`, and
`operator.arm_held` independently). EMNİYET KİLİDİ is the master
safety catch — a toggle, rarely changed, sent as `Arm()`/`Disarm()` —
while `arm_held` is the continuous per-shot dead-man switch that ATIŞ/
space itself drives while held. Conflating them into one control would
lose the fail-safe property that releasing the trigger — independent of
whatever the master switch is set to — always aborts back to S3.

**Commands split into direct-send and deferred-request, by what they
touch.** `request_estop`/`request_arm`/`request_zero`/`request_jog`/
`request_stop`/`request_goto` all call `LinkWorker.send()` straight from
the GUI thread — safe because `LinkWorker.send()` already takes its own
lock, and doing this avoids the emergency-stop path ever waiting behind
`PipelineWorker`'s own tick loop. `set_stage`/`select_layer` cannot do
this: they mutate `SystemState` fields that only `_tick_inner` may write
(every other read of `self._state` in that class is itself
unsynchronised), so they instead record a pending request under a
dedicated lock, consumed once at the top of the next tick — the same
deferred-request pattern `set_operator_input` already used. Leaving
Stage 1 with L3 (Tam Manuel, a Stage 1-only override) still selected
falls back to L2 there rather than leaving that combination sitting in
`SystemState`.

**Two different cross-thread synchronisation strategies, chosen by
mutation pattern, not by habit.** `OperatorInputBuilder` takes a real
`threading.Lock` around two small dicts, because both the gamepad
thread and the GUI thread call its `set_fire`/`set_arm` **incrementally**
and concurrently — a lock is cheap here and there is no single-object
swap that would make sense for "OR every source's own flag together."
`ColorDetectorConfig` deliberately does the opposite: no lock at all,
because `PipelineWorker.detector` docstring's contract is "replace the
whole object, never mutate a field" — a single reference reassignment
is already atomic under the GIL, and adding a lock there would protect
against a mutation pattern (field-by-field editing) the API contract
already forbids. Picking the wrong one of these two for a given piece
of shared state is the actual risk, not omitting synchronisation
entirely.

**`GamepadWorker` is constructed and started outside `MainWindow`, not
inside it — dependency injection, same reasoning as `PipelineWorker`
itself.** `MainWindow.__init__` takes `gamepad: GamepadWorker | None =
None` and only wires signal handlers; `ui/app.py`'s `build()` is what
constructs the real one and calls `.start()`, exactly like it already
does for `worker.start()`. Tests that only need `MainWindow`'s own
logic construct it with no gamepad at all and never spin up a real
background thread; `closeEvent` still stops/joins one if given, for
symmetry with `PipelineWorker`'s own shutdown path. `GamepadWorker.
jog_axis_changed` emits one *signed* `speed_dps` per axis (sign encodes
direction, since a 2D stick can move both axes independently and there
is no per-axis `Stop`), whereas `RightPanel.jog_pressed` emits a
separate unsigned `speed_dps` plus a `direction: int` (matching
`core.commands.Jog`'s own field shape exactly). `MainWindow.
_on_gamepad_jog_axis_changed` is the one place that converts between
the two shapes before calling `PipelineWorker.request_jog` — deliberate,
so neither signal has to pretend to be the other's shape.

**Click-to-aim on empty canvas builds a synthetic `Track`, not a raw
bearing.** Stage 1, clicking where nothing is detected, still goes
through `PipelineWorker.aim_solver.solve()` — the same solver every real
track uses — via a throwaway zero-area `Track` at the click point,
`range_m=None` (so `AimSolver` substitutes `config.DEFAULT_RANGE_M`
exactly like it would for any real track with unknown range) and
`track_id=-1` (never stored, read only transiently by `solve()`). The
click still benefits from ballistic drop and boresight correction this
way, which a bare `projection.point_to_angles()` call would skip.

**Tuning window re-runs `detect(debug=True)` itself, throttled and
visibility-gated.** `PipelineWorker._tick_inner` calls `detect(frame)`
without `debug=True` on every tick, forever — building `DebugMasks`
(HSV/morphed masks, per-contour accept/reject detail) has a real cost
that would otherwise be paid on every frame whether or not anyone is
tuning. Instead the dialog itself re-runs detection on the same detector
against fresh snapshots, at 10Hz, and only while `self.isVisible()` —
closing or hiding it stops the extra work with no separate teardown
needed.

**Visual QA is a required step before declaring any GUI work done, not
an optional nice-to-have.** A passing test suite has never once been
enough on its own — across two separate rounds of actually launching
the app (usually offscreen, via `QT_QPA_PLATFORM=offscreen`) and reading
back real rendered frames, this caught bugs no test caught, because no
test rendered a real frame at real widget geometry in the first place:
- Round 1 (shell/canvas, before `left_panel.py`/`right_panel.py`
  existed): `StatusStrip`'s text labels started at `""` and only
  reached real width on the *second* `update_from_snapshot()`, visibly
  squishing the strip for one frame at startup; `VideoCanvas`'s
  crosshair inherited a stale `QBrush` left set by `_draw_label_near`
  and painted as a filled disc instead of an outline, hiding whatever
  track was underneath; `SafeOverlay`'s event log used
  `QPlainTextEdit`'s default (white) palette against the rest of the
  dark shell; the off-screen crosshair's direction arrow was drawn
  entirely outward from the pinned edge position, so wherever there was
  no letterbox margin it landed outside the widget and Qt clipped it
  away invisible; `AngleGauge` was positioned against the canvas
  widget's raw bottom edge while its own PAN/EĞİM label used the
  letterboxed image's bottom edge, leaving a gap between them whenever
  a letterbox margin existed. Also this round: `KALİBRE DEĞİL` clipped
  at the frame edge, and the SAFE screen mixing English chrome
  (`ACKNOWLEDGE`) with Turkish content — see Part 0's own a-j list, the
  reason `core/strings.py`'s `UI_LABEL_TR` exists at all.
- Round 2 (left/right panels, tuning window, operator input):
  `REASON_CODE_TR` was ASCII-transliterated throughout (`degil` for
  `değil`, etc.) since before the GUI existed — nothing had ever
  displayed it prominently enough to notice until the AI recommendation
  card did. The lock banner and the telemetry strip's own OPERATOR
  AKTİF/TAKİP YARDIMI ON row were positioned from two independent
  magic-number offsets and visibly overlapped once a real scene
  exercised both at once. `QGroupBox`/`QScrollArea`/`QComboBox`/
  `QLineEdit`/`QCheckBox` had no rules in `theme.py`'s shared stylesheet
  at all, since nothing before the tuning window ever used them — left
  unstyled, it was light-text-on-light-background, nearly unreadable. A
  `QLabel` showing a long fire-blocked reason centre-clipped illegibly
  from both ends instead of eliding.

None of the above shows up as a failing assertion; every one only shows
up as a wrong-looking pixel. Screenshots for the current state live in
`docs/screenshots/` — see the Architecture section.

**A queued Qt paint event can outlive the widget it was queued for.**
A widget that calls `update()` (schedules a deferred repaint) and is
then torn down before Qt's event loop gets around to delivering that
paint can segfault — `RuntimeError: wrapped C/C++ object ... has been
deleted` inside `paintEvent`, immediately followed by a hard crash, not
a catchable Python exception. Hit repeatedly writing `tuning_window`'s
own tests. Fix: `qtbot.wait(10)` (or `qtbot.waitExposed(widget)` right
after `.show()`) before a test that touched a shown, paintable widget
returns, flushing any pending repaint while the widget is still alive
rather than leaving it queued across teardown into the next test.

**`isHidden()`, never `isVisible()`, for anything not itself a top-level
window.** `QWidget.isVisible()` also depends on every ancestor's own
visibility, so it never reads `True` for a widget embedded inside a
`MainWindow` that is itself never `.show()`n — which every test in this
codebase does deliberately, to stay headless and fast. Bit
`_toggle_help_overlay` for real (toggling it twice took the same "show"
branch both times) before it was caught. `TuningWindow` is the one
exception that is actually fine to check with `isVisible()`: it is a
genuine top-level `QDialog`, so its own visibility is never gated by
`MainWindow`'s.

## `--dev` mode

Field-testing escape hatch (`python -m celikkubbe.ui.app --dev`), never
for the competition: the GUI otherwise cannot run past M1_INIT without a
real turret providing genuine homing and clean e-stop/driver-alarm
telemetry, which blocks testing the vision/detection pipeline alone.
`PipelineWorker(dev_mode=True)` boots directly into `M2_STANDBY` (the
self-test never runs at startup at all), and `modes.step()` takes a
`dev_mode` flag that relaxes exactly three checks — homing for M2→M3,
and e-stop/driver-alarm for tripping M4_SAFE from M2/M3 — deliberately
**not** link timeout or camera health, since those aren't about missing
hardware (`SimTurretLink`, the only link this GUI drives, simulates
e-stop/driver-alarm/homing fully in software; a genuinely dead link or
camera is a real problem `dev_mode` must not mask).

A second, independent mechanism covers the case where the system lands
back in `M1_INIT` anyway (a real fault trips M4_SAFE — link timeout,
camera health, or, outside `dev_mode`, e-stop/driver-alarm — and the
operator acknowledges it): a genuine self-test runs there and displays
real pass/fail rows, since against `_DeadLink`-style non-cooperative
links or no real turret it may never pass on its own. `SelfTestOverlay`
gained a **GEÇ** (skip) button, enabled only when `dev_mode` is on,
which calls `PipelineWorker.skip_self_test()` — `modes.step()` honours
`operator_skip_self_test` unconditionally on the next tick, moving
straight to `M2_STANDBY` regardless of `self_test_result`. The existing
**TEKRAR DENE** (retry) button is a different, always-available
mechanism with the opposite effect: it forces conclusion of the
*current* self-test attempt, which trips `M4_SAFE` if it's still
failing — GEÇ bypasses that outcome entirely, TEKRAR DENE does not.

Safety requirement, not a suggestion: `dev_mode` must be impossible to
miss. `MainWindow` shows a permanent red banner
(`GELİŞTİRME MODU — EMNİYET KONTROLLERİ DEVRE DIŞI`) between the title
bar and the splitter — a real widget, not a toast or overlay that could
be dismissed or time out — plus a `"DEV"` `StatusBadge` as the *first*
badge in the status strip, before even the mode badge. Both are
constructed once from `PipelineWorker.dev_mode` (a read-only property,
the single source of truth every dev_mode-aware widget reads instead of
each caller carrying its own copy of the flag) and never toggle at
runtime — `dev_mode` is fixed for the process's lifetime, set only from
the `--dev` CLI flag. `ui/app.py` also logs a warning at startup, and
`PipelineWorker.__init__` logs its own warning independently, so the
condition is visible in the log even if a caller constructs
`PipelineWorker` directly (as every test in this codebase does) without
going through `app.py` at all.

## Confirmed hardware facts

- MCU: **Nucleo-G431KB** (STM32G431).
- Pan motor: **JK-HSD86** stepper, 5:1 gear reduction.
- Tilt motor: **JK-HSD57** stepper, 2.5:1 belt reduction.
- PC↔MCU link: **RS422** via **MAX490** transceiver, 921600 baud (see
  Protocol section above for the corrected PB4/PA5 pin assignment).
- Firing trigger: **PC817** optocoupler, driven from **PA5** (not PB4 —
  see Protocol section), with a 10kΩ pull-down on the LED anode.
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

## Physical measurements still outstanding

All marked `TODO(measurement)` in code — never silently guessed:

- **Camera offset from the pan/tilt axis intersection**
  (`geometry.frames.TurretGeometry`: `cam_offset_x/y/z_m`, plus mounting
  `cam_roll/pitch/yaw_deg`, all currently 0 or a placeholder). Measurement
  procedure is in `TurretGeometry`'s own docstring: turret at pan=0,
  tilt=0, measure from the axis intersection to the camera's front
  element on all three axes. `DEFAULT_TURRET_GEOMETRY`'s 0.20m Y offset
  is chosen to reproduce this project's own worked parallax example
  (~2.3° at 5m, ~0.76° at 15m), not a measurement — almost certainly not
  the true geometry.
- **Muzzle velocity** (`geometry.ballistics.DEFAULT_BALLISTIC_TABLE`,
  currently 100 m/s) — needs a chronograph reading against the actual
  HPA setup.
- **Ballistic drop table** (`DEFAULT_BALLISTIC_TABLE.drop_deg` at 5/10/15m)
  — currently `gravity_only_drop_deg()` predictions, no hop-up backspin
  lift accounted for. Calibration procedure (fire at a stationary target
  at each measured slant range, measure the vertical offset,
  `atan(offset_m / slant_range_m)`) is in `ballistics.py`'s module
  docstring.
- **Boresight correction** (`geometry.calibration.BoresightCorrection`) —
  no measurements exist yet; `config/boresight.json` doesn't exist, so
  the system runs uncalibrated (`load_boresight()` returns an empty
  `BoresightTable`, zero correction everywhere) rather than failing to
  start.
- **Muzzle offset from the rotation centre**
  (`TurretGeometry.muzzle_offset_z_m`, currently 0.0) — affects slant
  range, currently assumed negligible.
- **HSV thresholds for both target colours**
  (`vision.l2_color._DEFAULT_CLASSES`: hostile's two hue ranges,
  friendly's one, both classes' `sat_min`/`val_min`) — hex-to-HSV
  conversion math applied to the competition spec's stated hex colours
  (`#F50A0A`/`#00A3E0`), never measured against the actual printed
  target models under real (indoor competition) lighting. Theory, not
  measurement, same distinction as everything else in this section —
  printed ink, ambient colour temperature and camera white balance can
  all shift measured hue/saturation well away from the hex-derived
  value. `ui/tuning_window.py` exists specifically to recalibrate these
  once the real models and venue are available; `config/hsv/` presets
  are gitignored, so a recalibrated value never silently ships as a
  new default without someone deliberately changing
  `_DEFAULT_CLASSES` itself.

## Open questions / not yet built

- `ui/gamepad.py` targets a Logitech F310 in XInput mode (the `xpad`
  kernel driver's own mapping: `BTN_SOUTH`/`BTN_EAST`, `ABS_X`/`ABS_Y`,
  `ABS_RZ`) — derived from the F310's published layout, never verified
  against a physical device, since none exists in this environment. If
  the pad is left in DirectInput mode ("D" on its back switch), or a
  different pad entirely is used, none of this is guaranteed to match.
  Device discovery and event parsing are tested against a hand-written
  fake `evdev` device instead — the same honestly-flagged gap
  `io/serial_link.py` already carries for real MCU firmware.
- `assets/fonts/` has no bundled `.ttf`/`.otf` yet —
  `theme.load_monospace_font()` falls back to an installed system
  monospace family (`JetBrains Mono`/`IBM Plex Mono`/`DejaVu Sans Mono`/
  `Consolas`), then the generic `"monospace"` family, with a logged
  warning either way. Layout assumes a genuinely monospaced font either
  way, but glyph metrics (and therefore exact pixel widths in
  screenshots) will shift once a real bundled font lands.
- `ui.tuning_window.TuningWindow`'s draggable ROI and HSV
  save/load/reset all work against `SimTurretLink`/`SyntheticSource`
  only — never exercised against a real D435i feed, where actual sensor
  noise, auto-exposure and white balance will make real presets look
  nothing like the synthetic ones committed today (`config/hsv/` is
  gitignored, so no presets ship with the repo regardless).
- `RealSenseSource` — no camera in hand yet.
- L1/YOLO detection layer — doesn't exist; `cls` is always `None`
  everywhere in the system today. `cascade.py`'s L1 health path is
  unexercised until this lands.
- `io.link_worker.LinkWorker` is built and unit-tested but not wired into
  `demos/pipeline.py` — the demo talks to `SimTurretLink` directly
  (`send_heartbeat()`/`poll()` from the demo loop itself), so
  `LinkWorker`'s own ACK-retransmit and staleness logic is only exercised
  by its own tests, not the integration demo.
- `io.serial_link.SerialTurretLink` has never been exercised against real
  MCU firmware — tests mock `pyserial.Serial` directly; no hardware yet.
- `Telemetry.mcu_mode` is decoded/tracked (real MCU and `SimTurretLink`
  both report it) but nothing yet uses it to detect PC/MCU mode
  disagreement, which was the stated reason for promoting it onto
  `Telemetry` in the first place.
- `geometry.calibration.refine_geometry()` (camera-to-turret least
  squares) is a real, working implementation, verified against synthetic
  data only — never run against a physical rig.
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
  flakiness. Exception noted under Hard constraints
  (`io/link_worker.py`'s background-thread idle sleep, never exercised by
  a test that also checks timing).
- Coverage gate: `fail_under = 90` in `pyproject.toml`; actual is
  ~94-96% (632 tests) across every package including `ui/`. The range,
  not a single number, is real: `theme.ToggleSwitch`'s
  `QPropertyAnimation` runs on wall-clock time, not `FakeClock`, so how
  many intermediate animation frames a given test run happens to paint
  — and therefore exactly which `theme.py` branches get hit — varies
  slightly with real system load. The test *count* is stable; only
  coverage of that one file's paint code moves.
- `ui/` tests use `pytest-qt`'s `qtbot` fixture (no `conftest.py` needed
  — it is a pytest plugin, not project fixture code) and drive
  `PipelineWorker` the same way `test_pipeline_worker.py` does:
  `tick()`/`worker._link_worker.tick()` called directly against a
  `FakeClock`, never `worker.start()`. `isHidden()`, not `isVisible()`,
  to check whether an overlay/panel-internal widget is shown — see ui/
  decisions and why. A widget that calls `update()` and is shown for
  real needs `qtbot.wait(...)`/`qtbot.waitExposed(...)` before the test
  returns, or a queued repaint can outlive it into the next test's own
  setup — also covered there, found via an actual segfault. `evdev` is
  monkeypatched at the module level (`evdev.list_devices`,
  `evdev.InputDevice`) for device discovery, and a hand-written fake
  device (`capabilities()`/`read_loop()`) for event parsing — no
  physical gamepad exists in this environment.
- ruff: line length 100, `target-version = "py310"`,
  `select = ["E", "F", "I", "UP", "B"]`. Run `ruff format` too — several
  past commits needed a follow-up formatting pass after edits shortened
  lines below the wrap threshold.
- `tests/factories.py` — shared `make_track`/`make_telemetry`/`make_state`
  builders with sane defaults, not collected by pytest (no `test_`
  prefix). Same pattern per-package: `tests/tracking/helpers.py`. `io/`
  and `geometry/` tests instead define small **local** per-file builders
  (`test_solver.py`'s `_make_track`, `test_serial_link.py`'s
  `_FakeSerial`/`_FakePortInfo`, `test_sim_link.py`'s heartbeat-pumping
  `_run` helper) rather than a shared `helpers.py` — each file's needs
  differed enough that sharing wasn't worth it. `tests/ui/` does both:
  imports `tests/factories.py`'s `make_track`/`make_state` for
  `core.types` objects, but every file that needs a full `UiSnapshot`
  (which `factories.py` cannot build — it is a `ui/`-only type) defines
  its own local `_snapshot()`/`_health()`/`_timings()`, each slightly
  different, rather than one shared `ui/`-specific factories module.
- Vision/tracking tests use `SyntheticSource` + NumPy-generated frames
  exclusively — nothing depends on a file that might not exist.
  `VideoFileSource`/`WebcamSource` tests mock `cv2.VideoCapture` directly;
  `test_serial_link.py`'s `_FakeSerial` mirrors the same
  hand-written-fake-over-`MagicMock` pattern for `pyserial.Serial`.
- `SimTurretLink` enforces its own 200ms watchdog for real — a test that
  advances a `FakeClock` by more than that without calling
  `send_heartbeat()` will trip it and freeze motion. Bit several tests
  during development; the fix is always to pump heartbeats in the same
  loop that advances the clock, not to disable the watchdog.
- Conventional Commits, one logical change per commit, commit body
  explains *why* (root cause / trade-off), not what the diff already
  shows.
