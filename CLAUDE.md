# Çelikkubbe — project briefing

Dense reference for picking this codebase back up. Not user documentation —
assume familiarity with Python, control systems, and the code itself.

## Status

Layers built: `core/`, `vision/`, `tracking/`, `io/`, `geometry/`. All
committed, tested, pushed. 414 tests passing, 97.55% coverage (gate is
90% in `pyproject.toml`) across `core`/`vision`/`tracking`/`io`/`geometry`/
`demos`. Next up: the GUI layer — out of scope for every module built so
far by hard constraint (no Qt below it).

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
demos/      headless integration (no camera, no STM32, no GUI)
```

Non-package top-level dirs: `docs/protocol.md` (the spec above),
`config/` (calibration JSON, created on first save — empty and untracked
until then), `tools/` (dev-environment verification scripts, not part of
the installed package).

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
  `Telemetry.homed_pan`/`homed_tilt` — see io/ decisions.
- `core/engagement.py` — S1–S6 engagement machine, `StepResult`.
- `core/strings.py` — Turkish UI strings keyed by `ReasonCode` (now
  including `NOT_HOMED`). Nothing in decision logic imports this.
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

## Hard constraints

- No `PyQt6`/`PySide6` imports anywhere in `core/` or `io/`.
  `io/link_worker.py`'s own docstring states this explicitly — telemetry,
  events and command outcomes reach the owner through plain callbacks so
  it stays usable headless.
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
Lets a future GUI show the operator *why* the turret is pointing where it
is, and keeps debugging a bad solve tractable instead of a black box.
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

## Open questions / not yet built

- GUI — the actual next layer. Out of scope for every module built so far
  by hard constraint (no Qt in `core/` or `io/`); `geometry/`'s
  `AimSolution` reporting components separately was built specifically so
  a GUI can explain the pointing solution once it exists.
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
- Coverage gate: `fail_under = 90` in `pyproject.toml`; actual is 97.55%
  (414 tests) across `core`/`vision`/`tracking`/`io`/`geometry`/`demos`.
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
  differed enough that sharing wasn't worth it.
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
