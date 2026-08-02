"""PipelineWorker: the QThread owning perception, tracking and decision.

The GUI thread never blocks -- no serial reads, no camera reads, no
inference, no sleeps. This class is where all of that actually happens,
on its own thread, one tick at a time: read frame -> detect -> update
tracks -> score priorities -> solve aim for confirmed tracks -> step the
mode machine -> step the engagement machine -> send resulting commands
via LinkWorker -> build and emit a UiSnapshot.

Three threads exist in this application, no more: this one; the GUI
thread (widgets, painting, input -- nothing else); and LinkWorker's own
background thread (unchanged, started and owned here so its lifecycle
matches the pipeline's).

``tick()`` is a plain method, not hidden inside ``run()``'s loop, exactly
like ``io.link_worker.LinkWorker.tick()`` -- so a test can call it
directly, deterministically, against a ``FakeClock``, without ever
starting the real thread.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque

from PyQt6.QtCore import QThread, pyqtSignal

from celikkubbe.core import config, modes, priority
from celikkubbe.core.commands import Arm, Command, Disarm, Fire, Goto, Jog, SoftEstop, Stop, Zero
from celikkubbe.core.engagement import step as engagement_step
from celikkubbe.core.health import CameraHealth, DepthHealth, DetectionHealth, InferenceHealth
from celikkubbe.core.protocols import Clock, FrameSource, TurretLink
from celikkubbe.core.strings import SELF_TEST_DETAIL_TR
from celikkubbe.core.types import (
    Axis,
    EngagementState,
    HitResult,
    Layer,
    Mode,
    OperatorInput,
    ReasonCode,
    SelfTestItem,
    SelfTestResult,
    Stage,
    SystemState,
    Telemetry,
)
from celikkubbe.geometry.ballistics import DEFAULT_BALLISTIC_TABLE
from celikkubbe.geometry.calibration import DEFAULT_BORESIGHT_PATH, load_boresight
from celikkubbe.geometry.frames import DEFAULT_TURRET_GEOMETRY
from celikkubbe.geometry.projection import crosshair_with_indicator
from celikkubbe.geometry.solver import AimSolution, AimSolver
from celikkubbe.io.codec import AckResult, EventId, TelemetryFrame
from celikkubbe.io.link_worker import LinkWorker
from celikkubbe.tracking.manager import TrackManager
from celikkubbe.ui.snapshot import HealthSnapshot, PipelineTimings, UiSnapshot
from celikkubbe.vision.l2_color import ColorDetector
from celikkubbe.vision.sources import CameraIntrinsics

# CPU-yield only when a FrameSource has nothing new to offer (e.g. a
# VideoFileSource/WebcamSource paced source between frames) -- not a
# timing source. Every threshold this loop actually decides against
# (health windows, self-test, mode/engagement timeouts) compares against
# clock.now(), never against how long this sleep lasted. Same exception
# already documented for io/link_worker.py's background thread loop.
_IDLE_SLEEP_S = 0.001

# Sliding window for the SAFE overlay's event log -- old entries fall off
# the front rather than growing unbounded over a long session.
_EVENT_LOG_MAXLEN = 20

# Self-test (M1) thresholds. These gate M1 -> M2 (via modes.step()) but
# are computed entirely here, not core/config.py: core/modes.py only ever
# consumes the resulting SelfTestResult.passed, the same way io/'s own
# link-timing constants live in io/link_worker.py rather than
# core/config.py, since they are not a core/ decision threshold either.
# Deliberately stricter than the *ongoing* health monitors' thresholds
# (config.MIN_FPS=15, config.INFERENCE_FAIL_MS=100): a one-time bring-up
# check should demand more headroom than "still limping along" does.
_SELF_TEST_MIN_FPS = 30.0
_SELF_TEST_MAX_INFERENCE_MS = 30.0
_SELF_TEST_MOVE_TOLERANCE_DEG = 1.0
_SELF_TEST_MOVE_DELTA_DEG = 3.0  # small verification nudge, not an operational move
_SELF_TEST_MOVE_TIMEOUT_MS = 5000.0  # generous; a 3 deg nudge normally settles in well under 1s
# Float-precision slack on the FPS threshold: a source ticking at exactly
# FPS_TARGET's own period measures fractionally under 30.0 from ordinary
# float division error, not a real shortfall -- same idea as io/
# link_worker.py's _TIMING_EPSILON_MS on its own millisecond comparisons.
_SELF_TEST_FPS_EPSILON = 0.5


def _hit_result_from_ammo_delta(previous: int, current: int) -> HitResult | None:
    """Simulation convenience, not real hit verification: no vision-based
    hit confirmation exists anywhere in this codebase yet (see
    demos/pipeline.py's own docstring). Against SimTurretLink, a shot the
    sim actually accepted resolves to a KILL one tick later purely to
    exercise S6 -> S1; against real hardware (no ``ammo_fired`` attribute)
    this returns None forever and S6_ASSESS simply waits -- an honest gap,
    not a silently invented result.
    """
    return HitResult.KILL if current > previous else None


class PipelineWorker(QThread):
    snapshot_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        source: FrameSource,
        link: TurretLink,
        clock: Clock,
        stage: Stage = Stage.STAGE_2,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._link = link
        self._clock = clock
        self._event_log_lock = threading.Lock()
        self._event_log: deque[tuple[float, EventId, Axis | None]] = deque(maxlen=_EVENT_LOG_MAXLEN)
        self._link_worker = LinkWorker(link, clock, on_event=self._on_link_event)

        self._detector = ColorDetector()
        self._tracker = TrackManager(clock)
        self._aim_solver = AimSolver(
            DEFAULT_TURRET_GEOMETRY,
            DEFAULT_BALLISTIC_TABLE,
            load_boresight(DEFAULT_BORESIGHT_PATH),
        )

        self._inference_health = InferenceHealth()
        self._camera_health = CameraHealth(clock)
        self._detection_health = DetectionHealth()
        self._depth_health = DepthHealth()

        self._stop_event = threading.Event()
        self._self_test_retry = threading.Event()
        self._ack_fault = threading.Event()
        self._operator_lock = threading.Lock()
        self._operator_input = OperatorInput()
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot: UiSnapshot | None = None
        # Stage/layer changes mutate SystemState, which only this
        # worker's own thread may write (_tick_inner already does, every
        # tick, via dataclasses.replace()) -- a GUI-thread caller cannot
        # write self._state directly without racing that. Recorded here
        # instead and consumed once at the top of the next _tick_inner,
        # the same deferred-request pattern set_operator_input already
        # uses for OperatorInput. Commands (estop/zero/arm/jog/stop) need
        # no such deferral: LinkWorker.send() is already thread-safe on
        # its own, so those go straight through from the GUI thread.
        self._control_lock = threading.Lock()
        self._requested_stage: Stage | None = None
        self._requested_layer: Layer | None = None

        self._state = SystemState(
            stage=stage,
            mode=Mode.M1_INIT,
            engagement=EngagementState.S1_SEARCH,
            active_layer=Layer.L2,
            layer_manual_override=False,
            fallback_reason=None,
            tracks=(),
            selected_track_id=None,
            attempts={},
            deferred={},
            gate_fail_since=None,
            commanded_pan_deg=None,
            commanded_tilt_deg=None,
            telemetry=None,
            last_self_test=None,
        )
        self._previous_order: list[int] = []
        self._frames_seen = 0
        self._last_ammo_fired = 0
        self._last_tick_start: float | None = None
        self._last_telemetry_t: float | None = None
        self._last_self_test_display: SelfTestResult | None = None

        # Self-test (M1) active-check state -- see _reset_self_test_active_checks.
        self._pan_tilt_test_state = "idle"  # "idle" -> "moving" -> "done"
        self._pan_tilt_test_target: tuple[float, float] | None = None
        self._pan_tilt_test_started_t: float | None = None
        self._pan_tilt_test_passed: bool | None = None
        self._pan_tilt_test_error_deg: float | None = None
        self._fire_lock_test_passed: bool | None = None
        self._fire_lock_test_detail: str | None = None

    # --- GUI-thread-facing: thread-safe setters ---

    def set_operator_input(self, operator_input: OperatorInput) -> None:
        with self._operator_lock:
            self._operator_input = operator_input

    def request_estop(self) -> None:
        """Sent directly on LinkWorker, not queued behind this worker's
        own tick loop -- an emergency stop must reach the link the
        instant the operator presses it, not wait for the next
        _tick_inner() pass to get around to calling _send().
        """
        self._link_worker.send(SoftEstop())

    def request_arm(self, armed: bool) -> None:
        self._link_worker.send(Arm() if armed else Disarm())

    def request_zero(self, axis: Axis) -> None:
        self._link_worker.send(Zero(axis, 0.0))

    def request_jog(self, axis: Axis, direction: int, speed_dps: float) -> None:
        self._link_worker.send(Jog(axis, direction, speed_dps))

    def request_stop(self) -> None:
        self._link_worker.send(Stop())

    def set_stage(self, stage: Stage) -> None:
        with self._control_lock:
            self._requested_stage = stage

    def select_layer(self, layer: Layer) -> None:
        with self._control_lock:
            self._requested_layer = layer

    def retry_self_test(self) -> None:
        self._self_test_retry.set()

    def acknowledge_fault(self) -> None:
        self._ack_fault.set()

    @property
    def latest_snapshot(self) -> UiSnapshot | None:
        """Single-slot buffer, not a queue: a GUI that falls behind always
        reads whichever snapshot is newest at the moment it looks, rather
        than working through a backlog of every tick that happened while
        it was busy.
        """
        with self._snapshot_lock:
            return self._latest_snapshot

    @property
    def detector(self) -> ColorDetector:
        """The live L2 detector -- exposed so the HSV tuning window can
        read and live-apply ``ColorDetectorConfig`` changes. Unsynchronised:
        ``_tick_inner`` reads ``detector.config`` every tick on this
        worker's own thread with no lock, matching every other read of
        ``self._detector`` in this class, so a caller must replace the
        whole config object (``detector.config = new_config``) rather
        than mutating individual fields in place -- a single reference
        swap is atomic under the GIL, a field-by-field mutation is not.
        """
        return self._detector

    @property
    def aim_solver(self) -> AimSolver:
        """The same solver instance _tick_inner uses for every confirmed
        track, exposed so a manual click-to-aim (Stage 1, clicking empty
        canvas) solves through identical geometry/ballistics/boresight
        calibration rather than a second AimSolver silently drifting from
        whatever this one has loaded.
        """
        return self._aim_solver

    def request_goto(self, az_deg: float, el_deg: float) -> None:
        """Direct Goto, bypassing engagement.py entirely -- for Stage 1's
        click-to-aim on empty canvas, which is not a tracked target and
        so has no selected_track_id for the engagement FSM to solve for.
        """
        goto = Goto(az_deg, el_deg, config.AIM_MAX_VEL_DPS, config.AIM_MAX_ACCEL_DPS2)
        self._link_worker.send(goto)

    # --- LinkWorker callback: runs on LinkWorker's own background
    # thread, not this one, hence its own lock rather than reusing
    # _snapshot_lock or _operator_lock ---

    def _on_link_event(self, event: tuple[EventId, Axis | None, float]) -> None:
        event_id, axis, _mcu_t = event
        with self._event_log_lock:
            self._event_log.append((self._clock.now(), event_id, axis))

    # --- thread lifecycle ---

    def stop(self) -> None:
        """Sets the stop flag; the caller still joins with ``wait()``."""
        self._stop_event.set()

    def run(self) -> None:
        try:
            self._source.start()
        except Exception as exc:  # camera/file failing to open at all
            self.error.emit(f"failed to start source: {exc}")
            return
        self._link_worker.start()
        try:
            while not self._stop_event.is_set():
                self.tick()
        finally:
            self._link_worker.stop()
            self._source.stop()

    # --- the work itself ---

    def tick(self) -> None:
        """One full pass. Never raises: a bad frame must not kill the
        pipeline, so every exception is caught here, reported through
        ``error``, and the loop keeps running on the next tick.
        """
        try:
            self._tick_inner()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _tick_inner(self) -> None:
        tick_start = self._clock.now()
        fps = (
            1.0 / (tick_start - self._last_tick_start)
            if self._last_tick_start is not None and tick_start > self._last_tick_start
            else 0.0
        )
        self._last_tick_start = tick_start

        capture_start = self._clock.now()
        frame = self._source.read()
        capture_ms = (self._clock.now() - capture_start) * 1000.0
        if frame is None:
            time.sleep(_IDLE_SLEEP_S)
            return
        self._frames_seen += 1
        self._camera_health.record_frame(frame.t)

        detect_start = self._clock.now()
        detections, _ = self._detector.detect(frame)
        detect_ms = (self._clock.now() - detect_start) * 1000.0
        self._inference_health.record(detect_ms)
        self._detection_health.record(max((d.confidence for d in detections), default=None))
        self._depth_health.record(frame.has_depth)

        track_start = self._clock.now()
        tracks = self._tracker.update(detections, now=self._clock.now())
        track_ms = (self._clock.now() - track_start) * 1000.0

        scored = [
            dataclasses.replace(
                t, risk_score=priority.compute_risk_score(t.cls, t.range_m, t.confidence)
            )
            for t in tracks
        ]
        self._previous_order = priority.order_track_ids(scored, self._previous_order)

        telemetry = self._link_worker.latest_telemetry
        telemetry_frame: TelemetryFrame | None = getattr(self._link, "last_telemetry_frame", None)
        if telemetry is not None:
            self._last_telemetry_t = self._clock.now()

        solve_start = self._clock.now()
        solved: dict[int, AimSolution] = {}
        for t in scored:
            solution = self._aim_solver.solve(t, frame.intrinsics)
            if solution is not None:
                solved[t.track_id] = solution
        aim_solutions = {tid: (s.az_deg, s.el_deg) for tid, s in solved.items()}
        solve_ms = (self._clock.now() - solve_start) * 1000.0

        step_start = self._clock.now()
        with self._operator_lock:
            operator = self._operator_input
        self._apply_requested_stage_and_layer()

        ack_fault = self._ack_fault.is_set()
        if ack_fault:
            self._ack_fault.clear()
            if self._state.mode is Mode.M4_SAFE:
                # About to re-enter M1_INIT for a fresh self-test -- the
                # active checks (pan/tilt move, fire lock) must run again,
                # not report a stale result from the previous attempt.
                self._reset_self_test_active_checks()

        self_test_result = (
            self._advance_self_test(telemetry) if self._state.mode is Mode.M1_INIT else None
        )

        next_mode, mode_commands = modes.step(
            self._state.mode,
            telemetry,
            self_test_result,
            camera_healthy=self._camera_health.healthy,
            operator_requested_mode=None,  # mode-request control lands with the right panel
            operator_ack_fault=ack_fault,
            now=self._clock.now(),
        )
        self._send(mode_commands)

        ammo_fired = getattr(self._link, "ammo_fired", None)
        hit_result = None
        if ammo_fired is not None:
            hit_result = _hit_result_from_ammo_delta(self._last_ammo_fired, ammo_fired)
            self._last_ammo_fired = ammo_fired

        self._state = dataclasses.replace(
            self._state,
            mode=next_mode,
            tracks=tuple(scored),
            last_self_test=self._last_self_test_display,
        )
        engagement_fallback_reason: ReasonCode | None = None
        if next_mode is Mode.M1_INIT:
            # Self-test is still running: never let engagement's own aim
            # commands compete with the self-test's own pan/tilt
            # verification move (or move the turret at all before self-
            # test has confirmed the axes respond correctly). Detection
            # and tracking still run either way, for display -- only
            # engagement's own state advancement and commands pause.
            self._state = dataclasses.replace(self._state, telemetry=telemetry)
        else:
            engagement_result = engagement_step(
                state=self._state,
                tracks=scored,
                telemetry=telemetry,
                hit_result=hit_result,
                operator=operator,
                aim_solutions=aim_solutions,
                now=self._clock.now(),
            )
            self._send(engagement_result.commands)
            self._state = dataclasses.replace(
                self._state,
                engagement=engagement_result.engagement,
                selected_track_id=engagement_result.selected_track_id,
                attempts=engagement_result.attempts,
                deferred=engagement_result.deferred,
                gate_fail_since=engagement_result.gate_fail_since,
                commanded_pan_deg=engagement_result.commanded_pan_deg,
                commanded_tilt_deg=engagement_result.commanded_tilt_deg,
                telemetry=telemetry,
            )
            engagement_fallback_reason = engagement_result.fallback_reason
        step_ms = (self._clock.now() - step_start) * 1000.0

        crosshair_px, crosshair_offscreen, crosshair_bearing_deg = self._solve_crosshair(
            telemetry, solved, frame.intrinsics
        )

        tick_ms = (self._clock.now() - tick_start) * 1000.0
        timings = PipelineTimings(
            capture_ms=capture_ms,
            detect_ms=detect_ms,
            track_ms=track_ms,
            solve_ms=solve_ms,
            step_ms=step_ms,
            tick_ms=tick_ms,
            fps=fps,
        )
        health = HealthSnapshot(
            inference_healthy=self._inference_health.healthy,
            inference_ms=self._inference_health.value,
            camera_healthy=self._camera_health.healthy,
            camera_fps=self._camera_health.value if self._frames_seen >= 2 else 0.0,
            link_healthy=not self._link_worker.link_lost,
            link_stale_ms=(
                (self._clock.now() - self._last_telemetry_t) * 1000.0
                if self._last_telemetry_t is not None
                else float("inf")
            ),
            crc_error_count=self._link_worker.stats.crc_error_count,
            detection_healthy=self._detection_health.healthy,
            detection_confidence=self._detection_health.value,
            depth_healthy=self._depth_health.healthy,
            depth_valid_ratio=self._depth_health.value,
            active_layer=self._state.active_layer,
            fallback_reason=self._state.fallback_reason,
            l1_recovery_countdown_s=None,
        )

        with self._event_log_lock:
            event_log = tuple(self._event_log)

        snapshot = UiSnapshot(
            t=self._clock.now(),
            frame=frame,
            detections=tuple(detections),
            tracks=tuple(scored),
            ordered_track_ids=tuple(self._previous_order),
            state=self._state,
            telemetry=telemetry,
            telemetry_frame=telemetry_frame,
            aim=solved.get(self._state.selected_track_id),
            crosshair_px=crosshair_px,
            crosshair_offscreen=crosshair_offscreen,
            crosshair_bearing_deg=crosshair_bearing_deg,
            health=health,
            timings=timings,
            fault_reason=self._derive_fault_reason(telemetry),
            event_log=event_log,
            ammo_fired=ammo_fired,
            engagement_fallback_reason=engagement_fallback_reason,
        )
        with self._snapshot_lock:
            self._latest_snapshot = snapshot
        self.snapshot_ready.emit(snapshot)

        elapsed = self._clock.now() - tick_start
        target_tick_s = 1.0 / config.FPS_TARGET
        if elapsed < target_tick_s:
            time.sleep(target_tick_s - elapsed)

    def _send(self, commands: list[Command]) -> None:
        for cmd in commands:
            self._link_worker.send(cmd)

    def _apply_requested_stage_and_layer(self) -> None:
        """Consumes set_stage()/select_layer()'s pending requests, if any,
        exactly once per tick, on this worker's own thread -- the only
        thread allowed to write self._state.

        Layer.L3 (Tam Manuel) is a Stage 1-only override (see
        core/cascade.py's own docstring and select_manual()'s guard): if
        the operator leaves Stage 1 while it is active, it is cleared
        back to L2 here rather than leaving an otherwise-unreachable
        (stage, layer) combination sitting in SystemState. The right
        panel itself cannot cause this on its own -- L3's mode card is
        only ever rendered while Stage 1 is selected -- but a stage
        change and a stale prior layer choice can still combine into it.
        """
        with self._control_lock:
            requested_stage = self._requested_stage
            self._requested_stage = None
            requested_layer = self._requested_layer
            self._requested_layer = None

        if requested_stage is None and requested_layer is None:
            return

        new_stage = requested_stage if requested_stage is not None else self._state.stage
        new_layer = requested_layer if requested_layer is not None else self._state.active_layer
        new_override = self._state.layer_manual_override or requested_layer is not None
        if new_stage is not Stage.STAGE_1 and new_layer is Layer.L3:
            new_layer = Layer.L2
            new_override = True

        self._state = dataclasses.replace(
            self._state,
            stage=new_stage,
            active_layer=new_layer,
            layer_manual_override=new_override,
        )

    def _solve_crosshair(
        self,
        telemetry: Telemetry | None,
        solved: dict[int, AimSolution],
        intr: CameraIntrinsics,
    ) -> tuple[tuple[float, float] | None, bool, float | None]:
        """Where the barrel currently points, projected into this frame.

        Uses telemetry's real-time pan/tilt (what the MCU is actually at
        right now, mid-slew or not) rather than the FSM's last commanded
        setpoint, and the selected target's solved range for parallax
        when one exists -- config.DEFAULT_RANGE_M otherwise, the same
        assumption AimSolver itself substitutes when nothing is locked.
        """
        if telemetry is None:
            return None, False, None
        selected = solved.get(self._state.selected_track_id)
        range_m = selected.range_m if selected is not None else config.DEFAULT_RANGE_M
        pixel, offscreen, bearing_deg = crosshair_with_indicator(
            telemetry.pan_deg, telemetry.tilt_deg, range_m, intr, DEFAULT_TURRET_GEOMETRY
        )
        return pixel, offscreen, bearing_deg

    def _derive_fault_reason(self, telemetry: Telemetry | None) -> ReasonCode | None:
        """Explains *why* the system is in M4_SAFE, for the SAFE overlay.

        modes.step() itself returns only the next Mode and Commands, no
        reason -- it is a decision, not a diagnostic, and adding one would
        change a stable, tested core/ contract for a GUI-only display
        need. This mirrors modes.step()'s own condition order (estop,
        then link timeout, then driver alarm, then camera health) using
        the same public signals it checks, without duplicating its
        decision of whether to actually trip SAFE -- only explaining it
        after the fact. A self-test failure is deliberately not covered
        here: state.last_self_test already carries which item failed and
        why, which the overlay shows directly instead of forcing that
        detail through a ReasonCode that does not exist for it.
        """
        if self._state.mode is not Mode.M4_SAFE:
            return None
        if self._state.last_self_test is not None and not self._state.last_self_test.passed:
            return None
        if telemetry is not None and telemetry.estop:
            return ReasonCode.ESTOP_ACTIVE
        if self._link_worker.link_lost:
            return ReasonCode.LINK_TIMEOUT
        if telemetry is not None and (telemetry.driver_alarm_pan or telemetry.driver_alarm_tilt):
            return ReasonCode.DRIVER_ALARM
        if not self._camera_health.healthy:
            return ReasonCode.CAMERA_TIMEOUT
        return None

    # --- self-test (M1) ---

    def _reset_self_test_active_checks(self) -> None:
        """Called when re-entering M1_INIT (a fresh acknowledged fault, or
        construction): the multi-tick active checks must run again from
        scratch, not report a stale result left over from a previous
        attempt.
        """
        self._pan_tilt_test_state = "idle"
        self._pan_tilt_test_target = None
        self._pan_tilt_test_started_t = None
        self._pan_tilt_test_passed = None
        self._pan_tilt_test_error_deg = None
        self._fire_lock_test_passed = None
        self._fire_lock_test_detail = None

    def _advance_self_test(self, telemetry: Telemetry | None) -> SelfTestResult | None:
        """Six items, matching docs/protocol.md's startup sequence and the
        mode machine's own entry requirements: camera FPS, link response,
        a real pan/tilt verification move, the firing mechanism's fail-
        safe default, inference latency, and driver alarms. Runs
        continuously while in M1_INIT; ``state.last_self_test`` is always
        refreshed with the latest attempt for the overlay to show live
        rows. Only handed to ``modes.step()`` -- which is what actually
        moves the mode on -- once every item passes, or the operator
        explicitly retries: an item still failing within an ordinary
        startup grace window must not trip M4_SAFE on its own, e.g.
        before a webcam or the pan/tilt verification move has finished.
        """
        camera_ok, camera_measured, camera_detail = self._self_test_camera()
        link_ok, link_measured, link_detail = self._self_test_link()
        move_ok, move_measured, move_detail = self._self_test_pan_tilt_move(telemetry)
        fire_ok, fire_detail = self._self_test_fire_lock()
        inference_ok, inference_measured, inference_detail = self._self_test_inference()
        alarms_ok, alarms_detail = self._self_test_driver_alarms(telemetry)

        result = SelfTestResult(
            items=(
                SelfTestItem("camera", camera_ok, camera_detail, camera_measured),
                SelfTestItem("stm32_link", link_ok, link_detail, link_measured),
                SelfTestItem("pan_tilt_move", move_ok, move_detail, move_measured),
                SelfTestItem("fire_lock", fire_ok, fire_detail, None),
                SelfTestItem("inference_time", inference_ok, inference_detail, inference_measured),
                SelfTestItem("driver_alarms", alarms_ok, alarms_detail, None),
            )
        )
        self._last_self_test_display = result

        retry = self._self_test_retry.is_set()
        if retry:
            self._self_test_retry.clear()
        if result.passed or retry:
            return result
        return None

    def _self_test_camera(self) -> tuple[bool, float | None, str | None]:
        if self._frames_seen < 2:
            return False, None, SELF_TEST_DETAIL_TR["no_camera_signal"]
        measured = self._camera_health.value
        passed = measured >= _SELF_TEST_MIN_FPS - _SELF_TEST_FPS_EPSILON
        detail = None if passed else SELF_TEST_DETAIL_TR["fps_too_low"]
        return passed, measured, detail

    def _self_test_link(self) -> tuple[bool, float | None, str | None]:
        link_ok = not self._link_worker.link_lost
        measured = self._link_worker.stats.round_trip_latency_ms
        detail = None if link_ok else SELF_TEST_DETAIL_TR["no_telemetry"]
        return link_ok, measured, detail

    def _self_test_pan_tilt_move(
        self, telemetry: Telemetry | None
    ) -> tuple[bool, float | None, str | None]:
        """A small verification nudge, not an operational move: proves the
        commanded axes actually respond and settle within
        _SELF_TEST_MOVE_TOLERANCE_DEG of where they were told to go,
        using the same Goto path an operational aim would -- catching a
        disconnected or miswired axis before the competition run does,
        not just before a target is ever selected.

        Completion is gated on the *measured position* reaching
        tolerance, not on ``motion_complete`` alone: a trapezoidal
        profile starts from rest, so velocity -- and therefore
        ``motion_complete``, which SimTurretLink computes from velocity
        -- legitimately reads "complete" for the single instant the move
        is issued, before the axis has gone anywhere. See
        SimTurretLink._compute_mcu_mode's own docstring for the same
        quirk affecting a different field. A generous timeout still
        exists so a genuinely stuck axis eventually reports FAIL with
        its real error instead of waiting forever.
        """
        if telemetry is None:
            return False, None, SELF_TEST_DETAIL_TR["no_telemetry"]

        if self._pan_tilt_test_state == "idle":
            pan_lo, pan_hi = config.PAN_LIMIT_DEG
            tilt_lo, tilt_hi = config.TILT_LIMIT_DEG
            target_pan = _pick_verification_target(
                telemetry.pan_deg, _SELF_TEST_MOVE_DELTA_DEG, pan_lo, pan_hi
            )
            target_tilt = _pick_verification_target(
                telemetry.tilt_deg, _SELF_TEST_MOVE_DELTA_DEG, tilt_lo, tilt_hi
            )
            self._pan_tilt_test_target = (target_pan, target_tilt)
            self._pan_tilt_test_started_t = self._clock.now()
            self._link_worker.send(
                Goto(target_pan, target_tilt, config.AIM_MAX_VEL_DPS, config.AIM_MAX_ACCEL_DPS2)
            )
            self._pan_tilt_test_state = "moving"
            return False, None, SELF_TEST_DETAIL_TR["in_progress"]

        if self._pan_tilt_test_state == "moving":
            assert self._pan_tilt_test_target is not None
            assert self._pan_tilt_test_started_t is not None
            target_pan, target_tilt = self._pan_tilt_test_target
            error = max(abs(telemetry.pan_deg - target_pan), abs(telemetry.tilt_deg - target_tilt))
            within_tolerance = error <= _SELF_TEST_MOVE_TOLERANCE_DEG
            elapsed_ms = (self._clock.now() - self._pan_tilt_test_started_t) * 1000.0
            timed_out = elapsed_ms >= _SELF_TEST_MOVE_TIMEOUT_MS
            if not (within_tolerance or timed_out):
                return False, None, SELF_TEST_DETAIL_TR["in_progress"]
            self._pan_tilt_test_passed = within_tolerance
            self._pan_tilt_test_error_deg = error
            self._pan_tilt_test_state = "done"

        passed = bool(self._pan_tilt_test_passed)
        error = self._pan_tilt_test_error_deg
        detail = None if passed else SELF_TEST_DETAIL_TR["move_error"].format(error=error)
        return passed, error, detail

    def _self_test_fire_lock(self) -> tuple[bool, str | None]:
        """Confirms the fail-safe default: attempting to fire while
        disarmed (which is always true here -- nothing arms anything
        before M3) must be rejected, not accepted. Sent directly on the
        raw link, bypassing LinkWorker's queue: SimTurretLink's send()
        dispatches synchronously, so the result is available to read
        back the same tick -- no multi-tick wait needed, unlike the
        pan/tilt move above. Runs once and caches its result: repeatedly
        firing a test shot every tick while M1_INIT persists would be
        wasteful and, against real hardware, audible.
        """
        if self._fire_lock_test_passed is not None:
            return self._fire_lock_test_passed, self._fire_lock_test_detail

        self._link.send(Fire(count=1))
        last_ack = getattr(self._link, "last_ack", None)
        if last_ack is None:
            passed, detail = False, SELF_TEST_DETAIL_TR["cannot_verify"]
        else:
            passed = last_ack is not AckResult.OK
            detail = None if passed else SELF_TEST_DETAIL_TR["fire_not_rejected"]
        self._fire_lock_test_passed = passed
        self._fire_lock_test_detail = detail
        return passed, detail

    def _self_test_inference(self) -> tuple[bool, float | None, str | None]:
        if self._frames_seen < 1:
            return False, None, SELF_TEST_DETAIL_TR["no_camera_signal"]
        measured = self._inference_health.value
        passed = measured <= _SELF_TEST_MAX_INFERENCE_MS
        detail = None if passed else SELF_TEST_DETAIL_TR["inference_slow"]
        return passed, measured, detail

    def _self_test_driver_alarms(self, telemetry: Telemetry | None) -> tuple[bool, str | None]:
        if telemetry is None:
            return False, SELF_TEST_DETAIL_TR["no_telemetry"]
        if telemetry.driver_alarm_pan:
            return False, SELF_TEST_DETAIL_TR["driver_alarm_pan"]
        if telemetry.driver_alarm_tilt:
            return False, SELF_TEST_DETAIL_TR["driver_alarm_tilt"]
        return True, None


def _pick_verification_target(current_deg: float, delta_deg: float, lo: float, hi: float) -> float:
    """A small nudge in whichever direction stays inside the software
    limits -- the self-test move must never itself trip LimitGate.
    """
    candidate = current_deg + delta_deg
    if candidate > hi:
        candidate = current_deg - delta_deg
    return min(hi, max(lo, candidate))
