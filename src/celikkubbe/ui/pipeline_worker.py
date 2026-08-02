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
from celikkubbe.core.commands import Command
from celikkubbe.core.engagement import step as engagement_step
from celikkubbe.core.health import CameraHealth, DepthHealth, DetectionHealth, InferenceHealth
from celikkubbe.core.protocols import Clock, FrameSource, TurretLink
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
from celikkubbe.io.codec import EventId, TelemetryFrame
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

    # --- GUI-thread-facing: thread-safe setters ---

    def set_operator_input(self, operator_input: OperatorInput) -> None:
        with self._operator_lock:
            self._operator_input = operator_input

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

        self_test_result = self._advance_self_test() if self._state.mode is Mode.M1_INIT else None
        ack_fault = self._ack_fault.is_set()
        if ack_fault:
            self._ack_fault.clear()

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

    def _advance_self_test(self) -> SelfTestResult | None:
        """Runs continuously while in M1_INIT; ``state.last_self_test`` is
        always refreshed with the latest attempt for the overlay to show
        live PASS/FAIL rows. Only handed to ``modes.step()`` -- which is
        what actually moves the mode on -- once every item passes, or the
        operator explicitly retries: an item still failing within an
        ordinary startup grace window must not trip M4_SAFE on its own,
        e.g. before a webcam has even had time to open.
        """
        camera_ok = self._camera_health.healthy and self._frames_seen > 0
        link_ok = not self._link_worker.link_lost
        result = SelfTestResult(
            items=(
                SelfTestItem(
                    "camera",
                    camera_ok,
                    None if camera_ok else "no camera signal",
                    self._camera_health.value if self._frames_seen >= 2 else None,
                ),
                SelfTestItem(
                    "stm32_link",
                    link_ok,
                    None if link_ok else "no telemetry",
                    self._link_worker.stats.round_trip_latency_ms,
                ),
            )
        )
        self._last_self_test_display = result

        retry = self._self_test_retry.is_set()
        if retry:
            self._self_test_retry.clear()
        if result.passed or retry:
            return result
        return None
