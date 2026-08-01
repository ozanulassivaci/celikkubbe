"""Headless perception-to-decision pipeline demo.

Wires FrameSource -> ColorDetector -> TrackManager -> priority scoring ->
modes.step()/engagement.step() together with a simulated TurretLink, with
no camera, no STM32 and no GUI. This is what proves the perception chain
and the state machines actually agree with each other before any UI
exists — not a product, an integration check with a status line.

Usage:
    python -m celikkubbe.demos.pipeline --source synthetic --targets 3
    python -m celikkubbe.demos.pipeline --source video --path clip.mp4
    python -m celikkubbe.demos.pipeline --source webcam --device 0

The aim solver here is a placeholder bearing-only conversion (pixel
offset from centre -> angle via intrinsics, no lead compensation, no
ballistics) — real aim solving is out of scope and arrives in a later
prompt. Likewise there is no L1/YOLO pipeline yet, so the cascade always
reports L2.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time

from celikkubbe.core import config as core_config
from celikkubbe.core import modes, priority
from celikkubbe.core.clock import SystemClock
from celikkubbe.core.commands import Arm, Command, Disarm, Fire, Goto, SoftEstop
from celikkubbe.core.engagement import step as engagement_step
from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import (
    IFF,
    CameraIntrinsics,
    EngagementState,
    HitResult,
    Layer,
    Mode,
    SelfTestItem,
    SelfTestResult,
    Stage,
    SystemState,
    Telemetry,
    Track,
    TrackStatus,
)
from celikkubbe.tracking.manager import TrackManager
from celikkubbe.vision.l2_color import ColorDetector
from celikkubbe.vision.sources import (
    SyntheticSource,
    SyntheticSourceConfig,
    VideoFileSource,
    WebcamSource,
)


def _step_towards(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


class SimulatedTurretLink:
    """A TurretLink with no hardware: moves towards commanded Goto angles
    at a fixed slew rate and reports plausible Telemetry. There is no
    vision-based hit verification in this demo, so a Fire command is
    resolved to a KILL one tick later, purely to exercise S6 -> S1.
    """

    def __init__(self, clock: Clock, max_vel_dps: float = 90.0) -> None:
        self._clock = clock
        self._max_vel_dps = max_vel_dps
        self._pan_deg = 0.0
        self._tilt_deg = 0.0
        self._target_pan_deg = 0.0
        self._target_tilt_deg = 0.0
        self._armed = False
        self._estop = False
        self._last_t = clock.now()
        self._pending_hit_result: HitResult | None = None

    @property
    def connected(self) -> bool:
        return True

    def send(self, cmd: Command) -> None:
        if isinstance(cmd, Goto):
            self._target_pan_deg = cmd.az_deg
            self._target_tilt_deg = cmd.el_deg
        elif isinstance(cmd, Arm):
            self._armed = True
        elif isinstance(cmd, Disarm):
            self._armed = False
        elif isinstance(cmd, SoftEstop):
            self._estop = True
        elif isinstance(cmd, Fire):
            self._pending_hit_result = HitResult.KILL

    def poll(self) -> Telemetry | None:
        now = self._clock.now()
        dt = max(0.0, now - self._last_t)
        self._last_t = now
        max_delta = self._max_vel_dps * dt
        self._pan_deg = _step_towards(self._pan_deg, self._target_pan_deg, max_delta)
        self._tilt_deg = _step_towards(self._tilt_deg, self._target_tilt_deg, max_delta)
        # No measured position exists to check a following error against
        # (see Telemetry.pan_deg/tilt_deg) — _step_towards snaps exactly to
        # the target once within reach, so "arrived" is an exact equality
        # check here, standing in for the MCU's own trajectory generator
        # report.
        motion_complete = self._pan_deg == self._target_pan_deg and (
            self._tilt_deg == self._target_tilt_deg
        )
        return Telemetry(
            t=now,
            pan_deg=self._pan_deg,
            tilt_deg=self._tilt_deg,
            pan_vel_dps=0.0,
            tilt_vel_dps=0.0,
            target_pan_deg=self._target_pan_deg,
            target_tilt_deg=self._target_tilt_deg,
            motion_complete=motion_complete,
            armed=self._armed,
            estop=self._estop,
            position_valid=not self._estop,
            driver_alarm_pan=False,
            driver_alarm_tilt=False,
            fan_rpm=(3000, 3000, 3000),
            mcu_temp_c=40.0,
            loop_time_us=500,
            crc_error_count=0,
        )

    def take_hit_result(self) -> HitResult | None:
        result, self._pending_hit_result = self._pending_hit_result, None
        return result


def stub_aim_solutions(
    tracks: list[Track], intrinsics: CameraIntrinsics
) -> dict[int, tuple[float, float]]:
    """Placeholder bearing-only aim solver: pixel offset from centre ->
    angle via intrinsics. No current-pose offset, no lead, no ballistics.
    """
    solutions: dict[int, tuple[float, float]] = {}
    for track in tracks:
        if track.status is not TrackStatus.CONFIRMED:
            continue
        cx = (track.bbox[0] + track.bbox[2]) / 2.0
        cy = (track.bbox[1] + track.bbox[3]) / 2.0
        dx_px = (cx - 0.5) * intrinsics.width
        dy_px = (cy - 0.5) * intrinsics.height
        az_deg = math.degrees(math.atan2(dx_px, intrinsics.fx))
        el_deg = math.degrees(math.atan2(dy_px, intrinsics.fy))
        solutions[track.track_id] = (az_deg, el_deg)
    return solutions


def _build_source(args: argparse.Namespace, clock: Clock):
    if args.source == "synthetic":
        # Stationary by default: a moving target combined with the
        # bearing-only stub_aim_solutions (no lead compensation) would sit
        # in S4_AIM forever chasing an angle tolerance a non-lead solver
        # can never satisfy against a mover — an honest result, but not a
        # useful demo default. L2 does now set IFF from colour, so unlike
        # the balloon-era version of this demo, a hostile-coloured target
        # can pass IFFGate; class stays None regardless (RangeGate is what
        # blocks Stage 3 on that), which is the real remaining gap an
        # L1/YOLO layer is needed to close.
        #
        # range_m=3.0 rather than the library default of 10.0: a 30cm
        # drone at 10m renders at ~8px wide at 640x480, right at
        # MIN_TARGET_PX, and the fill-ratio confidence at that scale reads
        # ~0.54 — genuinely below CONFIDENCE_THRESHOLD, an honest
        # resolution limit rather than a bug, but not a useful default for
        # watching the chain reach S5/S6.
        #
        # With more than one target, expect priority.py to sometimes lock
        # S3/S4 onto a FRIENDLY track: engagement.py has no mechanism to
        # give up on a *selected* target that is permanently gate-rejected
        # (unlike an exhausted MAX_ENGAGEMENT_ATTEMPTS, TARGET_FRIENDLY
        # never clears) and try another one instead. That is a real gap
        # worth fixing in engagement.py, not something this demo works
        # around — reason=TARGET_FRIENDLY staying on screen indefinitely
        # is exactly this demo doing its job.
        config = SyntheticSourceConfig(num_targets=args.targets, speed=0.0, range_m=3.0)
        return SyntheticSource(clock, config)
    if args.source == "video":
        return VideoFileSource(args.path, clock)
    if args.source == "webcam":
        return WebcamSource(args.device, clock)
    raise ValueError(f"unknown source: {args.source}")  # pragma: no cover — argparse guards this


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Celikkubbe headless perception + decision pipeline demo"
    )
    parser.add_argument("--source", choices=["synthetic", "video", "webcam"], default="synthetic")
    parser.add_argument("--targets", type=int, default=3, help="synthetic source target count")
    parser.add_argument("--path", type=str, default=None, help="video file path")
    parser.add_argument("--device", type=int, default=0, help="webcam device index")
    parser.add_argument("--stage", choices=["1", "2", "3"], default="2")
    parser.add_argument(
        "--frames", type=int, default=None, help="stop after N ticks (default: run until Ctrl+C)"
    )
    args = parser.parse_args(argv)
    if args.source == "video" and not args.path:
        parser.error("--source video requires --path")
    return args


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    stage = {"1": Stage.STAGE_1, "2": Stage.STAGE_2, "3": Stage.STAGE_3}[args.stage]

    clock = SystemClock()
    source = _build_source(args, clock)
    source.start()

    detector = ColorDetector()
    tracker = TrackManager(clock)
    turret = SimulatedTurretLink(clock)

    self_test = SelfTestResult(
        items=(
            SelfTestItem("camera", True, None, core_config.FPS_TARGET),
            SelfTestItem("stm32_link", True, None, None),
        )
    )

    state = SystemState(
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

    previous_order: list[int] = []
    frame_count = 0
    target_tick_s = 1.0 / core_config.FPS_TARGET
    last_loop_t: float | None = None

    try:
        while args.frames is None or frame_count < args.frames:
            tick_start = time.perf_counter()
            fps = 1.0 / (tick_start - last_loop_t) if last_loop_t is not None else 0.0
            last_loop_t = tick_start

            frame = source.read()
            if frame is None:
                time.sleep(0.001)
                continue

            detect_start = time.perf_counter()
            detections, _ = detector.detect(frame)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0

            track_start = time.perf_counter()
            tracks = tracker.update(detections, now=clock.now())
            track_ms = (time.perf_counter() - track_start) * 1000.0

            scored = [
                dataclasses.replace(
                    t, risk_score=priority.compute_risk_score(t.cls, t.range_m, t.confidence)
                )
                for t in tracks
            ]
            previous_order = priority.order_track_ids(scored, previous_order)

            telemetry = turret.poll()

            self_test_result = self_test if state.mode is Mode.M1_INIT else None
            operator_requested_mode = Mode.M3_OPERATIONAL if state.mode is Mode.M2_STANDBY else None
            next_mode, mode_commands = modes.step(
                state.mode,
                telemetry,
                self_test_result,
                camera_healthy=True,
                operator_requested_mode=operator_requested_mode,
                operator_ack_fault=False,
                now=clock.now(),
            )
            for cmd in mode_commands:
                turret.send(cmd)

            aim_solutions = stub_aim_solutions(scored, frame.intrinsics)
            hit_result = turret.take_hit_result()
            state = dataclasses.replace(state, mode=next_mode, tracks=tuple(scored))

            engagement_result = engagement_step(
                state=state,
                tracks=scored,
                telemetry=telemetry,
                hit_result=hit_result,
                operator=None,
                aim_solutions=aim_solutions,
                now=clock.now(),
            )
            for cmd in engagement_result.commands:
                turret.send(cmd)

            state = dataclasses.replace(
                state,
                engagement=engagement_result.engagement,
                selected_track_id=engagement_result.selected_track_id,
                attempts=engagement_result.attempts,
                deferred=engagement_result.deferred,
                gate_fail_since=engagement_result.gate_fail_since,
                commanded_pan_deg=engagement_result.commanded_pan_deg,
                commanded_tilt_deg=engagement_result.commanded_tilt_deg,
                telemetry=telemetry,
            )

            frame_count += 1
            tick_ms = (time.perf_counter() - tick_start) * 1000.0
            commands = mode_commands + engagement_result.commands
            commands_str = ",".join(type(cmd).__name__ for cmd in commands) or "-"
            reason = engagement_result.fallback_reason
            reason_str = reason.value if reason is not None else "-"
            n_hostile = sum(1 for t in scored if t.iff is IFF.HOSTILE)
            n_friendly = sum(1 for t in scored if t.iff is IFF.FRIENDLY)
            n_unknown = len(scored) - n_hostile - n_friendly

            print(
                f"[{frame_count:05d}] mode={state.mode.value:<14} "
                f"eng={state.engagement.value:<10} layer=L2 "
                f"tracks={len(scored)}(H:{n_hostile},F:{n_friendly},U:{n_unknown}) "
                f"sel={state.selected_track_id} "
                f"cmds={commands_str} reason={reason_str} fps={fps:6.1f} "
                f"detect={detect_ms:5.2f}ms track={track_ms:5.2f}ms tick={tick_ms:5.2f}ms"
            )

            elapsed = time.perf_counter() - tick_start
            if elapsed < target_tick_s:
                time.sleep(target_tick_s - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()


if __name__ == "__main__":
    run()
