"""Shared construction helpers for core tests. Not collected by pytest."""

from __future__ import annotations

from celikkubbe.core.types import (
    IFF,
    EngagementState,
    Layer,
    McuMode,
    Mode,
    Stage,
    SystemState,
    TargetClass,
    Telemetry,
    Track,
    TrackStatus,
)


def make_track(
    track_id: int = 1,
    cls: TargetClass | None = TargetClass.UAV,
    confidence: float = 0.9,
    range_m: float | None = 5.0,
    range_source: str = "depth",
    iff: IFF = IFF.HOSTILE,
    status: TrackStatus = TrackStatus.CONFIRMED,
    risk_score: float = 50.0,
    cls_source: str | None = "model",
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
) -> Track:
    return Track(
        track_id=track_id,
        cls=cls,
        confidence=confidence,
        range_m=range_m,
        range_source=range_source,
        iff=iff,
        bbox=bbox,
        velocity=(0.0, 0.0),
        status=status,
        risk_score=risk_score,
        frames_confirmed=5,
        last_seen_t=0.0,
        cls_source=cls_source if cls is not None else None,
    )


def make_telemetry(
    t: float = 0.0,
    pan_deg: float = 0.0,
    tilt_deg: float = 0.0,
    target_pan_deg: float = 0.0,
    target_tilt_deg: float = 0.0,
    motion_complete: bool = True,
    armed: bool = True,
    estop: bool = False,
    position_valid: bool = True,
    driver_alarm_pan: bool = False,
    driver_alarm_tilt: bool = False,
    homed_pan: bool = True,
    homed_tilt: bool = True,
    mcu_mode: McuMode = McuMode.READY,
) -> Telemetry:
    return Telemetry(
        t=t,
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        pan_vel_dps=0.0,
        tilt_vel_dps=0.0,
        target_pan_deg=target_pan_deg,
        target_tilt_deg=target_tilt_deg,
        motion_complete=motion_complete,
        armed=armed,
        estop=estop,
        position_valid=position_valid,
        driver_alarm_pan=driver_alarm_pan,
        driver_alarm_tilt=driver_alarm_tilt,
        homed_pan=homed_pan,
        homed_tilt=homed_tilt,
        mcu_mode=mcu_mode,
        fan_rpm=(3000, 3000, 3000),
        mcu_temp_c=40.0,
        loop_time_us=500,
        crc_error_count=0,
    )


def make_state(
    stage: Stage = Stage.STAGE_2,
    mode: Mode = Mode.M3_OPERATIONAL,
    engagement: EngagementState = EngagementState.S1_SEARCH,
    active_layer: Layer = Layer.L1,
    layer_manual_override: bool = False,
    fallback_reason=None,
    tracks: tuple[Track, ...] | None = None,
    selected_track_id: int | None = None,
    attempts: dict[int, int] | None = None,
    deferred: dict[int, tuple] | None = None,
    gate_fail_since: float | None = None,
    commanded_pan_deg: float | None = None,
    commanded_tilt_deg: float | None = None,
    telemetry: Telemetry | None = None,
    last_self_test=None,
    class_overrides: dict[int, TargetClass] | None = None,
) -> SystemState:
    return SystemState(
        stage=stage,
        mode=mode,
        engagement=engagement,
        active_layer=active_layer,
        layer_manual_override=layer_manual_override,
        fallback_reason=fallback_reason,
        tracks=tracks if tracks is not None else (),
        selected_track_id=selected_track_id,
        attempts=attempts if attempts is not None else {},
        deferred=deferred if deferred is not None else {},
        gate_fail_since=gate_fail_since,
        commanded_pan_deg=commanded_pan_deg,
        commanded_tilt_deg=commanded_tilt_deg,
        telemetry=telemetry,
        last_self_test=last_self_test,
        class_overrides=class_overrides if class_overrides is not None else {},
    )
