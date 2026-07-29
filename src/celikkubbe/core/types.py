"""Enums and data contracts shared across the core decision layer.

No internal dependencies: every other module in ``core`` builds on top of
this one, never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class Stage(Enum):
    STAGE_1 = "STAGE_1"
    STAGE_2 = "STAGE_2"
    STAGE_3 = "STAGE_3"


class Mode(Enum):
    M1_INIT = "M1_INIT"
    M2_STANDBY = "M2_STANDBY"
    M3_OPERATIONAL = "M3_OPERATIONAL"
    M4_SAFE = "M4_SAFE"


class EngagementState(Enum):
    S1_SEARCH = "S1_SEARCH"
    S2_ACQUIRE = "S2_ACQUIRE"
    S3_TRACK = "S3_TRACK"
    S4_AIM = "S4_AIM"
    S5_ENGAGE = "S5_ENGAGE"
    S6_ASSESS = "S6_ASSESS"


class Layer(Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class IFF(Enum):
    FRIENDLY = "FRIENDLY"
    HOSTILE = "HOSTILE"
    UNKNOWN = "UNKNOWN"


class TargetClass(Enum):
    F16 = "F16"
    HELICOPTER = "HELICOPTER"
    MISSILE = "MISSILE"
    UAV = "UAV"
    BALLOON = "BALLOON"
    UNKNOWN = "UNKNOWN"


class TrackStatus(Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"
    LOST = "LOST"


class Axis(Enum):
    PAN = "PAN"
    TILT = "TILT"


class HitResult(Enum):
    KILL = "KILL"
    MISS = "MISS"
    UNVERIFIED = "UNVERIFIED"


class ReasonCode(Enum):
    """Why a gate rejected a target, or why the cascade fell back.

    Kept as codes, never human-readable strings, so language stays out of
    the decision logic. See ``strings.py`` for the Turkish UI mapping.
    """

    NOT_OPERATIONAL = "NOT_OPERATIONAL"
    NOT_ARMED = "NOT_ARMED"
    ESTOP_ACTIVE = "ESTOP_ACTIVE"
    POSITION_INVALID = "POSITION_INVALID"
    TARGET_FRIENDLY = "TARGET_FRIENDLY"
    CLASS_UNKNOWN = "CLASS_UNKNOWN"
    RANGE_UNKNOWN = "RANGE_UNKNOWN"
    RANGE_OUT_OF_BOUNDS = "RANGE_OUT_OF_BOUNDS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ANGLE_NOT_SETTLED = "ANGLE_NOT_SETTLED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INFERENCE_SLOW = "INFERENCE_SLOW"
    CAMERA_TIMEOUT = "CAMERA_TIMEOUT"
    LINK_TIMEOUT = "LINK_TIMEOUT"
    DEPTH_UNRELIABLE = "DEPTH_UNRELIABLE"
    OPERATOR_OVERRIDE = "OPERATOR_OVERRIDE"


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    quality: Literal["factory", "calibrated", "estimated"]

    @property
    def is_reliable(self) -> bool:
        """True when intrinsics are trustworthy enough to compute the crosshair.

        Factory and post-calibration values come from a measured optical
        model; "estimated" values are guessed defaults (e.g. for a webcam
        with no calibration) and must not be trusted to place a crosshair.
        """
        return self.quality in ("factory", "calibrated")


@dataclass(frozen=True)
class Frame:
    image: Any
    t: float
    intrinsics: CameraIntrinsics
    has_depth: bool
    depth: Any | None = None


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    cls: TargetClass | None
    confidence: float
    range_m: float | None
    source_layer: Layer
    iff: IFF


@dataclass
class Track:
    track_id: int
    cls: TargetClass | None
    confidence: float
    range_m: float | None
    iff: IFF
    bbox: tuple[float, float, float, float]
    velocity: tuple[float, float]
    status: TrackStatus
    risk_score: float
    frames_confirmed: int
    last_seen_t: float
    engagement_attempts: int


@dataclass(frozen=True)
class Telemetry:
    t: float
    pan_deg: float
    tilt_deg: float
    pan_vel_dps: float
    tilt_vel_dps: float
    target_pan_deg: float
    target_tilt_deg: float
    following_error_deg: float
    armed: bool
    estop: bool
    # False after an e-stop: the 48V rail is cut, unpowered steppers only
    # have detent torque, the tilt axis droops under gravity, and the
    # encoder reading no longer matches the real angle. The system must
    # refuse to fire and require re-homing while this is false.
    position_valid: bool
    driver_alarm_pan: bool
    driver_alarm_tilt: bool
    fan_rpm: tuple[int, int, int]
    mcu_temp_c: float
    loop_time_us: int
    crc_error_count: int


@dataclass(frozen=True)
class SelfTestItem:
    name: str
    passed: bool
    detail: str | None
    measured: float | None


@dataclass(frozen=True)
class SelfTestResult:
    items: tuple[SelfTestItem, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.items)


@dataclass
class SystemState:
    stage: Stage
    mode: Mode
    engagement: EngagementState
    active_layer: Layer
    layer_manual_override: bool
    fallback_reason: ReasonCode | None
    tracks: list[Track]
    selected_track_id: int | None
    telemetry: Telemetry | None
    last_self_test: SelfTestResult | None
