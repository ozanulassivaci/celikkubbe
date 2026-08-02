"""Enums and data contracts shared across the core decision layer.

No internal dependencies: every other module in ``core`` builds on top of
this one, never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

# bbox / roi convention used everywhere in this codebase: normalised
# (x1, y1, x2, y2) — min corner then max corner, each in [0, 1]. Never
# (cx, cy, w, h), and never pixels.
BoundingBox = tuple[float, float, float, float]

RangeSource = Literal["depth", "size", "none"]


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


class McuMode(Enum):
    """The MCU's own internal mode, echoed in Telemetry.status bits 11-13
    (docs/protocol.md section 4) -- distinct from the PC-side Mode above.
    Only IDLE/READY/SAFE are ever commanded (via SetMode); BOOT and MOVING
    are states the MCU reports on its own.
    """

    BOOT = "BOOT"
    IDLE = "IDLE"
    READY = "READY"
    MOVING = "MOVING"
    SAFE = "SAFE"


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
    RANGE_UNKNOWN = "RANGE_UNKNOWN"
    RANGE_OUT_OF_BOUNDS = "RANGE_OUT_OF_BOUNDS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INFERENCE_SLOW = "INFERENCE_SLOW"
    CAMERA_TIMEOUT = "CAMERA_TIMEOUT"
    LINK_TIMEOUT = "LINK_TIMEOUT"
    DEPTH_UNRELIABLE = "DEPTH_UNRELIABLE"
    OPERATOR_OVERRIDE = "OPERATOR_OVERRIDE"
    SETPOINT_NOT_ACKED = "SETPOINT_NOT_ACKED"
    NO_AIM_SOLUTION = "NO_AIM_SOLUTION"
    MOTION_IN_PROGRESS = "MOTION_IN_PROGRESS"
    DRIVER_ALARM = "DRIVER_ALARM"
    IFF_UNKNOWN = "IFF_UNKNOWN"
    NOT_HOMED = "NOT_HOMED"


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
    # Float array in METRES when present, same pixel grid as `image`.
    # 0.0 or NaN marks an invalid pixel (D435i dropout, out of range, ...).
    # Whoever produces a Frame is responsible for applying the sensor's
    # depth scale before it lands here — nothing downstream may assume a
    # raw sensor unit.
    depth: Any | None = None


@dataclass(frozen=True)
class Detection:
    bbox: BoundingBox
    cls: TargetClass | None
    confidence: float
    range_m: float | None
    # How range_m was derived. Depth is trustworthy; a size-based estimate
    # from a known real-world diameter is coarse and stage 3's range gate
    # should weight it with less confidence; "none" means range_m is None.
    range_source: RangeSource
    source_layer: Layer
    iff: IFF


@dataclass(frozen=True)
class Track:
    """A single frame's worth of perception output for one target.

    Frozen: the tracker (``tracking.py``) builds a fresh ``Track`` every
    frame rather than mutating one in place. Engagement bookkeeping that
    must survive across frames (shot attempts) does not belong here for
    exactly that reason — see ``SystemState.attempts``.
    """

    track_id: int
    cls: TargetClass | None
    confidence: float
    range_m: float | None
    range_source: RangeSource
    iff: IFF
    bbox: BoundingBox
    velocity: tuple[float, float]
    status: TrackStatus
    risk_score: float
    frames_confirmed: int
    last_seen_t: float


@dataclass(frozen=True)
class Telemetry:
    # PC-side receive timestamp, taken from the injected Clock when this
    # packet is parsed — not an MCU clock reading. The two time bases are
    # unsynchronised; if the MCU's own timestamp is ever needed, it belongs
    # in a separate `mcu_t` field, never compared directly against `now`.
    t: float
    # pan_deg / tilt_deg are the MCU's COMMANDED position, integrated from
    # the step pulses it has issued — NOT a measured position. The motor
    # encoders wire to the stepper drivers, which close the position loop
    # internally; the MCU never sees them. Reading these as "where the
    # turret actually is" is easy to get away with on the bench and wrong
    # the moment a motor stalls or loses steps. The only evidence that a
    # commanded move actually completed is `motion_complete` plus a clear
    # driver alarm — see AngleGate.
    pan_deg: float
    tilt_deg: float
    pan_vel_dps: float
    tilt_vel_dps: float
    target_pan_deg: float
    target_tilt_deg: float
    # True once the MCU's trajectory generator reports the current move
    # finished. There is no independent following-error measurement to
    # cross-check this against (see pan_deg/tilt_deg above), so it is
    # taken as-is.
    motion_complete: bool
    armed: bool
    estop: bool
    # False after an e-stop: the 48V rail is cut, unpowered steppers only
    # have detent torque, the tilt axis droops under gravity, and the
    # encoder reading no longer matches the real angle. The system must
    # refuse to fire and require re-homing while this is false.
    position_valid: bool
    driver_alarm_pan: bool
    driver_alarm_tilt: bool
    # Decision inputs, not diagnostics: modes.py refuses M3 OPERATIONAL
    # until both homing bits are set (docs/protocol.md section 5 — there
    # are no limit switches, so `zero` is the only homing mechanism), and
    # mcu_mode lets modes.py notice the MCU disagreeing with what the PC
    # last commanded via SetMode.
    homed_pan: bool
    homed_tilt: bool
    mcu_mode: McuMode
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


@dataclass(frozen=True)
class OperatorInput:
    """Manual controls sampled once per tick. Stage 2/3 ignore this entirely."""

    fire_requested: bool = False
    manual_target_id: int | None = None
    arm_held: bool = False  # Stage 1 RT hold-to-arm dead-man switch


@dataclass(frozen=True)
class SystemState:
    """The immutable snapshot the GUI reads and the control loop replaces.

    Nothing in ``core`` mutates a ``SystemState``. Each tick, the outer
    control loop builds the next one with ``dataclasses.replace()``, folding
    in whatever a state machine's result (e.g. ``engagement.StepResult``)
    says should change. ``tracks`` is a tuple for the same reason: a plain
    list is still a mutable container even when its elements are frozen,
    and the GUI thread reads this snapshot while the control worker
    produces the next one.
    """

    stage: Stage
    mode: Mode
    engagement: EngagementState
    active_layer: Layer
    layer_manual_override: bool
    fallback_reason: ReasonCode | None
    tracks: tuple[Track, ...]
    selected_track_id: int | None
    attempts: dict[int, int]  # track_id -> engagement attempts so far
    # track_id -> (defer-until timestamp, reason it was deferred). A track
    # stuck failing a recoverable gate (range, limits, aim solution — not
    # permanently-excluded FRIENDLY) for GATE_REJECT_TIMEOUT_MS lands here
    # so S3 tries the next candidate instead of spinning in S4 forever.
    deferred: dict[int, tuple[float, ReasonCode]]
    # When the currently selected S4 track's gate failures began, if any;
    # None while gates are passing or nothing is selected. Used to measure
    # GATE_REJECT_TIMEOUT_MS for the deferral above.
    gate_fail_since: float | None
    commanded_pan_deg: float | None
    commanded_tilt_deg: float | None
    telemetry: Telemetry | None
    last_self_test: SelfTestResult | None
