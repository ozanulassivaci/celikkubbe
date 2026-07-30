"""Safety / IFF / range / confidence / angle / limit gates.

Every gate is a stateless, pure check: given inputs, it returns whether the
action is allowed and, if not, why. ``evaluate_all`` runs every gate and
reports every failure at once so the operator sees the full picture.
"""

from __future__ import annotations

from dataclasses import dataclass

from celikkubbe.core import config
from celikkubbe.core.types import IFF, Mode, ReasonCode, Stage, TargetClass


class SafetyGate:
    @staticmethod
    def evaluate(
        mode: Mode, armed: bool, estop: bool, position_valid: bool
    ) -> tuple[bool, ReasonCode | None]:
        if estop:
            return False, ReasonCode.ESTOP_ACTIVE
        if mode is not Mode.M3_OPERATIONAL:
            return False, ReasonCode.NOT_OPERATIONAL
        if not armed:
            return False, ReasonCode.NOT_ARMED
        if not position_valid:
            return False, ReasonCode.POSITION_INVALID
        return True, None


class IFFGate:
    @staticmethod
    def evaluate(iff: IFF, cls: TargetClass | None) -> tuple[bool, ReasonCode | None]:
        if iff is IFF.FRIENDLY:
            return False, ReasonCode.TARGET_FRIENDLY
        if cls is None:
            # Conservative default: L2 never produces a class, so an
            # unclassified target must never be engaged automatically.
            return False, ReasonCode.CLASS_UNKNOWN
        return True, None


class RangeGate:
    @staticmethod
    def evaluate(
        cls: TargetClass | None, range_m: float | None, stage: Stage
    ) -> tuple[bool, ReasonCode | None]:
        # Fails closed: a class with no entry in RANGE_RULES (including
        # UNKNOWN and None) is rejected, same as an unknown range_m. "No
        # rule" must never be read as "no limit". Stage 1/2 stay lenient
        # so development against a depth-less webcam or an unlisted class
        # (e.g. balloons before classification) still works.
        bounds = config.RANGE_RULES.get(cls)
        if range_m is None or bounds is None:
            if stage is Stage.STAGE_3:
                return False, ReasonCode.RANGE_UNKNOWN
            return True, None
        lo, hi = bounds
        if lo <= range_m <= hi:
            return True, None
        return False, ReasonCode.RANGE_OUT_OF_BOUNDS


class ConfidenceGate:
    @staticmethod
    def evaluate(
        confidence: float, threshold: float = config.CONFIDENCE_THRESHOLD
    ) -> tuple[bool, ReasonCode | None]:
        if confidence >= threshold:
            return True, None
        return False, ReasonCode.LOW_CONFIDENCE


class AngleGate:
    @staticmethod
    def evaluate(
        current_pan_deg: float,
        current_tilt_deg: float,
        target_pan_deg: float,
        target_tilt_deg: float,
        commanded_pan_deg: float | None,
        commanded_tilt_deg: float | None,
        tolerance_deg: float = config.ANGLE_TOLERANCE_DEG,
        setpoint_ack_epsilon_deg: float = config.SETPOINT_ACK_EPSILON_DEG,
    ) -> tuple[bool, ReasonCode | None]:
        # telemetry's echoed setpoint (target_*) can still hold the *previous*
        # Goto for a few ticks after a new one is sent — the turret reads as
        # settled on stale data. Refuse to pass until telemetry has echoed
        # back the angle we actually last commanded.
        if (
            commanded_pan_deg is None
            or commanded_tilt_deg is None
            or abs(target_pan_deg - commanded_pan_deg) > setpoint_ack_epsilon_deg
            or abs(target_tilt_deg - commanded_tilt_deg) > setpoint_ack_epsilon_deg
        ):
            return False, ReasonCode.SETPOINT_NOT_ACKED
        pan_error = abs(current_pan_deg - target_pan_deg)
        tilt_error = abs(current_tilt_deg - target_tilt_deg)
        if pan_error <= tolerance_deg and tilt_error <= tolerance_deg:
            return True, None
        return False, ReasonCode.ANGLE_NOT_SETTLED


class LimitGate:
    @staticmethod
    def evaluate(
        target_pan_deg: float,
        target_tilt_deg: float,
        pan_limit_deg: tuple[float, float] = config.PAN_LIMIT_DEG,
        tilt_limit_deg: tuple[float, float] = config.TILT_LIMIT_DEG,
    ) -> tuple[bool, ReasonCode | None]:
        pan_lo, pan_hi = pan_limit_deg
        tilt_lo, tilt_hi = tilt_limit_deg
        if not (pan_lo <= target_pan_deg <= pan_hi):
            return False, ReasonCode.LIMIT_EXCEEDED
        if not (tilt_lo <= target_tilt_deg <= tilt_hi):
            return False, ReasonCode.LIMIT_EXCEEDED
        return True, None


@dataclass(frozen=True)
class GateContext:
    stage: Stage
    mode: Mode
    armed: bool
    estop: bool
    position_valid: bool
    iff: IFF
    cls: TargetClass | None
    range_m: float | None
    confidence: float
    current_pan_deg: float
    current_tilt_deg: float
    target_pan_deg: float
    target_tilt_deg: float
    commanded_pan_deg: float | None
    commanded_tilt_deg: float | None


def evaluate_all(ctx: GateContext) -> list[ReasonCode]:
    """Run every gate and return every failing reason, in gate-list order."""
    results = (
        SafetyGate.evaluate(ctx.mode, ctx.armed, ctx.estop, ctx.position_valid),
        IFFGate.evaluate(ctx.iff, ctx.cls),
        RangeGate.evaluate(ctx.cls, ctx.range_m, ctx.stage),
        ConfidenceGate.evaluate(ctx.confidence),
        AngleGate.evaluate(
            ctx.current_pan_deg,
            ctx.current_tilt_deg,
            ctx.target_pan_deg,
            ctx.target_tilt_deg,
            ctx.commanded_pan_deg,
            ctx.commanded_tilt_deg,
        ),
        LimitGate.evaluate(ctx.target_pan_deg, ctx.target_tilt_deg),
    )
    return [reason for passed, reason in results if not passed and reason is not None]
