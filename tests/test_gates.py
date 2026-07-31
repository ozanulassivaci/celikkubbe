from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.gates import (
    AngleGate,
    ConfidenceGate,
    GateContext,
    IFFGate,
    LimitGate,
    RangeGate,
    SafetyGate,
    evaluate_all,
)
from celikkubbe.core.types import IFF, Mode, ReasonCode, Stage, TargetClass


def test_safety_gate_passes_when_all_conditions_met() -> None:
    passed, reason = SafetyGate.evaluate(Mode.M3_OPERATIONAL, True, False, True)
    assert passed is True
    assert reason is None


def test_safety_gate_rejects_estop_first() -> None:
    passed, reason = SafetyGate.evaluate(Mode.M2_STANDBY, False, True, False)
    assert passed is False
    assert reason is ReasonCode.ESTOP_ACTIVE


def test_safety_gate_rejects_not_operational() -> None:
    passed, reason = SafetyGate.evaluate(Mode.M2_STANDBY, True, False, True)
    assert passed is False
    assert reason is ReasonCode.NOT_OPERATIONAL


def test_safety_gate_rejects_not_armed() -> None:
    passed, reason = SafetyGate.evaluate(Mode.M3_OPERATIONAL, False, False, True)
    assert passed is False
    assert reason is ReasonCode.NOT_ARMED


def test_safety_gate_rejects_position_invalid() -> None:
    passed, reason = SafetyGate.evaluate(Mode.M3_OPERATIONAL, True, False, False)
    assert passed is False
    assert reason is ReasonCode.POSITION_INVALID


def test_iff_gate_rejects_friendly() -> None:
    passed, reason = IFFGate.evaluate(IFF.FRIENDLY)
    assert passed is False
    assert reason is ReasonCode.TARGET_FRIENDLY


def test_iff_gate_rejects_unknown_iff() -> None:
    passed, reason = IFFGate.evaluate(IFF.UNKNOWN)
    assert passed is False
    assert reason is ReasonCode.IFF_UNKNOWN


def test_iff_gate_passes_hostile_regardless_of_class() -> None:
    # Competition targets are colour-coded, so L2 can set IFF from colour
    # even though it never determines class. Class absence is RangeGate's
    # job to police (via the Stage 3 rule), not this gate's.
    passed, reason = IFFGate.evaluate(IFF.HOSTILE)
    assert passed is True
    assert reason is None


def test_range_gate_rejects_unknown_range_in_stage3() -> None:
    passed, reason = RangeGate.evaluate(TargetClass.UAV, None, Stage.STAGE_3)
    assert passed is False
    assert reason is ReasonCode.RANGE_UNKNOWN


def test_range_gate_allows_unknown_range_in_stage1_and_2() -> None:
    for stage in (Stage.STAGE_1, Stage.STAGE_2):
        passed, reason = RangeGate.evaluate(TargetClass.UAV, None, stage)
        assert passed is True
        assert reason is None


def test_range_gate_enforces_bounds_for_known_class() -> None:
    lo, hi = config.RANGE_RULES[TargetClass.F16]
    passed, _ = RangeGate.evaluate(TargetClass.F16, lo - 0.1, Stage.STAGE_3)
    assert passed is False
    passed, _ = RangeGate.evaluate(TargetClass.F16, (lo + hi) / 2, Stage.STAGE_3)
    assert passed is True
    passed, _ = RangeGate.evaluate(TargetClass.F16, hi + 0.1, Stage.STAGE_3)
    assert passed is False


def test_range_gate_fails_closed_for_class_without_rule_in_stage3() -> None:
    passed, reason = RangeGate.evaluate(TargetClass.UNKNOWN, 5.0, Stage.STAGE_3)
    assert passed is False
    assert reason is ReasonCode.RANGE_UNKNOWN

    passed, reason = RangeGate.evaluate(None, 5.0, Stage.STAGE_3)
    assert passed is False
    assert reason is ReasonCode.RANGE_UNKNOWN


def test_range_gate_allows_class_without_rule_in_stage1_and_2() -> None:
    for stage in (Stage.STAGE_1, Stage.STAGE_2):
        passed, reason = RangeGate.evaluate(TargetClass.UNKNOWN, 100.0, stage)
        assert passed is True
        assert reason is None


def test_l2_detection_passes_iff_gate_but_fails_range_gate_in_stage3() -> None:
    # cls=None (L2 never classifies), iff=HOSTILE (L2 sets this from colour).
    passed, reason = IFFGate.evaluate(IFF.HOSTILE)
    assert passed is True
    assert reason is None

    passed, reason = RangeGate.evaluate(None, 8.0, Stage.STAGE_3)
    assert passed is False
    assert reason is ReasonCode.RANGE_UNKNOWN

    # Same detection is fine in Stage 1/2, where a missing rule is lenient.
    passed, reason = RangeGate.evaluate(None, 8.0, Stage.STAGE_2)
    assert passed is True
    assert reason is None


def test_confidence_gate_boundary() -> None:
    assert ConfidenceGate.evaluate(config.CONFIDENCE_THRESHOLD)[0] is True
    assert ConfidenceGate.evaluate(config.CONFIDENCE_THRESHOLD - 0.01)[0] is False


def test_angle_gate_passes_when_acked_complete_and_no_alarm() -> None:
    passed, reason = AngleGate.evaluate(
        target_pan_deg=10.0,
        target_tilt_deg=5.0,
        commanded_pan_deg=10.0,
        commanded_tilt_deg=5.0,
        motion_complete=True,
        driver_alarm_pan=False,
        driver_alarm_tilt=False,
    )
    assert passed is True
    assert reason is None


def test_angle_gate_rejects_stale_echoed_setpoint() -> None:
    # telemetry still echoes the previous Goto (target_pan_deg=0.0); we
    # last commanded 10.0. motion_complete reading True here would be
    # reporting completion of the OLD move, not the new one.
    passed, reason = AngleGate.evaluate(
        target_pan_deg=0.0,
        target_tilt_deg=0.0,
        commanded_pan_deg=10.0,
        commanded_tilt_deg=0.0,
        motion_complete=True,
        driver_alarm_pan=False,
        driver_alarm_tilt=False,
    )
    assert passed is False
    assert reason is ReasonCode.SETPOINT_NOT_ACKED


def test_angle_gate_rejects_when_nothing_commanded_yet() -> None:
    passed, reason = AngleGate.evaluate(
        target_pan_deg=0.0,
        target_tilt_deg=0.0,
        commanded_pan_deg=None,
        commanded_tilt_deg=None,
        motion_complete=True,
        driver_alarm_pan=False,
        driver_alarm_tilt=False,
    )
    assert passed is False
    assert reason is ReasonCode.SETPOINT_NOT_ACKED


def test_angle_gate_passes_once_echoed_setpoint_matches_within_epsilon() -> None:
    passed, reason = AngleGate.evaluate(
        target_pan_deg=10.0 + config.SETPOINT_ACK_EPSILON_DEG,
        target_tilt_deg=0.0,
        commanded_pan_deg=10.0,
        commanded_tilt_deg=0.0,
        motion_complete=True,
        driver_alarm_pan=False,
        driver_alarm_tilt=False,
    )
    assert passed is True
    assert reason is None


def test_angle_gate_fails_with_motion_in_progress() -> None:
    passed, reason = AngleGate.evaluate(
        target_pan_deg=10.0,
        target_tilt_deg=0.0,
        commanded_pan_deg=10.0,
        commanded_tilt_deg=0.0,
        motion_complete=False,
        driver_alarm_pan=False,
        driver_alarm_tilt=False,
    )
    assert passed is False
    assert reason is ReasonCode.MOTION_IN_PROGRESS


def test_angle_gate_fails_with_driver_alarm_pan() -> None:
    passed, reason = AngleGate.evaluate(
        target_pan_deg=10.0,
        target_tilt_deg=0.0,
        commanded_pan_deg=10.0,
        commanded_tilt_deg=0.0,
        motion_complete=True,
        driver_alarm_pan=True,
        driver_alarm_tilt=False,
    )
    assert passed is False
    assert reason is ReasonCode.DRIVER_ALARM


def test_angle_gate_fails_with_driver_alarm_tilt() -> None:
    passed, reason = AngleGate.evaluate(
        target_pan_deg=10.0,
        target_tilt_deg=0.0,
        commanded_pan_deg=10.0,
        commanded_tilt_deg=0.0,
        motion_complete=True,
        driver_alarm_pan=False,
        driver_alarm_tilt=True,
    )
    assert passed is False
    assert reason is ReasonCode.DRIVER_ALARM


def test_angle_gate_checks_setpoint_ack_before_driver_alarm() -> None:
    # Ordering matters for evaluate_all's single-reason-per-gate contract:
    # an un-acked setpoint is reported over a driver alarm that may just
    # be a stale leftover from the previous move.
    passed, reason = AngleGate.evaluate(
        target_pan_deg=0.0,
        target_tilt_deg=0.0,
        commanded_pan_deg=10.0,
        commanded_tilt_deg=0.0,
        motion_complete=True,
        driver_alarm_pan=True,
        driver_alarm_tilt=False,
    )
    assert passed is False
    assert reason is ReasonCode.SETPOINT_NOT_ACKED


def test_limit_gate_within_bounds() -> None:
    passed, reason = LimitGate.evaluate(0.0, 0.0)
    assert passed is True
    assert reason is None


def test_limit_gate_rejects_pan_out_of_bounds() -> None:
    passed, reason = LimitGate.evaluate(config.PAN_LIMIT_DEG[1] + 1.0, 0.0)
    assert passed is False
    assert reason is ReasonCode.LIMIT_EXCEEDED


def test_limit_gate_rejects_tilt_out_of_bounds() -> None:
    passed, reason = LimitGate.evaluate(0.0, config.TILT_LIMIT_DEG[0] - 1.0)
    assert passed is False
    assert reason is ReasonCode.LIMIT_EXCEEDED


def _ctx(**overrides) -> GateContext:
    defaults = dict(
        stage=Stage.STAGE_2,
        mode=Mode.M3_OPERATIONAL,
        armed=True,
        estop=False,
        position_valid=True,
        iff=IFF.HOSTILE,
        cls=TargetClass.UAV,
        range_m=10.0,
        confidence=0.9,
        target_pan_deg=5.0,
        target_tilt_deg=5.0,
        commanded_pan_deg=5.0,
        commanded_tilt_deg=5.0,
        motion_complete=True,
        driver_alarm_pan=False,
        driver_alarm_tilt=False,
    )
    defaults.update(overrides)
    return GateContext(**defaults)


def test_evaluate_all_returns_every_gates_failure() -> None:
    # Each gate reports at most one reason per call (e.g. estop takes
    # priority over the other SafetyGate checks); evaluate_all's "every
    # failure" guarantee is across the six gates, not within one.
    ctx = _ctx(
        stage=Stage.STAGE_3,
        mode=Mode.M2_STANDBY,
        iff=IFF.UNKNOWN,
        cls=None,
        range_m=None,
        confidence=0.1,
        target_pan_deg=200.0,
        commanded_pan_deg=200.0,
        motion_complete=False,
    )
    reasons = evaluate_all(ctx)
    assert reasons == [
        ReasonCode.NOT_OPERATIONAL,
        ReasonCode.IFF_UNKNOWN,
        ReasonCode.RANGE_UNKNOWN,
        ReasonCode.LOW_CONFIDENCE,
        ReasonCode.MOTION_IN_PROGRESS,
        ReasonCode.LIMIT_EXCEEDED,
    ]


def test_evaluate_all_safety_gate_reports_not_armed_in_isolation() -> None:
    assert evaluate_all(_ctx(armed=False)) == [ReasonCode.NOT_ARMED]


def test_evaluate_all_passes_with_no_failures() -> None:
    assert evaluate_all(_ctx()) == []
