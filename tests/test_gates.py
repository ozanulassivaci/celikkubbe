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
    passed, reason = IFFGate.evaluate(IFF.FRIENDLY, TargetClass.UAV)
    assert passed is False
    assert reason is ReasonCode.TARGET_FRIENDLY


def test_iff_gate_rejects_unknown_class() -> None:
    passed, reason = IFFGate.evaluate(IFF.HOSTILE, None)
    assert passed is False
    assert reason is ReasonCode.CLASS_UNKNOWN


def test_iff_gate_passes_hostile_with_known_class() -> None:
    passed, reason = IFFGate.evaluate(IFF.HOSTILE, TargetClass.UAV)
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


def test_range_gate_allows_class_without_rule() -> None:
    passed, reason = RangeGate.evaluate(TargetClass.BALLOON, 100.0, Stage.STAGE_3)
    assert passed is True
    assert reason is None


def test_confidence_gate_boundary() -> None:
    assert ConfidenceGate.evaluate(config.CONFIDENCE_THRESHOLD)[0] is True
    assert ConfidenceGate.evaluate(config.CONFIDENCE_THRESHOLD - 0.01)[0] is False


def test_angle_gate_boundary_passes_at_tolerance() -> None:
    passed, reason = AngleGate.evaluate(0.0, 0.0, config.ANGLE_TOLERANCE_DEG, 0.0)
    assert passed is True
    assert reason is None


def test_angle_gate_boundary_fails_past_tolerance() -> None:
    passed, reason = AngleGate.evaluate(0.0, 0.0, config.ANGLE_TOLERANCE_DEG + 0.01, 0.0)
    assert passed is False
    assert reason is ReasonCode.ANGLE_NOT_SETTLED


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


def test_evaluate_all_returns_every_gates_failure() -> None:
    # Each gate reports at most one reason per call (e.g. estop takes
    # priority over the other SafetyGate checks); evaluate_all's "every
    # failure" guarantee is across the six gates, not within one.
    ctx = GateContext(
        stage=Stage.STAGE_3,
        mode=Mode.M2_STANDBY,
        armed=True,
        estop=False,
        position_valid=True,
        iff=IFF.HOSTILE,
        cls=None,
        range_m=None,
        confidence=0.1,
        current_pan_deg=0.0,
        current_tilt_deg=0.0,
        target_pan_deg=200.0,
        target_tilt_deg=0.0,
    )
    reasons = evaluate_all(ctx)
    assert reasons == [
        ReasonCode.NOT_OPERATIONAL,
        ReasonCode.CLASS_UNKNOWN,
        ReasonCode.RANGE_UNKNOWN,
        ReasonCode.LOW_CONFIDENCE,
        ReasonCode.ANGLE_NOT_SETTLED,
        ReasonCode.LIMIT_EXCEEDED,
    ]


def test_evaluate_all_safety_gate_reports_not_armed_in_isolation() -> None:
    ctx = GateContext(
        stage=Stage.STAGE_2,
        mode=Mode.M3_OPERATIONAL,
        armed=False,
        estop=False,
        position_valid=True,
        iff=IFF.HOSTILE,
        cls=TargetClass.UAV,
        range_m=10.0,
        confidence=0.9,
        current_pan_deg=5.0,
        current_tilt_deg=5.0,
        target_pan_deg=5.0,
        target_tilt_deg=5.0,
    )
    assert evaluate_all(ctx) == [ReasonCode.NOT_ARMED]


def test_evaluate_all_passes_with_no_failures() -> None:
    ctx = GateContext(
        stage=Stage.STAGE_2,
        mode=Mode.M3_OPERATIONAL,
        armed=True,
        estop=False,
        position_valid=True,
        iff=IFF.HOSTILE,
        cls=TargetClass.UAV,
        range_m=10.0,
        confidence=0.9,
        current_pan_deg=5.0,
        current_tilt_deg=5.0,
        target_pan_deg=5.0,
        target_tilt_deg=5.0,
    )
    assert evaluate_all(ctx) == []
