from __future__ import annotations

from celikkubbe.core.types import (
    IFF,
    CameraIntrinsics,
    EngagementState,
    Layer,
    Mode,
    ReasonCode,
    SelfTestItem,
    SelfTestResult,
    Stage,
    SystemState,
    TargetClass,
    Track,
    TrackStatus,
)


def make_intrinsics(quality: str) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0, quality=quality
    )


def test_intrinsics_reliable_for_factory() -> None:
    assert make_intrinsics("factory").is_reliable is True


def test_intrinsics_reliable_for_calibrated() -> None:
    assert make_intrinsics("calibrated").is_reliable is True


def test_intrinsics_unreliable_for_estimated() -> None:
    assert make_intrinsics("estimated").is_reliable is False


def test_self_test_result_passes_when_all_items_pass() -> None:
    result = SelfTestResult(
        items=(
            SelfTestItem(name="camera", passed=True, detail=None, measured=30.0),
            SelfTestItem(name="link", passed=True, detail=None, measured=None),
        )
    )
    assert result.passed is True


def test_self_test_result_fails_when_any_item_fails() -> None:
    result = SelfTestResult(
        items=(
            SelfTestItem(name="camera", passed=True, detail=None, measured=30.0),
            SelfTestItem(name="link", passed=False, detail="timeout", measured=None),
        )
    )
    assert result.passed is False


def test_self_test_result_passes_with_no_items() -> None:
    assert SelfTestResult(items=()).passed is True


def test_system_state_holds_tracks() -> None:
    track = Track(
        track_id=1,
        cls=TargetClass.UAV,
        confidence=0.9,
        range_m=8.0,
        iff=IFF.HOSTILE,
        bbox=(0.1, 0.1, 0.2, 0.2),
        velocity=(0.0, 0.0),
        status=TrackStatus.CONFIRMED,
        risk_score=50.0,
        frames_confirmed=5,
        last_seen_t=1.0,
        engagement_attempts=0,
    )
    state = SystemState(
        stage=Stage.STAGE_2,
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S1_SEARCH,
        active_layer=Layer.L1,
        layer_manual_override=False,
        fallback_reason=None,
        tracks=[track],
        selected_track_id=1,
        telemetry=None,
        last_self_test=None,
    )
    assert state.tracks[0].track_id == 1
    assert state.fallback_reason is None


def test_reason_code_members_are_distinct_strings() -> None:
    values = [code.value for code in ReasonCode]
    assert len(values) == len(set(values))
