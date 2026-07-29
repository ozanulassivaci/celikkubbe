from __future__ import annotations

from celikkubbe.core import config, engagement
from celikkubbe.core.commands import Fire
from celikkubbe.core.types import (
    EngagementState,
    HitResult,
    Mode,
    ReasonCode,
    Stage,
    TrackStatus,
)

from .factories import make_state, make_telemetry, make_track


def test_s1_advances_to_s2_when_track_present() -> None:
    state = make_state(engagement=EngagementState.S1_SEARCH)
    tracks = [make_track(status=TrackStatus.TENTATIVE)]
    new_state, commands = engagement.step(state, tracks, None, None, now=0.0)
    assert new_state is EngagementState.S2_ACQUIRE
    assert commands == []


def test_s1_stays_without_tracks() -> None:
    state = make_state(engagement=EngagementState.S1_SEARCH)
    new_state, _ = engagement.step(state, [], None, None, now=0.0)
    assert new_state is EngagementState.S1_SEARCH


def test_s2_advances_to_s3_when_confirmed() -> None:
    state = make_state(engagement=EngagementState.S2_ACQUIRE)
    tracks = [make_track(status=TrackStatus.CONFIRMED)]
    new_state, _ = engagement.step(state, tracks, None, None, now=0.0)
    assert new_state is EngagementState.S3_TRACK


def test_s2_stays_while_only_tentative() -> None:
    state = make_state(engagement=EngagementState.S2_ACQUIRE)
    tracks = [make_track(status=TrackStatus.TENTATIVE)]
    new_state, _ = engagement.step(state, tracks, None, None, now=0.0)
    assert new_state is EngagementState.S2_ACQUIRE


def test_s2_falls_back_to_s1_when_track_lost() -> None:
    state = make_state(engagement=EngagementState.S2_ACQUIRE)
    new_state, _ = engagement.step(state, [], None, None, now=0.0)
    assert new_state is EngagementState.S1_SEARCH


def test_s3_selects_target_and_advances_to_s4_in_stage2() -> None:
    state = make_state(stage=Stage.STAGE_2, engagement=EngagementState.S3_TRACK)
    tracks = [make_track(track_id=7, status=TrackStatus.CONFIRMED)]
    new_state, _ = engagement.step(state, tracks, None, None, now=0.0)
    assert new_state is EngagementState.S4_AIM
    assert state.selected_track_id == 7


def test_s3_stays_in_stage1_even_with_confirmed_target() -> None:
    state = make_state(stage=Stage.STAGE_1, engagement=EngagementState.S3_TRACK)
    tracks = [make_track(track_id=7, status=TrackStatus.CONFIRMED)]
    new_state, commands = engagement.step(state, tracks, None, None, now=0.0)
    assert new_state is EngagementState.S3_TRACK
    assert commands == []
    assert state.selected_track_id == 7


def test_s3_falls_back_to_s1_when_no_eligible_target() -> None:
    state = make_state(engagement=EngagementState.S3_TRACK)
    new_state, _ = engagement.step(state, [], None, None, now=0.0)
    assert new_state is EngagementState.S1_SEARCH


def test_s3_skips_target_that_exhausted_attempts() -> None:
    state = make_state(engagement=EngagementState.S3_TRACK)
    tracks = [
        make_track(
            track_id=1,
            status=TrackStatus.CONFIRMED,
            engagement_attempts=config.MAX_ENGAGEMENT_ATTEMPTS,
        )
    ]
    new_state, _ = engagement.step(state, tracks, None, None, now=0.0)
    assert new_state is EngagementState.S1_SEARCH


def _passing_context_state_and_track():
    telemetry = make_telemetry(t=0.0, armed=True, estop=False, position_valid=True)
    track = make_track(track_id=1, status=TrackStatus.CONFIRMED, confidence=0.9, range_m=8.0)
    state = make_state(
        stage=Stage.STAGE_2,
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S4_AIM,
        selected_track_id=1,
    )
    return state, [track], telemetry


def test_s4_advances_to_s5_when_all_gates_pass() -> None:
    state, tracks, telemetry = _passing_context_state_and_track()
    new_state, commands = engagement.step(state, tracks, telemetry, None, now=0.0)
    assert new_state is EngagementState.S5_ENGAGE
    assert commands == []


def test_s4_stays_when_a_gate_fails() -> None:
    state, tracks, telemetry = _passing_context_state_and_track()
    telemetry = make_telemetry(t=0.0, armed=False)
    new_state, _ = engagement.step(state, tracks, telemetry, None, now=0.0)
    assert new_state is EngagementState.S4_AIM


def test_s4_falls_back_to_s1_when_target_lost() -> None:
    state = make_state(engagement=EngagementState.S4_AIM, selected_track_id=1)
    new_state, _ = engagement.step(state, [], make_telemetry(), None, now=0.0)
    assert new_state is EngagementState.S1_SEARCH


def test_s4_waits_without_telemetry() -> None:
    state = make_state(engagement=EngagementState.S4_AIM, selected_track_id=1)
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    new_state, _ = engagement.step(state, tracks, None, None, now=0.0)
    assert new_state is EngagementState.S4_AIM


def test_s5_fires_and_advances_to_s6() -> None:
    state = make_state(engagement=EngagementState.S5_ENGAGE, selected_track_id=1)
    track = make_track(track_id=1, engagement_attempts=0)
    new_state, commands = engagement.step(state, [track], None, None, now=0.0)
    assert new_state is EngagementState.S6_ASSESS
    assert commands == [Fire(count=1)]
    assert track.engagement_attempts == 1


def test_s6_waits_without_hit_result() -> None:
    state = make_state(engagement=EngagementState.S6_ASSESS, selected_track_id=1)
    track = make_track(track_id=1, engagement_attempts=1)
    new_state, commands = engagement.step(state, [track], None, None, now=0.0)
    assert new_state is EngagementState.S6_ASSESS
    assert commands == []


def test_s6_kill_returns_to_s1() -> None:
    state = make_state(engagement=EngagementState.S6_ASSESS, selected_track_id=1)
    track = make_track(track_id=1, engagement_attempts=1)
    new_state, _ = engagement.step(state, [track], None, HitResult.KILL, now=0.0)
    assert new_state is EngagementState.S1_SEARCH
    assert state.selected_track_id is None


def test_s6_miss_retries_when_attempts_remain() -> None:
    state = make_state(engagement=EngagementState.S6_ASSESS, selected_track_id=1)
    track = make_track(track_id=1, engagement_attempts=1)
    new_state, _ = engagement.step(state, [track], None, HitResult.MISS, now=0.0)
    assert new_state is EngagementState.S4_AIM


def test_s6_miss_skips_target_after_max_attempts() -> None:
    state = make_state(engagement=EngagementState.S6_ASSESS, selected_track_id=1)
    track = make_track(track_id=1, engagement_attempts=config.MAX_ENGAGEMENT_ATTEMPTS)
    new_state, _ = engagement.step(state, [track], None, HitResult.MISS, now=0.0)
    assert new_state is EngagementState.S1_SEARCH
    assert state.selected_track_id is None


def test_stage1_never_emits_autonomous_fire() -> None:
    state = make_state(
        stage=Stage.STAGE_1,
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S1_SEARCH,
    )
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    telemetry = make_telemetry(t=0.0)
    seen_states = set()
    for _ in range(20):
        new_state, commands = engagement.step(state, tracks, telemetry, None, now=0.0)
        state.engagement = new_state
        seen_states.add(new_state)
        assert Fire(count=1) not in commands
    assert EngagementState.S4_AIM not in seen_states
    assert EngagementState.S5_ENGAGE not in seen_states


def test_operator_fire_passes_through_gates_and_rejects() -> None:
    state = make_state(
        stage=Stage.STAGE_1,
        mode=Mode.M2_STANDBY,
        engagement=EngagementState.S3_TRACK,
        selected_track_id=1,
    )
    track = make_track(track_id=1, status=TrackStatus.CONFIRMED)
    telemetry = make_telemetry(t=0.0)
    new_state, commands, reasons = engagement.request_manual_fire(state, [track], telemetry)
    assert commands == []
    assert ReasonCode.NOT_OPERATIONAL in reasons
    assert new_state is EngagementState.S3_TRACK


def test_operator_fire_succeeds_when_gates_pass() -> None:
    state = make_state(
        stage=Stage.STAGE_1,
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S3_TRACK,
        selected_track_id=1,
    )
    track = make_track(track_id=1, status=TrackStatus.CONFIRMED, engagement_attempts=0)
    telemetry = make_telemetry(t=0.0, armed=True, estop=False, position_valid=True)
    new_state, commands, reasons = engagement.request_manual_fire(state, [track], telemetry)
    assert reasons == []
    assert commands == [Fire(count=1)]
    assert new_state is EngagementState.S6_ASSESS
    assert track.engagement_attempts == 1


def test_operator_fire_without_telemetry_is_rejected() -> None:
    state = make_state(engagement=EngagementState.S3_TRACK, selected_track_id=1)
    track = make_track(track_id=1)
    new_state, commands, reasons = engagement.request_manual_fire(state, [track], None)
    assert commands == []
    assert reasons == [ReasonCode.LINK_TIMEOUT]
    assert new_state is EngagementState.S3_TRACK


def test_operator_fire_without_selected_target_is_noop() -> None:
    state = make_state(engagement=EngagementState.S1_SEARCH, selected_track_id=None)
    new_state, commands, reasons = engagement.request_manual_fire(state, [], None)
    assert commands == []
    assert reasons == []
    assert new_state is EngagementState.S1_SEARCH
