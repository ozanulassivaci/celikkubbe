from __future__ import annotations

import dataclasses

from celikkubbe.core import config, engagement
from celikkubbe.core.commands import Fire, Goto
from celikkubbe.core.types import (
    IFF,
    EngagementState,
    HitResult,
    Mode,
    OperatorInput,
    ReasonCode,
    Stage,
    SystemState,
    TargetClass,
    Track,
    TrackStatus,
)

from .factories import make_state, make_telemetry, make_track


def _apply(state: SystemState, result: engagement.StepResult) -> SystemState:
    """Mimic what the outer control loop does with a StepResult."""
    return dataclasses.replace(
        state,
        engagement=result.engagement,
        selected_track_id=result.selected_track_id,
        attempts=result.attempts,
        deferred=result.deferred,
        gate_fail_since=result.gate_fail_since,
        commanded_pan_deg=result.commanded_pan_deg,
        commanded_tilt_deg=result.commanded_tilt_deg,
    )


def test_s1_advances_to_s2_when_track_present() -> None:
    state = make_state(engagement=EngagementState.S1_SEARCH)
    tracks = [make_track(status=TrackStatus.TENTATIVE)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S2_ACQUIRE
    assert result.commands == []


def test_s1_stays_without_tracks() -> None:
    state = make_state(engagement=EngagementState.S1_SEARCH)
    result = engagement.step(state, [], None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH


def test_s2_advances_to_s3_when_confirmed() -> None:
    state = make_state(engagement=EngagementState.S2_ACQUIRE)
    tracks = [make_track(status=TrackStatus.CONFIRMED)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S3_TRACK


def test_s2_stays_while_only_tentative() -> None:
    state = make_state(engagement=EngagementState.S2_ACQUIRE)
    tracks = [make_track(status=TrackStatus.TENTATIVE)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S2_ACQUIRE


def test_s2_falls_back_to_s1_when_track_lost() -> None:
    state = make_state(engagement=EngagementState.S2_ACQUIRE)
    result = engagement.step(state, [], None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH


def test_s3_selects_target_and_advances_to_s4_in_stage2() -> None:
    state = make_state(stage=Stage.STAGE_2, engagement=EngagementState.S3_TRACK)
    tracks = [make_track(track_id=7, status=TrackStatus.CONFIRMED)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM
    assert result.selected_track_id == 7


def test_s3_stays_in_stage1_without_arm_held() -> None:
    state = make_state(stage=Stage.STAGE_1, engagement=EngagementState.S3_TRACK)
    tracks = [make_track(track_id=7, status=TrackStatus.CONFIRMED)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S3_TRACK
    assert result.commands == []
    assert result.selected_track_id == 7


def test_s3_advances_to_s4_in_stage1_when_armed() -> None:
    state = make_state(stage=Stage.STAGE_1, engagement=EngagementState.S3_TRACK)
    tracks = [make_track(track_id=7, status=TrackStatus.CONFIRMED)]
    operator = OperatorInput(arm_held=True)
    result = engagement.step(state, tracks, None, None, operator, {}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM


def test_s3_reselects_when_previously_selected_target_is_lost() -> None:
    state = make_state(engagement=EngagementState.S3_TRACK, selected_track_id=1)
    tracks = [
        make_track(track_id=1, status=TrackStatus.LOST),
        make_track(track_id=2, status=TrackStatus.CONFIRMED),
    ]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM
    assert result.selected_track_id == 2


def test_s5_falls_back_to_s1_when_target_disappears_before_firing() -> None:
    state = make_state(engagement=EngagementState.S5_ENGAGE, selected_track_id=1)
    result = engagement.step(state, [], None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH
    assert result.selected_track_id is None
    assert result.commands == []


def test_s3_falls_back_to_s1_when_no_eligible_target() -> None:
    state = make_state(engagement=EngagementState.S3_TRACK)
    result = engagement.step(state, [], None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH


def test_s3_skips_target_that_exhausted_attempts() -> None:
    state = make_state(
        engagement=EngagementState.S3_TRACK, attempts={1: config.MAX_ENGAGEMENT_ATTEMPTS}
    )
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH


def test_s3_manual_target_id_overrides_auto_selection_in_stage1() -> None:
    state = make_state(stage=Stage.STAGE_1, engagement=EngagementState.S3_TRACK)
    tracks = [
        make_track(track_id=1, status=TrackStatus.CONFIRMED, risk_score=90.0),
        make_track(track_id=2, status=TrackStatus.CONFIRMED, risk_score=10.0),
    ]
    operator = OperatorInput(manual_target_id=2)
    result = engagement.step(state, tracks, None, None, operator, {}, now=0.0)
    assert result.selected_track_id == 2


def _armed_s4_state(**overrides) -> SystemState:
    defaults = dict(
        stage=Stage.STAGE_2,
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S4_AIM,
        selected_track_id=1,
        commanded_pan_deg=5.0,
        commanded_tilt_deg=5.0,
    )
    defaults.update(overrides)
    return make_state(**defaults)


def test_s4_advances_to_s5_when_all_gates_pass() -> None:
    state = _armed_s4_state()
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED, confidence=0.9, range_m=8.0)]
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0
    )
    result = engagement.step(state, tracks, telemetry, None, None, {1: (5.0, 5.0)}, now=0.0)
    assert result.engagement is EngagementState.S5_ENGAGE
    assert result.commands == []
    assert result.fallback_reason is None


def test_s4_stays_and_reports_first_gate_failure() -> None:
    state = _armed_s4_state()
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED, confidence=0.9, range_m=8.0)]
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0, armed=False
    )
    result = engagement.step(state, tracks, telemetry, None, None, {1: (5.0, 5.0)}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM
    assert result.fallback_reason is ReasonCode.NOT_ARMED


def test_s4_falls_back_to_s1_when_target_lost() -> None:
    state = make_state(engagement=EngagementState.S4_AIM, selected_track_id=1)
    result = engagement.step(state, [], make_telemetry(), None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH
    assert result.selected_track_id is None


def test_s4_waits_without_telemetry() -> None:
    state = make_state(engagement=EngagementState.S4_AIM, selected_track_id=1)
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM


def test_s4_emits_goto_and_blocks_on_stale_echoed_setpoint() -> None:
    # Fresh S4 entry: nothing commanded yet, aiming module wants (5, 5), but
    # telemetry still echoes the previous setpoint (0, 0) and the turret is
    # (coincidentally) settled there. Without the ack guard this would pass
    # AngleGate and fire at the old angle.
    state = make_state(
        stage=Stage.STAGE_2,
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S4_AIM,
        selected_track_id=1,
    )
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    telemetry = make_telemetry(
        t=0.0, pan_deg=0.0, tilt_deg=0.0, target_pan_deg=0.0, target_tilt_deg=0.0
    )
    result = engagement.step(state, tracks, telemetry, None, None, {1: (5.0, 5.0)}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM
    assert result.fallback_reason is ReasonCode.SETPOINT_NOT_ACKED
    assert result.commands == [
        Goto(
            az_deg=5.0,
            el_deg=5.0,
            max_vel_dps=config.AIM_MAX_VEL_DPS,
            max_accel_dps2=config.AIM_MAX_ACCEL_DPS2,
        )
    ]
    assert result.commanded_pan_deg == 5.0
    assert result.commanded_tilt_deg == 5.0


def test_s4_advances_once_telemetry_catches_up_to_commanded_angle() -> None:
    state = _armed_s4_state()  # commanded already (5.0, 5.0) from a prior tick
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0
    )
    # Aiming module reports the same solution as last tick; already acked,
    # so no new Goto is needed.
    result = engagement.step(state, tracks, telemetry, None, None, {1: (5.0, 5.0)}, now=0.0)
    assert result.engagement is EngagementState.S5_ENGAGE
    assert result.commands == []


def test_s4_blocked_when_no_aim_solution_for_selected_track() -> None:
    state = _armed_s4_state()  # selected_track_id=1
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0
    )
    # Solutions exist for other tracks, just not the selected one.
    result = engagement.step(
        state, tracks, telemetry, None, None, {2: (1.0, 1.0), 3: (2.0, 2.0)}, now=0.0
    )
    assert result.engagement is EngagementState.S4_AIM
    assert result.fallback_reason is ReasonCode.NO_AIM_SOLUTION
    assert result.commands == []


def test_s4_uses_the_solution_for_the_selected_track_among_several() -> None:
    state = _armed_s4_state()  # selected_track_id=1, commanded already (5.0, 5.0)
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0
    )
    solutions = {1: (5.0, 5.0), 2: (40.0, -10.0), 3: (-20.0, 15.0)}
    result = engagement.step(state, tracks, telemetry, None, None, solutions, now=0.0)
    assert result.engagement is EngagementState.S5_ENGAGE
    assert result.commands == []


def test_s5_fires_and_advances_to_s6() -> None:
    state = make_state(engagement=EngagementState.S5_ENGAGE, selected_track_id=1)
    tracks = [make_track(track_id=1)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S6_ASSESS
    assert result.commands == [Fire(count=1)]
    assert result.attempts[1] == 1


def test_s6_waits_without_hit_result() -> None:
    state = make_state(engagement=EngagementState.S6_ASSESS, selected_track_id=1, attempts={1: 1})
    tracks = [make_track(track_id=1)]
    result = engagement.step(state, tracks, None, None, None, {}, now=0.0)
    assert result.engagement is EngagementState.S6_ASSESS
    assert result.commands == []


def test_s6_kill_returns_to_s1() -> None:
    state = make_state(engagement=EngagementState.S6_ASSESS, selected_track_id=1, attempts={1: 1})
    tracks = [make_track(track_id=1)]
    result = engagement.step(state, tracks, None, HitResult.KILL, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH
    assert result.selected_track_id is None


def test_s6_miss_retries_when_attempts_remain() -> None:
    state = make_state(engagement=EngagementState.S6_ASSESS, selected_track_id=1, attempts={1: 1})
    tracks = [make_track(track_id=1)]
    result = engagement.step(state, tracks, None, HitResult.MISS, None, {}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM


def test_s6_miss_skips_target_after_max_attempts() -> None:
    state = make_state(
        engagement=EngagementState.S6_ASSESS,
        selected_track_id=1,
        attempts={1: config.MAX_ENGAGEMENT_ATTEMPTS},
    )
    tracks = [make_track(track_id=1)]
    result = engagement.step(state, tracks, None, HitResult.MISS, None, {}, now=0.0)
    assert result.engagement is EngagementState.S1_SEARCH
    assert result.selected_track_id is None


def test_attempts_survive_across_frames_as_tracks_are_rebuilt() -> None:
    # Each tick the tracker hands us a brand new Track object (frozen,
    # rebuilt every frame) sharing the same track_id; the attempt counter
    # must survive in SystemState.attempts regardless.
    state = make_state(engagement=EngagementState.S5_ENGAGE, selected_track_id=1)
    for expected in (1, 2, 3):
        fresh_track = make_track(track_id=1)
        result = engagement.step(state, [fresh_track], None, None, None, {}, now=0.0)
        assert result.attempts[1] == expected
        state = _apply(state, result)
        state = dataclasses.replace(state, engagement=EngagementState.S5_ENGAGE)


def test_step_mutates_nothing() -> None:
    state = make_state(
        stage=Stage.STAGE_2,
        engagement=EngagementState.S4_AIM,
        selected_track_id=1,
        commanded_pan_deg=5.0,
        commanded_tilt_deg=5.0,
        attempts={1: 0},
    )
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    tracks_snapshot = list(tracks)
    attempts_snapshot = dict(state.attempts)
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0
    )

    engagement.step(state, tracks, telemetry, None, None, {1: (5.0, 5.0)}, now=0.0)

    assert tracks == tracks_snapshot
    assert state.attempts == attempts_snapshot
    assert state.engagement is EngagementState.S4_AIM
    assert state.selected_track_id == 1


def test_stage1_full_cycle_reaches_s4_s5_s6_via_step() -> None:
    state = make_state(
        stage=Stage.STAGE_1,
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S1_SEARCH,
    )
    track = make_track(track_id=1, status=TrackStatus.CONFIRMED)
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0
    )
    seen: set[EngagementState] = set()

    # S1 -> S2 -> S3 (display only, no arm yet)
    for _ in range(3):
        result = engagement.step(state, [track], telemetry, None, None, {}, now=0.0)
        state = _apply(state, result)
        seen.add(result.engagement)
    assert state.engagement is EngagementState.S3_TRACK

    # Operator arms: S3 -> S4, and the aim Goto gets acked over two ticks.
    armed = OperatorInput(arm_held=True)
    result = engagement.step(state, [track], telemetry, None, armed, {1: (5.0, 5.0)}, now=0.0)
    state = _apply(state, result)
    seen.add(result.engagement)
    assert state.engagement is EngagementState.S4_AIM

    result = engagement.step(state, [track], telemetry, None, armed, {1: (5.0, 5.0)}, now=0.0)
    state = _apply(state, result)
    seen.add(result.engagement)
    assert state.engagement is EngagementState.S4_AIM  # gates pass, waiting on fire_requested

    # Operator pulls the trigger: S4 -> S5 -> S6.
    fire = OperatorInput(arm_held=True, fire_requested=True)
    result = engagement.step(state, [track], telemetry, None, fire, {1: (5.0, 5.0)}, now=0.0)
    state = _apply(state, result)
    seen.add(result.engagement)
    assert state.engagement is EngagementState.S5_ENGAGE

    result = engagement.step(state, [track], telemetry, None, fire, {1: (5.0, 5.0)}, now=0.0)
    seen.add(result.engagement)
    assert result.engagement is EngagementState.S6_ASSESS
    assert result.commands == [Fire(count=1)]

    assert seen >= {
        EngagementState.S2_ACQUIRE,
        EngagementState.S3_TRACK,
        EngagementState.S4_AIM,
        EngagementState.S5_ENGAGE,
        EngagementState.S6_ASSESS,
    }


def test_stage1_does_not_advance_s4_to_s5_without_fire_requested() -> None:
    state = _armed_s4_state(stage=Stage.STAGE_1)
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    telemetry = make_telemetry(
        t=0.0, pan_deg=5.0, tilt_deg=5.0, target_pan_deg=5.0, target_tilt_deg=5.0
    )
    operator = OperatorInput(arm_held=True, fire_requested=False)
    result = engagement.step(state, tracks, telemetry, None, operator, {1: (5.0, 5.0)}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM
    assert result.commands == []


def test_stage1_fire_is_blocked_when_a_gate_fails() -> None:
    state = _armed_s4_state(stage=Stage.STAGE_1)
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    telemetry = make_telemetry(
        t=0.0,
        pan_deg=5.0,
        tilt_deg=5.0,
        target_pan_deg=5.0,
        target_tilt_deg=5.0,
        armed=False,
    )
    operator = OperatorInput(arm_held=True, fire_requested=True)
    result = engagement.step(state, tracks, telemetry, None, operator, {1: (5.0, 5.0)}, now=0.0)
    assert result.engagement is EngagementState.S4_AIM
    assert result.commands == []
    assert result.fallback_reason is ReasonCode.NOT_ARMED


def test_stage1_releasing_arm_mid_aim_aborts_to_s3() -> None:
    state = _armed_s4_state(stage=Stage.STAGE_1)
    tracks = [make_track(track_id=1, status=TrackStatus.CONFIRMED)]
    operator = OperatorInput(arm_held=False)
    result = engagement.step(state, tracks, make_telemetry(), None, operator, {}, now=0.0)
    assert result.engagement is EngagementState.S3_TRACK
    assert result.commanded_pan_deg is None
    assert result.commanded_tilt_deg is None


# --- deferring/excluding targets that cannot be engaged ---


def test_friendly_track_is_never_selected_for_engagement() -> None:
    state = make_state(engagement=EngagementState.S3_TRACK)
    friendly = make_track(
        track_id=1, status=TrackStatus.CONFIRMED, iff=IFF.FRIENDLY, risk_score=99.0
    )
    hostile = make_track(track_id=2, status=TrackStatus.CONFIRMED, iff=IFF.HOSTILE, risk_score=1.0)
    result = engagement.step(state, [friendly, hostile], None, None, None, {}, now=0.0)
    # The hostile track wins selection despite the far lower risk_score:
    # the friendly one was never a candidate at all, not merely outscored.
    assert result.selected_track_id == 2
    assert result.engagement is EngagementState.S4_AIM


def test_friendly_track_never_selected_even_as_sole_confirmed_track() -> None:
    state = make_state(engagement=EngagementState.S3_TRACK)
    friendly = make_track(track_id=1, status=TrackStatus.CONFIRMED, iff=IFF.FRIENDLY)
    result = engagement.step(state, [friendly], None, None, None, {}, now=0.0)
    assert result.selected_track_id is None
    assert result.engagement is EngagementState.S1_SEARCH


def _out_of_range_track(track_id: int, risk_score: float) -> Track:
    return make_track(
        track_id=track_id,
        status=TrackStatus.CONFIRMED,
        cls=TargetClass.F16,  # rule is (10.0, 15.0)
        range_m=1.0,  # well outside it
        risk_score=risk_score,
    )


def test_track_stuck_failing_gates_is_deferred_and_next_candidate_selected() -> None:
    stuck = _out_of_range_track(1, risk_score=90.0)
    other = make_track(track_id=2, status=TrackStatus.CONFIRMED, risk_score=10.0)
    telemetry = make_telemetry(t=0.0, target_pan_deg=5.0, target_tilt_deg=5.0)
    solutions = {1: (5.0, 5.0), 2: (5.0, 5.0)}

    state = _armed_s4_state(selected_track_id=1)
    result = engagement.step(state, [stuck, other], telemetry, None, None, solutions, now=0.0)
    assert result.engagement is EngagementState.S4_AIM
    assert result.fallback_reason is ReasonCode.RANGE_OUT_OF_BOUNDS
    assert result.gate_fail_since == 0.0
    assert result.deferred == {}

    state = _apply(state, result)
    now = config.GATE_REJECT_TIMEOUT_MS / 1000.0
    result = engagement.step(state, [stuck, other], telemetry, None, None, solutions, now=now)
    assert result.engagement is EngagementState.S3_TRACK
    assert result.selected_track_id is None
    assert 1 in result.deferred
    defer_until, reason = result.deferred[1]
    assert reason is ReasonCode.RANGE_OUT_OF_BOUNDS
    assert defer_until == now + config.DEFER_COOLDOWN_MS / 1000.0

    # Immediately after: S3 should pick the other, non-deferred candidate.
    state = _apply(state, result)
    result = engagement.step(state, [stuck, other], telemetry, None, None, solutions, now=now)
    assert result.selected_track_id == 2
    assert result.engagement is EngagementState.S4_AIM


def test_track_not_yet_at_timeout_stays_selected_in_s4() -> None:
    stuck = _out_of_range_track(1, risk_score=90.0)
    telemetry = make_telemetry(t=0.0, target_pan_deg=5.0, target_tilt_deg=5.0)
    solutions = {1: (5.0, 5.0)}

    state = _armed_s4_state(selected_track_id=1)
    result = engagement.step(state, [stuck], telemetry, None, None, solutions, now=0.0)
    state = _apply(state, result)

    just_under = config.GATE_REJECT_TIMEOUT_MS / 1000.0 - 0.01
    result = engagement.step(state, [stuck], telemetry, None, None, solutions, now=just_under)
    assert result.engagement is EngagementState.S4_AIM
    assert result.selected_track_id == 1
    assert result.deferred == {}


def test_deferred_track_becomes_selectable_again_after_cooldown() -> None:
    stuck = _out_of_range_track(1, risk_score=90.0)
    defer_until = config.DEFER_COOLDOWN_MS / 1000.0
    state = make_state(
        engagement=EngagementState.S3_TRACK,
        deferred={1: (defer_until, ReasonCode.RANGE_OUT_OF_BOUNDS)},
    )
    # Strictly before the deadline: still excluded.
    result = engagement.step(state, [stuck], None, None, None, {}, now=defer_until - 0.01)
    assert result.selected_track_id is None
    assert result.engagement is EngagementState.S1_SEARCH

    # At (or past) the deadline, the cooldown has fully elapsed: selectable.
    result = engagement.step(state, [stuck], None, None, None, {}, now=defer_until)
    assert result.selected_track_id == 1
    assert result.deferred == {}


def test_target_moving_into_range_engages_normally_after_deferral_expires() -> None:
    now = config.DEFER_COOLDOWN_MS / 1000.0 + 0.01
    state = make_state(
        engagement=EngagementState.S3_TRACK,
        deferred={1: (config.DEFER_COOLDOWN_MS / 1000.0, ReasonCode.RANGE_OUT_OF_BOUNDS)},
    )
    # The target has since moved into range.
    now_in_range = make_track(
        track_id=1, status=TrackStatus.CONFIRMED, cls=TargetClass.F16, range_m=12.0
    )
    telemetry = make_telemetry(t=now, target_pan_deg=5.0, target_tilt_deg=5.0)

    result = engagement.step(state, [now_in_range], None, None, None, {}, now=now)
    assert result.selected_track_id == 1
    assert result.engagement is EngagementState.S4_AIM

    state = _apply(state, result)
    result = engagement.step(state, [now_in_range], telemetry, None, None, {1: (5.0, 5.0)}, now=now)
    assert result.engagement is EngagementState.S5_ENGAGE
    assert result.fallback_reason is None


def test_all_candidates_deferred_or_excluded_returns_to_s1() -> None:
    now = 5.0
    friendly = make_track(track_id=1, status=TrackStatus.CONFIRMED, iff=IFF.FRIENDLY)
    deferred_track = make_track(track_id=2, status=TrackStatus.CONFIRMED)
    state = make_state(
        engagement=EngagementState.S3_TRACK,
        deferred={2: (now + 10.0, ReasonCode.RANGE_OUT_OF_BOUNDS)},
    )
    result = engagement.step(state, [friendly, deferred_track], None, None, None, {}, now=now)
    assert result.selected_track_id is None
    assert result.engagement is EngagementState.S1_SEARCH
