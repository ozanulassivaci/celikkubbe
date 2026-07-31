"""S1-S6 engagement machine.

Pure function: ``step`` reads ``state`` and the tick's inputs and returns a
``StepResult`` describing what changed. It never mutates ``state``, any
``Track``, or executes a ``Command`` — the outer control loop applies the
result (typically via ``dataclasses.replace()`` on ``SystemState``) and
sends the commands.

Stage 1 runs through the same S1-S6 machine as Stage 2/3; only the S4 -> S5
trigger differs. Autonomous stages advance the instant every gate in
``gates.py`` passes. Stage 1 additionally requires the operator to hold the
arm switch (a continuous dead-man switch: releasing it before the shot is
taken aborts back to S3) and to request fire on that tick. There is exactly
one path to a ``Fire`` command, so "never fire without passing every gate"
only needs to be true in one place.

``step()`` chooses which track to engage (S3 -> S4, via prioritisation), so
the aim solution for the *selected* track cannot be computed by the outer
layer ahead of time without predicting that choice — and both ways of
predicting it are broken (a stale previous-tick selection lags by a frame;
re-running priority.py in the outer layer duplicates the selection logic
and can silently diverge from step()'s own hysteresis state). Instead the
outer layer computes a solution for every confirmed track and hands over
the whole map; step() looks up the one it selected.
"""

from __future__ import annotations

from dataclasses import dataclass

from celikkubbe.core import config
from celikkubbe.core.commands import Command, Fire, Goto
from celikkubbe.core.gates import GateContext, evaluate_all
from celikkubbe.core.types import (
    EngagementState,
    HitResult,
    OperatorInput,
    ReasonCode,
    Stage,
    SystemState,
    Telemetry,
    Track,
    TrackStatus,
)


@dataclass(frozen=True)
class StepResult:
    engagement: EngagementState
    commands: list[Command]
    selected_track_id: int | None
    attempts: dict[int, int]
    fallback_reason: ReasonCode | None
    commanded_pan_deg: float | None
    commanded_tilt_deg: float | None


def _find_track(tracks: list[Track], track_id: int | None) -> Track | None:
    if track_id is None:
        return None
    for track in tracks:
        if track.track_id == track_id:
            return track
    return None


def _pick_target(tracks: list[Track], attempts: dict[int, int]) -> Track | None:
    candidates = [
        t
        for t in tracks
        if t.status == TrackStatus.CONFIRMED
        and attempts.get(t.track_id, 0) < config.MAX_ENGAGEMENT_ATTEMPTS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.risk_score)


def _build_gate_context(
    state: SystemState,
    telemetry: Telemetry,
    track: Track,
    commanded_pan_deg: float | None,
    commanded_tilt_deg: float | None,
) -> GateContext:
    return GateContext(
        stage=state.stage,
        mode=state.mode,
        armed=telemetry.armed,
        estop=telemetry.estop,
        position_valid=telemetry.position_valid,
        iff=track.iff,
        cls=track.cls,
        range_m=track.range_m,
        confidence=track.confidence,
        target_pan_deg=telemetry.target_pan_deg,
        target_tilt_deg=telemetry.target_tilt_deg,
        commanded_pan_deg=commanded_pan_deg,
        commanded_tilt_deg=commanded_tilt_deg,
        motion_complete=telemetry.motion_complete,
        driver_alarm_pan=telemetry.driver_alarm_pan,
        driver_alarm_tilt=telemetry.driver_alarm_tilt,
    )


def step(
    state: SystemState,
    tracks: list[Track],
    telemetry: Telemetry | None,
    hit_result: HitResult | None,
    operator: OperatorInput | None,
    aim_solutions: dict[int, tuple[float, float]],
    now: float,
) -> StepResult:
    current = state.engagement
    attempts = dict(state.attempts)
    selected_track_id = state.selected_track_id
    commanded_pan_deg = state.commanded_pan_deg
    commanded_tilt_deg = state.commanded_tilt_deg
    fallback_reason: ReasonCode | None = None
    commands: list[Command] = []
    manual = state.stage is Stage.STAGE_1
    armed_by_operator = manual and operator is not None and operator.arm_held

    def finish(new_engagement: EngagementState) -> StepResult:
        return StepResult(
            engagement=new_engagement,
            commands=commands,
            selected_track_id=selected_track_id,
            attempts=attempts,
            fallback_reason=fallback_reason,
            commanded_pan_deg=commanded_pan_deg,
            commanded_tilt_deg=commanded_tilt_deg,
        )

    if current is EngagementState.S1_SEARCH:
        if any(t.status is not TrackStatus.LOST for t in tracks):
            return finish(EngagementState.S2_ACQUIRE)
        return finish(EngagementState.S1_SEARCH)

    if current is EngagementState.S2_ACQUIRE:
        if any(t.status is TrackStatus.CONFIRMED for t in tracks):
            return finish(EngagementState.S3_TRACK)
        if not tracks:
            return finish(EngagementState.S1_SEARCH)
        return finish(EngagementState.S2_ACQUIRE)

    if current is EngagementState.S3_TRACK:
        target = _find_track(tracks, selected_track_id)
        if target is not None and target.status is TrackStatus.LOST:
            target = None
        if manual and operator is not None and operator.manual_target_id is not None:
            picked = _find_track(tracks, operator.manual_target_id)
            if picked is not None and picked.status is not TrackStatus.LOST:
                target = picked
        if target is None:
            target = _pick_target(tracks, attempts)
        if target is None:
            selected_track_id = None
            commanded_pan_deg = None
            commanded_tilt_deg = None
            return finish(EngagementState.S1_SEARCH)
        selected_track_id = target.track_id
        if manual and not armed_by_operator:
            return finish(EngagementState.S3_TRACK)
        return finish(EngagementState.S4_AIM)

    if current is EngagementState.S4_AIM:
        target = _find_track(tracks, selected_track_id)
        if target is None or target.status is TrackStatus.LOST:
            selected_track_id = None
            commanded_pan_deg = None
            commanded_tilt_deg = None
            return finish(EngagementState.S1_SEARCH)

        if manual and not armed_by_operator:
            commanded_pan_deg = None
            commanded_tilt_deg = None
            return finish(EngagementState.S3_TRACK)

        solution = aim_solutions.get(target.track_id)
        if solution is None:
            fallback_reason = ReasonCode.NO_AIM_SOLUTION
            return finish(EngagementState.S4_AIM)

        new_pan, new_tilt = solution
        unacked = (
            commanded_pan_deg is None
            or commanded_tilt_deg is None
            or abs(new_pan - commanded_pan_deg) > config.SETPOINT_ACK_EPSILON_DEG
            or abs(new_tilt - commanded_tilt_deg) > config.SETPOINT_ACK_EPSILON_DEG
        )
        if unacked:
            commands.append(
                Goto(
                    az_deg=new_pan,
                    el_deg=new_tilt,
                    max_vel_dps=config.AIM_MAX_VEL_DPS,
                    max_accel_dps2=config.AIM_MAX_ACCEL_DPS2,
                )
            )
            commanded_pan_deg = new_pan
            commanded_tilt_deg = new_tilt

        if telemetry is None:
            return finish(EngagementState.S4_AIM)

        ctx = _build_gate_context(state, telemetry, target, commanded_pan_deg, commanded_tilt_deg)
        reasons = evaluate_all(ctx)
        if reasons:
            fallback_reason = reasons[0]
            return finish(EngagementState.S4_AIM)

        if manual:
            if operator is not None and operator.fire_requested:
                return finish(EngagementState.S5_ENGAGE)
            return finish(EngagementState.S4_AIM)

        return finish(EngagementState.S5_ENGAGE)

    if current is EngagementState.S5_ENGAGE:
        target = _find_track(tracks, selected_track_id)
        if target is None:
            selected_track_id = None
            commanded_pan_deg = None
            commanded_tilt_deg = None
            return finish(EngagementState.S1_SEARCH)
        attempts[target.track_id] = attempts.get(target.track_id, 0) + 1
        commands.append(Fire(count=1))
        return finish(EngagementState.S6_ASSESS)

    if current is EngagementState.S6_ASSESS:
        if hit_result is None:
            return finish(EngagementState.S6_ASSESS)
        target = _find_track(tracks, selected_track_id)
        if hit_result is HitResult.KILL:
            selected_track_id = None
            commanded_pan_deg = None
            commanded_tilt_deg = None
            return finish(EngagementState.S1_SEARCH)
        shots_so_far = attempts.get(selected_track_id, 0) if selected_track_id is not None else 0
        if target is not None and shots_so_far < config.MAX_ENGAGEMENT_ATTEMPTS:
            return finish(EngagementState.S4_AIM)
        selected_track_id = None
        commanded_pan_deg = None
        commanded_tilt_deg = None
        return finish(EngagementState.S1_SEARCH)

    raise AssertionError(f"unreachable engagement state {current}")
