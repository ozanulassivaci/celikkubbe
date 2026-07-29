"""S1-S6 engagement machine.

``step`` is the autonomous engine used for Stage 2/3 and for Stage 1's
display-only S1-S3 progression. It never fires on its own in Stage 1 (the
table says S4/S5 there are operator-driven); ``request_manual_fire`` is the
separate, explicitly-invoked path a Stage 1 operator uses to fire, and it
still runs the shot through every gate in ``gates.py``.

Not fully pure: to keep ``step``'s signature exactly as specified (no
extra "updated state" return value), per-track bookkeeping
(``selected_track_id``, ``engagement_attempts``) is mutated in place on the
``SystemState``/``Track`` objects passed in. Both are plain (non-frozen)
dataclasses for exactly this reason, mirroring how ``tracking.py`` is
expected to update a ``Track`` every frame. No ``Command`` is ever executed
here, only returned for an outer layer to send.
"""

from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.commands import Command, Fire
from celikkubbe.core.gates import GateContext, evaluate_all
from celikkubbe.core.types import (
    EngagementState,
    HitResult,
    ReasonCode,
    Stage,
    SystemState,
    Telemetry,
    Track,
    TrackStatus,
)


def _find_track(tracks: list[Track], track_id: int | None) -> Track | None:
    if track_id is None:
        return None
    for track in tracks:
        if track.track_id == track_id:
            return track
    return None


def _pick_target(tracks: list[Track]) -> Track | None:
    candidates = [
        t
        for t in tracks
        if t.status == TrackStatus.CONFIRMED
        and t.engagement_attempts < config.MAX_ENGAGEMENT_ATTEMPTS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.risk_score)


def _build_gate_context(state: SystemState, telemetry: Telemetry, track: Track) -> GateContext:
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
        current_pan_deg=telemetry.pan_deg,
        current_tilt_deg=telemetry.tilt_deg,
        target_pan_deg=telemetry.target_pan_deg,
        target_tilt_deg=telemetry.target_tilt_deg,
    )


def step(
    state: SystemState,
    tracks: list[Track],
    telemetry: Telemetry | None,
    hit_result: HitResult | None,
    now: float,
) -> tuple[EngagementState, list[Command]]:
    current = state.engagement

    if current is EngagementState.S1_SEARCH:
        if any(t.status is not TrackStatus.LOST for t in tracks):
            return EngagementState.S2_ACQUIRE, []
        return EngagementState.S1_SEARCH, []

    if current is EngagementState.S2_ACQUIRE:
        if any(t.status is TrackStatus.CONFIRMED for t in tracks):
            return EngagementState.S3_TRACK, []
        if not tracks:
            return EngagementState.S1_SEARCH, []
        return EngagementState.S2_ACQUIRE, []

    if current is EngagementState.S3_TRACK:
        target = _find_track(tracks, state.selected_track_id)
        if target is None or target.status is TrackStatus.LOST:
            target = _pick_target(tracks)
        if target is None:
            state.selected_track_id = None
            return EngagementState.S1_SEARCH, []
        state.selected_track_id = target.track_id
        if state.stage is Stage.STAGE_1:
            return EngagementState.S3_TRACK, []
        return EngagementState.S4_AIM, []

    if current is EngagementState.S4_AIM:
        target = _find_track(tracks, state.selected_track_id)
        if target is None or target.status is TrackStatus.LOST:
            state.selected_track_id = None
            return EngagementState.S1_SEARCH, []
        if telemetry is None:
            return EngagementState.S4_AIM, []
        if evaluate_all(_build_gate_context(state, telemetry, target)):
            return EngagementState.S4_AIM, []
        return EngagementState.S5_ENGAGE, []

    if current is EngagementState.S5_ENGAGE:
        target = _find_track(tracks, state.selected_track_id)
        if target is None:
            state.selected_track_id = None
            return EngagementState.S1_SEARCH, []
        target.engagement_attempts += 1
        return EngagementState.S6_ASSESS, [Fire(count=1)]

    if current is EngagementState.S6_ASSESS:
        if hit_result is None:
            return EngagementState.S6_ASSESS, []
        target = _find_track(tracks, state.selected_track_id)
        if hit_result is HitResult.KILL:
            state.selected_track_id = None
            return EngagementState.S1_SEARCH, []
        if target is not None and target.engagement_attempts < config.MAX_ENGAGEMENT_ATTEMPTS:
            return EngagementState.S4_AIM, []
        state.selected_track_id = None
        return EngagementState.S1_SEARCH, []

    raise AssertionError(f"unreachable engagement state {current}")


def request_manual_fire(
    state: SystemState,
    tracks: list[Track],
    telemetry: Telemetry | None,
) -> tuple[EngagementState, list[Command], list[ReasonCode]]:
    """Stage 1 operator-initiated fire request.

    Always runs the request through ``gates.evaluate_all``; returns the
    failing reasons instead of a Fire command when any gate rejects it.
    """
    target = _find_track(tracks, state.selected_track_id)
    if target is None:
        return state.engagement, [], []
    if telemetry is None:
        return state.engagement, [], [ReasonCode.LINK_TIMEOUT]

    reasons = evaluate_all(_build_gate_context(state, telemetry, target))
    if reasons:
        return state.engagement, [], reasons

    target.engagement_attempts += 1
    return EngagementState.S6_ASSESS, [Fire(count=1)], []
