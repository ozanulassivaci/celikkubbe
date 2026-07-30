from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.priority import compute_risk_score, order_track_ids
from celikkubbe.core.types import IFF, TargetClass, Track, TrackStatus


def make_track(track_id: int, risk_score: float) -> Track:
    return Track(
        track_id=track_id,
        cls=TargetClass.UAV,
        confidence=0.9,
        range_m=5.0,
        iff=IFF.HOSTILE,
        bbox=(0.0, 0.0, 0.1, 0.1),
        velocity=(0.0, 0.0),
        status=TrackStatus.CONFIRMED,
        risk_score=risk_score,
        frames_confirmed=5,
        last_seen_t=0.0,
    )


def test_risk_score_higher_for_higher_class_priority() -> None:
    f16_score = compute_risk_score(TargetClass.F16, 5.0, 0.9)
    balloon_score = compute_risk_score(TargetClass.BALLOON, 5.0, 0.9)
    assert f16_score > balloon_score


def test_risk_score_higher_for_closer_range() -> None:
    close = compute_risk_score(TargetClass.UAV, 1.0, 0.9)
    far = compute_risk_score(TargetClass.UAV, 14.0, 0.9)
    assert close > far


def test_risk_score_uses_default_range_when_unknown() -> None:
    known_default = compute_risk_score(TargetClass.UAV, config.DEFAULT_RANGE_M, 0.9)
    unknown = compute_risk_score(TargetClass.UAV, None, 0.9)
    assert known_default == unknown


def test_risk_score_bounded_0_to_100() -> None:
    assert 0.0 <= compute_risk_score(TargetClass.F16, 0.0, 1.0) <= 100.0
    assert 0.0 <= compute_risk_score(None, 100.0, 0.0) <= 100.0


def test_order_preserves_previous_order_within_hysteresis() -> None:
    tracks = [make_track(1, 50.0), make_track(2, 52.0)]
    order = order_track_ids(tracks, previous_order=[1, 2])
    assert order == [1, 2]


def test_order_swaps_when_gap_exceeds_hysteresis() -> None:
    tracks = [make_track(1, 30.0), make_track(2, 90.0)]
    order = order_track_ids(tracks, previous_order=[1, 2])
    assert order == [2, 1]


def test_order_inserts_new_track_by_score() -> None:
    tracks = [make_track(1, 40.0), make_track(2, 90.0)]
    order = order_track_ids(tracks, previous_order=[1])
    assert order == [2, 1]
