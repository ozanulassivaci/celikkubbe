from __future__ import annotations

import dataclasses

import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import IFF, TargetClass, TrackStatus
from celikkubbe.tracking.manager import TrackManager

from .helpers import bbox_center, make_detection


def test_new_unmatched_detection_creates_tentative_track() -> None:
    manager = TrackManager(FakeClock())
    tracks = manager.update([make_detection(0.5, 0.5)], now=0.0)
    assert len(tracks) == 1
    assert tracks[0].status is TrackStatus.TENTATIVE
    assert tracks[0].track_id == 1


def test_track_confirmed_at_exactly_acquire_frames() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=5)
    for i in range(4):
        tracks = manager.update([make_detection(0.5, 0.5)], now=float(i))
        assert tracks[0].status is TrackStatus.TENTATIVE

    tracks = manager.update([make_detection(0.5, 0.5)], now=4.0)
    assert tracks[0].status is TrackStatus.CONFIRMED
    assert tracks[0].frames_confirmed == 5


def test_track_coasts_on_missed_frame_and_predicts_forward() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=3, track_lost_ms=5000)
    for i in range(3):
        manager.update([make_detection(0.5, 0.5)], now=float(i))

    tracks = manager.update([], now=3.0)
    assert len(tracks) == 1
    assert tracks[0].status is TrackStatus.COASTING


def test_track_lost_and_removed_after_track_lost_ms() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1, track_lost_ms=1000)
    manager.update([make_detection(0.5, 0.5)], now=0.0)

    tracks = manager.update([], now=1.5)
    assert len(tracks) == 1
    assert tracks[0].status is TrackStatus.LOST

    tracks = manager.update([], now=2.0)
    assert tracks == []


def test_track_ids_stable_as_target_moves() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=100)
    x = 0.1
    track_id = None
    for i in range(10):
        tracks = manager.update([make_detection(x, 0.5)], now=float(i) * 0.033)
        assert len(tracks) == 1
        if track_id is None:
            track_id = tracks[0].track_id
        assert tracks[0].track_id == track_id
        x += 0.02


def test_track_ids_never_reused() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1, track_lost_ms=10)
    manager.update([make_detection(0.2, 0.5)], now=0.0)
    manager.update([], now=1.0)  # first track times out and is purged
    tracks = manager.update([make_detection(0.2, 0.5)], now=1.1)
    assert tracks[0].track_id == 2


def test_crossing_targets_do_not_swap_ids() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=3)
    left_x, right_x = 0.2, 0.8
    speed = 0.06
    now = 0.0

    ids_by_side: dict[str, int] = {}
    for _ in range(4):
        tracks = manager.update(
            [make_detection(left_x, 0.5), make_detection(right_x, 0.5)], now=now
        )
        left_x += speed
        right_x -= speed
        now += 1.0

    by_velocity = sorted(tracks, key=lambda t: t.velocity[0])
    ids_by_side["was_moving_right"] = by_velocity[-1].track_id
    ids_by_side["was_moving_left"] = by_velocity[0].track_id

    # Continue through and past the crossing point.
    for _ in range(6):
        tracks = manager.update(
            [make_detection(left_x, 0.5), make_detection(right_x, 0.5)], now=now
        )
        left_x += speed
        right_x -= speed
        now += 1.0

    moving_right_track = next(t for t in tracks if t.track_id == ids_by_side["was_moving_right"])
    moving_left_track = next(t for t in tracks if t.track_id == ids_by_side["was_moving_left"])
    assert moving_right_track.velocity[0] > 0
    assert moving_left_track.velocity[0] < 0


def test_class_voting_resists_a_single_misclassified_frame() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1)
    now = 0.0
    for _ in range(6):
        manager.update([make_detection(0.5, 0.5, cls=TargetClass.UAV)], now=now)
        now += 1.0

    tracks = manager.update([make_detection(0.5, 0.5, cls=TargetClass.HELICOPTER)], now=now)
    assert tracks[0].cls is TargetClass.UAV


def test_friendly_frame_within_window_keeps_track_friendly() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1)
    now = 0.0
    manager.update([make_detection(0.5, 0.5, iff=IFF.FRIENDLY)], now=now)
    for _ in range(4):
        now += 1.0
        tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.HOSTILE)], now=now)
        assert tracks[0].iff is IFF.FRIENDLY


def test_friendly_frame_ages_out_of_the_window_eventually() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1)
    now = 0.0
    manager.update([make_detection(0.5, 0.5, iff=IFF.FRIENDLY)], now=now)
    tracks = []
    for _ in range(10):
        now += 1.0
        tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.HOSTILE)], now=now)
    assert tracks[0].iff is IFF.HOSTILE


def test_emitted_tracks_are_frozen() -> None:
    manager = TrackManager(FakeClock())
    tracks = manager.update([make_detection(0.5, 0.5)], now=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tracks[0].risk_score = 99.0  # type: ignore[misc]


def test_range_fusion_reports_winning_source() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1)
    tracks = manager.update([make_detection(0.5, 0.5, range_m=6.0, range_source="depth")], now=0.0)
    assert tracks[0].range_source == "depth"
    assert tracks[0].range_m == pytest.approx(6.0, abs=0.5)


def test_range_stays_none_until_a_measurement_exists() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1)
    tracks = manager.update([make_detection(0.5, 0.5)], now=0.0)
    assert tracks[0].range_m is None
    assert tracks[0].range_source == "none"


def test_end_to_end_source_detector_tracker_priority_orders_sensibly() -> None:
    from celikkubbe.core.priority import compute_risk_score, order_track_ids
    from celikkubbe.vision.l2_color import ColorDetector
    from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig

    clock = FakeClock()
    source = SyntheticSource(
        clock, SyntheticSourceConfig(width=320, height=240, num_targets=2, speed=0.05)
    )
    source.start()
    detector = ColorDetector()
    manager = TrackManager(clock, confirm_frames=3)

    tracks = []
    for _ in range(6):
        clock.advance(1.0 / 30.0)
        frame = source.read()
        detections, _ = detector.detect(frame)
        tracks = manager.update(detections, now=clock.now())

    assert len(tracks) == 2
    scored = [
        dataclasses.replace(t, risk_score=compute_risk_score(t.cls, t.range_m, t.confidence))
        for t in tracks
    ]
    order = order_track_ids(scored, previous_order=[])
    assert set(order) == {t.track_id for t in scored}
    assert all(isinstance(tid, int) for tid in order)


def test_bbox_center_helper_matches_manual_calculation() -> None:
    assert bbox_center((0.0, 0.0, 0.2, 0.4)) == (0.1, 0.2)
