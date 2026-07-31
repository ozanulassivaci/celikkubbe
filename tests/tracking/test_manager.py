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


def test_friendly_frame_flips_a_hostile_track_to_friendly() -> None:
    # Association gates asymmetrically: a HOSTILE-voted track is only
    # soft-gated, so a single conflicting-colour detection can still land
    # on it (there's nothing better available) rather than being forced
    # onto a brand new track. Fail-safe voting then does its job.
    manager = TrackManager(FakeClock(), confirm_frames=1)
    tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.HOSTILE)], now=0.0)
    hostile_id = tracks[0].track_id

    tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.FRIENDLY)], now=1.0)
    assert len(tracks) == 1
    assert tracks[0].track_id == hostile_id
    assert tracks[0].iff is IFF.FRIENDLY


def test_hostile_track_survives_a_single_blue_frame_with_id_and_attempts_intact() -> None:
    # The whole point of the soft gate: identity survives a one-frame
    # colour misread (specular highlight, motion blur, momentary overlap
    # with another model), so FSM-owned bookkeeping keyed by track_id
    # (engagement_attempts, deferrals — neither lives in tracking.py) is
    # never invalidated by a spurious ID change.
    manager = TrackManager(FakeClock(), confirm_frames=1)
    tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.HOSTILE)], now=0.0)
    track_id = tracks[0].track_id
    simulated_engagement_attempts = {track_id: 2}

    tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.FRIENDLY)], now=1.0)

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
    assert simulated_engagement_attempts.get(tracks[0].track_id) == 2


def test_friendly_track_never_reverts_to_hostile() -> None:
    # Once fail-safe voting commits a track to FRIENDLY, it is hard-gated:
    # a HOSTILE-coloured detection can never land on it again to erode
    # the vote, so the commitment cannot be undone by more bad readings.
    # track_lost_ms is generous here since the point under test is voting
    # permanence, not coasting/lost lifecycle timing — the friendly track
    # never gets re-matched (by design) and would otherwise time out.
    manager = TrackManager(FakeClock(), confirm_frames=1, track_lost_ms=100_000.0)
    tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.FRIENDLY)], now=0.0)
    friendly_id = tracks[0].track_id

    for i in range(10):
        tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.HOSTILE)], now=float(i + 1))
        friendly_track = next(t for t in tracks if t.track_id == friendly_id)
        assert friendly_track.iff is IFF.FRIENDLY


def test_friendly_track_never_absorbs_a_hostile_detection() -> None:
    manager = TrackManager(FakeClock(), confirm_frames=1)
    tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.FRIENDLY)], now=0.0)
    friendly_id = tracks[0].track_id

    # A hostile-coloured detection at the exact same position must not
    # match the existing friendly track; it must start a new one.
    tracks = manager.update([make_detection(0.5, 0.5, iff=IFF.HOSTILE)], now=1.0)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert friendly_id in ids
    new_track = next(t for t in tracks if t.track_id != friendly_id)
    assert new_track.iff is IFF.HOSTILE


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
