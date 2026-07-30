from __future__ import annotations

from celikkubbe.tracking.association import associate, iou


def test_iou_identical_boxes_is_one() -> None:
    box = (0.1, 0.1, 0.3, 0.3)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    a = (0.0, 0.0, 0.1, 0.1)
    b = (0.5, 0.5, 0.6, 0.6)
    assert iou(a, b) == 0.0


def test_associate_matches_overlapping_pair() -> None:
    predicted = [(0.1, 0.1, 0.2, 0.2)]
    detections = [(0.11, 0.11, 0.21, 0.21)]
    matches, unmatched_tracks, unmatched_dets = associate(predicted, detections)
    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_dets == []


def test_associate_gates_out_implausibly_distant_pair() -> None:
    predicted = [(0.0, 0.0, 0.05, 0.05)]
    detections = [(0.9, 0.9, 0.95, 0.95)]
    matches, unmatched_tracks, unmatched_dets = associate(predicted, detections)
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_dets == [0]


def test_associate_handles_empty_inputs() -> None:
    assert associate([], []) == ([], [], [])
    assert associate([(0.0, 0.0, 0.1, 0.1)], []) == ([], [0], [])
    assert associate([], [(0.0, 0.0, 0.1, 0.1)]) == ([], [], [0])


def test_associate_prefers_globally_optimal_assignment() -> None:
    # Track 0 is closer to detection 1 by IoU, and track 1 to detection 0;
    # a greedy nearest-first match would get this wrong on the first pick.
    predicted = [
        (0.40, 0.40, 0.50, 0.50),
        (0.41, 0.41, 0.51, 0.51),
    ]
    detections = [
        (0.41, 0.41, 0.51, 0.51),
        (0.40, 0.40, 0.50, 0.50),
    ]
    matches, _, _ = associate(predicted, detections)
    assert set(matches) == {(0, 1), (1, 0)}
