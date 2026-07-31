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


def test_associate_hard_gates_out_mismatched_group_despite_perfect_overlap() -> None:
    # Same bbox, perfect IoU — but a track in a hard-gated group must not
    # absorb a mismatched detection regardless of how well their boxes
    # overlap.
    predicted = [(0.1, 0.1, 0.2, 0.2)]
    detections = [(0.1, 0.1, 0.2, 0.2)]
    matches, unmatched_tracks, unmatched_dets = associate(
        predicted,
        detections,
        track_groups=["red"],
        detection_groups=["blue"],
        hard_gate_groups=frozenset({"red"}),
    )
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_dets == [0]


def test_associate_allows_matching_group_with_perfect_overlap() -> None:
    predicted = [(0.1, 0.1, 0.2, 0.2)]
    detections = [(0.1, 0.1, 0.2, 0.2)]
    matches, _, _ = associate(predicted, detections, track_groups=["red"], detection_groups=["red"])
    assert matches == [(0, 0)]


def test_associate_without_groups_ignores_colour_entirely() -> None:
    # Backwards compatible: omitting both group lists means no colour gate.
    predicted = [(0.1, 0.1, 0.2, 0.2)]
    detections = [(0.1, 0.1, 0.2, 0.2)]
    matches, _, _ = associate(predicted, detections)
    assert matches == [(0, 0)]


def test_associate_soft_gates_a_mismatched_group_by_default() -> None:
    # No hard_gate_groups given: a mismatched colour is not forbidden,
    # only penalised — the only candidate still wins the match rather
    # than being left unmatched.
    predicted = [(0.1, 0.1, 0.2, 0.2)]
    detections = [(0.1, 0.1, 0.2, 0.2)]
    matches, unmatched_tracks, unmatched_dets = associate(
        predicted,
        detections,
        track_groups=["red"],
        detection_groups=["blue"],
        group_mismatch_cost=0.5,
    )
    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_dets == []


def test_associate_soft_gate_prefers_same_group_candidate() -> None:
    # Track 0 is a non-hard-gated group ("hostile"). Detection 0 is a
    # perfect-overlap mismatch; detection 1 is a same-group candidate with
    # slightly worse overlap. The mismatch penalty should tip the balance
    # towards the same-coloured, slightly-worse-overlap detection.
    predicted = [(0.10, 0.10, 0.20, 0.20)]
    detections = [
        (0.10, 0.10, 0.20, 0.20),  # perfect overlap, wrong colour
        (0.11, 0.11, 0.21, 0.21),  # good overlap, right colour
    ]
    matches, _, _ = associate(
        predicted,
        detections,
        track_groups=["hostile"],
        detection_groups=["friendly", "hostile"],
        group_mismatch_cost=0.5,
    )
    assert matches == [(0, 1)]


def test_associate_hard_gate_group_still_matches_its_own_colour() -> None:
    predicted = [(0.1, 0.1, 0.2, 0.2)]
    detections = [(0.1, 0.1, 0.2, 0.2)]
    matches, _, _ = associate(
        predicted,
        detections,
        track_groups=["friendly"],
        detection_groups=["friendly"],
        hard_gate_groups=frozenset({"friendly"}),
    )
    assert matches == [(0, 0)]
