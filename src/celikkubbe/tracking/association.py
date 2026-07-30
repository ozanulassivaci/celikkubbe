"""Detection-to-track matching.

Greedy nearest-match is tempting but produces identity swaps when two
targets cross — expected behaviour on Stage 2's three parallel lanes, not
an edge case. Instead: gate out pairs whose centroids are implausibly far
apart, build an IoU-distance cost matrix from what remains, and solve it
optimally with scipy's Hungarian algorithm implementation.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from celikkubbe.core.types import BoundingBox

# Normalised units. A track predicted to have moved further than this in
# one tick cannot plausibly be the same target as a candidate detection;
# gating these out before optimisation prevents the Hungarian solver from
# ever considering a physically implausible match.
MAX_ASSOCIATION_DISPLACEMENT = 0.25

# Cost assigned to a gated-out (ineligible) pair. Deliberately distinct
# from any real cost (1 - IoU is always in [0, 1]): a fast-moving or
# small target can legitimately have zero IoU with its own prediction
# while still being the obviously correct match, and that must not look
# identical to a pair the displacement gate rejected outright.
_INELIGIBLE_COST = 10.0


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _centroid(bbox: BoundingBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _centroid_distance(a: BoundingBox, b: BoundingBox) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return math.hypot(ax - bx, ay - by)


def associate(
    predicted_bboxes: list[BoundingBox],
    detection_bboxes: list[BoundingBox],
    max_displacement: float = MAX_ASSOCIATION_DISPLACEMENT,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted track boxes against this frame's detection boxes.

    Returns ``(matches, unmatched_track_indices, unmatched_detection_indices)``
    where ``matches`` is a list of ``(track_index, detection_index)`` pairs.
    """
    n_tracks = len(predicted_bboxes)
    n_dets = len(detection_bboxes)
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    cost = np.full((n_tracks, n_dets), _INELIGIBLE_COST, dtype=float)
    eligible = np.zeros((n_tracks, n_dets), dtype=bool)
    for i, track_bbox in enumerate(predicted_bboxes):
        for j, det_bbox in enumerate(detection_bboxes):
            if _centroid_distance(track_bbox, det_bbox) > max_displacement:
                continue
            eligible[i, j] = True
            cost[i, j] = 1.0 - iou(track_bbox, det_bbox)

    row_idx, col_idx = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for i, j in zip(row_idx, col_idx, strict=True):
        if not eligible[i, j]:
            continue
        matches.append((int(i), int(j)))
        matched_tracks.add(int(i))
        matched_dets.add(int(j))

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets
