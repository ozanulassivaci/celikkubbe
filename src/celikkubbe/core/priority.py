"""Threat scoring and hysteresis-stabilised ordering for the target list."""

from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.types import IFF, TargetClass, Track


def filter_engageable(tracks: list[Track]) -> list[Track]:
    """Tracks eligible for autonomous engagement selection.

    FRIENDLY is fail-safe and permanent — IFF voting never un-commits
    from it once earned — so a friendly track is excluded here rather
    than merely scored low; it must never enter the candidate list, let
    alone be selected. It still appears in ``SystemState.tracks`` and
    still gets a risk_score like any other track: the UI shows
    friendlies with blue boxes and counts them in the classification
    summary. This exclusion is scoped to engagement selection only.
    """
    return [t for t in tracks if t.iff is not IFF.FRIENDLY]


def compute_risk_score(
    cls: TargetClass | None,
    range_m: float | None,
    confidence: float,
) -> float:
    """Compute a 0-100 risk score for a target.

    The score is a weighted sum of three 0-100 sub-scores (weights from
    ``config.WEIGHT_CLASS`` / ``WEIGHT_RANGE`` / ``WEIGHT_CONFIDENCE``,
    which add up to 1.0):

    - class score: ``config.CLASS_PRIORITY[cls]``, already 0-100; unknown or
      unlisted classes score 0.
    - range score: linear falloff from 100 at range 0 to 0 at
      ``config.MAX_ENGAGEMENT_RANGE_M``, i.e. closer is higher risk. When
      ``range_m`` is unknown (no depth), ``config.DEFAULT_RANGE_M`` is used
      as the parallax assumption.
    - confidence score: ``confidence`` (0-1) scaled to 0-100.

    The weighted sum is clamped to [0, 100] to absorb rounding.
    """
    class_score = config.CLASS_PRIORITY.get(cls, 0)

    effective_range = range_m if range_m is not None else config.DEFAULT_RANGE_M
    range_fraction = effective_range / config.MAX_ENGAGEMENT_RANGE_M
    range_score = 100.0 * (1.0 - min(max(range_fraction, 0.0), 1.0))

    confidence_score = 100.0 * min(max(confidence, 0.0), 1.0)

    score = (
        config.WEIGHT_CLASS * class_score
        + config.WEIGHT_RANGE * range_score
        + config.WEIGHT_CONFIDENCE * confidence_score
    )
    return min(max(score, 0.0), 100.0)


def order_track_ids(
    tracks: list[Track],
    previous_order: list[int],
    hysteresis: float = config.REORDER_HYSTERESIS,
) -> list[int]:
    """Order track ids by risk score with hysteresis against re-sort thrash.

    Tracks already present in ``previous_order`` keep their relative
    position unless the score gap to a neighbour exceeds ``hysteresis``
    (a fraction of the 0-100 score range). New tracks are inserted sorted
    by score. This keeps the UI target list from reshuffling every frame
    when two scores are nearly tied.
    """
    scores = {t.track_id: t.risk_score for t in tracks}
    current_ids = set(scores)

    order = [tid for tid in previous_order if tid in current_ids]
    known = set(order)
    new_ids = sorted(
        (tid for tid in current_ids if tid not in known),
        key=lambda tid: scores[tid],
        reverse=True,
    )
    order.extend(new_ids)

    threshold = hysteresis * 100.0
    changed = True
    while changed:
        changed = False
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            if scores[b] - scores[a] > threshold:
                order[i], order[i + 1] = b, a
                changed = True
    return order
