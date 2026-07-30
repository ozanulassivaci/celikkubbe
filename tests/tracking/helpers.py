"""Shared construction helpers for tracking tests. Not collected by pytest."""

from __future__ import annotations

from celikkubbe.core.types import IFF, BoundingBox, Detection, Layer, TargetClass


def make_detection(
    cx: float,
    cy: float,
    w: float = 0.05,
    h: float = 0.05,
    cls: TargetClass | None = None,
    confidence: float = 0.8,
    range_m: float | None = None,
    range_source: str = "none",
    iff: IFF = IFF.UNKNOWN,
) -> Detection:
    bbox: BoundingBox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return Detection(
        bbox=bbox,
        cls=cls,
        confidence=confidence,
        range_m=range_m,
        range_source=range_source,
        source_layer=Layer.L2,
        iff=iff,
    )


def bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
