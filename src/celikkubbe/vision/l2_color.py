"""Classic-CV colour detector — the permanent L2 fallback.

This runs whenever the (future) YOLO L1 pipeline's health degrades, not
just during bring-up, so it is built to production standard rather than
as a scaffold. Output is always ``cls=None``: this layer cannot classify
a target and must never pretend otherwise. IFF is always ``UNKNOWN`` for
the same reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from celikkubbe.core.config import MIN_TARGET_PX
from celikkubbe.core.types import (
    IFF,
    BoundingBox,
    CameraIntrinsics,
    Detection,
    Frame,
    Layer,
    RangeSource,
)

DEPTH_SAMPLE_WINDOW_PX = 5  # odd window sampled around the centroid for depth range


@dataclass
class ColorDetectorConfig:
    hue_ranges: tuple[tuple[int, int], ...] = ((0, 10), (170, 180))
    sat_min: int = 120
    val_min: int = 80
    morph_kernel: int = 5
    min_area_px: int = 12
    circularity_min: float = 0.70
    roi: BoundingBox | None = None  # normalised (x1, y1, x2, y2)
    lowest_circle_rule: bool = True
    known_diameter_m: float = 0.14  # balloon, for size-based range


@dataclass(frozen=True)
class Contour:
    """One contour this frame considered, accepted or not — for the tuning panel."""

    bbox_px: tuple[int, int, int, int]  # x, y, w, h in full-frame pixels
    area_px: float
    circularity: float
    accepted: bool
    reject_reason: str | None = None


@dataclass(frozen=True)
class DebugMasks:
    hsv_mask: np.ndarray
    morphed_mask: np.ndarray
    contours: tuple[Contour, ...]


@dataclass
class _Circle:
    cx: float
    cy: float
    radius: float
    area: float
    circularity: float
    bbox_px: tuple[int, int, int, int]


def is_under_resolved(
    bbox: BoundingBox, intrinsics: CameraIntrinsics, min_target_px: int = MIN_TARGET_PX
) -> bool:
    """True when a normalised bbox's pixel diameter is below MIN_TARGET_PX.

    Computed from intrinsics.width rather than a hardcoded resolution
    table, so the same check is valid whether the frame came from a
    640x480 webcam or a 1920x1080 D435i.
    """
    diameter_px = (bbox[2] - bbox[0]) * intrinsics.width
    return diameter_px < min_target_px


def _group_lowest_circle(circles: list[_Circle]) -> list[_Circle]:
    """Where circles overlap horizontally (stacked vertically), keep the
    bottommost — the aircraft model sits above the balloon, never below it.
    """
    groups: list[list[_Circle]] = []
    for circle in sorted(circles, key=lambda c: c.cx):
        for group in groups:
            if any(abs(circle.cx - other.cx) < (circle.radius + other.radius) for other in group):
                group.append(circle)
                break
        else:
            groups.append([circle])
    return [max(group, key=lambda c: c.cy) for group in groups]


def _depth_range_m(depth: np.ndarray, cx_px: int, cy_px: int, window: int) -> float | None:
    half = window // 2
    y0, y1 = max(0, cy_px - half), min(depth.shape[0], cy_px + half + 1)
    x0, x1 = max(0, cx_px - half), min(depth.shape[1], cx_px + half + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _size_range_m(pixel_diameter: float, fx: float, known_diameter_m: float) -> float | None:
    if pixel_diameter <= 0:
        return None
    return fx * known_diameter_m / pixel_diameter


class ColorDetector:
    """The permanent L2 detection layer: HSV threshold -> contours -> circles."""

    def __init__(self, config: ColorDetectorConfig | None = None) -> None:
        self.config = config or ColorDetectorConfig()

    def detect(
        self, frame: Frame, debug: bool = False
    ) -> tuple[list[Detection], DebugMasks | None]:
        cfg = self.config
        width, height = frame.intrinsics.width, frame.intrinsics.height

        if cfg.roi is not None:
            x1n, y1n, x2n, y2n = cfg.roi
            px1, py1 = int(x1n * width), int(y1n * height)
            px2, py2 = int(x2n * width), int(y2n * height)
        else:
            px1, py1, px2, py2 = 0, 0, width, height

        # Crop before converting: at 1920x1080 full-frame HSV conversion and
        # morphology are expensive, and OpenCV releases the GIL during both,
        # so this method is safe to run in a worker thread alongside the
        # Python-level control loop.
        crop = frame.image[py1:py2, px1:px2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hue_lo, hue_hi in cfg.hue_ranges:
            lo = (hue_lo, cfg.sat_min, cfg.val_min)
            hi = (hue_hi, 255, 255)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

        kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        accepted: list[_Circle] = []
        debug_contours: list[Contour] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            bbox_px = (x + px1, y + py1, w, h)

            if perimeter <= 0:
                if debug:
                    debug_contours.append(Contour(bbox_px, area, 0.0, False, "zero_perimeter"))
                continue

            circularity = 4.0 * math.pi * area / (perimeter**2)
            reject_reason = None
            if area < cfg.min_area_px:
                reject_reason = "area"
            elif circularity < cfg.circularity_min:
                reject_reason = "circularity"

            if reject_reason is not None:
                if debug:
                    debug_contours.append(Contour(bbox_px, area, circularity, False, reject_reason))
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            accepted.append(_Circle(cx + px1, cy + py1, radius, area, circularity, bbox_px))

        if cfg.lowest_circle_rule:
            accepted = _group_lowest_circle(accepted)

        detections: list[Detection] = []
        for circle in accepted:
            diameter_px = 2.0 * circle.radius
            x1 = max(0.0, (circle.cx - circle.radius) / width)
            y1 = max(0.0, (circle.cy - circle.radius) / height)
            x2 = min(1.0, (circle.cx + circle.radius) / width)
            y2 = min(1.0, (circle.cy + circle.radius) / height)

            range_m: float | None = None
            range_source: RangeSource = "none"
            if frame.has_depth and frame.depth is not None:
                range_m = _depth_range_m(
                    frame.depth, int(circle.cx), int(circle.cy), DEPTH_SAMPLE_WINDOW_PX
                )
                if range_m is not None:
                    range_source = "depth"
            if range_m is None:
                # Coarse, and deliberately not gated on
                # CameraIntrinsics.is_reliable: that flag governs whether the
                # UI trusts a computed crosshair, a precision concern. A size
                # estimate from an assumed FOV is exactly the point here —
                # it is what lets the range gate be exercised against a
                # plain webcam, which never reports better than "estimated".
                range_m = _size_range_m(diameter_px, frame.intrinsics.fx, cfg.known_diameter_m)
                if range_m is not None:
                    range_source = "size"

            # No ML confidence score exists in a classic-CV pipeline;
            # circularity is the closest available signal for "how sure are
            # we this blob is actually a circle and not noise".
            confidence = min(1.0, circle.circularity)

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    cls=None,
                    confidence=confidence,
                    range_m=range_m,
                    range_source=range_source,
                    source_layer=Layer.L2,
                    iff=IFF.UNKNOWN,
                )
            )
            if debug:
                debug_contours.append(
                    Contour(circle.bbox_px, circle.area, circle.circularity, True, None)
                )

        debug_masks = None
        if debug:
            debug_masks = DebugMasks(
                hsv_mask=mask, morphed_mask=morphed, contours=tuple(debug_contours)
            )
        return detections, debug_masks
