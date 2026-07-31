"""Classic-CV colour detector — the permanent L2 fallback.

This runs whenever the (future) YOLO L1 pipeline's health degrades, not
just during bring-up, and is also what gets demonstrated in the
capability video, so it is built to production standard rather than as
a scaffold.

Output is always ``cls=None``: this layer cannot classify a target and
must never pretend otherwise. It **does** set ``iff``, though — the
competition's targets are colour-coded (hostile red, friendly blue), so
colour alone is enough to answer the IFF question even without knowing
whether a red blob is a drone or an F-16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class ColorClass:
    name: str
    iff: IFF
    hue_ranges: tuple[tuple[int, int], ...]
    sat_min: int
    val_min: int


# Confirmed competition target colours, converted to OpenCV HSV (H 0-179):
#   #F50A0A (hostile red)   -> H 0,  S 245, V 245
#   #00A3E0 (friendly blue) -> H 98, S 255, V 224 — an azure, not true cyan
#     (H 90); a range centred on 90 would miss it, hence (90, 108).
# Saturation floor is 90 rather than a tighter value: the printed models
# are highly saturated, but indoor competition lighting pulls measured
# saturation down, and a tight floor loses targets at range.
_DEFAULT_CLASSES: tuple[ColorClass, ...] = (
    ColorClass("hostile", IFF.HOSTILE, ((0, 10), (170, 180)), 90, 50),
    ColorClass("friendly", IFF.FRIENDLY, ((90, 108),), 90, 50),
)


@dataclass
class ColorDetectorConfig:
    classes: tuple[ColorClass, ...] = field(default_factory=lambda: _DEFAULT_CLASSES)
    morph_kernel: int = 5
    min_area_px: int = 12
    circularity_min: float = 0.70
    # Off by default: aircraft silhouettes are not circular (an F-16's
    # circularity sits well below 0.70) and would be rejected by a filter
    # that assumed balloons. Circularity is still computed and reported
    # for the tuning panel either way.
    require_circularity: bool = False
    roi: BoundingBox | None = None  # normalised (x1, y1, x2, y2)
    known_sizes_m: tuple[float, ...] = (0.30, 0.40, 0.50)


@dataclass(frozen=True)
class Contour:
    """One contour this frame considered, accepted or not — for the tuning panel."""

    class_name: str
    bbox_px: tuple[int, int, int, int]  # x, y, w, h in full-frame pixels
    area_px: float
    circularity: float
    accepted: bool
    reject_reason: str | None = None


@dataclass(frozen=True)
class DebugMasks:
    hsv_masks: dict[str, np.ndarray]  # class name -> raw HSV threshold mask
    morphed_masks: dict[str, np.ndarray]  # class name -> post-morphology mask
    contours: tuple[Contour, ...]


def is_under_resolved(
    bbox: BoundingBox, intrinsics: CameraIntrinsics, min_target_px: int = MIN_TARGET_PX
) -> bool:
    """True when a normalised bbox's pixel width is below MIN_TARGET_PX.

    Computed from intrinsics.width rather than a hardcoded resolution
    table, so the same check is valid whether the frame came from a
    1280x720 webcam or a 1920x1080 D435i. Reference points from the
    competition spec: a 50cm model at 15m is ~31px at 1280x720 and ~47px
    at 1920x1080; a 30cm drone is ~19px and ~28px respectively.
    """
    width_px = (bbox[2] - bbox[0]) * intrinsics.width
    return width_px < min_target_px


def _depth_range_m(depth: np.ndarray, cx_px: int, cy_px: int, window: int) -> float | None:
    half = window // 2
    y0, y1 = max(0, cy_px - half), min(depth.shape[0], cy_px + half + 1)
    x0, x1 = max(0, cx_px - half), min(depth.shape[1], cx_px + half + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _size_range_m(pixel_size: float, fx: float, known_sizes_m: tuple[float, ...]) -> float | None:
    if pixel_size <= 0 or not known_sizes_m:
        return None
    # Without a class, the true size is genuinely ambiguous. Assume the
    # largest plausible size: that yields the largest (most conservative)
    # range estimate, which is the safe direction of error for a gate
    # whose job is bounding how close/far is safe to engage — the wrong
    # way to be wrong is to underestimate range and pass something that
    # is actually too far.
    return fx * max(known_sizes_m) / pixel_size


class ColorDetector:
    """The permanent L2 detection layer: per-class HSV threshold -> contours."""

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
        kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)

        detections: list[Detection] = []
        debug_hsv_masks: dict[str, np.ndarray] = {}
        debug_morphed_masks: dict[str, np.ndarray] = {}
        debug_contours: list[Contour] = []

        for color_class in cfg.classes:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for hue_lo, hue_hi in color_class.hue_ranges:
                lo = (hue_lo, color_class.sat_min, color_class.val_min)
                hi = (hue_hi, 255, 255)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

            morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if debug:
                debug_hsv_masks[color_class.name] = mask
                debug_morphed_masks[color_class.name] = morphed

            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                detection, contour_debug = self._evaluate_contour(
                    contour, color_class, frame, width, height, px1, py1
                )
                if detection is not None:
                    detections.append(detection)
                if debug and contour_debug is not None:
                    debug_contours.append(contour_debug)

        debug_masks = None
        if debug:
            debug_masks = DebugMasks(
                hsv_masks=debug_hsv_masks,
                morphed_masks=debug_morphed_masks,
                contours=tuple(debug_contours),
            )
        return detections, debug_masks

    def _evaluate_contour(
        self,
        contour: np.ndarray,
        color_class: ColorClass,
        frame: Frame,
        width: int,
        height: int,
        px1: int,
        py1: int,
    ) -> tuple[Detection | None, Contour | None]:
        cfg = self.config
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_px = (x + px1, y + py1, w, h)
        circularity = 4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else 0.0

        reject_reason: str | None = None
        if area < cfg.min_area_px:
            reject_reason = "area"
        elif cfg.require_circularity and circularity < cfg.circularity_min:
            reject_reason = "circularity"

        if reject_reason is not None:
            return None, Contour(color_class.name, bbox_px, area, circularity, False, reject_reason)

        (ecx, ecy), _ = cv2.minEnclosingCircle(contour)
        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        pixel_size = max(rect_w, rect_h)
        cx_full, cy_full = ecx + px1, ecy + py1

        x1n = max(0.0, bbox_px[0] / width)
        y1n = max(0.0, bbox_px[1] / height)
        x2n = min(1.0, (bbox_px[0] + bbox_px[2]) / width)
        y2n = min(1.0, (bbox_px[1] + bbox_px[3]) / height)

        range_m: float | None = None
        range_source: RangeSource = "none"
        if frame.has_depth and frame.depth is not None:
            range_m = _depth_range_m(
                frame.depth, int(cx_full), int(cy_full), DEPTH_SAMPLE_WINDOW_PX
            )
            if range_m is not None:
                range_source = "depth"
        if range_m is None:
            range_m = _size_range_m(pixel_size, frame.intrinsics.fx, cfg.known_sizes_m)
            if range_m is not None:
                range_source = "size"

        # No ML confidence score exists in a classic-CV pipeline. Fill
        # ratio (contour area over its own bounding box) works for both
        # circular and elongated shapes, unlike circularity, which would
        # unfairly penalise a legitimately-detected aircraft silhouette.
        confidence = min(1.0, area / max(w * h, 1))

        detection = Detection(
            bbox=(x1n, y1n, x2n, y2n),
            cls=None,
            confidence=confidence,
            range_m=range_m,
            range_source=range_source,
            source_layer=Layer.L2,
            iff=color_class.iff,
        )
        contour_debug = Contour(color_class.name, bbox_px, area, circularity, True, None)
        return detection, contour_debug
