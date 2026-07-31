from __future__ import annotations

import cv2
import numpy as np

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import IFF, Frame, Layer
from celikkubbe.vision.l2_color import (
    ColorDetector,
    ColorDetectorConfig,
    is_under_resolved,
)
from celikkubbe.vision.sources import (
    FRIENDLY_HEX,
    HOSTILE_HEX,
    SyntheticSource,
    SyntheticSourceConfig,
    SyntheticTarget,
    estimated_intrinsics,
    hex_to_bgr,
)

WIDTH, HEIGHT = 320, 240


def _blank_frame(width: int = WIDTH, height: int = HEIGHT, has_depth: bool = False) -> Frame:
    image = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)
    return Frame(
        image=image,
        t=0.0,
        intrinsics=estimated_intrinsics(width, height),
        has_depth=has_depth,
        depth=np.zeros((height, width), dtype=np.float32) if has_depth else None,
    )


def _frame_with_image(image: np.ndarray, base: Frame, **overrides) -> Frame:
    kwargs = dict(t=base.t, intrinsics=base.intrinsics, has_depth=base.has_depth, depth=base.depth)
    kwargs.update(overrides)
    return Frame(image=image, **kwargs)


def _draw_hex_circle(
    image: np.ndarray, center: tuple[int, int], radius: int, hex_color: str
) -> np.ndarray:
    out = image.copy()
    cv2.circle(out, center, radius, hex_to_bgr(hex_color), thickness=-1)
    return out


def _draw_hsv_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    hue: int,
    sat: int = 200,
    val: int = 200,
) -> np.ndarray:
    """Draw a circle at an exact HSV hue by round-tripping through HSV space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.circle(hsv, center, radius, (hue, sat, val), thickness=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_exact_hostile_hex_detected_as_hostile() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 100), 20, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    detections, _ = ColorDetector().detect(frame)

    assert len(detections) == 1
    assert detections[0].iff is IFF.HOSTILE
    assert detections[0].cls is None
    assert detections[0].source_layer is Layer.L2


def test_exact_friendly_hex_detected_as_friendly() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 100), 20, FRIENDLY_HEX)
    frame = _frame_with_image(image, base)

    detections, _ = ColorDetector().detect(frame)

    assert len(detections) == 1
    assert detections[0].iff is IFF.FRIENDLY


def test_friendly_range_catches_both_true_cyan_and_azure() -> None:
    # H 90 is true cyan; H 98 is the confirmed #00A3E0 azure. Both must
    # fall inside the friendly hue range (90, 108).
    base = _blank_frame()
    image = _draw_hsv_circle(base.image, (80, 80), 18, hue=90)
    image = _draw_hsv_circle(image, (220, 80), 18, hue=98)
    frame = _frame_with_image(image, base)

    detections, _ = ColorDetector().detect(frame)

    assert len(detections) == 2
    assert all(d.iff is IFF.FRIENDLY for d in detections)


def test_red_hue_wraparound_both_ends_detected() -> None:
    base = _blank_frame()
    image = _draw_hsv_circle(base.image, (80, 80), 18, hue=0)
    image = _draw_hsv_circle(image, (220, 80), 18, hue=179)
    frame = _frame_with_image(image, base)

    detections, _ = ColorDetector().detect(frame)

    assert len(detections) == 2
    assert all(d.iff is IFF.HOSTILE for d in detections)


def test_non_circular_shape_accepted_without_circularity_requirement() -> None:
    base = _blank_frame()
    image = base.image.copy()
    cv2.rectangle(image, (100, 80), (160, 100), hex_to_bgr(HOSTILE_HEX), thickness=-1)
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(require_circularity=False))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1


def test_non_circular_shape_rejected_with_circularity_required() -> None:
    base = _blank_frame()
    image = base.image.copy()
    cv2.rectangle(image, (100, 80), (160, 100), hex_to_bgr(HOSTILE_HEX), thickness=-1)
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(require_circularity=True, circularity_min=0.85))
    detections, debug = detector.detect(frame, debug=True)

    assert detections == []
    assert debug is not None
    rejected = [c for c in debug.contours if not c.accepted]
    assert any(c.reject_reason == "circularity" for c in rejected)


def test_roi_cropping_excludes_blobs_outside_region() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (60, 60), 15, HOSTILE_HEX)
    image = _draw_hex_circle(image, (260, 200), 15, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    roi = (0.0, 0.0, 0.5, 0.5)
    detector = ColorDetector(ColorDetectorConfig(roi=roi))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    cx = (detections[0].bbox[0] + detections[0].bbox[2]) / 2
    cy = (detections[0].bbox[1] + detections[0].bbox[3]) / 2
    assert cx < 0.5
    assert cy < 0.5


def test_depth_range_uses_median_and_ignores_zeros() -> None:
    base = _blank_frame(has_depth=True)
    image = _draw_hex_circle(base.image, (100, 100), 15, HOSTILE_HEX)

    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    depth[95:106, 95:106] = 8.0
    depth[100, 100] = 0.0  # dropout at the very centroid must be ignored
    depth[99, 99] = np.nan  # invalid must be ignored too

    frame = _frame_with_image(image, base, depth=depth)

    detections, _ = ColorDetector().detect(frame)

    assert len(detections) == 1
    assert detections[0].range_source == "depth"
    assert detections[0].range_m == 8.0


def test_size_based_range_within_tolerance_for_synthetic_50cm_target() -> None:
    true_range_m = 10.0
    target = SyntheticTarget(
        color_hex=HOSTILE_HEX, size_m=0.50, lane_fraction=0.5, range_m=true_range_m
    )
    source = SyntheticSource(
        FakeClock(), SyntheticSourceConfig(width=WIDTH, height=HEIGHT, targets=(target,))
    )
    source.start()
    frame = source.read()
    assert frame is not None

    detector = ColorDetector(ColorDetectorConfig(known_sizes_m=(0.30, 0.40, 0.50)))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].range_source == "size"
    assert abs(detections[0].range_m - true_range_m) / true_range_m < 0.15


def test_range_source_is_none_when_pixel_size_is_degenerate() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 100), 0, HOSTILE_HEX)  # zero-radius, degenerate
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(min_area_px=0))
    detections, _ = detector.detect(frame)

    for detection in detections:
        if detection.range_m is None:
            assert detection.range_source == "none"


def test_under_resolved_detection_is_flagged() -> None:
    intrinsics = estimated_intrinsics(WIDTH, HEIGHT)
    tiny_bbox = (0.5, 0.5, 0.501, 0.501)
    normal_bbox = (0.5, 0.5, 0.6, 0.6)
    assert is_under_resolved(tiny_bbox, intrinsics) is True
    assert is_under_resolved(normal_bbox, intrinsics) is False


def test_detect_emits_under_resolved_detection_when_target_is_tiny() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 100), 3, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(min_area_px=1))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    assert is_under_resolved(detections[0].bbox, frame.intrinsics) is True


def test_debug_masks_populated_only_when_requested() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 100), 15, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    detections, debug_off = ColorDetector().detect(frame, debug=False)
    assert debug_off is None
    assert len(detections) == 1

    _, debug_on = ColorDetector().detect(frame, debug=True)
    assert debug_on is not None
    assert "hostile" in debug_on.hsv_masks
    assert "friendly" in debug_on.hsv_masks
    assert "hostile" in debug_on.morphed_masks
    assert len(debug_on.contours) >= 1
