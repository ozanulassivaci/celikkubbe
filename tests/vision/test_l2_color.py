from __future__ import annotations

import cv2
import numpy as np

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import Frame, Layer
from celikkubbe.vision.l2_color import (
    ColorDetector,
    ColorDetectorConfig,
    is_under_resolved,
)
from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig, estimated_intrinsics

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


def test_detector_finds_known_number_of_synthetic_circles_at_known_positions() -> None:
    source = SyntheticSource(
        FakeClock(),
        SyntheticSourceConfig(width=WIDTH, height=HEIGHT, num_targets=3, speed=0.0, radius_px=15),
    )
    source.start()
    frame = source.read()

    detector = ColorDetector()
    detections, _ = detector.detect(frame)

    assert len(detections) == 3
    expected_x = sorted(((i + 0.5) / 3) % 1.0 for i in range(3))
    found_x = sorted((d.bbox[0] + d.bbox[2]) / 2 for d in detections)
    for expected, found in zip(expected_x, found_x, strict=True):
        assert abs(expected - found) < 0.03
    for detection in detections:
        assert detection.cls is None
        assert detection.source_layer is Layer.L2


def test_circularity_rejects_square_and_accepts_circle() -> None:
    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (60, 60), 20, hue=0)
    cv2.rectangle(image, (200, 40), (240, 80), (0, 0, 220), thickness=-1)
    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=False, depth=None)

    detector = ColorDetector(ColorDetectorConfig(circularity_min=0.85))
    detections, debug = detector.detect(frame, debug=True)

    centers_x = [int((d.bbox[0] + d.bbox[2]) / 2 * WIDTH) for d in detections]
    assert any(abs(cx - 60) < 5 for cx in centers_x)
    assert all(abs(cx - 220) > 5 for cx in centers_x)
    assert debug is not None
    rejected = [c for c in debug.contours if not c.accepted]
    assert any(c.reject_reason == "circularity" for c in rejected)


def test_red_hue_wraparound_both_ends_detected() -> None:
    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (60, 60), 18, hue=0)
    image = _draw_hsv_circle(image, (220, 60), 18, hue=179)
    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=False, depth=None)

    detector = ColorDetector()
    detections, _ = detector.detect(frame)

    assert len(detections) == 2


def test_roi_cropping_excludes_blobs_outside_region() -> None:
    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (60, 60), 15, hue=0)
    image = _draw_hsv_circle(image, (260, 200), 15, hue=0)
    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=False, depth=None)

    roi = (0.0, 0.0, 0.5, 0.5)
    detector = ColorDetector(ColorDetectorConfig(roi=roi))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    cx = (detections[0].bbox[0] + detections[0].bbox[2]) / 2
    cy = (detections[0].bbox[1] + detections[0].bbox[3]) / 2
    assert cx < 0.5
    assert cy < 0.5


def test_lowest_circle_rule_picks_bottom_of_vertical_pair() -> None:
    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (100, 60), 15, hue=0)
    image = _draw_hsv_circle(image, (105, 150), 15, hue=0)
    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=False, depth=None)

    detector = ColorDetector(ColorDetectorConfig(lowest_circle_rule=True))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    cy = (detections[0].bbox[1] + detections[0].bbox[3]) / 2 * HEIGHT
    assert cy > 100


def test_lowest_circle_rule_disabled_keeps_both() -> None:
    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (100, 60), 15, hue=0)
    image = _draw_hsv_circle(image, (105, 150), 15, hue=0)
    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=False, depth=None)

    detector = ColorDetector(ColorDetectorConfig(lowest_circle_rule=False))
    detections, _ = detector.detect(frame)

    assert len(detections) == 2


def test_depth_range_uses_median_and_ignores_zeros() -> None:
    frame = _blank_frame(has_depth=True)
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (100, 100), 15, hue=0)

    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    depth[95:106, 95:106] = 8.0
    depth[100, 100] = 0.0  # dropout at the very centroid must be ignored
    depth[99, 99] = np.nan  # invalid must be ignored too

    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=True, depth=depth)

    detector = ColorDetector()
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].range_source == "depth"
    assert detections[0].range_m == 8.0


def test_size_based_range_within_tolerance_for_known_diameter() -> None:
    known_diameter_m = 0.14
    true_range_m = 3.0
    intrinsics = estimated_intrinsics(WIDTH, HEIGHT)
    pixel_diameter = intrinsics.fx * known_diameter_m / true_range_m
    radius_px = int(round(pixel_diameter / 2))

    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (160, 120), radius_px, hue=0)
    frame = Frame(image=image, t=0.0, intrinsics=intrinsics, has_depth=False, depth=None)

    detector = ColorDetector(ColorDetectorConfig(known_diameter_m=known_diameter_m))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].range_source == "size"
    assert abs(detections[0].range_m - true_range_m) / true_range_m < 0.1


def test_range_source_is_none_without_depth_or_reliable_size_basis() -> None:
    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (100, 100), 0, hue=0)  # degenerate, zero radius
    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=False, depth=None)

    detector = ColorDetector(ColorDetectorConfig(min_area_px=0, circularity_min=0.0))
    detections, _ = detector.detect(frame)

    for detection in detections:
        if detection.range_m is None:
            assert detection.range_source == "none"


def test_under_resolved_detection_is_flagged() -> None:
    intrinsics = estimated_intrinsics(WIDTH, HEIGHT)
    tiny_bbox = (0.5, 0.5, 0.501, 0.501)  # far smaller than MIN_TARGET_PX
    normal_bbox = (0.5, 0.5, 0.6, 0.6)
    assert is_under_resolved(tiny_bbox, intrinsics) is True
    assert is_under_resolved(normal_bbox, intrinsics) is False


def test_detect_emits_under_resolved_detection_when_target_is_tiny() -> None:
    frame = _blank_frame()
    image = frame.image.copy()
    image = _draw_hsv_circle(image, (100, 100), 3, hue=0)
    frame = Frame(image=image, t=0.0, intrinsics=frame.intrinsics, has_depth=False, depth=None)

    detector = ColorDetector(ColorDetectorConfig(min_area_px=1, circularity_min=0.5))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1
    assert is_under_resolved(detections[0].bbox, frame.intrinsics) is True
