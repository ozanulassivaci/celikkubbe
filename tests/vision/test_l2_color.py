from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import IFF, CameraIntrinsics, Frame, Layer
from celikkubbe.vision.l2_color import (
    ColorClass,
    ColorDetector,
    ColorDetectorConfig,
    compute_min_area_px,
    config_from_dict,
    config_to_dict,
    is_under_resolved,
    list_hsv_presets,
    load_hsv_preset,
    save_hsv_preset,
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


# --- compute_min_area_px (2a): scales with intrinsics and range ---


def _reliable_intrinsics(width: int, height: int, fx: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width, height=height, fx=fx, fy=fx, cx=width / 2, cy=height / 2, quality="factory"
    )


def test_compute_min_area_px_scales_with_fx() -> None:
    # expected_px is linear in fx, so the resulting area (squared) should
    # roughly quadruple when fx doubles.
    small = compute_min_area_px(_reliable_intrinsics(640, 480, fx=500.0), max_range_m=15.0)
    large = compute_min_area_px(_reliable_intrinsics(640, 480, fx=1000.0), max_range_m=15.0)
    assert large > small
    assert large == pytest.approx(small * 4, rel=0.05)


def test_compute_min_area_px_scales_inversely_with_max_range() -> None:
    near_range = compute_min_area_px(_reliable_intrinsics(640, 480, fx=800.0), max_range_m=5.0)
    far_range = compute_min_area_px(_reliable_intrinsics(640, 480, fx=800.0), max_range_m=15.0)
    assert near_range > far_range


def test_compute_min_area_px_falls_back_to_frame_area_fraction_when_estimated() -> None:
    # "estimated" (webcam, no calibration) intrinsics must not trust fx --
    # the floor scales with frame area instead, independent of resolution.
    small_frame = estimated_intrinsics(640, 480)
    large_frame = estimated_intrinsics(1920, 1080)
    small_val = compute_min_area_px(small_frame)
    large_val = compute_min_area_px(large_frame)
    assert large_val > small_val
    area_ratio = (1920 * 1080) / (640 * 480)
    assert large_val == pytest.approx(small_val * area_ratio, rel=0.02)


def test_default_detector_rejects_a_contour_below_the_computed_floor() -> None:
    base = _blank_frame()
    floor = compute_min_area_px(base.intrinsics)
    side = max(1, int((floor * 0.5) ** 0.5))
    image = base.image.copy()
    cv2.rectangle(image, (100, 100), (100 + side, 100 + side), hex_to_bgr(HOSTILE_HEX), -1)
    frame = _frame_with_image(image, base)

    detector = ColorDetector()  # min_area_px left at its None (auto) default
    detections, debug = detector.detect(frame, debug=True)

    assert detections == []
    assert debug is not None
    assert any(c.reject_reason == "area" for c in debug.contours)


def test_computed_min_area_px_is_logged(caplog) -> None:
    caplog.set_level(logging.INFO, logger="celikkubbe.vision.l2_color")
    frame = _blank_frame()

    ColorDetector().detect(frame)

    assert any("min_area_px" in record.message for record in caplog.records)


# --- max_detections_per_class (2b) ---


def test_five_same_colour_blobs_yield_exactly_three_detections() -> None:
    base = _blank_frame()
    image = base.image
    for cx in (40, 100, 160, 220, 280):
        image = _draw_hex_circle(image, (cx, 50), 15, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    detections, debug = ColorDetector().detect(frame, debug=True)

    assert len(detections) == 3
    assert debug is not None
    capped = [c for c in debug.contours if c.reject_reason == "class_cap"]
    assert len(capped) == 2


def test_max_detections_per_class_is_configurable() -> None:
    base = _blank_frame()
    image = base.image
    for cx in (40, 100, 160, 220, 280):
        image = _draw_hex_circle(image, (cx, 50), 15, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(max_detections_per_class=1))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1


def test_class_cap_keeps_the_largest_contours_by_area() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (60, 60), 25, HOSTILE_HEX)  # largest, must survive
    image = _draw_hex_circle(image, (200, 60), 8, HOSTILE_HEX)  # smallest, must be capped
    image = _draw_hex_circle(image, (60, 180), 20, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(max_detections_per_class=2))
    detections, debug = detector.detect(frame, debug=True)

    assert len(detections) == 2
    capped_areas = [c.area_px for c in debug.contours if c.reject_reason == "class_cap"]
    kept_areas = [c.area_px for c in debug.contours if c.accepted]
    assert capped_areas and kept_areas
    assert max(capped_areas) < min(kept_areas)


# --- solidity filter (2c) ---


def _draw_plus_shape(image: np.ndarray) -> np.ndarray:
    """A plus/cross silhouette: no interior hole (so RETR_EXTERNAL's outer
    contour genuinely reflects its concave shape, unlike a ring, where the
    outer contour traces only the full outer circle and the hole is
    invisible to cv2.contourArea), but a real concavity that scores well
    below a filled blob on convex-hull solidity -- around 0.47 by direct
    measurement, comfortably under the 0.75 default threshold.
    """
    out = image.copy()
    cv2.rectangle(out, (90, 60), (110, 180), hex_to_bgr(HOSTILE_HEX), thickness=-1)
    cv2.rectangle(out, (40, 105), (160, 125), hex_to_bgr(HOSTILE_HEX), thickness=-1)
    return out


def test_low_solidity_shape_rejected_as_solidity() -> None:
    base = _blank_frame()
    image = _draw_plus_shape(base.image)
    frame = _frame_with_image(image, base)

    detections, debug = ColorDetector().detect(frame, debug=True)

    assert detections == []
    assert debug is not None
    assert any(c.reject_reason == "solidity" for c in debug.contours if not c.accepted)


def test_solid_blob_passes_the_default_solidity_filter() -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 120), 30, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    detections, _ = ColorDetector().detect(frame)

    assert len(detections) == 1


def test_solidity_min_is_configurable() -> None:
    base = _blank_frame()
    image = _draw_plus_shape(base.image)
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(solidity_min=0.0))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1


# --- aspect ratio bounds (2d) ---


def test_thin_strip_rejected_as_aspect_ratio() -> None:
    base = _blank_frame()
    image = base.image.copy()
    cv2.rectangle(image, (40, 100), (190, 108), hex_to_bgr(HOSTILE_HEX), thickness=-1)  # 150x8
    frame = _frame_with_image(image, base)

    detections, debug = ColorDetector().detect(frame, debug=True)

    assert detections == []
    assert debug is not None
    assert any(c.reject_reason == "aspect_ratio" for c in debug.contours if not c.accepted)


def test_squarish_blob_passes_the_default_aspect_ratio_filter() -> None:
    base = _blank_frame()
    image = base.image.copy()
    cv2.rectangle(image, (100, 100), (140, 140), hex_to_bgr(HOSTILE_HEX), thickness=-1)  # 40x40
    frame = _frame_with_image(image, base)

    detections, _ = ColorDetector().detect(frame)

    assert len(detections) == 1


def test_aspect_ratio_range_is_configurable() -> None:
    base = _blank_frame()
    image = base.image.copy()
    cv2.rectangle(image, (40, 100), (190, 108), hex_to_bgr(HOSTILE_HEX), thickness=-1)  # 150x8
    frame = _frame_with_image(image, base)

    detector = ColorDetector(ColorDetectorConfig(aspect_ratio_range=(0.0, 50.0)))
    detections, _ = detector.detect(frame)

    assert len(detections) == 1


# --- config defaults and persistence for the new structural filters ---


def test_new_structural_filter_defaults() -> None:
    cfg = ColorDetectorConfig()
    assert cfg.min_area_px is None
    assert cfg.solidity_min == 0.75
    assert cfg.aspect_ratio_range == (0.2, 5.0)
    assert cfg.max_detections_per_class == 3


def test_config_to_dict_and_back_round_trips_new_filter_fields() -> None:
    config = ColorDetectorConfig(
        solidity_min=0.5, aspect_ratio_range=(0.1, 8.0), max_detections_per_class=5
    )
    restored = config_from_dict(config_to_dict(config))
    assert restored.solidity_min == 0.5
    assert restored.aspect_ratio_range == (0.1, 8.0)
    assert restored.max_detections_per_class == 5


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


# --- HSV preset persistence ---


def test_config_to_dict_and_back_round_trips() -> None:
    config = ColorDetectorConfig(morph_kernel=7, min_area_px=20, circularity_min=0.5)
    restored = config_from_dict(config_to_dict(config))
    assert restored.morph_kernel == 7
    assert restored.min_area_px == 20
    assert restored.circularity_min == 0.5
    assert restored.classes == config.classes


def test_config_from_dict_falls_back_to_defaults_for_missing_keys() -> None:
    defaults = ColorDetectorConfig()
    restored = config_from_dict({})
    assert restored.morph_kernel == defaults.morph_kernel
    assert restored.classes == defaults.classes


def test_config_from_dict_keeps_default_hue_for_a_class_missing_from_the_preset() -> None:
    defaults = ColorDetectorConfig()
    data = {"classes": {"hostile": {"sat_min": 120, "val_min": 60}}}
    restored = config_from_dict(data)
    hostile = next(c for c in restored.classes if c.name == "hostile")
    friendly = next(c for c in restored.classes if c.name == "friendly")
    assert hostile.sat_min == 120
    assert hostile.val_min == 60
    hostile_defaults = next(c for c in defaults.classes if c.name == "hostile")
    assert hostile.hue_ranges == hostile_defaults.hue_ranges
    friendly_defaults = next(c for c in defaults.classes if c.name == "friendly")
    assert friendly == friendly_defaults


def test_save_then_load_hsv_preset_round_trips(tmp_path) -> None:
    config = ColorDetectorConfig(
        morph_kernel=3,
        min_area_px=8,
        classes=(
            ColorClass("hostile", IFF.HOSTILE, ((0, 8), (172, 180)), 100, 60),
            ColorClass("friendly", IFF.FRIENDLY, ((92, 106),), 100, 60),
        ),
    )
    path = tmp_path / "indoor.json"
    save_hsv_preset(config, path)

    loaded = load_hsv_preset(path)
    assert loaded.morph_kernel == 3
    assert loaded.min_area_px == 8
    assert loaded.classes == config.classes


def test_load_hsv_preset_returns_defaults_when_file_is_missing(tmp_path) -> None:
    loaded = load_hsv_preset(tmp_path / "does_not_exist.json")
    assert loaded == ColorDetectorConfig()


def test_load_hsv_preset_returns_defaults_for_corrupt_json(tmp_path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    loaded = load_hsv_preset(path)
    assert loaded == ColorDetectorConfig()


def test_list_hsv_presets_returns_sorted_stems(tmp_path) -> None:
    (tmp_path / "hall.json").write_text("{}")
    (tmp_path / "indoor.json").write_text("{}")
    (tmp_path / "synthetic.json").write_text("{}")
    (tmp_path / "not_a_preset.txt").write_text("ignored")
    assert list_hsv_presets(tmp_path) == ("hall", "indoor", "synthetic")


def test_list_hsv_presets_returns_empty_tuple_when_directory_is_missing(tmp_path) -> None:
    assert list_hsv_presets(tmp_path / "nope") == ()
