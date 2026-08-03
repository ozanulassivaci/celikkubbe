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
    check_negative_sample,
    compute_min_area_px,
    config_from_dict,
    config_to_dict,
    derive_thresholds,
    is_range_estimate_unreliable,
    is_under_resolved,
    list_hsv_presets,
    load_hsv_preset,
    sample_rectangle,
    sample_region,
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

from ..factories import make_track

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


# --- --diag: per-contour, per-filter-stage logging ---


def _draw_hex_wings(image: np.ndarray, hex_color: str) -> np.ndarray:
    """A swept-wing silhouette (fuselage + wings + tailfins) filled in
    ``hex_color`` -- large bounding box, plenty of area, but its own
    convex hull is much bigger than its filled area because of the deep
    concave notches between fuselage and wingtips, exactly the shape
    class_min_area_px cannot see coming. Empirically measured at
    solidity ~0.37, well under the 0.75 default -- this is the shape
    that reproduces "a large model produces zero detections" without
    needing a real photographed target.
    """
    out = image.copy()
    color = hex_to_bgr(hex_color)
    cv2.rectangle(out, (140, 20), (160, 230), color, -1)
    cv2.fillPoly(out, [np.array([[140, 120], [40, 180], [60, 190], [140, 150]])], color)
    cv2.fillPoly(out, [np.array([[160, 120], [260, 180], [240, 190], [160, 150]])], color)
    cv2.fillPoly(out, [np.array([[145, 190], [110, 230], [150, 230]])], color)
    cv2.fillPoly(out, [np.array([[155, 190], [190, 230], [150, 230]])], color)
    return out


def test_diag_off_by_default_logs_nothing(caplog) -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 100), 20, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    with caplog.at_level(logging.INFO, logger="celikkubbe.vision.l2_color"):
        ColorDetector().detect(frame)

    assert "DIAG" not in caplog.text


def test_diag_logs_a_row_for_an_accepted_contour(caplog) -> None:
    base = _blank_frame()
    image = _draw_hex_circle(base.image, (100, 100), 20, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    with caplog.at_level(logging.INFO, logger="celikkubbe.vision.l2_color"):
        detections, _ = ColorDetector().detect(frame, diag=True)

    assert len(detections) == 1
    diag_lines = [line for line in caplog.text.splitlines() if "DIAG hostile#0" in line]
    assert len(diag_lines) == 1
    assert "verdict=ACCEPT" in diag_lines[0]
    assert "area=" in diag_lines[0] and "solidity=" in diag_lines[0]
    assert "hue=" in diag_lines[0] and "sat=" in diag_lines[0] and "val=" in diag_lines[0]


def test_diag_reports_area_pass_but_solidity_fail_for_a_wing_shape(caplog) -> None:
    # The reproduction of this session's actual bug report: a large,
    # well-saturated, correctly-coloured aircraft silhouette that the
    # area filter clears easily but the default solidity_min=0.75
    # rejects outright -- proving the model's contour DOES appear in the
    # raw contour list (ruling out an HSV/mask problem) and identifying
    # exactly which single stage discards it (not an inverted comparator
    # anywhere -- solidity_min is simply too strict for this silhouette).
    base = _blank_frame(width=300, height=250)
    image = _draw_hex_wings(base.image, HOSTILE_HEX)
    frame = _frame_with_image(image, base)

    with caplog.at_level(logging.INFO, logger="celikkubbe.vision.l2_color"):
        detections, _ = ColorDetector().detect(frame, diag=True)

    assert detections == []  # confirms the reported symptom: zero detections
    diag_lines = [line for line in caplog.text.splitlines() if "DIAG hostile#0" in line]
    assert len(diag_lines) == 1
    line = diag_lines[0]
    assert "area=" in line
    area_clause = line.split("area=")[1].split(")")[0]
    assert "PASS" in area_clause, f"expected the area stage to pass, got: {area_clause}"
    solidity_clause = line.split("solidity=")[1].split(")")[0]
    assert "FAIL" in solidity_clause, f"expected the solidity stage to fail, got: {solidity_clause}"
    assert "verdict=REJECT:solidity" in line


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


# --- eyedropper calibration ---


def _uniform_frame(hue: int, sat: int = 200, val: int = 200, width: int = 20, height: int = 20):
    hsv_img = np.full((height, width, 3), (hue, sat, val), dtype=np.uint8)
    bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return Frame(
        image=bgr,
        t=0.0,
        intrinsics=estimated_intrinsics(width, height),
        has_depth=False,
        depth=None,
    )


def test_sample_region_returns_expected_median_and_percentiles_for_a_uniform_patch() -> None:
    frame = _uniform_frame(hue=98, sat=210, val=180)
    sample = sample_region(frame, (0.5, 0.5), radius_px=8)
    assert sample.hue_median == pytest.approx(98.0, abs=1.0)
    assert sample.hue_p5 == pytest.approx(98.0, abs=1.0)
    assert sample.hue_p95 == pytest.approx(98.0, abs=1.0)
    assert sample.sat_median == pytest.approx(210.0, abs=1.0)
    assert sample.val_median == pytest.approx(180.0, abs=1.0)
    assert sample.pixel_count > 0
    assert sample.hue_wraps is False


def test_derived_thresholds_detect_the_sampled_patch() -> None:
    frame = _uniform_frame(hue=98, sat=210, val=180)
    sample = sample_region(frame, (0.5, 0.5), radius_px=8)
    hue_ranges, sat_min, val_min = derive_thresholds(sample)

    detector = ColorDetector(
        ColorDetectorConfig(
            classes=(ColorClass("sampled", IFF.HOSTILE, hue_ranges, sat_min, val_min),),
        )
    )
    detections, _ = detector.detect(frame)
    assert len(detections) == 1


def test_zero_spread_sample_still_derives_a_non_degenerate_hue_range() -> None:
    # A flat-filled synthetic patch (or a real photo of an evenly, flatly
    # lit surface) can have literally zero measured hue spread -- caught
    # empirically running the real eyedropper against SyntheticSource's
    # own flat-colour targets, where the naive margin*span padding
    # produced a single-hue-value range that visibly fragmented detection
    # on the very next frame's slightly different pixel.
    frame = _uniform_frame(hue=98, sat=210, val=180)
    sample = sample_region(frame, (0.5, 0.5), radius_px=8)
    assert sample.hue_p95 == sample.hue_p5  # confirms this sample is the zero-spread case

    hue_ranges, _sat_min, _val_min = derive_thresholds(sample)

    assert len(hue_ranges) == 1
    lo, hi = hue_ranges[0]
    assert hi > lo


def test_hue_wraparound_sample_produces_two_ranges_not_one_spanning_everything() -> None:
    # Half the patch at hue 2, half at hue 178 -- a real hostile-red
    # sample straddling the 0/180 seam, not a genuinely multi-hued patch.
    hsv_img = np.zeros((10, 20, 3), dtype=np.uint8)
    hsv_img[:, :10] = (2, 200, 200)
    hsv_img[:, 10:] = (178, 200, 200)
    bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    frame = Frame(
        image=bgr, t=0.0, intrinsics=estimated_intrinsics(20, 10), has_depth=False, depth=None
    )

    sample = sample_rectangle(frame, (0.0, 0.0, 1.0, 1.0))
    assert sample.hue_wraps is True

    hue_ranges, _sat_min, _val_min = derive_thresholds(sample)
    assert len(hue_ranges) == 2
    # Neither range may span more than a small neighbourhood of the two
    # sampled clusters -- the failure mode this must avoid is one range
    # wide enough to match nearly every hue.
    for lo, hi in hue_ranges:
        assert hi - lo < 30


def test_percentiles_ignore_stray_outlier_pixels() -> None:
    height, width = 10, 10
    hsv_img = np.full((height, width, 3), (98, 200, 200), dtype=np.uint8)
    hsv_img[0, 0] = (5, 255, 255)  # one stray pixel, far from the rest
    bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    frame = Frame(
        image=bgr,
        t=0.0,
        intrinsics=estimated_intrinsics(width, height),
        has_depth=False,
        depth=None,
    )

    sample = sample_rectangle(frame, (0.0, 0.0, 1.0, 1.0))
    # A single outlier among 100 pixels must not widen p5/p95 -- min/max
    # would have caught it, percentiles must not.
    assert sample.hue_p5 == pytest.approx(98.0, abs=1.0)
    assert sample.hue_p95 == pytest.approx(98.0, abs=1.0)


def test_negative_sample_reports_acceptance_by_current_thresholds() -> None:
    # A skin-tone-like sample sitting inside the default hostile range,
    # matching the false-positive scenario that motivated this tool.
    frame = _uniform_frame(hue=5, sat=120, val=220)
    sample = sample_rectangle(frame, (0.0, 0.0, 1.0, 1.0))

    reports = check_negative_sample(sample, ColorDetectorConfig())
    hostile_report = next(r for r in reports if r.class_name == "hostile")
    friendly_report = next(r for r in reports if r.class_name == "friendly")

    assert hostile_report.accepted is True
    assert hostile_report.tighten_field in ("sat_min", "val_min")
    assert hostile_report.suggested_value is not None
    assert friendly_report.accepted is False
    assert friendly_report.tighten_field is None


def test_negative_sample_outside_every_class_reports_not_accepted() -> None:
    frame = _uniform_frame(hue=60, sat=50, val=50)  # a dim green, in neither class
    sample = sample_rectangle(frame, (0.0, 0.0, 1.0, 1.0))
    reports = check_negative_sample(sample, ColorDetectorConfig())
    assert all(not r.accepted for r in reports)


def test_point_click_and_rectangle_drag_agree_for_a_uniform_region() -> None:
    frame = _uniform_frame(hue=3, sat=222, val=190, width=40, height=40)
    point_sample = sample_region(frame, (0.5, 0.5), radius_px=10)
    rect_sample = sample_rectangle(frame, (0.375, 0.375, 0.625, 0.625))

    assert point_sample.hue_median == pytest.approx(rect_sample.hue_median, abs=1.0)
    assert point_sample.sat_median == pytest.approx(rect_sample.sat_median, abs=1.0)
    assert point_sample.val_median == pytest.approx(rect_sample.val_median, abs=1.0)
    assert point_sample.hue_wraps == rect_sample.hue_wraps


def test_sample_region_outside_frame_returns_empty_sample() -> None:
    frame = _uniform_frame(hue=98)
    sample = sample_region(frame, (5.0, 5.0), radius_px=5)
    assert sample.pixel_count == 0


# --- implausible range-estimate flagging ---


def test_small_confident_bbox_flags_range_as_unreliable() -> None:
    intr = estimated_intrinsics(640, 480)
    tiny_confident = make_track(bbox=(0.5, 0.5, 0.52, 0.52), confidence=0.9)  # ~13px wide
    assert is_range_estimate_unreliable(tiny_confident, intr) is True


def test_normal_sized_bbox_is_not_flagged() -> None:
    intr = estimated_intrinsics(640, 480)
    normal = make_track(bbox=(0.3, 0.3, 0.5, 0.5), confidence=0.9)  # ~128px wide
    assert is_range_estimate_unreliable(normal, intr) is False


def test_extreme_aspect_ratio_flags_range_as_unreliable() -> None:
    intr = estimated_intrinsics(640, 480)
    sliver = make_track(bbox=(0.1, 0.1, 0.9, 0.12), confidence=0.9)  # very wide, very thin
    assert is_range_estimate_unreliable(sliver, intr) is True
