"""VideoCanvas tests. Pure geometry (letterbox, label placement) is unit
tested directly; overlay drawing is checked by rendering to an offscreen
QImage and sampling pixels -- shallower than the logic-layer tests
elsewhere, as expected for GUI code, but still exercising the real paint
path rather than only the helper functions it calls.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter

from celikkubbe.core.types import (
    IFF,
    CameraIntrinsics,
    Frame,
    Layer,
    TrackStatus,
)
from celikkubbe.ui import theme
from celikkubbe.ui.snapshot import HealthSnapshot, PipelineTimings, UiSnapshot
from celikkubbe.ui.video_canvas import (
    VideoCanvas,
    compute_letterbox,
    frame_to_pixmap,
    indicator_row_top,
    inset_point,
    lock_banner_rect,
    place_badge,
    place_label,
)

from ..factories import make_state, make_track


def _intrinsics(w: int = 640, h: int = 480, quality: str = "factory") -> CameraIntrinsics:
    return CameraIntrinsics(
        width=w, height=h, fx=500.0, fy=500.0, cx=w / 2, cy=h / 2, quality=quality
    )


def _frame(w: int = 640, h: int = 480, quality: str = "factory") -> Frame:
    image = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
    return Frame(
        image=image, t=0.0, intrinsics=_intrinsics(w, h, quality), has_depth=False, depth=None
    )


def _health() -> HealthSnapshot:
    return HealthSnapshot(
        inference_healthy=True,
        inference_ms=1.0,
        camera_healthy=True,
        camera_fps=30.0,
        link_healthy=True,
        link_stale_ms=0.0,
        crc_error_count=0,
        detection_healthy=True,
        detection_confidence=0.9,
        depth_healthy=True,
        depth_valid_ratio=1.0,
        active_layer=Layer.L2,
        fallback_reason=None,
        l1_recovery_countdown_s=None,
    )


def _timings() -> PipelineTimings:
    return PipelineTimings(
        capture_ms=1.0,
        detect_ms=1.0,
        track_ms=1.0,
        solve_ms=1.0,
        step_ms=1.0,
        tick_ms=5.0,
        fps=30.0,
    )


def _snapshot(
    frame: Frame | None = None,
    tracks=(),
    state=None,
    crosshair_px=None,
    crosshair_offscreen: bool = False,
    crosshair_bearing_deg=None,
    aim=None,
) -> UiSnapshot:
    tracks = tuple(tracks)
    return UiSnapshot(
        t=0.0,
        frame=frame if frame is not None else _frame(),
        detections=(),
        tracks=tracks,
        ordered_track_ids=tuple(t.track_id for t in tracks),
        state=state if state is not None else make_state(tracks=tracks),
        telemetry=None,
        telemetry_frame=None,
        aim=aim,
        crosshair_px=crosshair_px,
        crosshair_offscreen=crosshair_offscreen,
        crosshair_bearing_deg=crosshair_bearing_deg,
        health=_health(),
        timings=_timings(),
        fault_reason=None,
        event_log=(),
        ammo_fired=None,
        engagement_fallback_reason=None,
    )


def _render_to_image(canvas: VideoCanvas) -> QImage:
    image = QImage(canvas.size(), QImage.Format.Format_RGB32)
    painter = QPainter(image)
    canvas.render(painter)
    painter.end()
    return image


def _color_close(a: QColor, b: QColor, tol: int = 30) -> bool:
    return (
        abs(a.red() - b.red()) <= tol
        and abs(a.green() - b.green()) <= tol
        and abs(a.blue() - b.blue()) <= tol
    )


def _has_pixel_near(
    image: QImage, x: int, y: int, target: QColor, radius: int, tol: int = 30
) -> bool:
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            px, py = x + dx, y + dy
            if 0 <= px < image.width() and 0 <= py < image.height():
                if _color_close(QColor(image.pixel(px, py)), target, tol):
                    return True
    return False


def _find_exact_color(image: QImage, target: QColor) -> bool:
    target_rgb = target.rgb()
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixel(x, y) | 0xFF000000 == target_rgb | 0xFF000000:
                return True
    return False


def _images_differ(a: QImage, b: QImage) -> bool:
    return a.convertToFormat(QImage.Format.Format_RGB32) != b.convertToFormat(
        QImage.Format.Format_RGB32
    )


# --- letterbox geometry ---


def test_letterbox_fits_4_3_frame_inside_16_9_widget():
    t = compute_letterbox(widget_w=800, widget_h=450, frame_w=640, frame_h=480)
    assert t.offset_y == 0.0
    assert t.displayed_h == pytest.approx(450.0)
    assert t.displayed_w == pytest.approx(600.0)
    assert t.offset_x == pytest.approx(100.0)


def test_letterbox_fits_16_9_frame_inside_4_3_widget():
    t = compute_letterbox(widget_w=640, widget_h=480, frame_w=1920, frame_h=1080)
    assert t.offset_x == 0.0
    assert t.displayed_w == pytest.approx(640.0)
    assert t.displayed_h == pytest.approx(360.0)
    assert t.offset_y == pytest.approx(60.0)


@pytest.mark.parametrize("x_norm,y_norm", [(0.0, 0.0), (1.0, 1.0), (0.3, 0.7), (0.5, 0.5)])
def test_transform_round_trips(x_norm, y_norm):
    t = compute_letterbox(800, 450, 640, 480)
    x_px, y_px = t.to_widget(x_norm, y_norm)
    rx, ry = t.to_normalized(x_px, y_px)
    assert rx == pytest.approx(x_norm)
    assert ry == pytest.approx(y_norm)


def test_canvas_transform_round_trips_after_set_snapshot(qapp):
    canvas = VideoCanvas()
    canvas.resize(800, 450)
    canvas.set_snapshot(_snapshot(frame=_frame(640, 480)))

    x_px, y_px = canvas.normalized_to_widget(0.25, 0.6)
    rx, ry = canvas.widget_to_normalized(x_px, y_px)
    assert (rx, ry) == pytest.approx((0.25, 0.6))


# --- label placement ---


def test_place_label_stays_inside_frame_when_box_is_at_top_left_corner():
    frame = QRectF(0, 0, 400, 300)
    box = QRectF(0, 0, 20, 20)
    label = place_label(box, label_w=60, label_h=16, frame=frame)
    assert label.left() >= frame.left()
    assert label.top() >= frame.top()
    assert label.right() <= frame.right() + 1e-6
    assert label.bottom() <= frame.bottom() + 1e-6


def test_place_label_stays_inside_frame_when_box_is_at_bottom_right_corner():
    frame = QRectF(0, 0, 400, 300)
    box = QRectF(390, 290, 400, 300)
    label = place_label(box, label_w=60, label_h=16, frame=frame)
    assert label.left() >= frame.left()
    assert label.top() >= frame.top()
    assert label.right() <= frame.right() + 1e-6
    assert label.bottom() <= frame.bottom() + 1e-6


def test_place_label_sits_above_box_when_there_is_room():
    frame = QRectF(0, 0, 400, 300)
    box = QRectF(100, 100, 150, 150)
    label = place_label(box, label_w=40, label_h=16, frame=frame)
    assert label.bottom() <= box.top() + 1e-6


# --- crosshair ---


def test_crosshair_drawn_at_snapshot_position_not_widget_centre(qapp):
    canvas = VideoCanvas()
    canvas.resize(400, 300)
    canvas.set_snapshot(_snapshot(frame=_frame(400, 300), crosshair_px=(0.15, 0.2)))
    image = _render_to_image(canvas)

    expected_x, expected_y = canvas.normalized_to_widget(0.15, 0.2)
    ok_color = QColor(theme.OK)
    assert _has_pixel_near(image, int(expected_x), int(expected_y), ok_color, radius=20)

    center_x, center_y = canvas.width() // 2, canvas.height() // 2
    assert not _has_pixel_near(image, center_x, center_y, ok_color, radius=5)


def test_offscreen_crosshair_draws_extra_arrow_beyond_the_reticle(qapp):
    canvas = VideoCanvas()
    canvas.resize(400, 300)
    common = {"frame": _frame(400, 300), "crosshair_px": (0.95, 0.5)}

    canvas.set_snapshot(_snapshot(crosshair_offscreen=False, **common))
    without_arrow = _render_to_image(canvas)

    canvas.set_snapshot(_snapshot(crosshair_offscreen=True, crosshair_bearing_deg=90.0, **common))
    with_arrow = _render_to_image(canvas)

    assert _images_differ(without_arrow, with_arrow)


def test_offscreen_arrow_direction_follows_bearing(qapp):
    canvas = VideoCanvas()
    canvas.resize(400, 300)
    common = {"frame": _frame(400, 300), "crosshair_px": (0.95, 0.5), "crosshair_offscreen": True}

    canvas.set_snapshot(_snapshot(crosshair_bearing_deg=90.0, **common))
    pointing_right = _render_to_image(canvas)

    canvas.set_snapshot(_snapshot(crosshair_bearing_deg=270.0, **common))
    pointing_left = _render_to_image(canvas)

    assert _images_differ(pointing_right, pointing_left)


def test_dimmed_crosshair_shown_for_estimated_intrinsics(qapp):
    canvas = VideoCanvas()
    canvas.resize(400, 300)
    canvas.set_snapshot(
        _snapshot(frame=_frame(400, 300, quality="estimated"), crosshair_px=(0.5, 0.5))
    )
    image = _render_to_image(canvas)
    expected_x, expected_y = canvas.normalized_to_widget(0.5, 0.5)
    # Dimmed uses TEXT_MUTED, not the normal OK green.
    assert not _has_pixel_near(image, int(expected_x), int(expected_y), QColor(theme.OK), radius=16)
    assert _has_pixel_near(
        image, int(expected_x), int(expected_y), QColor(theme.TEXT_MUTED), radius=16
    )


# --- Part 0a/0b fixes: badge clipping, crosshair edge inset ---


@pytest.mark.parametrize(
    "x,y",
    [
        (0.0, 0.0),  # top-left corner
        (400.0, 0.0),  # top-right corner
        (0.0, 300.0),  # bottom-left corner
        (400.0, 300.0),  # bottom-right corner
        (200.0, 0.0),  # top edge, centred
        (200.0, 300.0),  # bottom edge, centred
    ],
)
def test_calibration_badge_stays_inside_image_area_at_every_crosshair_position(x, y):
    frame = QRectF(0.0, 0.0, 400.0, 300.0)
    label = place_badge(x, y, radius=14.0, label_w=90.0, label_h=16.0, frame=frame)
    assert label.left() >= frame.left() - 1e-6
    assert label.right() <= frame.right() + 1e-6
    assert label.top() >= frame.top() - 1e-6
    assert label.bottom() <= frame.bottom() + 1e-6


@pytest.mark.parametrize(
    "x,y",
    [
        (0.0, 150.0),  # left edge
        (400.0, 150.0),  # right edge
        (200.0, 0.0),  # top edge
        (200.0, 300.0),  # bottom edge
        (0.0, 0.0),  # corner
        (400.0, 300.0),  # opposite corner
    ],
)
def test_pinned_crosshair_is_fully_visible_at_all_four_edges(x, y):
    frame = QRectF(0.0, 0.0, 400.0, 300.0)
    radius = 14.0
    inset_x, inset_y = inset_point(x, y, radius + 4.0, frame)
    # The whole reticle circle (centre +/- radius) must be inside the
    # frame, not just its centre point.
    assert inset_x - radius >= frame.left() - 1e-6
    assert inset_x + radius <= frame.right() + 1e-6
    assert inset_y - radius >= frame.top() - 1e-6
    assert inset_y + radius <= frame.bottom() + 1e-6


def test_inset_point_leaves_a_point_already_inside_the_margin_unchanged():
    frame = QRectF(0.0, 0.0, 400.0, 300.0)
    assert inset_point(200.0, 150.0, 18.0, frame) == (200.0, 150.0)


def test_lock_banner_never_overlaps_the_indicator_row_above_the_telemetry_strip():
    # Regression: HEDEF KİLİTLİ and TAKİP YARDIMI ON used to be positioned
    # from two independent magic numbers and visibly overlapped whenever
    # both were on screen at once -- found by looking at a real
    # screenshot, not by any pixel-sampling test.
    frame = QRectF(0.0, 0.0, 640.0, 480.0)
    strip_top = 400.0
    banner = lock_banner_rect(frame, strip_top)
    assert banner.bottom() <= indicator_row_top(strip_top)


# --- box colours ---


def test_box_pen_colours_follow_iff():
    canvas = VideoCanvas()
    assert canvas._pen_by_iff[IFF.HOSTILE].color() == QColor(theme.HOSTILE)
    assert canvas._pen_by_iff[IFF.FRIENDLY].color() == QColor(theme.FRIENDLY)
    assert canvas._pen_by_iff[IFF.UNKNOWN].color() == QColor(theme.UNKNOWN)


def test_hostile_track_renders_in_hostile_colour(qapp):
    canvas = VideoCanvas()
    canvas.resize(400, 300)
    track = dataclasses.replace(
        make_track(track_id=1, iff=IFF.HOSTILE, status=TrackStatus.CONFIRMED),
        bbox=(0.3, 0.3, 0.6, 0.6),
    )
    canvas.set_snapshot(
        _snapshot(frame=_frame(400, 300), tracks=[track], state=make_state(tracks=(track,)))
    )
    image = _render_to_image(canvas)
    assert _find_exact_color(image, QColor(theme.HOSTILE))


def test_friendly_track_renders_in_friendly_colour(qapp):
    canvas = VideoCanvas()
    canvas.resize(400, 300)
    track = dataclasses.replace(
        make_track(track_id=1, iff=IFF.FRIENDLY, status=TrackStatus.CONFIRMED),
        bbox=(0.3, 0.3, 0.6, 0.6),
    )
    canvas.set_snapshot(
        _snapshot(frame=_frame(400, 300), tracks=[track], state=make_state(tracks=(track,)))
    )
    image = _render_to_image(canvas)
    assert _find_exact_color(image, QColor(theme.FRIENDLY))


# --- buffer-copy guard ---


def test_frame_to_pixmap_survives_source_buffer_mutation():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)  # BGR
    pixmap = frame_to_pixmap(image)

    image[:] = (200, 200, 200)  # mutate the ORIGINAL array after conversion

    result = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
    sampled = QColor(result.pixel(5, 5))
    assert (sampled.blue(), sampled.green(), sampled.red()) == (10, 20, 30)


# --- click to aim ---


def test_click_emits_normalized_coordinates(qtbot):
    canvas = VideoCanvas()
    canvas.resize(400, 300)
    canvas.set_snapshot(_snapshot(frame=_frame(400, 300)))
    received = []
    canvas.clicked_normalized.connect(lambda x, y: received.append((x, y)))

    qtbot.addWidget(canvas)
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))

    assert len(received) == 1
    x, y = received[0]
    assert x == pytest.approx(0.5, abs=0.02)
    assert y == pytest.approx(0.5, abs=0.02)
