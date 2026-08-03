"""VideoCanvas: the centre widget. Displays the current frame and paints
every overlay element with QPainter -- bounding boxes, crosshair, lock
banner, header/telemetry strips.

Overlays are never burned into the frame with OpenCV: that would couple
presentation to detection, scale badly across resolutions (dev happens on
a 640x480 webcam, competition runs a 1920x1080 D435i), and end up baked
into anything recorded from the feed. QPainter draws over the displayed
pixmap at whatever resolution the widget actually is, independent of the
source frame's own resolution.

Repaints happen only from ``set_snapshot`` -- never a QTimer. A snapshot
that hasn't changed has nothing new to show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
)
from PyQt6.QtWidgets import QWidget

from celikkubbe.core import config
from celikkubbe.core.strings import TARGET_CLASS_TR, UI_LABEL_TR
from celikkubbe.core.types import (
    IFF,
    CameraIntrinsics,
    EngagementState,
    Stage,
    TargetClass,
    Track,
    TrackStatus,
)
from celikkubbe.geometry.solver import AimSolution
from celikkubbe.ui import theme
from celikkubbe.ui.snapshot import UiSnapshot
from celikkubbe.ui.theme import AngleGauge
from celikkubbe.vision.l2_color import is_range_estimate_unreliable

_HEADER_HEIGHT_PX = 22
_TELEMETRY_HEIGHT_PX = 46
# OPERATOR AKTİF / TAKİP YARDIMI ON, drawn just above the telemetry
# strip's pan/tilt rows. lock_banner_rect reads this too, rather than the
# lock banner independently guessing its own clearance -- the two used to
# be positioned from separate magic numbers and silently overlapped
# whenever a locked target coincided with the indicator row.
_INDICATOR_ROW_HEIGHT_PX = 16
_LOCK_BANNER_HEIGHT_PX = 20
_LOCK_BANNER_GAP_PX = 4
_LOCK_ENGAGEMENT_STATES = (
    EngagementState.S4_AIM,
    EngagementState.S5_ENGAGE,
    EngagementState.S6_ASSESS,
)
# Engagement-envelope rings around a locked crosshair, in real metres,
# projected to pixels at the locked target's range -- a purely visual aid,
# not a core/ decision threshold, so it lives here rather than config.py.
_ENVELOPE_RING_RADII_M = (0.3, 0.6)


@dataclass(frozen=True)
class LetterboxTransform:
    """Maps normalised [0,1]x[0,1] frame coordinates to widget pixels and
    back, accounting for the letterbox/pillarbox bars added to preserve
    the frame's own aspect ratio inside a differently-shaped widget.
    """

    offset_x: float
    offset_y: float
    displayed_w: float
    displayed_h: float

    def to_widget(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        return (
            self.offset_x + x_norm * self.displayed_w,
            self.offset_y + y_norm * self.displayed_h,
        )

    def to_normalized(self, x_px: float, y_px: float) -> tuple[float, float]:
        if self.displayed_w <= 0.0 or self.displayed_h <= 0.0:
            return 0.0, 0.0
        return (
            (x_px - self.offset_x) / self.displayed_w,
            (y_px - self.offset_y) / self.displayed_h,
        )


_EMPTY_TRANSFORM = LetterboxTransform(0.0, 0.0, 0.0, 0.0)


def compute_letterbox(
    widget_w: int, widget_h: int, frame_w: int, frame_h: int
) -> LetterboxTransform:
    """Fit ``frame_w x frame_h`` inside ``widget_w x widget_h`` preserving
    aspect ratio -- never stretching a 4:3 webcam or a 16:9 D435i frame to
    fill a differently-shaped widget.
    """
    if widget_w <= 0 or widget_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return _EMPTY_TRANSFORM
    widget_aspect = widget_w / widget_h
    frame_aspect = frame_w / frame_h
    if frame_aspect > widget_aspect:
        displayed_w = float(widget_w)
        displayed_h = displayed_w / frame_aspect
        offset_x = 0.0
        offset_y = (widget_h - displayed_h) / 2.0
    else:
        displayed_h = float(widget_h)
        displayed_w = displayed_h * frame_aspect
        offset_y = 0.0
        offset_x = (widget_w - displayed_w) / 2.0
    return LetterboxTransform(offset_x, offset_y, displayed_w, displayed_h)


def frame_to_pixmap(image: np.ndarray) -> QPixmap:
    """BGR numpy array -> QPixmap, at the source frame's own resolution.

    ``QImage(buffer, ...)`` does not take ownership of the buffer it is
    given -- without ``.copy()``, a source that reuses its internal
    buffer on the next ``read()`` (``pyrealsense2`` may; no current
    ``FrameSource`` does -- ``cv2.VideoCapture.read()`` and
    ``SyntheticSource``'s own ``np.full()``/``np.zeros()`` calls both
    allocate a fresh array every call) would leave this pixmap pointing
    at memory the pipeline thread has already started overwriting,
    producing garbage pixels or a crash. Copying unconditionally rather
    than trusting today's sources' safety is what actually guards
    against a future ``RealSenseSource`` landing here silently.
    """
    contiguous = np.ascontiguousarray(image)
    height, width = contiguous.shape[:2]
    bytes_per_line = contiguous.strides[0]
    qimage = QImage(
        contiguous.data, width, height, bytes_per_line, QImage.Format.Format_BGR888
    ).copy()
    return QPixmap.fromImage(qimage)


def place_label(box: QRectF, label_w: float, label_h: float, frame: QRectF) -> QRectF:
    """Where a detection label should sit: just above ``box``, flipped
    below it if that would go above the frame's top edge, then clamped
    on every side so a box near any edge never produces a label that
    spills outside the visible frame.

    A pure function, not inlined in the paint call, so the edge-clamping
    logic is directly testable without a full paint cycle.
    """
    x, y = box.left(), box.top() - label_h
    if y < frame.top():
        y = box.bottom()
    if x + label_w > frame.right():
        x = frame.right() - label_w
    if x < frame.left():
        x = frame.left()
    if y + label_h > frame.bottom():
        y = frame.bottom() - label_h
    return QRectF(x, y, label_w, label_h)


def inset_point(x: float, y: float, inset: float, frame: QRectF) -> tuple[float, float]:
    """Pulls (x, y) inward from every edge of ``frame`` by ``inset``, so a
    glyph of that radius centred on the returned point never crosses the
    boundary. A pure function so the off-screen crosshair's inset -- and
    that it holds at all four edges, not just the one a given test frame
    happens to exercise -- is directly testable.
    """
    left, top = frame.left() + inset, frame.top() + inset
    right, bottom = frame.right() - inset, frame.bottom() - inset
    return min(max(x, left), right), min(max(y, top), bottom)


def place_badge(
    x: float, y: float, radius: float, label_w: float, label_h: float, frame: QRectF
) -> QRectF:
    """Where a small badge caption below a point-and-radius marker (the
    crosshair's KALİBRE DEĞİL label) should sit: centred under it,
    flipped above if there is no room below, and clamped horizontally
    so it never spills past the image's left/right edges regardless of
    how close to one the marker itself is.
    """
    label_x = min(max(x - label_w / 2.0, frame.left()), frame.right() - label_w)
    label_y = y + radius + 4.0
    if label_y + label_h > frame.bottom():
        label_y = y - radius - 4.0 - label_h
    return QRectF(label_x, label_y, label_w, label_h)


def indicator_row_top(strip_top: float) -> float:
    """Top y-coordinate of the OPERATOR AKTİF / TAKİP YARDIMI ON row,
    which sits just above the telemetry strip itself.
    """
    return strip_top - _INDICATOR_ROW_HEIGHT_PX


def lock_banner_rect(frame: QRectF, strip_top: float) -> QRectF:
    """Where the HEDEF KİLİTLİ banner sits: centred across the image,
    directly above the indicator row with a fixed clearance gap -- both
    derive from the same ``_INDICATOR_ROW_HEIGHT_PX`` so they cannot
    silently drift back into overlapping each other.
    """
    bottom = indicator_row_top(strip_top) - _LOCK_BANNER_GAP_PX
    top = bottom - _LOCK_BANNER_HEIGHT_PX
    return QRectF(frame.left(), top, frame.width(), _LOCK_BANNER_HEIGHT_PX)


def class_label(cls: TargetClass | None, cls_source: str | None = None) -> str:
    """A track's class, in Turkish -- or BİLİNMEYEN when there is none.
    Shared by the box label, the lock banner, the target list and the
    locked-target readout so all four stay in sync, and so this is the
    one place to change if TARGET_CLASS_TR grows.

    ``cls_source`` is optional (defaults to None, the pre-existing
    behaviour) but every call site that has a full ``Track`` should pass
    ``track.cls_source``: an operator-assigned class must never be
    visually indistinguishable from a model-produced one, and this
    suffix is the one place that distinction is actually drawn.
    """
    label = TARGET_CLASS_TR[cls] if cls is not None else UI_LABEL_TR["UNKNOWN_CLASS"]
    if cls_source == "operator":
        return f"{label} {UI_LABEL_TR['CLASS_SOURCE_OPERATOR_TAG']}"
    return label


_IFF_COLOR: dict[IFF, str] = {
    IFF.HOSTILE: theme.HOSTILE,
    IFF.FRIENDLY: theme.FRIENDLY,
    IFF.UNKNOWN: theme.UNKNOWN,
}


class VideoCanvas(QWidget):
    clicked_normalized = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self._snapshot: UiSnapshot | None = None
        self._pixmap: QPixmap | None = None
        self._transform = _EMPTY_TRANSFORM
        self._source_label = ""

        self._pan_gauge = AngleGauge(*config.PAN_LIMIT_DEG, parent=self)
        self._tilt_gauge = AngleGauge(*config.TILT_LIMIT_DEG, parent=self)

        self._label_font = QFont()
        self._label_font.setPointSizeF(9.0)
        self._strip_font = QFont()
        self._strip_font.setPointSizeF(9.5)

        # Built once, not per paintEvent -- see the module docstring's
        # performance note.
        self._pen_by_iff = {iff: QPen(QColor(color)) for iff, color in _IFF_COLOR.items()}
        for pen in self._pen_by_iff.values():
            pen.setWidth(2)
        self._pen_dim_text = QPen(QColor(theme.TEXT_DIM))
        self._pen_warn = QPen(QColor(theme.WARN))
        self._pen_danger = QPen(QColor(theme.DANGER))
        self._brush_label_bg = QBrush(QColor(theme.BG_BASE))

        self._layout_gauges()

    # --- public API ---

    def set_source_label(self, label: str) -> None:
        self._source_label = label
        self.update()

    def set_snapshot(self, snapshot: UiSnapshot) -> None:
        """The only place this widget repaints from -- never a timer."""
        self._snapshot = snapshot
        self._pixmap = frame_to_pixmap(snapshot.frame.image)
        self._recompute_transform()
        # Not just resizeEvent: the transform first becomes non-empty
        # here, on the first frame ever received, and resizeEvent may
        # not fire again for the rest of the session.
        self._layout_gauges()
        self._update_gauges(snapshot)
        self.update()

    def transform(self) -> LetterboxTransform:
        return self._transform

    def widget_to_normalized(self, x_px: float, y_px: float) -> tuple[float, float]:
        return self._transform.to_normalized(x_px, y_px)

    def normalized_to_widget(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        return self._transform.to_widget(x_norm, y_norm)

    # --- Qt overrides ---

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._recompute_transform()
        self._layout_gauges()
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            x_norm, y_norm = self.widget_to_normalized(pos.x(), pos.y())
            if 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0:
                self.clicked_normalized.emit(x_norm, y_norm)
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.BG_BASE))

        if self._pixmap is not None and self._transform.displayed_w > 0:
            target = QRectF(
                self._transform.offset_x,
                self._transform.offset_y,
                self._transform.displayed_w,
                self._transform.displayed_h,
            )
            painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if self._snapshot is not None:
            self._draw_overlay(painter, self._snapshot)
        painter.end()

    # --- layout ---

    def _recompute_transform(self) -> None:
        if self._pixmap is None:
            self._transform = _EMPTY_TRANSFORM
            return
        self._transform = compute_letterbox(
            self.width(), self.height(), self._pixmap.width(), self._pixmap.height()
        )

    def _layout_gauges(self) -> None:
        """Anchored to the letterboxed image's own bottom edge, like
        _draw_telemetry_strip's text -- not the canvas widget's raw
        bottom, which whenever there is a letterbox margin (e.g. a 4:3
        frame in a taller-than-4:3 widget) sits well below where the
        telemetry strip's text is actually drawn, leaving a dead gap
        between a value and its gauge. Falls back to the widget's own
        bottom before any frame has ever arrived, when the transform is
        still empty (offset 0, no displayed area).
        """
        if self._pixmap is not None:
            image_bottom = self._transform.offset_y + self._transform.displayed_h
        else:
            image_bottom = self.height()
        strip_top = int(image_bottom) - _TELEMETRY_HEIGHT_PX
        gauge_w = max(60, self.width() // 3)
        self._pan_gauge.setGeometry(90, strip_top + 8, gauge_w, 12)
        self._tilt_gauge.setGeometry(90, strip_top + 26, gauge_w, 12)

    def _update_gauges(self, snapshot: UiSnapshot) -> None:
        telemetry = snapshot.telemetry
        self._pan_gauge.set_value(telemetry.pan_deg if telemetry is not None else None)
        self._tilt_gauge.set_value(telemetry.tilt_deg if telemetry is not None else None)

    # --- overlay ---

    def _draw_overlay(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        self._draw_boxes(painter, snapshot)
        self._draw_crosshair(painter, snapshot)
        self._draw_lock_banner(painter, snapshot)
        self._draw_header_strip(painter, snapshot)
        self._draw_telemetry_strip(painter, snapshot)

    def _draw_boxes(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        painter.setFont(self._label_font)
        for track in snapshot.tracks:
            self._draw_one_box(painter, track, snapshot)

    def _draw_one_box(self, painter: QPainter, track: Track, snapshot: UiSnapshot) -> None:
        pen = QPen(self._pen_by_iff[track.iff])
        if track.status is not TrackStatus.CONFIRMED:
            pen.setStyle(Qt.PenStyle.DashLine)
        if track.track_id == snapshot.state.selected_track_id:
            pen.setWidth(4)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        x1, y1 = self.normalized_to_widget(track.bbox[0], track.bbox[1])
        x2, y2 = self.normalized_to_widget(track.bbox[2], track.bbox[3])
        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        painter.drawRect(rect)

        self._draw_label_near(
            painter, rect, self._format_label(track, snapshot.frame.intrinsics), pen.color()
        )

        bbox_w_px = (track.bbox[2] - track.bbox[0]) * self._transform.displayed_w
        if 0.0 < bbox_w_px < config.MIN_TARGET_PX:
            self._draw_under_resolution_marker(painter, rect)

    @staticmethod
    def _format_label(track: Track, intrinsics: CameraIntrinsics) -> str:
        cls_label = class_label(track.cls, track.cls_source)
        range_str = "--"
        if track.range_m is not None:
            prefix = "~" if track.range_source == "size" else ""
            range_str = f"{prefix}{track.range_m:.1f}m"
            if is_range_estimate_unreliable(track, intrinsics):
                range_str += "?"
        return f"{cls_label} {track.confidence:.0%} {range_str}"

    def _draw_label_near(self, painter: QPainter, rect: QRectF, text: str, color: QColor) -> None:
        metrics = painter.fontMetrics().boundingRect(text)
        pad = 3
        label_w = metrics.width() + 2 * pad
        label_h = metrics.height() + 2 * pad
        label_rect = place_label(rect, label_w, label_h, self._image_rect())

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._brush_label_bg)
        painter.drawRect(label_rect)
        painter.setBrush(Qt.BrushStyle.NoBrush)  # don't leak a solid brush into the next draw call
        painter.setPen(QPen(color))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_under_resolution_marker(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(self._pen_warn)
        marker_rect = QRectF(rect.left(), rect.bottom() + 1, rect.width(), 14)
        painter.drawText(marker_rect, Qt.AlignmentFlag.AlignCenter, "!")

    def _draw_crosshair(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        if snapshot.crosshair_px is None:
            return
        radius = 14.0
        x, y = self.normalized_to_widget(*snapshot.crosshair_px)
        x, y = self._inset_offscreen_position(x, y, snapshot.crosshair_offscreen, radius)
        low_confidence = snapshot.aim is not None and snapshot.aim.confidence == "low"
        estimated = not snapshot.frame.intrinsics.is_reliable
        dimmed = low_confidence or estimated
        color = QColor(theme.TEXT_MUTED if dimmed else theme.OK)

        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)  # an outline reticle, not a filled disc
        painter.drawEllipse(QPointF(x, y), radius, radius)
        for dx0, dx1, dy0, dy1 in (
            (-radius - 6, -radius + 4, 0, 0),
            (radius - 4, radius + 6, 0, 0),
            (0, 0, -radius - 6, -radius + 4),
            (0, 0, radius - 4, radius + 6),
        ):
            painter.drawLine(QPointF(x + dx0, y + dy0), QPointF(x + dx1, y + dy1))

        if snapshot.aim is not None:
            self._draw_envelope_rings(painter, x, y, snapshot.aim, snapshot.frame.intrinsics, color)

        if snapshot.crosshair_offscreen and snapshot.crosshair_bearing_deg is not None:
            self._draw_offscreen_arrow(painter, x, y, snapshot.crosshair_bearing_deg, color)

        if dimmed:
            self._draw_calibration_badge(painter, x, y, radius, color)

    def _inset_offscreen_position(
        self, x: float, y: float, offscreen: bool, radius: float
    ) -> tuple[float, float]:
        """Pulls the pinned crosshair in from the image edge by the
        reticle's own radius plus a small margin, so the full glyph --
        not just its centre point -- stays inside the visible image area.
        Without this, a crosshair pinned exactly at u_norm/v_norm 0 or 1
        (see crosshair_with_indicator) draws with its far half clipped by
        the image bounds, or by the canvas widget's own bounds when there
        is no letterbox margin on that side.
        """
        if not offscreen:
            return x, y
        frame = self._image_rect()
        return inset_point(x, y, radius + 4.0, frame)

    def _draw_calibration_badge(
        self, painter: QPainter, x: float, y: float, radius: float, color: QColor
    ) -> None:
        """Centred under the crosshair, flipped above it -- and always
        clamped horizontally inside the image area -- rather than a
        fixed-size rect that clips whenever the crosshair sits near an
        edge (the actual bug: KALİBRE DEĞİL's leading K cut off, or most
        of the text gone when pinned at the frame edge).
        """
        text = UI_LABEL_TR["NOT_CALIBRATED"]
        metrics = painter.fontMetrics().boundingRect(text)
        pad = 4
        label_w = metrics.width() + 2 * pad
        label_h = metrics.height() + 2 * pad

        label_rect = place_badge(x, y, radius, label_w, label_h, self._image_rect())
        painter.setPen(QPen(color))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _image_rect(self) -> QRectF:
        return QRectF(
            self._transform.offset_x,
            self._transform.offset_y,
            self._transform.displayed_w,
            self._transform.displayed_h,
        )

    def _draw_envelope_rings(
        self,
        painter: QPainter,
        x: float,
        y: float,
        aim: AimSolution,
        intr: CameraIntrinsics,
        color: QColor,
    ) -> None:
        """Rings sized in real metres, projected to pixels at the locked
        target's own range -- nearer targets get visibly bigger rings,
        the same way a real engagement envelope would look larger up
        close, rather than a fixed pixel radius that means nothing
        physical.
        """
        range_m = max(aim.range_m, 1e-3)
        ring_pen = QPen(color)
        ring_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(ring_pen)
        for radius_m in _ENVELOPE_RING_RADII_M:
            radius_px = intr.fx * radius_m / range_m
            painter.drawEllipse(QPointF(x, y), radius_px, radius_px)

    def _draw_offscreen_arrow(
        self, painter: QPainter, x: float, y: float, bearing_deg: float, color: QColor
    ) -> None:
        """(x, y) is already clamped to the image area's own edge (see
        crosshair_with_indicator), which -- whenever there is no
        letterbox margin on that side -- coincides exactly with the
        canvas widget's own boundary too, and Qt clips anything drawn
        outside a widget's rect. An arrowhead pointing further outward
        from (x, y) would therefore render entirely off-widget and be
        invisible exactly when it matters. Instead the tip sits at the
        anchor and the base trails inward, so the whole shape stays
        inside the visible canvas regardless of letterboxing.
        """
        painter.save()
        painter.translate(x, y)
        painter.rotate(bearing_deg)  # clockwise from straight up, matching crosshair_with_indicator
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        arrow = QPolygonF([QPointF(0, -2), QPointF(-6, 12), QPointF(6, 12)])
        painter.drawPolygon(arrow)
        painter.restore()

    def _draw_lock_banner(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        selected_id = snapshot.state.selected_track_id
        if selected_id is None or snapshot.state.engagement not in _LOCK_ENGAGEMENT_STATES:
            return
        track = next((t for t in snapshot.tracks if t.track_id == selected_id), None)
        if track is None:
            return
        cls_label = class_label(track.cls, track.cls_source)
        text = f"{UI_LABEL_TR['TARGET_LOCKED']} — {cls_label}"

        painter.setFont(self._strip_font)
        painter.setPen(self._pen_danger)
        strip_top = self._transform.offset_y + self._transform.displayed_h - _TELEMETRY_HEIGHT_PX
        rect = lock_banner_rect(self._image_rect(), strip_top)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_header_strip(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        painter.setFont(self._strip_font)
        intr = snapshot.frame.intrinsics
        n_hostile = sum(1 for t in snapshot.tracks if t.iff is IFF.HOSTILE)
        n_friendly = sum(1 for t in snapshot.tracks if t.iff is IFF.FRIENDLY)
        n_unknown = len(snapshot.tracks) - n_hostile - n_friendly

        rect = QRectF(
            self._transform.offset_x + 6,
            self._transform.offset_y + 2,
            self._transform.displayed_w - 12,
            _HEADER_HEIGHT_PX,
        )
        painter.setPen(self._pen_dim_text)
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._source_label} {intr.width}x{intr.height}",
        )
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"H:{n_hostile} F:{n_friendly} U:{n_unknown}",
        )

    def _draw_telemetry_strip(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        painter.setFont(self._strip_font)
        strip_top = self._transform.offset_y + self._transform.displayed_h - _TELEMETRY_HEIGHT_PX
        left = self._transform.offset_x + 6

        state = snapshot.state
        telemetry = snapshot.telemetry

        # Mode/engagement already have their own badges on the status
        # strip below the canvas -- showing "M3_OPERATIONAL / S4_AIM"
        # here too was pure duplication. This space is two fixed
        # indicators instead, each lit or dimmed by real state rather
        # than changing text, the same way the calibration badge above
        # is shown or not rather than reworded.
        operator_active = state.stage is Stage.STAGE_1 and bool(
            telemetry is not None and telemetry.armed
        )
        tracking_aid_on = state.selected_track_id is not None
        operator_pen = QPen(QColor(theme.OK if operator_active else theme.TEXT_MUTED))
        tracking_pen = QPen(QColor(theme.OK if tracking_aid_on else theme.TEXT_MUTED))

        indicator_top = indicator_row_top(strip_top)
        painter.setPen(operator_pen)
        painter.drawText(
            QRectF(left, indicator_top, self._transform.displayed_w - 12, 14),
            Qt.AlignmentFlag.AlignLeft,
            UI_LABEL_TR["OPERATOR_ACTIVE"],
        )
        painter.setPen(tracking_pen)
        painter.drawText(
            QRectF(left + 140, indicator_top, self._transform.displayed_w - 12, 14),
            Qt.AlignmentFlag.AlignLeft,
            UI_LABEL_TR["TRACKING_AID_ON"],
        )

        pan_label, tilt_label = UI_LABEL_TR["AXIS_PAN"], UI_LABEL_TR["AXIS_TILT"]
        if telemetry is not None:
            pan_text = f"{pan_label} {telemetry.pan_deg:+.1f}°"
            tilt_text = f"{tilt_label} {telemetry.tilt_deg:+.1f}°"
        else:
            pan_text, tilt_text = f"{pan_label} --", f"{tilt_label} --"
        align_left = Qt.AlignmentFlag.AlignLeft
        painter.drawText(QRectF(left, strip_top + 6, 80, 14), align_left, pan_text)
        painter.drawText(QRectF(left, strip_top + 24, 80, 14), align_left, tilt_text)
