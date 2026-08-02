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
from celikkubbe.core.strings import UI_LABEL_TR
from celikkubbe.core.types import IFF, CameraIntrinsics, EngagementState, Track, TrackStatus
from celikkubbe.geometry.solver import AimSolution
from celikkubbe.ui import theme
from celikkubbe.ui.snapshot import UiSnapshot
from celikkubbe.ui.theme import AngleGauge

_HEADER_HEIGHT_PX = 22
_TELEMETRY_HEIGHT_PX = 46
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
        strip_top = self.height() - _TELEMETRY_HEIGHT_PX
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

        self._draw_label_near(painter, rect, self._format_label(track), pen.color())

        bbox_w_px = (track.bbox[2] - track.bbox[0]) * self._transform.displayed_w
        if 0.0 < bbox_w_px < config.MIN_TARGET_PX:
            self._draw_under_resolution_marker(painter, rect)

    @staticmethod
    def _format_label(track: Track) -> str:
        cls_label = track.cls.value if track.cls is not None else UI_LABEL_TR["UNKNOWN_CLASS"]
        range_str = "--"
        if track.range_m is not None:
            prefix = "~" if track.range_source == "size" else ""
            range_str = f"{prefix}{track.range_m:.1f}m"
        return f"{cls_label} {track.confidence:.0%} {range_str}"

    def _draw_label_near(self, painter: QPainter, rect: QRectF, text: str, color: QColor) -> None:
        metrics = painter.fontMetrics().boundingRect(text)
        pad = 3
        label_w = metrics.width() + 2 * pad
        label_h = metrics.height() + 2 * pad
        frame = QRectF(
            self._transform.offset_x,
            self._transform.offset_y,
            self._transform.displayed_w,
            self._transform.displayed_h,
        )
        label_rect = place_label(rect, label_w, label_h, frame)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._brush_label_bg)
        painter.drawRect(label_rect)
        painter.setPen(QPen(color))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_under_resolution_marker(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(self._pen_warn)
        marker_rect = QRectF(rect.left(), rect.bottom() + 1, rect.width(), 14)
        painter.drawText(marker_rect, Qt.AlignmentFlag.AlignCenter, "!")

    def _draw_crosshair(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        if snapshot.crosshair_px is None:
            return
        x, y = self.normalized_to_widget(*snapshot.crosshair_px)
        low_confidence = snapshot.aim is not None and snapshot.aim.confidence == "low"
        estimated = not snapshot.frame.intrinsics.is_reliable
        dimmed = low_confidence or estimated
        color = QColor(theme.TEXT_MUTED if dimmed else theme.OK)

        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        radius = 14.0
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
            painter.setPen(QPen(color))
            painter.drawText(
                QRectF(x - 45, y + radius + 4, 90, 14),
                Qt.AlignmentFlag.AlignCenter,
                UI_LABEL_TR["NOT_CALIBRATED"],
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
        painter.save()
        painter.translate(x, y)
        painter.rotate(bearing_deg)  # clockwise from straight up, matching crosshair_with_indicator
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        arrow = QPolygonF([QPointF(0, -20), QPointF(-6, -10), QPointF(6, -10)])
        painter.drawPolygon(arrow)
        painter.restore()

    def _draw_lock_banner(self, painter: QPainter, snapshot: UiSnapshot) -> None:
        selected_id = snapshot.state.selected_track_id
        if selected_id is None or snapshot.state.engagement not in _LOCK_ENGAGEMENT_STATES:
            return
        track = next((t for t in snapshot.tracks if t.track_id == selected_id), None)
        if track is None:
            return
        cls_label = track.cls.value if track.cls is not None else UI_LABEL_TR["UNKNOWN_CLASS"]
        text = f"{UI_LABEL_TR['TARGET_LOCKED']} — {cls_label}"

        painter.setFont(self._strip_font)
        painter.setPen(self._pen_danger)
        frame_bottom = self._transform.offset_y + self._transform.displayed_h
        rect = QRectF(
            self._transform.offset_x,
            frame_bottom - _TELEMETRY_HEIGHT_PX - 26,
            self._transform.displayed_w,
            20,
        )
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
        status_text = f"{state.mode.value} / {state.engagement.value}"
        painter.setPen(self._pen_dim_text)
        painter.drawText(
            QRectF(left, strip_top - 16, self._transform.displayed_w - 12, 14),
            Qt.AlignmentFlag.AlignLeft,
            status_text,
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
