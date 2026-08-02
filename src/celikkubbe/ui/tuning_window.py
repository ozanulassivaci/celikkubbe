"""TuningWindow: live HSV tuning for the L2 colour detector.

A non-modal QDialog: sliders on the left mutate the *running*
PipelineWorker's own ColorDetector.config directly (see
PipelineWorker.detector's own docstring on why that means swapping the
whole config object, never individual fields in place), and the live
preview on the right re-runs detect(frame, debug=True) on every fresh
snapshot while this window is visible, showing the source frame, either
class's HSV/morphed mask, or the final accepted/rejected contours.

morph_kernel / min_area_px / circularity_min / require_circularity are
genuinely global fields on ColorDetectorConfig, shared by every class --
not, despite how a tuning UI might naively group them, a per-class
setting duplicated in each class's section. Only hue_ranges/sat_min/
val_min live on ColorClass itself.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from celikkubbe.core.strings import UI_LABEL_TR
from celikkubbe.core.types import BoundingBox
from celikkubbe.ui import theme
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.ui.snapshot import UiSnapshot
from celikkubbe.ui.video_canvas import LetterboxTransform, compute_letterbox, frame_to_pixmap
from celikkubbe.vision.l2_color import (
    DEFAULT_HSV_PRESETS_DIR,
    ColorDetectorConfig,
    DebugMasks,
    list_hsv_presets,
    load_hsv_preset,
    save_hsv_preset,
)

# Refresh cadence for the live preview -- deliberately below the
# pipeline's own 30Hz: a human tuning sliders by eye gets nothing from
# redrawing faster than they can perceive, and re-running detect() with
# debug=True on the GUI thread on every single snapshot would add
# needless load precisely while this window is open and being watched.
_PREVIEW_UPDATE_HZ = 10.0
# A drag shorter than this (normalised) is treated as an accidental
# click, not an intentional ROI -- clicking to dismiss/inspect the
# preview must not silently install a near-zero-area ROI.
_MIN_ROI_DRAG = 0.02
_EMPTY_TRANSFORM = LetterboxTransform(0.0, 0.0, 0.0, 0.0)

_PREVIEW_MODES = ("source", "hsv", "morphed", "contours")
_PREVIEW_MODE_LABEL_KEY = {
    "source": "PREVIEW_SOURCE",
    "hsv": "PREVIEW_HSV_MASK",
    "morphed": "PREVIEW_MORPHED_MASK",
    "contours": "PREVIEW_CONTOURS",
}
# ROI dragging is only meaningful over a full-frame preview: the hsv/
# morphed masks are already cropped to the current ROI (see
# l2_color.ColorDetector.detect), so overlaying a *new* ROI rectangle on
# top of an image that is already just the old ROI's contents would not
# line up with anything.
_ROI_DRAG_MODES = ("source", "contours")


def normalize_drag_to_roi(p1: tuple[float, float], p2: tuple[float, float]) -> BoundingBox | None:
    """Two normalised drag endpoints -> a clamped (x1,y1,x2,y2), or None
    if the drag is too small to be an intentional ROI. A pure function so
    the clamping and minimum-size rules are directly testable without a
    paint cycle, the same reasoning video_canvas.py's own placement
    helpers were extracted for.
    """
    x1, x2 = sorted((p1[0], p2[0]))
    y1, y2 = sorted((p1[1], p2[1]))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 - x1 < _MIN_ROI_DRAG or y2 - y1 < _MIN_ROI_DRAG:
        return None
    return (x1, y1, x2, y2)


class _PreviewWidget(QWidget):
    """Displays one pixmap, optionally with a ROI rectangle overlaid, and
    lets the operator drag out a new ROI directly on the image.
    """

    roi_dragged = pyqtSignal(tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 270)
        self._pixmap: QPixmap | None = None
        self._roi: BoundingBox | None = None
        self._roi_draggable = True
        self._transform = _EMPTY_TRANSFORM
        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None

    def set_pixmap(self, pixmap: QPixmap) -> None:
        # Recomputed here, not only in paintEvent: mouse handling needs an
        # up-to-date transform independent of paint timing, the same
        # reasoning video_canvas.VideoCanvas.set_snapshot already applies
        # to its own _recompute_transform call.
        self._pixmap = pixmap
        if pixmap.width() > 0:
            self._transform = compute_letterbox(
                self.width(), self.height(), pixmap.width(), pixmap.height()
            )
        self.update()

    def set_roi(self, roi: BoundingBox | None, draggable: bool) -> None:
        self._roi = roi
        self._roi_draggable = draggable
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.BG_BASE))

        if self._pixmap is not None and self._pixmap.width() > 0:
            self._transform = compute_letterbox(
                self.width(), self.height(), self._pixmap.width(), self._pixmap.height()
            )
            target = QRectF(
                self._transform.offset_x,
                self._transform.offset_y,
                self._transform.displayed_w,
                self._transform.displayed_h,
            )
            painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if self._roi is not None and self._roi_draggable:
            self._draw_roi(painter, self._roi, theme.OK)
        if self._drag_start is not None and self._drag_current is not None:
            pen = QPen(QColor(theme.WARN))
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self._drag_start, self._drag_current).normalized())
        painter.end()

    def _draw_roi(self, painter: QPainter, roi: BoundingBox, color: str) -> None:
        x1, y1 = self._transform.to_widget(roi[0], roi[1])
        x2, y2 = self._transform.to_widget(roi[2], roi[3])
        pen = QPen(QColor(color))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._roi_draggable and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position()
            self._drag_current = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag_start is not None:
            self._drag_current = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag_start is None or self._drag_current is None:
            return
        p1 = self._transform.to_normalized(self._drag_start.x(), self._drag_start.y())
        p2 = self._transform.to_normalized(self._drag_current.x(), self._drag_current.y())
        self._drag_start = None
        self._drag_current = None
        self.update()
        roi = normalize_drag_to_roi(p1, p2)
        if roi is not None:
            self.roi_dragged.emit(roi)


def _slider_row(label_text: str, lo: int, hi: int, value: int) -> tuple[QWidget, QSlider, QLabel]:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(label_text)
    label.setFixedWidth(130)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(value)
    value_label = QLabel(str(value))
    value_label.setFixedWidth(36)
    layout.addWidget(label)
    layout.addWidget(slider, stretch=1)
    layout.addWidget(value_label)
    return container, slider, value_label


class TuningWindow(QDialog):
    def __init__(
        self,
        worker: PipelineWorker,
        presets_dir: Path = DEFAULT_HSV_PRESETS_DIR,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(UI_LABEL_TR["TUNING_TITLE"])
        self.resize(1150, 720)

        self._worker = worker
        self._presets_dir = presets_dir
        self._detector = worker.detector
        self._working_config: ColorDetectorConfig = self._detector.config
        self._latest_frame = None
        self._latest_debug: DebugMasks | None = None
        self._last_preview_t: float | None = None
        self._sliders: dict[str, tuple[QSlider, QLabel]] = {}

        outer = QHBoxLayout(self)
        outer.addWidget(self._build_controls(), stretch=0)
        outer.addWidget(self._build_preview_column(), stretch=1)

        self._sync_widgets_from_config()
        worker.snapshot_ready.connect(self._on_snapshot)

    # --- construction: controls column ---

    def _build_controls(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._build_global_group())
        layout.addWidget(self._build_class_group("hostile", UI_LABEL_TR["HOSTILE_CLASS_TITLE"]))
        layout.addWidget(self._build_class_group("friendly", UI_LABEL_TR["FRIENDLY_CLASS_TITLE"]))
        layout.addWidget(self._build_presets_group())
        layout.addStretch(1)
        close_btn = QPushButton(UI_LABEL_TR["CLOSE_BUTTON"])
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        scroll.setWidget(container)
        return scroll

    def _build_global_group(self) -> QGroupBox:
        box = QGroupBox(UI_LABEL_TR["GLOBAL_SECTION_TITLE"])
        layout = QVBoxLayout(box)

        row, slider, label = _slider_row(UI_LABEL_TR["MORPH_KERNEL_LABEL"], 1, 15, 5)
        slider.valueChanged.connect(self._set_morph_kernel)
        self._sliders["morph_kernel"] = (slider, label)
        layout.addWidget(row)

        row, slider, label = _slider_row(UI_LABEL_TR["MIN_AREA_LABEL"], 1, 200, 12)
        slider.valueChanged.connect(self._set_min_area)
        self._sliders["min_area"] = (slider, label)
        layout.addWidget(row)

        row, slider, label = _slider_row(UI_LABEL_TR["CIRCULARITY_MIN_LABEL"], 0, 100, 70)
        slider.valueChanged.connect(self._set_circularity_min)
        self._sliders["circularity_min"] = (slider, label)
        layout.addWidget(row)

        self._require_circularity_checkbox = QCheckBox(UI_LABEL_TR["REQUIRE_CIRCULARITY_LABEL"])
        self._require_circularity_checkbox.toggled.connect(self._set_require_circularity)
        layout.addWidget(self._require_circularity_checkbox)
        return box

    def _build_class_group(self, class_name: str, title: str) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)

        hue_labels = (
            (UI_LABEL_TR["HUE_RANGE_1_LABEL"], UI_LABEL_TR["HUE_RANGE_2_LABEL"])
            if class_name == "hostile"
            else (UI_LABEL_TR["HUE_RANGE_LABEL"], None)
        )
        range_count = 2 if class_name == "hostile" else 1
        for range_index in range(range_count):
            for bound_index, suffix in ((0, "lo"), (1, "hi")):
                key = f"{class_name}_hue{range_index}_{suffix}"
                label_text = f"{hue_labels[range_index]} {suffix.upper()}"
                row, slider, label = _slider_row(label_text, 0, 179, 0)
                slider.valueChanged.connect(
                    lambda v, cn=class_name, ri=range_index, bi=bound_index: self._set_class_hue(
                        cn, ri, bi, v
                    )
                )
                self._sliders[key] = (slider, label)
                layout.addWidget(row)

        row, slider, label = _slider_row(UI_LABEL_TR["SAT_MIN_LABEL"], 0, 255, 90)
        slider.valueChanged.connect(lambda v, cn=class_name: self._set_class_sat(cn, v))
        self._sliders[f"{class_name}_sat"] = (slider, label)
        layout.addWidget(row)

        row, slider, label = _slider_row(UI_LABEL_TR["VAL_MIN_LABEL"], 0, 255, 50)
        slider.valueChanged.connect(lambda v, cn=class_name: self._set_class_val(cn, v))
        self._sliders[f"{class_name}_val"] = (slider, label)
        layout.addWidget(row)
        return box

    def _build_presets_group(self) -> QGroupBox:
        box = QGroupBox()
        layout = QVBoxLayout(box)

        save_row = QHBoxLayout()
        self._preset_name_edit = QLineEdit()
        self._preset_name_edit.setPlaceholderText(UI_LABEL_TR["PRESET_NAME_PLACEHOLDER"])
        save_btn = QPushButton(UI_LABEL_TR["SAVE_PRESET"])
        save_btn.clicked.connect(self._on_save_preset)
        save_row.addWidget(self._preset_name_edit)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

        load_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        load_btn = QPushButton(UI_LABEL_TR["LOAD_PRESET"])
        load_btn.clicked.connect(self._on_load_preset)
        load_row.addWidget(self._preset_combo, stretch=1)
        load_row.addWidget(load_btn)
        layout.addLayout(load_row)
        self._refresh_preset_list()

        reset_btn = QPushButton(UI_LABEL_TR["RESET_DEFAULTS"])
        reset_btn.clicked.connect(self._on_reset_defaults)
        layout.addWidget(reset_btn)
        return box

    # --- construction: preview column ---

    def _build_preview_column(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(UI_LABEL_TR["PREVIEW_MODE_LABEL"]))
        self._mode_combo = QComboBox()
        for mode in _PREVIEW_MODES:
            self._mode_combo.addItem(UI_LABEL_TR[_PREVIEW_MODE_LABEL_KEY[mode]], mode)
        self._mode_combo.currentIndexChanged.connect(lambda _i: self._render_preview())
        mode_row.addWidget(self._mode_combo)

        mode_row.addWidget(QLabel(UI_LABEL_TR["PREVIEW_CLASS_LABEL"]))
        self._class_combo = QComboBox()
        self._class_combo.addItem(UI_LABEL_TR["HOSTILE_CLASS_TITLE"], "hostile")
        self._class_combo.addItem(UI_LABEL_TR["FRIENDLY_CLASS_TITLE"], "friendly")
        self._class_combo.currentIndexChanged.connect(lambda _i: self._render_preview())
        mode_row.addWidget(self._class_combo)

        clear_roi_btn = QPushButton(UI_LABEL_TR["CLEAR_ROI"])
        clear_roi_btn.clicked.connect(self._on_clear_roi)
        mode_row.addWidget(clear_roi_btn)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self._preview = _PreviewWidget()
        self._preview.roi_dragged.connect(self._on_roi_dragged)
        layout.addWidget(self._preview, stretch=1)

        self._counts_label = QLabel("")
        layout.addWidget(self._counts_label)
        return container

    # --- live-apply: global fields ---

    def _apply_config(self, new_config: ColorDetectorConfig) -> None:
        self._working_config = new_config
        self._detector.config = new_config
        self._render_preview()

    def _set_morph_kernel(self, value: int) -> None:
        self._apply_config(dataclasses.replace(self._working_config, morph_kernel=value))

    def _set_min_area(self, value: int) -> None:
        self._apply_config(dataclasses.replace(self._working_config, min_area_px=value))

    def _set_circularity_min(self, value: int) -> None:
        self._apply_config(dataclasses.replace(self._working_config, circularity_min=value / 100.0))

    def _set_require_circularity(self, checked: bool) -> None:
        self._apply_config(dataclasses.replace(self._working_config, require_circularity=checked))

    # --- live-apply: per-class fields ---

    def _set_class_hue(
        self, class_name: str, range_index: int, bound_index: int, value: int
    ) -> None:
        classes = []
        for color_class in self._working_config.classes:
            if color_class.name != class_name:
                classes.append(color_class)
                continue
            ranges = list(color_class.hue_ranges)
            lo, hi = ranges[range_index]
            ranges[range_index] = (value, hi) if bound_index == 0 else (lo, value)
            classes.append(dataclasses.replace(color_class, hue_ranges=tuple(ranges)))
        self._apply_config(dataclasses.replace(self._working_config, classes=tuple(classes)))

    def _set_class_sat(self, class_name: str, value: int) -> None:
        classes = tuple(
            dataclasses.replace(c, sat_min=value) if c.name == class_name else c
            for c in self._working_config.classes
        )
        self._apply_config(dataclasses.replace(self._working_config, classes=classes))

    def _set_class_val(self, class_name: str, value: int) -> None:
        classes = tuple(
            dataclasses.replace(c, val_min=value) if c.name == class_name else c
            for c in self._working_config.classes
        )
        self._apply_config(dataclasses.replace(self._working_config, classes=classes))

    # --- ROI ---

    def _on_roi_dragged(self, roi: BoundingBox) -> None:
        self._apply_config(dataclasses.replace(self._working_config, roi=roi))

    def _on_clear_roi(self) -> None:
        self._apply_config(dataclasses.replace(self._working_config, roi=None))

    # --- presets ---

    def _refresh_preset_list(self) -> None:
        self._preset_combo.clear()
        self._preset_combo.addItems(list_hsv_presets(self._presets_dir))

    def _on_save_preset(self) -> None:
        name = self._preset_name_edit.text().strip()
        if not name:
            return
        save_hsv_preset(self._working_config, self._presets_dir / f"{name}.json")
        self._refresh_preset_list()

    def _on_load_preset(self) -> None:
        name = self._preset_combo.currentText()
        if not name:
            return
        loaded = load_hsv_preset(self._presets_dir / f"{name}.json")
        self._apply_config(loaded)
        self._sync_widgets_from_config()

    def _on_reset_defaults(self) -> None:
        self._apply_config(ColorDetectorConfig())
        self._sync_widgets_from_config()

    def _sync_widgets_from_config(self) -> None:
        """Pushes ``self._working_config`` into every slider/checkbox --
        needed after a preset load or a reset, both of which replace the
        whole config at once rather than one field at a time. Signals are
        blocked during the pushes: QSlider.valueChanged (unlike
        QAbstractButton.clicked -- see right_panel.py's own toggle-sync
        reasoning) fires on a programmatic setValue() too, and each
        slider individually re-triggering _apply_config here would walk
        self._working_config through a series of partially-updated
        states instead of jumping straight to the loaded one.
        """
        cfg = self._working_config
        hostile = next(c for c in cfg.classes if c.name == "hostile")
        friendly = next(c for c in cfg.classes if c.name == "friendly")
        values = {
            "morph_kernel": cfg.morph_kernel,
            "min_area": cfg.min_area_px,
            "circularity_min": round(cfg.circularity_min * 100),
            "hostile_sat": hostile.sat_min,
            "hostile_val": hostile.val_min,
            "friendly_sat": friendly.sat_min,
            "friendly_val": friendly.val_min,
        }
        for range_index, (lo, hi) in enumerate(hostile.hue_ranges):
            values[f"hostile_hue{range_index}_lo"] = lo
            values[f"hostile_hue{range_index}_hi"] = hi
        for range_index, (lo, hi) in enumerate(friendly.hue_ranges):
            values[f"friendly_hue{range_index}_lo"] = lo
            values[f"friendly_hue{range_index}_hi"] = hi

        for key, value in values.items():
            slider, label = self._sliders[key]
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            label.setText(str(value))

        self._require_circularity_checkbox.blockSignals(True)
        self._require_circularity_checkbox.setChecked(cfg.require_circularity)
        self._require_circularity_checkbox.blockSignals(False)

    # --- preview ---

    def _on_snapshot(self, snapshot: UiSnapshot) -> None:
        if not self.isVisible():
            return
        now = snapshot.t
        if (
            self._last_preview_t is not None
            and (now - self._last_preview_t) < 1.0 / _PREVIEW_UPDATE_HZ
        ):
            return
        self._last_preview_t = now
        self._latest_frame = snapshot.frame
        _detections, debug = self._detector.detect(snapshot.frame, debug=True)
        self._latest_debug = debug
        self._update_counts(debug)
        self._render_preview()

    def _update_counts(self, debug: DebugMasks | None) -> None:
        if debug is None:
            self._counts_label.setText("")
            return
        accepted = sum(1 for c in debug.contours if c.accepted)
        rejected = len(debug.contours) - accepted
        self._counts_label.setText(
            UI_LABEL_TR["TUNING_COUNTS"].format(accepted=accepted, rejected=rejected)
        )

    def _render_preview(self) -> None:
        if self._latest_frame is None:
            return
        mode = self._mode_combo.currentData()
        if mode == "source":
            pixmap = frame_to_pixmap(self._latest_frame.image)
        elif mode == "contours":
            pixmap = self._contours_pixmap()
        else:
            pixmap = self._mask_pixmap(mode)
        self._preview.set_pixmap(pixmap)
        self._preview.set_roi(self._working_config.roi, draggable=mode in _ROI_DRAG_MODES)

    def _mask_pixmap(self, mode: str) -> QPixmap:
        if self._latest_debug is None:
            return frame_to_pixmap(self._latest_frame.image)
        class_name = self._class_combo.currentData()
        masks = self._latest_debug.hsv_masks if mode == "hsv" else self._latest_debug.morphed_masks
        mask = masks.get(class_name)
        if mask is None:
            return frame_to_pixmap(self._latest_frame.image)
        rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return frame_to_pixmap(rgb)

    def _contours_pixmap(self) -> QPixmap:
        image = self._latest_frame.image.copy()
        if self._latest_debug is not None:
            for contour in self._latest_debug.contours:
                x, y, w, h = contour.bbox_px
                color = (0, 200, 0) if contour.accepted else (0, 0, 200)  # BGR
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return frame_to_pixmap(image)
