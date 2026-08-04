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
    QButtonGroup,
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
    ColorSample,
    DebugMasks,
    check_negative_sample,
    derive_thresholds,
    list_hsv_presets,
    load_hsv_preset,
    sample_rectangle,
    sample_region,
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

# Eyedropper calibration -- see l2_color.sample_region/sample_rectangle/
# derive_thresholds/check_negative_sample, the pure primitives this UI
# wires up. "off" leaves the preview's own click/drag interaction doing
# what it always did (ROI dragging); "positive" and "negative" both
# reuse that same click=point/drag=rectangle interaction to sample
# instead, never simultaneously with ROI dragging.
_EYEDROPPER_MODES = ("off", "positive", "negative")
_EYEDROPPER_MODE_LABEL_KEY = {
    "off": "EYEDROPPER_OFF",
    "positive": "EYEDROPPER_POSITIVE",
    "negative": "EYEDROPPER_NEGATIVE",
}
_TIGHTEN_FIELD_LABEL_KEY = {"sat_min": "SAT_MIN_LABEL", "val_min": "VAL_MIN_LABEL"}


def coerce_hue_range_count(
    hue_ranges: tuple[tuple[int, int], ...], count: int
) -> tuple[tuple[int, int], ...]:
    """derive_thresholds returns however many hue ranges the sample
    actually needs (one, or two if it wrapped) -- but this window's
    sliders are laid out with a fixed number of rows per class (2 for
    hostile, 1 for friendly, see _build_class_group), decided once at
    construction time. Padding by repeating the last range (rather than
    a degenerate (0, 0), which would still match hue=0 exactly) keeps
    the applied config's actual coverage unchanged when fewer ranges
    were produced than slots exist.
    """
    ranges = list(hue_ranges[:count])
    while len(ranges) < count:
        ranges.append(ranges[-1] if ranges else (0, 179))
    return tuple(ranges)


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
    # Emitted instead of roi_dragged when a press/release pair was too
    # small a drag to be an intentional ROI (see normalize_drag_to_roi) --
    # previously silently dropped, since nothing needed a plain click.
    # The eyedropper's point-sample mode needs exactly this: a click, not
    # a drag, is what sample_region's own point-vs-rectangle distinction
    # calls for.
    point_clicked = pyqtSignal(tuple)

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
        else:
            self.point_clicked.emit(p1)


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
        self._eyedropper_mode = "off"
        # (class_name, hue_ranges, sat_min, val_min) awaiting EYEDROPPER_APPLY,
        # or None -- see _show_positive_sample/_on_eyedropper_apply. Only a
        # positive sample ever has anything pending; a negative sample is
        # purely a report, nothing to apply.
        self._pending_sample: tuple[str, tuple[tuple[int, int], ...], int, int] | None = None

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

        # 0 means "auto": leaves ColorDetectorConfig.morph_kernel at None,
        # so the detector computes the closing kernel itself from optics
        # and an assumed highlight-gap width every frame (see
        # l2_color.compute_morph_kernel_px). Any value above 0 sets an
        # explicit fixed override instead.
        row, slider, label = _slider_row(UI_LABEL_TR["MORPH_KERNEL_LABEL"], 0, 25, 0)
        slider.setToolTip(UI_LABEL_TR["MORPH_KERNEL_AUTO_HINT"])
        slider.valueChanged.connect(self._set_morph_kernel)
        self._sliders["morph_kernel"] = (slider, label)
        layout.addWidget(row)

        # 0 means "auto": leaves ColorDetectorConfig.min_area_px at None,
        # so the detector computes the structural floor itself from
        # intrinsics every frame (see l2_color.compute_min_area_px).
        # Any value above 0 sets an explicit fixed override instead.
        row, slider, label = _slider_row(UI_LABEL_TR["MIN_AREA_LABEL"], 0, 2000, 0)
        slider.setToolTip(UI_LABEL_TR["MIN_AREA_AUTO_HINT"])
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

        self._require_solidity_checkbox = QCheckBox(UI_LABEL_TR["REQUIRE_SOLIDITY_LABEL"])
        self._require_solidity_checkbox.toggled.connect(self._set_require_solidity)
        layout.addWidget(self._require_solidity_checkbox)

        self._require_specular_bridging_checkbox = QCheckBox(
            UI_LABEL_TR["REQUIRE_SPECULAR_BRIDGING_LABEL"]
        )
        self._require_specular_bridging_checkbox.toggled.connect(
            self._set_require_specular_bridging
        )
        layout.addWidget(self._require_specular_bridging_checkbox)

        # Provisional default (230) -- the right value depends on the
        # venue's own lighting rig, not on anything derivable from
        # optics, so this is exposed for on-the-day tuning rather than
        # fixed in code. See l2_color.ColorDetectorConfig.highlight_v_min.
        row, slider, label = _slider_row(UI_LABEL_TR["HIGHLIGHT_V_MIN_LABEL"], 0, 255, 230)
        slider.valueChanged.connect(self._set_highlight_v_min)
        self._sliders["highlight_v_min"] = (slider, label)
        layout.addWidget(row)

        row, slider, label = _slider_row(UI_LABEL_TR["HIGHLIGHT_MAX_FRACTION_LABEL"], 0, 100, 50)
        slider.valueChanged.connect(self._set_highlight_max_fraction)
        self._sliders["highlight_max_fraction"] = (slider, label)
        layout.addWidget(row)
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

        layout.addLayout(self._build_eyedropper_row())

        self._preview = _PreviewWidget()
        self._preview.roi_dragged.connect(self._on_roi_dragged)
        self._preview.point_clicked.connect(self._on_point_clicked)
        layout.addWidget(self._preview, stretch=1)

        self._counts_label = QLabel("")
        layout.addWidget(self._counts_label)
        layout.addWidget(self._build_eyedropper_result_panel())
        return container

    def _build_eyedropper_row(self) -> QHBoxLayout:
        # Positive samples target whichever class is selected in
        # self._class_combo above -- deliberately the same combo the
        # hsv/morphed mask preview already uses, so "hostile selected"
        # means the same thing in both places rather than a second,
        # independent class picker.
        row = QHBoxLayout()
        row.addWidget(QLabel(UI_LABEL_TR["EYEDROPPER_HINT"]))
        row.addStretch(1)
        self._eyedropper_buttons: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode in _EYEDROPPER_MODES:
            btn = QPushButton(UI_LABEL_TR[_EYEDROPPER_MODE_LABEL_KEY[mode]])
            btn.setCheckable(True)
            is_off = mode == "off"
            btn.setChecked(is_off)
            if is_off:
                btn.setStyleSheet(
                    f"background-color: {theme.WARN}; border-color: {theme.WARN}; "
                    f"color: {theme.BG_BASE};"
                )
            btn.clicked.connect(lambda _checked=False, m=mode: self._set_eyedropper_mode(m))
            group.addButton(btn)
            self._eyedropper_buttons[mode] = btn
            row.addWidget(btn)
        return row

    def _build_eyedropper_result_panel(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self._eyedropper_result_label = QLabel("")
        self._eyedropper_result_label.setWordWrap(True)
        layout.addWidget(self._eyedropper_result_label, stretch=1)
        self._eyedropper_apply_btn = QPushButton(UI_LABEL_TR["EYEDROPPER_APPLY"])
        self._eyedropper_apply_btn.setEnabled(False)
        self._eyedropper_apply_btn.clicked.connect(self._on_eyedropper_apply)
        layout.addWidget(self._eyedropper_apply_btn)
        self._eyedropper_cancel_btn = QPushButton(UI_LABEL_TR["EYEDROPPER_CANCEL"])
        self._eyedropper_cancel_btn.setEnabled(False)
        self._eyedropper_cancel_btn.clicked.connect(self._on_eyedropper_cancel)
        layout.addWidget(self._eyedropper_cancel_btn)
        return container

    # --- live-apply: global fields ---

    def _apply_config(self, new_config: ColorDetectorConfig) -> None:
        self._working_config = new_config
        self._detector.config = new_config
        self._render_preview()

    def _set_morph_kernel(self, value: int) -> None:
        self._apply_config(
            dataclasses.replace(self._working_config, morph_kernel=value if value > 0 else None)
        )

    def _set_min_area(self, value: int) -> None:
        self._apply_config(
            dataclasses.replace(self._working_config, min_area_px=value if value > 0 else None)
        )

    def _set_circularity_min(self, value: int) -> None:
        self._apply_config(dataclasses.replace(self._working_config, circularity_min=value / 100.0))

    def _set_require_circularity(self, checked: bool) -> None:
        self._apply_config(dataclasses.replace(self._working_config, require_circularity=checked))

    def _set_require_solidity(self, checked: bool) -> None:
        self._apply_config(dataclasses.replace(self._working_config, require_solidity=checked))

    def _set_require_specular_bridging(self, checked: bool) -> None:
        self._apply_config(
            dataclasses.replace(self._working_config, require_specular_bridging=checked)
        )

    def _set_highlight_v_min(self, value: int) -> None:
        self._apply_config(dataclasses.replace(self._working_config, highlight_v_min=value))

    def _set_highlight_max_fraction(self, value: int) -> None:
        self._apply_config(
            dataclasses.replace(self._working_config, highlight_max_fraction=value / 100.0)
        )

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
    # While an eyedropper mode is active, the preview's own click/drag
    # interaction means "take a sample", not "set the ROI" -- the two
    # must never both react to the same drag.

    def _on_roi_dragged(self, roi: BoundingBox) -> None:
        if self._eyedropper_mode == "off":
            self._apply_config(dataclasses.replace(self._working_config, roi=roi))
            return
        self._sample_and_show(lambda frame: sample_rectangle(frame, roi))

    def _on_clear_roi(self) -> None:
        self._apply_config(dataclasses.replace(self._working_config, roi=None))

    # --- eyedropper calibration ---

    def _set_eyedropper_mode(self, mode: str) -> None:
        self._eyedropper_mode = mode
        self._clear_pending_sample()
        # QPushButton:checked has no distinct look under this theme's own
        # QSS (see theme.build_qss) -- right_panel.py's stage selector
        # hits the same gap and fixes it the same way, an explicit
        # per-button stylesheet rather than a generic checked rule, since
        # nothing else in this window needs one.
        for button_mode, btn in self._eyedropper_buttons.items():
            btn.setStyleSheet(
                f"background-color: {theme.WARN}; border-color: {theme.WARN}; "
                f"color: {theme.BG_BASE};"
                if button_mode == mode
                else ""
            )

    def _on_point_clicked(self, point: tuple[float, float]) -> None:
        if self._eyedropper_mode == "off":
            return
        self._sample_and_show(lambda frame: sample_region(frame, point))

    def _sample_and_show(self, sampler) -> None:
        if self._latest_frame is None:
            return
        sample = sampler(self._latest_frame)
        if sample.pixel_count == 0:
            return
        if self._eyedropper_mode == "positive":
            self._show_positive_sample(sample)
        elif self._eyedropper_mode == "negative":
            self._show_negative_sample(sample)

    def _show_positive_sample(self, sample: ColorSample) -> None:
        class_name = self._class_combo.currentData()
        range_count = 2 if class_name == "hostile" else 1
        hue_ranges, sat_min, val_min = derive_thresholds(sample)
        hue_ranges = coerce_hue_range_count(hue_ranges, range_count)
        self._pending_sample = (class_name, hue_ranges, sat_min, val_min)

        wrap_tag = UI_LABEL_TR["EYEDROPPER_WRAP_TAG"] if sample.hue_wraps else ""
        self._eyedropper_result_label.setText(
            UI_LABEL_TR["EYEDROPPER_POSITIVE_SUMMARY"].format(
                hue=sample.hue_median % 180.0,
                sat=sample.sat_median,
                val=sample.val_median,
                pixels=sample.pixel_count,
                wrap=wrap_tag,
            )
        )
        self._eyedropper_apply_btn.setEnabled(True)
        self._eyedropper_cancel_btn.setEnabled(True)

    def _show_negative_sample(self, sample: ColorSample) -> None:
        self._pending_sample = None
        reports = check_negative_sample(sample, self._working_config)
        lines = []
        for report in reports:
            class_title = (
                UI_LABEL_TR["HOSTILE_CLASS_TITLE"]
                if report.class_name == "hostile"
                else UI_LABEL_TR["FRIENDLY_CLASS_TITLE"]
            )
            if report.accepted:
                field_label = UI_LABEL_TR[_TIGHTEN_FIELD_LABEL_KEY[report.tighten_field]]
                lines.append(
                    UI_LABEL_TR["EYEDROPPER_NEGATIVE_HIT"].format(
                        cls=class_title, field=field_label, value=report.suggested_value
                    )
                )
            else:
                lines.append(UI_LABEL_TR["EYEDROPPER_NEGATIVE_CLEAR"].format(cls=class_title))
        self._eyedropper_result_label.setText("\n".join(lines))
        self._eyedropper_apply_btn.setEnabled(False)
        self._eyedropper_cancel_btn.setEnabled(True)

    def _on_eyedropper_apply(self) -> None:
        if self._pending_sample is None:
            return
        class_name, hue_ranges, sat_min, val_min = self._pending_sample
        classes = tuple(
            dataclasses.replace(c, hue_ranges=hue_ranges, sat_min=sat_min, val_min=val_min)
            if c.name == class_name
            else c
            for c in self._working_config.classes
        )
        self._apply_config(dataclasses.replace(self._working_config, classes=classes))
        self._sync_widgets_from_config()
        self._clear_pending_sample()

    def _on_eyedropper_cancel(self) -> None:
        self._clear_pending_sample()

    def _clear_pending_sample(self) -> None:
        self._pending_sample = None
        self._eyedropper_result_label.setText("")
        self._eyedropper_apply_btn.setEnabled(False)
        self._eyedropper_cancel_btn.setEnabled(False)

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
        self._clear_pending_sample()

    def _on_reset_defaults(self) -> None:
        self._apply_config(ColorDetectorConfig())
        self._sync_widgets_from_config()
        self._clear_pending_sample()

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
            "morph_kernel": cfg.morph_kernel if cfg.morph_kernel is not None else 0,
            "min_area": cfg.min_area_px if cfg.min_area_px is not None else 0,
            "circularity_min": round(cfg.circularity_min * 100),
            "highlight_v_min": cfg.highlight_v_min,
            "highlight_max_fraction": round(cfg.highlight_max_fraction * 100),
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

        self._require_solidity_checkbox.blockSignals(True)
        self._require_solidity_checkbox.setChecked(cfg.require_solidity)
        self._require_solidity_checkbox.blockSignals(False)

        self._require_specular_bridging_checkbox.blockSignals(True)
        self._require_specular_bridging_checkbox.setChecked(cfg.require_specular_bridging)
        self._require_specular_bridging_checkbox.blockSignals(False)

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
        # Reported separately from the rest of "RED": a persistently high
        # class-cap count means real targets are being discarded by the
        # per-class cap, not noise -- a different problem from a tight
        # HSV/area/solidity threshold, and one the operator should notice.
        capped = sum(1 for c in debug.contours if c.reject_reason == "class_cap")
        self._counts_label.setText(
            UI_LABEL_TR["TUNING_COUNTS"].format(accepted=accepted, rejected=rejected, capped=capped)
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
                if not contour.accepted and contour.reject_reason is not None:
                    # The whole point of 2e: which specific filter
                    # rejected this contour, so tuning is diagnostic
                    # rather than guesswork -- not just "red box, unknown
                    # why". The English slug itself, not
                    # REJECT_REASON_LABEL_TR's Turkish text: cv2.putText's
                    # Hershey fonts cannot render Turkish diacritics
                    # (İ/Ş/Ğ/Ü/Ö/Ç render as missing or wrong glyphs), and
                    # a corrupted label would be worse than an English one
                    # for a technical tuning tool.
                    cv2.putText(
                        image,
                        contour.reject_reason,
                        (x, max(0, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        return frame_to_pixmap(image)
