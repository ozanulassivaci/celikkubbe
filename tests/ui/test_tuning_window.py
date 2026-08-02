"""TuningWindow tests. normalize_drag_to_roi is a pure function, tested
directly. The dialog is driven against a real PipelineWorker (FakeClock,
SimTurretLink, SyntheticSource) exactly like test_pipeline_worker.py --
"live apply to the running detector" is the entire point of this window,
so a mock detector would not actually exercise it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.strings import UI_LABEL_TR
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.ui.tuning_window import TuningWindow, _PreviewWidget, normalize_drag_to_roi
from celikkubbe.vision.l2_color import ColorDetectorConfig, Contour, DebugMasks
from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig

_TICK_DT = 1.0 / 30.0


def _make_worker(clock: FakeClock, num_targets: int = 1) -> PipelineWorker:
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=num_targets, speed=0.0))
    source.start()
    return PipelineWorker(source, SimTurretLink(clock), clock)


def _tick(worker: PipelineWorker, clock: FakeClock) -> None:
    clock.advance(_TICK_DT)
    worker._link_worker.tick()
    worker.tick()


# --- normalize_drag_to_roi ---


def test_normalize_drag_to_roi_orders_reversed_corners() -> None:
    roi = normalize_drag_to_roi((0.6, 0.7), (0.2, 0.1))
    assert roi == (0.2, 0.1, 0.6, 0.7)


def test_normalize_drag_to_roi_clamps_to_the_unit_square() -> None:
    roi = normalize_drag_to_roi((-0.2, -0.1), (1.3, 1.4))
    assert roi == (0.0, 0.0, 1.0, 1.0)


def test_normalize_drag_to_roi_rejects_a_too_small_drag() -> None:
    assert normalize_drag_to_roi((0.5, 0.5), (0.505, 0.505)) is None


# --- widget: construction / sync ---


def test_sliders_initialize_from_the_detectors_current_config(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)

    cfg = worker.detector.config
    assert window._sliders["morph_kernel"][0].value() == cfg.morph_kernel
    # min_area_px defaults to None (auto -- computed from optics), shown
    # as 0 on the slider; see tuning_window._set_min_area's own docstring.
    assert cfg.min_area_px is None
    assert window._sliders["min_area"][0].value() == 0
    hostile = next(c for c in cfg.classes if c.name == "hostile")
    assert window._sliders["hostile_hue0_lo"][0].value() == hostile.hue_ranges[0][0]
    assert window._sliders["hostile_sat"][0].value() == hostile.sat_min


# --- widget: live apply ---


def test_moving_morph_kernel_slider_live_applies_to_the_detector(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)

    window._sliders["morph_kernel"][0].setValue(9)

    assert worker.detector.config.morph_kernel == 9


def test_moving_circularity_slider_applies_as_a_fraction(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)

    window._sliders["circularity_min"][0].setValue(55)

    assert worker.detector.config.circularity_min == 0.55


def test_moving_min_area_slider_sets_an_explicit_override(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)

    window._sliders["min_area"][0].setValue(500)
    assert worker.detector.config.min_area_px == 500

    window._sliders["min_area"][0].setValue(0)
    assert worker.detector.config.min_area_px is None


def test_require_circularity_checkbox_applies_to_the_detector(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)

    window._require_circularity_checkbox.setChecked(True)

    assert worker.detector.config.require_circularity is True


def test_moving_hostile_hue_slider_does_not_affect_friendly(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    friendly_before = next(c for c in worker.detector.config.classes if c.name == "friendly")

    window._sliders["hostile_hue0_lo"][0].setValue(3)

    hostile_after = next(c for c in worker.detector.config.classes if c.name == "hostile")
    friendly_after = next(c for c in worker.detector.config.classes if c.name == "friendly")
    assert hostile_after.hue_ranges[0][0] == 3
    assert friendly_after == friendly_before


def test_moving_friendly_sat_slider_updates_only_friendly(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)

    window._sliders["friendly_sat"][0].setValue(150)

    friendly = next(c for c in worker.detector.config.classes if c.name == "friendly")
    hostile = next(c for c in worker.detector.config.classes if c.name == "hostile")
    assert friendly.sat_min == 150
    assert hostile.sat_min != 150


def test_reset_defaults_restores_widgets_and_detector(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    window._sliders["morph_kernel"][0].setValue(1)
    window._sliders["hostile_sat"][0].setValue(200)

    window._on_reset_defaults()

    defaults = ColorDetectorConfig()
    assert worker.detector.config.morph_kernel == defaults.morph_kernel
    assert window._sliders["morph_kernel"][0].value() == defaults.morph_kernel
    hostile_default = next(c for c in defaults.classes if c.name == "hostile")
    assert window._sliders["hostile_sat"][0].value() == hostile_default.sat_min


# --- widget: presets ---


def test_save_then_load_preset_round_trips_through_the_detector(qtbot, tmp_path):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker, presets_dir=tmp_path)
    qtbot.addWidget(window)
    window._sliders["morph_kernel"][0].setValue(11)
    window._preset_name_edit.setText("hall")

    window._on_save_preset()
    assert (tmp_path / "hall.json").exists()

    window._on_reset_defaults()
    assert worker.detector.config.morph_kernel != 11

    window._preset_combo.setCurrentText("hall")
    window._on_load_preset()

    assert worker.detector.config.morph_kernel == 11
    assert window._sliders["morph_kernel"][0].value() == 11


def test_save_preset_with_empty_name_does_nothing(qtbot, tmp_path):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker, presets_dir=tmp_path)
    qtbot.addWidget(window)

    window._on_save_preset()

    assert list(tmp_path.glob("*.json")) == []


# --- widget: preview ---


def test_preview_does_not_update_while_hidden(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    assert not window.isVisible()

    _tick(worker, clock)

    assert window._latest_frame is None


def test_preview_updates_with_counts_once_shown(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock, num_targets=1)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    _tick(worker, clock)
    # Flushes set_pixmap's own update() request while the widget is still
    # alive -- left queued across teardown, Qt's event loop can deliver
    # the deferred paint to an already-deleted _PreviewWidget on the next
    # test's own event processing and segfault (seen empirically, not
    # hypothetically: this exact sequence crashed before this wait()).
    qtbot.wait(10)

    assert window._latest_frame is not None
    assert window._preview._pixmap is not None
    assert "KABUL" in window._counts_label.text()


def test_counts_label_reports_class_cap_discards_separately(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock, num_targets=1)
    window = TuningWindow(worker)
    qtbot.addWidget(window)

    debug = DebugMasks(
        hsv_masks={},
        morphed_masks={},
        contours=(
            Contour("hostile", (0, 0, 5, 5), 100.0, 0.9, 0.9, True),
            Contour("hostile", (10, 10, 5, 5), 50.0, 0.9, 0.9, False, "class_cap"),
            Contour("hostile", (20, 20, 5, 5), 10.0, 0.9, 0.9, False, "area"),
        ),
    )
    window._update_counts(debug)

    assert window._counts_label.text() == UI_LABEL_TR["TUNING_COUNTS"].format(
        accepted=1, rejected=2, capped=1
    )


def test_switching_preview_mode_does_not_crash_and_updates_pixmap(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock, num_targets=1)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    _tick(worker, clock)

    for index in range(window._mode_combo.count()):
        window._mode_combo.setCurrentIndex(index)
        assert window._preview._pixmap is not None
    qtbot.wait(10)  # flush the last mode switch's update() -- see above


def test_roi_not_draggable_in_mask_preview_modes(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock, num_targets=1)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    _tick(worker, clock)

    hsv_index = next(
        i for i in range(window._mode_combo.count()) if window._mode_combo.itemData(i) == "hsv"
    )
    window._mode_combo.setCurrentIndex(hsv_index)
    assert window._preview._roi_draggable is False

    source_index = next(
        i for i in range(window._mode_combo.count()) if window._mode_combo.itemData(i) == "source"
    )
    window._mode_combo.setCurrentIndex(source_index)
    qtbot.wait(10)  # flush the last mode switch's update() -- see above
    assert window._preview._roi_draggable is True


# --- widget: ROI drag on the preview itself ---
# Driven through qtbot's own QTest-backed mouse simulation, not
# hand-built QMouseEvents fed directly to the event handlers: a bare,
# never-shown widget receiving a hand-rolled event plus its own
# update()'s deferred repaint request is exactly the combination that
# segfaults under offscreen Qt once the widget is torn down before that
# repaint runs. Showing the widget for real and going through QTest
# gives Qt's own event loop a real widget to paint and dispose of.


def _drag(qtbot, widget: _PreviewWidget, start: tuple[int, int], end: tuple[int, int]) -> None:
    from PyQt6.QtCore import QPoint

    qtbot.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(*start))
    qtbot.mouseMove(widget, pos=QPoint(*end))
    qtbot.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(*end))


def test_dragging_on_the_preview_widget_emits_roi_dragged(qtbot):
    widget = _PreviewWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_pixmap(QPixmap(400, 300))

    received = []
    widget.roi_dragged.connect(received.append)
    _drag(qtbot, widget, (40, 30), (360, 270))
    qtbot.wait(10)  # flush the drag's own update() calls -- see above

    assert len(received) == 1
    x1, y1, x2, y2 = received[0]
    assert 0.0 < x1 < x2 <= 1.0
    assert 0.0 < y1 < y2 <= 1.0


def test_tiny_drag_on_the_preview_widget_emits_nothing(qtbot):
    widget = _PreviewWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_pixmap(QPixmap(400, 300))

    received = []
    widget.roi_dragged.connect(received.append)
    _drag(qtbot, widget, (100, 100), (101, 101))
    qtbot.wait(10)  # flush the drag's own update() calls -- see above

    assert received == []


def test_roi_dragged_signal_updates_working_config_and_detector(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock, num_targets=1)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    _tick(worker, clock)

    window._on_roi_dragged((0.1, 0.2, 0.8, 0.9))
    qtbot.wait(10)  # flush _apply_config's own render_preview()/update() -- see above

    assert worker.detector.config.roi == (0.1, 0.2, 0.8, 0.9)
    assert window._working_config.roi == (0.1, 0.2, 0.8, 0.9)


def test_clear_roi_resets_the_detectors_roi_to_none(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock, num_targets=1)
    window = TuningWindow(worker)
    qtbot.addWidget(window)
    window._on_roi_dragged((0.1, 0.2, 0.8, 0.9))

    window._on_clear_roi()

    assert worker.detector.config.roi is None
