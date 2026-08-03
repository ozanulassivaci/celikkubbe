"""MainWindow shell tests. Snapshots are driven the same way as
test_pipeline_worker.py: PipelineWorker.tick() called directly (never
worker.start()), which delivers snapshot_ready synchronously since
sender and receiver share this test's thread -- no event-loop spinning
needed to observe MainWindow react to a tick.
"""

from __future__ import annotations

import dataclasses

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QApplication, QLabel

from celikkubbe.core import config, strings
from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import Axis, Layer, Mode, ReasonCode, Stage
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.ui import theme
from celikkubbe.ui.main_window import MainWindow
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig

_TICK_DT = 1.0 / 30.0


def _make_source(clock: FakeClock) -> SyntheticSource:
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=1, speed=0.0, range_m=3.0))
    source.start()
    return source


def _make_window(clock: FakeClock) -> tuple[MainWindow, PipelineWorker, SimTurretLink]:
    link = SimTurretLink(clock)
    worker = PipelineWorker(_make_source(clock), link, clock)
    window = MainWindow(worker, source_label="SYNTHETIC")
    return window, worker, link


def _make_dev_window(clock: FakeClock) -> tuple[MainWindow, PipelineWorker, SimTurretLink]:
    link = SimTurretLink(clock)
    worker = PipelineWorker(_make_source(clock), link, clock, dev_mode=True)
    window = MainWindow(worker, source_label="SYNTHETIC")
    return window, worker, link


def _tick(worker: PipelineWorker, clock: FakeClock) -> None:
    clock.advance(_TICK_DT)
    worker._link_worker.tick()  # simulate LinkWorker's own background thread
    worker.tick()


def _settle_self_test(worker: PipelineWorker, clock: FakeClock, max_ticks: int = 60) -> None:
    """Self-test now includes a real pan/tilt verification move (Part 0e),
    which takes a handful of ticks to settle -- a fixed 3-tick budget from
    before that existed is no longer enough to reach M2_STANDBY.
    """
    for _ in range(max_ticks):
        if worker._state.mode is not Mode.M1_INIT:
            return
        _tick(worker, clock)


def test_status_strip_reflects_mode_engagement_and_layer(qtbot):
    clock = FakeClock()
    window, worker, link = _make_window(clock)
    qtbot.addWidget(window)

    _settle_self_test(worker, clock)
    _tick(worker, clock)

    snapshot = window._latest_snapshot
    assert snapshot is not None
    assert snapshot.state.mode is Mode.M2_STANDBY
    strip = window._status_strip
    assert strip._mode_badge._text == snapshot.state.mode.value[:2]
    assert strip._engagement_badge._text == snapshot.state.engagement.value[:2]
    assert strip._layer_badge._text == snapshot.state.active_layer.value

    link.inject_estop()
    for _ in range(3):
        _tick(worker, clock)

    updated = window._latest_snapshot
    assert updated.state.mode is Mode.M4_SAFE
    assert strip._mode_badge._text == "M4"


def test_selftest_overlay_shows_in_m1_and_hides_on_success(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    # LinkWorker starts with link_lost=True until its own tick() has run
    # at least once, so ticking the pipeline without it first guarantees
    # the stm32_link self-test item fails and mode stays in M1_INIT --
    # against a fully cooperative SimTurretLink, self-test can otherwise
    # pass within a single tick, too fast to reliably observe M1_INIT.
    clock.advance(_TICK_DT)
    worker.tick()
    assert window._latest_snapshot.state.mode is Mode.M1_INIT
    assert not window._selftest_overlay.isHidden()

    _settle_self_test(worker, clock)

    assert window._latest_snapshot.state.mode is Mode.M2_STANDBY
    assert window._selftest_overlay.isHidden()


def test_safe_overlay_shows_in_m4_and_requires_acknowledgement(qtbot):
    clock = FakeClock()
    window, worker, link = _make_window(clock)
    qtbot.addWidget(window)

    _settle_self_test(worker, clock)
    assert window._latest_snapshot.state.mode is Mode.M2_STANDBY

    link.inject_estop()
    for _ in range(3):
        _tick(worker, clock)
    assert window._latest_snapshot.state.mode is Mode.M4_SAFE
    assert not window._safe_overlay.isHidden()
    assert window._safe_overlay._fault_label.text() == strings.describe(ReasonCode.ESTOP_ACTIVE)


def test_dev_mode_shows_permanent_banner_and_status_badge(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_dev_window(clock)
    qtbot.addWidget(window)

    assert worker._state.mode is Mode.M2_STANDBY  # self-test skipped at boot
    banner_texts = [
        label.text()
        for label in window.findChildren(QLabel)
        if label.text() == strings.UI_LABEL_TR["DEV_MODE_BANNER"]
    ]
    assert banner_texts, "dev-mode banner not found in the window"

    dev_badges = [
        b for b in window._status_strip.findChildren(theme.StatusBadge) if b._text == "DEV"
    ]
    assert len(dev_badges) == 1


def test_dev_mode_absent_by_default(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    banner_texts = [
        label.text()
        for label in window.findChildren(QLabel)
        if label.text() == strings.UI_LABEL_TR["DEV_MODE_BANNER"]
    ]
    assert not banner_texts
    dev_badges = [
        b for b in window._status_strip.findChildren(theme.StatusBadge) if b._text == "DEV"
    ]
    assert dev_badges == []


def test_dev_mode_ignores_estop_and_reaches_m3_without_homing(qtbot):
    clock = FakeClock()
    window, worker, link = _make_dev_window(clock)
    qtbot.addWidget(window)

    link.inject_estop()
    _tick(worker, clock)
    assert worker._state.mode is Mode.M2_STANDBY  # e-stop ignored in dev_mode

    worker._state = dataclasses.replace(worker._state, mode=Mode.M3_OPERATIONAL)
    _tick(worker, clock)
    assert worker._state.mode is Mode.M3_OPERATIONAL


def test_left_panel_track_selection_sets_manual_target_id(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    _settle_self_test(worker, clock)
    _tick(worker, clock)

    assert window._latest_snapshot.tracks, "synthetic source should have produced a track by now"
    track_id = window._latest_snapshot.tracks[0].track_id

    window._left_panel.track_selected.emit(track_id)

    assert window._input_builder.build().manual_target_id == track_id


def test_right_panel_estop_signal_reaches_the_link(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)
    _settle_self_test(worker, clock)

    window._right_panel.estop_requested.emit()
    worker._link_worker.tick()

    # SoftEstop stops motion and disarms -- it must not trip the hardware
    # estop latch (that is inject_estop()'s job); see sim_link's own
    # _handle_soft_estop docstring.
    assert worker._link_worker.latest_telemetry.armed is False


def test_right_panel_armed_changed_signal_reaches_the_link(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)
    _settle_self_test(worker, clock)

    window._right_panel.armed_changed.emit(True)
    worker._link_worker.tick()

    assert worker._link_worker.latest_telemetry.armed


def test_right_panel_zero_and_jog_signals_reach_the_link(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    window._right_panel.jog_pressed.emit(Axis.PAN, 1, 20.0)
    worker._link_worker.tick()
    clock.advance(0.2)
    worker._link_worker.tick()
    assert worker._link_worker.latest_telemetry.pan_deg > 0.0

    window._right_panel.jog_released.emit()
    worker._link_worker.tick()

    window._right_panel.zero_requested.emit(Axis.PAN)
    worker._link_worker.tick()
    assert worker._link_worker.latest_telemetry.pan_deg == 0.0


def test_right_panel_stage_selected_updates_worker_and_app_accent(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    window._right_panel.stage_selected.emit(Stage.STAGE_1)
    _tick(worker, clock)

    assert worker._state.stage is Stage.STAGE_1
    app = QApplication.instance()
    assert theme.ACCENT_A1 in app.styleSheet()


def test_right_panel_layer_selected_reaches_the_worker(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    window._right_panel.layer_selected.emit(Layer.L2)
    _tick(worker, clock)

    assert worker._state.active_layer is Layer.L2
    assert worker._state.layer_manual_override is True


def test_right_panel_fire_press_and_release_updates_operator_input(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    window._right_panel.fire_pressed.emit()
    assert window._input_builder.build().fire_requested
    assert window._input_builder.build().arm_held

    window._right_panel.fire_released.emit()
    assert not window._input_builder.build().fire_requested
    assert not window._input_builder.build().arm_held


def test_window_deactivation_forces_stop_on_right_panel(qtbot, monkeypatch):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)
    window._right_panel._on_jog_pressed(Axis.PAN, 1)
    released = []
    window._right_panel.jog_released.connect(lambda: released.append(True))

    # A real OS/WM activation change cannot be triggered in a headless
    # test -- isActiveWindow() is monkeypatched instead so changeEvent's
    # own branch (not just force_stop_all in isolation) is what runs.
    monkeypatch.setattr(window, "isActiveWindow", lambda: False)
    window.changeEvent(QEvent(QEvent.Type.ActivationChange))

    assert released == [True]


def test_link_loss_visible_in_status_strip_within_stale_threshold(qtbot):
    clock = FakeClock()
    window, worker, link = _make_window(clock)
    qtbot.addWidget(window)

    _tick(worker, clock)
    assert "LINK OK" in window._status_strip._link_label.text()

    link.inject_link_dropout(duration_s=10.0)
    stale_s = config.TELEMETRY_STALE_MS / 1000.0
    clock.advance(stale_s + _TICK_DT)
    worker._link_worker.tick()
    worker.tick()

    assert "LINK LOST" in window._status_strip._link_label.text()


# --- Part 4: keyboard ---


def test_right_arrow_jogs_pan_and_release_stops_it(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    qtbot.keyPress(window, Qt.Key.Key_Right)
    worker._link_worker.tick()
    clock.advance(0.2)
    worker._link_worker.tick()
    moved_pan = worker._link_worker.latest_telemetry.pan_deg
    assert moved_pan > 0.0

    qtbot.keyRelease(window, Qt.Key.Key_Right)
    worker._link_worker.tick()
    stopped_pan = worker._link_worker.latest_telemetry.pan_deg
    clock.advance(0.2)
    worker._link_worker.tick()
    assert worker._link_worker.latest_telemetry.pan_deg == stopped_pan


def test_space_bar_sets_fire_and_arm_via_the_keyboard_source(qtbot):
    clock = FakeClock()
    window, _worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    qtbot.keyPress(window, Qt.Key.Key_Space)
    result = window._input_builder.build()
    assert result.fire_requested
    assert result.arm_held

    qtbot.keyRelease(window, Qt.Key.Key_Space)
    result = window._input_builder.build()
    assert not result.fire_requested
    assert not result.arm_held


def test_escape_key_triggers_estop_directly(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    qtbot.keyPress(window, Qt.Key.Key_Escape)
    worker._link_worker.tick()

    assert worker._link_worker.latest_telemetry.armed is False


def test_a_key_toggles_armed(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)
    _tick(worker, clock)
    assert window._latest_snapshot.telemetry is not None
    assert not window._latest_snapshot.telemetry.armed

    qtbot.keyPress(window, Qt.Key.Key_A)
    _tick(worker, clock)
    assert window._latest_snapshot.telemetry.armed

    qtbot.keyPress(window, Qt.Key.Key_A)
    _tick(worker, clock)
    assert not window._latest_snapshot.telemetry.armed


def test_number_keys_select_stage_and_swap_accent(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    qtbot.keyPress(window, Qt.Key.Key_1)
    _tick(worker, clock)
    assert worker._state.stage is Stage.STAGE_1
    app = QApplication.instance()
    assert theme.ACCENT_A1 in app.styleSheet()

    qtbot.keyPress(window, Qt.Key.Key_3)
    _tick(worker, clock)
    assert worker._state.stage is Stage.STAGE_3
    assert theme.ACCENT_A3 in app.styleSheet()


def test_h_key_toggles_the_tuning_window(qtbot):
    clock = FakeClock()
    window, _worker, _link = _make_window(clock)
    qtbot.addWidget(window)
    assert window._tuning_window is None

    qtbot.keyPress(window, Qt.Key.Key_H)
    assert window._tuning_window is not None
    assert window._tuning_window.isVisible()

    qtbot.keyPress(window, Qt.Key.Key_H)
    assert not window._tuning_window.isVisible()


def test_f1_key_toggles_the_help_overlay(qtbot):
    clock = FakeClock()
    window, _worker, _link = _make_window(clock)
    qtbot.addWidget(window)
    assert window._help_overlay.isHidden()

    qtbot.keyPress(window, Qt.Key.Key_F1)
    assert not window._help_overlay.isHidden()

    qtbot.keyPress(window, Qt.Key.Key_F1)
    assert window._help_overlay.isHidden()


def test_window_deactivation_also_releases_the_keyboard_fire_source(qtbot, monkeypatch):
    clock = FakeClock()
    window, _worker, _link = _make_window(clock)
    qtbot.addWidget(window)
    qtbot.keyPress(window, Qt.Key.Key_Space)
    assert window._input_builder.build().fire_requested

    monkeypatch.setattr(window, "isActiveWindow", lambda: False)
    window.changeEvent(QEvent(QEvent.Type.ActivationChange))

    assert not window._input_builder.build().fire_requested
    assert not window._input_builder.build().arm_held


# --- Part 4: click-to-aim on empty canvas ---


def test_clicking_empty_canvas_in_stage_1_sends_a_direct_goto(qtbot):
    clock = FakeClock()
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=0))
    source.start()
    link = SimTurretLink(clock)
    worker = PipelineWorker(source, link, clock, stage=Stage.STAGE_1)
    window = MainWindow(worker, source_label="SYNTHETIC")
    qtbot.addWidget(window)
    _settle_self_test(worker, clock)
    _tick(worker, clock)
    assert not window._latest_snapshot.tracks

    window._on_canvas_clicked(0.8, 0.2)
    worker._link_worker.tick()
    clock.advance(0.5)
    worker._link_worker.tick()

    telemetry = worker._link_worker.latest_telemetry
    assert abs(telemetry.pan_deg) > 0.5 or abs(telemetry.tilt_deg) > 0.5


def test_clicking_empty_canvas_outside_stage_1_does_nothing(qtbot):
    clock = FakeClock()
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=0))
    source.start()
    link = SimTurretLink(clock)
    worker = PipelineWorker(source, link, clock, stage=Stage.STAGE_2)
    window = MainWindow(worker, source_label="SYNTHETIC")
    qtbot.addWidget(window)
    _settle_self_test(worker, clock)
    # Self-test's own pan/tilt verification move (Part 0e) already
    # nudged both axes by _SELF_TEST_MOVE_DELTA_DEG -- "the click did
    # nothing" has to mean "no further change", not "stayed at zero".
    # Ticked well past self-test's own settling budget first so none of
    # that move's residual trapezoidal-profile motion is still in
    # flight when "before" is captured.
    for _ in range(20):
        _tick(worker, clock)
    before = worker._link_worker.latest_telemetry

    window._on_canvas_clicked(0.8, 0.2)
    worker._link_worker.tick()
    clock.advance(0.5)
    worker._link_worker.tick()

    after = worker._link_worker.latest_telemetry
    assert after.pan_deg == before.pan_deg
    assert after.tilt_deg == before.tilt_deg


# --- Part 4: gamepad wiring ---


def test_gamepad_signals_are_wired_through_to_the_worker_and_status_strip(qtbot):
    from celikkubbe.ui.gamepad import GamepadWorker

    clock = FakeClock()
    link = SimTurretLink(clock)
    worker = PipelineWorker(_make_source(clock), link, clock)
    gamepad = GamepadWorker()
    window = MainWindow(worker, source_label="SYNTHETIC", gamepad=gamepad)
    qtbot.addWidget(window)

    gamepad.jog_axis_changed.emit(Axis.PAN, 15.0)
    worker._link_worker.tick()
    clock.advance(0.2)
    worker._link_worker.tick()
    assert worker._link_worker.latest_telemetry.pan_deg > 0.0

    gamepad.fire_changed.emit(True)
    assert window._input_builder.build().fire_requested
    gamepad.fire_changed.emit(False)
    assert not window._input_builder.build().fire_requested

    gamepad.arm_changed.emit(True)
    assert window._input_builder.build().arm_held
    gamepad.arm_changed.emit(False)
    assert not window._input_builder.build().arm_held

    gamepad.estop_requested.emit()
    worker._link_worker.tick()
    assert worker._link_worker.latest_telemetry.armed is False

    gamepad.connected_changed.emit(True)
    assert window._status_strip._gamepad_label.text() == strings.UI_LABEL_TR["GAMEPAD_CONNECTED"]
    gamepad.connected_changed.emit(False)
    assert window._status_strip._gamepad_label.text() == strings.UI_LABEL_TR["GAMEPAD_DISCONNECTED"]


def test_gamepad_fire_does_not_also_set_arm_unlike_the_gui_button(qtbot):
    """RT and A are independent physical controls on the gamepad -- only
    ATIŞ/space double as both fire and arm for lack of a separate
    hold-to-arm control (see _set_fire_and_arm_source's own docstring).
    """
    from celikkubbe.ui.gamepad import GamepadWorker

    clock = FakeClock()
    worker = PipelineWorker(_make_source(clock), SimTurretLink(clock), clock)
    gamepad = GamepadWorker()
    window = MainWindow(worker, source_label="SYNTHETIC", gamepad=gamepad)
    qtbot.addWidget(window)

    gamepad.fire_changed.emit(True)

    result = window._input_builder.build()
    assert result.fire_requested
    assert not result.arm_held


def test_window_without_a_gamepad_still_constructs_and_closes_cleanly(qtbot):
    clock = FakeClock()
    worker = PipelineWorker(_make_source(clock), SimTurretLink(clock), clock)
    window = MainWindow(worker, source_label="SYNTHETIC")
    qtbot.addWidget(window)
    assert window.gamepad is None
