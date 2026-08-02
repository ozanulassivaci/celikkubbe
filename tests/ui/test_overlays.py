from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton

from celikkubbe.core import strings
from celikkubbe.core.types import Axis, ReasonCode, SelfTestItem, SelfTestResult
from celikkubbe.io.codec import EventId
from celikkubbe.ui.overlays.safe import SafeOverlay
from celikkubbe.ui.overlays.selftest import SelfTestOverlay


def test_selftest_overlay_shows_no_rows_with_no_result(qapp):
    overlay = SelfTestOverlay()
    overlay.set_result(None)
    assert overlay.current_items == ()
    assert overlay._rows_layout.count() == 0


def test_selftest_overlay_shows_one_row_per_item(qapp):
    overlay = SelfTestOverlay()
    result = SelfTestResult(
        items=(
            SelfTestItem("camera", True, None, 29.5),
            SelfTestItem("stm32_link", False, "no telemetry", None),
        )
    )
    overlay.set_result(result)
    assert overlay.current_items == result.items
    assert overlay._rows_layout.count() == 2


def test_selftest_overlay_rebuilds_rows_on_second_call(qapp):
    overlay = SelfTestOverlay()
    overlay.set_result(SelfTestResult(items=(SelfTestItem("camera", True, None, 30.0),)))
    overlay.set_result(
        SelfTestResult(
            items=(
                SelfTestItem("camera", True, None, 30.0),
                SelfTestItem("stm32_link", True, None, 5.0),
            )
        )
    )
    assert overlay._rows_layout.count() == 2


def test_selftest_overlay_retry_button_emits_signal(qtbot):
    overlay = SelfTestOverlay()
    qtbot.addWidget(overlay)
    received = []
    overlay.retry_requested.connect(lambda: received.append(True))

    buttons = overlay.findChildren(QPushButton)
    assert len(buttons) == 1
    qtbot.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    assert received == [True]


def test_safe_overlay_shows_reason_code_text(qapp):
    overlay = SafeOverlay()
    overlay.set_fault(ReasonCode.ESTOP_ACTIVE)
    assert overlay._fault_label.text() == strings.describe(ReasonCode.ESTOP_ACTIVE)


def test_safe_overlay_prefers_self_test_detail_over_reason_code(qapp):
    overlay = SafeOverlay()
    overlay.set_fault(ReasonCode.ESTOP_ACTIVE, self_test_detail="no camera signal")
    assert "no camera signal" in overlay._fault_label.text()


def test_safe_overlay_homing_warning_follows_position_valid(qapp):
    overlay = SafeOverlay()
    # isHidden() reflects the widget's own explicit visibility flag; unlike
    # isVisible(), it does not also require the whole window to be shown,
    # which this standalone-widget test never does.
    overlay.set_position_valid(False)
    assert not overlay._homing_warning.isHidden()
    overlay.set_position_valid(True)
    assert overlay._homing_warning.isHidden()


def test_safe_overlay_event_log_renders_all_entries(qapp):
    overlay = SafeOverlay()
    events = (
        (1.0, EventId.ESTOP_PRESSED, None),
        (1.5, EventId.DRIVER_ALARM_ASSERTED, Axis.PAN),
    )
    overlay.set_event_log(events)
    text = overlay._log.toPlainText()
    assert "ESTOP_PRESSED" in text
    assert "DRIVER_ALARM_ASSERTED" in text
    assert "PAN" in text


def test_safe_overlay_event_log_replaces_rather_than_accumulates(qapp):
    overlay = SafeOverlay()
    overlay.set_event_log(((1.0, EventId.ESTOP_PRESSED, None),))
    overlay.set_event_log(((2.0, EventId.ESTOP_RELEASED, None),))
    text = overlay._log.toPlainText()
    assert "ESTOP_PRESSED" not in text
    assert "ESTOP_RELEASED" in text


def test_safe_overlay_acknowledge_button_emits_signal(qtbot):
    overlay = SafeOverlay()
    qtbot.addWidget(overlay)
    received = []
    overlay.acknowledge_requested.connect(lambda: received.append(True))

    buttons = overlay.findChildren(QPushButton)
    assert len(buttons) == 1
    qtbot.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    assert received == [True]
