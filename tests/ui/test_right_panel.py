"""RightPanel tests. compute_fire_enabled is a pure function, tested
directly. The widget is driven the same lightweight way LeftPanel's
tests are: feeding UiSnapshot values straight to update_from_snapshot,
with qtbot simulating real button presses (not .click()) wherever
press/release-not-click semantics are the point of the test.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt

from celikkubbe.core import strings
from celikkubbe.core.strings import UI_LABEL_TR
from celikkubbe.core.types import (
    Axis,
    CameraIntrinsics,
    EngagementState,
    Frame,
    Layer,
    Mode,
    ReasonCode,
    Stage,
)
from celikkubbe.ui.right_panel import RightPanel, compute_fire_enabled
from celikkubbe.ui.snapshot import HealthSnapshot, PipelineTimings, UiSnapshot

from ..factories import make_state, make_telemetry, make_track


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0, quality="factory"
    )


def _frame() -> Frame:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(image=image, t=0.0, intrinsics=_intrinsics(), has_depth=False, depth=None)


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
    t: float = 0.0,
    tracks=(),
    state=None,
    telemetry=None,
    engagement_fallback_reason=None,
) -> UiSnapshot:
    tracks = tuple(tracks)
    return UiSnapshot(
        t=t,
        frame=_frame(),
        detections=(),
        tracks=tracks,
        ordered_track_ids=tuple(tr.track_id for tr in tracks),
        state=state if state is not None else make_state(tracks=tracks),
        telemetry=telemetry,
        telemetry_frame=None,
        aim=None,
        crosshair_px=None,
        crosshair_offscreen=False,
        crosshair_bearing_deg=None,
        health=_health(),
        timings=_timings(),
        fault_reason=None,
        event_log=(),
        ammo_fired=None,
        engagement_fallback_reason=engagement_fallback_reason,
    )


# --- compute_fire_enabled ---


def test_compute_fire_enabled_false_when_not_operational() -> None:
    state = make_state(mode=Mode.M2_STANDBY)
    snapshot = _snapshot(state=state, telemetry=make_telemetry(armed=True))
    enabled, reason = compute_fire_enabled(snapshot)
    assert enabled is False
    assert reason == strings.describe(ReasonCode.NOT_OPERATIONAL)


def test_compute_fire_enabled_false_when_not_armed() -> None:
    state = make_state(mode=Mode.M3_OPERATIONAL)
    snapshot = _snapshot(state=state, telemetry=make_telemetry(armed=False))
    enabled, reason = compute_fire_enabled(snapshot)
    assert enabled is False
    assert reason == strings.describe(ReasonCode.NOT_ARMED)


def test_compute_fire_enabled_false_with_no_telemetry_at_all() -> None:
    state = make_state(mode=Mode.M3_OPERATIONAL)
    snapshot = _snapshot(state=state, telemetry=None)
    enabled, reason = compute_fire_enabled(snapshot)
    assert enabled is False
    assert reason == strings.describe(ReasonCode.NOT_ARMED)


def test_compute_fire_enabled_false_and_explains_when_not_yet_aiming() -> None:
    state = make_state(mode=Mode.M3_OPERATIONAL, engagement=EngagementState.S3_TRACK)
    snapshot = _snapshot(state=state, telemetry=make_telemetry(armed=True))
    enabled, reason = compute_fire_enabled(snapshot)
    assert enabled is False
    assert reason  # some explanation, even if just "searching"


def test_compute_fire_enabled_false_when_s4_but_a_gate_blocks() -> None:
    track = make_track(track_id=1)
    state = make_state(
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S4_AIM,
        tracks=(track,),
        selected_track_id=1,
    )
    snapshot = _snapshot(
        state=state,
        telemetry=make_telemetry(armed=True),
        tracks=(track,),
        engagement_fallback_reason=ReasonCode.RANGE_OUT_OF_BOUNDS,
    )
    enabled, reason = compute_fire_enabled(snapshot)
    assert enabled is False
    assert reason is not None
    assert strings.describe(ReasonCode.RANGE_OUT_OF_BOUNDS) in reason


def test_compute_fire_enabled_true_when_s4_and_gates_pass() -> None:
    track = make_track(track_id=1)
    state = make_state(
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S4_AIM,
        tracks=(track,),
        selected_track_id=1,
    )
    snapshot = _snapshot(
        state=state,
        telemetry=make_telemetry(armed=True),
        tracks=(track,),
        engagement_fallback_reason=None,
    )
    enabled, reason = compute_fire_enabled(snapshot)
    assert enabled is True
    assert reason is None


# --- widget: pinned safety region ---


def test_estop_button_click_emits_estop_requested(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.estop_requested, timeout=1000):
        qtbot.mouseClick(panel._estop_button, Qt.MouseButton.LeftButton)


def test_safety_toggle_reflects_telemetry_armed_without_feedback_loop(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    received = []
    panel.armed_changed.connect(received.append)

    panel.update_from_snapshot(_snapshot(telemetry=make_telemetry(armed=True)))
    assert panel._safety_toggle.isChecked()
    panel.update_from_snapshot(_snapshot(telemetry=make_telemetry(armed=False)))
    assert not panel._safety_toggle.isChecked()
    # Programmatic sync from telemetry must never itself emit a command.
    assert received == []


def test_clicking_safety_toggle_emits_armed_changed(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.armed_changed, timeout=1000) as blocker:
        qtbot.mouseClick(panel._safety_toggle, Qt.MouseButton.LeftButton)
    assert blocker.args == [True]


def test_fire_button_press_and_release_emit_signals(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(
        _snapshot(
            state=make_state(mode=Mode.M3_OPERATIONAL, engagement=EngagementState.S4_AIM),
            telemetry=make_telemetry(armed=True),
        )
    )
    assert panel._fire_button.isEnabled()

    with qtbot.waitSignal(panel.fire_pressed, timeout=1000):
        qtbot.mousePress(panel._fire_button, Qt.MouseButton.LeftButton)
    with qtbot.waitSignal(panel.fire_released, timeout=1000):
        qtbot.mouseRelease(panel._fire_button, Qt.MouseButton.LeftButton)


def test_fire_button_disabled_when_not_armed_and_states_reason(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(
        _snapshot(
            state=make_state(mode=Mode.M3_OPERATIONAL),
            telemetry=make_telemetry(armed=False),
        )
    )
    assert not panel._fire_button.isEnabled()
    assert panel._fire_button.toolTip() == strings.describe(ReasonCode.NOT_ARMED)
    assert panel._warning_label.text() == UI_LABEL_TR["SAFETY_WARNING_LOCKED"]


def test_warning_line_shows_unlocked_message_when_ready_to_fire(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1)
    state = make_state(
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S4_AIM,
        tracks=(track,),
        selected_track_id=1,
    )
    panel.update_from_snapshot(
        _snapshot(state=state, telemetry=make_telemetry(armed=True), tracks=(track,))
    )
    assert panel._warning_label.text() == UI_LABEL_TR["SAFETY_WARNING_UNLOCKED"]


def test_warning_line_shows_blocking_reason_when_armed_but_gate_fails(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1)
    state = make_state(
        mode=Mode.M3_OPERATIONAL,
        engagement=EngagementState.S4_AIM,
        tracks=(track,),
        selected_track_id=1,
    )
    panel.update_from_snapshot(
        _snapshot(
            state=state,
            telemetry=make_telemetry(armed=True),
            tracks=(track,),
            engagement_fallback_reason=ReasonCode.NO_AIM_SOLUTION,
        )
    )
    assert strings.describe(ReasonCode.NO_AIM_SOLUTION) in panel._warning_label.text()


def test_force_stop_all_ends_an_in_progress_jog_and_fire(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel._on_jog_pressed(Axis.PAN, 1)
    panel._on_fire_pressed()

    jog_released = []
    fire_released = []
    panel.jog_released.connect(lambda: jog_released.append(True))
    panel.fire_released.connect(lambda: fire_released.append(True))

    panel.force_stop_all()

    assert jog_released == [True]
    assert fire_released == [True]
    # Idempotent: calling it again with nothing held must not re-emit.
    panel.force_stop_all()
    assert jog_released == [True]
    assert fire_released == [True]


# --- widget: manual control pad ---


def test_jog_button_press_emits_axis_direction_and_current_speed(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel._speed_slider.setValue(35)
    btn = panel._jog_buttons[(Axis.PAN, 1)]

    with qtbot.waitSignal(panel.jog_pressed, timeout=1000) as blocker:
        qtbot.mousePress(btn, Qt.MouseButton.LeftButton)
    assert blocker.args == [Axis.PAN, 1, 35.0]

    with qtbot.waitSignal(panel.jog_released, timeout=1000):
        qtbot.mouseRelease(btn, Qt.MouseButton.LeftButton)


def test_zero_buttons_emit_zero_requested_for_the_right_axis(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)

    with qtbot.waitSignal(panel.zero_requested, timeout=1000) as blocker:
        qtbot.mouseClick(panel._homing_pan_button, Qt.MouseButton.LeftButton)
    assert blocker.args == [Axis.PAN]

    with qtbot.waitSignal(panel.zero_requested, timeout=1000) as blocker:
        qtbot.mouseClick(panel._homing_tilt_button, Qt.MouseButton.LeftButton)
    assert blocker.args == [Axis.TILT]


# --- widget: stage selector / mode cards ---


def test_stage_button_click_emits_stage_selected(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.stage_selected, timeout=1000) as blocker:
        qtbot.mouseClick(panel._stage_buttons[Stage.STAGE_1], Qt.MouseButton.LeftButton)
    assert blocker.args == [Stage.STAGE_1]


def test_mode_cards_show_three_layers_in_stage_1_and_two_elsewhere(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(_snapshot(state=make_state(stage=Stage.STAGE_1)))
    assert set(panel._mode_card_buttons) == {Layer.L1, Layer.L2, Layer.L3}

    panel.update_from_snapshot(_snapshot(state=make_state(stage=Stage.STAGE_2)))
    assert set(panel._mode_card_buttons) == {Layer.L1, Layer.L2}


def test_l1_mode_card_is_disabled_and_unavailable(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(_snapshot(state=make_state(stage=Stage.STAGE_2)))
    assert not panel._mode_card_buttons[Layer.L1].isEnabled()


def test_clicking_l2_mode_card_emits_layer_selected(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(_snapshot(state=make_state(stage=Stage.STAGE_2)))
    with qtbot.waitSignal(panel.layer_selected, timeout=1000) as blocker:
        qtbot.mouseClick(panel._mode_card_buttons[Layer.L2], Qt.MouseButton.LeftButton)
    assert blocker.args == [Layer.L2]


def test_active_mode_card_is_checked_and_shows_who_chose_it(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    state = make_state(stage=Stage.STAGE_2, active_layer=Layer.L2, layer_manual_override=True)
    panel.update_from_snapshot(_snapshot(state=state))
    btn = panel._mode_card_buttons[Layer.L2]
    assert btn.isChecked()
    assert UI_LABEL_TR["OPERATOR_CHOSEN"] in btn.text()


# --- widget: locked target ---


def test_locked_target_shows_class_and_confidence_when_selected(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=5, confidence=0.81)
    state = make_state(tracks=(track,), selected_track_id=5)
    panel.update_from_snapshot(_snapshot(state=state, tracks=(track,)))
    assert panel._locked_confidence_label.text() == "81%"


def test_locked_target_shows_none_when_nothing_selected(qtbot) -> None:
    panel = RightPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(_snapshot())
    assert panel._locked_target_label.text() == UI_LABEL_TR["THREAT_NONE"]
    assert panel._locked_confidence_label.text() == ""
