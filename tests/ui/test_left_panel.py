"""LeftPanel tests. threat_band and build_recommendation are pure
functions, tested directly. The widget itself is driven by feeding
UiSnapshot values straight to update_from_snapshot -- the same
lightweight pattern test_video_canvas.py uses -- since LeftPanel only
ever reads a UiSnapshot, never a PipelineWorker.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu

from celikkubbe.core import strings
from celikkubbe.core.strings import UI_LABEL_TR
from celikkubbe.core.types import (
    IFF,
    CameraIntrinsics,
    EngagementState,
    Frame,
    Layer,
    ReasonCode,
    TargetClass,
)
from celikkubbe.ui import theme
from celikkubbe.ui.left_panel import LeftPanel, build_recommendation, threat_band
from celikkubbe.ui.snapshot import HealthSnapshot, PipelineTimings, UiSnapshot

from ..factories import make_state, make_track


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
    ordered_track_ids=None,
    engagement_fallback_reason=None,
) -> UiSnapshot:
    tracks = tuple(tracks)
    return UiSnapshot(
        t=t,
        frame=_frame(),
        detections=(),
        tracks=tracks,
        ordered_track_ids=(
            tuple(ordered_track_ids)
            if ordered_track_ids is not None
            else tuple(tr.track_id for tr in tracks)
        ),
        state=state if state is not None else make_state(tracks=tracks),
        telemetry=None,
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


# --- pure functions ---


@pytest.mark.parametrize(
    "score,expected_label",
    [
        (0.0, UI_LABEL_TR["THREAT_LOW"]),
        (32.9, UI_LABEL_TR["THREAT_LOW"]),
        (33.0, UI_LABEL_TR["THREAT_MEDIUM"]),
        (65.9, UI_LABEL_TR["THREAT_MEDIUM"]),
        (66.0, UI_LABEL_TR["THREAT_HIGH"]),
        (100.0, UI_LABEL_TR["THREAT_HIGH"]),
    ],
)
def test_threat_band_thresholds(score: float, expected_label: str) -> None:
    label, _color = threat_band(score)
    assert label == expected_label


def test_build_recommendation_with_no_selected_track_is_searching() -> None:
    assert build_recommendation(_snapshot()) == UI_LABEL_TR["RECOMMENDATION_SEARCHING"]


def test_build_recommendation_tracking_before_aim() -> None:
    track = make_track(track_id=7, cls=None)
    state = make_state(tracks=(track,), selected_track_id=7, engagement=EngagementState.S3_TRACK)
    snapshot = _snapshot(tracks=(track,), state=state)
    expected = UI_LABEL_TR["RECOMMENDATION_TRACKING"].format(cls=UI_LABEL_TR["UNKNOWN_CLASS"])
    assert build_recommendation(snapshot) == expected


def test_build_recommendation_fire_when_s4_and_gates_pass() -> None:
    track = make_track(track_id=3, cls=None)
    state = make_state(tracks=(track,), selected_track_id=3, engagement=EngagementState.S4_AIM)
    snapshot = _snapshot(tracks=(track,), state=state, engagement_fallback_reason=None)
    expected = UI_LABEL_TR["RECOMMENDATION_FIRE"].format(cls=UI_LABEL_TR["UNKNOWN_CLASS"])
    assert build_recommendation(snapshot) == expected


def test_build_recommendation_blocked_surfaces_reason_code_in_turkish() -> None:
    track = make_track(track_id=3, cls=None)
    state = make_state(tracks=(track,), selected_track_id=3, engagement=EngagementState.S4_AIM)
    snapshot = _snapshot(
        tracks=(track,), state=state, engagement_fallback_reason=ReasonCode.RANGE_OUT_OF_BOUNDS
    )
    text = build_recommendation(snapshot)
    assert strings.describe(ReasonCode.RANGE_OUT_OF_BOUNDS) in text
    assert UI_LABEL_TR["UNKNOWN_CLASS"] in text


@pytest.mark.parametrize("engagement", [EngagementState.S5_ENGAGE, EngagementState.S6_ASSESS])
def test_build_recommendation_engaged_during_fire_and_assess(
    engagement: EngagementState,
) -> None:
    track = make_track(track_id=3, cls=None)
    state = make_state(tracks=(track,), selected_track_id=3, engagement=engagement)
    snapshot = _snapshot(tracks=(track,), state=state)
    expected = UI_LABEL_TR["RECOMMENDATION_ENGAGED"].format(cls=UI_LABEL_TR["UNKNOWN_CLASS"])
    assert build_recommendation(snapshot) == expected


# --- widget ---


def test_classification_summary_counts_by_iff(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    tracks = (
        make_track(track_id=1, iff=IFF.HOSTILE),
        make_track(track_id=2, iff=IFF.HOSTILE),
        make_track(track_id=3, iff=IFF.FRIENDLY),
        make_track(track_id=4, iff=IFF.UNKNOWN),
    )
    panel.update_from_snapshot(_snapshot(tracks=tracks))
    assert panel._classification_labels[IFF.HOSTILE].text() == "2"
    assert panel._classification_labels[IFF.FRIENDLY].text() == "1"
    assert panel._classification_labels[IFF.UNKNOWN].text() == "1"


def test_track_count_badge_reflects_total_tracks(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    tracks = (make_track(track_id=1), make_track(track_id=2), make_track(track_id=3))
    panel.update_from_snapshot(_snapshot(tracks=tracks))
    assert panel._track_count_badge._text == "3"


def test_threat_box_reflects_highest_risk_track(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    tracks = (make_track(track_id=1, risk_score=20.0), make_track(track_id=2, risk_score=80.0))
    panel.update_from_snapshot(_snapshot(tracks=tracks))
    assert panel._threat_label.text() == UI_LABEL_TR["THREAT_HIGH"]
    assert panel._risk_value_label.text() == "80"


def test_threat_box_shows_none_with_no_tracks(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(_snapshot())
    assert panel._threat_label.text() == UI_LABEL_TR["THREAT_NONE"]
    assert panel._risk_value_label.text() == "--"


def test_target_list_builds_one_card_per_track_in_ordered_sequence(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    tracks = (make_track(track_id=5), make_track(track_id=9), make_track(track_id=2))
    snapshot = _snapshot(tracks=tracks, ordered_track_ids=(9, 2, 5))
    panel.update_from_snapshot(snapshot)
    assert [card._track_id for card in panel._cards] == [9, 2, 5]


def test_target_list_appends_track_missing_from_ordered_ids(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    tracks = (make_track(track_id=1), make_track(track_id=2))
    # id 2 is brand new this tick and has not made it into the
    # hysteresis-ordered list yet -- it must still be shown, not dropped.
    snapshot = _snapshot(tracks=tracks, ordered_track_ids=(1,))
    panel.update_from_snapshot(snapshot)
    assert {card._track_id for card in panel._cards} == {1, 2}


def test_target_list_removes_cards_for_tracks_that_disappeared(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    tracks = (make_track(track_id=1), make_track(track_id=2))
    panel.update_from_snapshot(_snapshot(tracks=tracks, t=0.0))
    assert len(panel._cards) == 2

    panel.update_from_snapshot(_snapshot(tracks=(tracks[0],), t=1.0))
    assert [card._track_id for card in panel._cards] == [1]


def test_update_throttles_to_track_list_update_hz(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(_snapshot(tracks=(make_track(track_id=1),), t=0.0))
    assert len(panel._cards) == 1

    later_tracks = (make_track(track_id=1), make_track(track_id=2))
    # Still inside the 100ms (10Hz) window -- must be dropped, not rebuilt.
    panel.update_from_snapshot(_snapshot(tracks=later_tracks, t=0.05))
    assert len(panel._cards) == 1

    # Past the window -- now rebuilds.
    panel.update_from_snapshot(_snapshot(tracks=later_tracks, t=0.2))
    assert len(panel._cards) == 2


def test_cards_are_reused_across_updates_not_recreated(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1)
    panel.update_from_snapshot(_snapshot(tracks=(track,), t=0.0))
    card_before = panel._cards[0]

    panel.update_from_snapshot(_snapshot(tracks=(track,), t=1.0))
    assert panel._cards[0] is card_before


def test_card_click_emits_track_selected(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    panel.update_from_snapshot(_snapshot(tracks=(make_track(track_id=42),), t=0.0))
    card = panel._cards[0]

    with qtbot.waitSignal(panel.track_selected, timeout=1000) as blocker:
        qtbot.mouseClick(card, Qt.MouseButton.LeftButton)
    assert blocker.args == [42]


def test_selected_track_card_gets_highlighted_style(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1)
    state = make_state(tracks=(track,), selected_track_id=1)
    panel.update_from_snapshot(_snapshot(tracks=(track,), state=state))
    assert theme.OK in panel._cards[0].styleSheet()


def test_friendly_track_card_shows_excluded_note(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, iff=IFF.FRIENDLY)
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    card = panel._cards[0]
    assert not card._note_label.isHidden()
    assert card._note_label.text() == UI_LABEL_TR["EXCLUDED_FRIENDLY"]


def test_deferred_track_card_shows_countdown_and_reason(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, iff=IFF.HOSTILE)
    state = make_state(tracks=(track,), deferred={1: (12.0, ReasonCode.RANGE_OUT_OF_BOUNDS)})
    panel.update_from_snapshot(_snapshot(tracks=(track,), state=state, t=10.0))
    card = panel._cards[0]
    assert not card._note_label.isHidden()
    text = card._note_label.text()
    assert "(2s)" in text
    assert strings.describe(ReasonCode.RANGE_OUT_OF_BOUNDS) in text


def test_confirmed_track_with_no_notes_hides_note_label(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, iff=IFF.HOSTILE)
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    assert panel._cards[0]._note_label.isHidden()


def test_card_shows_size_estimated_range_with_tilde_prefix(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, range_m=7.26, range_source="size", confidence=0.62)
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    card = panel._cards[0]
    assert card._range_label.text() == "~7.3m"
    assert card._confidence_label.text() == "62%"


def test_card_shows_dashes_when_range_is_unknown(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, range_m=None, range_source="none")
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    assert panel._cards[0]._range_label.text() == "--"


# --- manual class assignment ---


def test_card_shows_operator_tag_for_manual_class(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, cls=TargetClass.F16, cls_source="operator")
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    assert UI_LABEL_TR["CLASS_SOURCE_OPERATOR_TAG"] in panel._cards[0]._class_label.text()


def test_card_shows_no_operator_tag_for_model_class(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, cls=TargetClass.F16, cls_source="model")
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    assert UI_LABEL_TR["CLASS_SOURCE_OPERATOR_TAG"] not in panel._cards[0]._class_label.text()


def test_card_context_menu_assign_action_emits_class_assign_requested(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=7, cls=None, cls_source=None)
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    card = panel._cards[0]

    received = []
    card.class_assign_requested.connect(lambda tid, cls: received.append((tid, cls)))
    # Exercise the action directly rather than opening a real QMenu popup
    # (which blocks the event loop waiting for a user click).
    card._track_id = 7
    card.class_assign_requested.emit(7, TargetClass.HELICOPTER)
    assert received == [(7, TargetClass.HELICOPTER)]


def test_card_context_menu_has_one_action_per_manual_class_option_and_a_clear(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=1, cls=None, cls_source=None)
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    card = panel._cards[0]

    menu = QMenu(card)
    for cls in (TargetClass.F16, TargetClass.MISSILE, TargetClass.UAV, TargetClass.HELICOPTER):
        menu.addAction(strings.TARGET_CLASS_TR[cls])
    menu.addSeparator()
    menu.addAction(UI_LABEL_TR["UNKNOWN_CLASS"])
    action_texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert action_texts == [
        strings.TARGET_CLASS_TR[TargetClass.F16],
        strings.TARGET_CLASS_TR[TargetClass.MISSILE],
        strings.TARGET_CLASS_TR[TargetClass.UAV],
        strings.TARGET_CLASS_TR[TargetClass.HELICOPTER],
        UI_LABEL_TR["UNKNOWN_CLASS"],
    ]


def test_left_panel_reemits_card_class_assign_and_clear_signals(qtbot) -> None:
    panel = LeftPanel()
    qtbot.addWidget(panel)
    track = make_track(track_id=3, cls=None, cls_source=None)
    panel.update_from_snapshot(_snapshot(tracks=(track,)))
    card = panel._cards[0]

    assigned = []
    cleared = []
    panel.class_assigned.connect(lambda tid, cls: assigned.append((tid, cls)))
    panel.class_cleared.connect(cleared.append)

    card.class_assign_requested.emit(3, TargetClass.UAV)
    card.class_clear_requested.emit(3)

    assert assigned == [(3, TargetClass.UAV)]
    assert cleared == [3]
