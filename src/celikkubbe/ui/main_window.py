"""MainWindow: the application shell.

Three-column splitter (LeftPanel / VideoCanvas / RightPanel), the status
strip, and the two full-window overlays -- wired to a PipelineWorker this
window owns. Nothing in this module reads a camera, a serial port, or
runs inference -- the GUI thread only ever paints and reacts to signals
emitted from PipelineWorker's own thread. The product name lives only in
the OS window title bar (setWindowTitle) -- no in-app title row -- and
the version string lives in the status strip instead.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from celikkubbe import __version__
from celikkubbe.core import config
from celikkubbe.core.strings import UI_LABEL_TR
from celikkubbe.core.types import (
    IFF,
    Axis,
    EngagementState,
    Layer,
    Mode,
    Stage,
    TargetClass,
    Track,
    TrackStatus,
)
from celikkubbe.ui import theme
from celikkubbe.ui.gamepad import GamepadWorker
from celikkubbe.ui.left_panel import LeftPanel
from celikkubbe.ui.operator_input import OperatorInputBuilder
from celikkubbe.ui.overlays.help import HelpOverlay
from celikkubbe.ui.overlays.safe import SafeOverlay
from celikkubbe.ui.overlays.selftest import SelfTestOverlay
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.ui.right_panel import RightPanel
from celikkubbe.ui.snapshot import UiSnapshot
from celikkubbe.ui.theme import StatusBadge
from celikkubbe.ui.tuning_window import TuningWindow
from celikkubbe.ui.video_canvas import VideoCanvas

_STATUS_STRIP_HEIGHT_PX = 28
_ERROR_TOAST_MS = 5000
# A manual click-to-aim point has no detected size, so it is represented
# as a zero-area bbox exactly at the click -- AimSolver only ever reads
# its centre (see geometry.solver.AimSolver.solve's own u_center/v_center
# computation), so this is not degenerate for that purpose.
_JOG_KEYS: dict[int, tuple[Axis, int]] = {
    Qt.Key.Key_Up: (Axis.TILT, 1),
    Qt.Key.Key_Down: (Axis.TILT, -1),
    Qt.Key.Key_Left: (Axis.PAN, -1),
    Qt.Key.Key_Right: (Axis.PAN, 1),
}
_STAGE_KEYS: dict[int, Stage] = {
    Qt.Key.Key_1: Stage.STAGE_1,
    Qt.Key.Key_2: Stage.STAGE_2,
    Qt.Key.Key_3: Stage.STAGE_3,
}
# Manual class assignment on the currently selected track, for a
# demonstration where opening a context menu per shot is too slow. H
# would be the natural mnemonic for HELİKOPTER, but it is already bound
# to the tuning window (see keyPressEvent) -- K stands in instead rather
# than reassigning an established, tested shortcut.
_MANUAL_CLASS_KEYS: dict[int, TargetClass] = {
    Qt.Key.Key_F: TargetClass.F16,
    Qt.Key.Key_M: TargetClass.MISSILE,
    Qt.Key.Key_D: TargetClass.UAV,
    Qt.Key.Key_K: TargetClass.HELICOPTER,
}

_MODE_COLOR: dict[Mode, str] = {
    Mode.M1_INIT: theme.TEXT_DIM,
    Mode.M2_STANDBY: theme.WARN,
    Mode.M3_OPERATIONAL: theme.OK,
    Mode.M4_SAFE: theme.DANGER,
}
_ENGAGEMENT_COLOR: dict[EngagementState, str] = {
    EngagementState.S1_SEARCH: theme.TEXT_DIM,
    EngagementState.S2_ACQUIRE: theme.TEXT_DIM,
    EngagementState.S3_TRACK: theme.OK,
    EngagementState.S4_AIM: theme.WARN,
    EngagementState.S5_ENGAGE: theme.DANGER,
    EngagementState.S6_ASSESS: theme.DANGER,
}


def _layer_badge_color(layer: Layer, fallback_reason) -> str:
    """WARN is reserved for a genuine health-triggered fallback -- not
    for "L2 because that is the only detector that exists yet". With no
    L1/YOLO pipeline built, active_layer is always L2 and
    fallback_reason is always None (see HealthSnapshot's own
    docstring), so colouring L2 as a warning by default would
    permanently read as a degraded state for something that was never
    actually a fallback from anything.
    """
    if fallback_reason is not None:
        return theme.WARN
    if layer is Layer.L1:
        return theme.OK
    if layer is Layer.L3:
        return theme.ACCENT_A1
    return theme.TEXT_DIM


def _format_duration_ms(value_ms: float) -> str:
    """L2 colour detection runs in well under a millisecond, so a fixed
    "X.Xms" format has no resolution down there -- it always reads
    "0.0ms", telling the operator nothing. Below 1ms this switches to
    microseconds; below 10ms it keeps one decimal (enough resolution to
    actually see a real number change); at or above 10ms, whole
    milliseconds are plenty and reduce strip clutter.
    """
    if value_ms < 1.0:
        return f"{value_ms * 1000.0:.0f}µs"
    if value_ms < 10.0:
        return f"{value_ms:.1f}ms"
    return f"{value_ms:.0f}ms"


def _synthetic_click_track(x_norm: float, y_norm: float) -> Track:
    """A manual click has no detected size, velocity or range -- a
    zero-area bbox exactly at the click, with range_m=None so AimSolver
    substitutes config.DEFAULT_RANGE_M exactly like it would for any real
    CONFIRMED track with unknown range (see AimSolver.solve's own
    docstring). track_id=-1 is never a real track id and is only read
    transiently by solve() itself -- it is never stored anywhere.
    """
    return Track(
        track_id=-1,
        cls=None,
        confidence=1.0,
        range_m=None,
        range_source="none",
        iff=IFF.UNKNOWN,
        bbox=(x_norm, y_norm, x_norm, y_norm),
        velocity=(0.0, 0.0),
        status=TrackStatus.CONFIRMED,
        risk_score=0.0,
        frames_confirmed=0,
        last_seen_t=0.0,
    )


class StatusStrip(QWidget):
    """Always-visible bottom strip: mode/engagement/layer badges, link
    health, performance, homing status and counters. The operator's
    primary situational display -- see the module-level rationale in the
    prompt this was built from: without it there is no way to tell
    whether the system is searching, tracking or aiming, and a dropped
    link would otherwise look identical to a live one.
    """

    def __init__(self, dev_mode: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusStrip")
        self.setFixedHeight(_STATUS_STRIP_HEIGHT_PX)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(14)

        if dev_mode:
            # Deliberately the first badge, before even the mode badge --
            # the whole point is that this must be impossible to miss.
            dev_badge = StatusBadge("DEV", theme.DANGER)
            dev_badge.setFixedWidth(40)
            layout.addWidget(dev_badge)

        self._mode_badge = StatusBadge("M?", theme.TEXT_DIM)
        self._engagement_badge = StatusBadge("S?", theme.TEXT_DIM)
        self._layer_badge = StatusBadge("L?", theme.TEXT_DIM)
        for badge in (self._mode_badge, self._engagement_badge, self._layer_badge):
            badge.setFixedWidth(40)
            layout.addWidget(badge)

        self._link_label = self._add_text_label(layout, "LINK OK 999ms CRC:999")
        self._perf_label = self._add_text_label(layout, "FPS:99.9 INF:999µs L1:YOK")
        self._homing_label = self._add_text_label(layout, "HOMED P:Y T:Y")
        self._counters_label = self._add_text_label(layout, "AMMO:999 ATT:9/9 TRK:99")
        self._gamepad_label = self._add_text_label(layout, UI_LABEL_TR["GAMEPAD_DISCONNECTED"])
        self._error_label = self._add_text_label(layout, "")

        layout.addStretch(1)
        # Moved here from the now-removed in-app title row -- this is the
        # only place the version string is shown; the window title bar
        # itself (see MainWindow.__init__) carries the product name.
        version_label = QLabel(f"v{__version__}")
        version_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        layout.addWidget(version_label)

        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(lambda: self._error_label.setText(""))
        self.set_gamepad_connected(False)

    def _add_text_label(self, layout: QHBoxLayout, width_reference: str) -> QLabel:
        """``width_reference`` is never displayed -- it only sizes the
        label's minimum width before any real text arrives. Without it,
        an initially-empty QLabel's sizeHint is near zero, and the very
        first update_from_snapshot() call after construction lands
        before Qt has a chance to relayout for the real (much longer)
        text -- a one-frame squished status strip at startup. A fixed
        minimum width also stops later updates from jittering the
        layout as text length varies tick to tick (e.g. "LINK OK ..." vs
        the shorter "LINK LOST ...").
        """
        label = QLabel(width_reference)
        label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        label.setMinimumWidth(label.sizeHint().width())
        label.setText("")
        layout.addWidget(label)
        return label

    def update_from_snapshot(self, snapshot: UiSnapshot) -> None:
        state = snapshot.state
        health = snapshot.health
        telemetry = snapshot.telemetry

        self._mode_badge.set_status(state.mode.value[:2], _MODE_COLOR[state.mode])
        self._engagement_badge.set_status(
            state.engagement.value[:2], _ENGAGEMENT_COLOR[state.engagement]
        )
        layer_color = _layer_badge_color(state.active_layer, health.fallback_reason)
        self._layer_badge.set_status(state.active_layer.value, layer_color)

        self._update_link_label(health)
        self._update_perf_label(health)
        self._update_homing_label(telemetry)
        self._update_counters_label(snapshot)

    def _update_link_label(self, health) -> None:
        if health.link_healthy:
            self._link_label.setText(
                f"LINK OK {health.link_stale_ms:.0f}ms CRC:{health.crc_error_count}"
            )
            self._link_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        else:
            self._link_label.setText(f"LINK LOST CRC:{health.crc_error_count}")
            self._link_label.setStyleSheet(f"color: {theme.DANGER}; font-weight: 600;")

    def _update_perf_label(self, health) -> None:
        text = f"FPS:{health.camera_fps:.1f} INF:{_format_duration_ms(health.inference_ms)}"
        if health.fallback_reason is not None:
            text += f" FALLBACK:{health.fallback_reason.value}"
        elif health.active_layer is Layer.L2:
            # Not a fallback -- no L1/YOLO pipeline exists yet, so L2 is
            # simply the only detector available. See _layer_badge_color.
            text += f" {UI_LABEL_TR['L1_UNAVAILABLE']}"
        if health.l1_recovery_countdown_s is not None:
            text += f" RECOV:{health.l1_recovery_countdown_s:.0f}s"
        self._perf_label.setText(text)

    def _update_homing_label(self, telemetry) -> None:
        if telemetry is None:
            self._homing_label.setText("HOMED --/--")
            return
        pan = "Y" if telemetry.homed_pan else "N"
        tilt = "Y" if telemetry.homed_tilt else "N"
        self._homing_label.setText(f"HOMED P:{pan} T:{tilt}")

    def _update_counters_label(self, snapshot: UiSnapshot) -> None:
        state = snapshot.state
        ammo_text = str(snapshot.ammo_fired) if snapshot.ammo_fired is not None else "--"
        attempts = state.attempts.get(state.selected_track_id, 0) if state.selected_track_id else 0
        max_attempts = config.MAX_ENGAGEMENT_ATTEMPTS
        self._counters_label.setText(
            f"AMMO:{ammo_text} ATT:{attempts}/{max_attempts} TRK:{len(state.tracks)}"
        )

    def show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.setStyleSheet(f"color: {theme.DANGER};")
        self._error_timer.start(_ERROR_TOAST_MS)

    def set_gamepad_connected(self, connected: bool) -> None:
        text = UI_LABEL_TR["GAMEPAD_CONNECTED" if connected else "GAMEPAD_DISCONNECTED"]
        self._gamepad_label.setText(text)
        self._gamepad_label.setStyleSheet(f"color: {theme.OK if connected else theme.TEXT_DIM};")


def _build_dev_mode_banner() -> QLabel:
    """A permanent, full-width red strip -- not an overlay, not a toast,
    never hidden or throttled. dev_mode disables real safety checks (see
    modes.step's own docstring), so running it unnoticed at the
    competition would be the worst possible outcome; this banner exists
    to make that structurally impossible to miss, not just logged.
    """
    banner = QLabel(UI_LABEL_TR["DEV_MODE_BANNER"])
    banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
    banner.setStyleSheet(
        f"background-color: {theme.DANGER}; color: {theme.BG_BASE}; "
        "font-weight: 700; padding: 4px;"
    )
    return banner


class MainWindow(QMainWindow):
    def __init__(
        self,
        worker: PipelineWorker,
        source_label: str = "",
        font_family: str = "monospace",
        gamepad: GamepadWorker | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._worker = worker
        self._font_family = font_family
        self._gamepad = gamepad
        self._input_builder = OperatorInputBuilder()
        self._latest_snapshot: UiSnapshot | None = None
        self._held_jog_key: int | None = None
        self._tuning_window: TuningWindow | None = None

        self.setWindowTitle("OZU IEEE RAS ÇELİKKUBBE")
        self.resize(1400, 800)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        if self._worker.dev_mode:
            outer.addWidget(_build_dev_mode_banner())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._left_panel = LeftPanel()
        self._left_panel.setMinimumWidth(220)
        self._canvas = VideoCanvas()
        self._canvas.set_source_label(source_label)
        self._right_panel = RightPanel()  # owns its own minimum width -- see right_panel.py
        splitter.addWidget(self._left_panel)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._right_panel)
        splitter.setSizes([320, 900, 330])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        outer.addWidget(splitter, stretch=1)

        self._status_strip = StatusStrip(dev_mode=self._worker.dev_mode)
        outer.addWidget(self._status_strip)

        self._selftest_overlay = SelfTestOverlay(dev_mode=self._worker.dev_mode, parent=self)
        self._selftest_overlay.retry_requested.connect(self._worker.retry_self_test)
        self._selftest_overlay.skip_requested.connect(self._worker.skip_self_test)
        self._selftest_overlay.hide()

        self._safe_overlay = SafeOverlay(self)
        self._safe_overlay.acknowledge_requested.connect(self._worker.acknowledge_fault)
        self._safe_overlay.hide()

        self._help_overlay = HelpOverlay(self)
        self._help_overlay.close_requested.connect(self._hide_help_overlay)
        self._help_overlay.hide()

        self._canvas.clicked_normalized.connect(self._on_canvas_clicked)
        self._left_panel.track_selected.connect(self._select_track)
        self._left_panel.class_assigned.connect(self._worker.set_track_class)
        self._left_panel.class_cleared.connect(self._worker.clear_track_class)
        self._wire_right_panel()
        if self._gamepad is not None:
            self._wire_gamepad(self._gamepad)
        self._worker.snapshot_ready.connect(self._on_snapshot)
        self._worker.error.connect(self._on_error)

    @property
    def gamepad(self) -> GamepadWorker | None:
        """The GamepadWorker passed in at construction, if any -- exposed
        so app.py's aboutToQuit safety net can stop it without reaching
        into a private attribute (see closeEvent for the ordinary
        window-close path, which already stops it the same way).
        """
        return self._gamepad

    def _wire_right_panel(self) -> None:
        panel = self._right_panel
        panel.estop_requested.connect(self._worker.request_estop)
        panel.armed_changed.connect(self._worker.request_arm)
        panel.zero_requested.connect(self._worker.request_zero)
        panel.jog_pressed.connect(self._worker.request_jog)
        panel.jog_released.connect(self._worker.request_stop)
        panel.layer_selected.connect(self._worker.select_layer)
        panel.stage_selected.connect(self._on_stage_selected)
        panel.fire_pressed.connect(lambda: self._set_fire_and_arm_source("gui", True))
        panel.fire_released.connect(lambda: self._set_fire_and_arm_source("gui", False))
        panel.tuning_window_requested.connect(self._toggle_tuning_window)

    def _wire_gamepad(self, gamepad: GamepadWorker) -> None:
        gamepad.jog_axis_changed.connect(self._on_gamepad_jog_axis_changed)
        # RT and A are independent physical controls on the gamepad --
        # unlike ATIŞ/space, which double as both fire and the Stage 1
        # dead-man switch for lack of a separate hold-to-arm control (see
        # _set_fire_and_arm_source), these stay separate here.
        gamepad.fire_changed.connect(lambda held: self._set_fire_source("gamepad", held))
        gamepad.arm_changed.connect(lambda held: self._set_arm_source("gamepad", held))
        gamepad.estop_requested.connect(self._worker.request_estop)
        gamepad.connected_changed.connect(self._status_strip.set_gamepad_connected)

    # --- Qt overrides ---

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._reposition_overlays()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._gamepad is not None:
            self._gamepad.stop()
            self._gamepad.wait(2000)
        self._worker.stop()
        self._worker.wait(2000)
        super().closeEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        # A jog or fire button held down when the window loses focus (an
        # alt-tab, a modal dialog stealing activation) must not leave the
        # turret jogging, or the fire dead-man switch armed, with
        # nobody's finger actually on the control anymore.
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self._right_panel.force_stop_all()
            self._release_jog_key()
            self._set_fire_and_arm_source("keyboard", False)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in _JOG_KEYS:
            self._held_jog_key = key
            axis, direction = _JOG_KEYS[key]
            self._worker.request_jog(axis, direction, self._right_panel.jog_speed_dps)
        elif key == Qt.Key.Key_Space:
            self._set_fire_and_arm_source("keyboard", True)
        elif key == Qt.Key.Key_Escape:
            self._worker.request_estop()
        elif key == Qt.Key.Key_A:
            self._toggle_armed()
        elif key in _STAGE_KEYS:
            self._on_stage_selected(_STAGE_KEYS[key])
        elif key == Qt.Key.Key_H:
            self._toggle_tuning_window()
        elif key == Qt.Key.Key_F1:
            self._toggle_help_overlay()
        elif key in _MANUAL_CLASS_KEYS:
            self._assign_class_to_selected_track(_MANUAL_CLASS_KEYS[key])
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in _JOG_KEYS:
            self._release_jog_key(key)
        elif key == Qt.Key.Key_Space:
            self._set_fire_and_arm_source("keyboard", False)
        else:
            super().keyReleaseEvent(event)
            return
        event.accept()

    # --- wiring ---

    def _reposition_overlays(self) -> None:
        central = self.centralWidget()
        if central is None:
            return
        rect = central.rect()
        self._selftest_overlay.setGeometry(rect)
        self._safe_overlay.setGeometry(rect)
        self._help_overlay.setGeometry(rect)

    def _on_snapshot(self, snapshot: UiSnapshot) -> None:
        self._latest_snapshot = snapshot
        self._canvas.set_snapshot(snapshot)
        self._status_strip.update_from_snapshot(snapshot)
        self._left_panel.update_from_snapshot(snapshot)
        self._right_panel.update_from_snapshot(snapshot)
        self._update_overlays(snapshot)

    def _update_overlays(self, snapshot: UiSnapshot) -> None:
        """At most one of selftest/safe is ever shown -- enforced directly
        here (each branch hides the other explicitly) rather than left as
        an emergent property of two separate if/else blocks both keyed off
        the same mode, which is easy to silently break with a future edit
        to just one of them.
        """
        mode = snapshot.state.mode

        if mode is Mode.M1_INIT:
            self._safe_overlay.hide()
            self._selftest_overlay.set_result(snapshot.state.last_self_test)
            self._reposition_overlays()
            self._selftest_overlay.show()
            self._selftest_overlay.raise_()
        elif mode is Mode.M4_SAFE:
            self._selftest_overlay.hide()
            failure_detail = self._self_test_failure_detail(snapshot)
            self._safe_overlay.set_fault(snapshot.fault_reason, failure_detail)
            position_valid = (
                snapshot.telemetry.position_valid if snapshot.telemetry is not None else False
            )
            self._safe_overlay.set_position_valid(position_valid)
            self._safe_overlay.set_event_log(snapshot.event_log)
            self._reposition_overlays()
            self._safe_overlay.show()
            self._safe_overlay.raise_()
        else:
            self._selftest_overlay.hide()
            self._safe_overlay.hide()

    @staticmethod
    def _self_test_failure_detail(snapshot: UiSnapshot) -> str | None:
        last_self_test = snapshot.state.last_self_test
        if last_self_test is None or last_self_test.passed:
            return None
        failing = next((item for item in last_self_test.items if not item.passed), None)
        return failing.detail if failing is not None else None

    def _on_error(self, message: str) -> None:
        self._status_strip.show_error(message)

    def _on_canvas_clicked(self, x_norm: float, y_norm: float) -> None:
        track = self._find_track_at(x_norm, y_norm)
        if track is not None:
            self._select_track(track.track_id)
            return
        if self._latest_snapshot is None or self._latest_snapshot.state.stage is not Stage.STAGE_1:
            return
        self._goto_click_point(x_norm, y_norm)

    def _goto_click_point(self, x_norm: float, y_norm: float) -> None:
        """Stage 1, clicking empty canvas: a direct Goto through the same
        AimSolver _tick_inner uses for every real track (see
        PipelineWorker.aim_solver's own docstring), not a raw bearing --
        the click still benefits from ballistic drop and boresight
        correction even though it is not a detected target.
        """
        snapshot = self._latest_snapshot
        synthetic = _synthetic_click_track(x_norm, y_norm)
        solution = self._worker.aim_solver.solve(synthetic, snapshot.frame.intrinsics)
        if solution is None:
            return  # unreachable: _synthetic_click_track is always CONFIRMED
        self._worker.request_goto(solution.az_deg, solution.el_deg)

    def _select_track(self, track_id: int) -> None:
        self._input_builder.set_manual_target(track_id)
        self._worker.set_operator_input(self._input_builder.build())

    def _assign_class_to_selected_track(self, cls: TargetClass) -> None:
        """F/M/D/K on the currently selected (locked) track -- see
        _MANUAL_CLASS_KEYS's own docstring on the H/HELİKOPTER collision.
        A no-op with nothing selected, same as every other selected-
        track-only action in this class.
        """
        if self._latest_snapshot is None:
            return
        track_id = self._latest_snapshot.state.selected_track_id
        if track_id is None:
            return
        self._worker.set_track_class(track_id, cls)

    def _on_stage_selected(self, stage: Stage) -> None:
        self._worker.set_stage(stage)
        # Accent colour is applied application-wide (see theme.build_qss's
        # own docstring for why this rebuilds the whole stylesheet rather
        # than a per-widget override) -- stage changes happen at most a
        # handful of times per session, so the cost is irrelevant.
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.build_qss(theme.accent_for_stage(stage), self._font_family))

    def _set_fire_source(self, source: str, held: bool) -> None:
        self._input_builder.set_fire(source, held)
        self._worker.set_operator_input(self._input_builder.build())

    def _set_arm_source(self, source: str, held: bool) -> None:
        self._input_builder.set_arm(source, held)
        self._worker.set_operator_input(self._input_builder.build())

    def _set_fire_and_arm_source(self, source: str, held: bool) -> None:
        """ATIŞ and the space bar both double as fire_requested *and*
        Stage 1's own continuous dead-man switch (arm_held) -- see
        engagement.py's module docstring -- for lack of a separate
        hold-to-arm control on either. Releasing either anytime before
        the shot is taken aborts back to S3_TRACK on its own, via
        armed_by_operator turning False.
        """
        self._input_builder.set_fire(source, held)
        self._input_builder.set_arm(source, held)
        self._worker.set_operator_input(self._input_builder.build())

    def _release_jog_key(self, key: int | None = None) -> None:
        if key is not None and self._held_jog_key != key:
            return  # a different key is still shadowing this one -- see keyPressEvent
        if self._held_jog_key is None:
            return
        self._held_jog_key = None
        self._worker.request_stop()

    def _toggle_armed(self) -> None:
        telemetry = self._latest_snapshot.telemetry if self._latest_snapshot is not None else None
        currently_armed = telemetry is not None and telemetry.armed
        self._worker.request_arm(not currently_armed)

    def _on_gamepad_jog_axis_changed(self, axis: Axis, speed_dps: float) -> None:
        direction = 1 if speed_dps >= 0.0 else -1
        self._worker.request_jog(axis, direction, abs(speed_dps))

    def _build_menu_bar(self) -> None:
        # Second documented way into the tuning window, alongside
        # RightPanel's own button -- see right_panel.py's
        # _build_tuning_button. H still works too, unchanged.
        tools_menu = self.menuBar().addMenu(UI_LABEL_TR["TOOLS_MENU"])
        tuning_action = QAction(UI_LABEL_TR["OPEN_TUNING_WINDOW"], self)
        tuning_action.triggered.connect(self._toggle_tuning_window)
        tools_menu.addAction(tuning_action)

    def _toggle_tuning_window(self) -> None:
        if self._tuning_window is None:
            self._tuning_window = TuningWindow(self._worker, parent=self)
        if self._tuning_window.isVisible():
            self._tuning_window.hide()
        else:
            self._tuning_window.show()
            self._tuning_window.raise_()

    def _toggle_help_overlay(self) -> None:
        # isHidden(), not isVisible(): the latter also depends on every
        # ancestor's own visibility, so it never reads True for a widget
        # inside a MainWindow that itself is never shown (as in this
        # module's own tests) -- see SelfTestOverlay/SafeOverlay's own
        # isHidden()-based checks for the same established reason.
        if not self._help_overlay.isHidden():
            self._hide_help_overlay()
            return
        self._reposition_overlays()
        self._help_overlay.show()
        self._help_overlay.raise_()

    def _hide_help_overlay(self) -> None:
        self._help_overlay.hide()

    def _find_track_at(self, x_norm: float, y_norm: float) -> Track | None:
        if self._latest_snapshot is None:
            return None
        for track in self._latest_snapshot.tracks:
            x1, y1, x2, y2 = track.bbox
            if x1 <= x_norm <= x2 and y1 <= y_norm <= y2:
                return track
        return None
