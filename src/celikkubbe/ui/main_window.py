"""MainWindow: the application shell.

Title bar, three-column splitter (left panel / VideoCanvas / right
panel), the status strip, and the two full-window overlays -- wired to a
PipelineWorker this window owns. Left/right panels are placeholders here;
they are populated in the next prompt. Nothing in this module reads a
camera, a serial port, or runs inference -- the GUI thread only ever
paints and reacts to signals emitted from PipelineWorker's own thread.
"""

from __future__ import annotations

import dataclasses

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
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
from celikkubbe.core.types import EngagementState, Layer, Mode, OperatorInput, Track
from celikkubbe.ui import theme
from celikkubbe.ui.overlays.safe import SafeOverlay
from celikkubbe.ui.overlays.selftest import SelfTestOverlay
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.ui.snapshot import UiSnapshot
from celikkubbe.ui.theme import StatusBadge
from celikkubbe.ui.video_canvas import VideoCanvas

_STATUS_STRIP_HEIGHT_PX = 28
_TITLE_BAR_HEIGHT_PX = 36
_ERROR_TOAST_MS = 5000

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


class StatusStrip(QWidget):
    """Always-visible bottom strip: mode/engagement/layer badges, link
    health, performance, homing status and counters. The operator's
    primary situational display -- see the module-level rationale in the
    prompt this was built from: without it there is no way to tell
    whether the system is searching, tracking or aiming, and a dropped
    link would otherwise look identical to a live one.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusStrip")
        self.setFixedHeight(_STATUS_STRIP_HEIGHT_PX)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(14)

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
        self._error_label = self._add_text_label(layout, "")

        layout.addStretch(1)
        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(lambda: self._error_label.setText(""))

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


def _build_title_bar() -> QFrame:
    bar = QFrame()
    bar.setFixedHeight(_TITLE_BAR_HEIGHT_PX)
    bar.setProperty("role", "panel")
    layout = QHBoxLayout(bar)

    version_label = QLabel(f"v{__version__}")
    version_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
    layout.addWidget(version_label)
    layout.addStretch(1)

    title_label = QLabel("ÇELİKKUBBE")
    title_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-weight: 600; font-size: 12pt;")
    layout.addWidget(title_label)
    layout.addStretch(1)
    return bar


def _build_placeholder_panel(title: str) -> QFrame:
    """Left/right panel content lands in the next prompt; this is just
    the frame and minimum sizing the splitter needs to lay out correctly.
    """
    panel = QFrame()
    panel.setProperty("role", "panel")
    layout = QVBoxLayout(panel)
    label = QLabel(title)
    label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
    layout.addWidget(label)
    layout.addStretch(1)
    return panel


class MainWindow(QMainWindow):
    def __init__(
        self,
        worker: PipelineWorker,
        source_label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._worker = worker
        self._operator_input = OperatorInput()
        self._latest_snapshot: UiSnapshot | None = None

        self.setWindowTitle("Çelikkubbe")
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_build_title_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = _build_placeholder_panel("AI DECISION SUPPORT")
        left_panel.setMinimumWidth(220)
        self._canvas = VideoCanvas()
        self._canvas.set_source_label(source_label)
        right_panel = _build_placeholder_panel("ENGAGEMENT CONTROL")
        right_panel.setMinimumWidth(220)
        splitter.addWidget(left_panel)
        splitter.addWidget(self._canvas)
        splitter.addWidget(right_panel)
        splitter.setSizes([320, 900, 330])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        outer.addWidget(splitter, stretch=1)

        self._status_strip = StatusStrip()
        outer.addWidget(self._status_strip)

        self._selftest_overlay = SelfTestOverlay(self)
        self._selftest_overlay.retry_requested.connect(self._worker.retry_self_test)
        self._selftest_overlay.hide()

        self._safe_overlay = SafeOverlay(self)
        self._safe_overlay.acknowledge_requested.connect(self._worker.acknowledge_fault)
        self._safe_overlay.hide()

        self._canvas.clicked_normalized.connect(self._on_canvas_clicked)
        self._worker.snapshot_ready.connect(self._on_snapshot)
        self._worker.error.connect(self._on_error)

    # --- Qt overrides ---

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._reposition_overlays()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._worker.stop()
        self._worker.wait(2000)
        super().closeEvent(event)

    # --- wiring ---

    def _reposition_overlays(self) -> None:
        central = self.centralWidget()
        if central is None:
            return
        rect = central.rect()
        self._selftest_overlay.setGeometry(rect)
        self._safe_overlay.setGeometry(rect)

    def _on_snapshot(self, snapshot: UiSnapshot) -> None:
        self._latest_snapshot = snapshot
        self._canvas.set_snapshot(snapshot)
        self._status_strip.update_from_snapshot(snapshot)
        self._update_overlays(snapshot)

    def _update_overlays(self, snapshot: UiSnapshot) -> None:
        mode = snapshot.state.mode

        if mode is Mode.M1_INIT:
            self._selftest_overlay.set_result(snapshot.state.last_self_test)
            self._reposition_overlays()
            self._selftest_overlay.show()
            self._selftest_overlay.raise_()
        else:
            self._selftest_overlay.hide()

        if mode is Mode.M4_SAFE:
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
        if track is None:
            return
        self._operator_input = dataclasses.replace(
            self._operator_input, manual_target_id=track.track_id
        )
        self._worker.set_operator_input(self._operator_input)

    def _find_track_at(self, x_norm: float, y_norm: float) -> Track | None:
        if self._latest_snapshot is None:
            return None
        for track in self._latest_snapshot.tracks:
            x1, y1, x2, y2 = track.bbox
            if x1 <= x_norm <= x2 and y1 <= y_norm <= y2:
                return track
        return None
