"""SafeOverlay: the M4 SAFE fault screen.

No automatic dismissal: M4 requires a deliberate operator action, and
this overlay only ever leaves the screen through acknowledge_requested,
never on its own -- there is no timer, no auto-hide, nothing but the
button.

Layout is compact by design: fault description and homing warning first
(the two things the operator needs to read), a height-capped event log
below (context, not the main event), and the acknowledge button
centred under it -- not a full-height log with everything else
squeezed above it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from celikkubbe.core import strings
from celikkubbe.core.strings import UI_LABEL_TR
from celikkubbe.core.types import Axis, ReasonCode
from celikkubbe.io.codec import EventId
from celikkubbe.ui import theme

_OVERLAY_BG = QColor(0, 0, 0, 190)
_MAX_LOG_LINES = 20
_LOG_MAX_HEIGHT_PX = 200


class SafeOverlay(QWidget):
    acknowledge_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 40, 60, 40)

        title = QLabel(UI_LABEL_TR["SAFE_TITLE"])
        title.setStyleSheet(f"color: {theme.DANGER}; font-size: 16pt; font-weight: 600;")
        outer.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._fault_label = QLabel("")
        self._fault_label.setWordWrap(True)
        self._fault_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")
        outer.addWidget(self._fault_label)

        self._homing_warning = QLabel(UI_LABEL_TR["HOMING_WARNING"])
        self._homing_warning.setStyleSheet(f"color: {theme.WARN}; font-weight: 600;")
        self._homing_warning.setVisible(False)
        outer.addWidget(self._homing_warning)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(_MAX_LOG_LINES)
        self._log.setMaximumHeight(_LOG_MAX_HEIGHT_PX)
        # QPlainTextEdit's palette defaults to a white background
        # regardless of the app stylesheet's own colours -- without this
        # it renders as a jarring white box in an otherwise dark shell.
        self._log.setStyleSheet(
            f"background-color: {theme.BG_ELEVATED}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER};"
        )
        outer.addWidget(self._log)

        ack_button = QPushButton(UI_LABEL_TR["ACKNOWLEDGE"])
        ack_button.setProperty("role", "primary")
        ack_button.clicked.connect(self.acknowledge_requested.emit)
        outer.addWidget(ack_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

    def set_fault(self, reason: ReasonCode | None, self_test_detail: str | None = None) -> None:
        """``self_test_detail`` takes priority: a self-test failure is the
        one path into M4 with no telemetry-derived ReasonCode of its own
        (see PipelineWorker._derive_fault_reason), so the caller passes
        the failing item's own detail text instead.
        """
        if self_test_detail is not None:
            self._fault_label.setText(
                UI_LABEL_TR["SELF_TEST_PREFIX"].format(detail=self_test_detail)
            )
        elif reason is not None:
            self._fault_label.setText(strings.describe(reason))
        else:
            self._fault_label.setText("")

    def set_position_valid(self, valid: bool) -> None:
        self._homing_warning.setVisible(not valid)

    def set_event_log(self, events: tuple[tuple[float, EventId, Axis | None], ...]) -> None:
        """Rewritten in full from the buffered log every call rather than
        appending incrementally: the source buffer is a bounded sliding
        window (PipelineWorker keeps only the most recent entries), so an
        index-based "what's new since last time" would misbehave the
        moment old entries fall off the front. At <= 20 short lines this
        costs nothing to redo from scratch.
        """
        self._log.clear()
        for t, event_id, axis in events:
            axis_str = f" {axis.value}" if axis is not None else ""
            self._log.appendPlainText(f"[{t:8.2f}] {event_id.name}{axis_str}")

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), _OVERLAY_BG)
        painter.end()
        super().paintEvent(event)
