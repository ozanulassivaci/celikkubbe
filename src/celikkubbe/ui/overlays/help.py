"""HelpOverlay: the F1 keybinding reference.

Static content -- unlike SelfTestOverlay/SafeOverlay, nothing here comes
from a UiSnapshot. MainWindow toggles visibility directly on F1; there is
no PipelineWorker signal driving it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from celikkubbe.core.strings import HELP_CONTROLS_TR, UI_LABEL_TR
from celikkubbe.ui import theme

_OVERLAY_BG = QColor(0, 0, 0, 190)


class HelpOverlay(QWidget):
    close_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 60, 60, 60)

        title = QLabel(UI_LABEL_TR["HELP_TITLE"])
        title.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: 16pt; font-weight: 600;")
        outer.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)

        rows = QVBoxLayout()
        for key_label, description in HELP_CONTROLS_TR:
            row = QHBoxLayout()
            key_widget = QLabel(key_label)
            key_widget.setMinimumWidth(120)
            key_widget.setStyleSheet(f"color: {theme.OK}; font-weight: 600;")
            description_widget = QLabel(description)
            description_widget.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")
            row.addWidget(key_widget)
            row.addWidget(description_widget, stretch=1)
            rows.addLayout(row)
        outer.addLayout(rows)

        close_button = QPushButton(UI_LABEL_TR["CLOSE_BUTTON"])
        close_button.setProperty("role", "primary")
        close_button.clicked.connect(self.close_requested.emit)
        outer.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), _OVERLAY_BG)
        painter.end()
        super().paintEvent(event)
