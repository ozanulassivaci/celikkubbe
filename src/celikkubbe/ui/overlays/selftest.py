"""SelfTestOverlay: the M1 self-test screen.

One row per SelfTestItem with PASS/FAIL, its measured value and detail,
plus a retry button that asks PipelineWorker to conclude the current
attempt right now rather than waiting for every item to pass on its own
(see PipelineWorker._advance_self_test's docstring for why a still-failing
retry is allowed to trip M4_SAFE while an ordinary in-progress check is
not). This screen opens the capability video, so every item shown here
is real -- see PipelineWorker's six self-test checks, not a placeholder
pair.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from celikkubbe.core.strings import SELF_TEST_ITEM_LABEL_TR, UI_LABEL_TR
from celikkubbe.core.types import SelfTestItem, SelfTestResult
from celikkubbe.ui import theme

_OVERLAY_BG = QColor(0, 0, 0, 190)


class SelfTestOverlay(QWidget):
    retry_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_items: tuple[SelfTestItem, ...] = ()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 60, 60, 60)

        title = QLabel(UI_LABEL_TR["SELF_TEST_TITLE"])
        title.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: 16pt; font-weight: 600;")
        outer.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._rows_layout = QVBoxLayout()
        outer.addLayout(self._rows_layout)

        retry_button = QPushButton(UI_LABEL_TR["RETRY"])
        retry_button.setProperty("role", "primary")
        retry_button.clicked.connect(self.retry_requested.emit)
        outer.addWidget(retry_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

    def set_result(self, result: SelfTestResult | None) -> None:
        self.current_items = result.items if result is not None else ()
        self._clear_rows()
        for item in self.current_items:
            self._add_row(item)

    def _clear_rows(self) -> None:
        while self._rows_layout.count():
            entry = self._rows_layout.takeAt(0)
            row_layout = entry.layout()
            if row_layout is None:
                continue
            while row_layout.count():
                widget = row_layout.takeAt(0).widget()
                if widget is not None:
                    widget.deleteLater()
            row_layout.deleteLater()

    def _add_row(self, item: SelfTestItem) -> None:
        row = QHBoxLayout()

        name_label = QLabel(SELF_TEST_ITEM_LABEL_TR.get(item.name, item.name))
        name_label.setMinimumWidth(160)

        status_label = QLabel(UI_LABEL_TR["PASS"] if item.passed else UI_LABEL_TR["FAIL"])
        status_label.setMinimumWidth(80)
        status_label.setStyleSheet(
            f"color: {theme.OK if item.passed else theme.DANGER}; font-weight: 600;"
        )

        measured = f"{item.measured:.1f}" if item.measured is not None else "--"
        detail_label = QLabel(f"{measured}  {item.detail or ''}".strip())
        detail_label.setStyleSheet(f"color: {theme.TEXT_DIM};")

        row.addWidget(name_label)
        row.addWidget(status_label)
        row.addWidget(detail_label, stretch=1)
        self._rows_layout.addLayout(row)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), _OVERLAY_BG)
        painter.end()
        super().paintEvent(event)
