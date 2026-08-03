"""Colour tokens, QSS generation, font loading and the handful of widgets
QSS cannot express on its own.

Nothing outside this module may hardcode a colour literal for a themed
widget -- exactly the "no magic numbers" rule core/config.py enforces for
decision thresholds, applied here to appearance instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, pyqtProperty
from PyQt6.QtGui import QColor, QFontDatabase, QLinearGradient, QPainter, QPaintEvent
from PyQt6.QtWidgets import QAbstractButton, QWidget

from celikkubbe.core.types import Stage

logger = logging.getLogger(__name__)

# --- colour tokens ---

BG_BASE = "#0A0E1A"  # window background
BG_PANEL = "#0F1522"  # panel fill
BG_ELEVATED = "#141C2E"  # cards inside panels
BORDER = "#1E2A42"
BORDER_BRIGHT = "#2D3E5F"

TEXT_PRIMARY = "#E2E8F0"
TEXT_DIM = "#64748B"
TEXT_MUTED = "#475569"

HOSTILE = "#F50A0A"  # matches the physical target model colour
FRIENDLY = "#00A3E0"  # matches the physical target model colour
UNKNOWN = "#EAB308"

OK = "#10B981"
WARN = "#F59E0B"
DANGER = "#EF4444"

ACCENT_A1 = "#F97316"  # stage 1, orange
ACCENT_A2 = "#F59E0B"  # stage 2, amber
ACCENT_A3 = "#22D3EE"  # stage 3, cyan

# Full-window overlay backdrop (self-test, SAFE, help) -- was 190/255,
# which mathematically dims correctly (confirmed by direct pixel
# sampling) but leaves enough of the panels' own bright accent colours
# and badges legible at a glance that the overlay reads as "not really
# covering" them, especially next to the self-test table's own dense
# rows -- exactly the "panel content shows through" complaint. Every
# overlay must use this one constant, not its own literal, so a future
# contrast fix only has to change it here.
OVERLAY_BACKDROP = QColor(0, 0, 0, 235)

STAGE_ACCENT: dict[Stage, str] = {
    Stage.STAGE_1: ACCENT_A1,
    Stage.STAGE_2: ACCENT_A2,
    Stage.STAGE_3: ACCENT_A3,
}


def accent_for_stage(stage: Stage) -> str:
    return STAGE_ACCENT[stage]


# --- fonts ---

# In priority order: the two bundled candidates the mockup was built
# against, then whatever generic monospace family the platform provides.
# Checked against QFontDatabase.families() only as a fallback, after a
# bundled file has already failed to load -- see load_monospace_font.
_PREFERRED_SYSTEM_FALLBACKS = ("JetBrains Mono", "IBM Plex Mono", "DejaVu Sans Mono", "Consolas")
_GENERIC_FALLBACK = "monospace"


def _default_fonts_dir() -> Path:
    # src/celikkubbe/ui/theme.py -> repo root is 4 levels up.
    return Path(__file__).resolve().parents[3] / "assets" / "fonts"


def load_monospace_font(fonts_dir: Path | None = None) -> str:
    """Loads the bundled monospace font and returns its family name.

    Never relies on a system font: the competition laptop is not
    guaranteed to have JetBrains Mono or IBM Plex Mono installed, and the
    layout must be identical regardless of what the OS ships. Falls back
    to whatever generic monospace family is actually available, with a
    logged warning, if no bundled font file is found in ``fonts_dir`` or
    every candidate fails to load -- see assets/fonts/README.md for the
    TODO(asset) tracking the real bundled file this falls back from.
    """
    directory = fonts_dir if fonts_dir is not None else _default_fonts_dir()
    candidates: list[Path] = []
    if directory.is_dir():
        candidates = sorted(directory.glob("*.ttf")) + sorted(directory.glob("*.otf"))

    for font_file in candidates:
        font_id = QFontDatabase.addApplicationFont(str(font_file))
        if font_id == -1:
            logger.warning("failed to load bundled font file: %s", font_file)
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[0]
        logger.warning("bundled font file loaded but declared no family: %s", font_file)

    installed = set(QFontDatabase.families())
    for name in _PREFERRED_SYSTEM_FALLBACKS:
        if name in installed:
            logger.warning(
                "no bundled monospace font found in %s -- using installed "
                "system font %r instead of the mockup's intended family",
                directory,
                name,
            )
            return name

    logger.warning(
        "no bundled monospace font found in %s and none of %s are installed "
        "-- falling back to the generic 'monospace' family; layout may not "
        "match the mockup exactly",
        directory,
        _PREFERRED_SYSTEM_FALLBACKS,
    )
    return _GENERIC_FALLBACK


# --- QSS ---

_QSS_TEMPLATE = """
QMainWindow, QDialog {{
    background-color: {bg_base};
}}

QWidget {{
    color: {text_primary};
    font-family: "{font_family}";
}}

#panel, QFrame[role="panel"] {{
    background-color: {bg_panel};
    border: 1px solid {border};
}}

#card, QFrame[role="card"] {{
    background-color: {bg_elevated};
    border: 1px solid {border};
    border-radius: 4px;
}}

QGroupBox {{
    background-color: {bg_elevated};
    border: 1px solid {border};
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {text_dim};
}}

QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: transparent;
    border: none;
}}

QLabel {{
    color: {text_primary};
}}

QLabel[role="dim"] {{
    color: {text_dim};
}}

QLabel[role="muted"] {{
    color: {text_muted};
}}

QSplitter::handle {{
    background-color: {border};
}}

QSplitter::handle:hover {{
    background-color: {border_bright};
}}

QPushButton {{
    background-color: {bg_elevated};
    border: 1px solid {border_bright};
    border-radius: 3px;
    padding: 4px 12px;
    color: {text_primary};
}}

QPushButton:hover {{
    border-color: {accent};
}}

QPushButton:pressed {{
    background-color: {border};
}}

QPushButton:disabled {{
    color: {text_muted};
    border-color: {border};
}}

QPushButton[role="primary"] {{
    background-color: {accent};
    border-color: {accent};
    color: {bg_base};
    font-weight: 600;
}}

QComboBox, QLineEdit {{
    background-color: {bg_elevated};
    border: 1px solid {border_bright};
    border-radius: 3px;
    padding: 3px 8px;
    color: {text_primary};
}}

QComboBox:hover, QLineEdit:focus {{
    border-color: {accent};
}}

QComboBox QAbstractItemView {{
    background-color: {bg_elevated};
    border: 1px solid {border_bright};
    color: {text_primary};
    selection-background-color: {accent};
}}

QCheckBox {{
    color: {text_primary};
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {border_bright};
    border-radius: 2px;
    background-color: {bg_elevated};
}}

QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}

#statusStrip {{
    background-color: {bg_elevated};
    border-top: 1px solid {border};
}}

QScrollBar:vertical {{
    background: {bg_panel};
    width: 10px;
}}

QScrollBar::handle:vertical {{
    background: {border_bright};
    min-height: 20px;
    border-radius: 4px;
}}
"""


def build_qss(accent: str, font_family: str = _GENERIC_FALLBACK) -> str:
    """Renders the full stylesheet for one stage accent colour.

    A template string re-rendered and reapplied with ``setStyleSheet`` on
    the root widget on every stage change, rather than a fixed sheet plus
    per-widget accent overrides -- stage changes happen a handful of
    times per session at most, so the cost of rebuilding the whole sheet
    is irrelevant, and this keeps exactly one place that knows what
    "accented" means.
    """
    return _QSS_TEMPLATE.format(
        bg_base=BG_BASE,
        bg_panel=BG_PANEL,
        bg_elevated=BG_ELEVATED,
        border=BORDER,
        border_bright=BORDER_BRIGHT,
        text_primary=TEXT_PRIMARY,
        text_dim=TEXT_DIM,
        text_muted=TEXT_MUTED,
        accent=accent,
        font_family=font_family,
    )


# --- custom-paint widgets: QSS cannot express these ---


class StatusBadge(QWidget):
    """A small pill, filled with a state colour, holding short text --
    the M/S/L mode-badges in the status strip and similar. Built as
    direct paint rather than a QSS-styled QLabel so the fill colour can
    change per state without a QSS dynamic-property selector for every
    state this ever needs to represent.
    """

    def __init__(
        self, text: str = "", color: str = TEXT_DIM, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._color = QColor(color)
        self.setMinimumHeight(20)
        self.setMinimumWidth(48)

    def set_status(self, text: str, color: str) -> None:
        self._text = text
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawRoundedRect(rect, radius, radius)

        painter.setPen(QColor(BG_BASE))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._text)
        painter.end()


class ToggleSwitch(QAbstractButton):
    """A checkable pill switch -- Qt has no native equivalent. The knob's
    horizontal position is a ``pyqtProperty`` so ``QPropertyAnimation``
    can animate it smoothly between the off/on ends on every toggle.
    """

    def __init__(
        self,
        on_color: str = OK,
        off_color: str = BORDER_BRIGHT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(44, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_color = QColor(on_color)
        self._off_color = QColor(off_color)
        self._position = 0.0
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        # A hand-rolled animation would need its own QTimer bookkeeping;
        # QPropertyAnimation over the `position` property below already
        # does that, driven by Qt's own event loop.
        anim = QPropertyAnimation(self, b"position", self)
        anim.setDuration(120)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        anim.setStartValue(self._position)
        anim.setEndValue(1.0 if checked else 0.0)
        anim.start()
        self._anim = anim  # keep a reference so it is not garbage-collected mid-flight

    def get_position(self) -> float:
        return self._position

    def set_position(self, value: float) -> None:
        self._position = value
        self.update()

    position = pyqtProperty(float, get_position, set_position)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        track_color = _lerp_color(self._off_color, self._on_color, self._position)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)

        knob_diameter = rect.height() - 4.0
        travel = rect.width() - knob_diameter - 4.0
        knob_x = rect.left() + 2.0 + travel * self._position
        knob_rect = QRectF(knob_x, rect.top() + 2.0, knob_diameter, knob_diameter)
        painter.setBrush(QColor(TEXT_PRIMARY))
        painter.drawEllipse(knob_rect)
        painter.end()


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    t = min(1.0, max(0.0, t))
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


class RiskBar(QWidget):
    """Horizontal 0-100 risk score: a fixed green-amber-red gradient
    track (the zones are score bands, not per-target-relative, so the
    gradient itself never needs to change) with a marker at the current
    value and small zone labels underneath.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self.setMinimumHeight(28)
        self.setMinimumWidth(80)

    def set_value(self, value: float) -> None:
        self._value = min(100.0, max(0.0, value))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        track_h = max(6.0, h * 0.4)
        track = QRectF(0.0, 0.0, w, track_h)

        gradient = QLinearGradient(track.left(), 0.0, track.right(), 0.0)
        gradient.setColorAt(0.0, QColor(OK))
        gradient.setColorAt(0.5, QColor(WARN))
        gradient.setColorAt(1.0, QColor(DANGER))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(track, track_h / 2.0, track_h / 2.0)

        marker_x = (self._value / 100.0) * w
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.setBrush(QColor(TEXT_PRIMARY))
        painter.drawEllipse(QRectF(marker_x - 3.0, track_h / 2.0 - 3.0, 6.0, 6.0))

        painter.setPen(QColor(TEXT_MUTED))
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.75))
        painter.setFont(font)
        label_y = track_h + 2.0
        painter.drawText(
            QRectF(0, label_y, w * 0.34, h - label_y), Qt.AlignmentFlag.AlignLeft, "LOW"
        )
        painter.drawText(
            QRectF(w * 0.33, label_y, w * 0.34, h - label_y), Qt.AlignmentFlag.AlignCenter, "MED"
        )
        painter.drawText(
            QRectF(w * 0.66, label_y, w * 0.34, h - label_y), Qt.AlignmentFlag.AlignRight, "HIGH"
        )
        painter.end()


class AngleGauge(QWidget):
    """A thin bar showing one axis's current angle against its software
    limits (PAN_LIMIT_DEG / TILT_LIMIT_DEG), with the margin nearest each
    limit painted in the danger colour so the operator sees the turret
    approaching a limit before it is actually hit.
    """

    def __init__(
        self,
        min_deg: float,
        max_deg: float,
        warn_margin_deg: float = 10.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min_deg = min_deg
        self._max_deg = max_deg
        self._warn_margin_deg = warn_margin_deg
        self._value: float | None = None
        self.setMinimumHeight(14)
        self.setMinimumWidth(80)

    def set_value(self, value_deg: float | None) -> None:
        self._value = value_deg
        self.update()

    def _fraction(self, value_deg: float) -> float:
        span = self._max_deg - self._min_deg
        if span <= 0.0:
            return 0.0
        return min(1.0, max(0.0, (value_deg - self._min_deg) / span))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        track = QRectF(0.0, 0.0, w, h)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(BG_ELEVATED))
        painter.drawRoundedRect(track, h / 2.0, h / 2.0)

        warn_fraction = self._warn_margin_deg / max(1e-6, self._max_deg - self._min_deg)
        painter.setBrush(QColor(DANGER))
        painter.drawRect(QRectF(0.0, 0.0, w * warn_fraction, h))
        painter.drawRect(QRectF(w * (1.0 - warn_fraction), 0.0, w * warn_fraction, h))

        if self._value is not None:
            fraction = self._fraction(self._value)
            marker_x = fraction * w
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(TEXT_PRIMARY))
            painter.drawRect(QRectF(marker_x - 1.5, 0.0, 3.0, h))
        painter.end()
