"""RightPanel: safety controls, stage/layer selection, manual control.

Fixed-width column to the right of the video canvas, split top-to-bottom
into a scrollable configuration region and a pinned-bottom safety region.
This widget only ever emits signals -- it never talks to PipelineWorker
or LinkWorker directly, the same separation LeftPanel already
established (``track_selected``): MainWindow is the one place a ui/
widget's signal turns into a PipelineWorker call.

The pinned-bottom region (ACİL STOP, EMNİYET KİLİDİ, ATIŞ, the safety
warning line) lives outside the QScrollArea deliberately and must stay
that way: an operator scrolled down to look at the manual control pad
must never lose the ability to see or reach the e-stop. Do not move any
of these three controls into the scrollable region.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from celikkubbe.core import config, strings
from celikkubbe.core.strings import IFF_LABEL_TR, LAYER_LABEL_TR, STAGE_LABEL_TR, UI_LABEL_TR
from celikkubbe.core.types import (
    IFF,
    Axis,
    EngagementState,
    Layer,
    Mode,
    ReasonCode,
    Stage,
    SystemState,
)
from celikkubbe.ui import theme
from celikkubbe.ui.left_panel import build_recommendation
from celikkubbe.ui.snapshot import UiSnapshot
from celikkubbe.ui.theme import ToggleSwitch
from celikkubbe.ui.video_canvas import class_label

_LAYERS_BY_STAGE: dict[Stage, tuple[Layer, ...]] = {
    Stage.STAGE_1: (Layer.L1, Layer.L2, Layer.L3),
    Stage.STAGE_2: (Layer.L1, Layer.L2),
    Stage.STAGE_3: (Layer.L1, Layer.L2),
}

_IFF_TEXT_COLOR: dict[IFF, str] = {
    IFF.HOSTILE: theme.HOSTILE,
    IFF.FRIENDLY: theme.FRIENDLY,
    IFF.UNKNOWN: theme.UNKNOWN,
}

_MODE_CARD_HEIGHT_PX = 55


def compute_fire_enabled(snapshot: UiSnapshot) -> tuple[bool, str | None]:
    """Whether pressing ATIŞ right now could do anything, and if not, why
    -- a pure function of the snapshot so it is directly testable and so
    the button's own disabled reason can never silently diverge from
    what actually gates S4_AIM -> S5_ENGAGE in engagement.py.

    Deliberately does not check operator.arm_held/fire_requested: those
    are what *pressing* the (now-enabled) button sets, not a precondition
    for whether pressing it is worth doing.
    """
    state = snapshot.state
    telemetry = snapshot.telemetry
    if state.mode is not Mode.M3_OPERATIONAL:
        return False, strings.describe(ReasonCode.NOT_OPERATIONAL)
    if telemetry is None or not telemetry.armed:
        return False, strings.describe(ReasonCode.NOT_ARMED)
    if state.engagement is EngagementState.S4_AIM and snapshot.engagement_fallback_reason is None:
        return True, None
    return False, build_recommendation(snapshot)


class RightPanel(QWidget):
    estop_requested = pyqtSignal()
    armed_changed = pyqtSignal(bool)
    fire_pressed = pyqtSignal()
    fire_released = pyqtSignal()
    stage_selected = pyqtSignal(object)  # Stage -- see snapshot_ready's own precedent
    layer_selected = pyqtSignal(object)  # Layer
    zero_requested = pyqtSignal(object)  # Axis
    jog_pressed = pyqtSignal(object, int, float)  # Axis, direction, speed_dps
    jog_released = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "panel")
        self._rendered_stage: Stage | None = None
        self._mode_card_buttons: dict[Layer, QPushButton] = {}
        self._jog_active = False
        self._fire_active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_scroll_area(), stretch=1)
        outer.addLayout(self._build_pinned_safety_region())

    # --- construction: scrollable top ---

    def _build_scroll_area(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        layout.addLayout(self._build_stage_selector())
        self._mode_cards_layout = QVBoxLayout()
        self._mode_cards_layout.setSpacing(6)
        layout.addLayout(self._mode_cards_layout)
        layout.addWidget(self._build_config_summary())
        layout.addWidget(self._build_locked_target())
        layout.addWidget(self._build_manual_pad())
        layout.addWidget(self._build_speed_slider())
        layout.addLayout(self._build_homing_row())
        layout.addStretch(1)

        scroll.setWidget(container)
        return scroll

    def _build_stage_selector(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        self._stage_buttons: dict[Stage, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for stage in (Stage.STAGE_1, Stage.STAGE_2, Stage.STAGE_3):
            btn = QPushButton(STAGE_LABEL_TR[stage])
            btn.setCheckable(True)
            btn.setFixedHeight(55)
            btn.clicked.connect(lambda _checked=False, s=stage: self.stage_selected.emit(s))
            group.addButton(btn)
            self._stage_buttons[stage] = btn
            row.addWidget(btn)
        self._stage_button_group = group
        return row

    def _build_config_summary(self) -> QFrame:
        box = QFrame()
        box.setProperty("role", "card")
        box.setFixedHeight(55)
        layout = QVBoxLayout(box)
        self._config_summary_label = QLabel("")
        self._config_summary_label.setWordWrap(True)
        layout.addWidget(self._config_summary_label)
        return box

    def _build_locked_target(self) -> QFrame:
        box = QFrame()
        box.setProperty("role", "card")
        box.setFixedHeight(50)
        layout = QHBoxLayout(box)
        title = QLabel(UI_LABEL_TR["LOCKED_TARGET_TITLE"])
        title.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self._locked_target_label = QLabel(UI_LABEL_TR["THREAT_NONE"])
        self._locked_confidence_label = QLabel("")
        layout.addWidget(title)
        layout.addWidget(self._locked_target_label)
        layout.addStretch(1)
        layout.addWidget(self._locked_confidence_label)
        return box

    def _build_manual_pad(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(110)
        grid = QGridLayout(container)
        grid.setSpacing(4)

        self._jog_buttons: dict[tuple[Axis, int], QPushButton] = {}
        specs = (
            (Axis.TILT, 1, "▲", 0, 1),
            (Axis.PAN, -1, "◀", 1, 0),
            (Axis.PAN, 1, "▶", 1, 2),
            (Axis.TILT, -1, "▼", 2, 1),
        )
        for axis, direction, text, row, col in specs:
            btn = QPushButton(text)
            btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            btn.pressed.connect(lambda a=axis, d=direction: self._on_jog_pressed(a, d))
            btn.released.connect(self._on_jog_released)
            self._jog_buttons[(axis, direction)] = btn
            grid.addWidget(btn, row, col)

        stop_btn = QPushButton(UI_LABEL_TR["STOP_BUTTON"])
        stop_btn.clicked.connect(self._on_jog_released)
        grid.addWidget(stop_btn, 1, 1)
        return container

    def _build_speed_slider(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(40)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(UI_LABEL_TR["SPEED_LABEL"])
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(int(config.JOG_SPEED_MIN_DPS), int(config.JOG_SPEED_MAX_DPS))
        self._speed_slider.setValue(int(config.JOG_SPEED_DEFAULT_DPS))
        self._speed_value_label = QLabel(f"{config.JOG_SPEED_DEFAULT_DPS:.0f}°/s")
        self._speed_value_label.setFixedWidth(48)
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_value_label.setText(f"{v:.0f}°/s")
        )
        layout.addWidget(label)
        layout.addWidget(self._speed_slider, stretch=1)
        layout.addWidget(self._speed_value_label)
        return container

    def _build_homing_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._homing_pan_button = QPushButton(UI_LABEL_TR["ZERO_PAN"])
        self._homing_pan_button.setFixedHeight(40)
        self._homing_pan_button.clicked.connect(lambda: self.zero_requested.emit(Axis.PAN))
        self._homing_tilt_button = QPushButton(UI_LABEL_TR["ZERO_TILT"])
        self._homing_tilt_button.setFixedHeight(40)
        self._homing_tilt_button.clicked.connect(lambda: self.zero_requested.emit(Axis.TILT))
        row.addWidget(self._homing_pan_button)
        row.addWidget(self._homing_tilt_button)
        return row

    # --- construction: pinned-bottom safety region ---
    # Never move ESTOP/EMNİYET KİLİDİ/ATIŞ/warning into the scroll area
    # above -- see the module docstring.

    def _build_pinned_safety_region(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(10, 6, 10, 8)
        column.setSpacing(6)

        self._estop_button = QPushButton(UI_LABEL_TR["ESTOP_BUTTON"])
        self._estop_button.setFixedHeight(52)
        self._estop_button.setProperty("role", "primary")
        self._estop_button.setStyleSheet(
            f"background-color: {theme.DANGER}; border-color: {theme.DANGER}; "
            f"color: {theme.BG_BASE}; font-weight: 700;"
        )
        self._estop_button.clicked.connect(self.estop_requested.emit)
        column.addWidget(self._estop_button)

        lock_row = QHBoxLayout()
        lock_row.setSpacing(8)
        lock_label = QLabel(UI_LABEL_TR["SAFETY_LOCK"])
        self._safety_toggle = ToggleSwitch()
        self._safety_toggle.clicked.connect(self._on_safety_toggle_clicked)
        lock_row.addWidget(lock_label)
        lock_row.addStretch(1)
        lock_row.addWidget(self._safety_toggle)
        lock_container = QWidget()
        lock_container.setFixedHeight(40)
        lock_container.setLayout(lock_row)
        column.addWidget(lock_container)

        self._fire_button = QPushButton(UI_LABEL_TR["FIRE_BUTTON"])
        self._fire_button.setFixedHeight(52)
        self._fire_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._fire_button.pressed.connect(self._on_fire_pressed)
        self._fire_button.released.connect(self._on_fire_released)
        column.addWidget(self._fire_button)

        self._warning_label = QLabel(UI_LABEL_TR["SAFETY_WARNING_LOCKED"])
        self._warning_label.setFixedHeight(25)
        self._warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._warning_label)

        return column

    @property
    def jog_speed_dps(self) -> float:
        """The speed slider's current value -- exposed so the keyboard's
        arrow-key jog uses the exact same operator-chosen speed as this
        panel's own directional pad, rather than a second, independent
        default that could silently disagree with it.
        """
        return float(self._speed_slider.value())

    # --- jog / fire press-release handling ---

    def _on_jog_pressed(self, axis: Axis, direction: int) -> None:
        self._jog_active = True
        self.jog_pressed.emit(axis, direction, float(self._speed_slider.value()))

    def _on_jog_released(self) -> None:
        if not self._jog_active:
            return
        self._jog_active = False
        self.jog_released.emit()

    def _on_fire_pressed(self) -> None:
        self._fire_active = True
        self.fire_pressed.emit()

    def _on_fire_released(self) -> None:
        if not self._fire_active:
            return
        self._fire_active = False
        self.fire_released.emit()

    def force_stop_all(self) -> None:
        """Called by MainWindow on window deactivation (e.g. alt-tab) or
        focus loss, so an operator switching away with a jog or fire
        button visually held down can never leave the turret jogging, or
        the dead-man switch armed, with nobody's finger actually on it.
        """
        self._on_jog_released()
        self._on_fire_released()
        for btn in self._jog_buttons.values():
            btn.setDown(False)
        self._fire_button.setDown(False)

    def _on_safety_toggle_clicked(self) -> None:
        self.armed_changed.emit(self._safety_toggle.isChecked())

    # --- updates ---

    def update_from_snapshot(self, snapshot: UiSnapshot) -> None:
        state = snapshot.state
        if state.stage is not self._rendered_stage:
            self._rebuild_mode_cards(state.stage)
        for stage, btn in self._stage_buttons.items():
            btn.setChecked(stage is state.stage)
            btn.setStyleSheet(
                f"background-color: {theme.accent_for_stage(stage)}; "
                f"border-color: {theme.accent_for_stage(stage)}; color: {theme.BG_BASE};"
                if stage is state.stage
                else ""
            )
        self._update_mode_cards(state)
        self._update_config_summary(state)
        self._update_locked_target(state)

        telemetry = snapshot.telemetry
        armed = telemetry is not None and telemetry.armed
        self._safety_toggle.setChecked(armed)

        fire_enabled, fire_reason = compute_fire_enabled(snapshot)
        self._fire_button.setEnabled(fire_enabled)
        self._fire_button.setToolTip(fire_reason or "")
        self._update_warning_line(armed, fire_enabled, fire_reason)

    def _rebuild_mode_cards(self, stage: Stage) -> None:
        while self._mode_cards_layout.count():
            item = self._mode_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._mode_card_buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for layer in _LAYERS_BY_STAGE[stage]:
            btn = QPushButton(LAYER_LABEL_TR[layer])
            btn.setCheckable(True)
            btn.setFixedHeight(_MODE_CARD_HEIGHT_PX)
            if layer is Layer.L1:
                # No YOLO/L1 detector exists yet -- shown as unavailable,
                # not hidden, so the operator knows a third option exists
                # rather than wondering why only two are offered. See
                # main_window.py's _layer_badge_color for the same
                # "unavailable, not failed" distinction (Part 0j).
                btn.setEnabled(False)
                btn.setToolTip(UI_LABEL_TR["UNAVAILABLE"])
            else:
                btn.clicked.connect(lambda _checked=False, ly=layer: self.layer_selected.emit(ly))
            group.addButton(btn)
            self._mode_card_buttons[layer] = btn
            self._mode_cards_layout.addWidget(btn)
        self._mode_card_group = group
        self._rendered_stage = stage

    def _update_mode_cards(self, state: SystemState) -> None:
        chosen_by = (
            UI_LABEL_TR["OPERATOR_CHOSEN"]
            if state.layer_manual_override
            else UI_LABEL_TR["CASCADE_CHOSEN"]
        )
        for layer, btn in self._mode_card_buttons.items():
            is_active = layer is state.active_layer
            btn.setChecked(is_active)
            if layer is Layer.L1:
                btn.setText(f"{LAYER_LABEL_TR[layer]}\n{UI_LABEL_TR['UNAVAILABLE']}")
                continue
            label = LAYER_LABEL_TR[layer]
            btn.setText(f"{label}\n{chosen_by}" if is_active else label)

    def _update_config_summary(self, state: SystemState) -> None:
        chosen_by = (
            UI_LABEL_TR["OPERATOR_CHOSEN"]
            if state.layer_manual_override
            else UI_LABEL_TR["CASCADE_CHOSEN"]
        )
        text = f"{STAGE_LABEL_TR[state.stage]} · {LAYER_LABEL_TR[state.active_layer]} ({chosen_by})"
        self._config_summary_label.setText(text)

    def _update_locked_target(self, state: SystemState) -> None:
        track = next((t for t in state.tracks if t.track_id == state.selected_track_id), None)
        if track is None:
            self._locked_target_label.setText(UI_LABEL_TR["THREAT_NONE"])
            self._locked_target_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
            self._locked_confidence_label.setText("")
            return
        self._locked_target_label.setText(f"{class_label(track.cls)} — {IFF_LABEL_TR[track.iff]}")
        self._locked_target_label.setStyleSheet(
            f"color: {_IFF_TEXT_COLOR[track.iff]}; font-weight: 600;"
        )
        self._locked_confidence_label.setText(f"{track.confidence:.0%}")

    def _update_warning_line(
        self, armed: bool, fire_enabled: bool, fire_reason: str | None
    ) -> None:
        if not armed:
            self._set_warning_text(UI_LABEL_TR["SAFETY_WARNING_LOCKED"], theme.TEXT_DIM, bold=False)
        elif not fire_enabled and fire_reason is not None:
            self._set_warning_text(fire_reason, theme.WARN, bold=False)
        else:
            self._set_warning_text(UI_LABEL_TR["SAFETY_WARNING_UNLOCKED"], theme.DANGER, bold=True)

    def _set_warning_text(self, text: str, color: str, bold: bool) -> None:
        """QLabel never elides overflowing text on its own: left as-is, a
        reason longer than the panel is wide -- build_recommendation's
        own "{cls} — {reason}" text runs long -- centre-clips illegibly
        from both ends instead of truncating cleanly from one (found by
        looking at a real screenshot, not by any test). The untruncated
        text stays available as a tooltip.
        """
        metrics = self._warning_label.fontMetrics()
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, self._warning_label.width())
        self._warning_label.setText(elided)
        self._warning_label.setToolTip(text)
        weight = "font-weight: 600;" if bold else ""
        self._warning_label.setStyleSheet(f"color: {color}; {weight}")
