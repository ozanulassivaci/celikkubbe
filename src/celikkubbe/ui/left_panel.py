"""LeftPanel: AI decision support and target tracking.

Fixed-width column to the left of the video canvas. Everything here is
read-only display except the target cards: a left click selects a track
by emitting ``track_selected`` -- MainWindow turns that into
``OperatorInput.manual_target_id``, the same as a canvas click -- and a
right click opens a context menu to manually assign or clear that
track's class (``class_assigned``/``class_cleared``), unblocking Stage 3
range-rule testing without a real classifier.

Three things matter more than they look, per the prompt this was built
from:

- The target list refreshes at ``config.TRACK_LIST_UPDATE_HZ`` (10 Hz),
  not frame rate -- rebuilding a card per track 30 times a second for a
  read-only list is wasted work, and the operator cannot read numbers
  changing that fast anyway.
- Card order follows ``UiSnapshot.ordered_track_ids``, which already
  carries ``priority.order_track_ids``'s hysteresis -- recomputing an
  order here could silently disagree with it.
- A friendly (permanently excluded) or deferred (temporarily excluded)
  track says so on its own card. Without that, the operator sees a
  target on screen with no explanation for why the system will not
  engage it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QContextMenuEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from celikkubbe.core import config, strings
from celikkubbe.core.strings import IFF_LABEL_TR, TARGET_CLASS_TR, TRACK_STATUS_TR, UI_LABEL_TR
from celikkubbe.core.types import (
    IFF,
    CameraIntrinsics,
    EngagementState,
    ReasonCode,
    TargetClass,
    Track,
    TrackStatus,
)
from celikkubbe.ui import theme
from celikkubbe.ui.snapshot import UiSnapshot
from celikkubbe.ui.theme import RiskBar, StatusBadge
from celikkubbe.ui.video_canvas import class_label
from celikkubbe.vision.l2_color import is_range_estimate_unreliable

# The four classes an operator can actually assign -- BALLOON is a
# leftover from a pre-confirmation draft (see core/config.py's own
# RANGE_RULES comment) and not offered here; UNKNOWN is what clearing an
# assignment already produces, not something to assign.
_MANUAL_CLASS_OPTIONS: tuple[TargetClass, ...] = (
    TargetClass.F16,
    TargetClass.MISSILE,
    TargetClass.UAV,
    TargetClass.HELICOPTER,
)

_THREAT_HIGH = 66.0
_THREAT_MED = 33.0

_IFF_TEXT_COLOR: dict[IFF, str] = {
    IFF.HOSTILE: theme.HOSTILE,
    IFF.FRIENDLY: theme.FRIENDLY,
    IFF.UNKNOWN: theme.UNKNOWN,
}


def threat_band(score: float) -> tuple[str, str]:
    """(label, colour) for a 0-100 risk score, in the same thirds RiskBar
    itself paints its LOW/MED/HIGH gradient zones in.
    """
    if score >= _THREAT_HIGH:
        return UI_LABEL_TR["THREAT_HIGH"], theme.DANGER
    if score >= _THREAT_MED:
        return UI_LABEL_TR["THREAT_MEDIUM"], theme.WARN
    return UI_LABEL_TR["THREAT_LOW"], theme.OK


def build_recommendation(snapshot: UiSnapshot) -> str:
    """The operator-facing "why" behind the current engagement state --
    this is where ReasonCode becomes visible in the operator's own
    language, not just a status code. A pure function of the snapshot,
    so the recommendation text is directly testable without a widget.
    """
    state = snapshot.state
    selected = next((t for t in state.tracks if t.track_id == state.selected_track_id), None)
    if selected is None:
        return UI_LABEL_TR["RECOMMENDATION_SEARCHING"]

    cls = class_label(selected.cls, selected.cls_source)
    if state.engagement in (EngagementState.S5_ENGAGE, EngagementState.S6_ASSESS):
        return UI_LABEL_TR["RECOMMENDATION_ENGAGED"].format(cls=cls)
    if state.engagement is EngagementState.S4_AIM:
        reason = snapshot.engagement_fallback_reason
        if reason is not None:
            return UI_LABEL_TR["RECOMMENDATION_BLOCKED"].format(
                cls=cls, reason=strings.describe(reason)
            )
        return UI_LABEL_TR["RECOMMENDATION_FIRE"].format(cls=cls)
    return UI_LABEL_TR["RECOMMENDATION_TRACKING"].format(cls=cls)


class TrackCard(QFrame):
    """One target: class/status/IFF, then range/confidence/threat, then
    -- when applicable -- why the system will not engage it on its own.
    """

    clicked = pyqtSignal(int)
    class_assign_requested = pyqtSignal(int, object)  # track_id, TargetClass
    class_clear_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setMinimumHeight(90)
        self._track_id: int | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        top_row = QHBoxLayout()
        self._class_label = QLabel("")
        self._class_label.setStyleSheet("font-weight: 600;")
        self._status_badge = StatusBadge("", theme.TEXT_DIM)
        self._status_badge.setFixedWidth(76)
        self._iff_label = QLabel("")
        top_row.addWidget(self._class_label)
        top_row.addStretch(1)
        top_row.addWidget(self._status_badge)
        top_row.addWidget(self._iff_label)
        outer.addLayout(top_row)

        columns_row = QHBoxLayout()
        self._range_label = QLabel("")
        self._confidence_label = QLabel("")
        self._threat_label = QLabel("")
        for label in (self._range_label, self._confidence_label, self._threat_label):
            label.setStyleSheet(f"color: {theme.TEXT_DIM};")
            columns_row.addWidget(label)
        outer.addLayout(columns_row)

        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet(f"color: {theme.WARN}; font-size: 8pt;")
        self._note_label.setVisible(False)
        outer.addWidget(self._note_label)

    def update_track(
        self,
        track: Track,
        selected: bool,
        deferred: tuple[float, ReasonCode] | None,
        now: float,
        intrinsics: CameraIntrinsics,
    ) -> None:
        self._track_id = track.track_id
        self._class_label.setText(class_label(track.cls, track.cls_source))
        self._class_label.setStyleSheet(
            f"font-weight: 600; color: {theme.TEXT_PRIMARY if not selected else theme.OK};"
        )
        self._status_badge.set_status(
            TRACK_STATUS_TR[track.status], self._status_color(track.status)
        )
        self._iff_label.setText(IFF_LABEL_TR[track.iff])
        self._iff_label.setStyleSheet(f"color: {_IFF_TEXT_COLOR[track.iff]}; font-weight: 600;")

        range_str = "--" if track.range_m is None else f"{track.range_m:.1f}m"
        if track.range_m is not None and track.range_source == "size":
            range_str = f"~{range_str}"
        unreliable = track.range_m is not None and is_range_estimate_unreliable(track, intrinsics)
        if unreliable:
            range_str += "?"
        self._range_label.setText(range_str)
        self._range_label.setStyleSheet(f"color: {theme.WARN if unreliable else theme.TEXT_DIM};")
        self._confidence_label.setText(f"{track.confidence:.0%}")
        band_label, band_color = threat_band(track.risk_score)
        self._threat_label.setText(band_label)
        self._threat_label.setStyleSheet(f"color: {band_color};")

        self.setStyleSheet(
            f"QFrame[role='card'] {{ border: 2px solid {theme.OK if selected else theme.BORDER}; }}"
            if selected
            else ""
        )

        if track.iff is IFF.FRIENDLY:
            self._note_label.setText(UI_LABEL_TR["EXCLUDED_FRIENDLY"])
            self._note_label.setVisible(True)
        elif deferred is not None:
            defer_until, reason = deferred
            remaining = max(0.0, defer_until - now)
            self._note_label.setText(
                UI_LABEL_TR["DEFERRED_PREFIX"].format(
                    seconds=remaining, reason=strings.describe(reason)
                )
            )
            self._note_label.setVisible(True)
        else:
            self._note_label.setVisible(False)

    @staticmethod
    def _status_color(status: TrackStatus) -> str:
        if status is TrackStatus.CONFIRMED:
            return theme.OK
        if status is TrackStatus.COASTING:
            return theme.WARN
        return theme.TEXT_DIM

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._track_id is not None:
            self.clicked.emit(self._track_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802 - Qt override
        if self._track_id is None:
            return
        track_id = self._track_id
        menu = QMenu(self)
        for cls in _MANUAL_CLASS_OPTIONS:
            action = menu.addAction(TARGET_CLASS_TR[cls])
            action.triggered.connect(
                lambda _checked=False, c=cls: self.class_assign_requested.emit(track_id, c)
            )
        menu.addSeparator()
        clear_action = menu.addAction(UI_LABEL_TR["UNKNOWN_CLASS"])
        clear_action.triggered.connect(lambda: self.class_clear_requested.emit(track_id))
        menu.exec(event.globalPos())


class LeftPanel(QWidget):
    track_selected = pyqtSignal(int)
    class_assigned = pyqtSignal(int, object)  # track_id, TargetClass
    class_cleared = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "panel")
        self._last_update_t: float | None = None
        self._cards: list[TrackCard] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        outer.addLayout(self._build_header())
        outer.addWidget(self._build_threat_box())
        outer.addWidget(self._build_risk_bar())
        outer.addWidget(self._build_recommendation_box())
        outer.addLayout(self._build_classification_summary())
        outer.addLayout(self._build_target_list_header())
        outer.addWidget(self._build_target_list(), stretch=1)

    # --- construction ---

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel(UI_LABEL_TR["AI_PANEL_TITLE"])
        title.setStyleSheet("font-weight: 600;")
        self._live_badge = StatusBadge(UI_LABEL_TR["LIVE_INDICATOR"], theme.TEXT_DIM)
        self._live_badge.setFixedWidth(64)
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(self._live_badge)
        return row

    def _build_threat_box(self) -> QFrame:
        box = QFrame()
        box.setProperty("role", "card")
        box.setFixedHeight(60)
        layout = QVBoxLayout(box)
        self._threat_label = QLabel(UI_LABEL_TR["THREAT_NONE"])
        self._threat_label.setStyleSheet("font-weight: 700; font-size: 12pt;")
        self._threat_sub_label = QLabel("")
        self._threat_sub_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        layout.addWidget(self._threat_label)
        layout.addWidget(self._threat_sub_label)
        return box

    def _build_risk_bar(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(55)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self._risk_value_label = QLabel("--")
        top.addWidget(self._risk_value_label)
        top.addStretch(1)
        layout.addLayout(top)
        self._risk_bar = RiskBar()
        layout.addWidget(self._risk_bar)
        return container

    def _build_recommendation_box(self) -> QFrame:
        box = QFrame()
        box.setProperty("role", "card")
        box.setFixedHeight(75)
        layout = QVBoxLayout(box)
        self._recommendation_label = QLabel(UI_LABEL_TR["RECOMMENDATION_SEARCHING"])
        self._recommendation_label.setWordWrap(True)
        layout.addWidget(self._recommendation_label)
        return box

    def _build_classification_summary(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self._classification_labels: dict[IFF, QLabel] = {}
        for iff in (IFF.HOSTILE, IFF.FRIENDLY, IFF.UNKNOWN):
            box = QFrame()
            box.setProperty("role", "card")
            box.setFixedHeight(70)
            box_layout = QVBoxLayout(box)
            name_label = QLabel(IFF_LABEL_TR[iff])
            name_label.setStyleSheet(f"color: {_IFF_TEXT_COLOR[iff]}; font-weight: 600;")
            count_label = QLabel("0")
            count_label.setStyleSheet("font-size: 14pt; font-weight: 700;")
            box_layout.addWidget(name_label)
            box_layout.addWidget(count_label)
            self._classification_labels[iff] = count_label
            row.addWidget(box)
        return row

    def _build_target_list_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel(UI_LABEL_TR["TARGET_LIST_TITLE"])
        title.setStyleSheet("font-weight: 600;")
        self._track_count_badge = StatusBadge("0", theme.TEXT_DIM)
        self._track_count_badge.setFixedWidth(40)
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(self._track_count_badge)
        return row

    def _build_target_list(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self._list_layout = QVBoxLayout(container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    # --- updates ---

    def update_from_snapshot(self, snapshot: UiSnapshot) -> None:
        self._live_badge.set_status(UI_LABEL_TR["LIVE_INDICATOR"], theme.OK)

        if (
            self._last_update_t is not None
            and (snapshot.t - self._last_update_t) < 1.0 / config.TRACK_LIST_UPDATE_HZ
        ):
            return
        self._last_update_t = snapshot.t

        self._update_threat_and_risk(snapshot)
        self._recommendation_label.setText(build_recommendation(snapshot))
        self._update_classification_summary(snapshot)
        self._update_target_list(snapshot)

    def _update_threat_and_risk(self, snapshot: UiSnapshot) -> None:
        tracks = snapshot.tracks
        top = max(tracks, key=lambda t: t.risk_score, default=None)
        if top is None:
            self._threat_label.setText(UI_LABEL_TR["THREAT_NONE"])
            self._threat_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 12pt;")
            self._threat_sub_label.setText("")
            self._risk_value_label.setText("--")
            self._risk_bar.set_value(0.0)
            return
        label, color = threat_band(top.risk_score)
        self._threat_label.setText(label)
        self._threat_label.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 12pt;")
        self._threat_sub_label.setText(class_label(top.cls, top.cls_source))
        self._risk_value_label.setText(f"{top.risk_score:.0f}")
        self._risk_bar.set_value(top.risk_score)

    def _update_classification_summary(self, snapshot: UiSnapshot) -> None:
        counts = dict.fromkeys((IFF.HOSTILE, IFF.FRIENDLY, IFF.UNKNOWN), 0)
        for track in snapshot.tracks:
            counts[track.iff] += 1
        for iff, label in self._classification_labels.items():
            label.setText(str(counts[iff]))

    def _update_target_list(self, snapshot: UiSnapshot) -> None:
        self._track_count_badge.set_status(str(len(snapshot.tracks)), theme.TEXT_DIM)

        tracks_by_id = {t.track_id: t for t in snapshot.tracks}
        ordered = [tid for tid in snapshot.ordered_track_ids if tid in tracks_by_id]
        # Any track missing from the (possibly stale-for-one-tick) order
        # -- e.g. brand new this tick -- is still shown, appended at the
        # end, rather than silently dropped from the list.
        ordered += [tid for tid in tracks_by_id if tid not in ordered]

        while len(self._cards) < len(ordered):
            card = TrackCard()
            card.clicked.connect(self.track_selected.emit)
            card.class_assign_requested.connect(self.class_assigned.emit)
            card.class_clear_requested.connect(self.class_cleared.emit)
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)
            self._cards.append(card)
        while len(self._cards) > len(ordered):
            card = self._cards.pop()
            self._list_layout.removeWidget(card)
            card.deleteLater()

        for card, track_id in zip(self._cards, ordered, strict=True):
            track = tracks_by_id[track_id]
            card.update_track(
                track,
                selected=track_id == snapshot.state.selected_track_id,
                deferred=snapshot.state.deferred.get(track_id),
                now=snapshot.t,
                intrinsics=snapshot.frame.intrinsics,
            )
