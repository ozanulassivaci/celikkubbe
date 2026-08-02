"""OperatorInputBuilder: merges keyboard, gamepad, and click/GUI-button
intent into one core.types.OperatorInput.

Three input sources can each independently want fire/arm held (space
bar, gamepad A/RT, the right panel's own ATIŞ button), and none of them
know about the others. Without a single point of aggregation, whichever
source's own PipelineWorker.set_operator_input() call landed last would
silently win, so releasing a gamepad trigger could un-arm a shot the
keyboard's space bar is still physically holding down. Every source
instead reports its own held/not-held state under its own name here;
build() ORs them together, so *any* source still holding counts, and
clearing one source can never affect another's.

Not a QObject: this is a plain, synchronous aggregator with no signals
of its own, called directly from whichever thread learns a source's
state changed (the GUI thread for keyboard/clicks/buttons, the gamepad's
own worker thread for its inputs) -- see GamepadWorker's own docstring
for why that thread calls PipelineWorker.set_operator_input() itself
rather than marshalling through a Qt signal first.
"""

from __future__ import annotations

import threading

from celikkubbe.core.types import OperatorInput


class OperatorInputBuilder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fire_sources: dict[str, bool] = {}
        self._arm_sources: dict[str, bool] = {}
        self._manual_target_id: int | None = None

    def set_fire(self, source: str, held: bool) -> None:
        with self._lock:
            self._fire_sources[source] = held

    def set_arm(self, source: str, held: bool) -> None:
        with self._lock:
            self._arm_sources[source] = held

    def set_manual_target(self, track_id: int | None) -> None:
        with self._lock:
            self._manual_target_id = track_id

    def build(self) -> OperatorInput:
        with self._lock:
            return OperatorInput(
                fire_requested=any(self._fire_sources.values()),
                arm_held=any(self._arm_sources.values()),
                manual_target_id=self._manual_target_id,
            )
