"""MainWindow shell tests. Snapshots are driven the same way as
test_pipeline_worker.py: PipelineWorker.tick() called directly (never
worker.start()), which delivers snapshot_ready synchronously since
sender and receiver share this test's thread -- no event-loop spinning
needed to observe MainWindow react to a tick.
"""

from __future__ import annotations

from celikkubbe.core import config, strings
from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import Mode, ReasonCode
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.ui.main_window import MainWindow
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig

_TICK_DT = 1.0 / 30.0


def _make_source(clock: FakeClock) -> SyntheticSource:
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=1, speed=0.0, range_m=3.0))
    source.start()
    return source


def _make_window(clock: FakeClock) -> tuple[MainWindow, PipelineWorker, SimTurretLink]:
    link = SimTurretLink(clock)
    worker = PipelineWorker(_make_source(clock), link, clock)
    window = MainWindow(worker, source_label="SYNTHETIC")
    return window, worker, link


def _tick(worker: PipelineWorker, clock: FakeClock) -> None:
    clock.advance(_TICK_DT)
    worker._link_worker.tick()  # simulate LinkWorker's own background thread
    worker.tick()


def test_status_strip_reflects_mode_engagement_and_layer(qtbot):
    clock = FakeClock()
    window, worker, link = _make_window(clock)
    qtbot.addWidget(window)

    for _ in range(3):
        _tick(worker, clock)

    snapshot = window._latest_snapshot
    assert snapshot is not None
    strip = window._status_strip
    assert strip._mode_badge._text == snapshot.state.mode.value[:2]
    assert strip._engagement_badge._text == snapshot.state.engagement.value[:2]
    assert strip._layer_badge._text == snapshot.state.active_layer.value

    link.inject_estop()
    for _ in range(3):
        _tick(worker, clock)

    updated = window._latest_snapshot
    assert updated.state.mode is Mode.M4_SAFE
    assert strip._mode_badge._text == "M4"


def test_selftest_overlay_shows_in_m1_and_hides_on_success(qtbot):
    clock = FakeClock()
    window, worker, _link = _make_window(clock)
    qtbot.addWidget(window)

    # LinkWorker starts with link_lost=True until its own tick() has run
    # at least once, so ticking the pipeline without it first guarantees
    # the stm32_link self-test item fails and mode stays in M1_INIT --
    # against a fully cooperative SimTurretLink, self-test can otherwise
    # pass within a single tick, too fast to reliably observe M1_INIT.
    clock.advance(_TICK_DT)
    worker.tick()
    assert window._latest_snapshot.state.mode is Mode.M1_INIT
    assert not window._selftest_overlay.isHidden()

    for _ in range(10):
        _tick(worker, clock)
        if window._latest_snapshot.state.mode is not Mode.M1_INIT:
            break

    assert window._latest_snapshot.state.mode is Mode.M2_STANDBY
    assert window._selftest_overlay.isHidden()


def test_safe_overlay_shows_in_m4_and_requires_acknowledgement(qtbot):
    clock = FakeClock()
    window, worker, link = _make_window(clock)
    qtbot.addWidget(window)

    for _ in range(3):
        _tick(worker, clock)
    assert window._latest_snapshot.state.mode is Mode.M2_STANDBY

    link.inject_estop()
    for _ in range(3):
        _tick(worker, clock)
    assert window._latest_snapshot.state.mode is Mode.M4_SAFE
    assert not window._safe_overlay.isHidden()
    assert window._safe_overlay._fault_label.text() == strings.describe(ReasonCode.ESTOP_ACTIVE)

    # Time passing alone must never dismiss it -- M4 requires a
    # deliberate operator action.
    for _ in range(20):
        _tick(worker, clock)
    assert window._latest_snapshot.state.mode is Mode.M4_SAFE
    assert not window._safe_overlay.isHidden()

    link.release_estop()
    worker.acknowledge_fault()
    _tick(worker, clock)

    assert window._latest_snapshot.state.mode is Mode.M1_INIT
    assert window._safe_overlay.isHidden()


def test_link_loss_visible_in_status_strip_within_stale_threshold(qtbot):
    clock = FakeClock()
    window, worker, link = _make_window(clock)
    qtbot.addWidget(window)

    _tick(worker, clock)
    assert "LINK OK" in window._status_strip._link_label.text()

    link.inject_link_dropout(duration_s=10.0)
    stale_s = config.TELEMETRY_STALE_MS / 1000.0
    clock.advance(stale_s + _TICK_DT)
    worker._link_worker.tick()
    worker.tick()

    assert "LINK LOST" in window._status_strip._link_label.text()
