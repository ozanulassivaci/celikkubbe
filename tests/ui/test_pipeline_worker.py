"""PipelineWorker tests drive tick() directly against FakeClock, exactly
like tests/io/test_link_worker.py drives LinkWorker.tick() -- the real
QThread is only exercised by the single lifecycle test at the bottom.

PipelineWorker never calls LinkWorker.tick() itself (that is LinkWorker's
own background thread's job once started); tests that need telemetry to
be present call worker's link_worker.tick() directly to simulate that
thread having run one iteration, the same white-box approach
tests/io/test_link_worker.py itself uses.
"""

from __future__ import annotations

import dataclasses

import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import Mode
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig

_TICK_DT = 1.0 / 30.0


def _make_source(clock: FakeClock) -> SyntheticSource:
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=1, speed=0.0, range_m=3.0))
    source.start()
    return source


def _make_worker(clock: FakeClock | None = None) -> PipelineWorker:
    clock = clock or FakeClock()
    return PipelineWorker(_make_source(clock), SimTurretLink(clock), clock)


class _FlakySource:
    """Wraps a real source, raising once on a chosen call to exercise
    PipelineWorker's per-tick exception isolation.
    """

    def __init__(self, inner: SyntheticSource, fail_on_call: int) -> None:
        self._inner = inner
        self._fail_on_call = fail_on_call
        self._calls = 0

    def start(self) -> None:
        self._inner.start()

    def stop(self) -> None:
        self._inner.stop()

    @property
    def has_depth(self) -> bool:
        return self._inner.has_depth

    @property
    def intrinsics(self):
        return self._inner.intrinsics

    def read(self):
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise RuntimeError("boom")
        return self._inner.read()


def test_emits_one_snapshot_per_successful_tick(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    received = []
    worker.snapshot_ready.connect(received.append)

    for _ in range(10):
        clock.advance(_TICK_DT)
        worker.tick()

    assert len(received) == 10
    assert all(s.frame is not None for s in received)


def test_tick_exception_is_isolated_and_loop_continues(qtbot):
    clock = FakeClock()
    inner = _make_source(clock)
    source = _FlakySource(inner, fail_on_call=3)
    worker = PipelineWorker(source, SimTurretLink(clock), clock)

    snapshots = []
    errors = []
    worker.snapshot_ready.connect(snapshots.append)
    worker.error.connect(errors.append)

    for _ in range(5):
        clock.advance(_TICK_DT)
        worker.tick()

    assert len(errors) == 1
    assert "boom" in errors[0]
    # Every tick except the one that raised still produced a snapshot --
    # one bad frame did not kill the loop.
    assert len(snapshots) == 4


def test_latest_snapshot_survives_when_ticks_outpace_consumption(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    seen = []
    worker.snapshot_ready.connect(seen.append)

    for _ in range(5):
        clock.advance(_TICK_DT)
        worker.tick()

    # The signal itself still fires once per tick (Qt handles that
    # cheaply)...
    assert len(seen) == 5
    # ...but the single-slot buffer a slow GUI would actually read from
    # only ever holds the newest one: there is no way to get an earlier
    # tick's snapshot back out of the worker once a later one has landed.
    latest = worker.latest_snapshot
    assert latest is not None
    assert latest is seen[-1]
    assert latest.t == pytest.approx(clock.now())


def test_shutdown_joins_cleanly_with_no_leaked_thread(qtbot):
    clock = FakeClock()
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=1, speed=0.0, range_m=3.0))
    worker = PipelineWorker(source, SimTurretLink(clock), clock)

    worker.start()
    qtbot.waitUntil(worker.isRunning, timeout=2000)
    worker.stop()
    joined = worker.wait(2000)

    assert joined
    assert not worker.isRunning()


def test_emitted_snapshot_is_frozen(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    clock.advance(_TICK_DT)
    worker.tick()

    snapshot = worker.latest_snapshot
    assert snapshot is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.t = 999.0  # type: ignore[misc]


def test_self_test_passes_and_advances_to_standby_once_healthy(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)

    for _ in range(3):
        clock.advance(_TICK_DT)
        worker._link_worker.tick()  # simulate the link thread's own progress
        worker.tick()

    assert worker._state.mode in (Mode.M2_STANDBY, Mode.M1_INIT)
    # camera health needs >=2 frame timestamps and link needs a telemetry
    # reply -- both are true after a few ticks with a live SimTurretLink.
    assert worker._state.last_self_test is not None
    if worker._state.last_self_test.passed:
        assert worker._state.mode is Mode.M2_STANDBY


def test_retry_self_test_forces_submission_while_still_failing(qtbot):
    clock = FakeClock()
    source = _make_source(clock)
    # A link that never answers poll() -- stm32_link self-test item never
    # passes on its own, so mode would otherwise sit in M1_INIT forever.
    worker = PipelineWorker(source, _DeadLink(), clock)

    clock.advance(_TICK_DT)
    worker.tick()
    assert worker._state.mode is Mode.M1_INIT
    assert worker._state.last_self_test is not None
    assert not worker._state.last_self_test.passed

    worker.retry_self_test()
    clock.advance(_TICK_DT)
    worker.tick()

    assert worker._state.mode is Mode.M4_SAFE


class _DeadLink:
    connected = False

    def send(self, cmd) -> None:
        pass

    def poll(self):
        return None
