from __future__ import annotations

import threading
import time
from collections import deque

from celikkubbe.core.clock import FakeClock, SystemClock
from celikkubbe.core.commands import Arm
from celikkubbe.core.types import Telemetry
from celikkubbe.io.codec import AckResult
from celikkubbe.io.link_worker import LinkWorker

from ..factories import make_telemetry


class _FakeLink:
    """Minimal TurretLink-plus-extras double giving full control over
    telemetry/ack/event timing, so link_worker's own logic can be tested
    without SimTurretLink's physics/watchdog also being in play.
    """

    def __init__(self) -> None:
        self.connected = True
        self.sent_commands: list = []
        self.heartbeat_count = 0
        self.tracked_sends: list[tuple[int, object]] = []
        self.closed = False
        self.crc_error_count = 0
        self.dropped_byte_count = 0
        self._next_seq = 0
        self._telemetry: deque[Telemetry] = deque()
        self._acks: deque[tuple[int, AckResult]] = deque()
        self._events: deque[object] = deque()

    def send(self, cmd) -> None:
        self.sent_commands.append(cmd)

    def send_heartbeat(self) -> None:
        self.heartbeat_count += 1

    def send_tracked(self, cmd, seq: int | None = None) -> int:
        use_seq = self._next_seq if seq is None else seq
        if seq is None:
            self._next_seq += 1
        self.tracked_sends.append((use_seq, cmd))
        return use_seq

    def poll(self) -> Telemetry | None:
        return self._telemetry.popleft() if self._telemetry else None

    def drain_acks(self) -> list[tuple[int, AckResult]]:
        acks, self._acks = list(self._acks), deque()
        return acks

    def drain_events(self) -> list[object]:
        events, self._events = list(self._events), deque()
        return events

    def close(self) -> None:
        self.closed = True

    # test helpers
    def queue_telemetry(self, t: Telemetry) -> None:
        self._telemetry.append(t)

    def queue_ack(self, seq: int, result: AckResult) -> None:
        self._acks.append((seq, result))

    def queue_event(self, event: object) -> None:
        self._events.append(event)


def _tick_for(worker: LinkWorker, clock: FakeClock, seconds: float, step: float = 0.01) -> None:
    for _ in range(round(seconds / step)):
        clock.advance(step)
        worker.tick()


def test_heartbeats_emitted_at_20hz() -> None:
    clock = FakeClock()
    link = _FakeLink()
    worker = LinkWorker(link, clock)

    _tick_for(worker, clock, 1.0, step=0.01)

    assert link.heartbeat_count == 20


def test_link_declared_lost_at_300ms_not_200ms() -> None:
    clock = FakeClock()
    link = _FakeLink()
    worker = LinkWorker(link, clock)
    link.queue_telemetry(make_telemetry(t=0.0))
    worker.tick()  # consumes the one telemetry frame, now = 0.01... well now = clock.now()
    assert worker.link_lost is False

    clock.advance(0.20)
    worker.tick()
    assert worker.link_lost is False  # 200ms alone must not be enough

    clock.advance(0.11)  # total ~310ms since the last telemetry
    worker.tick()
    assert worker.link_lost is True


def test_telemetry_callback_fires_with_latest_snapshot() -> None:
    clock = FakeClock()
    link = _FakeLink()
    received: list[Telemetry] = []
    worker = LinkWorker(link, clock, on_telemetry=received.append)

    telem = make_telemetry(t=0.0, pan_deg=12.5)
    link.queue_telemetry(telem)
    worker.tick()

    assert received == [telem]
    assert worker.latest_telemetry is telem


def test_unacked_command_retransmitted_once_then_reported_failed() -> None:
    clock = FakeClock()
    link = _FakeLink()
    failed: list = []
    worker = LinkWorker(link, clock, on_command_failed=failed.append, ack_timeout_ms=100.0)

    cmd = Arm()
    worker.send(cmd)
    worker.tick()  # drains the outbox, sends seq 0
    assert link.tracked_sends == [(0, cmd)]

    clock.advance(0.11)  # past the 100ms ack timeout, no ack queued
    worker.tick()
    assert worker.stats.retransmissions == 1
    assert link.tracked_sends == [(0, cmd), (0, cmd)]  # same seq reused
    assert failed == []

    clock.advance(0.11)  # still no ack after the retransmit
    worker.tick()
    assert worker.stats.commands_failed == 1
    assert failed == [cmd]


def test_acked_command_is_not_retransmitted() -> None:
    clock = FakeClock()
    link = _FakeLink()
    rejected: list = []
    worker = LinkWorker(link, clock, on_command_rejected=rejected.append, ack_timeout_ms=100.0)

    worker.send(Arm())
    worker.tick()
    link.queue_ack(0, AckResult.OK)
    worker.tick()

    clock.advance(0.2)
    worker.tick()

    assert worker.stats.retransmissions == 0
    assert rejected == []
    assert worker.stats.round_trip_latency_ms is not None


def test_rejected_ack_invokes_callback() -> None:
    clock = FakeClock()
    link = _FakeLink()
    rejected: list = []
    worker = LinkWorker(
        link, clock, on_command_rejected=lambda cmd, result: rejected.append((cmd, result))
    )

    worker.send(Arm())
    worker.tick()
    link.queue_ack(0, AckResult.NOT_ARMED)
    worker.tick()

    assert rejected == [(Arm(), AckResult.NOT_ARMED)]


def test_event_callback_fires() -> None:
    clock = FakeClock()
    link = _FakeLink()
    events: list = []
    worker = LinkWorker(link, clock, on_event=events.append)

    link.queue_event(("estop", None, 0.0))
    worker.tick()

    assert events == [("estop", None, 0.0)]


def test_link_diagnostics_are_refreshed_from_the_link() -> None:
    clock = FakeClock()
    link = _FakeLink()
    worker = LinkWorker(link, clock)

    link.crc_error_count = 3
    link.dropped_byte_count = 7
    worker.tick()

    assert worker.stats.crc_error_count == 3
    assert worker.stats.frames_dropped == 7


def test_shutdown_joins_cleanly_with_no_leaked_threads() -> None:
    link = _FakeLink()
    worker = LinkWorker(link, SystemClock())

    before = threading.active_count()
    worker.start()
    time.sleep(0.05)
    thread = worker._thread
    assert thread is not None and thread.is_alive()

    worker.stop()

    assert not thread.is_alive()
    assert threading.active_count() == before
    assert link.closed is True


class _BareLink:
    """A TurretLink providing only the formal Protocol -- no send_tracked,
    drain_acks, drain_events, or diagnostic counters. LinkWorker must
    degrade to fire-and-forget sends rather than break against one.
    """

    def __init__(self) -> None:
        self.connected = True
        self.sent: list = []

    def send(self, cmd) -> None:
        self.sent.append(cmd)

    def poll(self) -> Telemetry | None:
        return None


def test_worker_degrades_gracefully_against_a_bare_turret_link() -> None:
    clock = FakeClock()
    link = _BareLink()
    worker = LinkWorker(link, clock)

    worker.send(Arm())
    worker.tick()

    assert link.sent == [Arm()]
    assert worker.stats.retransmissions == 0
    assert worker.stats.crc_error_count == 0


def test_unpaired_ack_is_ignored() -> None:
    clock = FakeClock()
    link = _FakeLink()
    worker = LinkWorker(link, clock)
    link.queue_ack(999, AckResult.OK)  # no command was ever sent with this seq
    worker.tick()
    assert worker.stats.round_trip_latency_ms is None


def test_start_is_idempotent() -> None:
    link = _FakeLink()
    worker = LinkWorker(link, SystemClock())
    worker.start()
    first_thread = worker._thread
    worker.start()  # already running -- must not spawn a second thread
    assert worker._thread is first_thread
    worker.stop()
