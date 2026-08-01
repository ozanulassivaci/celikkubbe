"""LinkWorker: the thread that owns a TurretLink.

No other layer touches serial I/O directly. All the actual work lives in
``tick()``, driven entirely by the injected ``Clock`` -- tests call it
directly in a loop with ``FakeClock.advance()`` and never touch the real
thread at all. ``start()``/``stop()`` wrap that same ``tick()`` in an
actual background thread for production use; the only real-wall-clock
``time.sleep`` in this module is the thread loop's idle throttle, which
exists purely to avoid pegging a CPU core and plays no part in any timing
decision -- every threshold check inside ``tick()`` compares against
``clock.now()``, never against how long the thread has actually slept.

No Qt anywhere: telemetry, events and command outcomes reach the owner
through plain callbacks, so this stays usable headless.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from celikkubbe.core import config
from celikkubbe.core.commands import Command
from celikkubbe.core.protocols import Clock, TurretLink
from celikkubbe.core.types import Telemetry
from celikkubbe.io.codec import AckResult

# Purely a CPU-yield between ticks in the background thread -- not a
# timing source. Short enough that the 20Hz heartbeat and 100Hz telemetry
# cadence are still met comfortably.
_IDLE_SLEEP_S = 0.005

# How long link_worker waits for an ACK before retransmitting once, then
# reporting failure. Distinct from the MCU's own 200ms no-command
# watchdog (config.WATCHDOG_TIMEOUT_MS) -- this is round-trip budget for a
# single command, not "how long since anything was heard at all".
DEFAULT_ACK_TIMEOUT_MS = 150.0

# Millisecond-scale slack on every elapsed-time-vs-threshold comparison
# below, so that a threshold meant to land exactly on a tick boundary
# (e.g. a 20Hz heartbeat checked every 50ms) doesn't miss by a float ULP.
_TIMING_EPSILON_MS = 1e-6

TelemetryCallback = Callable[[Telemetry], None]
EventCallback = Callable[[Any], None]
CommandRejectedCallback = Callable[[Command, AckResult], None]
CommandFailedCallback = Callable[[Command], None]


@dataclass
class LinkStats:
    round_trip_latency_ms: float | None = None
    crc_error_count: int = 0
    frames_dropped: int = 0
    retransmissions: int = 0
    commands_failed: int = 0


@dataclass
class _PendingCommand:
    cmd: Command
    sent_at: float
    retransmitted: bool = False


class LinkWorker:
    def __init__(
        self,
        link: TurretLink,
        clock: Clock,
        on_telemetry: TelemetryCallback | None = None,
        on_event: EventCallback | None = None,
        on_command_rejected: CommandRejectedCallback | None = None,
        on_command_failed: CommandFailedCallback | None = None,
        ack_timeout_ms: float = DEFAULT_ACK_TIMEOUT_MS,
    ) -> None:
        self._link = link
        self._clock = clock
        self._on_telemetry = on_telemetry
        self._on_event = on_event
        self._on_command_rejected = on_command_rejected
        self._on_command_failed = on_command_failed
        self._ack_timeout_ms = ack_timeout_ms

        self._lock = threading.Lock()
        self._latest_telemetry: Telemetry | None = None
        self._last_telemetry_t: float | None = None
        self._link_lost = True
        self._stats = LinkStats()

        self._outbox: deque[Command] = deque()
        self._pending: dict[int, _PendingCommand] = {}
        self._last_heartbeat_t = clock.now()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # --- owner-facing: thread lifecycle ---

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        close = getattr(self._link, "close", None)
        if callable(close):
            close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.tick()
            time.sleep(_IDLE_SLEEP_S)

    # --- owner-facing: state ---

    def send(self, cmd: Command) -> None:
        with self._lock:
            self._outbox.append(cmd)

    @property
    def latest_telemetry(self) -> Telemetry | None:
        with self._lock:
            return self._latest_telemetry

    @property
    def link_lost(self) -> bool:
        with self._lock:
            return self._link_lost

    @property
    def stats(self) -> LinkStats:
        with self._lock:
            return dataclasses.replace(self._stats)

    # --- the work itself ---

    def tick(self) -> None:
        now = self._clock.now()
        self._send_heartbeat_if_due(now)
        self._drain_outbox(now)
        self._poll_and_dispatch(now)
        self._check_staleness(now)
        self._check_retransmits(now)
        self._refresh_link_diagnostics()

    def _send_heartbeat_if_due(self, now: float) -> None:
        if (now - self._last_heartbeat_t) * 1000.0 < config.HEARTBEAT_MS - _TIMING_EPSILON_MS:
            return
        send_heartbeat = getattr(self._link, "send_heartbeat", None)
        if callable(send_heartbeat):
            send_heartbeat()
        self._last_heartbeat_t = now

    def _drain_outbox(self, now: float) -> None:
        with self._lock:
            to_send = list(self._outbox)
            self._outbox.clear()
        send_tracked = getattr(self._link, "send_tracked", None)
        for cmd in to_send:
            if callable(send_tracked):
                seq = send_tracked(cmd)
                with self._lock:
                    self._pending[seq] = _PendingCommand(cmd=cmd, sent_at=now)
            else:
                self._link.send(cmd)  # no ACK tracking possible; fire-and-forget

    def _poll_and_dispatch(self, now: float) -> None:
        telemetry = self._link.poll()
        if telemetry is not None:
            with self._lock:
                self._latest_telemetry = telemetry
                self._last_telemetry_t = now
            if self._on_telemetry is not None:
                self._on_telemetry(telemetry)

        drain_events = getattr(self._link, "drain_events", None)
        if callable(drain_events):
            events = drain_events()
            if self._on_event is not None:
                for event in events:
                    self._on_event(event)

        drain_acks = getattr(self._link, "drain_acks", None)
        if callable(drain_acks):
            for seq, result in drain_acks():
                self._handle_ack(seq, result, now)

    def _handle_ack(self, seq: int, result: AckResult, now: float) -> None:
        with self._lock:
            pending = self._pending.pop(seq, None)
        if pending is None:
            return
        latency_ms = (now - pending.sent_at) * 1000.0
        with self._lock:
            self._stats.round_trip_latency_ms = latency_ms
        if result is not AckResult.OK and self._on_command_rejected is not None:
            self._on_command_rejected(pending.cmd, result)

    def _check_staleness(self, now: float) -> None:
        with self._lock:
            last_t = self._last_telemetry_t
        lost = last_t is None or (
            (now - last_t) * 1000.0 > config.TELEMETRY_STALE_MS + _TIMING_EPSILON_MS
        )
        with self._lock:
            self._link_lost = lost

    def _check_retransmits(self, now: float) -> None:
        with self._lock:
            snapshot = list(self._pending.items())
        send_tracked = getattr(self._link, "send_tracked", None)
        for seq, pending in snapshot:
            age_ms = (now - pending.sent_at) * 1000.0
            if age_ms < self._ack_timeout_ms - _TIMING_EPSILON_MS:
                continue
            if not pending.retransmitted:
                if callable(send_tracked):
                    send_tracked(pending.cmd, seq)
                with self._lock:
                    if seq in self._pending:
                        self._pending[seq] = dataclasses.replace(
                            pending, sent_at=now, retransmitted=True
                        )
                        self._stats.retransmissions += 1
            else:
                with self._lock:
                    self._pending.pop(seq, None)
                    self._stats.commands_failed += 1
                if self._on_command_failed is not None:
                    self._on_command_failed(pending.cmd)

    def _refresh_link_diagnostics(self) -> None:
        crc = getattr(self._link, "crc_error_count", None)
        dropped = getattr(self._link, "dropped_byte_count", None)
        with self._lock:
            if crc is not None:
                self._stats.crc_error_count = crc
            if dropped is not None:
                self._stats.frames_dropped = dropped
