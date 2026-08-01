"""SerialTurretLink: the real RS422 link, per docs/protocol.md.

pyserial over ``/dev/ttyUSB*`` at 921600 8N1. Reads never wait for bytes
that aren't already buffered -- ``poll()`` only ever asks for exactly
``in_waiting`` bytes, so it returns immediately whether or not new data
has arrived -- and a disconnected port reconnects with backoff instead of
raising into whatever owns this object.

The user must be in the ``dialout`` group (or an equivalent udev rule must
grant the permission) to open ``/dev/ttyUSB*`` at all. A ``PermissionError``
here is that, not a hardware fault, but ``_connect`` folds it into the same
"treat as disconnected, retry later" path as every other open failure, so
it looks identical from the outside -- point anyone debugging a link that
never comes up at this comment before the wiring.
"""

from __future__ import annotations

from pathlib import Path

import serial
import serial.tools.list_ports as list_ports

from celikkubbe.core.commands import Command
from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import Telemetry
from celikkubbe.io.codec import FrameParser, TelemetryFrame, encode_command, encode_heartbeat

BAUD_RATE = 921600
READ_TIMEOUT_S = 0.02
RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 5.0)  # holds at the last value past this many attempts

# Waveshare isolated RS422-USB converter's default CH340 VID/PID. Override
# with an explicit `port=` for a different adapter or a udev symlink.
DEFAULT_VID = 0x1A86
DEFAULT_PID = 0x7523


def find_port(vid: int = DEFAULT_VID, pid: int = DEFAULT_PID) -> str | None:
    for info in list_ports.comports():
        if info.vid == vid and info.pid == pid:
            return info.device
    return None


class SerialTurretLink:
    """A ``TurretLink`` over a real serial port.

    Extra attributes beyond the Protocol, mirroring ``SimTurretLink``'s
    diagnostic surface so ``link_worker`` can read either uniformly:
    ``crc_error_count``, ``resync_count``, ``dropped_byte_count`` (from the
    underlying ``FrameParser``), and ``last_telemetry_frame`` (the full
    wire payload, for whatever a future GUI debug panel wants beyond
    ``core.types.Telemetry``).
    """

    def __init__(
        self,
        clock: Clock,
        port: str | None = None,
        baudrate: int = BAUD_RATE,
        vid: int = DEFAULT_VID,
        pid: int = DEFAULT_PID,
        log_path: str | Path | None = None,
    ) -> None:
        self._clock = clock
        self._explicit_port = port
        self._baudrate = baudrate
        self._vid = vid
        self._pid = pid
        self._serial: serial.Serial | None = None
        self._parser = FrameParser(clock)
        self._last_telemetry_frame: TelemetryFrame | None = None
        self._seq = 0
        self._backoff_index = 0
        self._next_reconnect_attempt_t = clock.now()
        self._log_file = open(log_path, "ab") if log_path is not None else None
        self._connect()

    # --- TurretLink protocol ---

    @property
    def connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def send(self, cmd: Command) -> None:
        if self._ensure_connected():
            self._write(encode_command(cmd, self._seq))
            self._seq = (self._seq + 1) & 0xFFFF

    def poll(self) -> Telemetry | None:
        if not self._ensure_connected():
            return None
        data = self._read_available()
        if data is None:
            return None
        if data and self._log_file is not None:
            self._log_file.write(data)
            self._log_file.flush()
        latest: Telemetry | None = None
        for frame in self._parser.feed(data):
            if isinstance(frame, TelemetryFrame):
                self._last_telemetry_frame = frame
                latest = frame.telemetry
        return latest

    # --- link-layer extras used by link_worker, beyond the Protocol ---

    def send_heartbeat(self) -> None:
        """``hb`` has no ``Command`` representation -- see codec.py."""
        if self._ensure_connected():
            self._write(encode_heartbeat(self._seq))
            self._seq = (self._seq + 1) & 0xFFFF

    @property
    def crc_error_count(self) -> int:
        return self._parser.crc_error_count

    @property
    def resync_count(self) -> int:
        return self._parser.resync_count

    @property
    def dropped_byte_count(self) -> int:
        return self._parser.dropped_byte_count

    @property
    def last_telemetry_frame(self) -> TelemetryFrame | None:
        return self._last_telemetry_frame

    def close(self) -> None:
        self._disconnect()
        if self._log_file is not None:
            self._log_file.close()

    # --- connection management ---

    def _resolve_port(self) -> str | None:
        if self._explicit_port is not None:
            return self._explicit_port
        return find_port(self._vid, self._pid)

    def _connect(self) -> None:
        port = self._resolve_port()
        if port is None:
            self._serial = None
            return
        try:
            self._serial = serial.Serial(
                port,
                self._baudrate,
                timeout=READ_TIMEOUT_S,
                write_timeout=READ_TIMEOUT_S,
            )
        except (serial.SerialException, OSError):
            self._serial = None

    def _disconnect(self) -> None:
        # Reached both from a failed write/read on an otherwise-open port
        # and, indirectly, from a failed reconnect attempt below -- either
        # way, the next attempt must still back off, or a flaky link would
        # get hammered with a reconnect attempt on every single send/poll.
        if self._serial is not None:
            try:
                self._serial.close()
            except (serial.SerialException, OSError):
                pass
        self._serial = None
        self._schedule_backoff()

    def _schedule_backoff(self) -> None:
        delay = RECONNECT_BACKOFF_S[min(self._backoff_index, len(RECONNECT_BACKOFF_S) - 1)]
        self._backoff_index += 1
        self._next_reconnect_attempt_t = self._clock.now() + delay

    def _ensure_connected(self) -> bool:
        if self._serial is not None and self._serial.is_open:
            return True
        now = self._clock.now()
        if now < self._next_reconnect_attempt_t:
            return False
        self._connect()
        if self._serial is not None:
            self._backoff_index = 0
            return True
        self._schedule_backoff()
        return False

    def _read_available(self) -> bytes | None:
        try:
            waiting = self._serial.in_waiting
            return self._serial.read(waiting) if waiting else b""
        except (serial.SerialException, OSError):
            self._disconnect()
            return None

    def _write(self, frame: bytes) -> None:
        try:
            self._serial.write(frame)
        except (serial.SerialException, OSError):
            self._disconnect()
