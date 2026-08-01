from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import serial

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.commands import Arm
from celikkubbe.io import codec
from celikkubbe.io.serial_link import SerialTurretLink, find_port


@dataclass
class _FakePortInfo:
    device: str
    vid: int | None
    pid: int | None


class _FakeSerial:
    """Minimal stand-in for serial.Serial, driven entirely in-memory --
    mirrors the _FakeCv2Capture pattern used for cv2.VideoCapture in
    tests/vision/test_sources.py.
    """

    def __init__(self, port: str, baudrate: int, **kwargs) -> None:
        self.port = port
        self.baudrate = baudrate
        self.is_open = True
        self._inbound = bytearray()
        self.written: list[bytes] = []
        self.raise_on_write = False
        self.raise_on_read = False

    @property
    def in_waiting(self) -> int:
        return len(self._inbound)

    def read(self, n: int) -> bytes:
        if self.raise_on_read:
            raise serial.SerialException("read failed")
        chunk = bytes(self._inbound[:n])
        del self._inbound[:n]
        return chunk

    def write(self, data: bytes) -> None:
        if self.raise_on_write:
            raise serial.SerialException("write failed")
        self.written.append(bytes(data))

    def close(self) -> None:
        self.is_open = False

    def feed_inbound(self, data: bytes) -> None:
        self._inbound.extend(data)


def _ack_frame(seq: int, ack_seq: int) -> bytes:
    payload = codec.struct.pack(codec._ACK_STRUCT, ack_seq, 0, 0)
    header = codec.struct.pack("<BBH", len(payload), codec.TYPE_ACK, seq)
    crc = codec.crc16_ccitt(header + payload)
    return codec.SYNC + header + payload + codec.struct.pack("<H", crc)


def test_find_port_matches_by_vid_and_pid() -> None:
    ports = [
        _FakePortInfo("/dev/ttyUSB0", vid=0x0403, pid=0x6001),
        _FakePortInfo("/dev/ttyUSB1", vid=0x1A86, pid=0x7523),
    ]
    with patch("celikkubbe.io.serial_link.list_ports.comports", return_value=ports):
        assert find_port(vid=0x1A86, pid=0x7523) == "/dev/ttyUSB1"
        assert find_port(vid=0x9999, pid=0x9999) is None


def test_connects_via_explicit_port_without_auto_detect() -> None:
    fake = _FakeSerial("/dev/ttyUSB3", 921600)
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake) as ctor:
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB3")
        assert link.connected is True
        ctor.assert_called_once()
        assert ctor.call_args[0][0] == "/dev/ttyUSB3"


def test_not_connected_when_no_port_found() -> None:
    with patch("celikkubbe.io.serial_link.list_ports.comports", return_value=[]):
        link = SerialTurretLink(FakeClock())
    assert link.connected is False
    assert link.poll() is None


def test_send_encodes_and_writes_command() -> None:
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
        link.send(Arm())
    assert len(fake.written) == 1
    assert fake.written[0].startswith(b'{"cmd":"arm"')


def test_send_heartbeat_writes_hb_frame() -> None:
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
        link.send_heartbeat()
    assert fake.written[0].startswith(b'{"cmd":"hb"')


def test_poll_returns_none_when_nothing_buffered() -> None:
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
        assert link.poll() is None


def test_poll_decodes_buffered_ack_frame_and_updates_diagnostics() -> None:
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    fake.feed_inbound(_ack_frame(seq=1, ack_seq=42))
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
        # ACK carries no Telemetry -- poll() correctly reports nothing new,
        # but the frame is still parsed and reachable via diagnostics.
        assert link.poll() is None
        assert link.last_telemetry_frame is None
        assert link.crc_error_count == 0


def test_poll_reads_partial_frame_across_two_reads() -> None:
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    frame = _ack_frame(seq=1, ack_seq=7)
    fake.feed_inbound(frame[:5])
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
        link.poll()
        fake.feed_inbound(frame[5:])
        link.poll()
    assert link.resync_count == 0
    assert link.dropped_byte_count == 0


def test_write_failure_disconnects_and_reconnect_is_backed_off() -> None:
    # A real port re-opens as a fresh handle each time; a mock that always
    # hands back the *same* already-closed instance would not exercise
    # that, so each serial.Serial(...) call gets its own fake here.
    fakes: list[_FakeSerial] = []

    def _open(port: str, baudrate: int, **kwargs) -> _FakeSerial:
        fake = _FakeSerial(port, baudrate, **kwargs)
        fakes.append(fake)
        return fake

    clock = FakeClock()
    with patch("celikkubbe.io.serial_link.serial.Serial", side_effect=_open):
        link = SerialTurretLink(clock, port="/dev/ttyUSB0")
        fakes[0].raise_on_write = True
        link.send(Arm())
        assert link.connected is False

        # immediate retry within the backoff window stays disconnected
        clock.advance(0.1)
        assert link.poll() is None
        assert link.connected is False
        assert len(fakes) == 1  # no reconnect attempt yet -- still backed off

        clock.advance(1.0)  # past the first backoff step
        link.send(Arm())
        assert link.connected is True
        assert len(fakes) == 2


def test_read_failure_disconnects() -> None:
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
        fake.raise_on_read = True
        fake._inbound.extend(b"\x00")  # make in_waiting truthy so read() is attempted
        assert link.poll() is None
        assert link.connected is False


def test_open_failure_leaves_link_disconnected() -> None:
    with patch(
        "celikkubbe.io.serial_link.serial.Serial", side_effect=serial.SerialException("busy")
    ):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
    assert link.connected is False


def test_raw_byte_logging_writes_to_file(tmp_path) -> None:
    log_path = tmp_path / "link.raw"
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    fake.feed_inbound(b"\x01\x02\x03")
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0", log_path=log_path)
        link.poll()
        link.close()
    assert log_path.read_bytes() == b"\x01\x02\x03"


def test_close_closes_port() -> None:
    fake = _FakeSerial("/dev/ttyUSB0", 921600)
    with patch("celikkubbe.io.serial_link.serial.Serial", return_value=fake):
        link = SerialTurretLink(FakeClock(), port="/dev/ttyUSB0")
        link.close()
    assert fake.is_open is False
    assert link.connected is False
