"""Wire codec for the PC <-> MCU link, per docs/protocol.md v1.0.

Pure functions plus one stateful buffering parser (``FrameParser``) — no
socket or serial I/O anywhere in this module. Byte layouts, field names and
constants below mirror that document exactly; it is the authoritative
source if the two ever disagree.

``core.types.Telemetry`` does not carry every field the wire TELEMETRY
payload has (``mcu_ms``, ``ammo_fired``, ``homed_pan``/``homed_tilt``,
``mcu_mode``, ``watchdog_tripped``, ``limit_pan``/``limit_tilt``) — core/ is
not touched by this codec, so ``TelemetryFrame`` below carries the full
payload with a ``telemetry`` field for the subset that already fits.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import Enum

from celikkubbe.core.commands import (
    Arm,
    Command,
    Disarm,
    Fire,
    Goto,
    Home,
    Jog,
    SetMode,
    SetParam,
    SoftEstop,
    Zero,
)
from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import Axis, Mode, Telemetry

# --- CRC16-CCITT, poly 0x1021, init 0xFFFF, no reflection/xorout (NMEA-style) ---


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# --- outbound: PC -> MCU, newline-delimited JSON (docs/protocol.md section 3.1) ---

MAX_OUTBOUND_FRAME_LEN = 256

# The wire `mode` command only accepts idle/ready/safe -- there is no MCU
# concept of "standby". M1_INIT and M2_STANDBY both mean "not operational,
# not armed" from the MCU's point of view, so both collapse to idle.
_MODE_TO_WIRE: dict[Mode, str] = {
    Mode.M1_INIT: "idle",
    Mode.M2_STANDBY: "idle",
    Mode.M3_OPERATIONAL: "ready",
    Mode.M4_SAFE: "safe",
}

_AXIS_TO_WIRE: dict[Axis, str] = {Axis.PAN: "pan", Axis.TILT: "tilt"}


def _axes_to_wire(axes: tuple[Axis, ...]) -> str:
    present = set(axes)
    if present == {Axis.PAN, Axis.TILT}:
        return "both"
    if present == {Axis.PAN}:
        return "pan"
    if present == {Axis.TILT}:
        return "tilt"
    raise ValueError(f"Home requires at least one axis, got {axes!r}")


def _command_payload(cmd: Command) -> dict[str, object]:
    if isinstance(cmd, SetMode):
        return {"cmd": "mode", "m": _MODE_TO_WIRE[cmd.mode]}
    if isinstance(cmd, Goto):
        return {
            "cmd": "aim",
            "az": round(cmd.az_deg, 2),
            "el": round(cmd.el_deg, 2),
            "vmax": round(cmd.max_vel_dps, 2),
            "amax": round(cmd.max_accel_dps2, 2),
        }
    if isinstance(cmd, Jog):
        return {
            "cmd": "jog",
            "axis": _AXIS_TO_WIRE[cmd.axis],
            "dir": cmd.direction,
            "speed": round(cmd.speed_dps, 2),
        }
    if isinstance(cmd, Home):
        return {"cmd": "home", "axes": _axes_to_wire(cmd.axes)}
    if isinstance(cmd, Zero):
        return {"cmd": "zero", "axis": _AXIS_TO_WIRE[cmd.axis], "value": round(cmd.value_deg, 2)}
    if isinstance(cmd, Arm):
        return {"cmd": "arm"}
    if isinstance(cmd, Disarm):
        return {"cmd": "disarm"}
    if isinstance(cmd, Fire):
        return {"cmd": "fire", "n": cmd.count}
    if isinstance(cmd, SoftEstop):
        return {"cmd": "estop"}
    if isinstance(cmd, SetParam):
        return {"cmd": "param", "id": cmd.param_id, "v": round(cmd.value, 2)}
    raise TypeError(f"unknown command type: {type(cmd).__name__}")  # pragma: no cover — exhaustive


def _encode_json_frame(payload: dict[str, object]) -> bytes:
    # allow_nan=False makes json.dumps raise on NaN/Infinity instead of
    # emitting the non-standard tokens the MCU's JSON parser can't read.
    json_bytes = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("ascii")
    crc_hex = f"{crc16_ccitt(json_bytes):04X}".encode("ascii")
    frame = json_bytes + b"*" + crc_hex + b"\n"
    if len(frame) > MAX_OUTBOUND_FRAME_LEN:
        raise ValueError(
            f"encoded frame is {len(frame)} bytes, exceeds the "
            f"{MAX_OUTBOUND_FRAME_LEN}-byte limit (docs/protocol.md section 3.1)"
        )
    return frame


def encode_command(cmd: Command, seq: int) -> bytes:
    """Serialise a Command to ``{json}*CRC\\n`` per docs/protocol.md section 3.1."""
    payload = _command_payload(cmd)
    payload["seq"] = seq & 0xFFFF
    return _encode_json_frame(payload)


def encode_heartbeat(seq: int) -> bytes:
    """``hb`` has no ``Command`` representation in core/ -- link_worker sends it directly."""
    return _encode_json_frame({"cmd": "hb", "seq": seq & 0xFFFF})


# --- inbound: MCU -> PC, fixed binary frames (docs/protocol.md section 3.2) ---

SYNC = b"\xaa\x55"

TYPE_TELEMETRY = 0x80
TYPE_ACK = 0x81
TYPE_EVENT = 0x82
TYPE_LOG = 0x83

# mcu_ms, pan_deg, tilt_deg, pan_vel_dps, tilt_vel_dps, target_pan_deg,
# target_tilt_deg, status, fan_rpm[3], mcu_temp_c10, loop_time_us,
# crc_error_count, ammo_fired -- little-endian, packed (docs section 4).
_TELEMETRY_STRUCT = "<IffffffHHHHhHHH"
_TELEMETRY_LEN = struct.calcsize(_TELEMETRY_STRUCT)
assert _TELEMETRY_LEN == 44, (
    f"telemetry_t packs to {_TELEMETRY_LEN} bytes, expected 44 "
    "(docs/protocol.md section 4) -- check _TELEMETRY_STRUCT against the C struct"
)

_ACK_STRUCT = "<HBB"  # ack_seq, result, detail
_ACK_LEN = struct.calcsize(_ACK_STRUCT)
assert _ACK_LEN == 4

_EVENT_STRUCT = "<BBIf"  # event_id, axis, mcu_ms, value
_EVENT_LEN = struct.calcsize(_EVENT_STRUCT)
assert _EVENT_LEN == 10

_MAX_LOG_LEN = 255  # LEN is a uint8; LOG is the only variable-length payload

_HEADER_STRUCT = "<BBH"  # LEN, TYPE, SEQ -- follows the 2 sync bytes
_HEADER_LEN = struct.calcsize(_HEADER_STRUCT)
_FRAME_OVERHEAD = len(SYNC) + _HEADER_LEN + 2  # + trailing CRC16
_MAX_FRAME_LEN = _FRAME_OVERHEAD + 255  # worst case: a full-length LOG frame
_BUFFER_CAP = 4 * _MAX_FRAME_LEN

_EXPECTED_PAYLOAD_LEN = {
    TYPE_TELEMETRY: _TELEMETRY_LEN,
    TYPE_ACK: _ACK_LEN,
    TYPE_EVENT: _EVENT_LEN,
}


class McuMode(Enum):
    BOOT = 0
    IDLE = 1
    READY = 2
    MOVING = 3
    SAFE = 4


class AckResult(Enum):
    OK = 0
    BAD_CRC = 1
    MALFORMED_JSON = 2
    UNKNOWN_COMMAND = 3
    PARAM_OUT_OF_RANGE = 4
    REJECTED_IN_MODE = 5
    NOT_ARMED = 6
    ESTOP_ACTIVE = 7
    POSITION_INVALID = 8
    DRIVER_ALARM = 9
    LIMIT_EXCEEDED = 10


class EventId(Enum):
    ESTOP_PRESSED = 1
    ESTOP_RELEASED = 2
    DRIVER_ALARM_ASSERTED = 3
    DRIVER_ALARM_CLEARED = 4
    WATCHDOG_TRIPPED = 5
    MOTION_COMPLETE = 6
    SHOT_FIRED = 7
    LIMIT_HIT = 8
    FAN_FAULT = 9
    OVER_TEMPERATURE = 10


_AXIS_FROM_WIRE = {0: Axis.PAN, 1: Axis.TILT}  # 255 = not applicable -> None


@dataclass(frozen=True)
class TelemetryFrame:
    seq: int
    telemetry: Telemetry
    mcu_ms: int
    ammo_fired: int
    homed_pan: bool
    homed_tilt: bool
    mcu_mode: McuMode
    watchdog_tripped: bool
    limit_pan: bool
    limit_tilt: bool


@dataclass(frozen=True)
class AckFrame:
    seq: int
    ack_seq: int
    result: AckResult
    detail: int


@dataclass(frozen=True)
class EventFrame:
    seq: int
    event_id: EventId
    axis: Axis | None
    mcu_ms: int
    value: float


@dataclass(frozen=True)
class LogFrame:
    seq: int
    text: str


ParsedFrame = TelemetryFrame | AckFrame | EventFrame | LogFrame


def _decode_telemetry(seq: int, payload: bytes, t: float) -> TelemetryFrame:
    (
        mcu_ms,
        pan_deg,
        tilt_deg,
        pan_vel_dps,
        tilt_vel_dps,
        target_pan_deg,
        target_tilt_deg,
        status,
        fan0,
        fan1,
        fan2,
        mcu_temp_c10,
        loop_time_us,
        crc_error_count,
        ammo_fired,
    ) = struct.unpack(_TELEMETRY_STRUCT, payload)

    telemetry = Telemetry(
        t=t,
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        pan_vel_dps=pan_vel_dps,
        tilt_vel_dps=tilt_vel_dps,
        target_pan_deg=target_pan_deg,
        target_tilt_deg=target_tilt_deg,
        motion_complete=bool(status & (1 << 3)),
        armed=bool(status & (1 << 0)),
        estop=bool(status & (1 << 1)),
        position_valid=bool(status & (1 << 2)),
        driver_alarm_pan=bool(status & (1 << 4)),
        driver_alarm_tilt=bool(status & (1 << 5)),
        fan_rpm=(fan0, fan1, fan2),
        mcu_temp_c=mcu_temp_c10 / 10.0,
        loop_time_us=loop_time_us,
        crc_error_count=crc_error_count,
    )
    return TelemetryFrame(
        seq=seq,
        telemetry=telemetry,
        mcu_ms=mcu_ms,
        ammo_fired=ammo_fired,
        homed_pan=bool(status & (1 << 6)),
        homed_tilt=bool(status & (1 << 7)),
        mcu_mode=McuMode((status >> 11) & 0b111),
        watchdog_tripped=bool(status & (1 << 8)),
        limit_pan=bool(status & (1 << 9)),
        limit_tilt=bool(status & (1 << 10)),
    )


def _decode_ack(seq: int, payload: bytes) -> AckFrame:
    ack_seq, result_raw, detail = struct.unpack(_ACK_STRUCT, payload)
    return AckFrame(seq=seq, ack_seq=ack_seq, result=AckResult(result_raw), detail=detail)


def _decode_event(seq: int, payload: bytes) -> EventFrame:
    event_id_raw, axis_raw, mcu_ms, value = struct.unpack(_EVENT_STRUCT, payload)
    return EventFrame(
        seq=seq,
        event_id=EventId(event_id_raw),
        axis=_AXIS_FROM_WIRE.get(axis_raw),
        mcu_ms=mcu_ms,
        value=value,
    )


def _decode_log(seq: int, payload: bytes) -> LogFrame:
    return LogFrame(seq=seq, text=payload.decode("utf-8", errors="replace"))


def _decode_payload(type_byte: int, seq: int, payload: bytes, t: float) -> ParsedFrame:
    if type_byte == TYPE_TELEMETRY:
        return _decode_telemetry(seq, payload, t)
    if type_byte == TYPE_ACK:
        return _decode_ack(seq, payload)
    if type_byte == TYPE_EVENT:
        return _decode_event(seq, payload)
    return _decode_log(seq, payload)  # only remaining plausible type; see _is_plausible


def _is_plausible(type_byte: int, length: int) -> bool:
    if type_byte == TYPE_LOG:
        return length <= _MAX_LOG_LEN
    return length == _EXPECTED_PAYLOAD_LEN.get(type_byte, -1)


class FrameParser:
    """Streaming MCU -> PC frame parser (docs/protocol.md section 3.2).

    Holds a byte buffer across ``feed()`` calls so frames split across
    serial reads reassemble, and several frames delivered in one read all
    come out. ``clock`` stamps ``Telemetry.t`` with the PC-side receive
    time -- never ``mcu_ms``, which is a different, unsynchronised clock
    kept only as a diagnostic field on ``TelemetryFrame``.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._buf = bytearray()
        self.crc_error_count = 0
        self.resync_count = 0
        self.dropped_byte_count = 0

    def feed(self, data: bytes) -> list[ParsedFrame]:
        self._buf.extend(data)
        frames: list[ParsedFrame] = []

        while True:
            sync_index = self._buf.find(SYNC)
            if sync_index == -1:
                break
            if sync_index > 0:
                del self._buf[:sync_index]
                self.dropped_byte_count += sync_index

            if len(self._buf) < len(SYNC) + _HEADER_LEN:
                break  # header not fully arrived yet

            length, type_byte, seq = struct.unpack(
                _HEADER_STRUCT, bytes(self._buf[len(SYNC) : len(SYNC) + _HEADER_LEN])
            )

            if not _is_plausible(type_byte, length):
                # A sync-looking byte pair that isn't a real frame start --
                # either garbage or AA 55 occurring inside payload data.
                # Drop one byte and keep scanning; do not discard the frame
                # that might start one byte later.
                self._resync()
                continue

            total_len = len(SYNC) + _HEADER_LEN + length + 2
            if len(self._buf) < total_len:
                break  # complete frame hasn't arrived yet

            header_and_payload = bytes(self._buf[len(SYNC) : len(SYNC) + _HEADER_LEN + length])
            (received_crc,) = struct.unpack(
                "<H", bytes(self._buf[len(SYNC) + _HEADER_LEN + length : total_len])
            )

            if crc16_ccitt(header_and_payload) != received_crc:
                self.crc_error_count += 1
                self._resync()
                continue

            payload = header_and_payload[_HEADER_LEN:]
            try:
                frame = _decode_payload(type_byte, seq, payload, self._clock.now())
            except (struct.error, ValueError):
                # CRC matched but a field decoded outside its defined range
                # (e.g. an mcu_mode/event_id/result the firmware doesn't
                # actually send) -- indistinguishable from corruption here.
                self.crc_error_count += 1
                self._resync()
                continue

            del self._buf[:total_len]
            frames.append(frame)

        self._cap_buffer()
        return frames

    def _resync(self, bytes_to_drop: int = 1) -> None:
        del self._buf[:bytes_to_drop]
        self.dropped_byte_count += bytes_to_drop
        self.resync_count += 1

    def _cap_buffer(self) -> None:
        if len(self._buf) > _BUFFER_CAP:
            overflow = len(self._buf) - _MAX_FRAME_LEN
            del self._buf[:overflow]
            self.dropped_byte_count += overflow
