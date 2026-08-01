from __future__ import annotations

import json
import struct

import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.commands import (
    Arm,
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
from celikkubbe.core.types import Axis, Mode
from celikkubbe.io import codec

# Independent transcription of docs/protocol.md section 4's telemetry_t,
# kept separate from codec._TELEMETRY_STRUCT so a bug in that constant
# doesn't get validated against itself.
_TELEMETRY_FMT = "<IffffffHHHHhHHH"


def _split_frame(frame: bytes) -> tuple[dict, str]:
    assert frame.endswith(b"\n")
    body, crc_hex = frame[:-1].rsplit(b"*", 1)
    return json.loads(body), crc_hex.decode("ascii")


def _build_inbound_frame(type_byte: int, seq: int, payload: bytes) -> bytes:
    header = struct.pack("<BBH", len(payload), type_byte, seq)
    crc = codec.crc16_ccitt(header + payload)
    return codec.SYNC + header + payload + struct.pack("<H", crc)


def _telemetry_payload(
    mcu_ms: int = 1000,
    pan_deg: float = 1.0,
    tilt_deg: float = 2.0,
    pan_vel_dps: float = 0.0,
    tilt_vel_dps: float = 0.0,
    target_pan_deg: float = 1.0,
    target_tilt_deg: float = 2.0,
    status: int = 0,
    fan_rpm: tuple[int, int, int] = (3000, 3000, 3000),
    mcu_temp_c10: int = 250,
    loop_time_us: int = 500,
    crc_error_count: int = 0,
    ammo_fired: int = 0,
) -> bytes:
    return struct.pack(
        _TELEMETRY_FMT,
        mcu_ms,
        pan_deg,
        tilt_deg,
        pan_vel_dps,
        tilt_vel_dps,
        target_pan_deg,
        target_tilt_deg,
        status,
        *fan_rpm,
        mcu_temp_c10,
        loop_time_us,
        crc_error_count,
        ammo_fired,
    )


# --- CRC ---


def test_crc16_matches_independent_known_vector() -> None:
    # Standard CRC-16/CCITT-FALSE check value (poly 0x1021, init 0xFFFF,
    # no reflection, no xorout) for ASCII "123456789".
    assert codec.crc16_ccitt(b"123456789") == 0x29B1


def test_telemetry_struct_size_matches_protocol_doc() -> None:
    assert struct.calcsize(_TELEMETRY_FMT) == 44


# --- outbound encoding ---


def test_encode_goto_as_aim() -> None:
    payload, crc_hex = _split_frame(codec.encode_command(Goto(10.0, -5.0, 90.0, 180.0), seq=7))
    assert payload == {"cmd": "aim", "az": 10.0, "el": -5.0, "vmax": 90.0, "amax": 180.0, "seq": 7}
    assert (
        crc_hex == f"{codec.crc16_ccitt(json.dumps(payload, separators=(',', ':')).encode()):04X}"
    )


def test_encode_command_appends_correct_crc() -> None:
    frame = codec.encode_command(Arm(), seq=1)
    json_bytes, crc_hex = frame[:-1].rsplit(b"*", 1)
    assert int(crc_hex, 16) == codec.crc16_ccitt(json_bytes)


@pytest.mark.parametrize(
    ("mode", "wire"),
    [
        (Mode.M1_INIT, "idle"),
        (Mode.M2_STANDBY, "idle"),
        (Mode.M3_OPERATIONAL, "ready"),
        (Mode.M4_SAFE, "safe"),
    ],
)
def test_encode_set_mode_maps_all_four_pc_modes(mode: Mode, wire: str) -> None:
    payload, _ = _split_frame(codec.encode_command(SetMode(mode), seq=1))
    assert payload == {"cmd": "mode", "m": wire, "seq": 1}


def test_encode_jog() -> None:
    payload, _ = _split_frame(codec.encode_command(Jog(Axis.PAN, 1, 15.0), seq=2))
    assert payload == {"cmd": "jog", "axis": "pan", "dir": 1, "speed": 15.0, "seq": 2}


@pytest.mark.parametrize(
    ("axes", "wire"),
    [
        ((Axis.PAN,), "pan"),
        ((Axis.TILT,), "tilt"),
        ((Axis.PAN, Axis.TILT), "both"),
    ],
)
def test_encode_home(axes: tuple[Axis, ...], wire: str) -> None:
    payload, _ = _split_frame(codec.encode_command(Home(axes), seq=3))
    assert payload == {"cmd": "home", "axes": wire, "seq": 3}


def test_encode_home_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        codec.encode_command(Home(()), seq=3)


def test_encode_zero() -> None:
    payload, _ = _split_frame(codec.encode_command(Zero(Axis.TILT, 12.34), seq=4))
    assert payload == {"cmd": "zero", "axis": "tilt", "value": 12.34, "seq": 4}


def test_encode_arm_disarm_estop() -> None:
    assert _split_frame(codec.encode_command(Arm(), seq=5))[0] == {"cmd": "arm", "seq": 5}
    assert _split_frame(codec.encode_command(Disarm(), seq=5))[0] == {"cmd": "disarm", "seq": 5}
    assert _split_frame(codec.encode_command(SoftEstop(), seq=5))[0] == {"cmd": "estop", "seq": 5}


def test_encode_fire() -> None:
    payload, _ = _split_frame(codec.encode_command(Fire(count=3), seq=6))
    assert payload == {"cmd": "fire", "n": 3, "seq": 6}


def test_encode_set_param() -> None:
    payload, _ = _split_frame(codec.encode_command(SetParam("trigger_pulse_ms", 60.0), seq=8))
    assert payload == {"cmd": "param", "id": "trigger_pulse_ms", "v": 60.0, "seq": 8}


def test_encode_heartbeat() -> None:
    payload, _ = _split_frame(codec.encode_heartbeat(seq=42))
    assert payload == {"cmd": "hb", "seq": 42}


def test_encode_rounds_floats_to_two_decimals() -> None:
    payload, _ = _split_frame(codec.encode_command(Goto(1.0 / 3.0, 0.0, 90.0, 180.0), seq=1))
    assert payload["az"] == 0.33


def test_encode_rejects_nan_angle() -> None:
    with pytest.raises(ValueError):
        codec.encode_command(Goto(float("nan"), 0.0, 90.0, 180.0), seq=1)


def test_encode_rejects_infinity() -> None:
    with pytest.raises(ValueError):
        codec.encode_command(Goto(float("inf"), 0.0, 90.0, 180.0), seq=1)


def test_encode_seq_wraps_to_uint16_range() -> None:
    payload, _ = _split_frame(codec.encode_command(Arm(), seq=65536 + 5))
    assert payload["seq"] == 5


def test_encode_oversized_frame_raises() -> None:
    with pytest.raises(ValueError, match="256-byte limit"):
        codec.encode_command(SetParam("x" * 300, 1.0), seq=1)


# --- inbound: FrameParser ---


def test_feed_parses_single_telemetry_frame() -> None:
    clock = FakeClock(42.0)
    parser = codec.FrameParser(clock)
    frame_bytes = _build_inbound_frame(codec.TYPE_TELEMETRY, 1, _telemetry_payload())
    (frame,) = parser.feed(frame_bytes)
    assert isinstance(frame, codec.TelemetryFrame)
    assert frame.telemetry.pan_deg == 1.0
    assert frame.telemetry.tilt_deg == 2.0


def test_telemetry_t_comes_from_clock_not_mcu_ms() -> None:
    clock = FakeClock(99.5)
    parser = codec.FrameParser(clock)
    frame_bytes = _build_inbound_frame(codec.TYPE_TELEMETRY, 1, _telemetry_payload(mcu_ms=123456))
    (frame,) = parser.feed(frame_bytes)
    assert frame.telemetry.t == 99.5
    assert frame.mcu_ms == 123456


def test_feed_reassembles_frame_split_across_multiple_reads() -> None:
    parser = codec.FrameParser(FakeClock())
    frame_bytes = _build_inbound_frame(codec.TYPE_TELEMETRY, 1, _telemetry_payload())
    assert parser.feed(frame_bytes[:10]) == []
    assert parser.feed(frame_bytes[10:20]) == []
    (frame,) = parser.feed(frame_bytes[20:])
    assert isinstance(frame, codec.TelemetryFrame)


def test_feed_parses_frame_fed_one_byte_at_a_time() -> None:
    parser = codec.FrameParser(FakeClock())
    frame_bytes = _build_inbound_frame(codec.TYPE_TELEMETRY, 1, _telemetry_payload())
    frames = []
    for b in frame_bytes:
        frames.extend(parser.feed(bytes([b])))
    assert len(frames) == 1


def test_feed_parses_multiple_frames_in_one_read() -> None:
    parser = codec.FrameParser(FakeClock())
    f1 = _build_inbound_frame(codec.TYPE_ACK, 1, struct.pack(codec._ACK_STRUCT, 1, 0, 0))
    f2 = _build_inbound_frame(codec.TYPE_ACK, 2, struct.pack(codec._ACK_STRUCT, 2, 0, 0))
    f3 = _build_inbound_frame(codec.TYPE_ACK, 3, struct.pack(codec._ACK_STRUCT, 3, 0, 0))
    frames = parser.feed(f1 + f2 + f3)
    assert [f.ack_seq for f in frames] == [1, 2, 3]


def test_feed_resyncs_after_corrupted_frame() -> None:
    parser = codec.FrameParser(FakeClock())
    good = _build_inbound_frame(codec.TYPE_ACK, 1, struct.pack(codec._ACK_STRUCT, 1, 0, 0))
    corrupt = bytearray(
        _build_inbound_frame(codec.TYPE_ACK, 2, struct.pack(codec._ACK_STRUCT, 2, 0, 0))
    )
    corrupt[-1] ^= 0xFF  # flip a CRC byte
    second_good = _build_inbound_frame(codec.TYPE_ACK, 3, struct.pack(codec._ACK_STRUCT, 3, 0, 0))

    frames = parser.feed(good + bytes(corrupt) + second_good)

    assert [f.ack_seq for f in frames] == [1, 3]
    assert parser.crc_error_count == 1
    assert parser.resync_count >= 1


def test_sync_pattern_inside_garbage_does_not_swallow_next_frame() -> None:
    parser = codec.FrameParser(FakeClock())
    # A fake sync candidate whose header claims an implausible (LEN, TYPE)
    # combination, followed by a real frame.
    fake = codec.SYNC + b"\x99\x99\x99\x99"
    real = _build_inbound_frame(codec.TYPE_ACK, 9, struct.pack(codec._ACK_STRUCT, 9, 0, 0))

    frames = parser.feed(fake + real)

    assert [f.ack_seq for f in frames] == [9]
    assert parser.resync_count >= 1


def test_sync_bytes_inside_a_log_payload_do_not_confuse_the_parser() -> None:
    parser = codec.FrameParser(FakeClock())
    payload = b"boot ok " + codec.SYNC + b" done"
    frame_bytes = _build_inbound_frame(codec.TYPE_LOG, 1, payload)

    (frame,) = parser.feed(frame_bytes)

    assert isinstance(frame, codec.LogFrame)
    assert frame.text == payload.decode("utf-8", errors="replace")
    assert parser.resync_count == 0


def test_buffer_growth_bounded_under_a_stream_of_garbage() -> None:
    parser = codec.FrameParser(FakeClock())
    parser.feed(b"\x00" * 100_000)
    assert len(parser._buf) <= codec._BUFFER_CAP
    assert parser.dropped_byte_count > 0


def test_buffer_growth_bounded_when_a_claimed_frame_never_completes() -> None:
    parser = codec.FrameParser(FakeClock())
    # A plausible LOG header (LEN=255) whose payload never arrives.
    header = struct.pack("<BBH", 255, codec.TYPE_LOG, 1)
    parser.feed(codec.SYNC + header)
    for _ in range(50):
        parser.feed(b"\x00" * 100)
    assert len(parser._buf) <= codec._BUFFER_CAP


# --- status bitfield ---


def test_status_bitfield_unpacks_every_boolean_bit() -> None:
    status = 0
    for bit in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        status |= 1 << bit
    parser = codec.FrameParser(FakeClock())
    frame_bytes = _build_inbound_frame(codec.TYPE_TELEMETRY, 1, _telemetry_payload(status=status))
    (frame,) = parser.feed(frame_bytes)

    assert frame.telemetry.armed is True
    assert frame.telemetry.estop is True
    assert frame.telemetry.position_valid is True
    assert frame.telemetry.motion_complete is True
    assert frame.telemetry.driver_alarm_pan is True
    assert frame.telemetry.driver_alarm_tilt is True
    assert frame.homed_pan is True
    assert frame.homed_tilt is True
    assert frame.watchdog_tripped is True
    assert frame.limit_pan is True
    assert frame.limit_tilt is True


def test_status_bitfield_all_clear() -> None:
    parser = codec.FrameParser(FakeClock())
    frame_bytes = _build_inbound_frame(codec.TYPE_TELEMETRY, 1, _telemetry_payload(status=0))
    (frame,) = parser.feed(frame_bytes)

    assert frame.telemetry.armed is False
    assert frame.homed_pan is False
    assert frame.mcu_mode is codec.McuMode.BOOT


@pytest.mark.parametrize(
    ("mode_bits", "expected"),
    [
        (0, codec.McuMode.BOOT),
        (1, codec.McuMode.IDLE),
        (2, codec.McuMode.READY),
        (3, codec.McuMode.MOVING),
        (4, codec.McuMode.SAFE),
    ],
)
def test_mcu_mode_boundary_values(mode_bits: int, expected: codec.McuMode) -> None:
    parser = codec.FrameParser(FakeClock())
    frame_bytes = _build_inbound_frame(
        codec.TYPE_TELEMETRY, 1, _telemetry_payload(status=mode_bits << 11)
    )
    (frame,) = parser.feed(frame_bytes)
    assert frame.mcu_mode is expected


def test_undefined_mcu_mode_value_is_treated_as_corrupt() -> None:
    parser = codec.FrameParser(FakeClock())
    # bits 11-13 = 7 is outside the 0-4 range the protocol defines.
    frame_bytes = _build_inbound_frame(codec.TYPE_TELEMETRY, 1, _telemetry_payload(status=7 << 11))
    frames = parser.feed(frame_bytes)
    assert frames == []
    assert parser.crc_error_count == 1


# --- ACK / EVENT / LOG ---


def test_ack_frame_decodes() -> None:
    parser = codec.FrameParser(FakeClock())
    payload = struct.pack(codec._ACK_STRUCT, 17, codec.AckResult.NOT_ARMED.value, 0)
    (frame,) = parser.feed(_build_inbound_frame(codec.TYPE_ACK, 5, payload))
    assert frame == codec.AckFrame(seq=5, ack_seq=17, result=codec.AckResult.NOT_ARMED, detail=0)


@pytest.mark.parametrize(("axis_byte", "expected"), [(0, Axis.PAN), (1, Axis.TILT), (255, None)])
def test_event_frame_decodes_axis(axis_byte: int, expected: Axis | None) -> None:
    parser = codec.FrameParser(FakeClock())
    payload = struct.pack(codec._EVENT_STRUCT, codec.EventId.SHOT_FIRED.value, axis_byte, 500, 1.5)
    (frame,) = parser.feed(_build_inbound_frame(codec.TYPE_EVENT, 1, payload))
    assert frame.axis is expected
    assert frame.event_id is codec.EventId.SHOT_FIRED
    assert frame.value == pytest.approx(1.5)


def test_log_frame_decodes_utf8_text() -> None:
    parser = codec.FrameParser(FakeClock())
    payload = "motor sıcaklığı normal".encode()
    (frame,) = parser.feed(_build_inbound_frame(codec.TYPE_LOG, 1, payload))
    assert frame.text == "motor sıcaklığı normal"
