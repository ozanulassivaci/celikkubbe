"""GamepadWorker tests. scale_stick_to_speed_dps/normalize_abs_value are
pure functions, tested directly. Device discovery and event parsing are
tested against a hand-written fake evdev device -- the same
hand-written-fake-over-MagicMock pattern test_serial_link.py's
_FakeSerial already uses for pyserial.Serial -- since no physical
gamepad exists in this environment, matching io/serial_link.py's own
"never exercised against real hardware" honestly-flagged gap.
"""

from __future__ import annotations

import evdev
import pytest

from celikkubbe.core import config
from celikkubbe.core.types import Axis
from celikkubbe.ui.gamepad import (
    GamepadWorker,
    find_gamepad_device_path,
    normalize_abs_value,
    scale_stick_to_speed_dps,
)

# --- scale_stick_to_speed_dps ---


def test_stick_within_deadzone_is_zero() -> None:
    assert scale_stick_to_speed_dps(0.05, deadzone=0.15) == 0.0
    assert scale_stick_to_speed_dps(-0.1, deadzone=0.15) == 0.0


def test_full_deflection_reaches_max_speed() -> None:
    assert scale_stick_to_speed_dps(1.0, deadzone=0.15, max_speed_dps=60.0) == 60.0
    assert scale_stick_to_speed_dps(-1.0, deadzone=0.15, max_speed_dps=60.0) == -60.0


def test_scaling_is_linear_from_the_deadzone_edge() -> None:
    # Halfway between the deadzone edge (0.15) and full deflection (1.0).
    half = 0.15 + (1.0 - 0.15) / 2.0
    result = scale_stick_to_speed_dps(half, deadzone=0.15, max_speed_dps=60.0)
    assert result == pytest.approx(30.0)


def test_sign_is_preserved() -> None:
    positive = scale_stick_to_speed_dps(0.5, deadzone=0.15)
    negative = scale_stick_to_speed_dps(-0.5, deadzone=0.15)
    assert positive > 0.0
    assert negative < 0.0
    assert positive == -negative


def test_out_of_range_magnitude_is_clamped() -> None:
    assert scale_stick_to_speed_dps(1.5, deadzone=0.15, max_speed_dps=60.0) == 60.0


def test_default_max_speed_matches_the_jog_speed_config() -> None:
    assert scale_stick_to_speed_dps(1.0) == config.JOG_SPEED_MAX_DPS


# --- normalize_abs_value ---


def test_normalize_abs_value_centres_a_symmetric_joystick_range() -> None:
    # -32768..32767 is the real, slightly-asymmetric range most evdev
    # joysticks report -- its true midpoint is -0.5, not 0, so a raw
    # centred reading of 0 normalises to a hair above zero, not exactly
    # zero. scale_stick_to_speed_dps's own deadzone is what actually
    # absorbs this in practice.
    assert normalize_abs_value(0, -32768, 32767) == pytest.approx(0.0, abs=1e-3)
    assert normalize_abs_value(32767, -32768, 32767) == pytest.approx(1.0)
    assert normalize_abs_value(-32768, -32768, 32767) == pytest.approx(-1.0)


def test_normalize_abs_value_handles_an_asymmetric_trigger_range() -> None:
    assert normalize_abs_value(0, 0, 255) == -1.0
    assert normalize_abs_value(255, 0, 255) == 1.0
    assert normalize_abs_value(127, 0, 255) < 0.05  # near the trigger's own midpoint


def test_normalize_abs_value_handles_a_degenerate_zero_span() -> None:
    assert normalize_abs_value(5, 5, 5) == 0.0


# --- find_gamepad_device_path ---


class _FakeInputDevice:
    def __init__(self, name: str) -> None:
        self.name = name


def test_find_gamepad_device_path_matches_by_name_hint(monkeypatch) -> None:
    devices = {"/dev/input/event3": _FakeInputDevice("Logitech Gamepad F310")}
    monkeypatch.setattr(evdev, "list_devices", lambda: list(devices))
    monkeypatch.setattr(evdev, "InputDevice", lambda path: devices[path])

    assert find_gamepad_device_path() == "/dev/input/event3"


def test_find_gamepad_device_path_returns_none_when_nothing_matches(monkeypatch) -> None:
    devices = {"/dev/input/event0": _FakeInputDevice("Some Other Keyboard")}
    monkeypatch.setattr(evdev, "list_devices", lambda: list(devices))
    monkeypatch.setattr(evdev, "InputDevice", lambda path: devices[path])

    assert find_gamepad_device_path() is None


def test_find_gamepad_device_path_skips_a_device_that_fails_to_open(monkeypatch) -> None:
    def _raise(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(evdev, "list_devices", lambda: ["/dev/input/event0"])
    monkeypatch.setattr(evdev, "InputDevice", _raise)

    assert find_gamepad_device_path() is None


# --- GamepadWorker: graceful no-device handling ---


def test_worker_reports_disconnected_and_stops_cleanly_with_no_device(qtbot):
    worker = GamepadWorker()
    with qtbot.waitSignal(worker.connected_changed, timeout=3000) as blocker:
        worker.start()
    assert blocker.args == [False]
    worker.stop()
    assert worker.wait(2000)


# --- GamepadWorker: event parsing against a fake device ---


class _FakeAbsInfo:
    def __init__(self, min_value: int, max_value: int) -> None:
        self.min = min_value
        self.max = max_value


class _FakeEvent:
    def __init__(self, event_type: int, code: int, value: int) -> None:
        self.type = event_type
        self.code = code
        self.value = value


class _FakeDevice:
    def __init__(self, events: list[_FakeEvent], abs_caps: dict[int, tuple[int, int]]) -> None:
        self.name = "Fake F310"
        self._events = events
        self._abs_caps = abs_caps

    def capabilities(self, absinfo: bool = True):
        return {
            evdev.ecodes.EV_ABS: [
                (code, _FakeAbsInfo(lo, hi)) for code, (lo, hi) in self._abs_caps.items()
            ]
        }

    def read_loop(self):
        yield from self._events


def _make_worker_with_events(events: list[_FakeEvent]) -> tuple[GamepadWorker, _FakeDevice]:
    device = _FakeDevice(
        events,
        abs_caps={
            evdev.ecodes.ABS_X: (-32768, 32767),
            evdev.ecodes.ABS_Y: (-32768, 32767),
            evdev.ecodes.ABS_RZ: (0, 255),
        },
    )
    return GamepadWorker(), device


def test_left_stick_x_emits_pan_jog() -> None:
    events = [_FakeEvent(evdev.ecodes.EV_ABS, evdev.ecodes.ABS_X, 32767)]
    worker, device = _make_worker_with_events(events)
    received = []
    worker.jog_axis_changed.connect(lambda axis, speed: received.append((axis, speed)))

    worker._read_until_disconnected(device)

    assert received == [(Axis.PAN, config.JOG_SPEED_MAX_DPS)]


def test_left_stick_y_is_inverted_for_tilt() -> None:
    # Full-forward on the stick is a negative raw Y in evdev's own axis
    # convention; the turret's tilt-positive-up must move up for that,
    # i.e. a *positive* speed -- see the module docstring.
    events = [_FakeEvent(evdev.ecodes.EV_ABS, evdev.ecodes.ABS_Y, -32768)]
    worker, device = _make_worker_with_events(events)
    received = []
    worker.jog_axis_changed.connect(lambda axis, speed: received.append((axis, speed)))

    worker._read_until_disconnected(device)

    assert received == [(Axis.TILT, config.JOG_SPEED_MAX_DPS)]


def test_rt_past_threshold_emits_arm_held(qtbot) -> None:
    events = [_FakeEvent(evdev.ecodes.EV_ABS, evdev.ecodes.ABS_RZ, 255)]
    worker, device = _make_worker_with_events(events)
    with qtbot.waitSignal(worker.arm_changed, timeout=1000) as blocker:
        worker._read_until_disconnected(device)
    assert blocker.args == [True]


def test_rt_released_emits_arm_not_held(qtbot) -> None:
    events = [_FakeEvent(evdev.ecodes.EV_ABS, evdev.ecodes.ABS_RZ, 0)]
    worker, device = _make_worker_with_events(events)
    with qtbot.waitSignal(worker.arm_changed, timeout=1000) as blocker:
        worker._read_until_disconnected(device)
    assert blocker.args == [False]


def test_button_a_emits_fire_changed(qtbot) -> None:
    events = [_FakeEvent(evdev.ecodes.EV_KEY, evdev.ecodes.BTN_SOUTH, 1)]
    worker, device = _make_worker_with_events(events)
    with qtbot.waitSignal(worker.fire_changed, timeout=1000) as blocker:
        worker._read_until_disconnected(device)
    assert blocker.args == [True]


def test_button_b_press_emits_estop_requested(qtbot) -> None:
    events = [_FakeEvent(evdev.ecodes.EV_KEY, evdev.ecodes.BTN_EAST, 1)]
    worker, device = _make_worker_with_events(events)
    with qtbot.waitSignal(worker.estop_requested, timeout=1000):
        worker._read_until_disconnected(device)


def test_button_b_release_does_not_emit_estop_requested() -> None:
    events = [_FakeEvent(evdev.ecodes.EV_KEY, evdev.ecodes.BTN_EAST, 0)]
    worker, device = _make_worker_with_events(events)
    received = []
    worker.estop_requested.connect(lambda: received.append(True))

    worker._read_until_disconnected(device)

    assert received == []


def test_disconnect_forces_jog_fire_and_arm_to_release(qtbot):
    worker = GamepadWorker()
    signals = []
    worker.jog_axis_changed.connect(lambda axis, speed: signals.append(("jog", axis, speed)))
    worker.fire_changed.connect(lambda held: signals.append(("fire", held)))
    worker.arm_changed.connect(lambda held: signals.append(("arm", held)))

    worker._force_release()

    assert ("jog", Axis.PAN, 0.0) in signals
    assert ("jog", Axis.TILT, 0.0) in signals
    assert ("fire", False) in signals
    assert ("arm", False) in signals
