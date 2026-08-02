"""GamepadWorker: Logitech F310 (or another XInput-compatible pad) via evdev.

Runs on its own thread: evdev's ``read_loop()`` blocks, and the GUI
thread must never block on hardware I/O (the same constraint
pipeline_worker.py's own module docstring states for perception/serial
I/O). Left stick drives proportional jog, RT is the hold-to-arm dead-man
switch, A fires, B is a direct soft-estop.

Targets the F310's XInput mode (its back switch set to "X"), read here
through the ``xpad``-style mapping most Linux systems expose it under:
BTN_SOUTH (A), BTN_EAST (B), ABS_X/ABS_Y (left stick), ABS_RZ (RT,
analog). No physical device exists in this development environment, so
this mapping is derived from the F310's published XInput layout, not
verified against real hardware -- the same honestly-flagged gap as
io/serial_link.py's own "never exercised against real MCU firmware."

No device present is not an error: this hardware is optional, and most
development happens without one plugged in. ``connected_changed(False)``
-- emitted both when no device is ever found and when a previously-open
one disconnects mid-session -- is Stage 1's own cue to treat a dropped
gamepad exactly like RT being released: this worker forces its own
arm/fire state to False on either path, never leaving a stale "still
armed" reading from a controller no longer there to report otherwise.
"""

from __future__ import annotations

import math
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from celikkubbe.core import config
from celikkubbe.core.types import Axis

try:
    import evdev
except ImportError:  # evdev is Linux-only; this hardware is optional everywhere
    evdev = None  # type: ignore[assignment]

# xpad reports the F310 (XInput mode) under a name containing one of
# these, case-insensitively -- "Xbox" because the xpad driver frequently
# exposes third-party XInput pads under a generic Xbox-compatible name.
DEFAULT_NAME_HINTS = ("f310", "gamepad f310", "xbox")
POLL_RETRY_S = 2.0
STICK_DEADZONE = 0.15
# RT past this normalised depth counts as held -- an analog trigger
# never reports a clean 0/1, and a hair-trigger threshold would make
# "held" flicker on noise near a resting position.
TRIGGER_HELD_THRESHOLD = 0.5


def find_gamepad_device_path(name_hints: tuple[str, ...] = DEFAULT_NAME_HINTS) -> str | None:
    """First evdev device whose reported name contains one of the hints,
    case-insensitively -- a pure-ish function over evdev.list_devices()
    so a test can monkeypatch that one call rather than the whole module.
    Returns None (never raises) when evdev is unavailable or no matching
    device is present, both ordinary conditions in this environment.
    """
    if evdev is None:
        return None
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
        except OSError:
            continue
        if any(hint in device.name.lower() for hint in name_hints):
            return path
    return None


def scale_stick_to_speed_dps(
    value: float,
    deadzone: float = STICK_DEADZONE,
    max_speed_dps: float = config.JOG_SPEED_MAX_DPS,
) -> float:
    """value: normalised stick position in [-1, 1]. Zero within the
    deadzone (a resting stick reporting a small nonzero value must never
    dribble the turret); scales linearly from the deadzone edge to
    max_speed_dps at full deflection -- proportional, unlike the GUI's
    own fixed-speed directional pad buttons.
    """
    magnitude = min(1.0, abs(value))
    if magnitude < deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(scaled * max_speed_dps, value)


def normalize_abs_value(raw: int, info_min: int, info_max: int) -> float:
    """Maps a raw ABS event value to [-1, 1] using the axis's own
    reported min/max -- never a hardcoded range, since sticks and
    triggers vary by device (some report -32768..32767, others 0..255).
    """
    span = info_max - info_min
    if span <= 0:
        return 0.0
    centered = raw - (info_min + info_max) / 2.0
    return max(-1.0, min(1.0, centered / (span / 2.0)))


class GamepadWorker(QThread):
    jog_axis_changed = pyqtSignal(object, float)  # Axis, speed_dps (0.0 = centred/stop this axis)
    fire_changed = pyqtSignal(bool)
    arm_changed = pyqtSignal(bool)
    estop_requested = pyqtSignal()
    connected_changed = pyqtSignal(bool)

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            device = self._open_device()
            if device is None:
                self.connected_changed.emit(False)
                self._stop_event.wait(POLL_RETRY_S)
                continue
            self.connected_changed.emit(True)
            self._read_until_disconnected(device)
            self._force_release()
            self.connected_changed.emit(False)

    def _open_device(self):
        if evdev is None:
            return None
        path = find_gamepad_device_path()
        if path is None:
            return None
        try:
            return evdev.InputDevice(path)
        except OSError:
            return None

    def _force_release(self) -> None:
        """Called whenever a device goes away (never found, or dropped
        mid-session) -- see the module docstring's dead-man reasoning.
        """
        self.jog_axis_changed.emit(Axis.PAN, 0.0)
        self.jog_axis_changed.emit(Axis.TILT, 0.0)
        self.fire_changed.emit(False)
        self.arm_changed.emit(False)

    def _read_until_disconnected(self, device) -> None:
        abs_ranges: dict[int, tuple[int, int]] = {}
        caps = device.capabilities(absinfo=True).get(evdev.ecodes.EV_ABS, [])
        for code, info in caps:
            abs_ranges[code] = (info.min, info.max)

        try:
            for event in device.read_loop():
                if self._stop_event.is_set():
                    return
                if event.type == evdev.ecodes.EV_ABS:
                    self._handle_abs(event, abs_ranges)
                elif event.type == evdev.ecodes.EV_KEY:
                    self._handle_key(event)
        except OSError:
            # Unplugged mid-read; the caller's own _force_release and
            # connected_changed(False) after this call covers it.
            return

    def _handle_abs(self, event, abs_ranges: dict[int, tuple[int, int]]) -> None:
        info = abs_ranges.get(event.code)
        if info is None:
            return
        value = normalize_abs_value(event.value, info[0], info[1])
        if event.code == evdev.ecodes.ABS_X:
            self.jog_axis_changed.emit(Axis.PAN, scale_stick_to_speed_dps(value))
        elif event.code == evdev.ecodes.ABS_Y:
            # Stick-forward (negative Y, matching the camera's own Y-down
            # convention -- see geometry/frames.py) should raise the
            # barrel: tilt positive-up, so this is inverted relative to
            # the raw stick reading.
            self.jog_axis_changed.emit(Axis.TILT, scale_stick_to_speed_dps(-value))
        elif event.code == evdev.ecodes.ABS_RZ:
            self.arm_changed.emit(value > TRIGGER_HELD_THRESHOLD)

    def _handle_key(self, event) -> None:
        held = bool(event.value)
        if event.code == evdev.ecodes.BTN_SOUTH:
            self.fire_changed.emit(held)
        elif event.code == evdev.ecodes.BTN_EAST and held:
            self.estop_requested.emit()
