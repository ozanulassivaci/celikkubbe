from __future__ import annotations

from celikkubbe.core import config, modes
from celikkubbe.core.commands import Arm, Disarm, SetMode, SoftEstop
from celikkubbe.core.types import Mode, SelfTestItem, SelfTestResult

from .factories import make_telemetry


def test_m1_stays_until_self_test_reported() -> None:
    mode, commands = modes.step(Mode.M1_INIT, None, None, True, None, False, now=0.0)
    assert mode is Mode.M1_INIT
    assert commands == []


def test_m1_advances_to_m2_when_self_test_passes() -> None:
    result = SelfTestResult(items=(SelfTestItem("camera", True, None, 30.0),))
    mode, commands = modes.step(Mode.M1_INIT, None, result, True, None, False, now=0.0)
    assert mode is Mode.M2_STANDBY
    assert SetMode(Mode.M2_STANDBY) in commands


def test_m1_drops_to_m4_when_self_test_fails() -> None:
    result = SelfTestResult(items=(SelfTestItem("camera", False, "no signal", None),))
    mode, commands = modes.step(Mode.M1_INIT, None, result, True, None, False, now=0.0)
    assert mode is Mode.M4_SAFE
    assert SoftEstop() in commands
    assert Disarm() in commands


def test_m2_advances_to_m3_on_operator_go() -> None:
    telemetry = make_telemetry(t=0.0)
    mode, commands = modes.step(
        Mode.M2_STANDBY, telemetry, None, True, Mode.M3_OPERATIONAL, False, now=0.0
    )
    assert mode is Mode.M3_OPERATIONAL
    assert Arm() in commands


def test_m2_stays_without_operator_request() -> None:
    telemetry = make_telemetry(t=0.0)
    mode, commands = modes.step(Mode.M2_STANDBY, telemetry, None, True, None, False, now=0.0)
    assert mode is Mode.M2_STANDBY
    assert commands == []


def test_m3_returns_to_m2_when_operator_requests_standby() -> None:
    telemetry = make_telemetry(t=0.0)
    mode, commands = modes.step(
        Mode.M3_OPERATIONAL, telemetry, None, True, Mode.M2_STANDBY, False, now=0.0
    )
    assert mode is Mode.M2_STANDBY
    assert Disarm() in commands


def test_estop_drives_m4_from_m3() -> None:
    telemetry = make_telemetry(t=0.0, estop=True)
    mode, commands = modes.step(Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=0.0)
    assert mode is Mode.M4_SAFE
    assert SoftEstop() in commands


def test_watchdog_timeout_drives_m4() -> None:
    telemetry = make_telemetry(t=0.0)
    now = config.TELEMETRY_STALE_MS / 1000.0 + 0.01
    mode, _ = modes.step(Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=now)
    assert mode is Mode.M4_SAFE


def test_fresh_telemetry_does_not_trigger_watchdog() -> None:
    telemetry = make_telemetry(t=0.0)
    now = config.TELEMETRY_STALE_MS / 1000.0 - 0.01
    mode, _ = modes.step(Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=now)
    assert mode is Mode.M3_OPERATIONAL


def test_watchdog_uses_telemetry_stale_not_mcu_watchdog_budget() -> None:
    # The MCU's own 200ms safing budget (WATCHDOG_TIMEOUT_MS) must not drive
    # this transition, or the PC races the MCU and drops to M4 on ordinary
    # scheduling jitter between 200ms and 300ms.
    telemetry = make_telemetry(t=0.0)
    now = config.WATCHDOG_TIMEOUT_MS / 1000.0 + 0.01
    mode, _ = modes.step(Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=now)
    assert mode is Mode.M3_OPERATIONAL


def test_missing_telemetry_drives_m4_from_m2() -> None:
    mode, _ = modes.step(Mode.M2_STANDBY, None, None, True, None, False, now=0.0)
    assert mode is Mode.M4_SAFE


def test_driver_alarm_drives_m4() -> None:
    telemetry = make_telemetry(t=0.0, driver_alarm_pan=True)
    mode, _ = modes.step(Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=0.0)
    assert mode is Mode.M4_SAFE


def test_camera_unhealthy_drives_m4() -> None:
    telemetry = make_telemetry(t=0.0)
    mode, _ = modes.step(Mode.M3_OPERATIONAL, telemetry, None, False, None, False, now=0.0)
    assert mode is Mode.M4_SAFE


def test_m4_never_exits_automatically() -> None:
    telemetry = make_telemetry(t=0.0)
    mode = Mode.M4_SAFE
    for _ in range(5):
        mode, commands = modes.step(mode, telemetry, None, True, Mode.M1_INIT, False, now=0.0)
        assert mode is Mode.M4_SAFE
        assert commands == []


def test_m4_exits_only_on_operator_acknowledgement() -> None:
    mode, commands = modes.step(Mode.M4_SAFE, None, None, False, None, True, now=0.0)
    assert mode is Mode.M1_INIT
    assert SetMode(Mode.M1_INIT) in commands
