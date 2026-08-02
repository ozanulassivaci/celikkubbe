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


def test_m2_advances_to_m3_on_operator_go_when_homed() -> None:
    telemetry = make_telemetry(t=0.0, homed_pan=True, homed_tilt=True)
    mode, commands = modes.step(
        Mode.M2_STANDBY, telemetry, None, True, Mode.M3_OPERATIONAL, False, now=0.0
    )
    assert mode is Mode.M3_OPERATIONAL
    assert Arm() in commands


def test_m2_blocked_from_m3_when_not_homed() -> None:
    # docs/protocol.md section 5: no limit switches exist, so the PC must
    # refuse OPERATIONAL until the operator has zeroed both axes.
    telemetry = make_telemetry(t=0.0, homed_pan=False, homed_tilt=False)
    mode, commands = modes.step(
        Mode.M2_STANDBY, telemetry, None, True, Mode.M3_OPERATIONAL, False, now=0.0
    )
    assert mode is Mode.M2_STANDBY
    assert commands == []


def test_m2_blocked_from_m3_when_only_one_axis_homed() -> None:
    telemetry = make_telemetry(t=0.0, homed_pan=True, homed_tilt=False)
    mode, commands = modes.step(
        Mode.M2_STANDBY, telemetry, None, True, Mode.M3_OPERATIONAL, False, now=0.0
    )
    assert mode is Mode.M2_STANDBY
    assert commands == []


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


# --- dev_mode: field-testing escape hatch, see modes.step's own docstring ---


def test_dev_mode_ignores_estop_for_mode_transitions() -> None:
    telemetry = make_telemetry(t=0.0, estop=True)
    mode, _ = modes.step(
        Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=0.0, dev_mode=True
    )
    assert mode is Mode.M3_OPERATIONAL


def test_dev_mode_ignores_driver_alarm_for_mode_transitions() -> None:
    telemetry = make_telemetry(t=0.0, driver_alarm_pan=True)
    mode, _ = modes.step(
        Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=0.0, dev_mode=True
    )
    assert mode is Mode.M3_OPERATIONAL


def test_dev_mode_still_trips_m4_on_link_timeout() -> None:
    # Only e-stop/driver-alarm/homing are relaxed -- link timeout and
    # camera health are not about missing turret hardware, so dev_mode
    # must not silently mask a genuinely dead link.
    telemetry = make_telemetry(t=0.0)
    now = config.TELEMETRY_STALE_MS / 1000.0 + 0.01
    mode, _ = modes.step(
        Mode.M3_OPERATIONAL, telemetry, None, True, None, False, now=now, dev_mode=True
    )
    assert mode is Mode.M4_SAFE


def test_dev_mode_still_trips_m4_on_camera_unhealthy() -> None:
    telemetry = make_telemetry(t=0.0)
    mode, _ = modes.step(
        Mode.M3_OPERATIONAL, telemetry, None, False, None, False, now=0.0, dev_mode=True
    )
    assert mode is Mode.M4_SAFE


def test_dev_mode_treats_homing_as_satisfied() -> None:
    telemetry = make_telemetry(t=0.0, homed_pan=False, homed_tilt=False)
    mode, commands = modes.step(
        Mode.M2_STANDBY,
        telemetry,
        None,
        True,
        Mode.M3_OPERATIONAL,
        False,
        now=0.0,
        dev_mode=True,
    )
    assert mode is Mode.M3_OPERATIONAL
    assert Arm() in commands


def test_dev_mode_without_skip_request_still_gates_on_self_test() -> None:
    mode, commands = modes.step(Mode.M1_INIT, None, None, True, None, False, now=0.0, dev_mode=True)
    assert mode is Mode.M1_INIT
    assert commands == []


def test_dev_mode_skip_self_test_moves_to_m2_regardless_of_result() -> None:
    failing = SelfTestResult(items=(SelfTestItem("camera", False, "no signal", None),))
    mode, commands = modes.step(
        Mode.M1_INIT,
        None,
        failing,
        True,
        None,
        False,
        now=0.0,
        dev_mode=True,
        operator_skip_self_test=True,
    )
    assert mode is Mode.M2_STANDBY
    assert SetMode(Mode.M2_STANDBY) in commands


def test_skip_self_test_ignored_outside_dev_mode() -> None:
    failing = SelfTestResult(items=(SelfTestItem("camera", False, "no signal", None),))
    mode, _ = modes.step(
        Mode.M1_INIT,
        None,
        failing,
        True,
        None,
        False,
        now=0.0,
        dev_mode=False,
        operator_skip_self_test=True,
    )
    assert mode is Mode.M4_SAFE
