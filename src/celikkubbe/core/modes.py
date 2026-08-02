"""M1-M4 mode machine.

Pure function: given the current mode and a snapshot of the world, returns
the next mode and the Commands an outer layer should send. M4 SAFE never
exits automatically; only an explicit operator acknowledgement moves the
system back to M1 INIT for a fresh self-test.
"""

from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.commands import Arm, Command, Disarm, SetMode, SoftEstop
from celikkubbe.core.types import Mode, SelfTestResult, Telemetry


def _link_timed_out(telemetry: Telemetry | None, now: float) -> bool:
    # TELEMETRY_STALE_MS (300ms), not WATCHDOG_TIMEOUT_MS (200ms): the 200ms
    # figure is the MCU's own budget for safing itself. Using it here would
    # race the MCU and drop to M4 on ordinary PC-side scheduling jitter; the
    # PC should notice slightly later than the MCU acts, not simultaneously.
    if telemetry is None:
        return True
    return (now - telemetry.t) * 1000.0 > config.TELEMETRY_STALE_MS


def _driver_alarm(telemetry: Telemetry | None) -> bool:
    return telemetry is not None and (telemetry.driver_alarm_pan or telemetry.driver_alarm_tilt)


def _estop_active(telemetry: Telemetry | None) -> bool:
    return telemetry is not None and telemetry.estop


def _homed(telemetry: Telemetry | None) -> bool:
    # docs/protocol.md section 5: no limit switches exist, so the PC must
    # refuse OPERATIONAL until the operator has zeroed both axes by eye.
    return telemetry is not None and telemetry.homed_pan and telemetry.homed_tilt


def _enter_safe() -> tuple[Mode, list[Command]]:
    return Mode.M4_SAFE, [SoftEstop(), Disarm(), SetMode(Mode.M4_SAFE)]


def step(
    mode: Mode,
    telemetry: Telemetry | None,
    self_test_result: SelfTestResult | None,
    camera_healthy: bool,
    operator_requested_mode: Mode | None,
    operator_ack_fault: bool,
    now: float,
) -> tuple[Mode, list[Command]]:
    if mode in (Mode.M2_STANDBY, Mode.M3_OPERATIONAL):
        if (
            _estop_active(telemetry)
            or _link_timed_out(telemetry, now)
            or _driver_alarm(telemetry)
            or not camera_healthy
        ):
            return _enter_safe()

    if mode is Mode.M1_INIT:
        if self_test_result is None:
            return Mode.M1_INIT, []
        if self_test_result.passed:
            return Mode.M2_STANDBY, [SetMode(Mode.M2_STANDBY)]
        return _enter_safe()

    if mode is Mode.M2_STANDBY:
        if operator_requested_mode is Mode.M3_OPERATIONAL and _homed(telemetry):
            return Mode.M3_OPERATIONAL, [SetMode(Mode.M3_OPERATIONAL), Arm()]
        return Mode.M2_STANDBY, []

    if mode is Mode.M3_OPERATIONAL:
        if operator_requested_mode is Mode.M2_STANDBY:
            return Mode.M2_STANDBY, [SetMode(Mode.M2_STANDBY), Disarm()]
        return Mode.M3_OPERATIONAL, []

    if mode is Mode.M4_SAFE:
        if operator_ack_fault:
            return Mode.M1_INIT, [SetMode(Mode.M1_INIT)]
        return Mode.M4_SAFE, []

    raise AssertionError(f"unreachable mode {mode}")
