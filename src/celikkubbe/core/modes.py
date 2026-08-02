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
    dev_mode: bool = False,
    operator_skip_self_test: bool = False,
) -> tuple[Mode, list[Command]]:
    """``dev_mode`` is a field-testing escape hatch, never used at the
    competition: it does not touch anything outside this function (fire/
    arm gating in engagement.py's SafetyGate is untouched), and only
    relaxes the three checks that assume a physical turret exists --
    e-stop/driver-alarm-triggered SAFE entry and the homing gate. Link
    timeout and camera health still trip M4_SAFE regardless, since those
    are not about missing hardware, and SimTurretLink (the only link this
    GUI drives) provides both estop/driver-alarm telemetry and a working
    homing/zero mechanism on its own without any physical turret at all --
    dev_mode exists for testing the vision/detection pipeline without
    wanting to run through that setup first.
    """
    if mode in (Mode.M2_STANDBY, Mode.M3_OPERATIONAL):
        unsafe = _link_timed_out(telemetry, now) or not camera_healthy
        if not dev_mode:
            unsafe = unsafe or _estop_active(telemetry) or _driver_alarm(telemetry)
        if unsafe:
            return _enter_safe()

    if mode is Mode.M1_INIT:
        # Checked before self_test_result: the self-test still runs and
        # displays real pass/fail rows in dev_mode (see
        # PipelineWorker._advance_self_test, unconditional on dev_mode) --
        # this is purely a manual override for when it cannot naturally
        # pass without real hardware, not a bypass of running it at all.
        # (Booting straight into M2_STANDBY without ever entering M1_INIT
        # is handled by PipelineWorker's own initial state, not here.)
        if dev_mode and operator_skip_self_test:
            return Mode.M2_STANDBY, [SetMode(Mode.M2_STANDBY)]
        if self_test_result is None:
            return Mode.M1_INIT, []
        if self_test_result.passed:
            return Mode.M2_STANDBY, [SetMode(Mode.M2_STANDBY)]
        return _enter_safe()

    if mode is Mode.M2_STANDBY:
        homed = dev_mode or _homed(telemetry)
        if operator_requested_mode is Mode.M3_OPERATIONAL and homed:
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
