from __future__ import annotations

import pytest

from celikkubbe.core import config
from celikkubbe.core.clock import FakeClock
from celikkubbe.core.commands import (
    Arm,
    Fire,
    Goto,
    Home,
    Jog,
    SetMode,
    SetParam,
    SetVelocity,
    SoftEstop,
    Zero,
)
from celikkubbe.core.types import Axis, Mode
from celikkubbe.io.codec import AckResult, EventId
from celikkubbe.io.sim_link import SimTurretLink, _Backlash


def _run(link: SimTurretLink, clock: FakeClock, seconds: float, step: float = 0.02):
    """Advance the clock in small steps, sending a heartbeat each step --
    a real PC never lets 200ms pass without one, and a bare clock.advance()
    of more than that trips the sim's own watchdog exactly as it should.
    """
    telem = None
    steps = max(1, round(seconds / step))
    for _ in range(steps):
        clock.advance(step)
        link.send_heartbeat()
        telem = link.poll()
    return telem


def test_trapezoidal_profile_reaches_target_and_completes() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Goto(100.0, 0.0, 10.0, 10.0))

    telem = _run(link, clock, 5.0)
    assert telem.motion_complete is False
    assert 0.0 < telem.pan_deg < 100.0

    # total 11s: t_acc=1s, d_acc=5deg, cruise 90deg/10dps=9s -> 11s
    telem = _run(link, clock, 6.0)
    assert telem.motion_complete is True
    assert telem.pan_deg == pytest.approx(100.0)
    assert telem.pan_vel_dps == 0.0


def test_backlash_lags_on_direction_reversal() -> None:
    backlash = _Backlash(gap_deg=0.1)
    backlash.drive_to(1.0)
    assert backlash.true_deg == pytest.approx(1.0)

    backlash.drive_to(0.95)  # reverses direction, travel (0.05) < gap -> fully absorbed
    assert backlash.true_deg == pytest.approx(1.0)

    backlash.drive_to(0.80)  # same direction: 0.05 finishes taking up slack, 0.10 moves output
    assert backlash.true_deg == pytest.approx(0.90)


def test_unidirectional_approach_compensation_removes_reversal_lag() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)

    # Settle pan at +10 deg moving in the consistent (+) direction.
    link.send(Goto(10.0, 0.0, 50.0, 50.0))
    _run(link, clock, 5.0)
    assert link.true_pan_deg == pytest.approx(10.0)

    # A small move in the - direction, shorter than the 0.1 deg backlash
    # gap -- uncompensated, this is exactly the case that leaves lag (see
    # test_backlash_lags_on_direction_reversal). BACKOFF_DEG (0.5) exceeds
    # the gap, so the firmware's overshoot-and-return guarantees the final
    # leg's travel alone exceeds the gap, cancelling it out.
    link.send(Goto(9.95, 0.0, 50.0, 50.0))
    telem = _run(link, clock, 5.0)

    assert telem.motion_complete is True
    assert telem.pan_deg == pytest.approx(9.95)
    assert link.true_pan_deg == pytest.approx(9.95)


def test_estop_clears_position_valid_and_tilt_droops() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Goto(0.0, 30.0, 50.0, 50.0))
    settled = _run(link, clock, 5.0)
    assert settled.tilt_deg == pytest.approx(30.0)

    link.inject_estop()
    telem = _run(link, clock, 1.0)

    assert telem.estop is True
    assert telem.position_valid is False
    assert telem.tilt_deg < 30.0  # drooped toward TILT_LIMIT_DEG's lower bound


def test_estop_droop_clamps_at_tilt_lower_limit() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.inject_estop()
    clock.advance(1000.0)
    telem = link.poll()
    assert telem.tilt_deg == config.TILT_LIMIT_DEG[0]


def test_fire_rejected_when_not_armed() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Fire(count=1))
    assert link.last_ack is AckResult.NOT_ARMED
    assert link.ammo_fired == 0


def test_fire_rejected_when_estopped() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Arm())
    link.inject_estop()
    link.send(Fire(count=1))
    assert link.last_ack is AckResult.ESTOP_ACTIVE
    assert link.ammo_fired == 0


def test_fire_rejected_when_position_invalid_via_driver_alarm() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Arm())
    link.inject_driver_alarm(Axis.PAN)
    assert link.poll().position_valid is False
    link.send(Fire(count=1))
    assert link.last_ack is AckResult.DRIVER_ALARM
    assert link.ammo_fired == 0


def test_fire_accepted_when_safe_and_armed() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Arm())
    link.send(Fire(count=3))
    assert link.last_ack is AckResult.OK
    assert link.ammo_fired == 3
    events = link.drain_events()
    assert any(e[0] is EventId.SHOT_FIRED for e in events)


def test_aim_beyond_software_limits_rejected() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    lo, hi = config.PAN_LIMIT_DEG
    link.send(Goto(hi + 10.0, 0.0, 10.0, 10.0))
    assert link.last_ack is AckResult.LIMIT_EXCEEDED
    telem = link.poll()
    assert telem.pan_deg == 0.0  # rejected -- no motion started

    link.send(Goto(lo - 10.0, 0.0, 10.0, 10.0))
    assert link.last_ack is AckResult.LIMIT_EXCEEDED


def test_watchdog_trips_200ms_after_contact_stops() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Arm())
    assert link.poll().armed is True

    clock.advance(0.199)
    assert link.watchdog_tripped is False

    clock.advance(0.002)  # total 201ms since last contact
    telem = link.poll()
    assert link.watchdog_tripped is True
    assert telem.armed is False


def test_heartbeat_resets_the_watchdog() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Arm())
    clock.advance(0.15)
    link.send_heartbeat()
    clock.advance(0.15)
    link.poll()
    assert link.watchdog_tripped is False


def test_injected_crc_errors_are_counted_and_recovered_from() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.inject_crc_errors(1.0)  # always corrupt

    assert link.poll() is None
    assert link.crc_error_count == 1

    link.inject_crc_errors(0.0)  # recovers
    telem = link.poll()
    assert telem is not None
    assert link.crc_error_count == 1  # no further errors once the fault clears


def test_injected_link_dropout_stops_telemetry() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.inject_link_dropout(0.3)

    assert link.poll() is None
    clock.advance(0.2)
    assert link.poll() is None
    clock.advance(0.2)  # total 0.4s > 0.3s dropout window
    assert link.poll() is not None


def test_injected_latency_delays_command_effect() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.set_latency(500.0)

    link.send(Goto(50.0, 0.0, 100.0, 100.0))
    telem = _run(link, clock, 0.1)
    assert telem.pan_deg == 0.0  # command hasn't been applied yet

    telem = _run(link, clock, 0.5)  # past the 500ms latency
    assert telem.pan_deg > 0.0  # now moving


def test_jog_moves_continuously_toward_the_limit() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Jog(Axis.TILT, 1, 5.0))
    telem = _run(link, clock, 1.0)
    assert telem.tilt_deg == pytest.approx(5.0, abs=0.5)


def test_zero_declares_current_position_and_resets_backlash() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Goto(10.0, 0.0, 50.0, 50.0))
    clock.advance(5.0)
    link.poll()

    link.send(Zero(Axis.PAN, 0.0))
    telem = link.poll()
    assert telem.pan_deg == 0.0
    assert link.true_pan_deg == 0.0


def test_soft_estop_command_disarms_and_trips_estop() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Arm())
    link.send(SoftEstop())
    telem = link.poll()
    assert telem.armed is False
    assert telem.estop is True


def test_release_estop_clears_the_flag() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.inject_estop()
    link.release_estop()
    telem = link.poll()
    assert telem.estop is False


def test_arm_rejected_during_estop() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.inject_estop()
    link.send(Arm())
    assert link.last_ack is AckResult.ESTOP_ACTIVE
    assert link.poll().armed is False


def test_zero_tilt_axis() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(Goto(0.0, 20.0, 50.0, 50.0))
    _run(link, clock, 2.0)

    link.send(Zero(Axis.TILT, 0.0))
    telem = link.poll()
    assert telem.tilt_deg == 0.0
    assert link.true_tilt_deg == 0.0


def test_clear_driver_alarm() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.inject_driver_alarm(Axis.TILT)
    assert link.poll().driver_alarm_tilt is True
    link.clear_driver_alarm(Axis.TILT)
    assert link.poll().driver_alarm_tilt is False


def test_set_mode_and_set_param_are_accepted_without_modelled_behaviour() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(SetMode(Mode.M3_OPERATIONAL))
    assert link.last_ack is AckResult.OK
    link.send(SetParam("trigger_pulse_ms", 60.0))
    assert link.last_ack is AckResult.OK
    link.send(Home((Axis.PAN,)))
    assert link.last_ack is AckResult.OK


def test_set_velocity_has_no_wire_equivalent_in_sim_either() -> None:
    clock = FakeClock()
    link = SimTurretLink(clock)
    link.send(SetVelocity(1.0, 2.0))
    assert link.last_ack is AckResult.UNKNOWN_COMMAND
