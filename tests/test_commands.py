from __future__ import annotations

import dataclasses

from celikkubbe.core.commands import Arm, Disarm, Fire, Goto, SetMode, SoftEstop, Stop
from celikkubbe.core.types import Mode


def test_commands_are_frozen() -> None:
    cmd = SetMode(mode=Mode.M2_STANDBY)
    assert dataclasses.is_dataclass(cmd)
    try:
        cmd.mode = Mode.M4_SAFE  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Command must be immutable")


def test_goto_holds_all_fields() -> None:
    cmd = Goto(az_deg=10.0, el_deg=5.0, max_vel_dps=30.0, max_accel_dps2=60.0)
    assert cmd.az_deg == 10.0
    assert cmd.max_accel_dps2 == 60.0


def test_fire_and_estop_are_distinct_types() -> None:
    assert Fire(count=1) != SoftEstop()
    assert Arm() != Disarm()


def test_stop_is_frozen_and_has_no_fields() -> None:
    cmd = Stop()
    assert dataclasses.is_dataclass(cmd)
    assert cmd == Stop()
    assert not dataclasses.fields(cmd)
