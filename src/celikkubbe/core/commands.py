"""Command objects returned by the state machines.

Pure data, no behaviour: nothing in ``core`` executes a ``Command``. An
outer layer (serial link, simulator, ...) is responsible for that.
"""

from __future__ import annotations

from dataclasses import dataclass

from celikkubbe.core.types import Axis, Mode


@dataclass(frozen=True)
class SetMode:
    mode: Mode


@dataclass(frozen=True)
class Goto:
    az_deg: float
    el_deg: float
    max_vel_dps: float
    max_accel_dps2: float


@dataclass(frozen=True)
class Jog:
    axis: Axis
    direction: int
    speed_dps: float


@dataclass(frozen=True)
class Home:
    axes: tuple[Axis, ...]


@dataclass(frozen=True)
class Arm:
    pass


@dataclass(frozen=True)
class Disarm:
    pass


@dataclass(frozen=True)
class Fire:
    count: int


@dataclass(frozen=True)
class SoftEstop:
    pass


@dataclass(frozen=True)
class SetParam:
    param_id: str
    value: float


@dataclass(frozen=True)
class Zero:
    """Declare the current position of one axis to be ``value_deg``.

    The only homing mechanism the electronics actually have (see
    docs/protocol.md section 5): no limit switches exist, so the operator
    centres the turret by eye and the GUI sends this per axis. ``Home``
    (below) corresponds to the protocol's ``home`` command, reserved until
    limit switches exist.
    """

    axis: Axis
    value_deg: float


Command = SetMode | Goto | Jog | Home | Zero | Arm | Disarm | Fire | SoftEstop | SetParam
