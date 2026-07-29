"""Interfaces implemented by outer layers (I/O, simulation, ...).

``core`` only depends on these Protocols, never on a concrete camera,
serial, or clock implementation.
"""

from __future__ import annotations

from typing import Protocol

from celikkubbe.core.commands import Command
from celikkubbe.core.types import CameraIntrinsics, Frame, Telemetry


class Clock(Protocol):
    def now(self) -> float: ...


class FrameSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> Frame | None: ...

    def stop(self) -> None: ...

    @property
    def has_depth(self) -> bool: ...

    @property
    def intrinsics(self) -> CameraIntrinsics: ...


class TurretLink(Protocol):
    def send(self, cmd: Command) -> None: ...

    def poll(self) -> Telemetry | None: ...

    @property
    def connected(self) -> bool: ...
