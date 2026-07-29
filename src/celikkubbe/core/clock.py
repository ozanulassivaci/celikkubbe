"""Injectable time source. Nothing in core calls time.monotonic() directly.

Implements the ``Clock`` protocol from ``protocols.py`` structurally; no
inheritance is required since it is a ``typing.Protocol``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class SystemClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass
class FakeClock:
    _t: float = field(default=0.0)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds
