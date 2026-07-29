from __future__ import annotations

from celikkubbe.core.clock import FakeClock, SystemClock


def test_fake_clock_starts_at_zero() -> None:
    clock = FakeClock()
    assert clock.now() == 0.0


def test_fake_clock_advances() -> None:
    clock = FakeClock()
    clock.advance(1.5)
    clock.advance(2.0)
    assert clock.now() == 3.5


def test_system_clock_returns_increasing_values() -> None:
    clock = SystemClock()
    first = clock.now()
    second = clock.now()
    assert second >= first
