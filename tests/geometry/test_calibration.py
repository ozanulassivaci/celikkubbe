from __future__ import annotations

import pytest

from celikkubbe.geometry.calibration import BoresightCorrection, BoresightTable


def test_empty_boresight_table_yields_zero_correction() -> None:
    table = BoresightTable()
    assert table.correction_at(10.0) == (0.0, 0.0)


def test_single_correction_applies_at_every_range() -> None:
    table = BoresightTable((BoresightCorrection(0.5, -0.3, "2026-01-01", 10.0, "single point"),))
    assert table.correction_at(1.0) == (0.5, -0.3)
    assert table.correction_at(10.0) == (0.5, -0.3)
    assert table.correction_at(30.0) == (0.5, -0.3)


def test_interpolation_between_range_indexed_corrections() -> None:
    table = BoresightTable(
        (
            BoresightCorrection(0.0, 0.0, "2026-01-01", 5.0, "near"),
            BoresightCorrection(1.0, 0.4, "2026-01-01", 15.0, "far"),
        )
    )
    az, el = table.correction_at(10.0)  # halfway between 5m and 15m
    assert az == pytest.approx(0.5)
    assert el == pytest.approx(0.2)


def test_interpolation_is_order_independent() -> None:
    # Constructor order shouldn't matter -- correction_at sorts by range.
    table = BoresightTable(
        (
            BoresightCorrection(1.0, 0.4, "2026-01-01", 15.0, "far"),
            BoresightCorrection(0.0, 0.0, "2026-01-01", 5.0, "near"),
        )
    )
    az, el = table.correction_at(10.0)
    assert az == pytest.approx(0.5)
    assert el == pytest.approx(0.2)


def test_correction_beyond_span_clamps_to_nearest() -> None:
    table = BoresightTable(
        (
            BoresightCorrection(0.0, 0.0, "2026-01-01", 5.0, "near"),
            BoresightCorrection(1.0, 0.4, "2026-01-01", 15.0, "far"),
        )
    )
    assert table.correction_at(1.0) == (0.0, 0.0)
    assert table.correction_at(30.0) == (1.0, 0.4)
