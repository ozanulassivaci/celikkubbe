from __future__ import annotations

import math

import pytest

from celikkubbe.geometry.ballistics import (
    DEFAULT_BALLISTIC_TABLE,
    BallisticTable,
    angular_velocity_dps,
    gravity_only_drop_deg,
    lead_angle,
)
from celikkubbe.vision.sources import estimated_intrinsics

_INTR = estimated_intrinsics(640, 480)


def test_zero_table_produces_zero_correction() -> None:
    table = BallisticTable(ranges_m=(5.0, 15.0), drop_deg=(0.0, 0.0), muzzle_velocity_ms=100.0)
    for r in (1.0, 5.0, 10.0, 15.0, 20.0):
        drop, _ = table.interpolate_drop_deg(r)
        assert drop == 0.0


def test_interpolation_between_table_entries() -> None:
    table = BallisticTable(
        ranges_m=(5.0, 10.0, 15.0), drop_deg=(0.10, 0.30, 0.50), muzzle_velocity_ms=100.0
    )
    drop, clamped = table.interpolate_drop_deg(7.5)
    assert drop == pytest.approx(0.20)  # halfway between 0.10 and 0.30
    assert clamped is False

    drop, clamped = table.interpolate_drop_deg(12.5)
    assert drop == pytest.approx(0.40)  # halfway between 0.30 and 0.50
    assert clamped is False


def test_interpolation_at_exact_table_entries() -> None:
    table = BallisticTable(
        ranges_m=(5.0, 10.0, 15.0), drop_deg=(0.10, 0.30, 0.50), muzzle_velocity_ms=100.0
    )
    assert table.interpolate_drop_deg(10.0) == (0.30, False)


def test_extrapolation_beyond_span_clamps_and_warns() -> None:
    table = BallisticTable(
        ranges_m=(5.0, 10.0, 15.0), drop_deg=(0.10, 0.30, 0.50), muzzle_velocity_ms=100.0
    )
    drop, clamped = table.interpolate_drop_deg(20.0)
    assert drop == 0.50
    assert clamped is True

    drop, clamped = table.interpolate_drop_deg(1.0)
    assert drop == 0.10
    assert clamped is True


def test_time_of_flight_is_range_over_muzzle_velocity() -> None:
    table = BallisticTable(ranges_m=(15.0,), drop_deg=(0.0,), muzzle_velocity_ms=100.0)
    assert table.time_of_flight(15.0) == pytest.approx(0.15)


def test_gravity_only_drop_is_zero_for_nonpositive_range_or_velocity() -> None:
    assert gravity_only_drop_deg(0.0, 100.0) == 0.0
    assert gravity_only_drop_deg(15.0, 0.0) == 0.0
    assert gravity_only_drop_deg(-5.0, 100.0) == 0.0


def test_time_of_flight_is_zero_for_nonpositive_muzzle_velocity() -> None:
    table = BallisticTable(ranges_m=(15.0,), drop_deg=(0.0,), muzzle_velocity_ms=0.0)
    assert table.time_of_flight(15.0) == 0.0


def test_gravity_only_drop_matches_the_worked_example_at_15m() -> None:
    # docs-level worked example: 100 m/s muzzle velocity, 15 m range ->
    # ~0.15s flight, ~11cm drop, ~0.42 deg.
    drop_deg = gravity_only_drop_deg(15.0, 100.0)
    assert drop_deg == pytest.approx(0.4215, abs=1e-3)


def test_default_ballistic_table_is_gravity_only_and_covers_5_10_15() -> None:
    assert DEFAULT_BALLISTIC_TABLE.ranges_m == (5.0, 10.0, 15.0)
    assert DEFAULT_BALLISTIC_TABLE.muzzle_velocity_ms == pytest.approx(100.0)
    for r, drop in zip(
        DEFAULT_BALLISTIC_TABLE.ranges_m, DEFAULT_BALLISTIC_TABLE.drop_deg, strict=False
    ):
        assert drop == pytest.approx(gravity_only_drop_deg(r, 100.0))


def test_stationary_target_produces_zero_lead() -> None:
    assert lead_angle((0.0, 0.0), time_of_flight_s=0.2) == (0.0, 0.0)


def test_lead_scales_linearly_with_time_of_flight() -> None:
    lead_short = lead_angle((10.0, -5.0), time_of_flight_s=0.1)
    lead_long = lead_angle((10.0, -5.0), time_of_flight_s=0.2)
    assert lead_long[0] == pytest.approx(2 * lead_short[0])
    assert lead_long[1] == pytest.approx(2 * lead_short[1])


def test_angular_velocity_identical_normalised_velocity_differs_centre_vs_edge() -> None:
    centre = (_INTR.cx / _INTR.width, _INTR.cy / _INTR.height)
    edge = (0.95, _INTR.cy / _INTR.height)  # near the right edge of the frame
    velocity_norm = (0.1, 0.0)  # identical apparent rightward drift

    az_centre, _ = angular_velocity_dps(centre, velocity_norm, _INTR)
    az_edge, _ = angular_velocity_dps(edge, velocity_norm, _INTR)

    # tan(angle) grows faster than angle near the edge, so the same
    # normalised-velocity drift corresponds to a *smaller* true angular
    # velocity there than at centre -- angular_velocity_dps must undo the
    # non-linearity, not apply it again.
    assert az_centre > 0.0
    assert az_edge > 0.0
    assert az_edge < az_centre


def test_angular_velocity_elevation_sign_is_y_down_to_up_inverted() -> None:
    centre = (_INTR.cx / _INTR.width, _INTR.cy / _INTR.height)
    # Moving toward smaller v (upward in the image) must be positive
    # elevation velocity (upward), matching point_to_angles' convention.
    _, el_vel = angular_velocity_dps(centre, (0.0, -0.1), _INTR)
    assert el_vel > 0.0


def test_worked_example_2ms_crossing_at_15m_needs_about_1_15_degrees_lead() -> None:
    # This project's own worked example: a target crossing at 2 m/s at
    # 15 m needs about 1.15 deg of lead, at 100 m/s muzzle velocity.
    range_m = 15.0
    crossing_speed_mps = 2.0
    angular_rate_dps = math.degrees(crossing_speed_mps / range_m)
    tof = DEFAULT_BALLISTIC_TABLE.time_of_flight(range_m)

    lead_az, _ = lead_angle((angular_rate_dps, 0.0), tof)

    assert lead_az == pytest.approx(1.15, abs=0.01)
