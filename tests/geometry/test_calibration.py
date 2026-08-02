from __future__ import annotations

import pytest

from celikkubbe.geometry.calibration import (
    BoresightCorrection,
    BoresightTable,
    GeometryObservation,
    load_boresight,
    load_turret_geometry,
    refine_geometry,
    save_boresight,
    save_turret_geometry,
)
from celikkubbe.geometry.frames import DEFAULT_TURRET_GEOMETRY, TurretGeometry
from celikkubbe.geometry.projection import crosshair_pixel
from celikkubbe.vision.sources import estimated_intrinsics


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


# --- persistence ---


def test_boresight_round_trips_through_save_and_load(tmp_path) -> None:
    path = tmp_path / "boresight.json"
    table = BoresightTable(
        (
            BoresightCorrection(0.2, -0.1, "2026-01-15", 5.0, "range day 1"),
            BoresightCorrection(0.3, -0.15, "2026-01-15", 15.0, "range day 1"),
        )
    )
    save_boresight(table, path)
    loaded = load_boresight(path)

    assert loaded.correction_at(5.0) == pytest.approx((0.2, -0.1))
    assert loaded.correction_at(15.0) == pytest.approx((0.3, -0.15))


def test_boresight_missing_file_yields_empty_table(tmp_path) -> None:
    loaded = load_boresight(tmp_path / "does_not_exist.json")
    assert loaded == BoresightTable()
    assert loaded.correction_at(10.0) == (0.0, 0.0)


def test_boresight_corrupt_file_yields_empty_table(tmp_path) -> None:
    path = tmp_path / "boresight.json"
    path.write_text("not valid json{{{")
    assert load_boresight(path) == BoresightTable()


def test_turret_geometry_round_trips_through_save_and_load(tmp_path) -> None:
    path = tmp_path / "turret_geometry.json"
    geom = TurretGeometry(
        cam_offset_x_m=0.01,
        cam_offset_y_m=0.22,
        cam_offset_z_m=-0.03,
        cam_roll_deg=1.0,
        cam_pitch_deg=-2.0,
        cam_yaw_deg=0.5,
        muzzle_offset_z_m=0.1,
    )
    save_turret_geometry(geom, path)
    loaded = load_turret_geometry(path)
    assert loaded == geom


def test_turret_geometry_missing_file_yields_default(tmp_path) -> None:
    loaded = load_turret_geometry(tmp_path / "does_not_exist.json")
    assert loaded == DEFAULT_TURRET_GEOMETRY


def test_turret_geometry_corrupt_file_yields_default(tmp_path) -> None:
    path = tmp_path / "turret_geometry.json"
    path.write_text("{not json")
    assert load_turret_geometry(path) == DEFAULT_TURRET_GEOMETRY


# --- camera-to-turret refinement ---


def test_refine_geometry_recovers_a_known_offset() -> None:
    intr = estimated_intrinsics(640, 480)
    true_geom = TurretGeometry(cam_offset_x_m=0.03, cam_offset_y_m=0.18, cam_offset_z_m=-0.02)
    pan_tilt_range = [
        (0.0, 0.0, 10.0),
        (10.0, 5.0, 8.0),
        (-8.0, -3.0, 12.0),
        (15.0, 10.0, 6.0),
        (-12.0, 8.0, 9.0),
    ]
    observations = []
    for pan, tilt, rng in pan_tilt_range:
        pixel = crosshair_pixel(pan, tilt, rng, intr, true_geom)
        assert pixel is not None
        observations.append(GeometryObservation(pan, tilt, rng, pixel[0], pixel[1]))

    # Start from a deliberately wrong initial guess.
    initial_guess = TurretGeometry(cam_offset_x_m=0.0, cam_offset_y_m=0.0, cam_offset_z_m=0.0)
    refined = refine_geometry(observations, intr, initial_guess)

    assert refined.cam_offset_x_m == pytest.approx(true_geom.cam_offset_x_m, abs=1e-4)
    assert refined.cam_offset_y_m == pytest.approx(true_geom.cam_offset_y_m, abs=1e-4)
    assert refined.cam_offset_z_m == pytest.approx(true_geom.cam_offset_z_m, abs=1e-4)


def test_refine_geometry_preserves_muzzle_offset() -> None:
    intr = estimated_intrinsics(640, 480)
    initial_guess = TurretGeometry(
        cam_offset_x_m=0.0, cam_offset_y_m=0.2, cam_offset_z_m=0.0, muzzle_offset_z_m=0.07
    )
    observations = [
        GeometryObservation(0.0, 0.0, 10.0, *crosshair_pixel(0.0, 0.0, 10.0, intr, initial_guess)),
        GeometryObservation(5.0, 0.0, 10.0, *crosshair_pixel(5.0, 0.0, 10.0, intr, initial_guess)),
        GeometryObservation(0.0, 5.0, 10.0, *crosshair_pixel(0.0, 5.0, 10.0, intr, initial_guess)),
    ]
    refined = refine_geometry(observations, intr, initial_guess)
    assert refined.muzzle_offset_z_m == 0.07
