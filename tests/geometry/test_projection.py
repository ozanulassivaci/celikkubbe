from __future__ import annotations

import math

import pytest

from celikkubbe.geometry.frames import TurretGeometry
from celikkubbe.geometry.projection import (
    crosshair_pixel,
    crosshair_with_indicator,
    pixel_to_ray,
    pixel_to_turret_point,
    point_to_angles,
)
from celikkubbe.vision.sources import estimated_intrinsics

_ZERO_GEOM = TurretGeometry(cam_offset_x_m=0.0, cam_offset_y_m=0.0, cam_offset_z_m=0.0)
_INTR = estimated_intrinsics(640, 480)


def _principal_point_norm():
    return _INTR.cx / _INTR.width, _INTR.cy / _INTR.height


def test_pixel_to_ray_round_trips_through_pinhole_projection() -> None:
    u_norm, v_norm = 0.7, 0.35
    ray = pixel_to_ray(u_norm, v_norm, _INTR)
    u_px = _INTR.fx * ray[0] / ray[2] + _INTR.cx
    v_px = _INTR.fy * ray[1] / ray[2] + _INTR.cy
    assert u_px / _INTR.width == pytest.approx(u_norm)
    assert v_px / _INTR.height == pytest.approx(v_norm)


def test_principal_point_with_zero_offset_yields_zero_az_el() -> None:
    u_norm, v_norm = _principal_point_norm()
    p = pixel_to_turret_point(u_norm, v_norm, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    az_deg, el_deg = point_to_angles(p)
    assert az_deg == pytest.approx(0.0, abs=1e-9)
    assert el_deg == pytest.approx(0.0, abs=1e-9)


def test_target_right_of_centre_gives_positive_azimuth() -> None:
    cu, cv = _principal_point_norm()
    p = pixel_to_turret_point(cu + 0.1, cv, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    az_deg, _ = point_to_angles(p)
    assert az_deg > 0.0


def test_target_left_of_centre_gives_negative_azimuth() -> None:
    cu, cv = _principal_point_norm()
    p = pixel_to_turret_point(cu - 0.1, cv, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    az_deg, _ = point_to_angles(p)
    assert az_deg < 0.0


def test_target_above_centre_gives_positive_elevation() -> None:
    # "Above" == smaller v (v grows downward).
    cu, cv = _principal_point_norm()
    p = pixel_to_turret_point(cu, cv - 0.1, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    _, el_deg = point_to_angles(p)
    assert el_deg > 0.0


def test_target_below_centre_gives_negative_elevation() -> None:
    cu, cv = _principal_point_norm()
    p = pixel_to_turret_point(cu, cv + 0.1, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    _, el_deg = point_to_angles(p)
    assert el_deg < 0.0


def test_parallax_differs_measurably_between_5m_and_15m_and_matches_prediction() -> None:
    # Default-shaped geometry: 0.2 m purely-vertical offset, matching this
    # project's own worked example (~2.3 deg at 5 m, ~0.76 deg at 15 m).
    geom = TurretGeometry(cam_offset_x_m=0.0, cam_offset_y_m=0.2, cam_offset_z_m=0.0)
    cu, cv = _principal_point_norm()

    p5 = pixel_to_turret_point(cu, cv, range_m=5.0, intr=_INTR, geom=geom)
    p15 = pixel_to_turret_point(cu, cv, range_m=15.0, intr=_INTR, geom=geom)
    _, el5 = point_to_angles(p5)
    _, el15 = point_to_angles(p15)

    predicted_5 = -math.degrees(math.atan(0.2 / 5.0))
    predicted_15 = -math.degrees(math.atan(0.2 / 15.0))

    assert el5 == pytest.approx(predicted_5, abs=1e-6)
    assert el15 == pytest.approx(predicted_15, abs=1e-6)
    assert abs(el5) > abs(el15)  # parallax shrinks with range
    assert el5 == pytest.approx(-2.2906, abs=1e-3)
    assert el15 == pytest.approx(-0.7639, abs=1e-3)


def test_crosshair_at_zero_pan_tilt_zero_offset_lands_on_principal_point() -> None:
    pixel = crosshair_pixel(0.0, 0.0, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    assert pixel is not None
    cu, cv = _principal_point_norm()
    assert pixel[0] == pytest.approx(cu)
    assert pixel[1] == pytest.approx(cv)


def test_crosshair_moves_right_as_pan_increases() -> None:
    # Self-consistency requirement, not intuition: point_to_angles (az =
    # atan2(x, z)) combined with the right-hand "azimuth positive right"
    # convention forces barrel_direction.x, and therefore the projected
    # crosshair pixel, to move toward larger u (right) as pan increases.
    base = crosshair_pixel(0.0, 0.0, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    panned = crosshair_pixel(5.0, 0.0, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM)
    assert base is not None and panned is not None
    assert panned[0] > base[0]


def test_crosshair_beyond_fov_is_flagged_clamped_and_points_right() -> None:
    # Half of a 69 deg HFOV is ~34.5 deg; 40 deg pan is outside it.
    pixel, out_of_frame, bearing_deg = crosshair_with_indicator(
        40.0, 0.0, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM
    )
    assert out_of_frame is True
    assert pixel[0] == pytest.approx(1.0)  # clamped to the right edge
    assert bearing_deg == pytest.approx(90.0, abs=1e-6)  # right == 90 deg


def test_crosshair_in_frame_reports_no_indicator() -> None:
    pixel, out_of_frame, bearing_deg = crosshair_with_indicator(
        0.0, 0.0, range_m=10.0, intr=_INTR, geom=_ZERO_GEOM
    )
    assert out_of_frame is False
    assert bearing_deg is None
    cu, cv = _principal_point_norm()
    assert pixel == pytest.approx((cu, cv))


def test_point_behind_camera_returns_none() -> None:
    # Camera placed 5 m forward of the rotation centre: a 1 m-range barrel
    # point ends up well behind the camera's own optical centre.
    geom = TurretGeometry(cam_offset_x_m=0.0, cam_offset_y_m=0.0, cam_offset_z_m=5.0)
    assert crosshair_pixel(0.0, 0.0, range_m=1.0, intr=_INTR, geom=geom) is None


def test_crosshair_with_indicator_handles_behind_camera_case() -> None:
    geom = TurretGeometry(cam_offset_x_m=0.0, cam_offset_y_m=0.0, cam_offset_z_m=5.0)
    pixel, out_of_frame, bearing_deg = crosshair_with_indicator(
        10.0, 0.0, range_m=1.0, intr=_INTR, geom=geom
    )
    assert out_of_frame is True
    assert pixel == (0.5, 0.5)
    assert bearing_deg is not None
