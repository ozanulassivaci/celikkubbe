from __future__ import annotations

import numpy as np
import pytest

from celikkubbe.geometry.frames import (
    DEFAULT_TURRET_GEOMETRY,
    TurretGeometry,
    barrel_direction,
    camera_to_turret,
    rotation_matrix,
    turret_to_camera,
)


def test_rotation_matrix_identity_at_zero_pan_tilt() -> None:
    r = rotation_matrix(0.0, 0.0)
    assert np.allclose(r, np.eye(3))


def test_barrel_direction_forward_at_zero_pan_tilt() -> None:
    assert np.allclose(barrel_direction(0.0, 0.0), [0.0, 0.0, 1.0])


def test_barrel_direction_positive_pan_moves_toward_positive_x() -> None:
    # Azimuth positive right (viewed from above): Z rotates toward +X.
    d = barrel_direction(90.0, 0.0)
    assert np.allclose(d, [1.0, 0.0, 0.0], atol=1e-9)


def test_barrel_direction_negative_pan_moves_toward_negative_x() -> None:
    d = barrel_direction(-90.0, 0.0)
    assert np.allclose(d, [-1.0, 0.0, 0.0], atol=1e-9)


def test_barrel_direction_positive_tilt_moves_toward_negative_y() -> None:
    # Elevation positive up == negative Y (Y is down).
    d = barrel_direction(0.0, 90.0)
    assert np.allclose(d, [0.0, -1.0, 0.0], atol=1e-9)


def test_barrel_direction_is_unit_length_everywhere() -> None:
    for pan in (-170.0, -45.0, 0.0, 30.0, 170.0):
        for tilt in (-20.0, 0.0, 45.0, 60.0):
            d = barrel_direction(pan, tilt)
            assert np.linalg.norm(d) == pytest.approx(1.0)


def test_camera_to_turret_of_origin_is_the_offset() -> None:
    geom = TurretGeometry(cam_offset_x_m=0.1, cam_offset_y_m=0.2, cam_offset_z_m=0.3)
    p = camera_to_turret(np.array([0.0, 0.0, 0.0]), geom)
    assert np.allclose(p, [0.1, 0.2, 0.3])


def test_camera_to_turret_and_back_round_trips() -> None:
    geom = TurretGeometry(
        cam_offset_x_m=0.05,
        cam_offset_y_m=-0.10,
        cam_offset_z_m=0.15,
        cam_roll_deg=3.0,
        cam_pitch_deg=-2.0,
        cam_yaw_deg=1.5,
    )
    p_cam = np.array([1.23, -0.45, 2.5])
    p_turret = camera_to_turret(p_cam, geom)
    back = turret_to_camera(p_turret, geom)
    assert np.allclose(back, p_cam)


def test_camera_to_turret_identity_geometry_is_a_pure_translation() -> None:
    geom = TurretGeometry(cam_offset_x_m=0.0, cam_offset_y_m=0.2, cam_offset_z_m=0.0)
    p_cam = np.array([1.0, 2.0, 3.0])
    p_turret = camera_to_turret(p_cam, geom)
    assert np.allclose(p_turret, p_cam + [0.0, 0.2, 0.0])


def test_default_turret_geometry_has_the_documented_lateral_offset() -> None:
    assert DEFAULT_TURRET_GEOMETRY.cam_offset_y_m == pytest.approx(0.20)
    assert DEFAULT_TURRET_GEOMETRY.cam_offset_x_m == 0.0
    assert DEFAULT_TURRET_GEOMETRY.cam_offset_z_m == 0.0
