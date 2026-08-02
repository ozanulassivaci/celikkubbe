"""Pixel <-> ray, and turret-frame point <-> pixel projections.

Pixels are (u, v) with the origin at the top-left, u rightward, v
downward. Every external interface here uses normalised [0, 1]
coordinates, consistent with core.types.BoundingBox/Detection/Track;
pixel units only exist inside this module, derived from CameraIntrinsics
(whose fx/fy/cx/cy are themselves in pixels).

Angles are degrees at the boundary, radians internally -- see
frames.py's module docstring for the full azimuth/elevation convention,
including the Y-down/elevation-up sign inversion.
"""

from __future__ import annotations

import math

import numpy as np

from celikkubbe.geometry.frames import (
    TurretGeometry,
    barrel_direction,
    camera_to_turret,
    turret_to_camera,
)
from celikkubbe.vision.sources import CameraIntrinsics


def pixel_to_ray(u_norm: float, v_norm: float, intr: CameraIntrinsics) -> np.ndarray:
    """Unit vector in camera frame for a normalised pixel.

    Standard pinhole back-projection: x=(u-cx)/fx, y=(v-cy)/fy, z=1,
    normalised.
    """
    u_px = u_norm * intr.width
    v_px = v_norm * intr.height
    x = (u_px - intr.cx) / intr.fx
    y = (v_px - intr.cy) / intr.fy
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


def pixel_to_turret_point(
    u_norm: float,
    v_norm: float,
    range_m: float,
    intr: CameraIntrinsics,
    geom: TurretGeometry,
) -> np.ndarray:
    """3D point in turret frame for a pixel at a known range.

    range_m is Z-depth (distance along the camera's optical axis), matching
    the depth sensor's own convention and Track.range_m's actual meaning
    when range_source == "depth" (vision/l2_color.py takes the median of
    raw per-pixel depth values, which are Z-depth, not slant range). This
    is not an arbitrary choice: at typical off-axis angles within a 69 deg
    FOV, Z-depth and straight-line camera-to-point range differ by up to
    ~15-20%, far outside the 0.10 deg aim tolerance if confused.
    Implemented by scaling pixel_to_ray's unit vector so its Z component
    equals range_m -- algebraically identical to the more familiar
    pinhole back-projection X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy, Z=Z.
    """
    ray = pixel_to_ray(u_norm, v_norm, intr)
    p_cam = ray * (range_m / ray[2])
    return camera_to_turret(p_cam, geom)


def point_to_angles(p_turret: np.ndarray) -> tuple[float, float]:
    """Turret-frame point -> (az_deg, el_deg) the barrel must point to face it.

    az = atan2(x, z); el = atan2(-y, hypot(x, z)). The negation on y is the
    Y-down (frame convention) to elevation-up (angle convention)
    conversion -- the single most likely place to introduce a sign error
    in this module. This is the exact inverse of
    frames.barrel_direction(): point_to_angles(barrel_direction(pan, el))
    == (pan, el) for any pan/el.
    """
    x, y, z = p_turret
    az_deg = math.degrees(math.atan2(x, z))
    el_deg = math.degrees(math.atan2(-y, math.hypot(x, z)))
    return az_deg, el_deg


def crosshair_pixel(
    pan_deg: float,
    tilt_deg: float,
    range_m: float,
    intr: CameraIntrinsics,
    geom: TurretGeometry,
) -> tuple[float, float] | None:
    """Normalised pixel where the barrel axis, at the given range, projects
    onto the camera image.

    Range-dependent because of parallax: aiming at a target 5 m away puts
    the barrel axis at a different pixel than aiming at one 15 m away,
    even at identical pan/tilt (see frames.TurretGeometry's camera
    offset). This is what the GUI draws instead of a fixed centre
    crosshair.

    Returns None when the projected point falls behind the camera
    (p_cam.z <= 0) -- possible for a large camera offset combined with a
    short range and extreme tilt, where the geometric point along the
    barrel axis at that range is physically behind the camera's optical
    centre.
    """
    p_turret = barrel_direction(pan_deg, tilt_deg) * range_m
    p_cam = turret_to_camera(p_turret, geom)
    if p_cam[2] <= 0.0:
        return None
    u_px = intr.fx * p_cam[0] / p_cam[2] + intr.cx
    v_px = intr.fy * p_cam[1] / p_cam[2] + intr.cy
    return u_px / intr.width, v_px / intr.height


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def crosshair_with_indicator(
    pan_deg: float,
    tilt_deg: float,
    range_m: float,
    intr: CameraIntrinsics,
    geom: TurretGeometry,
) -> tuple[tuple[float, float], bool, float | None]:
    """Like crosshair_pixel, but always returns a drawable position.

    Returns (clamped_pixel, out_of_frame, bearing_deg). When the true
    crosshair position falls outside [0, 1] in either axis -- or there is
    no valid pixel at all, because the point is behind the camera -- the
    GUI pins it to the frame edge (or centre, in the behind-camera case)
    with a direction arrow, rather than losing the indicator entirely.
    Otherwise the operator loses track of where the barrel is pointing
    whenever it swings past the camera's 69 deg field of view.

    bearing_deg is the direction of the unclamped position from frame
    centre, measured clockwise from straight up in the image (0=up,
    90=right, 180=down, 270=left) -- what the GUI draws the arrow along.
    None when the position is already in-frame.
    """
    pixel = crosshair_pixel(pan_deg, tilt_deg, range_m, intr, geom)
    if pixel is not None:
        u_norm, v_norm = pixel
        if 0.0 <= u_norm <= 1.0 and 0.0 <= v_norm <= 1.0:
            return (u_norm, v_norm), False, None
        du, dv = u_norm - 0.5, v_norm - 0.5
    else:
        # No valid pixel to clamp -- pin to centre, but still point the
        # arrow the right way using pan/tilt directly: increasing pan
        # moves the crosshair right (+u), increasing tilt moves it up
        # (-v), matching crosshair_pixel's own sign behaviour.
        u_norm, v_norm = 0.5, 0.5
        du, dv = pan_deg, -tilt_deg

    bearing_deg = math.degrees(math.atan2(du, -dv)) % 360.0
    return (_clamp01(u_norm), _clamp01(v_norm)), True, bearing_deg
