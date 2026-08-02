"""Coordinate frames and transforms between them.

Two frames:

Camera frame (OpenCV convention): X right, Y **down**, Z forward along the
optical axis. Origin at the camera's optical centre.

Turret frame: origin at the intersection of the pan and tilt axes. This is
a *fixed* frame, not a body frame -- it does not rotate with pan/tilt. At
pan = 0, tilt = 0 its axes align with the camera frame convention (X
right, Y down, Z forward), which is what "when pan=0, tilt=0" in the
class docstring below means: the turret-to-camera relationship is a
single fixed rigid transform (mounting offset + mounting rotation),
independent of the current pan/tilt commanded position. The camera is
chassis-mounted and does not rotate with the turret, so this fixed
relationship is physically correct, not a simplification.

Angles (azimuth/elevation) are degrees at every function boundary in this
module, radians internally. Azimuth (pan) is positive to the **right**
when viewed from above -- the same right-handed convention as aeronautical
yaw about a down-pointing axis (0=forward, 90=right), which for a
right-handed X-right/Y-down/Z-forward frame means rotating Z *toward* +X
for positive azimuth. Elevation (tilt) is positive **up**, which is
negative Y (Y is down) -- the sign inversion made explicit everywhere it
appears.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TurretGeometry:
    """Fixed spatial relationship between camera and turret rotation centre.

    Translation is expressed in the turret frame at pan=0, tilt=0: the
    camera's optical centre relative to the turret rotation centre (i.e.
    ``camera_to_turret(origin) == (cam_offset_x_m, cam_offset_y_m,
    cam_offset_z_m)``).

    Measurement procedure: with the turret at pan=0, tilt=0, measure from
    the pan/tilt axis intersection to the camera's front element (the
    lens, approximating the optical centre), along all three axes, using
    the camera-frame convention (X right, Y down, Z forward) for the sign
    of each component.
    """

    cam_offset_x_m: float  # +right
    cam_offset_y_m: float  # +down
    cam_offset_z_m: float  # +forward
    cam_roll_deg: float = 0.0
    cam_pitch_deg: float = 0.0
    cam_yaw_deg: float = 0.0
    muzzle_offset_z_m: float = 0.0  # muzzle forward of rotation centre along the barrel


# TODO(measurement): replace with the real mounting offset. This placeholder
# is deliberately chosen to reproduce this project's own worked parallax
# example (docs/protocol.md-adjacent Prompt 4 intro: ~2.3 deg at 5 m, ~0.76
# deg at 15 m for a ~0.2 m offset) so the numbers in that example are
# directly checkable in tests -- it is almost certainly not the true
# geometry. Measure with the turret at pan=0, tilt=0 per the docstring
# above before this is used against real hardware.
DEFAULT_TURRET_GEOMETRY = TurretGeometry(
    cam_offset_x_m=0.0,
    cam_offset_y_m=0.20,
    cam_offset_z_m=0.0,
    cam_roll_deg=0.0,
    cam_pitch_deg=0.0,
    cam_yaw_deg=0.0,
    muzzle_offset_z_m=0.0,
)


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle_rad: float) -> np.ndarray:
    # Positive angle rotates +Z toward +X -- see the module docstring for
    # why that is the correct sign for "azimuth positive right".
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrix(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """Turret orientation for a given pan/tilt, applied pan then tilt.

    Physically: the pan motor rotates the whole tilt-axis-and-barrel
    assembly about the turret frame's Y axis; the tilt motor then rotates
    the barrel about the *tilt* axis, which is horizontal and rotates
    along with the pan platform. Composing "pan, then tilt about the new
    local axis" (intrinsic rotations) as fixed-frame matrices gives
    ``R_pan @ R_tilt`` -- pan is applied last when the composed matrix
    acts on a vector, not first; do not swap the order.
    """
    pan_rad = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_deg)
    return _rotation_y(pan_rad) @ _rotation_x(tilt_rad)


def barrel_direction(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """Unit vector in turret frame the barrel points along for pan/tilt.

    ``rotation_matrix(pan, tilt) @ [0, 0, 1]`` -- the barrel's own +Z
    (forward) axis, expressed in the turret frame. This is the exact
    inverse of ``projection.point_to_angles``: for any pan/tilt,
    ``point_to_angles(barrel_direction(pan, tilt)) == (pan, tilt)``. That
    round trip is what the whole aiming pipeline depends on -- if it ever
    breaks, the turret will not point where the solver computed.
    """
    return rotation_matrix(pan_deg, tilt_deg) @ np.array([0.0, 0.0, 1.0])


def muzzle_position(pan_deg: float, tilt_deg: float, geom: TurretGeometry) -> np.ndarray:
    """Muzzle tip position in turret frame: ``muzzle_offset_z_m`` forward
    of the rotation centre along the current barrel direction.

    Used to get the true muzzle-to-target distance (slant range) for
    ballistics, rather than the rotation-centre-to-target distance --
    see ballistics.py's module docstring.
    """
    return barrel_direction(pan_deg, tilt_deg) * geom.muzzle_offset_z_m


def _mount_rotation(geom: TurretGeometry) -> np.ndarray:
    """Fixed camera-mounting rotation: yaw about turret Y (matching pan),
    pitch about turret X (matching tilt), roll about the camera's own
    forward Z, composed intrinsically yaw -> pitch -> roll. These are
    expected to be small (near-zero) mounting-misalignment corrections,
    not operational angles, so the exact composition order matters far
    less than for rotation_matrix -- documented for reproducibility, not
    because a different order would be wrong at these magnitudes.
    """
    yaw = _rotation_y(math.radians(geom.cam_yaw_deg))
    pitch = _rotation_x(math.radians(geom.cam_pitch_deg))
    roll = _rotation_z(math.radians(geom.cam_roll_deg))
    return yaw @ pitch @ roll


def camera_to_turret(p_cam: np.ndarray, geom: TurretGeometry) -> np.ndarray:
    """Transform a point from camera frame to turret frame.

    Fixed rigid transform (mounting rotation then translation by the
    camera offset) -- does not depend on pan/tilt; see the module
    docstring for why.
    """
    offset = np.array([geom.cam_offset_x_m, geom.cam_offset_y_m, geom.cam_offset_z_m])
    return _mount_rotation(geom) @ np.asarray(p_cam, dtype=float) + offset


def turret_to_camera(p_turret: np.ndarray, geom: TurretGeometry) -> np.ndarray:
    """Inverse of ``camera_to_turret``."""
    offset = np.array([geom.cam_offset_x_m, geom.cam_offset_y_m, geom.cam_offset_z_m])
    return _mount_rotation(geom).T @ (np.asarray(p_turret, dtype=float) - offset)
