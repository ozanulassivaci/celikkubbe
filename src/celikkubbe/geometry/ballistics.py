"""Time of flight, empirical drop correction, and lead angle.

Calibration procedure for BallisticTable.drop_deg (live fire, one entry
per range): fire at a stationary target at each range, measure the
vertical offset of the impact point from the aim point, convert to
degrees with ``atan(offset_m / range_m)``, and record the result here
replacing the gravity-only placeholder.

Degrees at every function boundary, matching frames.py/projection.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from celikkubbe.vision.sources import CameraIntrinsics

GRAVITY_MS2 = 9.81


def gravity_only_drop_deg(range_m: float, muzzle_velocity_ms: float) -> float:
    """Unassisted projectile drop angle at range_m -- gravity only, no
    hop-up backspin lift.

    For comparison against a live-fire-measured BallisticTable, not a
    substitute for one: hop-up lift partially offsets gravity by an
    amount with no closed-form model accurate enough for the 0.10 deg
    aim tolerance (see the module docstring's calibration procedure).
    Real measured drop should come in below this prediction.
    """
    if range_m <= 0.0 or muzzle_velocity_ms <= 0.0:
        return 0.0
    t = range_m / muzzle_velocity_ms
    drop_m = 0.5 * GRAVITY_MS2 * t * t
    return math.degrees(math.atan(drop_m / range_m))


@dataclass(frozen=True)
class BallisticTable:
    """Empirical drop correction, indexed by range.

    Values must come from live fire. Hop-up backspin generates lift that
    partially offsets gravity, so no closed-form model predicts real
    trajectory accurately enough for our 0.10 degree tolerance.

    ranges_m must be strictly ascending; drop_deg is the corresponding
    positive-is-downward correction (degrees to add, pointing the barrel
    up, to compensate for drop) at each range.
    """

    ranges_m: tuple[float, ...]
    drop_deg: tuple[float, ...]
    muzzle_velocity_ms: float

    def time_of_flight(self, range_m: float) -> float:
        """range_m / muzzle_velocity_ms.

        TODO(measurement): add a drag term once live-fire measurements
        exist to characterise it; this is a muzzle-velocity-only estimate.
        """
        if self.muzzle_velocity_ms <= 0.0:
            return 0.0
        return range_m / self.muzzle_velocity_ms

    def interpolate_drop_deg(self, range_m: float) -> tuple[float, bool]:
        """Linear interpolation between measured entries.

        Returns (drop_deg, clamped). Clamped to the nearest endpoint's
        value when range_m falls outside the measured span, with
        clamped=True so the caller can warn that the table is being
        extrapolated rather than interpolated.
        """
        ranges = self.ranges_m
        drops = self.drop_deg
        if range_m <= ranges[0]:
            return drops[0], range_m < ranges[0]
        if range_m >= ranges[-1]:
            return drops[-1], range_m > ranges[-1]
        for i in range(len(ranges) - 1):
            r0, r1 = ranges[i], ranges[i + 1]
            if r0 <= range_m <= r1:
                t = (range_m - r0) / (r1 - r0)
                return drops[i] + t * (drops[i + 1] - drops[i]), False
        return drops[-1], False  # pragma: no cover — the two clamp checks above are exhaustive


_DEFAULT_RANGES_M = (5.0, 10.0, 15.0)
_DEFAULT_MUZZLE_VELOCITY_MS = 100.0

# TODO(measurement): every value here needs live-fire replacement, per
# BallisticTable's own docstring. These are gravity_only_drop_deg()
# predictions -- no hop-up lift -- so real measured drop should come in
# below these figures. muzzle_velocity_ms=100.0 m/s is itself an assumed
# planning figure (TODO: chronograph the actual HPA setup), not measured.
DEFAULT_BALLISTIC_TABLE = BallisticTable(
    ranges_m=_DEFAULT_RANGES_M,
    drop_deg=tuple(
        gravity_only_drop_deg(r, _DEFAULT_MUZZLE_VELOCITY_MS) for r in _DEFAULT_RANGES_M
    ),
    muzzle_velocity_ms=_DEFAULT_MUZZLE_VELOCITY_MS,
)


def angular_velocity_dps(
    bbox_center_norm: tuple[float, float],
    velocity_norm: tuple[float, float],
    intr: CameraIntrinsics,
) -> tuple[float, float]:
    """Convert the Kalman filter's normalised centroid velocity into true
    angular velocity, in degrees/second.

    tracking/kalman.py's CentroidKalmanFilter docstring: normalised
    coordinates are proportional to tan(angle), not angle itself, so a
    target moving at a constant *angular* rate produces a *larger*
    apparent normalised-coordinate velocity near the frame edge than at
    the centre (~47% faster for a 69 deg FOV). This is the inverse
    conversion the lead solver needs -- treating velocity_norm as linear
    in angle would under-lead a target near frame centre and over-lead
    one near the edge.

    Local derivative: dtheta/dx_px = fx / (fx^2 + x_px^2), x_px = u_px -
    cx (symmetrically for v/fy/cy). The el component carries the same
    Y-down to elevation-up sign inversion as projection.point_to_angles.
    """
    u_norm, v_norm = bbox_center_norm
    du_dt, dv_dt = velocity_norm

    x_px = u_norm * intr.width - intr.cx
    y_px = v_norm * intr.height - intr.cy

    dtheta_az_du_px = intr.fx / (intr.fx**2 + x_px**2)
    dtheta_el_dv_px = intr.fy / (intr.fy**2 + y_px**2)

    az_vel_rad_s = dtheta_az_du_px * intr.width * du_dt
    el_vel_rad_s = -dtheta_el_dv_px * intr.height * dv_dt

    return math.degrees(az_vel_rad_s), math.degrees(el_vel_rad_s)


def lead_angle(
    target_angular_velocity_dps: tuple[float, float],
    time_of_flight_s: float,
) -> tuple[float, float]:
    """How far the target's angular position moves during the
    projectile's flight, assuming constant angular velocity over that
    (short) interval -- a linear extrapolation, not a full intercept
    solve.
    """
    az_vel_dps, el_vel_dps = target_angular_velocity_dps
    return az_vel_dps * time_of_flight_s, el_vel_dps * time_of_flight_s
