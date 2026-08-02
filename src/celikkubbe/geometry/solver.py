"""AimSolver: bbox -> ray -> point at range -> angles -> lead -> drop ->
boresight -> clamp.

Pipeline order matters: lead and drop are angular corrections computed
independently and summed, not re-derived from a corrected aim point --
each is small enough (aim tolerance 0.10 deg; lead and drop are each
sub-degree at these ranges) that the small-angle interactions between
them are negligible, and keeping them additive is what lets
AimSolution report them as separate, inspectable components.

Two distinct "range" values, both on AimSolution: ``range_m`` is Z-depth
(projection.pixel_to_turret_point's input convention -- see its
docstring), used to place the target's turret-frame point.
``slant_range_m`` is the actual muzzle-to-target distance -- what the BB
travels -- used for everything ballistic (time of flight, drop lookup).
They agree only on-axis; off-axis they diverge as 1/cos(theta), and
because the camera is chassis-fixed, targets are genuinely engaged well
off-axis, not just near boresight. At 15 m Z-depth and 34.5 deg off-axis
(the edge of a 69 deg FOV), slant range is 18.2 m, not 15 m -- confusing
the two here would burn most of the 0.10 deg aim tolerance on drop alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from celikkubbe.core import config
from celikkubbe.core.types import Track, TrackStatus
from celikkubbe.geometry.ballistics import BallisticTable, angular_velocity_dps, lead_angle
from celikkubbe.geometry.calibration import BoresightTable
from celikkubbe.geometry.frames import TurretGeometry, muzzle_position
from celikkubbe.geometry.projection import pixel_to_turret_point, point_to_angles
from celikkubbe.vision.sources import CameraIntrinsics

AimRangeSource = Literal["depth", "size", "none", "assumed"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class AimSolution:
    az_deg: float
    el_deg: float
    range_m: float
    slant_range_m: float
    range_source: AimRangeSource
    lead_az_deg: float
    lead_el_deg: float
    drop_deg: float
    confidence: Confidence
    warnings: tuple[str, ...]


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


class AimSolver:
    def __init__(
        self,
        geom: TurretGeometry,
        ballistics: BallisticTable,
        boresight: BoresightTable | None = None,
    ) -> None:
        self._geom = geom
        self._ballistics = ballistics
        self._boresight = boresight if boresight is not None else BoresightTable()

    def solve(self, track: Track, intr: CameraIntrinsics) -> AimSolution | None:
        """None only for a track that is not yet CONFIRMED -- not aiming
        at tentative detections. For any CONFIRMED track, always returns an
        AimSolution, even with range assumed and confidence "low": the
        engagement FSM already treats a missing dict entry as
        NO_AIM_SOLUTION, and silently dropping a target the operator can
        see on screen would be worse than a low-confidence solution.
        """
        if track.status is not TrackStatus.CONFIRMED:
            return None

        warnings: list[str] = []
        range_m = track.range_m
        range_source: AimRangeSource = track.range_source
        range_assumed = range_m is None
        if range_m is None:
            range_m = config.DEFAULT_RANGE_M
            range_source = "assumed"
            warnings.append(
                f"no range available, assumed DEFAULT_RANGE_M={config.DEFAULT_RANGE_M}m"
            )

        u_center = (track.bbox[0] + track.bbox[2]) / 2.0
        v_center = (track.bbox[1] + track.bbox[3]) / 2.0

        p_turret = pixel_to_turret_point(u_center, v_center, range_m, intr, self._geom)
        base_az_deg, base_el_deg = point_to_angles(p_turret)

        # Ballistics needs the actual distance the BB travels -- the norm
        # of the muzzle-to-target vector -- not range_m's Z-depth. They
        # agree on-axis but diverge fast off-axis (1/cos(theta)), and the
        # camera being chassis-fixed means targets are genuinely engaged
        # near the frame edge, not just close to boresight. base_az/el
        # (pre lead/drop/boresight) are accurate enough to place the
        # muzzle for this -- see frames.muzzle_position.
        muzzle_pos = muzzle_position(base_az_deg, base_el_deg, self._geom)
        slant_range_m = float(np.linalg.norm(p_turret - muzzle_pos))

        ang_vel_dps = angular_velocity_dps((u_center, v_center), track.velocity, intr)
        time_of_flight_s = self._ballistics.time_of_flight(slant_range_m)
        lead_az_deg, lead_el_deg = lead_angle(ang_vel_dps, time_of_flight_s)

        drop_deg, drop_clamped = self._ballistics.interpolate_drop_deg(slant_range_m)
        if drop_clamped:
            warnings.append(
                f"ballistic table extrapolated beyond its measured span at {slant_range_m:.2f}m"
            )

        boresight_az_deg, boresight_el_deg = self._boresight.correction_at(range_m)

        total_az = base_az_deg + lead_az_deg + boresight_az_deg
        # Drop is a vertical-only correction (pitch up to compensate for
        # the projectile falling); it has no azimuth component.
        total_el = base_el_deg + lead_el_deg + drop_deg + boresight_el_deg

        pan_lo, pan_hi = config.PAN_LIMIT_DEG
        tilt_lo, tilt_hi = config.TILT_LIMIT_DEG
        clamped_az = _clamp(total_az, pan_lo, pan_hi)
        clamped_el = _clamp(total_el, tilt_lo, tilt_hi)
        if clamped_az != total_az or clamped_el != total_el:
            warnings.append("solved angle clamped to software limits")

        confidence = self._confidence(range_assumed, range_source, intr)

        return AimSolution(
            az_deg=clamped_az,
            el_deg=clamped_el,
            range_m=range_m,
            slant_range_m=slant_range_m,
            range_source=range_source,
            lead_az_deg=lead_az_deg,
            lead_el_deg=lead_el_deg,
            drop_deg=drop_deg,
            confidence=confidence,
            warnings=tuple(warnings),
        )

    def solve_all(
        self, tracks: list[Track], intr: CameraIntrinsics
    ) -> dict[int, tuple[float, float]]:
        """Exactly the aim_solutions dict engagement.step() expects: one
        (az_deg, el_deg) entry per CONFIRMED track, keyed by track_id.
        """
        solutions: dict[int, tuple[float, float]] = {}
        for track in tracks:
            solution = self.solve(track, intr)
            if solution is not None:
                solutions[track.track_id] = (solution.az_deg, solution.el_deg)
        return solutions

    @staticmethod
    def _confidence(
        range_assumed: bool, range_source: AimRangeSource, intr: CameraIntrinsics
    ) -> Confidence:
        if range_assumed or not intr.is_reliable:
            return "low"
        if range_source == "depth":
            return "high"
        if range_source == "size":
            return "medium"
        return "low"  # pragma: no cover — range_source == "none" implies range_m is None
