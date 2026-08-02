"""Boresight and camera-to-turret calibration: types, persistence, refinement.

Two calibrations, both persisted as JSON under ``config/`` at the repo
root (not ``src/celikkubbe/core/config.py``, which holds compile-time
thresholds -- this is runtime calibration data, machine- and rig-specific,
loaded at startup).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from celikkubbe.geometry.frames import DEFAULT_TURRET_GEOMETRY, TurretGeometry
from celikkubbe.geometry.projection import crosshair_pixel
from celikkubbe.vision.sources import CameraIntrinsics

# Repo-root config/ -- runtime calibration data, not the compile-time
# thresholds in core/config.py. Created on first save; absent entirely
# until then.
DEFAULT_BORESIGHT_PATH = Path("config/boresight.json")
DEFAULT_GEOMETRY_PATH = Path("config/turret_geometry.json")


@dataclass(frozen=True)
class BoresightCorrection:
    """Residual angular correction measured at one range.

    After geometric correction (frames.py/projection.py) and ballistic
    compensation (ballistics.py), residual angular error remains from
    measurement imprecision and barrel alignment. Correct it empirically:
    fire at a known point, record where the shot lands relative to the
    aim point, convert the offset to degrees, and store it here.
    """

    az_offset_deg: float
    el_offset_deg: float
    calibrated_at: str
    range_m: float
    notes: str


@dataclass(frozen=True)
class BoresightTable:
    """A set of BoresightCorrections at different ranges, since residual
    error is not necessarily range-independent.

    ``corrections`` need not be pre-sorted; ``correction_at`` sorts by
    range once. An empty table (the "uncalibrated" default) returns a
    zero correction everywhere -- the system runs uncalibrated rather
    than failing to start.
    """

    corrections: tuple[BoresightCorrection, ...] = ()

    def correction_at(self, range_m: float) -> tuple[float, float]:
        """Interpolated (az_offset_deg, el_offset_deg) at range_m.

        Same linear-interpolation-with-clamp shape as
        ballistics.BallisticTable.interpolate_drop_deg, minus the
        clamped flag: an out-of-range boresight lookup silently reuses
        the nearest measured correction rather than warning, since
        residual boresight error changes far more slowly with range than
        ballistic drop does.
        """
        if not self.corrections:
            return 0.0, 0.0
        ordered = sorted(self.corrections, key=lambda c: c.range_m)
        if len(ordered) == 1 or range_m <= ordered[0].range_m:
            return ordered[0].az_offset_deg, ordered[0].el_offset_deg
        if range_m >= ordered[-1].range_m:
            return ordered[-1].az_offset_deg, ordered[-1].el_offset_deg
        for c0, c1 in zip(ordered, ordered[1:], strict=False):
            if c0.range_m <= range_m <= c1.range_m:
                t = (range_m - c0.range_m) / (c1.range_m - c0.range_m)
                az = c0.az_offset_deg + t * (c1.az_offset_deg - c0.az_offset_deg)
                el = c0.el_offset_deg + t * (c1.el_offset_deg - c0.el_offset_deg)
                return az, el
        return ordered[-1].az_offset_deg, ordered[-1].el_offset_deg  # pragma: no cover — exhaustive


def save_boresight(table: BoresightTable, path: Path = DEFAULT_BORESIGHT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"corrections": [dataclasses.asdict(c) for c in table.corrections]}
    path.write_text(json.dumps(data, indent=2))


def load_boresight(path: Path = DEFAULT_BORESIGHT_PATH) -> BoresightTable:
    """Missing or unreadable file -> the empty (uncalibrated) table, not
    a crash: the system must run uncalibrated rather than fail to start.
    """
    if not path.exists():
        return BoresightTable()
    try:
        data = json.loads(path.read_text())
        corrections = tuple(
            BoresightCorrection(
                az_offset_deg=c["az_offset_deg"],
                el_offset_deg=c["el_offset_deg"],
                calibrated_at=c["calibrated_at"],
                range_m=c["range_m"],
                notes=c.get("notes", ""),
            )
            for c in data.get("corrections", [])
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return BoresightTable()
    return BoresightTable(corrections)


def save_turret_geometry(geom: TurretGeometry, path: Path = DEFAULT_GEOMETRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(geom), indent=2))


def load_turret_geometry(path: Path = DEFAULT_GEOMETRY_PATH) -> TurretGeometry:
    """Missing or unreadable file -> DEFAULT_TURRET_GEOMETRY, not a
    crash -- see its own TODO(measurement) note for why that default is
    itself a placeholder.
    """
    if not path.exists():
        return DEFAULT_TURRET_GEOMETRY
    try:
        data = json.loads(path.read_text())
        return TurretGeometry(**data)
    except (json.JSONDecodeError, OSError, TypeError):
        return DEFAULT_TURRET_GEOMETRY


@dataclass(frozen=True)
class GeometryObservation:
    """One observation of a fixed physical target for camera-to-turret
    refinement: the pan/tilt the turret was commanded to, the (measured
    or assumed) range to the target, and the pixel where the target
    actually appeared in the camera image at that pan/tilt.
    """

    pan_deg: float
    tilt_deg: float
    range_m: float
    observed_u_norm: float
    observed_v_norm: float


def refine_geometry(
    observations: list[GeometryObservation],
    intr: CameraIntrinsics,
    initial_geom: TurretGeometry = DEFAULT_TURRET_GEOMETRY,
) -> TurretGeometry:
    """Least-squares refinement of TurretGeometry's camera offset and
    mounting rotation from several observations of the same physical
    target at different known pan/tilt angles.

    For each observation, the model predicts where the barrel-axis
    crosshair should land in the image (projection.crosshair_pixel at
    that pan/tilt/range); refinement adjusts the 6 free parameters
    (cam_offset_x/y/z_m, cam_roll/pitch/yaw_deg -- muzzle_offset_z_m is
    left untouched, since it cannot be observed from crosshair position
    alone) to minimise the sum of squared pixel-space residuals against
    what was actually observed. Optional and lower priority than the
    rest of this module -- present so the entry point exists, not
    because it has been exercised against a real rig yet.

    Needs at least 3 observations to be well-determined (6 free
    parameters, 2 residuals each); with fewer, scipy will still return
    *a* result but it is not meaningfully constrained.
    """
    from scipy.optimize import least_squares

    def residuals(params: np.ndarray) -> np.ndarray:
        geom = TurretGeometry(
            cam_offset_x_m=params[0],
            cam_offset_y_m=params[1],
            cam_offset_z_m=params[2],
            cam_roll_deg=params[3],
            cam_pitch_deg=params[4],
            cam_yaw_deg=params[5],
            muzzle_offset_z_m=initial_geom.muzzle_offset_z_m,
        )
        out: list[float] = []
        for obs in observations:
            predicted = crosshair_pixel(obs.pan_deg, obs.tilt_deg, obs.range_m, intr, geom)
            if predicted is None:
                # Behind the camera under the current parameter guess --
                # no meaningful residual to report; contribute nothing
                # rather than let a NaN poison the whole solve.
                out.extend([0.0, 0.0])
                continue
            out.append(predicted[0] - obs.observed_u_norm)
            out.append(predicted[1] - obs.observed_v_norm)
        return np.array(out)

    x0 = np.array(
        [
            initial_geom.cam_offset_x_m,
            initial_geom.cam_offset_y_m,
            initial_geom.cam_offset_z_m,
            initial_geom.cam_roll_deg,
            initial_geom.cam_pitch_deg,
            initial_geom.cam_yaw_deg,
        ]
    )
    result = least_squares(residuals, x0)
    x = result.x
    return TurretGeometry(
        cam_offset_x_m=float(x[0]),
        cam_offset_y_m=float(x[1]),
        cam_offset_z_m=float(x[2]),
        cam_roll_deg=float(x[3]),
        cam_pitch_deg=float(x[4]),
        cam_yaw_deg=float(x[5]),
        muzzle_offset_z_m=initial_geom.muzzle_offset_z_m,
    )
