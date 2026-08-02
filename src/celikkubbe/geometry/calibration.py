"""Boresight and camera-to-turret calibration: types, persistence, refinement.

Two calibrations, both persisted as JSON under ``config/`` at the repo
root (not ``src/celikkubbe/core/config.py``, which holds compile-time
thresholds -- this is runtime calibration data, machine- and rig-specific,
loaded at startup).
"""

from __future__ import annotations

from dataclasses import dataclass


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
