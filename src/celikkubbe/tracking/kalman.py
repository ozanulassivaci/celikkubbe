"""Per-track Kalman filters: constant-velocity centroid + a separate 1D range filter.

Two filters, not one, because they have different availability: the
centroid is measured every frame a detection matches, but range may be
unavailable for long stretches (no depth, unreliable size estimate). A
single combined filter would need special-casing for a missing range
component on every predict/update; keeping range separate means it simply
idles — predict-only, no update — for exactly as long as no measurement
arrives, and resumes the moment one does.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

# Empirical starting points for an indoor-range Stage 2/3 target moving at
# a few m/s; retune once real (non-synthetic) detections are available.
CENTROID_PROCESS_NOISE = 1e-3
CENTROID_MEASUREMENT_NOISE = 1e-2
CENTROID_INITIAL_COVARIANCE = 0.05

RANGE_PROCESS_NOISE = 1e-2
RANGE_INITIAL_COVARIANCE = 1.0
# Depth measurements are trustworthy; a size-based estimate from an assumed
# FOV and a known real-world diameter is coarse. Range fusion weights each
# measurement by these noise figures rather than treating them as equals.
RANGE_MEASUREMENT_NOISE_DEPTH = 0.05
RANGE_MEASUREMENT_NOISE_SIZE = 4.0


class CentroidKalmanFilter:
    """Constant-velocity filter on the normalised (x, y) centroid.

    State is [x, y, vx, vy]. Normalised coordinates are proportional to
    tan(angle), not angle itself: a target moving at a constant angular
    rate appears to accelerate as it approaches the frame edge (roughly
    47% faster apparent rate at the edge than centre for a 69 degree FOV).
    At 30 Hz the per-frame error this introduces is small and process
    noise absorbs it, but whoever consumes this state to aim (the lead
    solver) must convert to angles using intrinsics rather than treating
    it as linear in angle.
    """

    def __init__(
        self,
        x: float,
        y: float,
        process_noise: float = CENTROID_PROCESS_NOISE,
        measurement_noise: float = CENTROID_MEASUREMENT_NOISE,
    ) -> None:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([x, y, 0.0, 0.0])
        kf.F = np.eye(4)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        kf.P *= CENTROID_INITIAL_COVARIANCE
        kf.R *= measurement_noise
        kf.Q *= process_noise
        self._kf = kf

    def predict(self, dt: float) -> None:
        self._kf.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        self._kf.predict()

    def update(self, x: float, y: float) -> None:
        self._kf.update(np.array([x, y]))

    @property
    def position(self) -> tuple[float, float]:
        return float(self._kf.x[0]), float(self._kf.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self._kf.x[2]), float(self._kf.x[3])


class RangeKalmanFilter:
    """1D filter on range, fed measurement noise per Detection.range_source.

    Only ever constructed once a first real range measurement exists; a
    track with no range yet simply has no RangeKalmanFilter (range_m stays
    None on the emitted Track) rather than filtering a fabricated zero.
    """

    def __init__(self, range_m: float, measurement_noise: float) -> None:
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([range_m])
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.P *= RANGE_INITIAL_COVARIANCE
        kf.R = np.array([[measurement_noise]])
        kf.Q = np.array([[RANGE_PROCESS_NOISE]])
        self._kf = kf

    def predict(self) -> None:
        self._kf.predict()

    def update(self, range_m: float, measurement_noise: float) -> None:
        self._kf.R = np.array([[measurement_noise]])
        self._kf.update(np.array([range_m]))

    @property
    def value(self) -> float:
        return float(self._kf.x[0])


def measurement_noise_for_source(range_source: str) -> float:
    if range_source == "depth":
        return RANGE_MEASUREMENT_NOISE_DEPTH
    return RANGE_MEASUREMENT_NOISE_SIZE
