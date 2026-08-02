from __future__ import annotations

import pytest

from celikkubbe.core import config
from celikkubbe.core.types import IFF, Track, TrackStatus
from celikkubbe.geometry.ballistics import BallisticTable
from celikkubbe.geometry.calibration import BoresightCorrection, BoresightTable
from celikkubbe.geometry.frames import TurretGeometry
from celikkubbe.geometry.projection import pixel_to_turret_point, point_to_angles
from celikkubbe.geometry.solver import AimSolver
from celikkubbe.vision.sources import CameraIntrinsics, estimated_intrinsics

_ZERO_GEOM = TurretGeometry(cam_offset_x_m=0.0, cam_offset_y_m=0.0, cam_offset_z_m=0.0)
_ZERO_BALLISTICS = BallisticTable(
    ranges_m=(5.0, 15.0), drop_deg=(0.0, 0.0), muzzle_velocity_ms=100.0
)
_INTR = estimated_intrinsics(640, 480)  # quality="estimated"
_RELIABLE_INTR = CameraIntrinsics(
    width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0, quality="factory"
)


def _principal_point_norm(intr: CameraIntrinsics = _INTR) -> tuple[float, float]:
    return intr.cx / intr.width, intr.cy / intr.height


def _make_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (0.45, 0.45, 0.55, 0.55),
    velocity: tuple[float, float] = (0.0, 0.0),
    range_m: float | None = 10.0,
    range_source: str = "depth",
    status: TrackStatus = TrackStatus.CONFIRMED,
) -> Track:
    return Track(
        track_id=track_id,
        cls=None,
        confidence=0.9,
        range_m=range_m,
        range_source=range_source,
        iff=IFF.HOSTILE,
        bbox=bbox,
        velocity=velocity,
        status=status,
        risk_score=50.0,
        frames_confirmed=5,
        last_seen_t=0.0,
    )


def test_end_to_end_known_position_and_range_gives_expected_angles() -> None:
    solver = AimSolver(_ZERO_GEOM, _ZERO_BALLISTICS)
    cu, cv = _principal_point_norm()
    track = _make_track(bbox=(cu - 0.01, cv - 0.01, cu + 0.01, cv + 0.01), range_m=10.0)

    solution = solver.solve(track, _INTR)

    assert solution is not None
    assert solution.az_deg == pytest.approx(0.0, abs=1e-6)
    assert solution.el_deg == pytest.approx(0.0, abs=1e-6)
    assert solution.range_m == 10.0
    assert solution.lead_az_deg == 0.0
    assert solution.lead_el_deg == 0.0
    assert solution.drop_deg == 0.0


def test_components_sum_to_the_final_angle() -> None:
    ballistics = BallisticTable(ranges_m=(5.0, 15.0), drop_deg=(0.2, 0.6), muzzle_velocity_ms=100.0)
    solver = AimSolver(_ZERO_GEOM, ballistics)
    bbox = (0.55, 0.40, 0.65, 0.50)
    track = _make_track(bbox=bbox, velocity=(0.05, -0.02), range_m=10.0)

    solution = solver.solve(track, _INTR)
    assert solution is not None

    u_center = (bbox[0] + bbox[2]) / 2.0
    v_center = (bbox[1] + bbox[3]) / 2.0
    p_turret = pixel_to_turret_point(u_center, v_center, 10.0, _INTR, _ZERO_GEOM)
    base_az, base_el = point_to_angles(p_turret)

    assert solution.az_deg == pytest.approx(base_az + solution.lead_az_deg)
    assert solution.el_deg == pytest.approx(base_el + solution.lead_el_deg + solution.drop_deg)


@pytest.mark.parametrize(
    ("range_source", "intr", "expected"),
    [
        ("depth", _RELIABLE_INTR, "high"),
        ("size", _RELIABLE_INTR, "medium"),
        ("depth", _INTR, "low"),  # estimated intrinsics downgrade even depth-derived range
        ("size", _INTR, "low"),
    ],
)
def test_confidence_downgrades_per_range_source_and_intrinsics(
    range_source: str, intr: CameraIntrinsics, expected: str
) -> None:
    solver = AimSolver(_ZERO_GEOM, _ZERO_BALLISTICS)
    track = _make_track(range_source=range_source, range_m=10.0)
    solution = solver.solve(track, intr)
    assert solution is not None
    assert solution.confidence == expected


def test_missing_range_yields_low_confidence_solution_not_none() -> None:
    solver = AimSolver(_ZERO_GEOM, _ZERO_BALLISTICS)
    track = _make_track(range_m=None, range_source="none")

    solution = solver.solve(track, _RELIABLE_INTR)

    assert solution is not None
    assert solution.confidence == "low"
    assert solution.range_source == "assumed"
    assert solution.range_m == config.DEFAULT_RANGE_M
    assert any("no range available" in w for w in solution.warnings)


def test_non_confirmed_track_returns_none() -> None:
    solver = AimSolver(_ZERO_GEOM, _ZERO_BALLISTICS)
    track = _make_track(status=TrackStatus.TENTATIVE)
    assert solver.solve(track, _INTR) is None


def test_angles_clamped_to_software_limits_with_warning() -> None:
    # An unrealistic boresight offset, well beyond TILT_LIMIT_DEG, forces
    # the clamp deterministically rather than relying on an extreme bbox
    # position.
    boresight = BoresightTable((BoresightCorrection(0.0, 100.0, "test", 10.0, "forced clamp"),))
    solver = AimSolver(_ZERO_GEOM, _ZERO_BALLISTICS, boresight=boresight)
    track = _make_track(range_m=10.0)

    solution = solver.solve(track, _INTR)

    assert solution is not None
    assert solution.el_deg == config.TILT_LIMIT_DEG[1]
    assert any("clamped" in w for w in solution.warnings)


def test_ballistic_extrapolation_warns() -> None:
    ballistics = BallisticTable(ranges_m=(5.0, 10.0), drop_deg=(0.1, 0.2), muzzle_velocity_ms=100.0)
    solver = AimSolver(_ZERO_GEOM, ballistics)
    track = _make_track(range_m=20.0)  # beyond the table's 10m span

    solution = solver.solve(track, _INTR)

    assert solution is not None
    assert any("extrapolated" in w for w in solution.warnings)


def test_solve_all_returns_one_entry_per_confirmed_track() -> None:
    solver = AimSolver(_ZERO_GEOM, _ZERO_BALLISTICS)
    tracks = [
        _make_track(track_id=1, status=TrackStatus.CONFIRMED),
        _make_track(track_id=2, status=TrackStatus.TENTATIVE),
        _make_track(track_id=3, status=TrackStatus.CONFIRMED),
    ]

    solutions = solver.solve_all(tracks, _INTR)

    assert set(solutions) == {1, 3}
    for az_deg, el_deg in solutions.values():
        assert isinstance(az_deg, float)
        assert isinstance(el_deg, float)
