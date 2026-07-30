from __future__ import annotations

import dataclasses
import io
from contextlib import redirect_stdout

import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.commands import Arm, Disarm, Fire, Goto, SoftEstop
from celikkubbe.core.types import IFF, HitResult, Track, TrackStatus
from celikkubbe.demos.pipeline import SimulatedTurretLink, run, stub_aim_solutions
from celikkubbe.vision.sources import estimated_intrinsics


def test_pipeline_runs_synthetic_source_for_a_few_frames() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run(["--source", "synthetic", "--targets", "2", "--frames", "5"])
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert len(lines) == 5
    assert "mode=" in lines[0]
    assert "eng=" in lines[0]
    assert "tracks=" in lines[-1]


def test_pipeline_reaches_s4_aim_and_reports_class_unknown_reason() -> None:
    # With only the L2 colour detector (cls is always None), autonomous
    # engagement should reliably stall in S4_AIM with CLASS_UNKNOWN once a
    # target is confirmed and selected — this is IFFGate working as
    # designed, not a demo bug.
    buf = io.StringIO()
    with redirect_stdout(buf):
        run(["--source", "synthetic", "--targets", "1", "--frames", "20"])
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert any("eng=S4_AIM" in line and "reason=CLASS_UNKNOWN" in line for line in lines)


def test_pipeline_requires_path_for_video_source() -> None:
    with pytest.raises(SystemExit):
        run(["--source", "video"])


def test_stub_aim_solutions_only_covers_confirmed_tracks() -> None:
    intrinsics = estimated_intrinsics(640, 480)
    confirmed = Track(
        track_id=1,
        cls=None,
        confidence=0.9,
        range_m=None,
        range_source="none",
        iff=IFF.UNKNOWN,
        bbox=(0.55, 0.55, 0.75, 0.75),
        velocity=(0.0, 0.0),
        status=TrackStatus.CONFIRMED,
        risk_score=0.0,
        frames_confirmed=5,
        last_seen_t=0.0,
    )
    tentative = dataclasses.replace(confirmed, track_id=2, status=TrackStatus.TENTATIVE)

    solutions = stub_aim_solutions([confirmed, tentative], intrinsics)

    assert set(solutions) == {1}
    az_deg, el_deg = solutions[1]
    assert az_deg > 0  # bbox is right-of-centre
    assert el_deg > 0  # bbox is below-centre (image y grows downward)


def test_simulated_turret_link_arms_and_slews_towards_goto() -> None:
    clock = FakeClock()
    link = SimulatedTurretLink(clock, max_vel_dps=90.0)
    assert link.connected is True

    link.send(Arm())
    link.send(Goto(az_deg=10.0, el_deg=0.0, max_vel_dps=90.0, max_accel_dps2=180.0))
    clock.advance(0.05)
    telemetry = link.poll()

    assert telemetry is not None
    assert telemetry.armed is True
    assert 0.0 < telemetry.pan_deg <= 10.0
    assert telemetry.target_pan_deg == 10.0


def test_simulated_turret_link_estop_marks_position_invalid() -> None:
    clock = FakeClock()
    link = SimulatedTurretLink(clock)
    link.send(Arm())
    link.send(SoftEstop())
    telemetry = link.poll()
    assert telemetry is not None
    assert telemetry.estop is True
    assert telemetry.position_valid is False


def test_simulated_turret_link_disarm_clears_armed_flag() -> None:
    clock = FakeClock()
    link = SimulatedTurretLink(clock)
    link.send(Arm())
    link.send(Disarm())
    telemetry = link.poll()
    assert telemetry is not None
    assert telemetry.armed is False


def test_simulated_turret_link_fire_schedules_a_kill_next_poll() -> None:
    clock = FakeClock()
    link = SimulatedTurretLink(clock)
    assert link.take_hit_result() is None
    link.send(Fire(count=1))
    assert link.take_hit_result() is HitResult.KILL
    assert link.take_hit_result() is None  # consumed, not repeated


def test_stub_aim_solutions_centred_track_points_straight_ahead() -> None:
    intrinsics = estimated_intrinsics(640, 480)
    centred = Track(
        track_id=1,
        cls=None,
        confidence=0.9,
        range_m=None,
        range_source="none",
        iff=IFF.UNKNOWN,
        bbox=(0.45, 0.45, 0.55, 0.55),
        velocity=(0.0, 0.0),
        status=TrackStatus.CONFIRMED,
        risk_score=0.0,
        frames_confirmed=5,
        last_seen_t=0.0,
    )
    az_deg, el_deg = stub_aim_solutions([centred], intrinsics)[1]
    assert az_deg == pytest.approx(0.0, abs=1e-9)
    assert el_deg == pytest.approx(0.0, abs=1e-9)
