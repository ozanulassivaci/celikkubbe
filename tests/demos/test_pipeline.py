from __future__ import annotations

import dataclasses
import io
from contextlib import redirect_stdout

import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.commands import Arm, Disarm, Fire, Goto, SoftEstop
from celikkubbe.core.types import IFF, HitResult, Track, TrackStatus
from celikkubbe.demos.pipeline import (
    SimulatedTurretLink,
    _apply_injection,
    _parse_injection,
    run,
    stub_aim_solutions,
)
from celikkubbe.io.sim_link import SimTurretLink
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


def test_pipeline_completes_full_s1_to_s6_cycle_for_a_single_hostile_target() -> None:
    # L2 sets IFF from colour, so a single hostile-coloured target passes
    # every gate given the demo's closer default range: the chain should
    # actually fire and cycle back to S1, not just reach S4_AIM.
    buf = io.StringIO()
    with redirect_stdout(buf):
        run(["--source", "synthetic", "--targets", "1", "--frames", "20"])
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert any("eng=S6_ASSESS" in line and "cmds=Fire" in line for line in lines)
    assert any("eng=S1_SEARCH" in line for line in lines[9:])  # cycles back after S6


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
    assert telemetry.motion_complete is False  # 90 deg/s * 0.05s = 4.5 deg, short of 10

    clock.advance(1.0)  # plenty of time to finish a 10 degree move
    telemetry = link.poll()
    assert telemetry.pan_deg == 10.0
    assert telemetry.motion_complete is True


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


def test_pipeline_runs_with_sim_link() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run(["--source", "synthetic", "--targets", "1", "--frames", "5", "--link", "sim"])
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert len(lines) == 5
    assert "mode=" in lines[0]


def test_pipeline_applies_injection_with_sim_link() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run(
            [
                "--source",
                "synthetic",
                "--targets",
                "1",
                "--frames",
                "5",
                "--link",
                "sim",
                "--inject",
                "estop@0.0s",
            ]
        )
    output = buf.getvalue()
    assert "[INJECT]" in output
    assert "estop@0.0s" in output


def test_pipeline_inject_requires_link_sim() -> None:
    with pytest.raises(SystemExit):
        run(["--source", "synthetic", "--inject", "estop@1s"])


def test_pipeline_inject_rejects_malformed_spec() -> None:
    with pytest.raises(SystemExit):
        run(["--source", "synthetic", "--link", "sim", "--inject", "bogus"])


def test_parse_injection_simple_spec() -> None:
    injection = _parse_injection("estop@5s")
    assert injection.at_t == 5.0
    assert injection.name == "estop"
    assert injection.value is None


def test_parse_injection_with_value() -> None:
    injection = _parse_injection("crc_errors:0.3@2.5s")
    assert injection.at_t == 2.5
    assert injection.name == "crc_errors"
    assert injection.value == 0.3


@pytest.mark.parametrize("spec", ["estop", "estop@5", "estop@fives"])
def test_parse_injection_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        _parse_injection(spec)


def test_apply_injection_dispatches_every_documented_fault() -> None:
    link = SimTurretLink(FakeClock())
    for name, value in [
        ("estop", None),
        ("release_estop", None),
        ("driver_alarm_pan", None),
        ("driver_alarm_tilt", None),
        ("clear_driver_alarm_pan", None),
        ("clear_driver_alarm_tilt", None),
        ("link_dropout", 0.5),
        ("crc_errors", 0.1),
        ("latency", 50.0),
    ]:
        _apply_injection(link, name, value)  # must not raise


def test_apply_injection_rejects_unknown_fault() -> None:
    link = SimTurretLink(FakeClock())
    with pytest.raises(ValueError):
        _apply_injection(link, "not_a_real_fault", None)


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
