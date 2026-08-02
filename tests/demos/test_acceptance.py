"""Acceptance tests for the full perception-to-fire loop against
SimTurretLink, driven by FakeClock for determinism.

These mirror Capabilities 3 and 4 from the competition video
requirements -- treat them as acceptance tests, not ordinary unit tests.
Deliberately white-box (direct access to SimTurretLink's own state,
precise control over simulated time) rather than driven through the CLI
like tests/demos/test_pipeline.py's black-box smoke tests: verifying
"e-stop injected mid-slew" needs to know exactly which tick that is, and
"the shot appears in ammo_fired" needs the turret object itself, neither
of which a stdout-scraping test can get at.
"""

from __future__ import annotations

import dataclasses

from celikkubbe.core import modes, priority
from celikkubbe.core.clock import FakeClock
from celikkubbe.core.commands import Zero
from celikkubbe.core.engagement import step as engagement_step
from celikkubbe.core.types import (
    Axis,
    EngagementState,
    HitResult,
    Layer,
    Mode,
    SelfTestItem,
    SelfTestResult,
    Stage,
    SystemState,
)
from celikkubbe.geometry.ballistics import DEFAULT_BALLISTIC_TABLE
from celikkubbe.geometry.frames import DEFAULT_TURRET_GEOMETRY
from celikkubbe.geometry.solver import AimSolver
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.tracking.manager import TrackManager
from celikkubbe.vision.l2_color import ColorDetector
from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig

_TICK_DT = 1.0 / 30.0

_SELF_TEST = SelfTestResult(
    items=(
        SelfTestItem("camera", True, None, None),
        SelfTestItem("stm32_link", True, None, None),
    )
)


class _Rig:
    """The same detect -> track -> prioritise -> aim (via the real
    geometry.solver.AimSolver) -> gate -> fire wiring demos/pipeline.py's
    run() assembles, but driven by an injected FakeClock instead of
    SystemClock + real-time pacing, so a test can advance time instantly
    and land an injection on an exact tick.
    """

    def __init__(self, clock: FakeClock, num_targets: int = 1) -> None:
        self.clock = clock
        self.source = SyntheticSource(
            clock, SyntheticSourceConfig(num_targets=num_targets, speed=0.0, range_m=3.0)
        )
        self.source.start()
        self.detector = ColorDetector()
        self.tracker = TrackManager(clock)
        self.turret = SimTurretLink(clock)
        # docs/protocol.md section 9's startup sequence: the operator
        # centres the turret by eye and the GUI sends `zero` per axis
        # before the PC will ever leave M2_STANDBY (modes.py refuses
        # OPERATIONAL until both homing bits are set).
        self.turret.send(Zero(Axis.PAN, 0.0))
        self.turret.send(Zero(Axis.TILT, 0.0))
        self.aim_solver = AimSolver(DEFAULT_TURRET_GEOMETRY, DEFAULT_BALLISTIC_TABLE)
        self.state = SystemState(
            stage=Stage.STAGE_2,
            mode=Mode.M1_INIT,
            engagement=EngagementState.S1_SEARCH,
            active_layer=Layer.L2,
            layer_manual_override=False,
            fallback_reason=None,
            tracks=(),
            selected_track_id=None,
            attempts={},
            deferred={},
            gate_fail_since=None,
            commanded_pan_deg=None,
            commanded_tilt_deg=None,
            telemetry=None,
            last_self_test=None,
        )
        self._last_ammo_fired = 0
        self.last_telemetry = None

    def tick(self) -> None:
        self.clock.advance(_TICK_DT)
        self.turret.send_heartbeat()

        frame = self.source.read()
        if frame is None:
            return

        detections, _ = self.detector.detect(frame)
        tracks = self.tracker.update(detections, now=self.clock.now())
        scored = [
            dataclasses.replace(
                t, risk_score=priority.compute_risk_score(t.cls, t.range_m, t.confidence)
            )
            for t in tracks
        ]

        telemetry = self.turret.poll()
        self.last_telemetry = telemetry

        self_test_result = _SELF_TEST if self.state.mode is Mode.M1_INIT else None
        operator_requested_mode = (
            Mode.M3_OPERATIONAL if self.state.mode is Mode.M2_STANDBY else None
        )
        next_mode, mode_commands = modes.step(
            self.state.mode,
            telemetry,
            self_test_result,
            camera_healthy=True,
            operator_requested_mode=operator_requested_mode,
            operator_ack_fault=False,
            now=self.clock.now(),
        )
        for cmd in mode_commands:
            self.turret.send(cmd)

        aim_solutions = self.aim_solver.solve_all(scored, frame.intrinsics)
        hit_result = None
        if self.turret.ammo_fired > self._last_ammo_fired:
            hit_result = HitResult.KILL
        self._last_ammo_fired = self.turret.ammo_fired

        self.state = dataclasses.replace(self.state, mode=next_mode, tracks=tuple(scored))
        result = engagement_step(
            state=self.state,
            tracks=scored,
            telemetry=telemetry,
            hit_result=hit_result,
            operator=None,
            aim_solutions=aim_solutions,
            now=self.clock.now(),
        )
        for cmd in result.commands:
            self.turret.send(cmd)
        self.state = dataclasses.replace(
            self.state,
            engagement=result.engagement,
            selected_track_id=result.selected_track_id,
            attempts=result.attempts,
            deferred=result.deferred,
            gate_fail_since=result.gate_fail_since,
            commanded_pan_deg=result.commanded_pan_deg,
            commanded_tilt_deg=result.commanded_tilt_deg,
            telemetry=telemetry,
        )

    def run_ticks(self, n: int) -> None:
        for _ in range(n):
            self.tick()


def test_full_loop_detects_tracks_aims_gates_and_fires_on_a_hostile_target() -> None:
    rig = _Rig(FakeClock())

    # Run tick-by-tick rather than a fixed count: the real (geometrically
    # correct) AimSolver converges on SETPOINT_ACK faster than the old
    # bearing-only stub did, so exactly how many engagement cycles fit in
    # a fixed tick budget is not a stable thing to assert on -- what
    # matters is that a shot lands and the state machine then cycles back
    # toward search, whenever that happens to occur.
    cycled_back_after_firing = False
    for _ in range(90):
        rig.tick()
        if rig.turret.ammo_fired >= 1 and rig.state.engagement in (
            EngagementState.S1_SEARCH,
            EngagementState.S2_ACQUIRE,
        ):
            cycled_back_after_firing = True
            break

    assert rig.turret.ammo_fired >= 1
    assert cycled_back_after_firing


def test_estop_injected_mid_slew_stops_motion_and_drives_mode_to_m4() -> None:
    rig = _Rig(FakeClock())

    # Run until the turret is actually mid-Goto (S4_AIM, motion not yet
    # complete) rather than injecting blind at a fixed tick count.
    for _ in range(60):
        rig.tick()
        if rig.state.engagement is EngagementState.S4_AIM and not rig.turret.true_pan_deg == 0.0:
            break
    assert rig.state.engagement is EngagementState.S4_AIM

    pan_at_injection = rig.turret.true_pan_deg
    rig.turret.inject_estop()
    rig.run_ticks(5)

    assert rig.state.mode is Mode.M4_SAFE
    assert rig.last_telemetry.estop is True
    assert rig.last_telemetry.position_valid is False
    # Motion actually stopped -- pan does not keep slewing toward the
    # target it had picked before the e-stop (only tilt droops; pan has
    # nothing pulling it, so "frozen" means "exactly unchanged").
    assert rig.turret.true_pan_deg == pan_at_injection
    assert rig.last_telemetry.pan_vel_dps == 0.0


def test_estop_injected_during_fire_sequence_stops_further_shots() -> None:
    rig = _Rig(FakeClock())

    for _ in range(60):
        rig.tick()
        if rig.state.engagement is EngagementState.S5_ENGAGE:
            break
    assert rig.state.engagement is EngagementState.S5_ENGAGE

    rig.turret.inject_estop()
    ammo_at_estop = rig.turret.ammo_fired

    rig.run_ticks(30)

    assert rig.turret.ammo_fired == ammo_at_estop  # no further shots after the estop
    assert rig.state.mode is Mode.M4_SAFE
