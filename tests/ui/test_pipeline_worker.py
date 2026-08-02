"""PipelineWorker tests drive tick() directly against FakeClock, exactly
like tests/io/test_link_worker.py drives LinkWorker.tick() -- the real
QThread is only exercised by the single lifecycle test at the bottom.

PipelineWorker never calls LinkWorker.tick() itself (that is LinkWorker's
own background thread's job once started); tests that need telemetry to
be present call worker's link_worker.tick() directly to simulate that
thread having run one iteration, the same white-box approach
tests/io/test_link_worker.py itself uses.
"""

from __future__ import annotations

import dataclasses

import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.commands import Arm
from celikkubbe.core.types import Axis, Layer, Mode, Stage
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.vision.sources import SyntheticSource, SyntheticSourceConfig, SyntheticTarget

_TICK_DT = 1.0 / 30.0


def _make_source(clock: FakeClock) -> SyntheticSource:
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=1, speed=0.0, range_m=3.0))
    source.start()
    return source


def _make_worker(clock: FakeClock | None = None) -> PipelineWorker:
    clock = clock or FakeClock()
    return PipelineWorker(_make_source(clock), SimTurretLink(clock), clock)


class _FlakySource:
    """Wraps a real source, raising once on a chosen call to exercise
    PipelineWorker's per-tick exception isolation.
    """

    def __init__(self, inner: SyntheticSource, fail_on_call: int) -> None:
        self._inner = inner
        self._fail_on_call = fail_on_call
        self._calls = 0

    def start(self) -> None:
        self._inner.start()

    def stop(self) -> None:
        self._inner.stop()

    @property
    def has_depth(self) -> bool:
        return self._inner.has_depth

    @property
    def intrinsics(self):
        return self._inner.intrinsics

    def read(self):
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise RuntimeError("boom")
        return self._inner.read()


def test_emits_one_snapshot_per_successful_tick(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    received = []
    worker.snapshot_ready.connect(received.append)

    for _ in range(10):
        clock.advance(_TICK_DT)
        worker.tick()

    assert len(received) == 10
    assert all(s.frame is not None for s in received)


def test_tick_exception_is_isolated_and_loop_continues(qtbot):
    clock = FakeClock()
    inner = _make_source(clock)
    source = _FlakySource(inner, fail_on_call=3)
    worker = PipelineWorker(source, SimTurretLink(clock), clock)

    snapshots = []
    errors = []
    worker.snapshot_ready.connect(snapshots.append)
    worker.error.connect(errors.append)

    for _ in range(5):
        clock.advance(_TICK_DT)
        worker.tick()

    assert len(errors) == 1
    assert "boom" in errors[0]
    # Every tick except the one that raised still produced a snapshot --
    # one bad frame did not kill the loop.
    assert len(snapshots) == 4


def test_latest_snapshot_survives_when_ticks_outpace_consumption(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    seen = []
    worker.snapshot_ready.connect(seen.append)

    for _ in range(5):
        clock.advance(_TICK_DT)
        worker.tick()

    # The signal itself still fires once per tick (Qt handles that
    # cheaply)...
    assert len(seen) == 5
    # ...but the single-slot buffer a slow GUI would actually read from
    # only ever holds the newest one: there is no way to get an earlier
    # tick's snapshot back out of the worker once a later one has landed.
    latest = worker.latest_snapshot
    assert latest is not None
    assert latest is seen[-1]
    assert latest.t == pytest.approx(clock.now())


def test_shutdown_joins_cleanly_with_no_leaked_thread(qtbot):
    clock = FakeClock()
    source = SyntheticSource(clock, SyntheticSourceConfig(num_targets=1, speed=0.0, range_m=3.0))
    worker = PipelineWorker(source, SimTurretLink(clock), clock)

    worker.start()
    qtbot.waitUntil(worker.isRunning, timeout=2000)
    worker.stop()
    joined = worker.wait(2000)

    assert joined
    assert not worker.isRunning()


def test_emitted_snapshot_is_frozen(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    clock.advance(_TICK_DT)
    worker.tick()

    snapshot = worker.latest_snapshot
    assert snapshot is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.t = 999.0  # type: ignore[misc]


def _settle(worker: PipelineWorker, clock: FakeClock, max_ticks: int = 200) -> None:
    for _ in range(max_ticks):
        if worker._state.mode is not Mode.M1_INIT:
            return
        clock.advance(_TICK_DT)
        worker._link_worker.tick()
        worker.tick()


def test_self_test_passes_and_advances_to_standby_once_healthy(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)

    _settle(worker, clock)

    assert worker._state.mode is Mode.M2_STANDBY
    assert worker._state.last_self_test is not None
    assert worker._state.last_self_test.passed


def test_all_six_self_test_items_appear_and_pass(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)

    _settle(worker, clock)

    result = worker._state.last_self_test
    assert result is not None
    names = {item.name for item in result.items}
    assert names == {
        "camera",
        "stm32_link",
        "pan_tilt_move",
        "fire_lock",
        "inference_time",
        "driver_alarms",
    }
    assert all(item.passed for item in result.items), result.items


def test_fire_lock_check_rejects_fire_while_disarmed(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)

    _settle(worker, clock)

    fire_lock = next(i for i in worker._state.last_self_test.items if i.name == "fire_lock")
    assert fire_lock.passed
    # Confirmed by never having armed anything during self-test: no shot
    # was actually accepted by the link.
    assert worker._link.ammo_fired == 0


def test_engagement_does_not_command_the_turret_during_self_test(qtbot):
    """Regression test: engagement.step() runs regardless of Mode by
    design (see CLAUDE.md), so a target visible while M1_INIT is still
    running would otherwise select it and start commanding Goto/Fire
    before self-test has confirmed the axes even work -- and directly
    fight the pan/tilt verification move's own Goto, since both target
    the same link. PipelineWorker must not run engagement_step at all
    while still in M1_INIT.
    """
    clock = FakeClock()
    worker = _make_worker(clock)  # source has one real target, per _make_source

    for _ in range(5):
        clock.advance(_TICK_DT)
        worker._link_worker.tick()
        worker.tick()
        if worker._state.mode is Mode.M1_INIT:
            assert worker._state.engagement.value == "S1_SEARCH"
            assert worker._state.selected_track_id is None

    _settle(worker, clock)
    assert worker._state.mode is Mode.M2_STANDBY


def test_off_boresight_target_produces_real_slew_and_reaches_fire(qtbot):
    """0i: a target well off boresight must actually move the turret
    (not stay at the synthetic default of 0,0), and motion_complete must
    correctly gate S4_AIM -> S5_ENGAGE -- i.e. AngleGate genuinely blocks
    until the axes have arrived, not just once selected.
    """
    clock = FakeClock()
    targets = (
        SyntheticTarget(
            color_hex="#F50A0A", size_m=0.4, lane_fraction=0.2, range_m=5.0, start_x=0.1
        ),
    )
    source = SyntheticSource(clock, SyntheticSourceConfig(targets=targets))
    source.start()
    link = SimTurretLink(clock)
    worker = PipelineWorker(source, link, clock)

    _settle(worker, clock)
    assert worker._state.mode is Mode.M2_STANDBY

    worker._state = dataclasses.replace(worker._state, mode=Mode.M3_OPERATIONAL)
    link.send(Arm())  # what modes.step() itself sends on a real M2 -> M3 transition

    reached_s5 = False
    for _ in range(150):
        clock.advance(_TICK_DT)
        worker._link_worker.tick()
        worker.tick()
        if worker._state.engagement.value in ("S5_ENGAGE", "S6_ASSESS") or link.ammo_fired > 0:
            reached_s5 = True
            break

    assert reached_s5, worker._state.engagement
    telemetry = worker.latest_snapshot.telemetry
    # The target is well off-axis; a turret that never actually slewed
    # would still read (0, 0).
    assert abs(telemetry.pan_deg) > 5.0 or abs(telemetry.tilt_deg) > 5.0


def test_driver_alarm_fails_that_self_test_item_specifically(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    worker._link.inject_driver_alarm(Axis.PAN)

    for _ in range(10):
        clock.advance(_TICK_DT)
        worker._link_worker.tick()
        worker.tick()

    result = worker._state.last_self_test
    assert result is not None
    alarms_item = next(i for i in result.items if i.name == "driver_alarms")
    assert not alarms_item.passed
    assert worker._state.mode is Mode.M1_INIT


def test_retry_self_test_forces_submission_while_still_failing(qtbot):
    clock = FakeClock()
    source = _make_source(clock)
    # A link that never answers poll() -- stm32_link self-test item never
    # passes on its own, so mode would otherwise sit in M1_INIT forever.
    worker = PipelineWorker(source, _DeadLink(), clock)

    clock.advance(_TICK_DT)
    worker.tick()
    assert worker._state.mode is Mode.M1_INIT
    assert worker._state.last_self_test is not None
    assert not worker._state.last_self_test.passed

    worker.retry_self_test()
    clock.advance(_TICK_DT)
    worker.tick()

    assert worker._state.mode is Mode.M4_SAFE


class _DeadLink:
    connected = False

    def send(self, cmd) -> None:
        pass

    def poll(self):
        return None


# --- right-panel control methods ---


def test_request_estop_reaches_the_link_without_a_pipeline_tick(qtbot):
    """The whole point of request_estop() is that it must not wait for
    _tick_inner() to get around to calling _send() -- so this deliberately
    never calls worker.tick() at all, only worker._link_worker.tick() to
    simulate LinkWorker's own independent background thread having run.
    """
    clock = FakeClock()
    worker = _make_worker(clock)
    worker.request_estop()
    worker._link_worker.tick()
    telemetry = worker._link_worker.latest_telemetry
    assert telemetry is not None
    assert telemetry.estop


def test_request_arm_and_disarm_toggle_telemetry_armed(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    worker.request_arm(True)
    worker._link_worker.tick()
    assert worker._link_worker.latest_telemetry.armed

    worker.request_arm(False)
    worker._link_worker.tick()
    assert not worker._link_worker.latest_telemetry.armed


def test_request_zero_sets_position_and_homed_flag_for_the_given_axis(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    worker.request_zero(Axis.PAN)
    worker._link_worker.tick()
    telemetry = worker._link_worker.latest_telemetry
    assert telemetry.pan_deg == pytest.approx(0.0)
    assert telemetry.homed_pan
    assert not telemetry.homed_tilt


def test_request_jog_moves_the_axis_and_request_stop_halts_it(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    worker.request_jog(Axis.PAN, 1, 20.0)
    worker._link_worker.tick()
    clock.advance(0.2)
    worker._link_worker.tick()
    moved_pan = worker._link_worker.latest_telemetry.pan_deg
    assert moved_pan > 0.0

    worker.request_stop()
    worker._link_worker.tick()
    stopped_pan = worker._link_worker.latest_telemetry.pan_deg
    clock.advance(0.2)
    worker._link_worker.tick()
    assert worker._link_worker.latest_telemetry.pan_deg == pytest.approx(stopped_pan, abs=0.05)


def test_set_stage_updates_state_on_the_next_tick(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)
    assert worker._state.stage is not Stage.STAGE_1

    worker.set_stage(Stage.STAGE_1)
    clock.advance(_TICK_DT)
    worker._link_worker.tick()
    worker.tick()

    assert worker._state.stage is Stage.STAGE_1


def test_select_layer_updates_active_layer_and_marks_operator_chosen(qtbot):
    clock = FakeClock()
    worker = _make_worker(clock)

    worker.select_layer(Layer.L2)
    clock.advance(_TICK_DT)
    worker._link_worker.tick()
    worker.tick()

    assert worker._state.active_layer is Layer.L2
    assert worker._state.layer_manual_override is True


def test_leaving_stage_1_clears_an_active_l3_override_back_to_l2(qtbot):
    """L3 (Tam Manuel) is a Stage 1-only override (core/cascade.py's own
    docstring) -- leaving Stage 1 while it is active must not leave that
    otherwise-unreachable (stage, layer) combination sitting in
    SystemState.
    """
    clock = FakeClock()
    worker = _make_worker(clock)

    worker.set_stage(Stage.STAGE_1)
    clock.advance(_TICK_DT)
    worker._link_worker.tick()
    worker.tick()
    worker.select_layer(Layer.L3)
    clock.advance(_TICK_DT)
    worker._link_worker.tick()
    worker.tick()
    assert worker._state.active_layer is Layer.L3

    worker.set_stage(Stage.STAGE_2)
    clock.advance(_TICK_DT)
    worker._link_worker.tick()
    worker.tick()

    assert worker._state.stage is Stage.STAGE_2
    assert worker._state.active_layer is Layer.L2
