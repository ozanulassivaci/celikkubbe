from __future__ import annotations

import pytest

from celikkubbe.core import config
from celikkubbe.core.cascade import Cascade, l1_health
from celikkubbe.core.clock import FakeClock
from celikkubbe.core.health import CameraHealth, DetectionHealth, InferenceHealth
from celikkubbe.core.types import Layer, ReasonCode, Stage


def test_cascade_starts_on_l1() -> None:
    cascade = Cascade(FakeClock())
    assert cascade.active_layer is Layer.L1
    assert cascade.fallback_reason is None


def test_cascade_falls_back_to_l2_when_l1_unhealthy() -> None:
    clock = FakeClock()
    cascade = Cascade(clock)
    layer, reason = cascade.update(False, ReasonCode.INFERENCE_SLOW, clock.now())
    assert layer is Layer.L2
    assert reason is ReasonCode.INFERENCE_SLOW


def test_cascade_recovers_after_five_healthy_probes_ten_seconds_apart() -> None:
    clock = FakeClock()
    cascade = Cascade(clock)
    cascade.update(False, ReasonCode.CAMERA_TIMEOUT, clock.now())
    assert cascade.active_layer is Layer.L2

    for _ in range(config.L1_RECOVERY_FRAMES - 1):
        clock.advance(config.L1_RECOVERY_INTERVAL_S)
        layer, _ = cascade.update(True, None, clock.now())
        assert layer is Layer.L2

    clock.advance(config.L1_RECOVERY_INTERVAL_S)
    layer, reason = cascade.update(True, None, clock.now())
    assert layer is Layer.L1
    assert reason is None


def test_cascade_probe_does_not_count_before_interval_elapses() -> None:
    clock = FakeClock()
    cascade = Cascade(clock)
    cascade.update(False, ReasonCode.CAMERA_TIMEOUT, clock.now())

    clock.advance(config.L1_RECOVERY_INTERVAL_S / 2)
    cascade.update(True, None, clock.now())
    clock.advance(config.L1_RECOVERY_INTERVAL_S / 2)
    layer, _ = cascade.update(True, None, clock.now())
    assert layer is Layer.L2


def test_cascade_probe_resets_consecutive_count_on_unhealthy_probe() -> None:
    clock = FakeClock()
    cascade = Cascade(clock)
    cascade.update(False, ReasonCode.CAMERA_TIMEOUT, clock.now())

    for _ in range(config.L1_RECOVERY_FRAMES - 1):
        clock.advance(config.L1_RECOVERY_INTERVAL_S)
        cascade.update(True, None, clock.now())

    clock.advance(config.L1_RECOVERY_INTERVAL_S)
    cascade.update(False, ReasonCode.CAMERA_TIMEOUT, clock.now())

    for _ in range(config.L1_RECOVERY_FRAMES - 1):
        clock.advance(config.L1_RECOVERY_INTERVAL_S)
        layer, _ = cascade.update(True, None, clock.now())
        assert layer is Layer.L2


def test_manual_override_suppresses_automatic_switching() -> None:
    clock = FakeClock()
    cascade = Cascade(clock)
    cascade.select_manual(Layer.L2, Stage.STAGE_2)
    assert cascade.manual_override is True

    layer, _ = cascade.update(True, None, clock.now())
    assert layer is Layer.L2

    clock.advance(config.L1_RECOVERY_INTERVAL_S * config.L1_RECOVERY_FRAMES)
    layer, _ = cascade.update(True, None, clock.now())
    assert layer is Layer.L2


def test_clearing_manual_override_restores_automatic_switching() -> None:
    clock = FakeClock()
    cascade = Cascade(clock)
    cascade.select_manual(Layer.L2, Stage.STAGE_2)
    cascade.clear_manual_override()
    assert cascade.manual_override is False

    layer, _ = cascade.update(False, ReasonCode.CAMERA_TIMEOUT, clock.now())
    assert layer is Layer.L2


def test_l3_manual_selection_requires_stage_1() -> None:
    cascade = Cascade(FakeClock())
    with pytest.raises(ValueError):
        cascade.select_manual(Layer.L3, Stage.STAGE_2)
    cascade.select_manual(Layer.L3, Stage.STAGE_1)
    assert cascade.active_layer is Layer.L3


def test_l1_health_reports_first_unhealthy_monitor() -> None:
    clock = FakeClock()
    inference = InferenceHealth()
    camera = CameraHealth(clock)
    camera.record_frame(clock.now())
    detection = DetectionHealth()

    healthy, reason = l1_health(inference, camera, detection)
    assert healthy is True
    assert reason is None

    for _ in range(config.INFERENCE_FAIL_FRAMES):
        inference.record(config.INFERENCE_FAIL_MS + 1)
    healthy, reason = l1_health(inference, camera, detection)
    assert healthy is False
    assert reason is ReasonCode.INFERENCE_SLOW
