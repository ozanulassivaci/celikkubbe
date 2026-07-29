from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.clock import FakeClock
from celikkubbe.core.health import (
    CameraHealth,
    DepthHealth,
    DetectionHealth,
    InferenceHealth,
    LinkHealth,
)
from celikkubbe.core.types import ReasonCode


def test_inference_health_ignores_single_slow_frame() -> None:
    health = InferenceHealth()
    for _ in range(config.INFERENCE_FAIL_FRAMES - 1):
        health.record(config.INFERENCE_FAIL_MS + 1)
    assert health.healthy is True
    assert health.reason is None


def test_inference_health_trips_on_consecutive_slow_frames() -> None:
    health = InferenceHealth()
    for _ in range(config.INFERENCE_FAIL_FRAMES):
        health.record(config.INFERENCE_FAIL_MS + 1)
    assert health.healthy is False
    assert health.reason is ReasonCode.INFERENCE_SLOW


def test_inference_health_recovers_after_a_fast_frame() -> None:
    health = InferenceHealth()
    for _ in range(config.INFERENCE_FAIL_FRAMES - 1):
        health.record(config.INFERENCE_FAIL_MS + 1)
    health.record(1.0)
    for _ in range(config.INFERENCE_FAIL_FRAMES - 1):
        health.record(config.INFERENCE_FAIL_MS + 1)
    assert health.healthy is True


def test_camera_health_unhealthy_before_first_frame() -> None:
    clock = FakeClock()
    health = CameraHealth(clock)
    assert health.healthy is False
    assert health.reason is ReasonCode.CAMERA_TIMEOUT


def test_camera_health_times_out_without_new_frames() -> None:
    clock = FakeClock()
    health = CameraHealth(clock)
    health.record_frame(clock.now())
    clock.advance(config.CAMERA_TIMEOUT_MS / 1000.0 + 0.01)
    assert health.healthy is False
    assert health.reason is ReasonCode.CAMERA_TIMEOUT


def test_camera_health_healthy_with_fresh_frames() -> None:
    clock = FakeClock()
    health = CameraHealth(clock)
    health.record_frame(clock.now())
    clock.advance(0.01)
    health.record_frame(clock.now())
    assert health.healthy is True
    assert health.reason is None


def test_link_health_unhealthy_before_first_packet() -> None:
    clock = FakeClock()
    health = LinkHealth(clock)
    assert health.healthy is False
    assert health.reason is ReasonCode.LINK_TIMEOUT


def test_link_health_goes_stale() -> None:
    clock = FakeClock()
    health = LinkHealth(clock)
    health.record(clock.now())
    clock.advance(config.TELEMETRY_STALE_MS / 1000.0 + 0.01)
    assert health.healthy is False


def test_detection_health_unhealthy_below_threshold() -> None:
    health = DetectionHealth()
    for _ in range(config.ACQUIRE_FRAMES):
        health.record(0.1)
    assert health.healthy is False
    assert health.reason is ReasonCode.LOW_CONFIDENCE


def test_detection_health_healthy_above_threshold() -> None:
    health = DetectionHealth()
    for _ in range(config.ACQUIRE_FRAMES):
        health.record(0.95)
    assert health.healthy is True


def test_depth_health_unhealthy_when_mostly_invalid() -> None:
    health = DepthHealth()
    for _ in range(config.ACQUIRE_FRAMES):
        health.record(False)
    assert health.healthy is False
    assert health.reason is ReasonCode.DEPTH_UNRELIABLE


def test_depth_health_healthy_when_mostly_valid() -> None:
    health = DepthHealth()
    for _ in range(config.ACQUIRE_FRAMES):
        health.record(True)
    assert health.healthy is True
