"""Sliding-window health monitors consumed by the cascade and mode machine.

Each monitor exposes ``healthy``, ``value``, and ``reason`` and never
executes anything: callers decide what to do with an unhealthy monitor.
"""

from __future__ import annotations

from collections import deque

from celikkubbe.core import config
from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import ReasonCode


class InferenceHealth:
    """Tracks GPU inference latency with a consecutive-failure counter.

    A rolling average would let a single slow frame (garbage collection, OS
    scheduling) drag down the mean without ever being wrong on its own. A
    consecutive counter only trips after ``fail_frames`` bad frames in a
    row, which is what actually indicates the pipeline is unhealthy.
    """

    def __init__(
        self,
        fail_ms: float = config.INFERENCE_FAIL_MS,
        fail_frames: int = config.INFERENCE_FAIL_FRAMES,
    ) -> None:
        self._fail_ms = fail_ms
        self._fail_frames = fail_frames
        self._consecutive_slow = 0
        self._last_latency_ms = 0.0

    def record(self, latency_ms: float) -> None:
        self._last_latency_ms = latency_ms
        if latency_ms > self._fail_ms:
            self._consecutive_slow += 1
        else:
            self._consecutive_slow = 0

    @property
    def healthy(self) -> bool:
        return self._consecutive_slow < self._fail_frames

    @property
    def value(self) -> float:
        return self._last_latency_ms

    @property
    def reason(self) -> ReasonCode | None:
        return None if self.healthy else ReasonCode.INFERENCE_SLOW


class CameraHealth:
    """Tracks camera liveness (timeout) and measured FPS over a rolling window."""

    def __init__(
        self,
        clock: Clock,
        timeout_ms: float = config.CAMERA_TIMEOUT_MS,
        min_fps: float = config.MIN_FPS,
        fps_window_s: float = config.CAMERA_FPS_WINDOW_S,
    ) -> None:
        self._clock = clock
        self._timeout_ms = timeout_ms
        self._min_fps = min_fps
        self._fps_window_s = fps_window_s
        self._last_frame_t: float | None = None
        self._frame_times: deque[float] = deque()

    def record_frame(self, t: float) -> None:
        self._last_frame_t = t
        self._frame_times.append(t)
        cutoff = t - self._fps_window_s
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()

    @property
    def _timed_out(self) -> bool:
        if self._last_frame_t is None:
            return True
        return (self._clock.now() - self._last_frame_t) * 1000.0 > self._timeout_ms

    @property
    def value(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    @property
    def healthy(self) -> bool:
        if self._timed_out:
            return False
        if len(self._frame_times) < 2:
            return True
        return self.value >= self._min_fps

    @property
    def reason(self) -> ReasonCode | None:
        return None if self.healthy else ReasonCode.CAMERA_TIMEOUT


class LinkHealth:
    """Tracks how stale the last telemetry packet from the MCU is."""

    def __init__(self, clock: Clock, stale_ms: float = config.TELEMETRY_STALE_MS) -> None:
        self._clock = clock
        self._stale_ms = stale_ms
        self._last_t: float | None = None

    def record(self, t: float) -> None:
        self._last_t = t

    @property
    def value(self) -> float:
        if self._last_t is None:
            return float("inf")
        return (self._clock.now() - self._last_t) * 1000.0

    @property
    def healthy(self) -> bool:
        if self._last_t is None:
            return False
        return self.value <= self._stale_ms

    @property
    def reason(self) -> ReasonCode | None:
        return None if self.healthy else ReasonCode.LINK_TIMEOUT


class DetectionHealth:
    """Tracks mean detection confidence over the last few frames."""

    def __init__(
        self,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        window: int = config.ACQUIRE_FRAMES,
    ) -> None:
        self._threshold = confidence_threshold
        self._window: deque[float] = deque(maxlen=window)

    def record(self, confidence: float | None) -> None:
        self._window.append(confidence if confidence is not None else 0.0)

    @property
    def value(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def healthy(self) -> bool:
        if not self._window:
            return True
        return self.value >= self._threshold

    @property
    def reason(self) -> ReasonCode | None:
        return None if self.healthy else ReasonCode.LOW_CONFIDENCE


class DepthHealth:
    """Tracks the fraction of recent frames reporting usable depth."""

    def __init__(
        self,
        min_valid_ratio: float = config.DEPTH_MIN_VALID_RATIO,
        window: int = config.ACQUIRE_FRAMES,
    ) -> None:
        self._min_valid_ratio = min_valid_ratio
        self._window: deque[bool] = deque(maxlen=window)

    def record(self, depth_valid: bool) -> None:
        self._window.append(depth_valid)

    @property
    def value(self) -> float:
        if not self._window:
            return 1.0
        return sum(1 for v in self._window if v) / len(self._window)

    @property
    def healthy(self) -> bool:
        if not self._window:
            return True
        return self.value >= self._min_valid_ratio

    @property
    def reason(self) -> ReasonCode | None:
        return None if self.healthy else ReasonCode.DEPTH_UNRELIABLE
