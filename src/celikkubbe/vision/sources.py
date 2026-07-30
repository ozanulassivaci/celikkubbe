"""FrameSource implementations that run with no real camera hardware.

``RealSenseSource`` is deliberately not here — it arrives once the D435i
is in hand and needs pyrealsense2. Everything downstream (detection,
tracking, the demo pipeline) develops and tests against these instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import CameraIntrinsics, Frame

# Stand-in horizontal FOV used to derive "estimated" intrinsics when no real
# calibration exists (matches the D435i RGB stream, a reasonable default for
# an arbitrary webcam too).
ASSUMED_HORIZONTAL_FOV_DEG = 69.0

# BGR, since every source here hands out images in cv2's native channel order.
_RED_BGR = (0, 0, 220)
_BACKGROUND_BGR = (40, 40, 40)


def estimated_intrinsics(
    width: int, height: int, hfov_deg: float = ASSUMED_HORIZONTAL_FOV_DEG
) -> CameraIntrinsics:
    """Plausible pinhole intrinsics for a source with no real calibration.

    Square pixels are assumed (fy = fx): with no calibration there is no
    basis to do otherwise.
    """
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fx,
        cx=width / 2.0,
        cy=height / 2.0,
        quality="estimated",
    )


@dataclass
class SyntheticSourceConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    num_targets: int = 3
    # Normalised (0-1) y position of each lane. Targets are assigned to
    # lanes round-robin. None -> evenly spaced across the frame.
    lane_fractions: tuple[float, ...] | None = None
    speed: float = 0.15  # normalised x-units per second
    radius_px: int = 18
    noise_std: float = 0.0  # gaussian pixel noise std, 0 disables it
    emit_depth: bool = False
    depth_m: float = 10.0  # simulated distance painted onto each target
    seed: int = 0


class SyntheticSource:
    """Procedurally draws red circles (balloons) moving along horizontal lanes.

    No files, no hardware, fully deterministic for a given seed and read()
    call count. This is the workhorse for tests and GUI development against
    a resolution and frame rate nothing else can guarantee.
    """

    def __init__(self, clock: Clock, config: SyntheticSourceConfig | None = None) -> None:
        self._clock = clock
        self._config = config or SyntheticSourceConfig()
        self._intrinsics = estimated_intrinsics(self._config.width, self._config.height)
        self._frame_index = 0
        self._rng: np.random.Generator = np.random.default_rng(self._config.seed)
        self._lane_fractions = self._config.lane_fractions or self._default_lanes()
        self._started = False

    def _default_lanes(self) -> tuple[float, ...]:
        n = max(1, min(3, self._config.num_targets))
        return tuple((i + 1) / (n + 1) for i in range(n))

    def start(self) -> None:
        self._frame_index = 0
        self._rng = np.random.default_rng(self._config.seed)
        self._started = True

    def stop(self) -> None:
        self._started = False

    @property
    def has_depth(self) -> bool:
        return self._config.emit_depth

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def _target_positions(self, t_sim: float) -> list[tuple[float, float]]:
        """Return (x, y) normalised centroids for every target at t_sim."""
        cfg = self._config
        lanes = self._lane_fractions
        positions = []
        for i in range(cfg.num_targets):
            lane_y = lanes[i % len(lanes)]
            # Offset by half a slot so a target never starts exactly at
            # x=0, where its circle would be clipped by the frame edge.
            start_x = ((i + 0.5) / max(cfg.num_targets, 1)) % 1.0
            x = (start_x + cfg.speed * t_sim) % 1.0
            positions.append((x, lane_y))
        return positions

    def read(self) -> Frame | None:
        if not self._started:
            return None
        cfg = self._config
        t_sim = self._frame_index / cfg.fps
        self._frame_index += 1

        image = np.full((cfg.height, cfg.width, 3), _BACKGROUND_BGR, dtype=np.uint8)
        depth = None
        if cfg.emit_depth:
            depth = np.zeros((cfg.height, cfg.width), dtype=np.float32)

        for x_norm, y_norm in self._target_positions(t_sim):
            cx = int(x_norm * cfg.width)
            cy = int(y_norm * cfg.height)
            cv2.circle(image, (cx, cy), cfg.radius_px, _RED_BGR, thickness=-1)
            if depth is not None:
                cv2.circle(depth, (cx, cy), cfg.radius_px, float(cfg.depth_m), thickness=-1)

        if cfg.noise_std > 0:
            noise = self._rng.normal(0.0, cfg.noise_std, size=image.shape)
            image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        return Frame(
            image=image,
            t=self._clock.now(),
            intrinsics=self._intrinsics,
            has_depth=cfg.emit_depth,
            depth=depth,
        )


class VideoFileSource:
    """Reads a video file, pacing reads to the file's native frame rate.

    ``read()`` never blocks: if the next frame is not due yet according to
    the injected Clock, it returns None, exactly like a live camera with
    nothing new to offer. This keeps tests deterministic under a FakeClock
    instead of requiring real sleeps.
    """

    def __init__(self, path: str, clock: Clock, loop: bool = True) -> None:
        self._path = path
        self._clock = clock
        self._loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._frame_period_s = 1.0 / 30.0
        self._next_due_t: float | None = None
        self._intrinsics: CameraIntrinsics | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video file: {self._path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        native_fps = fps if fps and fps > 0 else 30.0
        self._frame_period_s = 1.0 / native_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._intrinsics = estimated_intrinsics(width, height)
        self._next_due_t = None
        self._cap = cap

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        now = self._clock.now()
        if self._next_due_t is not None and now < self._next_due_t:
            return None

        ok, image = self._cap.read()
        if not ok:
            if not self._loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._cap.read()
            if not ok:
                return None

        self._next_due_t = now + self._frame_period_s
        assert self._intrinsics is not None
        return Frame(image=image, t=now, intrinsics=self._intrinsics, has_depth=False, depth=None)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def has_depth(self) -> bool:
        return False

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("start() has not been called")
        return self._intrinsics


class WebcamSource:
    """Reads a webcam by device index.

    Webcams silently substitute the closest resolution they support, so
    the intrinsics reported here reflect what the device actually
    delivered, queried back after ``set()``, never the requested value.
    """

    def __init__(
        self,
        device: int,
        clock: Clock,
        requested_width: int = 1280,
        requested_height: int = 720,
    ) -> None:
        self._device = device
        self._clock = clock
        self._requested_width = requested_width
        self._requested_height = requested_height
        self._cap: cv2.VideoCapture | None = None
        self._intrinsics: CameraIntrinsics | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise RuntimeError(f"could not open webcam device {self._device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        delivered_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        delivered_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._intrinsics = estimated_intrinsics(delivered_width, delivered_height)
        self._cap = cap

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        ok, image = self._cap.read()
        if not ok:
            return None
        assert self._intrinsics is not None
        return Frame(
            image=image,
            t=self._clock.now(),
            intrinsics=self._intrinsics,
            has_depth=False,
            depth=None,
        )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def has_depth(self) -> bool:
        return False

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("start() has not been called")
        return self._intrinsics
