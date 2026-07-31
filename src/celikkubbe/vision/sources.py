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

_BACKGROUND_BGR = (40, 40, 40)

# Confirmed competition target colours (see vision/l2_color.py) and the
# three known physical sizes, used to auto-generate a plausible mix of
# targets when the caller does not specify one explicitly.
HOSTILE_HEX = "#F50A0A"
FRIENDLY_HEX = "#00A3E0"
_DEFAULT_COLORS_HEX = (HOSTILE_HEX, FRIENDLY_HEX)
_DEFAULT_SIZES_M = (0.30, 0.40, 0.50)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert "#RRGGBB" to the (B, G, R) tuple cv2 drawing calls expect."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


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


def _default_lanes(num_targets: int) -> tuple[float, ...]:
    n = max(1, min(3, num_targets))
    return tuple((i + 1) / (n + 1) for i in range(n))


@dataclass(frozen=True)
class SyntheticTarget:
    """One rendered target: colour, true physical size, lane and motion."""

    color_hex: str
    size_m: float
    lane_fraction: float  # normalised (0-1) y position
    speed: float = 0.0  # normalised x-units per second
    range_m: float = 10.0  # simulated distance -> apparent pixel size + depth
    start_x: float = 0.0  # normalised x position at t=0


@dataclass
class SyntheticSourceConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    num_targets: int = 3
    # Explicit target list. None -> auto-generate num_targets targets
    # cycling through the confirmed colours and known sizes, using the
    # speed/range_m/lane_fractions below.
    targets: tuple[SyntheticTarget, ...] | None = None
    lane_fractions: tuple[float, ...] | None = None
    speed: float = 0.15  # normalised x-units per second
    range_m: float = 10.0
    noise_std: float = 0.0  # gaussian pixel noise std, 0 disables it
    emit_depth: bool = False
    seed: int = 0

    def resolve_targets(self) -> tuple[SyntheticTarget, ...]:
        if self.targets is not None:
            return self.targets
        n = max(self.num_targets, 0)
        if n == 0:
            return ()
        lanes = self.lane_fractions or _default_lanes(n)
        resolved = []
        for i in range(n):
            resolved.append(
                SyntheticTarget(
                    color_hex=_DEFAULT_COLORS_HEX[i % len(_DEFAULT_COLORS_HEX)],
                    size_m=_DEFAULT_SIZES_M[i % len(_DEFAULT_SIZES_M)],
                    lane_fraction=lanes[i % len(lanes)],
                    speed=self.speed,
                    range_m=self.range_m,
                    start_x=((i + 0.5) / n) % 1.0,
                )
            )
        return tuple(resolved)


class SyntheticSource:
    """Procedurally draws coloured targets moving along horizontal lanes.

    No files, no hardware, fully deterministic for a given seed and read()
    call count. This is the workhorse for tests and GUI development against
    a resolution and frame rate nothing else can guarantee. Apparent pixel
    size is derived from each target's true size_m and simulated range_m
    through the same pinhole model l2_color.py uses to estimate range, so
    size-based range estimation can be exercised end to end against it.
    """

    def __init__(self, clock: Clock, config: SyntheticSourceConfig | None = None) -> None:
        self._clock = clock
        self._config = config or SyntheticSourceConfig()
        self._intrinsics = estimated_intrinsics(self._config.width, self._config.height)
        self._targets = self._config.resolve_targets()
        self._frame_index = 0
        self._rng: np.random.Generator = np.random.default_rng(self._config.seed)
        self._started = False

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

    def _target_position(self, target: SyntheticTarget, t_sim: float) -> tuple[float, float]:
        x = (target.start_x + target.speed * t_sim) % 1.0
        return x, target.lane_fraction

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

        for target in self._targets:
            x_norm, y_norm = self._target_position(target, t_sim)
            cx = int(x_norm * cfg.width)
            cy = int(y_norm * cfg.height)
            diameter_px = self._intrinsics.fx * target.size_m / target.range_m
            radius_px = max(1, int(round(diameter_px / 2.0)))
            color_bgr = hex_to_bgr(target.color_hex)
            cv2.circle(image, (cx, cy), radius_px, color_bgr, thickness=-1)
            if depth is not None:
                cv2.circle(depth, (cx, cy), radius_px, float(target.range_m), thickness=-1)

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
