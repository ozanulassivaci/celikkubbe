from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.vision.sources import (
    SyntheticSource,
    SyntheticSourceConfig,
    VideoFileSource,
    WebcamSource,
)


def _started_source(**overrides) -> SyntheticSource:
    config = SyntheticSourceConfig(**overrides)
    source = SyntheticSource(FakeClock(), config)
    source.start()
    return source


def test_synthetic_source_produces_requested_resolution() -> None:
    source = _started_source(width=320, height=240)
    frame = source.read()
    assert frame is not None
    assert frame.image.shape == (240, 320, 3)
    assert frame.intrinsics.width == 320
    assert frame.intrinsics.height == 240
    assert frame.intrinsics.quality == "estimated"


def test_synthetic_source_returns_none_before_start() -> None:
    source = SyntheticSource(FakeClock(), SyntheticSourceConfig())
    assert source.read() is None


def test_synthetic_source_reproducible_for_seed() -> None:
    a = _started_source(seed=42, noise_std=5.0)
    b = _started_source(seed=42, noise_std=5.0)
    for _ in range(5):
        frame_a = a.read()
        frame_b = b.read()
        assert np.array_equal(frame_a.image, frame_b.image)


def test_synthetic_source_different_seeds_diverge_with_noise() -> None:
    a = _started_source(seed=1, noise_std=20.0)
    b = _started_source(seed=2, noise_std=20.0)
    frame_a = a.read()
    frame_b = b.read()
    assert not np.array_equal(frame_a.image, frame_b.image)


def test_synthetic_source_targets_move_over_time() -> None:
    source = _started_source(width=640, height=480, num_targets=1, speed=0.2, fps=30.0)
    first = source.read()
    for _ in range(9):
        source.read()
    tenth = source.read()

    def reddest_column(image: np.ndarray) -> int:
        redness = image[:, :, 2].astype(int) - image[:, :, 0].astype(int)
        return int(np.argmax(redness.sum(axis=0)))

    assert reddest_column(tenth.image) != reddest_column(first.image)


def test_synthetic_source_has_depth_flag() -> None:
    with_depth = _started_source(emit_depth=True)
    without_depth = _started_source(emit_depth=False)
    assert with_depth.has_depth is True
    assert without_depth.has_depth is False
    frame = without_depth.read()
    assert frame.has_depth is False
    assert frame.depth is None


def test_synthetic_source_depth_is_valid_only_near_target() -> None:
    source = _started_source(
        width=320, height=240, num_targets=1, emit_depth=True, depth_m=7.5, radius_px=10
    )
    frame = source.read()
    assert frame.depth is not None
    # Background depth is 0.0 (invalid); only the target's footprint is valid.
    assert frame.depth.min() == 0.0
    assert np.isclose(frame.depth.max(), 7.5)
    valid_fraction = np.count_nonzero(frame.depth) / frame.depth.size
    assert 0 < valid_fraction < 0.2


class _FakeCv2Capture:
    """Minimal stand-in for cv2.VideoCapture, driven entirely in-memory."""

    def __init__(self, frames: list[np.ndarray], fps: float, width: int, height: int) -> None:
        self._frames = frames
        self._fps = fps
        self._width = width
        self._height = height
        self._index = 0
        self._opened = True
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int) -> float:
        import cv2

        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        import cv2

        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._index = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True
        self._opened = False


def _fake_frames(n: int, width: int = 64, height: int = 48) -> list[np.ndarray]:
    return [np.full((height, width, 3), i, dtype=np.uint8) for i in range(n)]


def test_video_file_source_paces_to_native_fps() -> None:
    clock = FakeClock()
    fake_cap = _FakeCv2Capture(_fake_frames(3), fps=10.0, width=64, height=48)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = VideoFileSource("clip.mp4", clock, loop=False)
        source.start()

        first = source.read()
        assert first is not None

        # Not due yet: native period is 0.1s at 10 fps.
        clock.advance(0.05)
        assert source.read() is None

        clock.advance(0.05)
        second = source.read()
        assert second is not None
        assert not np.array_equal(first.image, second.image)


def test_video_file_source_loops() -> None:
    clock = FakeClock()
    fake_cap = _FakeCv2Capture(_fake_frames(2), fps=10.0, width=64, height=48)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = VideoFileSource("clip.mp4", clock, loop=True)
        source.start()

        seen = []
        for _ in range(4):
            frame = source.read()
            assert frame is not None
            seen.append(frame.image[0, 0, 0])
            clock.advance(0.1)

        assert seen == [0, 1, 0, 1]


def test_video_file_source_stops_without_looping() -> None:
    clock = FakeClock()
    fake_cap = _FakeCv2Capture(_fake_frames(1), fps=10.0, width=64, height=48)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = VideoFileSource("clip.mp4", clock, loop=False)
        source.start()
        assert source.read() is not None
        clock.advance(0.1)
        assert source.read() is None


def test_video_file_source_read_before_start_returns_none() -> None:
    source = VideoFileSource("clip.mp4", FakeClock())
    assert source.read() is None


def test_video_file_source_stop_releases_and_blocks_further_reads() -> None:
    clock = FakeClock()
    fake_cap = _FakeCv2Capture(_fake_frames(2), fps=10.0, width=64, height=48)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = VideoFileSource("clip.mp4", clock, loop=False)
        source.start()
        source.stop()
        assert fake_cap.released is True
        assert source.read() is None


def test_video_file_source_intrinsics_before_start_raises() -> None:
    source = VideoFileSource("clip.mp4", FakeClock())
    with pytest.raises(RuntimeError):
        _ = source.intrinsics


def test_video_file_source_looping_empty_file_returns_none() -> None:
    clock = FakeClock()
    fake_cap = _FakeCv2Capture(_fake_frames(0), fps=10.0, width=64, height=48)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = VideoFileSource("clip.mp4", clock, loop=True)
        source.start()
        assert source.read() is None


def test_video_file_source_raises_if_file_cannot_be_opened() -> None:
    fake_cap = _FakeCv2Capture([], fps=30.0, width=64, height=48)
    fake_cap._opened = False
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = VideoFileSource("missing.mp4", FakeClock())
        with pytest.raises(RuntimeError):
            source.start()


def test_webcam_source_reports_delivered_not_requested_resolution() -> None:
    fake_cap = _FakeCv2Capture(_fake_frames(1), fps=30.0, width=640, height=480)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = WebcamSource(0, FakeClock(), requested_width=1920, requested_height=1080)
        source.start()
        assert source.intrinsics.width == 640
        assert source.intrinsics.height == 480


def test_webcam_source_has_no_depth() -> None:
    fake_cap = _FakeCv2Capture(_fake_frames(1), fps=30.0, width=640, height=480)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = WebcamSource(0, FakeClock())
        source.start()
        assert source.has_depth is False
        frame = source.read()
        assert frame is not None
        assert frame.has_depth is False


def test_webcam_source_read_before_start_returns_none() -> None:
    source = WebcamSource(0, FakeClock())
    assert source.read() is None


def test_webcam_source_stop_releases_and_blocks_further_reads() -> None:
    fake_cap = _FakeCv2Capture(_fake_frames(1), fps=30.0, width=640, height=480)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = WebcamSource(0, FakeClock())
        source.start()
        source.stop()
        assert fake_cap.released is True
        assert source.read() is None


def test_webcam_source_intrinsics_before_start_raises() -> None:
    source = WebcamSource(0, FakeClock())
    with pytest.raises(RuntimeError):
        _ = source.intrinsics


def test_webcam_source_read_failure_returns_none() -> None:
    fake_cap = _FakeCv2Capture(_fake_frames(0), fps=30.0, width=640, height=480)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = WebcamSource(0, FakeClock())
        source.start()
        assert source.read() is None


def test_webcam_source_raises_if_device_cannot_be_opened() -> None:
    fake_cap = _FakeCv2Capture([], fps=30.0, width=640, height=480)
    fake_cap._opened = False
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = WebcamSource(0, FakeClock())
        with pytest.raises(RuntimeError):
            source.start()
