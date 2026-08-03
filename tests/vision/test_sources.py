from __future__ import annotations

import logging
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.vision.sources import (
    FRIENDLY_HEX,
    HOSTILE_HEX,
    SyntheticSource,
    SyntheticSourceConfig,
    SyntheticTarget,
    VideoFileSource,
    WebcamSource,
    estimated_intrinsics,
    hex_to_bgr,
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
    source = _started_source(width=320, height=240, num_targets=1, emit_depth=True, range_m=7.5)
    frame = source.read()
    assert frame.depth is not None
    # Background depth is 0.0 (invalid); only the target's footprint is valid.
    assert frame.depth.min() == 0.0
    assert np.isclose(frame.depth.max(), 7.5)
    valid_fraction = np.count_nonzero(frame.depth) / frame.depth.size
    assert 0 < valid_fraction < 0.2


def test_synthetic_source_auto_generates_alternating_colours() -> None:
    resolved = SyntheticSourceConfig(num_targets=4, speed=0.0).resolve_targets()
    colors = [t.color_hex for t in resolved]
    assert colors == [HOSTILE_HEX, FRIENDLY_HEX, HOSTILE_HEX, FRIENDLY_HEX]


def test_synthetic_source_renders_both_colours() -> None:
    source = _started_source(width=320, height=240, num_targets=2, speed=0.0)
    frame = source.read()
    hostile_bgr = np.array(hex_to_bgr(HOSTILE_HEX))
    friendly_bgr = np.array(hex_to_bgr(FRIENDLY_HEX))
    pixels = frame.image.reshape(-1, 3)
    assert np.any(np.all(pixels == hostile_bgr, axis=1))
    assert np.any(np.all(pixels == friendly_bgr, axis=1))


def test_synthetic_source_apparent_size_follows_pinhole_model() -> None:
    # A target twice as far away should render at roughly half the pixel
    # diameter — the same fx * size_m / range_m relationship l2_color.py
    # uses for size-based range estimation.
    near = SyntheticTarget(color_hex=HOSTILE_HEX, size_m=0.5, lane_fraction=0.5, range_m=5.0)
    far = SyntheticTarget(color_hex=HOSTILE_HEX, size_m=0.5, lane_fraction=0.5, range_m=10.0)

    def rendered_width(target: SyntheticTarget) -> int:
        cfg = SyntheticSourceConfig(width=640, height=480, targets=(target,))
        source = SyntheticSource(FakeClock(), cfg)
        source.start()
        frame = source.read()
        mask = np.any(frame.image != (40, 40, 40), axis=-1)
        cols = np.where(mask.any(axis=0))[0]
        return int(cols.max() - cols.min())

    assert rendered_width(near) > rendered_width(far)
    assert abs(rendered_width(near) - 2 * rendered_width(far)) <= 2


def test_synthetic_source_explicit_targets_override_num_targets() -> None:
    targets = (SyntheticTarget(color_hex=FRIENDLY_HEX, size_m=0.5, lane_fraction=0.3, range_m=8.0),)
    cfg = SyntheticSourceConfig(num_targets=5, targets=targets)
    assert cfg.resolve_targets() == targets


def test_hex_to_bgr_matches_confirmed_hostile_and_friendly_values() -> None:
    assert hex_to_bgr(HOSTILE_HEX) == (10, 10, 245)
    assert hex_to_bgr(FRIENDLY_HEX) == (224, 163, 0)


def test_estimated_intrinsics_matches_worked_reference_pixel_sizes() -> None:
    # Reference numbers from the competition spec: a 50cm model at 15m and
    # a 30cm drone at 15m, at both candidate resolutions.
    for width, height, expected_50cm, expected_30cm in (
        (1280, 720, 31, 19),
        (1920, 1080, 47, 28),
    ):
        intrinsics = estimated_intrinsics(width, height)
        px_50cm = intrinsics.fx * 0.50 / 15.0
        px_30cm = intrinsics.fx * 0.30 / 15.0
        assert round(px_50cm) == expected_50cm
        assert round(px_30cm) == expected_30cm


class _FakeCv2Capture:
    """Minimal stand-in for cv2.VideoCapture, driven entirely in-memory."""

    def __init__(
        self,
        frames: list[np.ndarray],
        fps: float,
        width: int,
        height: int,
        fourcc: int = 0,
    ) -> None:
        self._frames = frames
        self._fps = fps
        self._width = width
        self._height = height
        self._fourcc = fourcc
        self.requested_fourcc: int | None = None
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
        if prop_id == cv2.CAP_PROP_FOURCC:
            return float(self._fourcc)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        import cv2

        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._index = int(value)
        elif prop_id == cv2.CAP_PROP_FOURCC:
            self.requested_fourcc = int(value)
            self._fourcc = int(value)
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


def test_webcam_source_requests_mjpg_before_resolution() -> None:
    # Confirmed against a real device (v4l2-ctl --list-formats-ext): YUYV
    # at 1280x720 negotiates only 10 fps on hardware that supports MJPG
    # at 30 fps at every resolution it offers. WebcamSource must request
    # MJPG explicitly rather than accepting the backend's default.
    fake_cap = _FakeCv2Capture(_fake_frames(1), fps=30.0, width=1280, height=720)
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = WebcamSource(0, FakeClock())
        source.start()
        assert fake_cap.requested_fourcc == cv2.VideoWriter_fourcc(*"MJPG")


def test_webcam_source_logs_the_negotiated_format(caplog) -> None:
    caplog.set_level(logging.INFO, logger="celikkubbe.vision.sources")
    fake_cap = _FakeCv2Capture(
        _fake_frames(1), fps=30.0, width=1280, height=720, fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    )
    with patch("celikkubbe.vision.sources.cv2.VideoCapture", return_value=fake_cap):
        source = WebcamSource(0, FakeClock())
        source.start()

    assert any(
        "1280x720" in record.message and "MJPG" in record.message and "30.0" in record.message
        for record in caplog.records
    )


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
