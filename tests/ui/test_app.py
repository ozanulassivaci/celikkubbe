"""app.py tests. ``build()`` starts a real PipelineWorker thread, so
anything that needs to observe a snapshot arrive must actually spin the
Qt event loop (qtbot.waitUntil/qtbot.wait) rather than sleep -- a cross-
thread signal only gets delivered to its slot once the receiving
thread's event loop processes it.
"""

from __future__ import annotations

import pytest

from celikkubbe.core.clock import FakeClock
from celikkubbe.core.types import Stage
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.ui.app import _build_link, _build_source, _parse_args, build
from celikkubbe.ui.main_window import MainWindow
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.vision.sources import SyntheticSource, VideoFileSource, WebcamSource


def test_parse_args_defaults():
    args = _parse_args([])
    assert args.source == "synthetic"
    assert args.link == "sim"
    assert args.stage == "2"
    assert args.fullscreen is False


def test_parse_args_video_requires_path():
    with pytest.raises(SystemExit):
        _parse_args(["--source", "video"])


def test_build_source_synthetic():
    args = _parse_args(["--source", "synthetic", "--targets", "2"])
    source, label = _build_source(args, FakeClock())
    assert isinstance(source, SyntheticSource)
    assert label == "SYNTHETIC"


def test_build_source_video():
    args = _parse_args(["--source", "video", "--path", "clip.mp4"])
    source, label = _build_source(args, FakeClock())
    assert isinstance(source, VideoFileSource)
    assert "clip.mp4" in label


def test_build_source_webcam():
    args = _parse_args(["--source", "webcam", "--device", "1"])
    source, label = _build_source(args, FakeClock())
    assert isinstance(source, WebcamSource)
    assert "1" in label


def test_build_link_sim():
    args = _parse_args(["--link", "sim"])
    link = _build_link(args, FakeClock())
    assert isinstance(link, SimTurretLink)


def test_build_constructs_and_starts_a_running_app(qtbot):
    app, window, worker = build(["--source", "synthetic", "--link", "sim", "--stage", "1"])
    try:
        assert isinstance(window, MainWindow)
        assert isinstance(worker, PipelineWorker)
        assert worker.isRunning()
        assert worker._state.stage is Stage.STAGE_1

        qtbot.waitUntil(lambda: window._latest_snapshot is not None, timeout=3000)
        assert window._latest_snapshot is not None
    finally:
        worker.stop()
        worker.wait(2000)
        window.close()


def test_build_fullscreen_flag_shows_window_fullscreen(qtbot):
    app, window, worker = build(["--source", "synthetic", "--link", "sim", "--fullscreen"])
    try:
        assert window.isFullScreen()
    finally:
        worker.stop()
        worker.wait(2000)
        window.close()
