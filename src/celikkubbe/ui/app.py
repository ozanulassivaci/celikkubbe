"""GUI entry point.

    python -m celikkubbe.ui.app --source synthetic --link sim
    python -m celikkubbe.ui.app --source webcam --device 0 --link sim
    python -m celikkubbe.ui.app --source video --path clip.mp4 --link sim --fullscreen

Same --source selection as demos/pipeline.py (synthetic/video/webcam), so
the GUI runs against synthetic data, a webcam, or later the real camera
with no code change. --link is sim-only here, unlike the demo's
{stub, sim}: PipelineWorker drives the link through LinkWorker, whose
ACK/retransmit tracking needs send_tracked()/drain_acks(), which the
demo's lightweight bearing-tracker stub does not implement -- so it is
not a fit for anything built on LinkWorker. SerialTurretLink is the
natural next addition here once real hardware exists.

``build()`` does everything except run the Qt event loop, which is what
keeps this module testable: a test calls it directly, inspects the
constructed window and worker, and tears them down itself -- never
blocking on a real ``app.exec()``. ``main()`` is the thin wrapper that
actually runs the event loop and is what ``__main__`` calls.
"""

from __future__ import annotations

import argparse
import signal
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from celikkubbe.core.clock import SystemClock
from celikkubbe.core.protocols import Clock, FrameSource, TurretLink
from celikkubbe.core.types import Stage
from celikkubbe.io.sim_link import SimTurretLink
from celikkubbe.ui import theme
from celikkubbe.ui.main_window import MainWindow
from celikkubbe.ui.pipeline_worker import PipelineWorker
from celikkubbe.vision.sources import (
    SyntheticSource,
    SyntheticSourceConfig,
    VideoFileSource,
    WebcamSource,
)

_STAGE_FROM_ARG: dict[str, Stage] = {"1": Stage.STAGE_1, "2": Stage.STAGE_2, "3": Stage.STAGE_3}

# Gives the Python interpreter a periodic chance to run between Qt event
# loop iterations, so a SIGINT handler actually gets dispatched instead of
# waiting for the next real Qt event -- app.exec() otherwise blocks
# entirely inside Qt's C++ loop between events.
_SIGINT_POLL_MS = 200


def _build_source(args: argparse.Namespace, clock: Clock) -> tuple[FrameSource, str]:
    if args.source == "synthetic":
        cfg = SyntheticSourceConfig(num_targets=args.targets, speed=0.05, range_m=5.0)
        return SyntheticSource(clock, cfg), "SYNTHETIC"
    if args.source == "video":
        return VideoFileSource(args.path, clock), f"VIDEO:{args.path}"
    if args.source == "webcam":
        return WebcamSource(args.device, clock), f"WEBCAM:{args.device}"
    raise ValueError(f"unknown source: {args.source}")  # pragma: no cover — argparse guards this


def _build_link(args: argparse.Namespace, clock: Clock) -> TurretLink:
    if args.link == "sim":
        return SimTurretLink(clock)
    raise ValueError(f"unknown link: {args.link}")  # pragma: no cover — argparse guards this


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Celikkubbe operator GUI")
    parser.add_argument("--source", choices=["synthetic", "video", "webcam"], default="synthetic")
    parser.add_argument("--targets", type=int, default=3, help="synthetic source target count")
    parser.add_argument("--path", type=str, default=None, help="video file path")
    parser.add_argument("--device", type=int, default=0, help="webcam device index")
    parser.add_argument("--stage", choices=["1", "2", "3"], default="2")
    parser.add_argument(
        "--link",
        choices=["sim"],
        default="sim",
        help="sim: io.sim_link.SimTurretLink (default and, for now, only option -- see "
        "the module docstring for why the demo's --link stub is not offered here)",
    )
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args(argv)
    if args.source == "video" and not args.path:
        parser.error("--source video requires --path")
    return args


def build(argv: list[str] | None = None) -> tuple[QApplication, MainWindow, PipelineWorker]:
    args = _parse_args(argv)
    stage = _STAGE_FROM_ARG[args.stage]

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)  # pragma: no cover — tests reuse pytest-qt's session qapp
    font_family = theme.load_monospace_font()
    app.setStyleSheet(theme.build_qss(theme.accent_for_stage(stage), font_family))

    clock = SystemClock()
    source, source_label = _build_source(args, clock)
    link = _build_link(args, clock)
    worker = PipelineWorker(source, link, clock, stage=stage)
    window = MainWindow(worker, source_label=source_label)

    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    worker.start()

    return app, window, worker


def main(argv: list[str] | None = None) -> int:
    app, _window, worker = build(argv)

    # Referenced only by this frame, which stays on the stack for the
    # entire blocking app.exec() call below -- safe from GC without
    # needing to be attached to anything else.
    poll_timer = QTimer()
    poll_timer.timeout.connect(lambda: None)
    poll_timer.start(_SIGINT_POLL_MS)

    signal.signal(signal.SIGINT, lambda *_args: app.quit())
    app.aboutToQuit.connect(lambda: (worker.stop(), worker.wait(2000)))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
