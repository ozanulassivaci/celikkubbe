"""UiSnapshot: the one immutable object PipelineWorker hands to the GUI thread.

Frozen throughout, so the GUI thread never needs a lock: it reads whatever
snapshot it last received while PipelineWorker builds the next one on its
own thread. Delivered by a Qt queued-connection signal, which marshals the
object onto the GUI thread without either side blocking on the other.
"""

from __future__ import annotations

from dataclasses import dataclass

from celikkubbe.core.types import (
    Axis,
    Detection,
    Frame,
    Layer,
    ReasonCode,
    SystemState,
    Telemetry,
    Track,
)
from celikkubbe.geometry.solver import AimSolution
from celikkubbe.io.codec import EventId, TelemetryFrame


@dataclass(frozen=True)
class PipelineTimings:
    """Per-stage wall-clock duration for one tick, in milliseconds.

    The performance strip reads this directly so a slowdown's origin is
    visible at a glance instead of inferred from a single aggregate FPS.
    """

    capture_ms: float
    detect_ms: float
    track_ms: float
    solve_ms: float
    step_ms: float
    tick_ms: float
    fps: float


@dataclass(frozen=True)
class HealthSnapshot:
    """A tick's worth of readings from core.health's monitors, plus the
    cascade's layer selection, aggregated for display. Owned by ui/, not
    core/: core.health exposes the monitors themselves, never a combined
    snapshot of them, since no core/ decision logic needs one -- only the
    status strip does.

    active_layer/fallback_reason mirror SystemState verbatim rather than
    running core.cascade.Cascade: L1/YOLO does not exist yet (see
    CLAUDE.md's open questions), so there is nothing for a cascade to
    monitor or fall back from. active_layer is hardcoded to L2 exactly
    like demos/pipeline.py, and l1_recovery_countdown_s stays None until
    a real L1 health signal exists to recover towards.
    """

    inference_healthy: bool
    inference_ms: float
    camera_healthy: bool
    camera_fps: float
    link_healthy: bool
    link_stale_ms: float
    crc_error_count: int
    detection_healthy: bool
    detection_confidence: float
    depth_healthy: bool
    depth_valid_ratio: float
    active_layer: Layer
    fallback_reason: ReasonCode | None
    l1_recovery_countdown_s: float | None


@dataclass(frozen=True)
class UiSnapshot:
    """One tick of the pipeline, frame and detections together.

    Coupling the frame to the detections computed from it (rather than
    letting a separate capture thread feed the display at full rate while
    processing lags behind) means the canvas never draws a box next to a
    newer frame than the one it was computed from. L2 colour detection on
    a cropped ROI and even a future YOLOv8n pass are both fast enough
    (single-digit milliseconds) that this pipeline sustains FPS_TARGET
    coupled, at no framerate cost. If a future model is slow enough to
    change that trade-off, the escape hatch is decoupling the capture and
    processing loops and forward-predicting box positions onto the newer
    frame -- not building that until a real model needs it.
    """

    t: float
    frame: Frame
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    state: SystemState
    telemetry: Telemetry | None
    telemetry_frame: TelemetryFrame | None
    aim: AimSolution | None
    crosshair_px: tuple[float, float] | None
    crosshair_offscreen: bool
    crosshair_bearing_deg: float | None
    health: HealthSnapshot
    timings: PipelineTimings
    # Populated only in M4_SAFE, for the SAFE overlay: the telemetry-
    # derived condition that tripped it (mirrors modes.step()'s own
    # check order), or None when the trip instead came from a self-test
    # failure -- state.last_self_test already carries that detail, so
    # there is no separate ReasonCode invented for it.
    fault_reason: ReasonCode | None
    # Bounded sliding window of MCU events (EventId, axis, PC-receive
    # timestamp), oldest first, for the SAFE overlay's event log. Only
    # ever populated against a link that actually surfaces events
    # (SimTurretLink does; SerialTurretLink does not yet -- see
    # CLAUDE.md's open questions), so this stays empty against real
    # hardware today.
    event_log: tuple[tuple[float, EventId, Axis | None], ...]
    # Cumulative shots fired, read directly off the link (SimTurretLink
    # exposes it; SerialTurretLink does not -- no ammo counter exists on
    # the wire protocol). None when the link does not report it, rather
    # than a misleading 0.
    ammo_fired: int | None
