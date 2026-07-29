"""L1/L2/L3 detection-layer selection and the L1 recovery monitor.

L1 is primary, L2 is the automatic fallback, L3 is a Stage 1 manual-only
override. Nothing here inspects pixels or runs inference; it only decides
which layer's output the rest of the system should trust, based on health
booleans supplied by the caller (see ``health.py``).
"""

from __future__ import annotations

from celikkubbe.core import config
from celikkubbe.core.health import CameraHealth, DetectionHealth, InferenceHealth
from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import Layer, ReasonCode, Stage


def l1_health(
    inference: InferenceHealth,
    camera: CameraHealth,
    detection: DetectionHealth,
) -> tuple[bool, ReasonCode | None]:
    """Combine the individual L1 pipeline health monitors into one signal."""
    for monitor in (inference, camera, detection):
        if not monitor.healthy:
            return False, monitor.reason
    return True, None


class Cascade:
    """Owns the active detection layer and the L1 recovery monitor's timers."""

    def __init__(
        self,
        clock: Clock,
        recovery_interval_s: float = config.L1_RECOVERY_INTERVAL_S,
        recovery_frames: int = config.L1_RECOVERY_FRAMES,
    ) -> None:
        self._clock = clock
        self._recovery_interval_s = recovery_interval_s
        self._recovery_frames = recovery_frames
        self._active_layer = Layer.L1
        self._manual_override = False
        self._fallback_reason: ReasonCode | None = None
        self._last_probe_t: float | None = None
        self._consecutive_healthy_probes = 0

    @property
    def active_layer(self) -> Layer:
        return self._active_layer

    @property
    def manual_override(self) -> bool:
        return self._manual_override

    @property
    def fallback_reason(self) -> ReasonCode | None:
        return self._fallback_reason

    def select_manual(self, layer: Layer, stage: Stage) -> None:
        """Operator manually selects a layer; disables automatic switching."""
        if layer is Layer.L3 and stage is not Stage.STAGE_1:
            raise ValueError("L3 may only be selected manually in Stage 1")
        self._active_layer = layer
        self._manual_override = True
        self._fallback_reason = ReasonCode.OPERATOR_OVERRIDE if layer is not Layer.L1 else None
        self._last_probe_t = None
        self._consecutive_healthy_probes = 0

    def clear_manual_override(self) -> None:
        """Return control to the automatic L1/L2 selection logic."""
        self._manual_override = False
        self._last_probe_t = None
        self._consecutive_healthy_probes = 0

    def update(
        self,
        l1_healthy: bool,
        l1_unhealthy_reason: ReasonCode | None,
        now: float,
    ) -> tuple[Layer, ReasonCode | None]:
        """Advance the automatic L1/L2 selection by one tick.

        No-op while a manual override is active. Falls back to L2 the
        instant L1 is unhealthy; recovers to L1 only after
        ``recovery_frames`` consecutive healthy probes spaced
        ``recovery_interval_s`` apart, so a flapping L1 does not thrash.
        """
        if self._manual_override:
            return self._active_layer, self._fallback_reason

        if self._active_layer is Layer.L1:
            if not l1_healthy:
                self._active_layer = Layer.L2
                self._fallback_reason = l1_unhealthy_reason
                self._last_probe_t = now
                self._consecutive_healthy_probes = 0
            return self._active_layer, self._fallback_reason

        if self._last_probe_t is None or now - self._last_probe_t >= self._recovery_interval_s:
            self._last_probe_t = now
            if l1_healthy:
                self._consecutive_healthy_probes += 1
            else:
                self._consecutive_healthy_probes = 0
            if self._consecutive_healthy_probes >= self._recovery_frames:
                self._active_layer = Layer.L1
                self._fallback_reason = None
                self._consecutive_healthy_probes = 0

        return self._active_layer, self._fallback_reason
