"""Every threshold used by the core decision layer.

Nothing in ``core`` may hardcode a number that belongs here.
"""

from __future__ import annotations

from celikkubbe.core.types import TargetClass

# --- timing ---
FPS_TARGET = 30
HEARTBEAT_MS = 50  # PC -> MCU keepalive period
WATCHDOG_TIMEOUT_MS = 200  # MCU safes the turret if no heartbeat arrives
TELEMETRY_STALE_MS = 300  # PC-side: UI shows "LINK LOST" past this

# --- motion ---
ANGLE_TOLERANCE_DEG = 0.10  # servo settling; S4 -> S5 condition
BACKOFF_DEG = 0.50  # retreat distance before final approach
UNIDIRECTIONAL_APPROACH = True  # always settle from the same side (pan gear backlash)
PAN_LIMIT_DEG = (-170.0, 170.0)  # cable wrap limit
TILT_LIMIT_DEG = (-20.0, 60.0)

# --- health ---
INFERENCE_WARN_MS = 50
INFERENCE_FAIL_MS = 100
INFERENCE_FAIL_FRAMES = 15  # CONSECUTIVE; a single spike must not trigger fallback
CAMERA_TIMEOUT_MS = 500
MIN_FPS = 15
CAMERA_FPS_WINDOW_S = 1.0  # rolling window used to measure camera FPS
DEPTH_MIN_VALID_RATIO = 0.5  # min fraction of recent frames needing usable depth

# --- detection ---
CONFIDENCE_THRESHOLD = 0.65
ACQUIRE_FRAMES = 5  # consecutive consistent frames to confirm in S2
TRACK_LOST_MS = 1000
MIN_TARGET_PX = 8  # below this the UI warns "target under resolution"

# --- cascade ---
L1_RECOVERY_INTERVAL_S = 10.0  # silently probe L1 while running L2
L1_RECOVERY_FRAMES = 5  # consecutive healthy probes required to return

# --- engagement ---
MAX_ENGAGEMENT_ATTEMPTS = 3  # then skip the target
HIT_VERIFY_MS = 1500
DEFAULT_RANGE_M = 10.0  # parallax assumption when nothing is locked

# --- class/range rule (Stage 3) ---
RANGE_RULES: dict[TargetClass, tuple[float, float]] = {
    TargetClass.F16: (10.0, 15.0),
    TargetClass.HELICOPTER: (5.0, 15.0),
    TargetClass.MISSILE: (5.0, 15.0),
    TargetClass.UAV: (0.0, 15.0),
}

# --- prioritisation ---
CLASS_PRIORITY: dict[TargetClass, int] = {
    TargetClass.F16: 100,
    TargetClass.MISSILE: 80,
    TargetClass.HELICOPTER: 60,
    TargetClass.UAV: 40,
    TargetClass.BALLOON: 20,
}
WEIGHT_CLASS, WEIGHT_RANGE, WEIGHT_CONFIDENCE = 0.5, 0.3, 0.2
MAX_ENGAGEMENT_RANGE_M = 15.0  # far reference for the range score; matches RANGE_RULES

# --- UI stability ---
TRACK_LIST_UPDATE_HZ = 10
REORDER_HYSTERESIS = 0.10  # order changes only if score gap exceeds 10%
