"""Classic-CV colour detector — the permanent L2 fallback.

This runs whenever the (future) YOLO L1 pipeline's health degrades, not
just during bring-up, and is also what gets demonstrated in the
capability video, so it is built to production standard rather than as
a scaffold.

Output is always ``cls=None``: this layer cannot classify a target and
must never pretend otherwise. It **does** set ``iff``, though — the
competition's targets are colour-coded (hostile red, friendly blue), so
colour alone is enough to answer the IFF question even without knowing
whether a red blob is a drone or an F-16.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from celikkubbe.core.config import (
    IMPLAUSIBLE_RANGE_ASPECT_RATIO_RANGE,
    IMPLAUSIBLE_RANGE_MAX_PX,
    IMPLAUSIBLE_RANGE_MIN_CONFIDENCE,
    MAX_ENGAGEMENT_RANGE_M,
    MIN_TARGET_PX,
)
from celikkubbe.core.types import (
    IFF,
    BoundingBox,
    CameraIntrinsics,
    Detection,
    Frame,
    Layer,
    RangeSource,
    Track,
)

logger = logging.getLogger(__name__)

DEPTH_SAMPLE_WINDOW_PX = 5  # odd window sampled around the centroid for depth range

# --- structural noise floor for ColorDetectorConfig.min_area_px, applied
# whenever it is left at its default None -- see compute_min_area_px. ---
_MIN_AREA_TARGET_SIZE_M = 0.30  # the drone -- smallest of the three known target sizes
_MIN_AREA_FRACTION = 0.3  # allows a non-square aircraft silhouette, not a filled square
# "estimated"-quality intrinsics (a webcam with no calibration) have a
# guessed fx -- computing the floor from it would compound one guess with
# another, so this instead falls back to a fixed fraction of frame area.
# Derived by evaluating the same formula once at the assumed 69-degree HFOV
# (vision.sources.estimated_intrinsics) against a 1280x720 frame: fx =
# 640/tan(34.5deg) = 931, expected_px = 931*0.30/15 = 18.6px,
# area = 18.6**2*0.3 = 104px, fraction = 104/921600 = 0.000113 -- rounded
# down slightly since the same derivation at other aspect ratios (e.g.
# 640x480) yields a somewhat smaller fraction.
_MIN_AREA_ESTIMATED_FRAME_FRACTION = 0.0001

# Repo-root config/ -- runtime tuning data, not the compile-time
# thresholds in core/config.py. Created on first save; absent entirely
# until then, same convention as geometry.calibration's config/ paths.
DEFAULT_HSV_PRESETS_DIR = Path("config/hsv")


@dataclass(frozen=True)
class ColorClass:
    name: str
    iff: IFF
    hue_ranges: tuple[tuple[int, int], ...]
    sat_min: int
    val_min: int


# Confirmed competition target colours, converted to OpenCV HSV (H 0-179):
#   #F50A0A (hostile red)   -> H 0,  S 245, V 245
#   #00A3E0 (friendly blue) -> H 98, S 255, V 224 — an azure, not true cyan
#     (H 90); a range centred on 90 would miss it, hence (90, 108).
# Saturation floor is 90 rather than a tighter value: the printed models
# are highly saturated, but indoor competition lighting pulls measured
# saturation down, and a tight floor loses targets at range.
_DEFAULT_CLASSES: tuple[ColorClass, ...] = (
    ColorClass("hostile", IFF.HOSTILE, ((0, 10), (170, 180)), 90, 50),
    ColorClass("friendly", IFF.FRIENDLY, ((90, 108),), 90, 50),
)


@dataclass
class ColorDetectorConfig:
    classes: tuple[ColorClass, ...] = field(default_factory=lambda: _DEFAULT_CLASSES)
    morph_kernel: int = 5
    # None -- the structural floor is computed per frame from optics (see
    # compute_min_area_px), which is what a webcam-vs-D435i, or a change
    # to the course's engagement range, should actually drive. Set an int
    # here to override it with a fixed value instead (e.g. a competition-
    # day preset tuned by eye against the real venue).
    min_area_px: int | None = None
    circularity_min: float = 0.70
    # Off by default: aircraft silhouettes are not circular (an F-16's
    # circularity sits well below 0.70) and would be rejected by a filter
    # that assumed balloons. Circularity is still computed and reported
    # for the tuning panel either way.
    require_circularity: bool = False
    # On by default, unlike circularity: a solid printed model scores
    # around 0.9 regardless of its silhouette (solidity does not assume
    # anything is round), while scattered shadow patches, specular
    # streaks and fragmented blobs score well below this.
    solidity_min: float = 0.75
    # (short side / long side is never below this, long/short never
    # above) -- rejects long thin colour bands (a strip of wall trim, a
    # doorframe edge) that no competition target's silhouette resembles.
    aspect_ratio_range: tuple[float, float] = (0.2, 5.0)
    # The course physically cannot present more than three models of one
    # colour; anything beyond this per class, this frame, made it past
    # every HSV/area/solidity/aspect-ratio filter but is still noise, not
    # a real fourth target.
    max_detections_per_class: int = 3
    roi: BoundingBox | None = None  # normalised (x1, y1, x2, y2)
    known_sizes_m: tuple[float, ...] = (0.30, 0.40, 0.50)


@dataclass(frozen=True)
class Contour:
    """One contour this frame considered, accepted or not — for the tuning panel."""

    class_name: str
    bbox_px: tuple[int, int, int, int]  # x, y, w, h in full-frame pixels
    area_px: float
    circularity: float
    solidity: float
    accepted: bool
    # "area" | "circularity" | "solidity" | "aspect_ratio" | "class_cap"
    reject_reason: str | None = None


@dataclass(frozen=True)
class DebugMasks:
    hsv_masks: dict[str, np.ndarray]  # class name -> raw HSV threshold mask
    morphed_masks: dict[str, np.ndarray]  # class name -> post-morphology mask
    contours: tuple[Contour, ...]


@dataclass(frozen=True)
class _ContourMetrics:
    """Internal: every measurement taken for one contour, before the
    per-class area-descending cap decides which structural survivors are
    actually kept. Not exposed outside this module -- Contour (above) is
    the public, tuning-panel-facing record.
    """

    contour: np.ndarray
    bbox_px: tuple[int, int, int, int]
    area: float
    circularity: float
    solidity: float
    rect_w: float
    rect_h: float
    aspect_ratio: float
    reject_reason: str | None


def is_under_resolved(
    bbox: BoundingBox, intrinsics: CameraIntrinsics, min_target_px: int = MIN_TARGET_PX
) -> bool:
    """True when a normalised bbox's pixel width is below MIN_TARGET_PX.

    Computed from intrinsics.width rather than a hardcoded resolution
    table, so the same check is valid whether the frame came from a
    1280x720 webcam or a 1920x1080 D435i. Reference points from the
    competition spec: a 50cm model at 15m is ~31px at 1280x720 and ~47px
    at 1920x1080; a 30cm drone is ~19px and ~28px respectively.
    """
    width_px = (bbox[2] - bbox[0]) * intrinsics.width
    return width_px < min_target_px


def is_range_estimate_unreliable(track: Track, intrinsics: CameraIntrinsics) -> bool:
    """True when a track's own bounding box looks like a fragment of a
    larger object rather than the whole target -- small yet highly
    confident, or an aspect ratio no printed model's silhouette would
    produce (see IMPLAUSIBLE_RANGE_* in core/config.py). Its range
    estimate is not wrong in the sense of a bug -- size-based range
    (_size_range_m) assumes the box covers the whole model, and even a
    depth-derived range samples a window centred on the fragment's own,
    wrong centroid -- it is simply meaningless. Display-only: this never
    gates engagement, and is checked independently of range_source.
    """
    width_px = (track.bbox[2] - track.bbox[0]) * intrinsics.width
    height_px = (track.bbox[3] - track.bbox[1]) * intrinsics.height
    if width_px <= 0.0 or height_px <= 0.0:
        return False
    if width_px < IMPLAUSIBLE_RANGE_MAX_PX and track.confidence >= IMPLAUSIBLE_RANGE_MIN_CONFIDENCE:
        return True
    aspect_ratio = max(width_px, height_px) / min(width_px, height_px)
    lo, hi = IMPLAUSIBLE_RANGE_ASPECT_RATIO_RANGE
    return not (lo <= aspect_ratio <= hi)


def compute_min_area_px(
    intrinsics: CameraIntrinsics,
    max_range_m: float = MAX_ENGAGEMENT_RANGE_M,
    min_target_size_m: float = _MIN_AREA_TARGET_SIZE_M,
    area_fraction: float = _MIN_AREA_FRACTION,
) -> int:
    """Structural noise floor for accepted contour area, derived from
    optics rather than guessed.

    A contour genuinely produced by the smallest competition target (the
    0.30m drone) at the course's own maximum engagement range projects to
    roughly ``expected_px`` pixels across; squaring that and scaling by
    ``area_fraction`` (not 1.0) allows for a non-square aircraft
    silhouette rather than assuming a filled square bounding box. Anything
    smaller than this floor cannot possibly be a real target at any range
    the course allows, and is therefore noise (a specular highlight, a
    shadow fragment, a slice of a red doorframe) regardless of how tight
    the HSV thresholds are tuned.

    "estimated"-quality intrinsics (a webcam with no calibration -- see
    CameraIntrinsics.is_reliable) have an ``fx`` that is itself a guess, so
    computing from it here would compound one guess with another; see
    _MIN_AREA_ESTIMATED_FRAME_FRACTION's own comment for the fallback used
    instead, which scales with frame area rather than assuming a fixed
    pixel count.
    """
    if not intrinsics.is_reliable:
        return round(intrinsics.width * intrinsics.height * _MIN_AREA_ESTIMATED_FRAME_FRACTION)
    expected_px = intrinsics.fx * min_target_size_m / max_range_m
    return round((expected_px**2) * area_fraction)


def _depth_range_m(depth: np.ndarray, cx_px: int, cy_px: int, window: int) -> float | None:
    half = window // 2
    y0, y1 = max(0, cy_px - half), min(depth.shape[0], cy_px + half + 1)
    x0, x1 = max(0, cx_px - half), min(depth.shape[1], cx_px + half + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _size_range_m(pixel_size: float, fx: float, known_sizes_m: tuple[float, ...]) -> float | None:
    if pixel_size <= 0 or not known_sizes_m:
        return None
    # Without a class, the true size is genuinely ambiguous. Assume the
    # largest plausible size: that yields the largest (most conservative)
    # range estimate, which is the safe direction of error for a gate
    # whose job is bounding how close/far is safe to engage — the wrong
    # way to be wrong is to underestimate range and pass something that
    # is actually too far.
    return fx * max(known_sizes_m) / pixel_size


class ColorDetector:
    """The permanent L2 detection layer: per-class HSV threshold -> contours."""

    def __init__(self, config: ColorDetectorConfig | None = None) -> None:
        self.config = config or ColorDetectorConfig()
        self._last_logged_min_area_px: int | None = None
        # known_sizes_m is a theoretical assumption (see the module's own
        # "Physical measurements still outstanding" entry in CLAUDE.md),
        # not a measurement -- if the printed models were scaled down for
        # a desktop printer, every size-derived range is wrong by exactly
        # that factor, silently. Logged once at construction so the
        # mismatch is visible rather than discovered downstream in a
        # wrong range reading.
        logger.info(
            "L2 detector: assumed target sizes (m) = %s -- verify these match "
            "the printed models as actually produced, not the spec",
            self.config.known_sizes_m,
        )

    def detect(
        self, frame: Frame, debug: bool = False, diag: bool = False
    ) -> tuple[list[Detection], DebugMasks | None]:
        cfg = self.config
        width, height = frame.intrinsics.width, frame.intrinsics.height
        min_area_px = self._resolve_min_area_px(cfg, frame.intrinsics)

        if cfg.roi is not None:
            x1n, y1n, x2n, y2n = cfg.roi
            px1, py1 = int(x1n * width), int(y1n * height)
            px2, py2 = int(x2n * width), int(y2n * height)
        else:
            px1, py1, px2, py2 = 0, 0, width, height

        # Crop before converting: at 1920x1080 full-frame HSV conversion and
        # morphology are expensive, and OpenCV releases the GIL during both,
        # so this method is safe to run in a worker thread alongside the
        # Python-level control loop.
        crop = frame.image[py1:py2, px1:px2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)

        detections: list[Detection] = []
        debug_hsv_masks: dict[str, np.ndarray] = {}
        debug_morphed_masks: dict[str, np.ndarray] = {}
        debug_contours: list[Contour] = []

        for color_class in cfg.classes:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for hue_lo, hue_hi in color_class.hue_ranges:
                lo = (hue_lo, color_class.sat_min, color_class.val_min)
                hi = (hue_hi, 255, 255)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

            morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if debug:
                debug_hsv_masks[color_class.name] = mask
                debug_morphed_masks[color_class.name] = morphed

            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            measured = [
                self._measure_contour(contour, cfg, min_area_px, px1, py1) for contour in contours
            ]

            # Structural survivors only, then capped by area descending --
            # the course cannot physically present more than
            # max_detections_per_class models of one colour, so once every
            # HSV/area/solidity/aspect-ratio filter has run, the largest N
            # remaining are the plausible real targets and everything
            # smaller is still noise that happened to pass every other
            # filter.
            survivors = sorted(
                (m for m in measured if m.reject_reason is None),
                key=lambda m: m.area,
                reverse=True,
            )
            cap = cfg.max_detections_per_class
            kept, capped = survivors[:cap], survivors[cap:]

            if diag:
                self._log_diag(color_class.name, measured, kept, capped, min_area_px, hsv, cfg)

            for m in kept:
                detections.append(
                    self._build_detection(m, color_class, frame, width, height, px1, py1)
                )
            if debug:
                for m in kept:
                    debug_contours.append(
                        Contour(
                            color_class.name, m.bbox_px, m.area, m.circularity, m.solidity, True
                        )
                    )
                for m in capped:
                    debug_contours.append(
                        Contour(
                            color_class.name,
                            m.bbox_px,
                            m.area,
                            m.circularity,
                            m.solidity,
                            False,
                            "class_cap",
                        )
                    )
                for m in measured:
                    if m.reject_reason is not None:
                        debug_contours.append(
                            Contour(
                                color_class.name,
                                m.bbox_px,
                                m.area,
                                m.circularity,
                                m.solidity,
                                False,
                                m.reject_reason,
                            )
                        )

        debug_masks = None
        if debug:
            debug_masks = DebugMasks(
                hsv_masks=debug_hsv_masks,
                morphed_masks=debug_morphed_masks,
                contours=tuple(debug_contours),
            )
        return detections, debug_masks

    def _resolve_min_area_px(self, cfg: ColorDetectorConfig, intrinsics: CameraIntrinsics) -> int:
        if cfg.min_area_px is not None:
            return cfg.min_area_px
        computed = compute_min_area_px(intrinsics)
        if computed != self._last_logged_min_area_px:
            logger.info(
                "L2 detector: computed min_area_px floor = %d px "
                "(intrinsics quality=%s, fx=%.1f, %dx%d)",
                computed,
                intrinsics.quality,
                intrinsics.fx,
                intrinsics.width,
                intrinsics.height,
            )
            self._last_logged_min_area_px = computed
        return computed

    def _measure_contour(
        self,
        contour: np.ndarray,
        cfg: ColorDetectorConfig,
        min_area_px: int,
        px1: int,
        py1: int,
    ) -> _ContourMetrics:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_px = (x + px1, y + py1, w, h)
        circularity = 4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else 0.0

        # Solidity: contour area over its own convex hull area. A solid
        # printed model is close to its own hull (~0.9); scattered shadow
        # patches, specular streaks and fragmented blobs are not, since
        # morphological closing does not make a fragmented mask convex,
        # only connected.
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0

        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        short_side, long_side = min(rect_w, rect_h), max(rect_w, rect_h)
        aspect_ratio = long_side / short_side if short_side > 0 else math.inf

        reject_reason: str | None = None
        aspect_lo, aspect_hi = cfg.aspect_ratio_range
        if area < min_area_px:
            reject_reason = "area"
        elif solidity < cfg.solidity_min:
            reject_reason = "solidity"
        elif not (aspect_lo <= aspect_ratio <= aspect_hi):
            reject_reason = "aspect_ratio"
        elif cfg.require_circularity and circularity < cfg.circularity_min:
            reject_reason = "circularity"

        return _ContourMetrics(
            contour,
            bbox_px,
            area,
            circularity,
            solidity,
            rect_w,
            rect_h,
            aspect_ratio,
            reject_reason,
        )

    def _log_diag(
        self,
        class_name: str,
        measured: list[_ContourMetrics],
        kept: list[_ContourMetrics],
        capped: list[_ContourMetrics],
        min_area_px: int,
        hsv: np.ndarray,
        cfg: ColorDetectorConfig,
    ) -> None:
        """One row per contour this class's mask produced this frame,
        evaluating every stage independently of the elif chain's own
        short-circuiting -- ``reject_reason`` only ever names the first
        filter that failed, but a --diag row needs every stage's own
        pass/fail so "does the model's contour appear at all, and if so
        which single stage kills it" is answered directly from the log
        rather than inferred. mean_hue/mean_sat/mean_val are restricted to
        the contour's own filled mask (cv2.mean with a per-contour mask),
        not the bounding box, so a concave silhouette's background pixels
        never leak into the colour reading.
        """
        kept_ids = {id(m) for m in kept}
        capped_ids = {id(m) for m in capped}
        aspect_lo, aspect_hi = cfg.aspect_ratio_range
        for i, m in enumerate(measured):
            contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [m.contour], -1, 255, -1)
            mean_h, mean_s, mean_v = cv2.mean(hsv, mask=contour_mask)[:3]

            area_pass = m.area >= min_area_px
            solidity_pass = m.solidity >= cfg.solidity_min
            aspect_pass = aspect_lo <= m.aspect_ratio <= aspect_hi
            circularity_pass = (not cfg.require_circularity) or m.circularity >= cfg.circularity_min

            if id(m) in kept_ids:
                cap_status = "PASS"
                verdict = "ACCEPT"
            elif id(m) in capped_ids:
                cap_status = "FAIL"
                verdict = "REJECT:class_cap"
            else:
                cap_status = "N/A"
                verdict = f"REJECT:{m.reject_reason}"

            logger.info(
                "DIAG %s#%d bbox=%s area=%.1f(min=%d %s) solidity=%.3f(min=%.2f %s) "
                "aspect_ratio=%.2f(range=%.1f-%.1f %s) circularity=%.3f(min=%.2f req=%s %s) "
                "class_cap=%s hue=%.1f sat=%.1f val=%.1f verdict=%s",
                class_name,
                i,
                m.bbox_px,
                m.area,
                min_area_px,
                "PASS" if area_pass else "FAIL",
                m.solidity,
                cfg.solidity_min,
                "PASS" if solidity_pass else "FAIL",
                m.aspect_ratio,
                aspect_lo,
                aspect_hi,
                "PASS" if aspect_pass else "FAIL",
                m.circularity,
                cfg.circularity_min,
                cfg.require_circularity,
                "PASS" if circularity_pass else "FAIL",
                cap_status,
                mean_h,
                mean_s,
                mean_v,
                verdict,
            )

    def _build_detection(
        self,
        m: _ContourMetrics,
        color_class: ColorClass,
        frame: Frame,
        width: int,
        height: int,
        px1: int,
        py1: int,
    ) -> Detection:
        cfg = self.config
        pixel_size = max(m.rect_w, m.rect_h)
        (ecx, ecy), _ = cv2.minEnclosingCircle(m.contour)
        cx_full, cy_full = ecx + px1, ecy + py1

        x1n = max(0.0, m.bbox_px[0] / width)
        y1n = max(0.0, m.bbox_px[1] / height)
        x2n = min(1.0, (m.bbox_px[0] + m.bbox_px[2]) / width)
        y2n = min(1.0, (m.bbox_px[1] + m.bbox_px[3]) / height)

        range_m: float | None = None
        range_source: RangeSource = "none"
        if frame.has_depth and frame.depth is not None:
            range_m = _depth_range_m(
                frame.depth, int(cx_full), int(cy_full), DEPTH_SAMPLE_WINDOW_PX
            )
            if range_m is not None:
                range_source = "depth"
        if range_m is None:
            range_m = _size_range_m(pixel_size, frame.intrinsics.fx, cfg.known_sizes_m)
            if range_m is not None:
                range_source = "size"

        # No ML confidence score exists in a classic-CV pipeline. Fill
        # ratio (contour area over its own bounding box) works for both
        # circular and elongated shapes, unlike circularity, which would
        # unfairly penalise a legitimately-detected aircraft silhouette.
        confidence = min(1.0, m.area / max(m.bbox_px[2] * m.bbox_px[3], 1))

        return Detection(
            bbox=(x1n, y1n, x2n, y2n),
            cls=None,
            confidence=confidence,
            range_m=range_m,
            range_source=range_source,
            source_layer=Layer.L2,
            iff=color_class.iff,
        )


# --- eyedropper calibration ---
# The tuning window's most useful addition: instead of guessing HSV
# thresholds against a hex-derived theory (see CLAUDE.md's "Physical
# measurements still outstanding"), the operator samples the actual
# printed model under the actual venue lighting and the detector derives
# its thresholds from those pixels.

# Below this linear hue spread (in OpenCV's 0-179 H units), a sample is
# already a tight, non-wrapping cluster -- no need to even check for
# wraparound. Above it, a genuinely wide/multi-coloured sample and a
# tight cluster straddling the 0/180 seam both show a large linear
# spread, so the circular check below is what actually tells them apart.
_HUE_WRAP_LINEAR_SPREAD_THRESHOLD = 90.0
# Mean resultant vector length (0 = uniformly spread around the circle,
# 1 = a single point) above which a wide-looking linear spread is judged
# to actually be a tight circular cluster, i.e. a wraparound rather than
# a genuinely multi-hued sample.
_HUE_WRAP_RESULTANT_THRESHOLD = 0.5
_DEFAULT_SAMPLE_MARGIN = 0.15
# A sample taken from a perfectly uniform patch (zero measured hue
# spread -- a real risk against a flat-lit or computer-generated swatch,
# confirmed empirically against SyntheticSource's own flat-fill targets)
# would otherwise derive a single-hue-value range: proportional padding
# (margin * span) is 0 when span is 0. A real camera always has some
# sensor/compression noise even on a uniform surface, so this floor is
# accounting for that inherent measurement noise, not guessing a wider
# range than what was actually observed.
_MIN_HUE_PAD = 2.0


@dataclass(frozen=True)
class ColorSample:
    """Percentile statistics from one eyedropper sample -- percentiles,
    not min/max, so a single stray outlier pixel (sensor noise, an
    anti-aliased edge pixel) cannot widen the derived range.

    ``hue_median``/``hue_p5``/``hue_p95`` are reported in a *shifted*
    domain when ``hue_wraps`` is true: every raw hue below 90 has 180
    added to it before the percentiles are computed, so a cluster that
    straddles the real 0/180 seam (red) becomes one contiguous interval
    instead of one that appears to span the whole axis. ``derive_
    thresholds`` knows how to split that shifted interval back into the
    two real hue ranges it actually covers -- callers that want the
    plain real-domain value should take ``hue_median % 180`` etc.
    themselves.
    """

    hue_median: float
    hue_p5: float
    hue_p95: float
    sat_median: float
    sat_p5: float
    val_median: float
    val_p5: float
    pixel_count: int
    hue_wraps: bool


_EMPTY_SAMPLE = ColorSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)


def _hue_wraps_around(hue: np.ndarray) -> bool:
    """Circular, not linear, spread is what actually distinguishes a
    sample that straddles the 0/180 seam from one that is genuinely
    multi-hued -- both look identical under a plain max-min or
    percentile spread on the raw values. Getting this wrong in either
    direction is exactly the failure mode to avoid: missing a real wrap
    produces one derived range spanning nearly the whole hue axis
    (matches everything); falsely detecting one on a genuinely wide
    sample would incorrectly split it in two.
    """
    if hue.size == 0:
        return False
    linear_spread = float(np.percentile(hue, 95) - np.percentile(hue, 5))
    if linear_spread < _HUE_WRAP_LINEAR_SPREAD_THRESHOLD:
        return False
    angles = hue * (2.0 * math.pi / 180.0)
    mean_cos = float(np.mean(np.cos(angles)))
    mean_sin = float(np.mean(np.sin(angles)))
    resultant = math.hypot(mean_cos, mean_sin)
    return resultant > _HUE_WRAP_RESULTANT_THRESHOLD


def _sample_from_pixels(hsv_pixels: np.ndarray) -> ColorSample:
    """``hsv_pixels`` is an (N, 3) array of raw HSV pixel values, in
    whatever pixel order/shape they were extracted -- a circular
    eyedropper click and a rectangular drag both funnel into this one
    statistics computation so they agree exactly on a uniform region
    (only the pixel *selection* differs between the two call sites).
    """
    if hsv_pixels.size == 0:
        return _EMPTY_SAMPLE
    hue = hsv_pixels[:, 0].astype(np.float64)
    sat = hsv_pixels[:, 1].astype(np.float64)
    val = hsv_pixels[:, 2].astype(np.float64)

    wraps = _hue_wraps_around(hue)
    hue_for_stats = np.where(hue < 90.0, hue + 180.0, hue) if wraps else hue

    return ColorSample(
        hue_median=float(np.median(hue_for_stats)),
        hue_p5=float(np.percentile(hue_for_stats, 5)),
        hue_p95=float(np.percentile(hue_for_stats, 95)),
        sat_median=float(np.median(sat)),
        sat_p5=float(np.percentile(sat, 5)),
        val_median=float(np.median(val)),
        val_p5=float(np.percentile(val, 5)),
        pixel_count=int(hsv_pixels.shape[0]),
        hue_wraps=wraps,
    )


def sample_region(
    frame: Frame, center_norm: tuple[float, float], radius_px: int = 15
) -> ColorSample:
    """Circular eyedropper sample centred on one clicked point.
    ``center_norm`` is normalised (x, y), matching every other bbox/point
    convention in this codebase; ``radius_px`` is real image pixels, not
    normalised, since a calibration click has a fixed physical purpose
    (how large a patch of the model to average) independent of frame
    resolution.
    """
    height, width = frame.image.shape[:2]
    cx = int(round(center_norm[0] * width))
    cy = int(round(center_norm[1] * height))
    y0, y1 = max(0, cy - radius_px), min(height, cy + radius_px + 1)
    x0, x1 = max(0, cx - radius_px), min(width, cx + radius_px + 1)
    patch = frame.image[y0:y1, x0:x1]
    if patch.size == 0:
        return _EMPTY_SAMPLE
    yy, xx = np.mgrid[y0:y1, x0:x1]
    circular_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return _sample_from_pixels(hsv_patch[circular_mask])


def sample_rectangle(frame: Frame, roi_norm: BoundingBox) -> ColorSample:
    """Rectangular eyedropper sample from a drag -- the same statistics
    as ``sample_region``, over every pixel in the rectangle rather than a
    circle, so the two agree exactly for a uniform region.
    """
    height, width = frame.image.shape[:2]
    x1n, y1n, x2n, y2n = roi_norm
    x0, y0 = int(round(x1n * width)), int(round(y1n * height))
    x1, y1 = int(round(x2n * width)), int(round(y2n * height))
    patch = frame.image[y0:y1, x0:x1]
    if patch.size == 0:
        return _EMPTY_SAMPLE
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return _sample_from_pixels(hsv_patch.reshape(-1, 3))


def derive_thresholds(
    sample: ColorSample, margin: float = _DEFAULT_SAMPLE_MARGIN
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    """(hue_ranges, sat_min, val_min) derived from a sample, widened by a
    safety margin: the hue interval is padded outward by ``margin`` of
    its own span on each side, and sat_min/val_min are each reduced by
    ``margin`` of their own measured p5 value, so a sample taken under
    slightly different lighting than the next frame still falls inside
    the derived thresholds.

    Returns *two* hue ranges, not one, when ``sample.hue_wraps`` is set
    -- reporting one wide range spanning ``[hue_p5, hue_p95]`` in that
    case would span nearly the entire hue axis and match everything,
    exactly the failure mode a wraparound sample must avoid.
    """
    sat_min = max(0, round(sample.sat_p5 * (1.0 - margin)))
    val_min = max(0, round(sample.val_p5 * (1.0 - margin)))

    span = sample.hue_p95 - sample.hue_p5
    pad = max(margin * span, _MIN_HUE_PAD)
    lo = sample.hue_p5 - pad
    hi = sample.hue_p95 + pad

    if not sample.hue_wraps:
        hue_ranges = ((max(0, round(lo)), min(179, round(hi))),)
        return hue_ranges, sat_min, val_min

    # Wrapped: lo/hi live in the shifted domain _sample_from_pixels used
    # to detect the wrap (raw hue < 90 shifted by +180, so the cluster
    # sits contiguously somewhere in roughly [90, 270)). Clamp to that
    # domain before splitting: extreme padding on a tiny sample could
    # otherwise push an endpoint past the shift point itself, which
    # would need a second wrap to interpret correctly and is not a
    # realistic calibration sample.
    lo = max(90.0, lo)
    hi = min(270.0, hi)
    if lo < 180.0 <= hi:
        range_high = (max(0, round(lo)), 179)
        range_low = (0, min(179, round(hi - 180.0)))
        return (range_high, range_low), sat_min, val_min
    # Padding never actually crossed the seam (a very tight, barely-
    # wrapping sample) -- one real range is enough.
    unshift = lambda v: v - 180.0 if v >= 180.0 else v  # noqa: E731
    hue_ranges = ((max(0, round(unshift(lo))), min(179, round(unshift(hi)))),)
    return hue_ranges, sat_min, val_min


@dataclass(frozen=True)
class NegativeSampleReport:
    """Whether one ``ColorClass`` currently accepts a "must not detect"
    sample (skin, a wall, clothing), and if so, the cheaper of sat_min/
    val_min to raise to exclude it -- "cheaper" meaning whichever is
    already closer to excluding the sample, on the reasoning that a
    smaller change is less likely to also exclude genuine targets whose
    own saturation/value sits close to the negative sample's.
    """

    class_name: str
    accepted: bool
    tighten_field: Literal["sat_min", "val_min"] | None
    suggested_value: int | None


def check_negative_sample(
    sample: ColorSample, config: ColorDetectorConfig
) -> tuple[NegativeSampleReport, ...]:
    """Turns "does this negative sample sit inside our thresholds" from
    guesswork into a measurement -- see the module's own hex-vs-measured
    HSV caveat. Checked against the sample's median (not every pixel):
    a calibration sample is deliberately taken from a fairly uniform
    patch, so the central tendency is what a real detection would key
    off too.
    """
    reports = []
    for color_class in config.classes:
        real_hue = sample.hue_median % 180.0
        hue_hits = any(lo <= real_hue <= hi for lo, hi in color_class.hue_ranges)
        accepted = (
            hue_hits
            and sample.sat_median >= color_class.sat_min
            and sample.val_median >= color_class.val_min
        )
        tighten_field: Literal["sat_min", "val_min"] | None = None
        suggested_value: int | None = None
        if accepted:
            sat_gap = sample.sat_p5 - color_class.sat_min
            val_gap = sample.val_p5 - color_class.val_min
            if sat_gap <= val_gap:
                tighten_field = "sat_min"
                suggested_value = round(sample.sat_p5) + 1
            else:
                tighten_field = "val_min"
                suggested_value = round(sample.val_p5) + 1
        reports.append(
            NegativeSampleReport(color_class.name, accepted, tighten_field, suggested_value)
        )
    return tuple(reports)


# --- tuning preset persistence ---
# The HSV tuning window is the only consumer of these; they live here
# rather than in ui/ because ColorDetectorConfig -- what a preset
# actually captures -- belongs to this module, the same reasoning
# geometry.calibration owns BoresightTable's own JSON persistence.


def config_to_dict(config: ColorDetectorConfig) -> dict:
    return {
        "morph_kernel": config.morph_kernel,
        "min_area_px": config.min_area_px,
        "circularity_min": config.circularity_min,
        "require_circularity": config.require_circularity,
        "solidity_min": config.solidity_min,
        "aspect_ratio_range": list(config.aspect_ratio_range),
        "max_detections_per_class": config.max_detections_per_class,
        "classes": {
            c.name: {
                "hue_ranges": [list(r) for r in c.hue_ranges],
                "sat_min": c.sat_min,
                "val_min": c.val_min,
            }
            for c in config.classes
        },
    }


def config_from_dict(data: dict) -> ColorDetectorConfig:
    """Missing keys fall back to the hex-derived defaults -- a preset
    need not cover every field, and a class absent from
    ``data["classes"]`` keeps its own default hue/sat/val rather than
    vanishing from the detector entirely.
    """
    defaults = ColorDetectorConfig()
    by_name = {c.name: c for c in defaults.classes}
    classes = []
    for name, class_defaults in by_name.items():
        entry = data.get("classes", {}).get(name)
        if entry is None:
            classes.append(class_defaults)
            continue
        classes.append(
            ColorClass(
                name=name,
                iff=class_defaults.iff,
                hue_ranges=tuple(
                    tuple(r) for r in entry.get("hue_ranges", class_defaults.hue_ranges)
                ),
                sat_min=entry.get("sat_min", class_defaults.sat_min),
                val_min=entry.get("val_min", class_defaults.val_min),
            )
        )
    aspect_range = data.get("aspect_ratio_range", defaults.aspect_ratio_range)
    return ColorDetectorConfig(
        classes=tuple(classes),
        morph_kernel=data.get("morph_kernel", defaults.morph_kernel),
        min_area_px=data.get("min_area_px", defaults.min_area_px),
        circularity_min=data.get("circularity_min", defaults.circularity_min),
        require_circularity=data.get("require_circularity", defaults.require_circularity),
        solidity_min=data.get("solidity_min", defaults.solidity_min),
        aspect_ratio_range=(aspect_range[0], aspect_range[1]),
        max_detections_per_class=data.get(
            "max_detections_per_class", defaults.max_detections_per_class
        ),
    )


def save_hsv_preset(config: ColorDetectorConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2))


def load_hsv_preset(path: Path) -> ColorDetectorConfig:
    """Missing or unreadable file -> the hex-derived defaults, not a
    crash -- same contract as geometry.calibration.load_boresight.
    """
    if not path.exists():
        return ColorDetectorConfig()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return ColorDetectorConfig()
    return config_from_dict(data)


def list_hsv_presets(directory: Path = DEFAULT_HSV_PRESETS_DIR) -> tuple[str, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(p.stem for p in directory.glob("*.json")))
