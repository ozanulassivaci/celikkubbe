"""TrackManager: turns per-frame Detections into stable, frozen Tracks.

Confirmation lives here, not in the engagement FSM: this module has the
frame-by-frame association data (frames_confirmed is exactly a count of
consecutive matches), so it owns the TENTATIVE -> CONFIRMED promotion. S2
ACQUIRE in engagement.py simply waits for a track to already be CONFIRMED.
Duplicating that counting logic in the FSM would be a second source of
truth for the same fact.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from celikkubbe.core import config
from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import (
    IFF,
    BoundingBox,
    Detection,
    Track,
    TrackStatus,
)
from celikkubbe.tracking.association import associate
from celikkubbe.tracking.kalman import (
    CentroidKalmanFilter,
    RangeKalmanFilter,
    measurement_noise_for_source,
)

CLASS_VOTE_WINDOW = 10
IFF_VOTE_WINDOW = 5  # Stage 3 IFF refresh cadence; see IFFGate


def _bbox_size(bbox: BoundingBox) -> tuple[float, float]:
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_from_centroid(cx: float, cy: float, size: tuple[float, float]) -> BoundingBox:
    w, h = size
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _majority(votes: deque) -> object:
    return Counter(votes).most_common(1)[0][0]


@dataclass
class _TrackState:
    track_id: int
    centroid_filter: CentroidKalmanFilter
    last_bbox_size: tuple[float, float]
    last_seen_t: float
    status: TrackStatus = TrackStatus.TENTATIVE
    frames_confirmed: int = 1
    ever_confirmed: bool = False
    range_filter: RangeKalmanFilter | None = None
    last_range_source: str = "none"
    confidence: float = 0.0
    cls_votes: deque = field(default_factory=lambda: deque(maxlen=CLASS_VOTE_WINDOW))
    iff_votes: deque = field(default_factory=lambda: deque(maxlen=IFF_VOTE_WINDOW))

    def predicted_bbox(self) -> BoundingBox:
        cx, cy = self.centroid_filter.position
        return _bbox_from_centroid(cx, cy, self.last_bbox_size)

    def to_track(self) -> Track:
        cx, cy = self.centroid_filter.position
        bbox = _bbox_from_centroid(cx, cy, self.last_bbox_size)
        range_m = self.range_filter.value if self.range_filter is not None else None
        return Track(
            track_id=self.track_id,
            cls=_majority(self.cls_votes) if self.cls_votes else None,
            confidence=self.confidence,
            range_m=range_m,
            range_source=self.last_range_source if range_m is not None else "none",
            iff=self._voted_iff(),
            bbox=bbox,
            velocity=self.centroid_filter.velocity,
            status=self.status,
            risk_score=0.0,  # filled in by priority.py downstream
            frames_confirmed=self.frames_confirmed,
            last_seen_t=self.last_seen_t,
        )

    def _voted_iff(self) -> IFF:
        # Fail-safe: any FRIENDLY in the recent window wins outright. A
        # friendly aircraft misread as hostile for a single frame must
        # never open the fire chain.
        if any(v is IFF.FRIENDLY for v in self.iff_votes):
            return IFF.FRIENDLY
        if not self.iff_votes:
            return IFF.UNKNOWN
        return _majority(self.iff_votes)


class TrackManager:
    """Owns the active track list. Emits frozen Track objects every update()."""

    def __init__(
        self,
        clock: Clock,
        confirm_frames: int = config.ACQUIRE_FRAMES,
        track_lost_ms: float = config.TRACK_LOST_MS,
        max_association_displacement: float | None = None,
    ) -> None:
        self._clock = clock
        self._confirm_frames = confirm_frames
        self._track_lost_s = track_lost_ms / 1000.0
        self._association_kwargs = (
            {}
            if max_association_displacement is None
            else {"max_displacement": max_association_displacement}
        )
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1
        self._last_update_t: float | None = None

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        # Tracks marked LOST were already emitted once, last call; they are
        # genuinely gone from this point on.
        self._tracks = {
            tid: state
            for tid, state in self._tracks.items()
            if state.status is not TrackStatus.LOST
        }

        dt = 0.0 if self._last_update_t is None else max(0.0, now - self._last_update_t)
        self._last_update_t = now

        for state in self._tracks.values():
            state.centroid_filter.predict(dt)
            if state.range_filter is not None:
                state.range_filter.predict()

        track_ids = list(self._tracks.keys())
        predicted_bboxes = [self._tracks[tid].predicted_bbox() for tid in track_ids]
        detection_bboxes = [d.bbox for d in detections]
        # Gate on the fail-safe *voted* IFF, not the single most recent
        # detection's colour: a track that has ever shown a FRIENDLY frame
        # stays gated as FRIENDLY for association too, consistent with the
        # same fail-safe reasoning behind Track.iff itself. The
        # alternative — gating on the latest raw colour — would let one
        # HOSTILE-coloured measurement immediately re-open a track that
        # fail-safe voting has already decided to protect.
        track_groups = [self._tracks[tid]._voted_iff() for tid in track_ids]
        detection_groups = [d.iff for d in detections]

        matches, unmatched_track_idx, unmatched_det_idx = associate(
            predicted_bboxes,
            detection_bboxes,
            track_groups=track_groups,
            detection_groups=detection_groups,
            **self._association_kwargs,
        )

        for track_idx, det_idx in matches:
            self._apply_match(self._tracks[track_ids[track_idx]], detections[det_idx], now)

        for track_idx in unmatched_track_idx:
            self._apply_miss(self._tracks[track_ids[track_idx]], now)

        for det_idx in unmatched_det_idx:
            self._create_track(detections[det_idx], now)

        return [state.to_track() for state in self._tracks.values()]

    def _apply_match(self, state: _TrackState, detection: Detection, now: float) -> None:
        cx, cy = _bbox_center(detection.bbox)
        state.centroid_filter.update(cx, cy)
        state.last_bbox_size = _bbox_size(detection.bbox)
        state.last_seen_t = now
        state.confidence = detection.confidence
        state.cls_votes.append(detection.cls)
        state.iff_votes.append(detection.iff)

        if detection.range_m is not None:
            noise = measurement_noise_for_source(detection.range_source)
            if state.range_filter is None:
                state.range_filter = RangeKalmanFilter(detection.range_m, noise)
            else:
                state.range_filter.update(detection.range_m, noise)
            state.last_range_source = detection.range_source

        if state.ever_confirmed:
            state.status = TrackStatus.CONFIRMED
            state.frames_confirmed += 1
        else:
            state.frames_confirmed += 1
            if state.frames_confirmed >= self._confirm_frames:
                state.ever_confirmed = True
                state.status = TrackStatus.CONFIRMED
            else:
                state.status = TrackStatus.TENTATIVE

    def _apply_miss(self, state: _TrackState, now: float) -> None:
        if now - state.last_seen_t > self._track_lost_s:
            state.status = TrackStatus.LOST
            return
        if not state.ever_confirmed:
            state.frames_confirmed = 0
        state.status = TrackStatus.COASTING

    def _create_track(self, detection: Detection, now: float) -> None:
        cx, cy = _bbox_center(detection.bbox)
        state = _TrackState(
            track_id=self._next_id,
            centroid_filter=CentroidKalmanFilter(cx, cy),
            last_bbox_size=_bbox_size(detection.bbox),
            last_seen_t=now,
            confidence=detection.confidence,
        )
        self._next_id += 1
        state.cls_votes.append(detection.cls)
        state.iff_votes.append(detection.iff)
        if detection.range_m is not None:
            noise = measurement_noise_for_source(detection.range_source)
            state.range_filter = RangeKalmanFilter(detection.range_m, noise)
            state.last_range_source = detection.range_source
        self._tracks[state.track_id] = state
