"""Conservative local temporal consensus; track IDs are not staff identities."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np


@dataclass(frozen=True)
class ConsensusConfig:
    strong_score: float = 0.9
    support_score: float = 0.84
    min_strong_hits: int = 2
    min_support_hits: int = 3
    evidence_window_seconds: float = 1.2
    propagation_seconds: float = 3.0
    max_track_gap_seconds: float = 0.4
    max_center_jump_ratio: float = 0.8

    def __post_init__(self):
        if not 0 < self.support_score <= self.strong_score <= 1:
            raise ValueError('Require 0 < support_score <= strong_score <= 1')
        if self.min_strong_hits < 1 or self.min_support_hits < self.min_strong_hits:
            raise ValueError('Invalid evidence counts')
        for name in ('evidence_window_seconds', 'propagation_seconds', 'max_track_gap_seconds', 'max_center_jump_ratio'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


def compress_frame_ranges(frames):
    frames = sorted(set(frames))
    ranges = []
    for frame in frames:
        if ranges and frame == ranges[-1][1]+1:
            ranges[-1][1] = frame
        else:
            ranges.append([frame, frame])
    return ranges


def classify_tracks(frames, fps, config=None):
    """Split discontinuities; require repeated local evidence; bound propagation.

    Offline: confirmation may propagate backwards. Never invent boxes on missed or
    invalid frames, bridge blackouts, or merge unrelated tracker IDs.
    """
    config = config or ConsensusConfig()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive')
    segments, last = defaultdict(list), {}
    serial = 0
    for item in frames:
        f = item['frame_id']
        if not item['quality']['valid']:
            last.clear()
            for d in item['detections']:
                d.update(status='unknown', segment_id=None, reason='invalid_frame')
            continue
        for d in item['detections']:
            tid = d['track_id']
            previous = last.get(tid)
            split = previous is None
            if previous:
                pf, pd = previous
                box = pd['bbox']
                scale = max(1, np.hypot(box[2]-box[0], box[3]-box[1]))
                jump = np.linalg.norm(np.asarray(d['center'])-pd['center'])/scale
                split = f-pf > max(1, round(config.max_track_gap_seconds*fps)) or jump > config.max_center_jump_ratio
            if split:
                serial += 1
                sid = serial
            else:
                sid = previous[1]['segment_id']
            d.update(segment_id=sid, status='unknown', reason='insufficient_marker_evidence')
            segments[sid].append((f,d))
            last[tid] = (f,d)

    audits = []
    for sid, observations in segments.items():
        measured = [(f,d) for f,d in observations if d.get('badge') is not None]
        support = [(f,d) for f,d in measured if d['badge']['score'] >= config.support_score]
        confirmed_anchors = set()
        for start, (frame, _) in enumerate(support):
            window = []
            for other, d in support[start:]:
                if other-frame > config.evidence_window_seconds*fps:
                    break
                window.append((other,d))
            strong = [(f,d) for f,d in window if d['badge']['score'] >= config.strong_score]
            if len(window) < config.min_support_hits or len(strong) < config.min_strong_hits:
                continue
            # The badge must stay on a consistent part of the person box. This
            # rejects matches jumping between objects included in a loose bbox.
            relative = []
            for _, d in strong:
                a,b,c,e = d['badge']['bbox']
                x,y,x2,y2 = d['bbox']
                relative.append([((a+c)/2-x)/(x2-x), ((b+e)/2-y)/(y2-y)])
            if np.max(np.ptp(np.asarray(relative), axis=0)) > 0.35:
                continue
            confirmed_anchors.update(f for f,_ in window)
        anchor_array = np.asarray(sorted(confirmed_anchors), dtype=int)
        for f,d in observations:
            if len(anchor_array) and np.min(np.abs(anchor_array-f)) <= config.propagation_seconds*fps:
                d.update(status='staff', reason='repeated_marker_evidence_on_continuous_track')
            elif d.get('badge') and d['badge']['score'] >= config.support_score:
                d.update(status='candidate', reason='unconfirmed_marker_match')
        ranked = sorted(measured, key=lambda pair: pair[1]['badge']['score'], reverse=True)
        audits.append(dict(segment_id=sid, track_id=observations[0][1]['track_id'],
                           start_frame=observations[0][0], end_frame=observations[-1][0],
                           observed_frames=len(observations), sampled_frames=len(measured),
                           support_hits=len(support), strong_hits=sum(d['badge']['score'] >= config.strong_score for _,d in measured),
                           confirmed_anchor_frames=sorted(confirmed_anchors),
                           staff_frames=sum(d['status']=='staff' for _,d in observations),
                           best_evidence=[dict(frame_id=f, **d['badge']) for f,d in ranked[:3]]))
    return audits
