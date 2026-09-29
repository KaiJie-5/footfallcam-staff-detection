"""Staff = one clear marker, or repeated marker evidence moving with a person."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ConsensusConfig:
    marker_threshold: float = 0.87
    clear_marker_threshold: float = 0.97
    propagation_seconds: float = 3.0
    evidence_window_seconds: float = 1.2
    max_track_gap_seconds: float = 0.4
    max_center_jump_ratio: float = 0.8

    def __post_init__(self):
        if not 0.05 < self.marker_threshold <= self.clear_marker_threshold <= 1:
            raise ValueError('Require 0.05 < marker_threshold <= clear_marker_threshold <= 1')
        for name in ('propagation_seconds', 'evidence_window_seconds',
                     'max_track_gap_seconds', 'max_center_jump_ratio'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')

    @property
    def support_threshold(self):
        # A second view may be slightly blurrier than the best marker view.
        return self.marker_threshold - 0.05


def compress_frame_ranges(frames):
    ranges = []
    for frame in sorted(set(frames)):
        if ranges and frame == ranges[-1][1]+1:
            ranges[-1][1] = frame
        else:
            ranges.append([frame, frame])
    return ranges


def _valid_badge(d):
    badge = d.get('badge')
    if not badge or badge.get('bbox') is None or not badge.get('template'):
        return False
    score, contrast = badge.get('score', 0), badge.get('contrast', 0)
    if not math.isfinite(score) or not 0 <= score <= 1 or not math.isfinite(contrast) or contrast < 20:
        return False
    x, y, x2, y2 = d['bbox']
    a, b, c, e = badge['bbox']
    return x <= a < c <= x2 and y <= b < e <= y2


def _badge_center(d):
    a, b, c, e = d['badge']['bbox']
    return np.array([(a+c)/2, (b+e)/2])


def _moves_with_person(previous, current):
    """Reject fixed desk/chair patterns even if repeated scores are high.

    Compare source-image motion as well as relative location. At image edges the
    person box changes height as more of the body becomes visible.
    """
    if previous['badge']['template'] != current['badge']['template']:
        return False
    locations, sizes = [], []
    for d in (previous, current):
        x, y, x2, y2 = d['bbox']
        locations.append((_badge_center(d)-[x, y])/[x2-x, y2-y])
        sizes.append(math.hypot(x2-x, y2-y))
    if np.max(np.abs(locations[1]-locations[0])) > 0.4:
        return False
    scale = max(1.0, sum(sizes)/2)
    body_delta = np.asarray(current['center'])-previous['center']
    badge_delta = _badge_center(current)-_badge_center(previous)
    body_motion, badge_motion = np.linalg.norm(body_delta), np.linalg.norm(badge_delta)
    if min(body_motion, badge_motion) < 0.1*scale:
        return False
    same_direction = np.dot(body_delta, badge_delta)/(body_motion*badge_motion) >= 0.7
    similar_motion = np.linalg.norm(badge_delta-body_delta) <= 0.25*scale
    return bool(same_direction and similar_motion)


def _confirmed_frames(measured, fps, config):
    samples = [(f, d) for f, d in measured
               if _valid_badge(d) and d['badge']['score'] >= config.support_threshold]
    anchors = {f: 'clear_marker' for f, d in samples
               if d['badge']['score'] >= config.clear_marker_threshold}
    for index, (frame, first) in enumerate(samples):
        for other_frame, second in samples[index+1:]:
            if other_frame-frame > config.evidence_window_seconds*fps:
                break
            # Avoid treating nearly identical consecutive frames as confirmation.
            if other_frame-frame < 0.12*fps:
                continue
            if max(first['badge']['score'], second['badge']['score']) < config.marker_threshold:
                continue
            if _moves_with_person(first, second):
                anchors.setdefault(frame, 'consistent_marker')
                anchors.setdefault(other_frame, 'consistent_marker')
    return anchors


def classify_tracks(frames, fps, config=None):
    """Offline confirmation, with bounded propagation and no fabricated positions.

    Clear markers also support stationary staff. Weaker evidence requires motion
    to distinguish a badge from static patterns inside a loose person box.
    """
    config = config or ConsensusConfig()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive')
    segments, last = defaultdict(list), {}
    serial = 0
    for item in frames:
        frame = item['frame_id']
        for d in item['detections']:
            d.update(status='unknown', segment_id=None, reason='insufficient_marker_evidence',
                     evidence_frame=None, classification_source='unconfirmed')
        if not item['quality']['valid']:
            last.clear()
            for d in item['detections']:
                d['reason'] = 'invalid_frame'
            continue
        for d in item['detections']:
            tid = d['track_id']
            previous = last.get(tid)
            split = previous is None
            if previous:
                pf, pd = previous
                x, y, x2, y2 = pd['bbox']
                scale = max(1, math.hypot(x2-x, y2-y))
                jump = np.linalg.norm(np.asarray(d['center'])-pd['center'])/scale
                split = (frame-pf > max(1, round(config.max_track_gap_seconds*fps))
                         or jump > config.max_center_jump_ratio)
            if split:
                serial += 1
                sid = serial
            else:
                sid = previous[1]['segment_id']
            d['segment_id'] = sid
            segments[sid].append((frame, d))
            last[tid] = (frame, d)

    audits = []
    for sid, observations in segments.items():
        measured = [(f, d) for f, d in observations if d.get('badge') is not None]
        anchors = _confirmed_frames(measured, fps, config)
        anchor_array = np.asarray(sorted(anchors), dtype=int)
        for frame, d in observations:
            nearest = int(anchor_array[np.argmin(np.abs(anchor_array-frame))]) if len(anchor_array) else None
            if nearest is not None and abs(nearest-frame) <= config.propagation_seconds*fps:
                direct = nearest == frame
                d.update(status='staff', reason=anchors[nearest] if direct else 'track_continuity',
                         evidence_frame=nearest, classification_source='marker' if direct else 'tracked')
            elif _valid_badge(d) and d['badge']['score'] >= config.support_threshold:
                d.update(status='candidate', reason='unconfirmed_marker_match')
        ranked = sorted(measured, key=lambda pair: pair[1]['badge']['score'], reverse=True)
        audits.append(dict(segment_id=sid, track_id=observations[0][1]['track_id'],
                           start_frame=observations[0][0], end_frame=observations[-1][0],
                           observed_frames=len(observations), sampled_frames=len(measured),
                           confirmed_anchor_frames=sorted(anchors),
                           staff_frames=sum(d['status']=='staff' for _, d in observations),
                           best_evidence=[dict(frame_id=f, **d['badge']) for f, d in ranked[:3]]))
    return audits
