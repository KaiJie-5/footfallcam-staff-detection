"""Shared video contracts and auditable frame quality checks."""
from collections import deque
from pathlib import Path
import cv2
import numpy as np


def open_video(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Cannot decode video: {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = [int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError('Video has invalid FPS or dimensions; no implicit 25 FPS fallback')
    return cap, dict(video_name=Path(path).name, video_size_bytes=Path(path).stat().st_size,
                     width=width, height=height, fps=fps,
                     reported_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


def validate_video_metadata(expected, actual):
    """Basic cache mistake check. Regenerate caches when input content changes."""
    keys = ('video_name', 'video_size_bytes', 'width', 'height', 'fps', 'reported_frame_count')
    if any(expected.get(k) != actual.get(k) for k in keys):
        raise ValueError('Cache metadata does not match this video; regenerate the cache')


class FrameQuality:
    """Flag blackout/recovery from recent valid luminance, without clip frame numbers."""
    def __init__(self):
        self.history = deque(maxlen=50)

    def inspect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        reference = float(np.median(self.history)) if self.history else mean
        reason = 'ok'
        if mean < 15:
            reason = 'blackout'
        elif len(self.history) >= 5 and mean < 0.55*reference:
            reason = 'exposure_drop'
        if reason == 'ok':
            self.history.append(mean)
        return dict(valid=reason == 'ok', reason=reason, brightness=round(mean, 3))
