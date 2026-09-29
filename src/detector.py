"""Person detection and association, independent of staff identity."""
from __future__ import annotations
import inspect
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np


def restore_boxes(boxes, rotation, width, height):
    """Invert np.rot90; xyxy boxes use exclusive right/bottom edges."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    a, b, c, d = boxes.T
    if rotation == 0:
        return boxes.copy()
    if rotation == 90:
        return np.stack((width-d, a, width-b, c), axis=1)
    if rotation == 180:
        return np.stack((width-c, height-d, width-a, height-b), axis=1)
    if rotation == 270:
        return np.stack((b, height-c, d, height-a), axis=1)
    raise ValueError('rotation must be 0, 90, 180 or 270')


class OverheadPersonTracker:
    """Fuse rotated detections BEFORE one association in source coordinates."""
    def __init__(self, model_weights='yolov8x.pt', device=None, conf_thresh=0.1,
                 imgsz=960, fps=25.0, rotations=(0,), buffer_seconds=0.6):
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        from torchvision.ops import nms
        import torch
        if not Path(model_weights).is_file():
            raise FileNotFoundError(f'Model file required: {model_weights}. Download weights explicitly first.')
        if not 0 < conf_thresh <= 0.1:
            raise ValueError('conf must be in (0, 0.1] to preserve ByteTrack low-score recovery')
        if not rotations or any(r not in (0, 90, 180, 270) for r in rotations):
            raise ValueError('Invalid rotations')
        self.model = YOLO(model_weights)
        self.device, self.conf_thresh, self.imgsz = device, conf_thresh, imgsz
        self.rotations = tuple(dict.fromkeys(rotations))
        self.Boxes, self.nms, self.torch = Boxes, nms, torch
        args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                               new_track_thresh=0.35, track_buffer=max(1, round(fps*buffer_seconds)),
                               match_thresh=0.8, fuse_score=True)
        if 'frame_rate' in inspect.signature(BYTETracker).parameters:
            self.tracker = BYTETracker(args, frame_rate=30)
        else:
            self.tracker = BYTETracker(args)

    def track_frame(self, frame, valid=True):
        h, w = frame.shape[:2]
        detections = []
        if valid:
            for angle in self.rotations:
                rotated = np.ascontiguousarray(np.rot90(frame, angle//90))
                result = self.model.predict(rotated, classes=[0], conf=self.conf_thresh,
                                            imgsz=self.imgsz, device=self.device, verbose=False)[0]
                if len(result.boxes):
                    data = result.boxes.data.cpu().numpy().copy()
                    data[:, :4] = restore_boxes(data[:, :4], angle, w, h)
                    detections.append(data)
        data = np.concatenate(detections) if detections else np.empty((0, 6), np.float32)
        if len(data):
            data[:, [0, 2]] = data[:, [0, 2]].clip(0, w)
            data[:, [1, 3]] = data[:, [1, 3]].clip(0, h)
            keep = self.nms(self.torch.from_numpy(data[:, :4].copy()),
                            self.torch.from_numpy(data[:, 4].copy()), 0.5).numpy()
            data = data[keep]
        tracks = self.tracker.update(self.Boxes(data, (h, w)), frame)
        output = []
        for track in tracks:
            raw = data[int(track[-1])]
            x1, y1 = np.floor(raw[:2]).astype(int)
            x2, y2 = np.ceil(raw[2:4]).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            output.append(dict(track_id=int(track[4]), bbox=[int(x1), int(y1), int(x2), int(y2)],
                               center=[float((x1+x2)/2), float((y1+y2)/2)], conf=float(raw[4])))
        return output


class CorridorZoneFilter:
    """Optional normalized output ROI; never evidence of staff identity."""
    def __init__(self, polygon=None):
        self.polygon = None if polygon is None else np.asarray(polygon, dtype=np.float32)
        if self.polygon is not None:
            if self.polygon.ndim != 2 or self.polygon.shape[1] != 2 or len(self.polygon) < 3:
                raise ValueError('ROI must have at least three [x, y] vertices')
            if not np.isfinite(self.polygon).all() or np.any((self.polygon < 0) | (self.polygon > 1)):
                raise ValueError('ROI coordinates must be normalized to [0, 1]')
            if cv2.contourArea(self.polygon) <= 0:
                raise ValueError('ROI polygon has zero area')

    def is_inside(self, x, y, width, height):
        return self.polygon is None or cv2.pointPolygonTest(self.polygon, (x/width, y/height), False) >= 0

    def draw_zone(self, image):
        if self.polygon is not None:
            h, w = image.shape[:2]
            polygon = np.rint(self.polygon * [w, h]).astype(np.int32)
            cv2.polylines(image, [polygon], True, (0, 220, 255), 2)
