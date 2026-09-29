"""
FootfallCam Staff Identification & Tracking Engine
Contains:
1. StaffTagDetector: Forensic visual verification of white staff name tags.
2. CorridorZoneFilter: Spatial polygon gating to filter seated office staff.
3. OverheadPersonTracker: YOLOv8 + ByteTrack multi-object tracker.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class StaffTagDetector:
    """
    Detects rectangular white staff name tags on the upper torso/chest region
    of overhead fisheye person crops.
    """
    def __init__(self, min_area=30, max_area=3500, min_wh_ratio=0.3, max_wh_ratio=3.0):
        self.min_area = min_area
        self.max_area = max_area
        self.min_wh_ratio = min_wh_ratio
        self.max_wh_ratio = max_wh_ratio

    def inspect_crop(self, person_crop):
        """
        Inspects person bounding box crop for a staff name tag.
        Returns:
            has_tag (bool): True if candidate tag matches badge profile.
            confidence (float): Metric [0.0 - 1.0].
            tag_rect (tuple): (x, y, w, h) relative to person crop, or None.
        """
        if person_crop is None or person_crop.size == 0:
            return False, 0.0, None

        h, w = person_crop.shape[:2]
        if h < 35 or w < 35:
            return False, 0.0, None

        # Focus on upper torso / chest area (avoid legs/shoes and top of head)
        y1, y2 = int(h * 0.15), int(h * 0.75)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        torso = person_crop[y1:y2, x1:x2]

        if torso.size == 0:
            return False, 0.0, None

        gray_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        
        # 1. White / high-luminance mask (Tag is predominantly bright white)
        # Adapt to average torso illumination
        torso_mean = np.mean(gray_torso)
        thresh_val = max(175, int(torso_mean + 35))
        _, white_mask = cv2.threshold(gray_torso, thresh_val, 255, cv2.THRESH_BINARY)

        # 2. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Contour geometry evaluation
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cand = None
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            wh_ratio = bw / float(bh)

            if not (self.min_wh_ratio <= wh_ratio <= self.max_wh_ratio):
                continue

            # Evaluate rectangularity (solidity)
            rect_area = bw * bh
            solidity = area / float(rect_area) if rect_area > 0 else 0
            if solidity < 0.55:
                continue

            # Contrast evaluation against immediate surrounding fabric
            pad = 4
            sy1 = max(0, by - pad)
            sy2 = min(gray_torso.shape[0], by + bh + pad)
            sx1 = max(0, bx - pad)
            sx2 = min(gray_torso.shape[1], bx + bw + pad)
            
            tag_pixels = gray_torso[by:by+bh, bx:bx+bw]
            surrounding_pixels = gray_torso[sy1:sy2, sx1:sx2]
            
            tag_lum = np.mean(tag_pixels) if tag_pixels.size > 0 else 0
            bg_lum = np.mean(surrounding_pixels) if surrounding_pixels.size > 0 else 0
            contrast_diff = tag_lum - bg_lum

            # Confidence based on brightness and contrast
            conf = min(1.0, (tag_lum / 255.0) * 0.6 + max(0, contrast_diff / 80.0) * 0.4)

            if conf > best_conf:
                best_conf = conf
                # Convert back to person_crop coordinate space
                best_cand = (bx + x1, by + y1, bw, bh)

        has_tag = best_conf >= 0.50
        return has_tag, float(best_conf), best_cand


class CorridorZoneFilter:
    """
    Defines the walkable corridor polygon to isolate subjects walking along
    the central walkway from seated background desk employees.
    """
    def __init__(self, polygon=None):
        # Default corridor bounds for 960x720 FootfallCam ceiling sensor
        if polygon is None:
            self.polygon = np.array([
                [330, 60],
                [640, 60],
                [670, 710],
                [310, 710]
            ], dtype=np.int32)
        else:
            self.polygon = np.array(polygon, dtype=np.int32)

    def is_inside(self, x, y):
        """Tests if a coordinate (x, y) falls inside the corridor polygon."""
        point = (float(x), float(y))
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def draw_zone(self, image, color=(0, 200, 255), thickness=2):
        """Draws corridor boundary lines on output visualization."""
        cv2.polylines(image, [self.polygon], isClosed=True, color=color, thickness=thickness)


class OverheadPersonTracker:
    """
    YOLOv8 + ByteTrack integration for persistent tracking of subjects in
    fisheye overhead surveillance video.
    """
    def __init__(self, model_weights="yolov8n.pt", device=None, conf_thresh=0.25):
        self.model = YOLO(model_weights)
        self.device = device
        self.conf_thresh = conf_thresh

    def track_frame(self, frame):
        """
        Runs tracking on a single frame.
        Returns:
            detections: List of dicts with:
                - 'track_id': int (or -1 if unassigned)
                - 'bbox': [x1, y1, x2, y2]
                - 'center': (cx, cy)
                - 'conf': float
        """
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],  # Person class only
            conf=self.conf_thresh,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [-1] * len(boxes)

        for bbox, conf, tid in zip(boxes, confs, ids):
            x1, y1, x2, y2 = map(int, bbox)
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            detections.append({
                "track_id": int(tid),
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "conf": float(conf)
            })

        return detections