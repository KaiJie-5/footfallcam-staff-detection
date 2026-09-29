"""
FootfallCam Staff Identification & Tracking Engine (High-Precision Edition)
Contains:
1. StaffTagDetector: Forensic visual verification of white staff name tags.
2. CorridorZoneFilter: Narrowed spatial polygon to strictly isolate the walkway.
3. OverheadPersonTracker: High-precision YOLOv8x / YOLO11x + ByteTrack at native resolution.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class StaffTagDetector:
    """
    Detects rectangular white staff name tags on the upper torso/chest region
    of overhead fisheye person crops.
    """
    def __init__(self, min_area=25, max_area=3500, min_wh_ratio=0.35, max_wh_ratio=2.8):
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
        y1, y2 = int(h * 0.15), int(h * 0.72)
        x1, x2 = int(w * 0.18), int(w * 0.82)
        torso = person_crop[y1:y2, x1:x2]

        if torso.size == 0:
            return False, 0.0, None

        gray_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        
        # Adapt luminance threshold to ambient torso illumination
        torso_mean = np.mean(gray_torso)
        thresh_val = max(175, int(torso_mean + 38))
        _, white_mask = cv2.threshold(gray_torso, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        # Contour geometry evaluation
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
            if solidity < 0.58:
                continue

            # Contrast evaluation against surrounding fabric
            pad = 5
            sy1 = max(0, by - pad)
            sy2 = min(gray_torso.shape[0], by + bh + pad)
            sx1 = max(0, bx - pad)
            sx2 = min(gray_torso.shape[1], bx + bw + pad)
            
            tag_pixels = gray_torso[by:by+bh, bx:bx+bw]
            surrounding_pixels = gray_torso[sy1:sy2, sx1:sx2]
            
            tag_lum = np.mean(tag_pixels) if tag_pixels.size > 0 else 0
            bg_lum = np.mean(surrounding_pixels) if surrounding_pixels.size > 0 else 0
            contrast_diff = tag_lum - bg_lum

            conf = min(1.0, (tag_lum / 255.0) * 0.55 + max(0, contrast_diff / 75.0) * 0.45)

            if conf > best_conf:
                best_conf = conf
                best_cand = (bx + x1, by + y1, bw, bh)

        has_tag = best_conf >= 0.52
        return has_tag, float(best_conf), best_cand


class CorridorZoneFilter:
    """
    Defines a narrowed corridor polygon to strictly track the central aisle
    and eliminate background employees seated at desks.
    """
    PRESETS = {
        # Tight aisle: strictly covers the floor space between desk edges
        "tight": np.array([
            [440, 50],   # Top-Left
            [595, 50],   # Top-Right
            [615, 710],  # Bottom-Right
            [455, 710]   # Bottom-Left
        ], dtype=np.int32),
        # Ultra-tight aisle: center-line walking track only
        "ultratight": np.array([
            [465, 80],
            [575, 80],
            [590, 680],
            [475, 680]
        ], dtype=np.int32),
        # Standard: slightly wider tolerance
        "standard": np.array([
            [410, 50],
            [620, 50],
            [640, 710],
            [425, 710]
        ], dtype=np.int32)
    }

    def __init__(self, preset="tight", custom_polygon=None):
        if custom_polygon is not None:
            self.polygon = np.array(custom_polygon, dtype=np.int32)
        else:
            self.polygon = self.PRESETS.get(preset, self.PRESETS["tight"])

    def is_inside(self, x, y):
        """Tests if coordinate (x, y) falls inside the active corridor polygon."""
        point = (float(x), float(y))
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def draw_zone(self, image, color=(0, 220, 255), thickness=2):
        """Draws corridor boundary lines on output visualization."""
        cv2.polylines(image, [self.polygon], isClosed=True, color=color, thickness=thickness)
        # Add visual label at top of aisle
        tx, ty = self.polygon[0][0] + 10, self.polygon[0][1] + 25
        cv2.putText(image, "CORRIDOR ZONE", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


class OverheadPersonTracker:
    """
    High-Precision YOLOv8x / YOLO11x + ByteTrack integration.
    Operates at native resolution (imgsz=960) to avoid downsampling blur.
    """
    def __init__(self, model_weights="yolov8x.pt", device=None, conf_thresh=0.30, imgsz=960):
        print(f"[INFO] Loading High-Precision Detection Model: {model_weights} (imgsz={imgsz})")
        self.model = YOLO(model_weights)
        self.device = device
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

    def track_frame(self, frame):
        """
        Runs high-precision inference and ByteTrack association.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],  # Person class only
            conf=self.conf_thresh,
            imgsz=self.imgsz,
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