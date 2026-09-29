"""Matched-filter evidence for the supplied three-bar marker, not generic whiteness.

Scores are image correlations, NOT calibrated probabilities. Templates come from
the company's supplied task illustrations; they are explicit reference examples.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np


@dataclass
class BadgeEvidence:
    score: float = 0.0
    bbox: list | None = None
    contrast: float = 0.0
    angle: int = 0
    template: str = ''

    def as_dict(self):
        return dict(score=round(self.score, 5), bbox=self.bbox,
                    contrast=round(self.contrast, 3), angle=self.angle, template=self.template)


class StaffTagDetector:
    """Rotation/scale search over the complete crop; no upright torso assumption."""
    def __init__(self, widths=(16, 20, 24, 30, 38), angle_step=15, reference_dir=None):
        if angle_step < 1 or not widths or any(w < 8 for w in widths):
            raise ValueError('Marker widths must be at least 8 and angle_step positive')
        self.templates = []
        reference_dir = Path(reference_dir or Path(__file__).resolve().parents[1]/'assets')
        manifest = json.loads((reference_dir/'marker_references.json').read_text(encoding='utf-8'))
        if not manifest.get('templates'):
            raise ValueError('Marker reference list is empty')
        for entry in manifest['templates']:
            original = cv2.imread(str(reference_dir/entry['file']), cv2.IMREAD_GRAYSCALE)
            if original is None or original.std() < 5:
                raise ValueError(f'Invalid marker template: {entry["file"]}')
            for width in sorted(set(widths) | {original.shape[1]}):
                height = max(6, round(width*original.shape[0]/original.shape[1]))
                source = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
                for angle in range(0, 360, angle_step):
                    transform = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), angle, 1)
                    c, s = abs(transform[0, 0]), abs(transform[0, 1])
                    bw, bh = int(np.ceil(width*c+height*s-1e-9)), int(np.ceil(height*c+width*s-1e-9))
                    transform[:, 2] += [(bw-width)/2, (bh-height)/2]
                    template = cv2.warpAffine(source, transform, (bw, bh), borderValue=127)
                    mask = cv2.warpAffine(np.full_like(source, 255), transform, (bw, bh), flags=cv2.INTER_NEAREST)
                    values = template[mask > 0]
                    white = (template >= np.percentile(values, 75)) & (mask > 0)
                    black = (template <= np.percentile(values, 25)) & (mask > 0)
                    self.templates.append((template, mask, angle, entry['file'], white, black))

    def inspect_crop(self, crop):
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 20:
            return BadgeEvidence()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        original_h, original_w = gray.shape
        # Limit compute and make large person crops comparable across resolutions.
        scale = min(1.0, 240.0/max(gray.shape))
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = gray.shape
        best = BadgeEvidence()
        for template, mask, angle, name, white, black in self.templates:
            th, tw = template.shape
            if th > h or tw > w or max(th, tw) > 0.55*max(h, w):
                continue
            scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            np.nan_to_num(scores, copy=False, nan=-1, posinf=-1, neginf=-1)
            # A box can include adjacent desks; reject its outermost strip.
            ys, xs = np.ogrid[:scores.shape[0], :scores.shape[1]]
            valid = ((xs+tw/2 > w*0.12) & (xs+tw/2 < w*0.88) &
                     (ys+th/2 > h*0.10) & (ys+th/2 < h*0.90))
            scores[~valid] = -1
            _, score, _, loc = cv2.minMaxLoc(scores)
            if score <= best.score:
                continue
            x, y = loc
            patch = gray[y:y+th, x:x+tw]
            if not white.any() or not black.any():
                continue
            contrast = float(patch[white].mean() - patch[black].mean())
            # Correlation alone is unstable on smooth/noisy nearly-flat patches.
            if contrast < 20:
                continue
            best = BadgeEvidence(float(np.clip(score, 0, 1)),
                                 [int(x/scale), int(y/scale), min(original_w,int(np.ceil((x+tw)/scale))),
                                  min(original_h,int(np.ceil((y+th)/scale)))], contrast, angle, name)
        return best
