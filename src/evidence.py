"""Replay cached detections and collect compact, inspectable marker evidence."""
from pathlib import Path
import json
import math
import time
from .badge import StaffTagDetector
from .video_io import open_video, validate_video_metadata


def collect_evidence(video, cache, out, sample_seconds=0.2, reference_dir=None):
    cache = Path(cache)
    out = Path(out)
    if cache.resolve() == out.resolve():
        raise ValueError('Evidence output must not overwrite the person input cache')
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(cache.with_suffix('.meta.json').read_text(encoding='utf-8'))
    cap, video_meta = open_video(video)
    try:
        validate_video_metadata(meta, video_meta)
        if not math.isfinite(sample_seconds) or sample_seconds <= 0:
            raise ValueError('sample_seconds must be positive')
        detector = StaffTagDetector(reference_dir=reference_dir)
    except Exception:
        cap.release()
        raise
    step = max(1, round(meta['fps']*sample_seconds))
    count, start = 0, time.perf_counter()
    partial = out.with_suffix('.partial')
    try:
        with cache.open(encoding='utf-8') as source, partial.open('w', encoding='utf-8') as dest:
            for line in source:
                item = json.loads(line)
                if item['frame_id'] != count:
                    raise ValueError('Cache must contain each frame exactly once in source order')
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Video ended before the detection cache')
                for d in item['detections']:
                    d['badge'] = None
                    if item['quality']['valid'] and count % step == 0:
                        x1,y1,x2,y2 = d['bbox']
                        if not (0 <= x1 < x2 <= meta['width'] and 0 <= y1 < y2 <= meta['height']):
                            raise ValueError('Invalid cached bounding box')
                        e = detector.inspect_crop(frame[y1:y2, x1:x2]).as_dict()
                        if e['bbox']:
                            a,b,c,f = e['bbox']
                            e['bbox'] = [a+x1,b+y1,c+x1,f+y1]
                        d['badge'] = e
                dest.write(json.dumps(item)+'\n')
                count += 1
                if count % 100 == 0:
                    print(f'Marker evidence: {count}/{meta["decoded_frame_count"]}; {time.perf_counter()-start:.1f}s', flush=True)
            if cap.read()[0] or count != meta['decoded_frame_count']:
                raise ValueError('Cache/video frame count mismatch')
    finally:
        cap.release()
    partial.replace(out)
    meta.update(marker_sample_step=step, marker_elapsed_seconds=time.perf_counter()-start,
                reference_dir=str(reference_dir or Path(__file__).resolve().parents[1]/'assets'))
    out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta
