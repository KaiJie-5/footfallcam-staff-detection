"""Cache expensive person detections independently of marker/temporal experiments."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detector import OverheadPersonTracker
from src.video_io import FrameQuality, open_video


def cache_people(video, weights, out, device=None, imgsz=960, rotations=(0,), conf=0.1):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap, meta = open_video(video)
    meta.update(schema_version=2,
                weights=str(weights), imgsz=imgsz, rotations=list(rotations), conf=conf,
                ultralytics=importlib.metadata.version('ultralytics'))
    quality = FrameQuality()
    start = time.perf_counter()
    count = 0
    partial = out.with_suffix('.partial')
    try:
        tracker = OverheadPersonTracker(weights, device, conf, imgsz, meta['fps'], rotations)
        with partial.open('w', encoding='utf-8') as stream:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                q = quality.inspect(frame)
                detections = tracker.track_frame(frame, q['valid'])
                stream.write(json.dumps(dict(frame_id=count, quality=q, detections=detections))+'\n')
                count += 1
                if count % 100 == 0:
                    print(f'{count}/{meta["reported_frame_count"]} frames; {time.perf_counter()-start:.1f}s', flush=True)
    finally:
        cap.release()
    if not count:
        raise ValueError('No decoded frames')
    if meta['reported_frame_count'] > 0 and count != meta['reported_frame_count']:
        raise RuntimeError(f'Decoded {count} frames; container reports {meta["reported_frame_count"]}. Partial cache retained.')
    meta.update(decoded_frame_count=count, elapsed_seconds=time.perf_counter()-start)
    partial.replace(out)
    out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--weights', default='yolov8x.pt')
    parser.add_argument('--out', default='data/cache/people.jsonl')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--device', default=None)
    parser.add_argument('--rotations', type=int, nargs='+', default=[0], choices=[0,90,180,270])
    parser.add_argument('--conf', type=float, default=0.1)
    cache_people(**vars(parser.parse_args()))
