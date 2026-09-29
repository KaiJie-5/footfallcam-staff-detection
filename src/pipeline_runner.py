"""Offline staff identification from person tracks and supplied marker references."""
from __future__ import annotations
import argparse
import csv
from contextlib import nullcontext
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detector import CorridorZoneFilter
from src.evidence import collect_evidence
from src.temporal import ConsensusConfig, classify_tracks, compress_frame_ranges
from src.video_io import open_video, validate_video_metadata

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ['frame_id', 'timestamp_sec', 'track_id', 'segment_id', 'x', 'y',
              'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'person_conf',
              'marker_sampled', 'marker_score', 'position_source', 'classification_source']


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', default='data/raw/sample.mp4')
    p.add_argument('--out', default='data/output/improved')
    p.add_argument('--weights', default='yolov8x.pt')
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--device', default=None)
    p.add_argument('--rotations', type=int, nargs='+', choices=[0,90,180,270], default=[0,180])
    p.add_argument('--people-cache', help='Reuse a detection JSONL and its .meta.json; records its producer settings')
    p.add_argument('--evidence-cache', help='Reuse marker evidence; rebuild after changing inputs or marker settings')
    p.add_argument('--sample-seconds', type=float, default=0.2)
    p.add_argument('--reference-dir', default=str(ROOT/'assets'))
    p.add_argument('--marker-threshold', type=float, default=0.87, help='Marker match requiring a consistent second view')
    p.add_argument('--clear-marker-threshold', type=float, default=0.97, help='Clear match that can confirm staff on its own')
    p.add_argument('--propagation-seconds', type=float, default=3.0)
    p.add_argument('--roi', help='JSON file containing normalized polygon vertices; default is whole image')
    p.add_argument('--save-video', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--debug', action='store_true', help='Also write every frame decision and the track audit')
    p.add_argument('--review-frames', type=int, nargs='+', help='Only export review PNGs around these frame IDs; skip detection')
    p.add_argument('--review-context', type=int, default=10, help='Frames before/after each review target (default: 10)')
    return p.parse_args(argv)


def load_evidence(path, video):
    path = Path(path)
    meta = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    cap, actual = open_video(video)
    cap.release()
    validate_video_metadata(meta, actual)
    with path.open(encoding='utf-8') as stream:
        frames = [json.loads(line) for line in stream]
    if len(frames) != meta['decoded_frame_count'] or any(item['frame_id'] != i for i,item in enumerate(frames)):
        raise ValueError('Evidence cache is incomplete or out of order')
    h,w = meta['height'], meta['width']
    for item in frames:
        ids = set()
        for d in item['detections']:
            x,y,a,b = d['bbox']
            if not (0 <= x < a <= w and 0 <= y < b <= h) or d['track_id'] in ids:
                raise ValueError('Invalid or duplicated cached person detection')
            ids.add(d['track_id'])
    return frames, meta


def export_results(frames, meta, out, roi, debug=False):
    staff_frames, rows = [], []
    log = (out/'frame_decisions.jsonl').open('w', encoding='utf-8') if debug else nullcontext()
    with log as decisions:
        for item in frames:
            fid = item['frame_id']
            exported = []
            for d in item['detections']:
                x,y = d['center']
                d['in_roi'] = roi.is_inside(x,y,meta['width'],meta['height'])
                if d['status'] != 'staff' or not d['in_roi']:
                    continue
                a,b,c,e = d['bbox']
                badge = d.get('badge')
                rows.append(dict(frame_id=fid, timestamp_sec=fid/meta['fps'], track_id=d['track_id'],
                                 segment_id=d['segment_id'], x=x, y=y, bbox_x1=a, bbox_y1=b, bbox_x2=c, bbox_y2=e,
                                 person_conf=d['conf'], marker_sampled=badge is not None,
                                 marker_score='' if badge is None else badge['score'], position_source='observed_box_center',
                                 classification_source=d['classification_source']))
                exported.append(d['segment_id'])
            if exported:
                staff_frames.append(fid)
            if decisions is not None:
                decisions.write(json.dumps(dict(**item, staff_present=bool(exported), staff_segment_ids=exported))+'\n')
    with (out/'staff_trajectories.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    task1 = dict(frame_index_base=0, frame_range_end='inclusive', total_staff_frames=len(staff_frames),
                 frame_ranges=compress_frame_ranges(staff_frames), frames=staff_frames)
    (out/'staff_frames.json').write_text(json.dumps(task1,indent=2),encoding='utf-8')
    return task1, rows


def render_video(video, out, frames, meta, roi, support_score=0.82):
    path = out/'annotated_output.mp4'
    if path.resolve() == Path(video).resolve():
        raise ValueError('Annotated output must not overwrite the input video')
    cap, _ = open_video(video)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), meta['fps'], (meta['width'],meta['height']))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video writer: {path}')
    count = 0
    try:
        for item in frames:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Source video ended during rendering')
            roi.draw_zone(frame)
            present = False
            for d in item['detections']:
                if not d.get('in_roi', True):
                    continue
                a,b,c,e = d['bbox']
                staff = d['status'] == 'staff'
                present |= staff
                color = (40,220,40) if staff else ((0,180,255) if d['status']=='candidate' else (130,130,130))
                cv2.rectangle(frame,(a,b),(c,e),color,2 if staff else 1)
                state = d['status']
                if staff:
                    state += ' (marker)' if d['classification_source'] == 'marker' else ' (tracked)'
                label = f'{state} T{d["track_id"]}/S{d["segment_id"]}'
                cv2.putText(frame,label,(a,max(15,b-5)),0,.43,color,1)
                badge=d.get('badge')
                if badge and badge['bbox'] and badge['score'] >= support_score:
                    x,y,x2,y2=badge['bbox']
                    cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),1)
                if staff:
                    x,y=map(round,d['center'])
                    cv2.drawMarker(frame,(x,y),(0,255,255),cv2.MARKER_CROSS,12,1)
            status='STAFF: marker + track evidence' if present else 'No confirmed staff evidence'
            if not item['quality']['valid']:
                status='UNOBSERVABLE: '+item['quality']['reason']
            cv2.rectangle(frame,(0,0),(min(meta['width'],550),52),(25,25,25),-1)
            cv2.putText(frame,f'Frame {item["frame_id"]} | {item["frame_id"]/meta["fps"]:.2f}s',(10,20),0,.5,(255,255,255),1)
            cv2.putText(frame,status,(10,42),0,.5,(255,255,255),1)
            writer.write(frame)
            count += 1
    finally:
        cap.release()
        writer.release()
    check, check_meta = open_video(path)
    check.release()
    if check_meta['reported_frame_count'] != count:
        raise RuntimeError('Rendered video frame count mismatch')


def evidence_sheet(video, frames, audits, out, threshold=.86):
    selected = sorted([a for a in audits if a['best_evidence'] and a['best_evidence'][0]['score'] >= threshold],
                      key=lambda a:(bool(a['staff_frames']), a['best_evidence'][0]['score']),reverse=True)[:24]
    if not selected:
        return
    cap,_ = open_video(video)
    sheet=np.full(((len(selected)+5)//6*240,6*200,3),30,np.uint8)
    try:
        for i,a in enumerate(selected):
            b=a['best_evidence'][0]
            cap.set(cv2.CAP_PROP_POS_FRAMES,b['frame_id'])
            ok,f=cap.read()
            if not ok:
                raise RuntimeError('Cannot read evidence frame')
            d=next(d for d in frames[b['frame_id']]['detections'] if d['segment_id']==a['segment_id'])
            x,y,x2,y2=d['bbox']
            crop=f[y:y2,x:x2].copy()
            bx,by,bx2,by2=b['bbox']
            cv2.rectangle(crop,(bx-x,by-y),(bx2-x,by2-y),(0,0,255),1)
            scale=min(190/crop.shape[1],160/crop.shape[0])
            crop=cv2.resize(crop,None,fx=scale,fy=scale)
            py,px=i//6*240,i%6*200
            sheet[py+70:py+70+crop.shape[0],px:px+crop.shape[1]]=crop
            cv2.putText(sheet,f'S{a["segment_id"]} T{a["track_id"]} f{b["frame_id"]}',(px+3,py+18),0,.42,(255,255,255),1)
            cv2.putText(sheet,f'{b["score"]:.3f} | {d["status"]}',(px+3,py+38),0,.42,(255,255,255),1)
            rule=d['reason']
            cv2.putText(sheet,rule,(px+3,py+56),0,.33,(255,255,255),1)
    finally:
        cap.release()
    if not cv2.imwrite(str(out/'marker_evidence.jpg'),sheet):
        raise RuntimeError('Cannot write evidence sheet')


def run_pipeline(argv=None):
    args=parse_arguments(argv)
    if args.review_frames is not None:
        from src.review_frames import export_review_sheets
        return export_review_sheets(args.video,args.review_frames,Path(args.out)/'review',args.review_context)
    start=time.perf_counter()
    cfg=ConsensusConfig(marker_threshold=args.marker_threshold,clear_marker_threshold=args.clear_marker_threshold,
                        propagation_seconds=args.propagation_seconds)
    if not math.isfinite(args.sample_seconds) or args.sample_seconds <= 0 or args.imgsz < 32:
        raise ValueError('sample-seconds must be positive and imgsz at least 32')
    out=Path(args.out)
    if args.save_video and (out/'annotated_output.mp4').resolve() == Path(args.video).resolve():
        raise ValueError('Choose an output directory different from the source video location')
    out.mkdir(parents=True,exist_ok=True)
    roi=CorridorZoneFilter(None if not args.roi else json.loads(Path(args.roi).read_text(encoding='utf-8')))
    evidence=args.evidence_cache
    if evidence is None:
        people=args.people_cache
        if people is None:
            from scripts.cache_people import cache_people
            people=out/'cache'/'people.jsonl'
            cache_people(args.video,args.weights,people,args.device,args.imgsz,args.rotations)
        evidence=out/'cache'/'evidence.jsonl'
        collect_evidence(args.video,people,evidence,args.sample_seconds,args.reference_dir)
    frames,meta=load_evidence(evidence,args.video)
    audits=classify_tracks(frames,meta['fps'],cfg)
    task1,rows=export_results(frames,meta,out,roi,args.debug)
    if args.debug:
        (out/'track_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    evidence_sheet(args.video,frames,audits,out)
    if args.save_video:
        render_video(args.video,out,frames,meta,roi,cfg.support_threshold)
    summary=dict(schema_version=3,task_1_staff_frame_count=len(task1['frames']),
                 task_1_frame_ranges=task1['frame_ranges'],task_2_trajectory_points=len(rows),
                 identified_staff_track_ids=sorted({r['track_id'] for r in rows}),
                 staff_segment_ids=sorted({r['segment_id'] for r in rows}),
                 invalid_frame_ranges=compress_frame_ranges([f['frame_id'] for f in frames if not f['quality']['valid']]),
                 video=meta,consensus=asdict(cfg),roi=None if roi.polygon is None else roi.polygon.tolist(),
                 confirmation_policy='one clear marker, or two nearby matches moving with the person',
                 coordinate_system='source image pixels, top-left origin, bounding-box center; not calibrated floor coordinates',
                 mode='offline; future marker evidence can support earlier observations within propagation_seconds',
                 uncertainty='No confirmed evidence is not proof of absence. Missing detections are not interpolated.',
                 annotated_video='annotated_output.mp4' if args.save_video else None,
                 elapsed_seconds=time.perf_counter()-start,
                 packages={p:importlib.metadata.version(p) for p in ['numpy','opencv-python','ultralytics','torch']})
    (out/'staff_presence_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['task_1_staff_frame_count','task_1_frame_ranges','task_2_trajectory_points','identified_staff_track_ids']},indent=2))
    return summary


if __name__=='__main__':
    run_pipeline()
