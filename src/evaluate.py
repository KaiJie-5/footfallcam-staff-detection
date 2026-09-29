"""Evaluate explicitly labelled frames; unlabelled frames are never negatives."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(predicted_frames, annotations, trajectories=None, match_radius=40.0):
    if not np.isfinite(match_radius) or match_radius <= 0:
        raise ValueError('match_radius must be finite and positive')
    predicted_frames=set(predicted_frames)
    trajectories=trajectories or {}
    seen=set()
    tp=fp=fn=tn=ignored=0
    location_tp=location_fp=location_fn=0
    location_frames=0
    errors=[]
    for row in annotations:
        fid=row['frame_id']
        if not isinstance(fid,int) or fid<0 or fid in seen:
            raise ValueError('Annotations need unique nonnegative integer frame IDs')
        seen.add(fid)
        truth=row['staff_present']
        if truth is None:
            ignored+=1
            continue
        if not isinstance(truth,bool):
            raise ValueError('staff_present must be true, false, or null (uncertain)')
        prediction=fid in predicted_frames
        tp+=int(truth and prediction)
        fp+=int(not truth and prediction)
        fn+=int(truth and not prediction)
        tn+=int(not truth and not prediction)
        if 'points' not in row:
            continue
        target=np.asarray(row['points'],dtype=float).reshape(-1,2)
        points=np.asarray(trajectories.get(fid,[]),dtype=float).reshape(-1,2)
        if not np.isfinite(target).all() or not np.isfinite(points).all():
            raise ValueError('Coordinates must be finite')
        if bool(len(target))!=truth:
            raise ValueError('Point annotations contradict staff_present')
        location_frames+=1
        matches=0
        if len(target) and len(points):
            distances=np.linalg.norm(target[:,None]-points[None,:],axis=2)
            # Large penalty maximizes number of valid matches before distance.
            penalty=(min(len(target),len(points))+1)*match_radius
            costs=np.where(distances<=match_radius,distances,penalty)
            a,b=linear_sum_assignment(costs)
            for i,j in zip(a,b):
                if distances[i,j]<=match_radius:
                    matches+=1
                    errors.append(float(distances[i,j]))
        location_tp+=matches
        location_fn+=len(target)-matches
        location_fp+=len(points)-matches
    def ratio(a,b):
        return a/b if b else None
    return dict(labelled_frames=tp+fp+fn+tn,ignored_uncertain_frames=ignored,
                tp=tp,fp=fp,fn=fn,tn=tn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
                f1=ratio(2*tp,2*tp+fp+fn),
                localization=dict(labelled_frames=location_frames,match_radius_pixels=match_radius,
                                  matched=location_tp,false_positives=location_fp,missed=location_fn,
                                  mean_error_pixels=float(np.mean(errors)) if errors else None,
                                  median_error_pixels=float(np.median(errors)) if errors else None),
                scope='Only explicitly annotated frames; these labels may be development data, not a held-out test')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,help='Pipeline output directory')
    parser.add_argument('--annotations',required=True,help='JSON list of frame annotations')
    parser.add_argument('--out',required=True)
    parser.add_argument('--match-radius',type=float,default=40.)
    args=parser.parse_args()
    results=Path(args.results)
    frames=json.loads((results/'staff_frames.json').read_text(encoding='utf-8'))['frames']
    annotations=json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    trajectories={}
    with (results/'staff_trajectories.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            trajectories.setdefault(int(row['frame_id']),[]).append([float(row['x']),float(row['y'])])
    metrics=evaluate(frames,annotations,trajectories,args.match_radius)
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
