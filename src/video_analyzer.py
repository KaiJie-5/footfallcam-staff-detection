"""Video diagnostics. Motion is a quality/activity signal, not staff evidence."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video_io import FrameQuality, open_video


def analyze_video_stream(video_path, sample_rate=1):
    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise ValueError('sample_rate must be a positive integer')
    cap, meta = open_video(video_path)
    quality = FrameQuality()
    rows = []
    previous, previous_valid = None, False
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            q = quality.inspect(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = None
            if previous is not None and q['valid'] and previous_valid:
                delta = gray.astype(np.float32)-previous.astype(np.float32)
                # Subtract global exposure change before interpreting local change.
                motion = float(np.mean(np.abs(delta-np.median(delta))))
            if count % sample_rate == 0:
                rows.append(dict(frame_id=count, timestamp_sec=count/meta['fps'],
                                 brightness=q['brightness'], contrast=float(gray.std()),
                                 sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                                 motion_score=motion, valid=q['valid'], quality_reason=q['reason']))
            previous, previous_valid = gray, q['valid']
            count += 1
    finally:
        cap.release()
    if count == 0:
        raise ValueError('No video frames decoded')
    if meta['reported_frame_count'] > 0 and count != meta['reported_frame_count']:
        raise ValueError('Video decoding ended before the reported frame count')
    meta.update(decoded_frame_count=count, sample_rate=sample_rate, duration_seconds=count/meta['fps'])
    return meta, pd.DataFrame(rows)


def detect_motion_intervals(df, threshold_factor=3.0, sample_rate=1):
    """Robust activity threshold excludes sensor dropouts and NaN comparisons."""
    scores = df.loc[df['valid'], 'motion_score'].dropna()
    if scores.empty:
        return []
    median = float(scores.median())
    mad = float((scores-median).abs().median())
    cutoff = median + threshold_factor*max(1.4826*mad,0.1)
    intervals = []
    for row in df.itertuples():
        active = row.valid and pd.notna(row.motion_score) and row.motion_score > cutoff
        if active:
            if intervals and row.frame_id-intervals[-1]['end_frame'] <= sample_rate:
                intervals[-1]['end_frame'] = int(row.frame_id)
            else:
                intervals.append(dict(start_frame=int(row.frame_id),end_frame=int(row.frame_id)))
    return intervals


def generate_visual_reports(df, meta, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(4,1,figsize=(12,9),sharex=True)
    for ax,column in zip(axes,['motion_score','sharpness','brightness','contrast']):
        ax.plot(df.timestamp_sec,df[column],linewidth=1)
        ax.set_ylabel(column.replace('_',' '))
        ax.grid(alpha=.3)
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'Video diagnostics: {meta["width"]} x {meta["height"]}, {meta["fps"]:g} FPS')
    fig.tight_layout()
    fig.savefig(out/'video_eda_metrics.png',dpi=150)
    plt.close(fig)


def save_keyframe_contact_sheet(video_path, out_dir, num_samples=16):
    if num_samples < 1:
        raise ValueError('num_samples must be positive')
    cap, meta = open_video(video_path)
    if meta['reported_frame_count'] < 1:
        cap.release()
        raise ValueError('Cannot seek keyframes without a frame count')
    count=min(num_samples,meta['reported_frame_count'])
    indices=np.linspace(0,meta['reported_frame_count']-1,count,dtype=int)
    cols=min(4,count)
    fig,axes=plt.subplots(int(np.ceil(count/cols)),cols,figsize=(4*cols,3*np.ceil(count/cols)),squeeze=False)
    try:
        for ax in axes.flat:
            ax.axis('off')
        for ax,idx in zip(axes.flat,indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
            ok,frame=cap.read()
            if not ok:
                raise ValueError(f'Cannot read keyframe {idx}')
            ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            ax.set_title(f'Frame {idx} ({idx/meta["fps"]:.2f}s)')
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        fig.tight_layout()
        fig.savefig(Path(out_dir)/'keyframe_contact_sheet.jpg',dpi=120)
    finally:
        cap.release()
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True)
    p.add_argument('--out',default='data/output/eda')
    p.add_argument('--sample-rate',type=int,default=1)
    p.add_argument('--save-keyframes',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    meta,df=analyze_video_stream(args.video,args.sample_rate)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'frame_metrics.csv',index=False)
    summary=dict(metadata=meta,motion_intervals=detect_motion_intervals(df,sample_rate=args.sample_rate),
                 sampled_invalid_frames=df.loc[~df.valid,'frame_id'].tolist(),
                 interpretation='Activity and image quality only; neither marker identity nor person count')
    (out/'video_eda_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    generate_visual_reports(df,meta,out)
    if args.save_keyframes:
        save_keyframe_contact_sheet(args.video,out)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
