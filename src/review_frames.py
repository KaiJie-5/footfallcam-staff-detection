"""Create report-ready contact sheets from consecutive original video frames."""
import argparse
import math
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video_io import open_video


def export_review_sheets(video, targets, out, context=3):
    """Export one 300-DPI PNG per target, with full frames and a highlighted target.

    Frame IDs are zero-based, matching staff_trajectories.csv. Decode in source
    order instead of seeking independently, so adjacent panels really are adjacent
    frames. A window crossing the video boundary is rejected, never padded with
    repeated frames or silently shortened.
    """
    targets = list(dict.fromkeys(targets))
    if not targets or any(not isinstance(f, int) or f < 0 for f in targets):
        raise ValueError('Provide at least one nonnegative integer target frame')
    if not isinstance(context, int) or context < 0:
        raise ValueError('context must be a nonnegative integer')

    cap, meta = open_video(video)
    wanted = set()
    frames = {}
    try:
        for target in targets:
            start, end = target-context, target+context
            if start < 0 or (meta['reported_frame_count'] > 0 and end >= meta['reported_frame_count']):
                raise ValueError(f'Frame {target} cannot have {context} frames on each side in this video')
            wanted.update(range(start, end+1))
        for frame_id in range(max(wanted)+1):
            if frame_id in wanted:
                ok, frame = cap.read()
                if ok:
                    frames[frame_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                ok = cap.grab()
            if not ok:
                raise ValueError(f'Video ended at frame {frame_id}; cannot complete the requested review windows')
    finally:
        cap.release()

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    count = 2*context+1
    columns = min(7, count)
    rows = math.ceil(count/columns)
    # Size panels for native-resolution frames. Keep header spacing in inches
    # so the title, subtitle and frame labels also fit on a single-row sheet.
    dpi = 300
    panel_width = max(4.0, meta['width']/(dpi*0.9))
    panel_height = panel_width*meta['height']/meta['width']+0.36
    figure_height = rows*panel_height+1.0
    paths = []
    for target in targets:
        start, end = target-context, target+context
        fig, axes = plt.subplots(rows, columns, squeeze=False,
                                 figsize=(columns*panel_width, figure_height), dpi=dpi)
        try:
            fig.patch.set_facecolor('white')
            fig.suptitle(f'Frame-by-frame review | Target frame {target}', fontsize=22, fontweight='bold',
                         y=1-0.08/figure_height)
            fig.text(0.5, 1-0.48/figure_height,
                     f'Frames {start}-{end} inclusive | {count} consecutive frames | '
                     f'{meta["fps"]:g} FPS | Original {meta["width"]} x {meta["height"]} pixels | Zero-based frame IDs',
                     ha='center', va='top', fontsize=12, color='#444444')
            for ax in axes.flat:
                ax.set_axis_off()
            for ax, frame_id in zip(axes.flat, range(start, end+1)):
                ax.set_axis_on()
                ax.imshow(frames[frame_id], interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                is_target = frame_id == target
                offset = 'TARGET' if is_target else f'{frame_id-target:+d}'
                color = '#bd5600' if is_target else '#333333'
                ax.set_title(f'Frame {frame_id} | {frame_id/meta["fps"]:.2f}s | {offset}',
                             fontsize=11, fontweight='bold' if is_target else 'normal', color=color, pad=8)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('#e87900' if is_target else '#bbbbbb')
                    spine.set_linewidth(3.5 if is_target else 0.6)
            fig.subplots_adjust(left=0.012, right=0.988, bottom=0.015,
                                top=1-1.0/figure_height, wspace=0.045, hspace=0.16)
            path = out/f'review_frame_{target:06d}.png'
            fig.savefig(path, dpi=dpi, facecolor='white',
                        metadata={'Title': f'Frame {target}: consecutive frames {start}-{end}',
                                  'Description': f'{count} original frames; zero-based numbering; target outlined in orange.'})
            paths.append(path)
            print(f'Saved {path} | frames {start}-{end} | {count} panels | '
                  f'{round(fig.get_figwidth()*dpi)} x {round(fig.get_figheight()*dpi)} pixels', flush=True)
        finally:
            plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default='data/raw/sample.mp4')
    parser.add_argument('--frames', type=int, nargs='+', required=True, help='Target frame IDs, zero-based')
    parser.add_argument('--context', type=int, default=3, help='Number of consecutive frames on each side (default: 3)')
    parser.add_argument('--out', default='data/output/refined/review')
    args = parser.parse_args()
    export_review_sheets(args.video, args.frames, args.out, args.context)


if __name__ == '__main__':
    main()
