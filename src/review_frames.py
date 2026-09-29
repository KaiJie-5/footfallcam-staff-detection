"""Create report-ready contact sheets of tracked-person crops."""
import argparse
import csv
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


def _review_boxes(trajectories, targets, context, track_ids):
    """Follow the target's staff track and segment through its entire window."""
    if track_ids is not None and len(track_ids) != len(targets):
        raise ValueError('Provide one review track ID per target frame')
    wanted = {f for target in targets for f in range(target-context, target+context+1)}
    by_frame = {}
    with Path(trajectories).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'frame_id', 'track_id', 'segment_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Review requires a staff_trajectories.csv with track, segment and bounding-box columns')
        for row in reader:
            frame_id = int(row['frame_id'])
            if frame_id in wanted:
                by_frame.setdefault(frame_id, []).append(dict(
                    track_id=int(row['track_id']), segment_id=int(row['segment_id']),
                    bbox=tuple(float(row[key]) for key in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'))))
    windows = {}
    for index, target in enumerate(targets):
        candidates = by_frame.get(target, [])
        if track_ids is not None:
            candidates = [row for row in candidates if row['track_id'] == track_ids[index]]
        if len(candidates) != 1:
            raise ValueError(f'Frame {target}: expected one staff box, found {len(candidates)}. '
                             'Check the trajectories CSV; specify review track IDs if multiple staff are present.')
        anchor = candidates[0]
        boxes = {}
        for frame_id in range(target-context, target+context+1):
            matches = [row for row in by_frame.get(frame_id, [])
                       if (row['track_id'], row['segment_id']) == (anchor['track_id'], anchor['segment_id'])]
            if len(matches) != 1:
                raise ValueError(f'Frame {frame_id}: missing or duplicate box for track {anchor["track_id"]}, '
                                 f'segment {anchor["segment_id"]}; cannot complete the review window')
            boxes[frame_id] = matches[0]['bbox']
        windows[target] = (anchor['track_id'], boxes)
    return windows


def _person_crop(frame, bbox, padding):
    """Add a fraction of box width/height on each side, clipped to the source."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(v) for v in bbox) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'Invalid person box {bbox} for a {width}x{height} frame')
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    left, top = max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy))
    right, bottom = min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy))
    return frame[top:bottom, left:right]


def export_review_sheets(video, targets, out, context=3, *, trajectories=None, padding=0.10, track_ids=None):
    """Export one 300-DPI PNG per target, following its padded person box.

    Frame IDs are zero-based, matching staff_trajectories.csv. Decode in source
    order instead of seeking independently, so adjacent panels really are adjacent
    frames. A window crossing the video boundary is rejected, never padded with
    repeated frames or silently shortened. Missing boxes also fail explicitly;
    review does not rerun detection or guess a different person's box.
    """
    targets = list(dict.fromkeys(targets))
    if not targets or any(not isinstance(f, int) or f < 0 for f in targets):
        raise ValueError('Provide at least one nonnegative integer target frame')
    if not isinstance(context, int) or context < 0:
        raise ValueError('context must be a nonnegative integer')
    if not math.isfinite(padding) or padding < 0:
        raise ValueError('padding must be a finite nonnegative fraction')
    out = Path(out)
    trajectories = Path(trajectories) if trajectories is not None else out.parent/'staff_trajectories.csv'
    if not trajectories.is_file():
        raise FileNotFoundError(f'Missing {trajectories}. Run the staff pipeline first or supply a trajectories CSV.')
    windows = _review_boxes(trajectories, targets, context, track_ids)

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

    # Prepare every requested crop before writing any sheets.
    crops = {target: [_person_crop(frames[f], bbox, padding) for f, bbox in boxes.items()]
             for target, (_, boxes) in windows.items()}
    out.mkdir(parents=True, exist_ok=True)
    count = 2*context+1
    columns = min(4, count)
    row_counts = [min(columns, count-offset) for offset in range(0, count, columns)]
    # Every row fills the sheet width, including an incomplete last row.
    # Use each crop's aspect ratio and a shared height per row. This fills the
    # row without distorting people or adding empty cells/black bars.
    dpi = 300
    margin, gap = 0.10, 0.10
    header_height, label_height = 0.70, 0.32
    panel_width = 4.0
    figure_width = columns*panel_width+(columns-1)*gap+2*margin
    paths = []
    for target in targets:
        start, end = target-context, target+context
        track_id, _ = windows[target]
        panels = crops[target]
        row_layout = []
        offset = 0
        for n in row_counts:
            aspects = [crop.shape[1]/crop.shape[0] for crop in panels[offset:offset+n]]
            image_height = (figure_width-2*margin-(n-1)*gap)/sum(aspects)
            row_layout.append((image_height, [image_height*aspect for aspect in aspects]))
            offset += n
        figure_height = (header_height+sum(height for height, _ in row_layout)+len(row_counts)*label_height
                         +(len(row_counts)-1)*gap+margin)
        fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
        try:
            fig.patch.set_facecolor('white')
            fig.suptitle(f'Frame-by-frame review | Target frame {target}', fontsize=22, fontweight='bold',
                         y=1-0.08/figure_height)
            fig.text(0.5, 1-0.46/figure_height,
                     f'Frames {start}-{end} inclusive | {count} consecutive frames | '
                     f'Track {track_id} | Person box + {padding:.0%} per side | Zero-based frame IDs',
                     ha='center', va='top', fontsize=12, color='#444444')
            axes = []
            row_top = figure_height-header_height
            for image_height, image_widths in row_layout:
                image_bottom = row_top-label_height-image_height
                image_left = margin
                for image_width in image_widths:
                    axes.append(fig.add_axes([
                        image_left/figure_width,
                        image_bottom/figure_height,
                        image_width/figure_width,
                        image_height/figure_height,
                    ]))
                    image_left += image_width+gap
                row_top = image_bottom-gap
            for ax, frame_id, crop in zip(axes, range(start, end+1), panels):
                ax.imshow(crop, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                is_target = frame_id == target
                offset = 'TARGET' if is_target else f'{frame_id-target:+d}'
                color = '#bd5600' if is_target else '#333333'
                ax.set_title(f'Frame {frame_id} | {frame_id/meta["fps"]:.2f}s | {offset}',
                             fontsize=11, fontweight='bold' if is_target else 'normal', color=color, pad=6)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('#e87900' if is_target else '#bbbbbb')
                    spine.set_linewidth(3.5 if is_target else 0.6)
            path = out/f'review_frame_{target:06d}.png'
            fig.savefig(path, dpi=dpi, facecolor='white', transparent=False,
                        metadata={'Title': f'Frame {target}: consecutive frames {start}-{end}',
                                  'Description': f'{count} person crops from consecutive frames; track {track_id}; '
                                                 f'{padding:.0%} padding per side, clipped to source edges; '
                                                 'zero-based numbering; target outlined in orange.'})
            paths.append(path)
            pixel_width, pixel_height = fig.canvas.get_width_height()
            print(f'Saved {path} | frames {start}-{end} | {count} panels | '
                  f'{pixel_width} x {pixel_height} pixels', flush=True)
        finally:
            plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default='data/raw/sample.mp4')
    parser.add_argument('--frames', type=int, nargs='+', required=True, help='Target frame IDs, zero-based')
    parser.add_argument('--context', type=int, default=3, help='Number of consecutive frames on each side (default: 3)')
    parser.add_argument('--out', default='data/output/refined/review')
    parser.add_argument('--trajectories', help='Staff CSV (default: staff_trajectories.csv in the parent of --out)')
    parser.add_argument('--padding', type=float, default=0.10, help='Extra box width/height on each side (default: 0.10)')
    parser.add_argument('--track-ids', type=int, nargs='+', help='Optional track ID per target, in the same order as --frames')
    args = parser.parse_args()
    export_review_sheets(args.video, args.frames, args.out, args.context,
                         trajectories=args.trajectories, padding=args.padding, track_ids=args.track_ids)


if __name__ == '__main__':
    main()
