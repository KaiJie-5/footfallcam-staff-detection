"""
FootfallCam Staff Identification & Tracking Pipeline Runner (High-Precision Edition)
Solves:
- Task 1: Identify which frames in the clip have the staff present.
- Task 2 [Bonus]: Locate the staff (x, y) coordinates when present.
"""

import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from detector import OverheadPersonTracker, CorridorZoneFilter, StaffTagDetector


def parse_arguments():
    parser = argparse.ArgumentParser(description="FootfallCam High-Precision Staff Detection Pipeline")
    parser.add_argument("--video", type=str, default="data/raw/sample.mp4", help="Path to input sample.mp4")
    parser.add_argument("--out", type=str, default="data/output", help="Directory to save generated outputs")
    # Upgraded default model: yolov8x.pt (Extra-Large)
    parser.add_argument("--weights", type=str, default="yolov8x.pt", help="YOLO model weights (e.g. yolov8x.pt, yolo11x.pt, yolov8l.pt)")
    # Upgraded default inference resolution: 960 (Native video width)
    parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution (default: 960 native)")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    # Narrow corridor preset: 'tight' by default
    parser.add_argument("--corridor-preset", type=str, default="tight", choices=["tight", "ultratight", "standard"], help="Corridor width preset")
    parser.add_argument("--min-tag-ratio", type=float, default=0.08, help="Min ratio of tag-positive frames to classify track as staff")
    parser.add_argument("--min-displacement", type=float, default=90.0, help="Min pixel displacement to filter out stationary workers")
    parser.add_argument("--save-video", action="store_true", default=True, help="Render annotated demonstration MP4")
    return parser.parse_args()


def run_pipeline():
    args = parse_arguments()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n========================================================")
    print(f"  FOOTFALLCAM HIGH-PRECISION STAFF TRACKING PIPELINE")
    print(f"========================================================")
    print(f"Model Architecture : {args.weights}")
    print(f"Inference Res      : {args.imgsz}x{args.imgsz}")
    print(f"Compute Device     : {device.upper()}")
    print(f"Corridor Zone      : {args.corridor_preset.upper()} (Narrowed Aisle)")
    print(f"========================================================\n")

    tracker = OverheadPersonTracker(
        model_weights=args.weights,
        device=device,
        conf_thresh=args.conf,
        imgsz=args.imgsz
    )
    corridor_filter = CorridorZoneFilter(preset=args.corridor_preset)
    tag_detector = StaffTagDetector()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    annotated_video_path = os.path.join(args.out, "annotated_output.mp4")
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    track_records = {}
    frame_detections_history = {}

    pbar = tqdm(total=total_frames, desc="Tracking & Tag Verification", unit="frame")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Automatically skip blackout sensor glitches (Frame 1300-1304)
        if np.mean(frame) < 15.0:
            frame_detections_history[frame_idx] = []
            frame_idx += 1
            pbar.update(1)
            continue

        raw_detections = tracker.track_frame(frame)
        frame_detections_history[frame_idx] = []

        for det in raw_detections:
            tid = det["track_id"]
            if tid == -1:
                continue

            cx, cy = det["center"]
            x1, y1, x2, y2 = det["bbox"]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            in_corridor = corridor_filter.is_inside(cx, cy)
            person_crop = frame[y1:y2, x1:x2]

            has_tag, tag_conf, tag_rect = tag_detector.inspect_crop(person_crop)

            record_item = {
                "track_id": tid,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "in_corridor": in_corridor,
                "has_tag": has_tag,
                "tag_conf": tag_conf,
                "tag_rect": tag_rect,
                "crop": person_crop
            }
            frame_detections_history[frame_idx].append(record_item)

            if tid not in track_records:
                track_records[tid] = {
                    "centers": [],
                    "frames": [],
                    "corridor_frames": 0,
                    "tag_hits": 0,
                    "max_tag_conf": 0.0,
                    "sample_tag_crop": None
                }

            track_records[tid]["centers"].append((cx, cy, frame_idx))
            track_records[tid]["frames"].append(frame_idx)
            if in_corridor:
                track_records[tid]["corridor_frames"] += 1
            if has_tag:
                track_records[tid]["tag_hits"] += 1
                if tag_conf > track_records[tid]["max_tag_conf"]:
                    track_records[tid]["max_tag_conf"] = tag_conf
                    track_records[tid]["sample_tag_crop"] = person_crop

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Pass 2: Track Temporal Consensus
    print("\n[INFO] Evaluating Track-Level Consensus...")
    staff_track_ids = set()

    for tid, data in track_records.items():
        frames_seen = len(data["frames"])
        if frames_seen < 12:
            continue

        first_pt = np.array(data["centers"][0][:2])
        last_pt = np.array(data["centers"][-1][:2])
        displacement = float(np.linalg.norm(last_pt - first_pt))

        corridor_ratio = data["corridor_frames"] / float(frames_seen)
        tag_ratio = data["tag_hits"] / float(frames_seen)

        # Criteria: Traverses corridor + Verified Tag
        is_moving_in_corridor = (displacement >= args.min_displacement) and (corridor_ratio >= 0.25)
        has_verified_badge = (tag_ratio >= args.min_tag_ratio) or (data["tag_hits"] >= 5)

        if is_moving_in_corridor and has_verified_badge:
            staff_track_ids.add(tid)
            print(f"  --> Identified Staff Track [ID {tid}]: {frames_seen} frames, "
                  f"Displacement: {displacement:.1f}px, Corridor Ratio: {corridor_ratio*100:.1f}%, Tag Hits: {data['tag_hits']}")

    # Pass 3: Extract Task 1 & Task 2 Solutions
    staff_presence_frames = set()
    staff_trajectory_rows = []

    for f_idx in range(total_frames):
        dets = frame_detections_history.get(f_idx, [])
        for d in dets:
            tid = d["track_id"]
            if tid in staff_track_ids:
                staff_presence_frames.add(f_idx)
                cx, cy = d["center"]
                x1, y1, x2, y2 = d["bbox"]
                timestamp = round(f_idx / fps, 3)

                staff_trajectory_rows.append({
                    "frame_id": f_idx,
                    "timestamp_sec": timestamp,
                    "track_id": tid,
                    "x": cx,
                    "y": cy,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "tag_detected": d["has_tag"],
                    "tag_conf": round(d["tag_conf"], 3)
                })

    sorted_staff_frames = sorted(list(staff_presence_frames))

    # Save Task 1
    task1_json_path = os.path.join(args.out, "staff_frames.json")
    with open(task1_json_path, "w") as f:
        json.dump({
            "total_staff_frames": len(sorted_staff_frames),
            "frame_ranges": compress_frame_ranges(sorted_staff_frames),
            "frames": sorted_staff_frames
        }, f, indent=2)

    # Save Task 2
    df_traj = pd.DataFrame(staff_trajectory_rows)
    task2_csv_path = os.path.join(args.out, "staff_trajectories.csv")
    df_traj.to_csv(task2_csv_path, index=False)

    print(f"\n[TASK 1 COMPLETE] Saved {len(sorted_staff_frames)} staff presence frames to: {task1_json_path}")
    print(f"[TASK 2 COMPLETE] Saved staff (X, Y) trajectories to: {task2_csv_path}")

    # Pass 4: Render Demonstration Video
    if args.save_video and video_writer is not None:
        print("[INFO] Rendering High-Definition Annotated Video...")
        cap = cv2.VideoCapture(args.video)
        f_idx = 0
        pbar_v = tqdm(total=total_frames, desc="Rendering Annotated MP4", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw the narrowed corridor polygon
            corridor_filter.draw_zone(frame, color=(0, 220, 255), thickness=2)

            dets = frame_detections_history.get(f_idx, [])
            staff_in_this_frame = False
            active_badge_crop = None

            for d in dets:
                tid = d["track_id"]
                x1, y1, x2, y2 = d["bbox"]
                cx, cy = d["center"]

                if tid in staff_track_ids:
                    staff_in_this_frame = True
                    color = (0, 255, 0)
                    label = f"STAFF ID {tid} (Tag Verified)"
                    
                    if d["tag_rect"] is not None:
                        bx, by, bw, bh = d["tag_rect"]
                        cv2.rectangle(frame, (x1 + bx, y1 + by), (x1 + bx + bw, y1 + by + bh), (0, 0, 255), 2)
                        active_badge_crop = d["crop"]

                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                    cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    color = (255, 120, 0) if d["in_corridor"] else (90, 90, 90)
                    label = f"Person {tid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            # HUD Status Display
            cv2.rectangle(frame, (10, 10), (370, 92), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, 10), (370, 92), (120, 120, 120), 1)

            status_text = "STAFF DETECTED" if staff_in_this_frame else "NO STAFF IN CORRIDOR"
            status_color = (0, 255, 0) if staff_in_this_frame else (0, 165, 255)
            
            cv2.putText(frame, f"FootfallCam 3D Sensor ({args.weights})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
            cv2.putText(frame, f"Frame: {f_idx:04d} ({f_idx/fps:.2f}s)", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            cv2.putText(frame, f"Status: {status_text}", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            # Picture-in-Picture Badge Inspection window
            if active_badge_crop is not None and active_badge_crop.size > 0:
                pip_size = 140
                pip_crop = cv2.resize(active_badge_crop, (pip_size, pip_size))
                px1 = width - pip_size - 15
                py1 = 15
                frame[py1:py1+pip_size, px1:px1+pip_size] = pip_crop
                cv2.rectangle(frame, (px1, py1), (px1+pip_size, py1+pip_size), (0, 255, 0), 2)
                cv2.putText(frame, "NAME TAG CROP", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            video_writer.write(frame)
            f_idx += 1
            pbar_v.update(1)

        pbar_v.close()
        cap.release()
        video_writer.release()
        print(f"[INFO] Annotated video exported to: {annotated_video_path}")

    # Pass 5: Output Summary Report
    summary_report = {
        "model_architecture": args.weights,
        "inference_resolution": f"{args.imgsz}x{args.imgsz}",
        "corridor_preset": args.corridor_preset,
        "task_1_staff_frame_count": len(sorted_staff_frames),
        "task_1_frame_ranges": compress_frame_ranges(sorted_staff_frames),
        "task_2_trajectory_points": len(staff_trajectory_rows),
        "identified_staff_track_ids": list(staff_track_ids),
        "traversal_time_intervals": [
            f"{start/fps:.2f}s - {end/fps:.2f}s (Frames {start}-{end})"
            for start, end in compress_frame_ranges(sorted_staff_frames)
        ]
    }
    summary_path = os.path.join(args.out, "staff_presence_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 65)
    print("        FOOTFALLCAM HIGH-PRECISION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Task 1: Staff present in {len(sorted_staff_frames)} frames.")
    for rng in summary_report["traversal_time_intervals"]:
        print(f"  -> Interval: {rng}")
    print(f"Task 2: Exported {len(staff_trajectory_rows)} spatial (X, Y) points.")
    print(f"Output files saved to: {os.path.abspath(args.out)}")
    print("=" * 65)


def compress_frame_ranges(frame_list):
    """Compresses [10, 11, 12, 50, 51] into [[10, 12], [50, 51]]."""
    if not frame_list:
        return []
    ranges = []
    start = frame_list[0]
    prev = frame_list[0]
    for f in frame_list[1:]:
        if f == prev + 1:
            prev = f
        else:
            ranges.append([start, prev])
            start = f
            prev = f
    ranges.append([start, prev])
    return ranges


if __name__ == "__main__":
    run_pipeline()