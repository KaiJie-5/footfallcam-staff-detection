"""
Outputs:
- staff_frames.json (Task 1 answer)
- staff_trajectories.csv (Task 2 bonus coordinates)
- staff_presence_summary.json (Executive metrics)
- annotated_output.mp4 (Full visualization with PiP badge crop)
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
    parser = argparse.ArgumentParser(description="FootfallCam Staff Detection & Tracking Pipeline")
    parser.add_argument("--video", type=str, default="data/raw/sample.mp4", help="Path to input sample.mp4")
    parser.add_argument("--out", type=str, default="data/output", help="Directory to save generated outputs")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO model weights (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Person detection confidence threshold")
    parser.add_argument("--min-tag-ratio", type=float, default=0.10, help="Min ratio of tag-positive frames to classify track as staff")
    parser.add_argument("--min-displacement", type=float, default=80.0, help="Min pixel displacement to filter out seated workers")
    parser.add_argument("--save-video", action="store_true", default=True, help="Render annotated output MP4")
    return parser.parse_args()


def run_pipeline():
    args = parse_arguments()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing Pipeline on Device: {device.upper()}")

    tracker = OverheadPersonTracker(model_weights=args.weights, device=device, conf_thresh=args.conf)
    corridor_filter = CorridorZoneFilter()
    tag_detector = StaffTagDetector()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video loaded: {width}x{height} @ {fps:.1f} FPS ({total_frames} total frames)")

    # Video Writer
    video_writer = None
    annotated_video_path = os.path.join(args.out, "annotated_output.mp4")
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    # Pass 1 Storage: collect frame observations per track ID
    # track_id -> { "centers": [(cx, cy, frame_idx)], "tag_hits": int, "total_seen": int, "frames": [int], ... }
    track_records = {}
    frame_detections_history = {}  # frame_idx -> list of person detection dicts

    pbar = tqdm(total=total_frames, desc="Pass 1: Tracking & Tag Inspection", unit="frame")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Avoid processing blackout sensor glitch frames (Frame 1300-1304)
        mean_lum = np.mean(frame)
        if mean_lum < 15.0:
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

            # Clamp coordinates to frame boundary
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

            # Accumulate track statistics
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

    # Pass 2: Temporal Consensus & Staff Track Identification
    print("\n[INFO] Evaluating Track-Level Consensus...")
    staff_track_ids = set()

    for tid, data in track_records.items():
        frames_seen = len(data["frames"])
        if frames_seen < 10:
            continue  # Transient noise

        # Calculate total spatial displacement across life of track
        first_pt = np.array(data["centers"][0][:2])
        last_pt = np.array(data["centers"][-1][:2])
        displacement = float(np.linalg.norm(last_pt - first_pt))

        corridor_ratio = data["corridor_frames"] / float(frames_seen)
        tag_ratio = data["tag_hits"] / float(frames_seen)

        # A track is a Staff Member if:
        # 1. Subject actively traverses corridor (displacement > threshold & inside corridor)
        # 2. Tag detector confirmed badge presence across temporal instances
        is_moving_in_corridor = (displacement >= args.min_displacement) and (corridor_ratio >= 0.25)
        has_verified_badge = (tag_ratio >= args.min_tag_ratio) or (data["tag_hits"] >= 5)

        if is_moving_in_corridor and has_verified_badge:
            staff_track_ids.add(tid)
            print(f"  --> Identified Staff Track [ID {tid}]: {frames_seen} frames, "
                  f"Displacement: {displacement:.1f}px, Tag Hits: {data['tag_hits']} ({tag_ratio*100:.1f}%)")

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

    # Save Task 1 Output
    task1_json_path = os.path.join(args.out, "staff_frames.json")
    with open(task1_json_path, "w") as f:
        json.dump({
            "total_staff_frames": len(sorted_staff_frames),
            "frame_ranges": compress_frame_ranges(sorted_staff_frames),
            "frames": sorted_staff_frames
        }, f, indent=2)

    # Save Task 2 Output
    df_traj = pd.DataFrame(staff_trajectory_rows)
    task2_csv_path = os.path.join(args.out, "staff_trajectories.csv")
    df_traj.to_csv(task2_csv_path, index=False)

    print(f"\n[TASK 1 COMPLETE] Saved {len(sorted_staff_frames)} staff presence frames to: {task1_json_path}")
    print(f"[TASK 2 COMPLETE] Saved staff (X, Y) trajectories to: {task2_csv_path}")

    # Pass 4: Render Annotated Video
    if args.save_video and video_writer is not None:
        print("[INFO] Rendering Annotated Demonstration Video...")
        cap = cv2.VideoCapture(args.video)
        f_idx = 0
        pbar_v = tqdm(total=total_frames, desc="Rendering Annotated MP4", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw corridor polygon
            corridor_filter.draw_zone(frame, color=(255, 180, 0), thickness=2)

            dets = frame_detections_history.get(f_idx, [])
            staff_in_this_frame = False
            active_badge_crop = None

            for d in dets:
                tid = d["track_id"]
                x1, y1, x2, y2 = d["bbox"]
                cx, cy = d["center"]

                if tid in staff_track_ids:
                    staff_in_this_frame = True
                    # Green bounding box for Staff
                    color = (0, 255, 0)
                    label = f"STAFF ID {tid} (Tag Verified)"
                    
                    # Highlight badge bounding box if detected
                    if d["tag_rect"] is not None:
                        bx, by, bw, bh = d["tag_rect"]
                        cv2.rectangle(frame, (x1 + bx, y1 + by), (x1 + bx + bw, y1 + by + bh), (0, 0, 255), 2)
                        active_badge_crop = d["crop"]

                    # Draw coordinate crosshair
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                    coord_str = f"({cx}, {cy})"
                    cv2.putText(frame, coord_str, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    # Blue bounding box for non-staff corridor pedestrians or grey for seated
                    color = (255, 100, 0) if d["in_corridor"] else (120, 120, 120)
                    label = f"Person {tid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # HUD Status Overlay
            hud_bg = (30, 30, 30)
            cv2.rectangle(frame, (10, 10), (360, 90), hud_bg, -1)
            cv2.rectangle(frame, (10, 10), (360, 90), (100, 100, 100), 1)

            status_text = "STAFF DETECTED" if staff_in_this_frame else "NO STAFF PRESENT"
            status_color = (0, 255, 0) if staff_in_this_frame else (0, 165, 255)
            
            cv2.putText(frame, f"FootfallCam 3D Sensor Evaluation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(frame, f"Frame: {f_idx:04d} ({f_idx/fps:.2f}s)", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Status: {status_text}", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Picture-in-Picture Badge Inspection window (top right)
            if active_badge_crop is not None and active_badge_crop.size > 0:
                pip_size = 140
                pip_crop = cv2.resize(active_badge_crop, (pip_size, pip_size))
                px1 = width - pip_size - 15
                py1 = 15
                frame[py1:py1+pip_size, px1:px1+pip_size] = pip_crop
                cv2.rectangle(frame, (px1, py1), (px1+pip_size, py1+pip_size), (0, 255, 0), 2)
                cv2.putText(frame, "TAG CROP", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            video_writer.write(frame)
            f_idx += 1
            pbar_v.update(1)

        pbar_v.close()
        cap.release()
        video_writer.release()
        print(f"[INFO] Annotated video exported to: {annotated_video_path}")

    # Pass 5: Output Summary JSON
    summary_report = {
        "task_1_staff_frame_count": len(sorted_staff_frames),
        "task_1_frame_ranges": compress_frame_ranges(sorted_staff_frames),
        "task_2_trajectory_points": len(staff_trajectory_rows),
        "identified_staff_track_ids": list(staff_track_ids),
        "sample_time_intervals": [
            f"{start/fps:.2f}s - {end/fps:.2f}s"
            for start, end in compress_frame_ranges(sorted_staff_frames)
        ]
    }
    summary_path = os.path.join(args.out, "staff_presence_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 60)
    print("        FOOTFALLCAM EVALUATION PIPELINE RESULTS")
    print("=" * 60)
    print(f"Task 1: Staff present in {len(sorted_staff_frames)} frames.")
    for rng in summary_report["sample_time_intervals"]:
        print(f"  -> Interval: {rng}")
    print(f"Task 2: Exported {len(staff_trajectory_rows)} spatial (X, Y) points.")
    print(f"Outputs written to: {os.path.abspath(args.out)}")
    print("=" * 60)


def compress_frame_ranges(frame_list):
    """Compresses a list of frames [10, 11, 12, 50, 51] into [[10, 12], [50, 51]]."""
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