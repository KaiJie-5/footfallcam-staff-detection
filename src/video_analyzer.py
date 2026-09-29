"""
FootfallCam AI Evaluation: Video Characteristics & EDA Analyzer
Analyzes MP4 files for metadata, lighting,
motion variance, and sharpness before building staff identification pipelines.
"""

import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="FootfallCam Video EDA & Characteristic Inspection")
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 file")
    parser.add_argument("--out", type=str, default="data/output", help="Directory to save analysis results")
    parser.add_argument("--sample-rate", type=int, default=1, help="Process every Nth frame (default=1 for full analysis)")
    parser.add_argument("--save-keyframes", action="store_true", default=True, help="Save a contact sheet of keyframes")
    return parser.parse_args()


def get_video_properties(cap):
    """Extract standard metadata from the OpenCV VideoCapture object."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps else 0

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    return {
        "width": width,
        "height": height,
        "aspect_ratio": f"{width}:{height}",
        "native_fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration_sec, 2),
        "duration_formatted": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}",
        "fourcc_codec": codec
    }


def analyze_video_stream(video_path, sample_rate=1):
    """
    Scans the video to compute per-frame metrics:
    - Mean Luminance (brightness)
    - Contrast (standard deviation of pixel intensities)
    - Sharpness (variance of the Laplacian)
    - Inter-frame Motion Difference (Mean Absolute Difference with previous frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    meta = get_video_properties(cap)
    print(f"\n[INFO] Loaded: {video_path}")
    print(f"[INFO] Resolution: {meta['width']}x{meta['height']} @ {meta['native_fps']} FPS")
    print(f"[INFO] Duration: {meta['duration_formatted']} ({meta['total_frames']} frames)")

    records = []
    prev_gray = None
    frame_idx = 0

    pbar = tqdm(total=meta['total_frames'], desc="Analyzing frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Luminance & Contrast
            mean_brightness = float(np.mean(gray))
            contrast = float(np.std(gray))

            # 2. Sharpness (Laplacian variance - sensitive to motion blur)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 3. Inter-frame difference (Motion proxy)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = float(np.mean(diff))
            else:
                motion_score = 0.0

            timestamp_sec = frame_idx / meta['native_fps']

            records.append({
                "frame_id": frame_idx,
                "timestamp_sec": round(timestamp_sec, 2),
                "brightness": round(mean_brightness, 2),
                "contrast": round(contrast, 2),
                "sharpness": round(sharpness, 2),
                "motion_score": round(motion_score, 2)
            })

            prev_gray = gray

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    df = pd.DataFrame(records)
    return meta, df


def generate_visual_reports(df, meta, out_dir):
    """Generates time-series inspection plots and a contact sheet."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Video EDA Report: {meta['width']}x{meta['height']} @ {meta['native_fps']} FPS", fontsize=14, fontweight="bold")

    # Motion Activity
    axes[0].plot(df["timestamp_sec"], df["motion_score"], color="crimson", linewidth=1.5)
    axes[0].set_ylabel("Motion Score\n(Frame Diff)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].set_title("Motion Activity (Peaks indicate people walking in the aisle)")

    # Sharpness / Blur
    axes[1].plot(df["timestamp_sec"], df["sharpness"], color="teal", linewidth=1.2)
    axes[1].set_ylabel("Sharpness\n(Laplacian Var)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_title("Frame Sharpness (Monitors motion blur)")

    # Brightness
    axes[2].plot(df["timestamp_sec"], df["brightness"], color="darkorange", linewidth=1.2)
    axes[2].set_ylabel("Luminance")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].set_title("Average Brightness Level")

    # Contrast
    axes[3].plot(df["timestamp_sec"], df["contrast"], color="purple", linewidth=1.2)
    axes[3].set_ylabel("Contrast (Std)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].set_title("Pixel Intensity Contrast")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "video_eda_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved metrics plot to: {plot_path}")


def save_keyframe_contact_sheet(video_path, out_dir, num_samples=16):
    """Creates a 4x4 grid of representative frames across the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame_rgb, f"Frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            frames.append(frame_rgb)
    cap.release()

    if not frames:
        return

    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(frames):
            ax.imshow(frames[i])
            ax.axis("off")
    plt.suptitle("Sample Keyframe Overview (Evenly Distributed Across Video)", fontsize=16)
    plt.tight_layout()
    sheet_path = os.path.join(out_dir, "keyframe_contact_sheet.jpg")
    plt.savefig(sheet_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved keyframe contact sheet to: {sheet_path}")


def detect_motion_intervals(df, threshold_factor=1.5):
    """Identifies time intervals where significant movement occurs (people walking)."""
    baseline = df["motion_score"].median()
    std = df["motion_score"].std()
    cutoff = baseline + threshold_factor * std

    active_frames = df[df["motion_score"] > cutoff]
    if active_frames.empty:
        return []

    # Group consecutive frames into active segments
    intervals = []
    start_frame = None
    last_frame = None

    for f in active_frames["frame_id"]:
        if start_frame is None:
            start_frame = f
            last_frame = f
        elif f - last_frame <= 5:  # tolerance gap of 5 frames
            last_frame = f
        else:
            intervals.append({"start_frame": int(start_frame), "end_frame": int(last_frame)})
            start_frame = f
            last_frame = f

    if start_frame is not None:
        intervals.append({"start_frame": int(start_frame), "end_frame": int(last_frame)})

    return intervals


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Analyze video
    meta, df = analyze_video_stream(args.video, sample_rate=args.sample_rate)

    # 2. Find high motion intervals (potential staff walking timestamps)
    active_intervals = detect_motion_intervals(df)

    # 3. Compile summary report
    summary = {
        "metadata": meta,
        "signal_statistics": {
            "mean_brightness": round(float(df["brightness"].mean()), 2),
            "std_brightness": round(float(df["brightness"].std()), 2),
            "mean_sharpness": round(float(df["sharpness"].mean()), 2),
            "mean_motion_energy": round(float(df["motion_score"].mean()), 2)
        },
        "motion_intervals": active_intervals
    }

    # 4. Save CSV & JSON
    csv_path = os.path.join(args.out, "frame_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved frame metrics to: {csv_path}")

    json_path = os.path.join(args.out, "video_eda_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved summary report to: {json_path}")

    # 5. Generate Visuals
    generate_visual_reports(df, meta, args.out)

    if args.save_keyframes:
        save_keyframe_contact_sheet(args.video, args.out)

    print("\n" + "=" * 50)
    print("           VIDEO EDA SUMMARY")
    print("=" * 50)
    print(f"Dimensions     : {meta['width']}x{meta['height']}")
    print(f"FPS            : {meta['native_fps']}")
    print(f"Total Frames   : {meta['total_frames']}")
    print(f"Total Duration : {meta['duration_formatted']} ({meta['duration_seconds']}s)")
    print(f"High-Motion Segments Detected: {len(active_intervals)}")
    for i, interval in enumerate(active_intervals):
        t_start = interval["start_frame"] / meta["native_fps"]
        t_end = interval["end_frame"] / meta["native_fps"]
        print(f"  Segment {i+1}: Frames [{interval['start_frame']} - {interval['end_frame']}] (~{t_start:.1f}s - {t_end:.1f}s)")
    print("=" * 50)


if __name__ == "__main__":
    main()