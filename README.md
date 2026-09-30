# FootfallCam Staff Identification

Finds the frames of an overhead video in which a staff member wearing the company name-tag (three dark bars on a light card) appears, and exports each staff member's image coordinates.

The pipeline detects and tracks every person, matches the name-tag inside each person's bounding box, and labels a person as staff only when the matches are confirmed along that person's track.

## 1. Set up the environment

Python 3.10+ is recommended. Create and activate a conda environment, install the PyTorch/torchvision build that matches your CPU or GPU (see [pytorch.org](https://pytorch.org/get-started/locally/)), then install the remaining requirements:

```bash
conda create -n footfallcam python=3.10 -y
conda activate footfallcam
python -m pip install -r requirements.txt
```

Place the YOLOv8x weights (`yolov8x.pt`) in the project root and the video at `data/raw/sample.mp4`. The code does not download weights automatically.

## 2. Run the pipeline

```bash
python -m src.pipeline_runner --video data/raw/sample.mp4 --weights yolov8x.pt --out data/output/improved
```

Add `--device 0` for a GPU (use `--device cpu` without CUDA) and `--no-save-video` to skip the annotated video.

| Output | Contents |
| --- | --- |
| `staff_frames.json` | Zero-based staff frame IDs and inclusive frame ranges |
| `staff_trajectories.csv` | Bounding-box centre `(x, y)` in original image pixels, top-left origin |
| `annotated_output.mp4` | Green: staff, amber: candidate, grey: unknown |
| `marker_evidence.jpg` | Best name-tag matches, for a quick visual check |
| `staff_presence_summary.json` | Results, parameters and timings |

Add `--debug` to also write `track_audit.json` and `frame_decisions.jsonl`.

## 3. Review specific frames

```bash
python -m src.pipeline_runner --video data/raw/sample.mp4 --out data/output/improved --review-frames 506 860 --review-track-ids 128 207
```

The sheets are saved in `data/output/improved/review/`. Each shows the target frame with three frames before and after it, cropped to the same person.
