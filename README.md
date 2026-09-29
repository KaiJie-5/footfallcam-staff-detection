# FootfallCam Staff Identification & Tracking

This repository contains the end-to-end solution for the FootfallCam AI Evaluation test:
1. **Task 1**: Identify frames where staff (wearing a company name tag) is present in the 3D overhead sensor view.
2. **Task 2 [Bonus]**: Locate staff $(x, y)$ coordinates across detected frames.

---

## 1. Environment Setup

### Conda Environment
Create and activate an isolated Python 3.10 environment:

```bash
conda create -n footfallcam_env python=3.10 -y
conda activate footfallcam_env
```

### PyTorch Installation

- **For CUDA 12.1 / 12.4 / 12.8:**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Verify Environment & GPU Access
Run this to verify PyTorch and CUDA availability:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

---

## 2. Directory Layout Setup

```bash
mkdir -p data/raw data/output src
```

Copy your video into the raw data folder:
```bash
cp /path/to/sample.mp4 data/raw/sample.mp4
```

---

## 3. Video Characteristics & Exploratory Analysis (EDA)

Before running detection models, run the video analyzer to extract metadata, motion signals, and image quality metrics:

```bash
python src/video_analyzer.py --video data/raw/sample.mp4 --out data/output --sample-rate 1
```

### What this tool extracts:
- **Video Metadata**: Codec, container, native FPS, duration, resolution, and total frame count.
- **Motion Energy Curve**: Inter-frame pixel difference across time (detects exactly when people walk along the corridor).
- **Luminance & Contrast**: Detects lighting variations, glare, and shadow issues.
- **Sharpness Profile (Laplacian Variance)**: Evaluates motion blur to ensure staff tag readability.
- **Frame Contact Sheet**: Automatically generates sampled keyframe grids and identifies high-activity intervals.