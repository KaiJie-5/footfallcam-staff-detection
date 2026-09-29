# FootfallCam staff identification

Identify frames containing the company's name-tag marker, then export staff image coordinates. The supplied badge contains **three dark bars on a light card**. A white patch, corridor location, or movement alone does not establish staff identity.

See [the analysis](docs/analysis.md) for the original-result audit, assumptions, tradeoffs, and remaining limitations. [The short solution note](docs/solution.md) describes the implementation for the evaluation task.

## Run

Python 3.10+ is recommended (the code also supports Python 3.9). Install an appropriate PyTorch/torchvision pair for your CPU or GPU first, then:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.pipeline_runner --video data/raw/sample.mp4 --weights yolov8x.pt --out data/output/improved
```

On the local Windows checkout, an isolated environment and your existing model copy are already available:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.pipeline_runner --video data/raw/sample.mp4 --weights yolov8x.pt --device 0 --out data/output/improved
```

Tests were added but **not run**, at the user's request. The final pipeline has not completed an end-to-end validation run. The earlier person-only diagnostic pass is not validation of the final staff output.

Weights must exist locally. The pipeline does not silently download a model. For a fresh machine, explicitly download your chosen Ultralytics detection model or copy your existing weights. A smaller person model can reduce inference time but needs a recall check on overhead people. The supplied three template PNGs are included; the PDF is not required at runtime.

Use `--device cpu` without CUDA, `--rotations 0` for one detector view, and `--no-save-video` to skip MP4 rendering. Defaults use source and 180-degree views, fused into source-image coordinates before tracking. These settings are a starting point, not a measured optimum.

## Outputs

| File | Contents |
| --- | --- |
| `staff_frames.json` | Sorted, zero-based staff frame IDs and inclusive ranges |
| `staff_trajectories.csv` | Observed box-center xy coordinates in original image pixels |
| `annotated_output.mp4` | Green confirmed staff, amber candidates, gray unknown people |
| `marker_evidence.jpg` | Best marker matches for reviewing confirmed and questionable segments |
| `track_audit.json` | Evidence counts, split segments, and confirmation anchors |
| `frame_decisions.jsonl` | All observations, scores, quality flags, and classification reasons |
| `staff_presence_summary.json` | Results, parameters, timings, and basic video metadata |

Coordinates use a top-left origin, with x rightward and y downward. They are bounding-box centers, not head centers, floor positions, or calibrated 3D coordinates. Right/bottom bbox edges are exclusive. Missing observations are not interpolated. Frame time is `frame_id / FPS`, assuming constant-frame-rate input.

`staff` means repeated marker evidence on a continuous local track. `candidate` means an unconfirmed pattern match. `unknown` means insufficient evidence; it does not prove that someone is not staff. The algorithm is offline: later evidence can support earlier observations within a configurable time limit.

## Cache expensive work

The normal command creates person and marker caches in the output directory. Reuse person detections while modifying marker settings:

```powershell
python -m src.pipeline_runner --video data/raw/sample.mp4 --people-cache data/output/improved/people.jsonl --out data/output/marker_revision
```

Reuse marker evidence while changing only temporal thresholds:

```powershell
python -m src.pipeline_runner --video data/raw/sample.mp4 --evidence-cache data/output/improved/evidence.jsonl --strong-score 0.90 --support-score 0.84 --propagation-seconds 3 --out data/output/threshold_revision
```

Caches have a JSONL file and matching `.meta.json` sidecar. Basic filename, size, dimensions, FPS, and frame-count checks catch common mismatches. **Rebuild when the video, weights, detector settings, code, or marker templates change.** There is no content hashing or automatic cache invalidation. Explicit cache reuse uses its recorded producer settings; detector flags do not rerun an existing cache. Interrupted `.partial` files are not valid inputs. Use separate output folders for comparisons.

`--sample-seconds 0.2` controls marker sampling, not person tracking: every source frame is tracked. Larger intervals reduce marker cost but can miss short badge views or prevent the default three-observation consensus within 1.2 seconds. For very short clips, use a smaller interval.

An optional `--roi path/to/polygon.json` filters exported box centers. The JSON is a list of normalized vertices, for example `[[0.2,0.1],[0.8,0.1],[0.8,0.9],[0.2,0.9]]`. No corridor is assumed by default; ROI membership never confirms identity.

## Inspect and evaluate

```powershell
python -m src.video_analyzer --video data/raw/sample.mp4 --out data/output/eda --sample-rate 1
```

For accuracy measurement, create a JSON list of explicit frame annotations. This is a **format example**, not ground truth for the sample:

```json
[
  {"frame_id": 10, "staff_present": false, "points": []},
  {"frame_id": 20, "staff_present": true, "points": [[500, 300]]},
  {"frame_id": 30, "staff_present": null}
]
```

Omit `points` when only presence is labelled. Set `staff_present` to `null` for uncertain frames. Point annotations must use the same bounding-box-center convention as predictions. Evaluate with:

```powershell
python -m src.evaluate --results data/output/improved --annotations data/annotations.json --out data/output/improved/evaluation.json --match-radius 40
```

The evaluator reports frame precision/recall/F1, plus matched localization errors, missed locations, and false locations on explicitly labelled frames. Unlabelled frames are not negatives. Reserve separate clips or traversals for final evaluation; nearby frames are highly correlated.

## Reference assets

The templates are small crops of the company's PDF illustrations, not manually selected video detections. `assets/marker_references.json` records their locations. To recreate them from the provided PDF:

```powershell
python scripts/extract_references.py --pdf "C:/Users/liang/Downloads/AI Evaluation Test.pdf"
```

Replace these references or train a badge detector when the marker changes. The supplied task images are development references; matching them is not evidence of accuracy on unseen footage.
