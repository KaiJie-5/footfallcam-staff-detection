# Analysis of the original pipeline and saved results

This review inspected the repository's Python code, README and dependencies, the supplied company PDF and its three illustrations, the source video, and the saved outputs at `W:/footfallcam-staff-detection/data/output`. The two provided copies of `sample.mp4` were identical at inspection. Changes are in the local GitHub checkout; the W: results were read, not overwritten.

## Start from the task

The task defines staff by a **specific visible name tag**, illustrated as three dark bars on a light rectangle. It asks for presence frames and optionally xy coordinates. Corridor walking describes the supplied scene; it is not the definition of staff. The supplied snapshots show the marker against both dark and light clothing. Hardcoding shirt color, an upright torso, one pixel corridor, or a particular tracker ID would not implement the task on another clip.

Three different questions must be answered:

1. **Observation:** is a person visible, and where is their image box?
2. **Identity evidence:** does the marker pattern appear on that person?
3. **Temporal association:** which nearby observations belong to that same person while the marker is hidden?

Person-detector confidence answers only the first. A tracking ID is an association hypothesis, not a staff identity. Motion and a white patch do not answer the second. Temporal smoothing cannot repair a systematically wrong identity cue.

## What the existing outputs actually establish

The clip has 1,341 frames, 960 x 720 pixels, 25 FPS, and 53.64 seconds. The old output reports 915 staff-positive frames and 941 coordinate rows. These are prediction counts, not an accuracy measurement.

| Old track | Observations labelled staff | Badge-positive observations | Positive fraction | First/last observed frame |
| --- | ---: | ---: | ---: | --- |
| 6 | 856 | 6 | 0.70% | 5 / 879 |
| 217 | 37 | 3 | 8.11% | 828 / 865 |
| 290 | 48 | 10 | 20.83% | 1077 / 1144 |

Track 6 qualified because `(tag_ratio >= 0.08) OR (tag_hits >= 5)` lets five isolated hits override poor support over any track length. Its six hits are at frames 382, 531, and 873–876. All 856 observations then inherit the label. The diagnostic crops show background/clothing contamination in the supposed badge observations. Other crops show the actual patterned marker, so the saved detections contain a mixture of useful and invalid evidence. No supplied file provides exhaustive ground truth; a defensible precision/recall number cannot be derived from these outputs alone.

## Main defects and implemented responses

| Defect | Why it matters | Change |
| --- | --- | --- |
| Bright rectangular contour treated as verified badge | Can accept chairs, desk edges, clothing, or highlights without checking the printed marker | Match the actual supplied marker patterns across scales and rotations; expose raw correlation scores and match boxes |
| Upright-only torso crop | Overhead people rotate and can face either image direction | Search the person crop without assuming a top-facing torso |
| Inference confidence 0.30 before ByteTrack | Removes the low-score observations needed for recovery | Retain detections down to 0.10; use higher thresholds for new tracks |
| Upright general-purpose person detector only | Fisheye views include inverted bodies and unusual geometry | Optional orthogonal views, fused before a single tracker; default 0 and 180 degrees |
| Entire-track label from a few hits | Correlated false positives and ID switches contaminate long spans | Repeated evidence in a local time window, spatial consistency, track splits, and bounded propagation |
| Hardcoded pixel corridor and 90-pixel displacement | Excludes legitimate stationary staff, different resolutions, and out-and-back trajectories | Whole-frame default; optional normalized ROI only filters output |
| Motion interpreted as staff activity | Lighting and sensor failures can dominate global frame difference | Separate exposure quality from adjacent-frame activity; no identity claim from EDA |
| Stored NumPy crop views for every detection | A small crop can retain its full source frame, causing memory growth with video pixels | Store boxes and scalar evidence; reread source frames for visualization |
| FPS fallback and metadata-only loops | Can silently mis-time or omit decoded observations | Validate FPS and count actual decoded frames; reject incomplete caches |
| No useful audit of rejected tracks | Difficult to diagnose false matches and missing staff | Emit all frame decisions, track evidence, and review sheets |
| No tests, no labelled evaluation | Larger models and nicer video overlays can look like accuracy improvements | Add regression tests and an evaluator for explicit frame/point labels |

The low-score recovery change follows the [ByteTrack paper](https://arxiv.org/abs/2110.06864) and the [Ultralytics tracker configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml). Rotation is a relevant concern in overhead imagery, as demonstrated by [RAPiD](https://arxiv.org/abs/2005.11623). The added orthogonal inference views are a practical engineering baseline, not a replacement for a trained fisheye detector or measured evidence of improved recall on this clip.

The old contrast calculation also included the proposed badge inside its background window. Its score was a hand-weighted brightness expression labelled as confidence, with no probability calibration. The replacement explicitly reports correlation rather than a probability or a forensic verification claim.

The old EDA selected frame 900 and frames 1300–1304 as its main activity intervals. Saved luminance drops from about 107 to 8.18 at frame 1300, then recovers through 34.21, 43.87, and 46.29 before returning to 96.0. The old `<15` blackout rule only covers the darkest part. The new quality flag compares against recent valid illumination and advances the tracker on invalid frames; identity evidence does not cross these boundaries.

## Why this implementation, and what it does not solve

The supplied material includes a marker and reference appearances, but no labelled badge training set. Reference matching is therefore an immediately implementable baseline with inspectable failure cases. It checks the stated identity cue and needs no invented training labels. A synthetic three-bar template alone did not describe the supplied blurred views well enough, so the runtime uses the actual task illustrations. Those illustrations are development exemplars; they are not an independent test set.

Defaults require two strong marker samples (correlation at least 0.90) and three supporting samples (at least 0.84) within 1.2 seconds. Strong matches must lie in a consistent relative part of the person box. Accepted evidence can propagate at most three seconds forward/backward over observed, continuous track segments. These thresholds are initial engineering settings, not calibrated optima. The propagation limit prevents one badge observation from classifying a 35-second trajectory, but it may omit long periods where the real marker is hidden.

Detection runs every frame. Marker matching is sampled every 0.2 seconds to reduce CPU cost. Neighboring samples remain correlated; repeated hits are a consistency gate, not independent statistical trials. A persistent background pattern can still pass. Loose person boxes, small tags, blur, perspective shear, overlapping people, and moderate ID switches remain important failure modes. Template width search and large-crop normalization do not provide complete scale or projective invariance.

No tracking gaps are filled with invented coordinates, and separate IDs are not merged solely because their clothes look similar. These choices favor inspectability and precision but leave recall limitations. The generic person model can miss staff before marker recognition is even attempted. An exposure shift that stays far below recent illumination may be conservatively marked unobservable for an extended period.

For the company's stated few-hundred-video setting, the more scalable approach is a dedicated badge detector trained with rotated, perspective-distorted positives and hard negatives from person crops, combined with an overhead-person detector and a tracker evaluated for ID switches. Split by video/camera or person traversal, not random neighboring frames. Compare marker localization precision/recall, frame presence precision/recall, entry/exit error, coordinate error with missed detections counted, and processing speed on the intended laptop. OCR is not the first choice for the supplied small bar pattern. Camera calibration would be needed for meaningful floor/3D coordinates; the MP4 alone supplies image pixels.

## Validation status and next comparison

Before the user requested code-only delivery, a person-only diagnostic pass using the existing YOLOv8x weights and 0/180-degree views processed all 1,341 frames in approximately 178 seconds on the local GPU. A marker scan was started and then stopped at the user's request. This does not establish final pipeline accuracy. The final revised code, test suite, and complete new outputs have **not been executed or validated** after the subsequent edits. No improved frame-count or accuracy claim is made.

Run the commands in the README. Inspect the marker evidence sheet and annotated video, label positive and negative frames independently of model scores, and evaluate both the old W: outputs and the new output directory using the same labels. Separate label-development clips from the final holdout. Compare single-view and rotated-view runs before selecting a deployment configuration. Content hashes have been removed as requested; cache reuse relies on explicit user choice and basic video metadata.
