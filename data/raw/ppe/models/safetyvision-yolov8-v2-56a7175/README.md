---
license: agpl-3.0
language:
- en
library_name: ultralytics
pipeline_tag: object-detection
tags:
- yolov8
- yolov8s
- object-detection
- ppe-detection
- workplace-safety
- computer-vision
- onnx
- albumentations
- safety
- osha
---

# SafetyVision YOLOv8 — PPE Detection (v1 nano · v2 small)

YOLOv8 fine-tuned for Personal Protective Equipment (PPE) detection at industrial worksites. Backbone model for [SafetyVision](https://github.com/ayushgupta07xx/SafetyVision), an open-source AI workplace safety monitor.

This repo hosts **two versions**:
- **v2 (current, production)** — YOLOv8s, trained on 80k images with Albumentations augmentation. Weights at `v2/`.
- **v1 (original)** — YOLOv8n, trained on 58k images. Weights at the repo root, kept for reproducibility and the v1→v2 comparison.

| Headline metric (v2, held-out test) | Value |
|---|---|
| **Test mAP@0.5 (imgsz 896)** | **0.766** |
| Test mAP@0.5 (imgsz 640) | 0.754 |
| Deployed ONNX mAP@0.5 (imgsz 640) | 0.738 |
| Test mAP@0.5:0.95 (imgsz 896) | 0.487 |
| Validation mAP@0.5 | 0.787 |
| Parameters | 11,130,615 (~11.1M) |
| FLOPs | 28.5 GFLOPs |

> **Honest note on the target.** The Phase-2 goal was mAP@0.5 ≥ 0.78 on the held-out **test** split. Validation cleared it (0.787); the held-out test came in at **0.766** (imgsz 896) — short of 0.78 by 0.014. We report the test number as the headline generalization figure rather than leading with the higher validation value. See [Evaluation](#evaluation).

## What's new in v2 (v1 → v2)

| Aspect | v1 (YOLOv8n) | v2 (YOLOv8s) |
|---|---|---|
| Backbone | nano | small |
| Parameters | ~3.0M | ~11.1M |
| Training images | 57,904 (1 dataset) | 80,304 (5 datasets merged + MD5 dedup) |
| Augmentation | ultralytics defaults | + Albumentations (CoarseDropout, MotionBlur, RandomGamma, CLAHE) + perspective |
| Epochs | 100 | 150 (cosine LR) |
| Train image size | 640 | 896 |
| Hardware | Kaggle 2× T4 (16GB) | GCP L4 (24GB), single 61.25 hr run |
| Test mAP@0.5 | 0.701 | **0.766** (896) / 0.754 (640) |
| Test mAP@0.5:0.95 | 0.441 | **0.487** (896) / 0.485 (640) |
| Deployed weights | `best.onnx` (640) | `v2/best_640.onnx` + `v2/best_896.onnx` |

Test-vs-test improvement: **+6.5 mAP@0.5 / +4.6 mAP@0.5:0.95** at imgsz 896 (+5.3 mAP@0.5 at 640). Two failure-mode classes improved dramatically — see [Failure modes](#failure-modes).

## Model description

13-class PPE detection covering hard hats, safety vests, goggles, gloves, masks, their "missing/no" violation counterparts, fall detection, fall-harness absence, and a Person class.

- **Base model:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) — `yolov8s.pt` for v2, `yolov8n.pt` for v1
- **Output:** 17 channels × N anchors (8400 at 640, 16464 at 896) → NMS → boxes + class labels + confidence
- **Use it for:** flagging likely PPE violations in static images and short video clips for human review
- **Do not use it for:** automated disciplinary action, medical/clinical PPE, food safety, hazmat suits, or any standalone enforcement decision

## Classes

| ID | Class | Type |
|----|---|---|
| 0 | Fall-Detected | Event |
| 1 | Gloves | PPE worn ✓ |
| 2 | Goggles | PPE worn ✓ |
| 3 | Hardhat | PPE worn ✓ |
| 4 | Mask | PPE worn ✓ |
| 5 | NO-Gloves | Violation ✗ |
| 6 | NO-Goggles | Violation ✗ |
| 7 | NO-Hardhat | Violation ✗ |
| 8 | NO-Mask | Violation ✗ |
| 9 | NO-Safety Vest | Violation ✗ |
| 10 | No_Harness | Violation ✗ |
| 11 | Person | Person detection |
| 12 | Safety Vest | PPE worn ✓ |

## Training data

### v2 (current)

Five Roboflow Universe datasets merged into one corpus, deduplicated by MD5 hash and remapped to the 13 canonical classes:

- `ppe-combined-9bprl-mmcaf` (v1)
- `hardhat-safetyvest` (v1)
- `fall-detection-ca3o8` (v4)
- `safety_ppe` (v1)
- `construction-safety-gears-vcbdq` (v1)

| Split | Images |
|---|---|
| Train | 68,253 |
| Validation | 8,025 |
| Test (held-out) | 4,026 |
| **Total (post-dedup)** | **80,304** |

Stratified 85/10/5 split (~6.4 GB). Dataset selection deliberately favored side/back/occluded poses, low-light and high-glare scenes, and non-frontal workers to address v1's frontal bias.

### v1 (original)

[PPE-Combined v1](https://universe.roboflow.com/mazz-maxx/ppe-combined-9bprl-mmcaf) — 57,904 images (41,922 train / 10,834 val / 5,148 test).

## Training procedure

### v2 (current)

- **Hardware:** GCP L4 24GB (`g2-standard-8`, `asia-southeast1-c`)
- **Framework:** Ultralytics 8.4.51, PyTorch 2.12.0 + CUDA 13.0
- **Epochs:** 150 · **Batch:** 24 · **Image size:** 896 · **LR schedule:** cosine
- **Augmentation:** Albumentations (CoarseDropout, MotionBlur, RandomGamma, CLAHE — non-spatial) + native perspective, mosaic, mixup, HSV jitter
- **multi_scale:** False (see ADR-012 — `multi_scale=True` OOMs at batch=24 on a 24GB L4 at peak image size; the marginal benefit isn't worth halving the batch / ~95 hr wall time for a fixed-resolution deployment)
- **Class balancing:** none applied — augmentation alone hit target; the planned class-weighted loss was not needed (NO-Mask remained trainable at recall 0.789)
- **Wall time:** 61.25 hours, single uninterrupted run (no session cap, no resume), ~24 min/epoch, GPU memory 10–21 GB

### v1 (original)

- Kaggle Notebooks, 2× Tesla T4 · Ultralytics 8.3.40 · 100 epochs · batch 32 · imgsz 640 · SGD
- ~15 hr across two Kaggle Save Versions (12-hr cap forced a resume at epoch 82)

### Experiment tracking

- **W&B run (v2, public):** https://wandb.ai/agcr7jw-vellore-institute-of-technology/Ultralytics/runs/yolov8s-ppe-v2_20260519_065053
  - Logged under the `Ultralytics` project (the ultralytics W&B callback hardcodes the project name and ignores `WANDB_PROJECT`).
- **MLflow (v2):** local file store committed at [`mlruns/`](https://github.com/ayushgupta07xx/SafetyVision/tree/main/mlruns), experiment `621501274199551492`, run `0af3bb3c50b84db3ac376d7e63e558d8`.
- The v1 W&B run (`9nctv2ai`) has expired; v1 canonical metrics live in `model/yolov8n-ppe-v1/results.csv` in the repo.

## Evaluation

Honest numbers, no cherry-picking. The held-out test split (4,026 images, never seen during training/validation) is the canonical generalization measure.

### v2 headline (held-out test, 4,026 images, 12,080 instances)

| Measurement | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|
| `.pt` @ imgsz 896 (model ceiling) | **0.766** | 0.487 | 0.731 | 0.757 |
| `.pt` @ imgsz 640 | 0.754 | 0.485 | 0.724 | 0.736 |
| **ONNX @ imgsz 640 (deployed, Lambda)** | **0.738** | 0.463 | 0.723 | 0.715 |
| Validation @ imgsz 896 | 0.787 | 0.504 | 0.755 | 0.778 |

The ~0.016 ONNX-vs-`.pt` gap at 640 is fp32 numerical drift through onnxslim/opset-20 (precision is unchanged, recall dips slightly at the detection threshold), not a broken export. The 640 ONNX ships on AWS Lambda (CPU budget); the 896 ONNX ships on Hugging Face Spaces (16GB RAM) for the full 0.766 ceiling.

### v2 per-class test metrics (imgsz 896)

| Class | Instances | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Fall-Detected | 765 | 0.886 | 0.937 | **0.959** | 0.704 |
| Hardhat | 5,589 | 0.888 | 0.912 | **0.937** | 0.608 |
| Goggles | 256 | 0.857 | 0.887 | 0.919 | 0.545 |
| Safety Vest | 1,015 | 0.816 | 0.831 | 0.892 | 0.648 |
| Person | 1,038 | 0.870 | 0.798 | 0.861 | 0.584 |
| No_Harness | 256 | 0.728 | 0.773 | 0.830 | 0.533 |
| Gloves | 669 | 0.810 | 0.677 | 0.786 | 0.423 |
| NO-Hardhat | 865 | 0.687 | 0.788 | 0.754 | 0.474 |
| NO-Gloves | 713 | 0.771 | 0.685 | 0.751 | 0.400 |
| NO-Goggles | 439 | 0.765 | 0.608 | 0.711 | 0.387 |
| NO-Mask | 115 | 0.559 | 0.694 | 0.598 | 0.430 |
| Mask | 143 | 0.387 | 0.825 | 0.575 | 0.376 |
| **NO-Safety Vest** | 217 | 0.478 | 0.431 | **0.386** | 0.224 |
| **all** | 12,080 | 0.731 | 0.757 | **0.766** | 0.487 |

Confusion matrices and PR curves (640 and 896) are committed in [`docs/assets/eval/v2/`](https://github.com/ayushgupta07xx/SafetyVision/tree/main/docs/assets/eval/v2).

### v1 (for reference)

YOLOv8n test mAP@0.5 = **0.701**, mAP@0.5:0.95 = 0.441. Full v1 per-class metrics and curves in `model/yolov8n-ppe-v1/`.

## Inference performance

- **v2 GPU (L4), per image:** ~7.5 ms inference @ 896, ~3.5 ms @ 640 (plus ~1 ms pre/post)
- **v2 ONNX CPU (AWS Lambda, 3008 MB):** ~500–800 ms warm inference @ 640. Cold start adds ~5–8 s container init plus a one-time ~10 s S3 fetch of the ONNX weights into `/tmp` (cached for subsequent warm invocations on the same container).
- **v2 ONNX CPU (HF Spaces, 16 GB):** sub-second warm detection @ 896; visible end-to-end latency on the public Space is dominated by the explainability + Gemini-multimodal report stages downstream of YOLO, not the forward pass itself.
- **ONNX files:** `best_640.onnx` 42.7 MB, `best_896.onnx` 42.8 MB (fp32, opset 20, onnxslim 0.1.94, no external-data sidecar)

## Explainability

Per-violation results carry two attribution signals alongside the bounding boxes: a GradCAM heatmap and a SHAP pixel attribution. Both are surfaced in three places — the web UI tabs on the Upload result page, the API response (`gradcam_b64` and `shap_chart_b64`), and embedded side-by-side in the downloadable PDF incident report alongside the annotated detection image, the OSHA citation, and the Gemini-generated incident narrative.

- **GradCAM heatmap reliability** — most informative on single-subject close-up scenes with one dominant detected object. In diffuse multi-person scenes, wide shots, or scenes where the violation target is small (<50 px), the heatmap can render flat or uninformative — a known limitation of class-activation-map techniques on dense scenes with small targets. The SPPF backbone layer (`model.model[9]`) is the only consistently usable GradCAM target identified during development on this model; targeting earlier or later layers produced noisier results in testing. Kept in the pipeline because the cases where it works clearly are exactly the ones that warrant visual confirmation; in diffuse scenes the SHAP attribution alongside it provides a useful complementary signal.
- **SHAP attribution** — `shap.GradientExplainer` against the YOLO classification head at 320×320 (host machine needs ≥11 GB RAM for the backward pass — relevant on WSL where the default 7.6 GB is insufficient). Rendered as a per-pixel attribution chart in the web UI and PDF report.

## Intended use

Pre-screening tool to **assist** human workplace safety officers by surfacing likely PPE violations in images and short video clips for human review. Designed for construction sites, warehouses, manufacturing floors, and pre-shift safety walkthroughs.

**Not a replacement for human judgment.** Predictions must be reviewed by qualified safety personnel before any disciplinary, compliance, or insurance action.

## Out of scope

- Medical/clinical settings (gowns, N95 fit testing, sterile gloves)
- Food processing (hairnets, beard guards, lab coats)
- Chemical/hazmat operations (full-face respirators, encapsulating suits)
- Drone or overhead camera angles (training data is ground/eye level)
- Crowded scenes with heavy mutual occlusion
- Real-time alerting where missing a single violation is unacceptable

## Failure modes

Documented from training-data review and observed v2 test errors:

- **NO-Safety Vest is the weakest class** (test mAP@0.5 0.386, only 217 instances). High false-negative rate — do not rely on it as the sole vest-compliance signal.
- **Mask / NO-Mask are weak** (0.58 / 0.60). One source dataset (`construction-safety-gears`) mixes COVID-style face-mask close-ups into the industrial-mask class, adding domain noise. Mask precision in particular suffers (0.39).
- **Low light / high glare** — confidence drops; expect both false positives and false negatives.
- **Partial occlusion** — workers behind machinery/other workers may have PPE missed (improved vs v1 but not solved).
- **Small workers (<50 px height)** — distant figures often missed.
- **Fast motion in video** — motion blur causes missed frames; aggregate across frames rather than trusting any single frame.
- **Rare PPE colors** — training skews to high-vis vests and standard hard-hat colors.

**Improved in v2 (previously failure modes):**
- **No_Harness** was effectively unusable in v1 (1 test instance, mAP 0.000). v2 adds fall-detection data → 256 test instances at mAP@0.5 **0.83**. Now a usable signal, though still validate before relying on it for fall-arrest compliance.
- **Frontal bias / Person detection** — v1 Person precision was 0.37; v2 reaches **0.87** (P) with mAP 0.86 on 7× more test instances, reflecting the deliberate inclusion of side/back/occluded poses in the v2 dataset.

## Bias and limitations

- Training data over-represents Western construction/industrial sites; PPE conventions in South/Southeast Asia, Africa, and the Middle East may be underrepresented.
- Heavily skewed toward male-presenting workers.
- The Person class inherits biases from YOLOv8 COCO pretraining.
- Indoor warehouse lighting overrepresented; bright outdoor sun and underground/tunnel environments may degrade performance.
- 13 classes is a fixed taxonomy — site-specific PPE (arc-flash hoods, cut-resistant sleeves) is not detected.

## Files

### v2 (`v2/`)

| File | Size | Description |
|---|---|---|
| `v2/best.pt` | ~22.5 MB | PyTorch weights — `ultralytics.YOLO("best.pt")` |
| `v2/last.pt` | ~22.5 MB | Final-epoch checkpoint |
| `v2/best_640.onnx` | ~42.7 MB | ONNX (imgsz 640) — AWS Lambda deployment |
| `v2/best_896.onnx` | ~42.8 MB | ONNX (imgsz 896) — HF Spaces deployment |

### v1 (repo root)

`best.pt`, `best.onnx`, `best.onnx.data` (v1 ONNX uses an external-data sidecar that must be co-located).

## Usage

### PyTorch (ultralytics)

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

weights = hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="v2/best.pt")
model = YOLO(weights)
results = model("worksite_image.jpg")
results[0].show()
```

### ONNX Runtime (CPU-friendly, used in AWS Lambda)

```python
import cv2, numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

onnx_path = hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="v2/best_640.onnx")
session = ort.InferenceSession(onnx_path)

img = cv2.imread("worksite_image.jpg")
img = cv2.resize(img, (640, 640))           # letterbox in production; see core/detector.py
inp = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
outputs = session.run(None, {"images": inp})
# outputs[0] shape: (1, 17, 8400) — apply your own NMS for final boxes
```

For the full 0.766 ceiling on a higher-RAM host, swap `v2/best_640.onnx` → `v2/best_896.onnx` and resize to 896.

## License

- **Model weights:** AGPL-3.0 (inherited from Ultralytics YOLOv8 base model)
- **SafetyVision repository code:** see [LICENSE](https://github.com/ayushgupta07xx/SafetyVision/blob/main/LICENSE)

## Citation

```bibtex
@software{safetyvision_2026,
  author = {Gupta, Ayush},
  title  = {SafetyVision: Open-Source AI Workplace Safety Monitor},
  year   = {2026},
  url    = {https://github.com/ayushgupta07xx/SafetyVision}
}
```

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 and the training framework
- [Roboflow Universe](https://universe.roboflow.com) and the PPE dataset maintainers
- [OSHA](https://www.osha.gov) for the public-domain regulation corpus
- [Kaggle Notebooks](https://www.kaggle.com/code) (v1 training) and Google Cloud L4 (v2 training)