# Third-party notices

This repository contains or references third-party model artefacts. Their own
licenses and usage conditions continue to apply.

## SafetyVision YOLOv8s v2

- Provider: Hugging Face
- Repository: <https://huggingface.co/ayushgupta7777/safetyvision-yolov8>
- Pinned commit: `56a71758b55f0e9f2b4b2d6b51a779a1f882da10`
- Source checkpoint: `v2/best.pt`
- Checkpoint SHA-256:
  `7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a`
- Declared license: AGPL-3.0
- License text: <https://www.gnu.org/licenses/agpl-3.0.txt>

The local manifest and upstream model card are preserved at:

- `data/manifests/ppe-safetyvision-yolov8s-v2-challenger-r1.json`
- `data/manifests/ppe-safetyvision-yolov8s-v2-checkpoint-quarantine-r2.json`
- `data/raw/ppe/models/safetyvision-yolov8-v2-56a7175/README.md`

## Ultralytics YOLO11s person detector

The person lane uses Ultralytics YOLO11s exported to ONNX. The canonical source
checkpoint used in the originating workspace had SHA-256
`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
The delivered exports are pinned below:

- 640 ONNX: `ad0bb6e414adacde5d4aea652197670d2265a68b4b69cde61fd99d40d7306814`
- 640 external data: `c0034f112991337967c913470adf1f12aa5fcb4c7f1bbd3a2851cb8be9b26f28`
- 960 ONNX: `271dafd5234f19c2487daf2a75cee97f940ae1cba695e70d1cb179d328cf5f9c`
- 960 external data: `0945ff9843953c75889dc45e017748e06dc3bb1892e7c168a62beaf221830a39`

Confirm the Ultralytics licensing terms that apply to the team's intended
distribution and deployment before commercial release.

## NVIDIA runtime

DeepStream, CUDA, TensorRT and NVIDIA Container Toolkit are NVIDIA products and
are governed by NVIDIA's respective license terms. The target machine obtains
the NVIDIA base image directly from NGC and builds the project layer locally.

## Sample media

Sample videos are intentionally excluded from the Git history. The frozen
handoff Release contains the test media used for this internal transfer. Do not
redistribute that media publicly without checking each source's rights and
provenance.
