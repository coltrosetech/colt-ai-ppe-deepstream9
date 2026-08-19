# v2026.08.19-handoff

Initial public technical handoff for the COLT AI – COLLBRAI PPE pipeline.

## Release assets

- `COLT-AI-PPE-RUNTIME-GITHUB-20260819.tar.zst`: portable source and model
  archive; contains no videos, private media or TensorRT `.engine` files.
- `COLT-AI-PPE-TEKNIK-DEVIR-GITHUB-20260819.zip`: standalone Turkish installation
  and handoff documentation.
- `COLT-AI-PPE-GITHUB-RELEASE-20260819.sha256`: SHA-256 verification file for both
  assets.

## Verified state

- DeepStream 9.0 / CUDA 13.1 / TensorRT 10.14.1.48
- PPE profiles: 640 and 960, FP16, dynamic batch 1–12
- Reference GPU: NVIDIA RTX A5000 Laptop 16 GiB, compute capability 8.6
- Runtime archive integrity and ZIP integrity verified before upload
- Portable archive smoke suite: 131 passed, 1 skipped

## Important

TensorRT engines are intentionally not portable and are not distributed. Let
the runner build fresh engines from the checked ONNX files on the target GPU.
Smoke media is intentionally not distributed; supply a separately authorized
PPE video on the target machine.
