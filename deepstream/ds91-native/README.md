# DeepStream 9.1 native build path (R1)

This directory is an additive, versioned build path for the native DeepSafe
components. It does not modify the frozen DeepStream 9.0 `deepstream/Dockerfile`
or the repository-root `.dockerignore`.

The build uses the exact cached amd64 image
`nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`.
That image contains DeepStream 9.1.0, CUDA 13.2.0.046, TensorRT 10.16.0.72,
and GStreamer 1.24.2. Every project source enters through a local named build
context, and every `RUN` layer is isolated with BuildKit `network=none`.

## Components

- RT-DETRv4 person custom parser and its CPU parser contract tests
- pose decoder, association core, and DeepStream tensor adapter
- PPE association and semantic postprocess core
- DeepStream metadata fusion shared library
- NVIDIA parallel-infer sample patched with the R2 fusion hook

The pose and PPE static libraries are also retained as build evidence. They are
linked into `libdeepsafe_fusion.so.1` for runtime use.

## Verify and build

Static policy verification does not start a container:

```bash
python3 deepstream/ds91-native/verify_static.py
python3 deepstream/ds91-native/test_static.py
```

The native build is CPU-only. It neither requests a GPU device nor runs
`deepstream-parallel-infer`; only unit/contract tests and ELF checks run:

```bash
bash deepstream/ds91-native/build.sh
```

The default output tag is `deepsafe-deepstream:9.1-native-r1`. Override it with
`DEEPSAFE_DS91_NATIVE_TAG`. The runtime artifacts and their `SHA256SUMS` live at
`/opt/deepsafe/ds91-native` in the resulting image.

The successful CPU-only R1 build is recorded in `build-receipt-r1.json`,
including the immutable image ID and all seven artifact hashes.

The build demonstrates compilation, CPU tests, DS9.1 header/ABI linkage, and
artifact integrity. It intentionally does not claim GPU engine generation,
TensorRT inference, pipeline execution, FPS, accuracy, or production readiness.
