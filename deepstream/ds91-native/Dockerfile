FROM nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994 AS native-build

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# This stage deliberately consumes only local named build contexts.  It has no
# package-manager or remote-source step and is built with network=none.
RUN test "$(dpkg --print-architecture)" = "amd64" \
    && test "$(readlink -f /opt/nvidia/deepstream/deepstream)" = "/opt/nvidia/deepstream/deepstream-9.1" \
    && grep -Fxq "Version: 9.1.0" /opt/nvidia/deepstream/deepstream/version \
    && test "${CUDA_VERSION}" = "13.2.0.046" \
    && test "$(readlink -f /usr/local/cuda)" = "/usr/local/cuda-13.2" \
    && test "$(dpkg-query -W -f='${Version}' libnvinfer10)" = "10.16.0.72-1+cuda13.2" \
    && test "$(pkg-config --modversion gstreamer-1.0)" = "1.24.2" \
    && grep -Eq '^#define NVDS_VERSION_MAJOR 9$' /opt/nvidia/deepstream/deepstream/sources/includes/nvds_version.h \
    && grep -Eq '^#define NVDS_VERSION_MINOR 1$' /opt/nvidia/deepstream/deepstream/sources/includes/nvds_version.h \
    && test -f /opt/nvidia/deepstream/deepstream/sources/includes/gstnvdsinfer.h \
    && test -s /opt/nvidia/deepstream/deepstream/lib/libnvds_meta.so \
    && test -s /opt/nvidia/deepstream/deepstream/lib/libnvdsgst_meta.so

COPY --from=person / /build/src/models/person/postprocess/rtdetrv4_ds9/
COPY --from=pose / /build/src/models/pose/postprocess/
COPY --from=ppe / /build/src/models/ppe/postprocess/
COPY --from=fusion / /build/src/deepstream/fusion/
COPY --from=fusion_runner / /build/src/deepstream/fusion-r2/
COPY --from=patches / /build/src/deepstream/patches/
COPY --from=upstream_app / /build/src/upstream-app/
COPY native-build-contract-v1.json /build/policy/native-build-contract-v1.json

RUN mkdir -p /build/audit /build/out \
    && cd /build/src \
    && find . -type f -print0 \
       | LC_ALL=C sort -z \
       | xargs -0 sha256sum > /build/audit/input-source-files.sha256 \
    && python3 -m json.tool /build/policy/native-build-contract-v1.json >/dev/null

# RT-DETRv4 person parser: all tests are parser-only CPU tests.  The CUDA path
# is used for headers only; this component contains no CUDA kernel launch.
RUN cmake \
      -S /build/src/models/person/postprocess/rtdetrv4_ds9 \
      -B /build/person-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DDEEPSTREAM_INCLUDE_DIR=/opt/nvidia/deepstream/deepstream/sources/includes \
      -DCUDA_INCLUDE_DIR=/usr/local/cuda-13.2/include \
    && cmake --build /build/person-build --parallel 2 \
    && ctest --test-dir /build/person-build --output-on-failure \
    && /build/person-build/test_rtdetrv4_parser \
    && readelf --dyn-syms --wide /build/person-build/libdeepsafe_rtdetrv4_parser.so \
       | grep -Fq 'NvDsInferParseCustomRTDETRv4Person@@DEEPSAFE_RTDETRV4_PARSER_1.0' \
    && readelf --dyn-syms --wide /build/person-build/libdeepsafe_rtdetrv4_parser.so \
       | grep -Fq 'NvDsInferInitializeInputLayers@@DEEPSAFE_RTDETRV4_PARSER_1.0' \
    && test "$(ldd -r /build/person-build/libdeepsafe_rtdetrv4_parser.so 2>&1 \
       | grep -c 'undefined symbol' || true)" = "0" \
    && install -m 0555 /build/person-build/libdeepsafe_rtdetrv4_parser.so \
       /build/out/libdeepsafe_rtdetrv4_parser.so

# The R2 builder compiles and tests the pose and PPE postprocess cores, the
# DeepStream metadata fusion adapter, and a locally patched NVIDIA parallel
# inference application.  The application is linked but never executed here.
RUN bash /build/src/deepstream/fusion-r2/build-runtime.sh \
    && install -m 0444 \
       /build/fusion-build/pose-core/libdeepsafe_pose_postprocess.a \
       /build/out/libdeepsafe_pose_postprocess.a \
    && install -m 0444 \
       /build/fusion-build/pose-core/libdeepsafe_pose_deepstream_adapter.a \
       /build/out/libdeepsafe_pose_deepstream_adapter.a \
    && install -m 0444 \
       /build/fusion-build/ppe-core/libdeepsafe_ppe_postprocess.a \
       /build/out/libdeepsafe_ppe_postprocess.a \
    && test -x /build/out/deepstream-parallel-infer \
    && test -x /build/out/libdeepsafe_fusion.so.1 \
    && test -s /build/out/fusion-runtime.conf \
    && readelf --dynamic /build/out/deepstream-parallel-infer \
       | grep -Fq 'Shared library: [libdeepsafe_fusion.so.1]' \
    && readelf --dyn-syms --wide /build/out/libdeepsafe_fusion.so.1 \
       | grep -Fq 'deepsafe_fusion_process_gst_buffer_v1@@DEEPSAFE_FUSION_1.0' \
    && cd /build/out \
    && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
       | LC_ALL=C sort -z \
       | xargs -0 sha256sum > SHA256SUMS \
    && sha256sum --check SHA256SUMS

FROM scratch AS native-artifacts
COPY --from=native-build /build/out/ /
COPY --from=native-build /build/audit/ /audit/
COPY --from=native-build /build/policy/native-build-contract-v1.json /audit/native-build-contract-v1.json

FROM nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994 AS runtime

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

COPY --from=native-build /build/out/ /opt/deepsafe/ds91-native/
COPY --from=native-build /build/audit/ /opt/deepsafe/ds91-native/audit/
COPY --from=native-build /build/policy/native-build-contract-v1.json /opt/deepsafe/ds91-native/audit/native-build-contract-v1.json

RUN cd /opt/deepsafe/ds91-native \
    && sha256sum --check SHA256SUMS \
    && test -x deepstream-parallel-infer \
    && test -x libdeepsafe_fusion.so.1 \
    && test -x libdeepsafe_rtdetrv4_parser.so \
    && test -s libdeepsafe_pose_postprocess.a \
    && test -s libdeepsafe_pose_deepstream_adapter.a \
    && test -s libdeepsafe_ppe_postprocess.a \
    && python3 -m json.tool audit/native-build-contract-v1.json >/dev/null

LABEL com.deepsafe.native-build.schema="deepsafe.ds91-native-build/v1" \
      com.deepsafe.deepstream.base-ref="nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994" \
      com.deepsafe.deepstream.base-digest="sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994" \
      com.deepsafe.deepstream.version="9.1.0" \
      com.deepsafe.cuda.version="13.2.0.046" \
      com.deepsafe.tensorrt.version="10.16.0.72" \
      com.deepsafe.gstreamer.version="1.24.2" \
      com.deepsafe.build.platform="linux/amd64" \
      com.deepsafe.build.network="none" \
      com.deepsafe.build.gpu-used="false"

ENV DEEPSAFE_DS91_NATIVE_ROOT=/opt/deepsafe/ds91-native
ENV PATH=/opt/deepsafe/ds91-native:${PATH}

WORKDIR /opt/deepsafe/ds91-native
