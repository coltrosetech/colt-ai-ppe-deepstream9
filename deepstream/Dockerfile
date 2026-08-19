ARG DEEPSTREAM_BASE_REF
FROM ${DEEPSTREAM_BASE_REF} AS parser-build

ARG DEEPSTREAM_BASE_REF
ARG DEEPSTREAM_BASE_DIGEST
ARG DEEPSTREAM_YOLO_COMMIT=2894babce8e75c49115dbe0c7b516289ed853565
ARG DEEPSTREAM_YOLO_TREE=1740cc4bc7e925f30e4eea0160064bfde729f8d8
ARG DEEPSTREAM_YOLO_PATCH_SHA256=dd85619bf62da249d17d99e967caf53de96367a0f139286c52a86f5e67b7623e
ARG DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256=a63299206550f1f8dd413cb9328352db304cdc020a47451170fd4c7eda0adf4d
ARG DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256=642e2875d67c3528c7ea301bcd1e973ea31fb019865f2aa116122583e9a765e3
ARG DEEPSTREAM_YOLO_PATCHED_TREE=753acbd2995f9f8c0b791f6152f0793baa11b71a
RUN test "$DEEPSTREAM_BASE_REF" = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch@${DEEPSTREAM_BASE_DIGEST}" \
    && printf '%s' "$DEEPSTREAM_BASE_DIGEST" | grep -Eq '^sha256:[0-9a-f]{64}$' \
    && test "$DEEPSTREAM_BASE_DIGEST" != "sha256:0000000000000000000000000000000000000000000000000000000000000000" \
    && test "$DEEPSTREAM_BASE_DIGEST" != "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
COPY deepstream/patches/deepstream-yolo-ds9-cuda-kernel-proof.patch /tmp/deepsafe-yolo-kernel-proof.patch
RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo.git /opt/deepsafe/DeepStream-Yolo \
    && cd /opt/deepsafe/DeepStream-Yolo \
    && git checkout "$DEEPSTREAM_YOLO_COMMIT" \
    && test "$(git rev-parse HEAD)" = "$DEEPSTREAM_YOLO_COMMIT" \
    && test "$(git rev-parse 'HEAD^{tree}')" = "$DEEPSTREAM_YOLO_TREE" \
    && test -z "$(git status --porcelain --untracked-files=no)" \
    && test "$(sha256sum nvdsinfer_custom_impl_Yolo/Makefile | cut -d' ' -f1)" = "fd2c03b810b8dae9d9d3a60b503616bbf6ed67a6f614843dd6a29f7f87ff8ad0" \
    && test "$(sha256sum nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu | cut -d' ' -f1)" = "$DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256" \
    && test "$(sha256sum /tmp/deepsafe-yolo-kernel-proof.patch | cut -d' ' -f1)" = "$DEEPSTREAM_YOLO_PATCH_SHA256" \
    && git apply --check /tmp/deepsafe-yolo-kernel-proof.patch \
    && git apply /tmp/deepsafe-yolo-kernel-proof.patch \
    && test "$(git diff --name-only)" = "nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu" \
    && test "$(sha256sum nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu | cut -d' ' -f1)" = "$DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256" \
    && git diff --check \
    && git add nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu \
    && test "$(git diff --cached --name-only)" = "nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu" \
    && test "$(git write-tree)" = "$DEEPSTREAM_YOLO_PATCHED_TREE" \
    && test -x /usr/bin/x86_64-linux-gnu-strip \
    && test "$(sha256sum /usr/bin/x86_64-linux-gnu-strip | cut -d' ' -f1)" = "4dad0d12aa5d6a49b117b4551b897175ad5b43b9525e8f9efd661133a1c8ea0d" \
    && test "$(/usr/bin/x86_64-linux-gnu-strip --version | sed -n '1p')" = "GNU strip (GNU Binutils for Ubuntu) 2.42" \
    && nice -n 10 make -C nvdsinfer_custom_impl_Yolo -j2 CUDA_VER=13.1 \
       'CUFLAGS=-I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-13.1/include -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86' \
    && test -s nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
    && /usr/bin/x86_64-linux-gnu-strip --strip-unneeded nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
    && test "$(readelf --section-headers --wide nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so | grep -Ec '[[:space:]]\.symtab[[:space:]]')" = "0" \
    && test "$(readelf --section-headers --wide nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so | grep -Ec '[[:space:]]\.strtab[[:space:]]')" = "0" \
    && test "$(readelf --section-headers --wide nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so | grep -Ec '[[:space:]]\.dynsym[[:space:]]')" = "1" \
    && test "$(readelf --section-headers --wide nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so | grep -Ec '[[:space:]]\.dynstr[[:space:]]')" = "1" \
    && mkdir -p /opt/deepsafe/parser-audit \
    && sha256sum nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
       | cut -d' ' -f1 > /opt/deepsafe/parser-audit/parser.sha256 \
    && printf '%s\n' \
        '{' \
        '  "schema_version": "deepsafe.ds9-parser-source-lineage/v3",' \
        '  "repository": "https://github.com/marcoslucianops/DeepStream-Yolo.git",' \
        "  \"commit\": \"${DEEPSTREAM_YOLO_COMMIT}\"," \
        "  \"tree\": \"${DEEPSTREAM_YOLO_TREE}\"," \
        '  "patched_source_path": "nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu",' \
        "  \"upstream_source_sha256\": \"${DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256}\"," \
        '  "patch_path": "deepstream/patches/deepstream-yolo-ds9-cuda-kernel-proof.patch",' \
        "  \"patch_sha256\": \"${DEEPSTREAM_YOLO_PATCH_SHA256}\"," \
        "  \"patched_source_sha256\": \"${DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256}\"," \
        "  \"patched_tree\": \"${DEEPSTREAM_YOLO_PATCHED_TREE}\"," \
        '  "build_makefile_path": "nvdsinfer_custom_impl_Yolo/Makefile",' \
        '  "build_makefile_sha256": "fd2c03b810b8dae9d9d3a60b503616bbf6ed67a6f614843dd6a29f7f87ff8ad0",' \
        '  "build_makefile_modified": false,' \
        '  "cuda_cubin_architecture": "sm_86",' \
        '  "cuda_ptx_architecture": "compute_86",' \
        '  "cuda_gencode_flags": ["-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86"],' \
        '  "instrumentation_schema": "deepsafe.ds9-cuda-kernel-proof/v1",' \
        "  \"build_command\": \"nice -n 10 make -C nvdsinfer_custom_impl_Yolo -j2 CUDA_VER=13.1 'CUFLAGS=-I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-13.1/include -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86'\"," \
        '  "build_command_sha256": "e244df2d9c424fe7d027d62205ff21c820b58eb2cc00aa61df2d32cbfe329ac1",' \
        '  "post_link_tool_path": "/usr/bin/x86_64-linux-gnu-strip",' \
        '  "post_link_tool_sha256": "4dad0d12aa5d6a49b117b4551b897175ad5b43b9525e8f9efd661133a1c8ea0d",' \
        '  "post_link_tool_version": "GNU strip (GNU Binutils for Ubuntu) 2.42",' \
        '  "post_link_command": "/usr/bin/x86_64-linux-gnu-strip --strip-unneeded nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so",' \
        '  "post_link_command_sha256": "5e19627d403e984d9e349f8c81332f7280a1ac8477d56056c9b27be63aeef7ca",' \
        '  "post_link_removed_sections": [".symtab", ".strtab"],' \
        '  "post_link_retained_sections": [".dynsym", ".dynstr"]' \
        '}' > /opt/deepsafe/parser-audit/source-lineage.json \
    && python3 -m json.tool /opt/deepsafe/parser-audit/source-lineage.json >/dev/null

# Pass 1 deliberately has no DEEPSTREAM_YOLO_PARSER_SHA256 input.  BuildKit's
# local exporter gives the auditor the real ELF, its measured digest and exact
# source lineage.  Only the resulting immutable receipt may feed pass 2.
FROM scratch AS parser-audit-export
COPY --from=parser-build /opt/deepsafe/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so /parser/libnvdsinfer_custom_impl_Yolo.so
COPY --from=parser-build /opt/deepsafe/parser-audit/parser.sha256 /parser/parser.sha256
COPY --from=parser-build /opt/deepsafe/parser-audit/source-lineage.json /parser/source-lineage.json

FROM parser-build AS runtime
ARG DEEPSTREAM_BASE_REF
ARG DEEPSTREAM_BASE_DIGEST
ARG DEEPSTREAM_YOLO_COMMIT=2894babce8e75c49115dbe0c7b516289ed853565
ARG DEEPSTREAM_YOLO_TREE=1740cc4bc7e925f30e4eea0160064bfde729f8d8
ARG DEEPSTREAM_YOLO_PATCH_SHA256=dd85619bf62da249d17d99e967caf53de96367a0f139286c52a86f5e67b7623e
ARG DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256=a63299206550f1f8dd413cb9328352db304cdc020a47451170fd4c7eda0adf4d
ARG DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256=642e2875d67c3528c7ea301bcd1e973ea31fb019865f2aa116122583e9a765e3
ARG DEEPSTREAM_YOLO_PATCHED_TREE=753acbd2995f9f8c0b791f6152f0793baa11b71a
ARG DEEPSTREAM_YOLO_PARSER_SHA256
ARG DEEPSAFE_RUNTIME_CONTROLLER_SHA256
ARG DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256
ARG DEEPSAFE_DOCKERIGNORE_SHA256
RUN test "$DEEPSTREAM_BASE_REF" = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch@${DEEPSTREAM_BASE_DIGEST}" \
    && printf '%s' "$DEEPSTREAM_BASE_DIGEST" | grep -Eq '^sha256:[0-9a-f]{64}$' \
    && printf '%s' "$DEEPSTREAM_YOLO_PARSER_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
    && printf '%s' "$DEEPSAFE_RUNTIME_CONTROLLER_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
    && printf '%s' "$DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
    && printf '%s' "$DEEPSAFE_DOCKERIGNORE_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
    && test "$(cat /opt/deepsafe/parser-audit/parser.sha256)" = "$DEEPSTREAM_YOLO_PARSER_SHA256" \
    && test "$(sha256sum /opt/deepsafe/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so | cut -d' ' -f1)" = "$DEEPSTREAM_YOLO_PARSER_SHA256" \
    && test "$(sha256sum /tmp/deepsafe-yolo-kernel-proof.patch | cut -d' ' -f1)" = "$DEEPSTREAM_YOLO_PATCH_SHA256" \
    && test "$(sha256sum /opt/deepsafe/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu | cut -d' ' -f1)" = "$DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256" \
    && test "$(cd /opt/deepsafe/DeepStream-Yolo && git write-tree)" = "$DEEPSTREAM_YOLO_PATCHED_TREE"
WORKDIR /app
COPY admin/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir PyYAML==6.0.3
COPY admin /app/admin
COPY deepstream /app/deepstream
COPY validation/__init__.py /app/validation/__init__.py
COPY validation/ds9_runtime_compatibility.py /app/validation/ds9_runtime_compatibility.py
COPY validation/ds9_parser_bootstrap.py /app/validation/ds9_parser_bootstrap.py
COPY validation/ds9_gpu_smoke.py /app/validation/ds9_gpu_smoke.py
COPY validation/open_video_manual_review.py /app/validation/open_video_manual_review.py
COPY validation/schemas/ds9-cuda-kernel-proof-v1.schema.json /app/validation/schemas/ds9-cuda-kernel-proof-v1.schema.json
COPY validation/schemas/site-distance-evaluation-plan-v1.schema.json /app/validation/schemas/site-distance-evaluation-plan-v1.schema.json
COPY validation/schemas/distance-validation-v1.schema.json /app/validation/schemas/distance-validation-v1.schema.json
COPY validation/schemas/validation-campaign-report-v1.schema.json /app/validation/schemas/validation-campaign-report-v1.schema.json
COPY deepstream/runtime-control-manifest.json /opt/deepsafe/runtime-control-manifest.json
COPY .dockerignore /opt/deepsafe/dockerignore.policy
RUN test "$(sha256sum /app/validation/ds9_runtime_compatibility.py | cut -d' ' -f1)" = "$DEEPSAFE_RUNTIME_CONTROLLER_SHA256" \
    && test "$(sha256sum /opt/deepsafe/runtime-control-manifest.json | cut -d' ' -f1)" = "$DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256" \
    && test "$(sha256sum /opt/deepsafe/dockerignore.policy | cut -d' ' -f1)" = "$DEEPSAFE_DOCKERIGNORE_SHA256" \
    && printf '%s\n' \
        '{' \
        '  "schema_version": "deepsafe.deepstream-build-lineage/v3",' \
        "  \"base_ref\": \"${DEEPSTREAM_BASE_REF}\"," \
        "  \"base_digest\": \"${DEEPSTREAM_BASE_DIGEST}\"," \
        '  "deepstream_version": "9.0.0",' \
        '  "cuda_version": "13.1",' \
        '  "tensorrt_version": "10.14.1.48",' \
        '  "gstreamer_version": "1.24.2",' \
        '  "deepstream_yolo_repository": "https://github.com/marcoslucianops/DeepStream-Yolo.git",' \
        "  \"deepstream_yolo_commit\": \"${DEEPSTREAM_YOLO_COMMIT}\"," \
        "  \"deepstream_yolo_tree\": \"${DEEPSTREAM_YOLO_TREE}\"," \
        "  \"deepstream_yolo_patch_sha256\": \"${DEEPSTREAM_YOLO_PATCH_SHA256}\"," \
        "  \"deepstream_yolo_upstream_source_sha256\": \"${DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256}\"," \
        "  \"deepstream_yolo_patched_source_sha256\": \"${DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256}\"," \
        "  \"deepstream_yolo_patched_tree\": \"${DEEPSTREAM_YOLO_PATCHED_TREE}\"," \
        '  "deepstream_yolo_parser_build_makefile_sha256": "fd2c03b810b8dae9d9d3a60b503616bbf6ed67a6f614843dd6a29f7f87ff8ad0",' \
        '  "deepstream_yolo_parser_cuda_cubin_architecture": "sm_86",' \
        '  "deepstream_yolo_parser_cuda_ptx_architecture": "compute_86",' \
        '  "deepstream_yolo_parser_cuda_gencode_flags": "-gencode=arch=compute_86,code=sm_86;-gencode=arch=compute_86,code=compute_86",' \
        '  "deepstream_yolo_parser_build_command_sha256": "e244df2d9c424fe7d027d62205ff21c820b58eb2cc00aa61df2d32cbfe329ac1",' \
        '  "deepstream_yolo_parser_post_link_tool_path": "/usr/bin/x86_64-linux-gnu-strip",' \
        '  "deepstream_yolo_parser_post_link_tool_sha256": "4dad0d12aa5d6a49b117b4551b897175ad5b43b9525e8f9efd661133a1c8ea0d",' \
        '  "deepstream_yolo_parser_post_link_tool_version": "GNU strip (GNU Binutils for Ubuntu) 2.42",' \
        '  "deepstream_yolo_parser_post_link_command": "/usr/bin/x86_64-linux-gnu-strip --strip-unneeded nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so",' \
        '  "deepstream_yolo_parser_post_link_command_sha256": "5e19627d403e984d9e349f8c81332f7280a1ac8477d56056c9b27be63aeef7ca",' \
        '  "deepstream_yolo_parser_post_link_removed_sections": ".symtab;.strtab",' \
        '  "deepstream_yolo_parser_post_link_retained_sections": ".dynsym;.dynstr",' \
        '  "cuda_kernel_proof_schema": "deepsafe.ds9-cuda-kernel-proof/v1",' \
        "  \"parser_sha256\": \"${DEEPSTREAM_YOLO_PARSER_SHA256}\"," \
        "  \"controller_sha256\": \"${DEEPSAFE_RUNTIME_CONTROLLER_SHA256}\"," \
        "  \"runtime_control_manifest_sha256\": \"${DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256}\"," \
        "  \"dockerignore_sha256\": \"${DEEPSAFE_DOCKERIGNORE_SHA256}\"" \
        '}' > /opt/deepsafe/build-lineage.json \
    && python3 -m json.tool /opt/deepsafe/build-lineage.json >/dev/null
LABEL com.deepsafe.build-lineage.schema="deepsafe.deepstream-build-lineage/v3" \
      com.deepsafe.deepstream.base-ref="${DEEPSTREAM_BASE_REF}" \
      com.deepsafe.deepstream.base-digest="${DEEPSTREAM_BASE_DIGEST}" \
      com.deepsafe.deepstream.version="9.0.0" \
      com.deepsafe.cuda.version="13.1" \
      com.deepsafe.tensorrt.version="10.14.1.48" \
      com.deepsafe.gstreamer.version="1.24.2" \
      com.deepsafe.deepstream-yolo.repository="https://github.com/marcoslucianops/DeepStream-Yolo.git" \
      com.deepsafe.deepstream-yolo.commit="${DEEPSTREAM_YOLO_COMMIT}" \
      com.deepsafe.deepstream-yolo.tree="${DEEPSTREAM_YOLO_TREE}" \
      com.deepsafe.deepstream-yolo.patch-sha256="${DEEPSTREAM_YOLO_PATCH_SHA256}" \
      com.deepsafe.deepstream-yolo.upstream-source-sha256="${DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256}" \
      com.deepsafe.deepstream-yolo.patched-source-sha256="${DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256}" \
      com.deepsafe.deepstream-yolo.patched-tree="${DEEPSTREAM_YOLO_PATCHED_TREE}" \
      com.deepsafe.deepstream-yolo.parser-build-makefile-sha256="fd2c03b810b8dae9d9d3a60b503616bbf6ed67a6f614843dd6a29f7f87ff8ad0" \
      com.deepsafe.deepstream-yolo.parser-cuda-cubin-architecture="sm_86" \
      com.deepsafe.deepstream-yolo.parser-cuda-ptx-architecture="compute_86" \
      com.deepsafe.deepstream-yolo.parser-cuda-gencode-flags="-gencode=arch=compute_86,code=sm_86;-gencode=arch=compute_86,code=compute_86" \
      com.deepsafe.deepstream-yolo.parser-build-command-sha256="e244df2d9c424fe7d027d62205ff21c820b58eb2cc00aa61df2d32cbfe329ac1" \
      com.deepsafe.deepstream-yolo.parser-post-link-tool-path="/usr/bin/x86_64-linux-gnu-strip" \
      com.deepsafe.deepstream-yolo.parser-post-link-tool-sha256="4dad0d12aa5d6a49b117b4551b897175ad5b43b9525e8f9efd661133a1c8ea0d" \
      com.deepsafe.deepstream-yolo.parser-post-link-tool-version="GNU strip (GNU Binutils for Ubuntu) 2.42" \
      com.deepsafe.deepstream-yolo.parser-post-link-command="/usr/bin/x86_64-linux-gnu-strip --strip-unneeded nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so" \
      com.deepsafe.deepstream-yolo.parser-post-link-command-sha256="5e19627d403e984d9e349f8c81332f7280a1ac8477d56056c9b27be63aeef7ca" \
      com.deepsafe.deepstream-yolo.parser-post-link-removed-sections=".symtab;.strtab" \
      com.deepsafe.deepstream-yolo.parser-post-link-retained-sections=".dynsym;.dynstr" \
      com.deepsafe.cuda-kernel-proof.schema="deepsafe.ds9-cuda-kernel-proof/v1" \
      com.deepsafe.deepstream-yolo.parser-sha256="${DEEPSTREAM_YOLO_PARSER_SHA256}" \
      com.deepsafe.runtime-compatibility-controller.sha256="${DEEPSAFE_RUNTIME_CONTROLLER_SHA256}" \
      com.deepsafe.runtime-control-manifest.sha256="${DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256}" \
      com.deepsafe.dockerignore.sha256="${DEEPSAFE_DOCKERIGNORE_SHA256}"
ENV PYTHONPATH=/app
CMD ["python3", "-m", "uvicorn", "admin.app:app", "--host", "0.0.0.0", "--port", "8080"]
