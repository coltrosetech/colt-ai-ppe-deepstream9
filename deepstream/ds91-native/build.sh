#!/usr/bin/env bash
set -euo pipefail

umask 022

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="$(cd "${HERE}/../.." && pwd)"
readonly BASE_DIGEST="sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994"
readonly BASE_REF="nvcr.io/nvidia/deepstream@${BASE_DIGEST}"
readonly IMAGE_TAG="${DEEPSAFE_DS91_NATIVE_TAG:-deepsafe-deepstream:9.1-native-r1}"

python3 "${HERE}/verify_static.py"

test "$(uname -m)" = "x86_64"
test "$(docker image inspect "${BASE_REF}" --format '{{.Id}}')" = "${BASE_DIGEST}"
test "$(docker image inspect "${BASE_REF}" --format '{{.Architecture}}/{{.Os}}')" = "amd64/linux"

docker buildx build \
  --file "${HERE}/Dockerfile" \
  --network=none \
  --platform=linux/amd64 \
  --load \
  --provenance=false \
  --build-context person="${ROOT}/models/person/postprocess/rtdetrv4_ds9" \
  --build-context pose="${ROOT}/models/pose/postprocess" \
  --build-context ppe="${ROOT}/models/ppe/postprocess" \
  --build-context fusion="${ROOT}/deepstream/fusion" \
  --build-context fusion_runner="${ROOT}/deepstream/fusion-r2" \
  --build-context patches="${ROOT}/deepstream/patches" \
  --build-context upstream_app="${ROOT}/third_party/deepstream_reference_apps/deepstream_parallel_inference_app/tritonclient/sample/apps/deepstream-parallel-infer" \
  --tag "${IMAGE_TAG}" \
  "${HERE}"

test "$(docker image inspect "${IMAGE_TAG}" --format '{{.Architecture}}/{{.Os}}')" = "amd64/linux"
test "$(docker image inspect "${IMAGE_TAG}" --format '{{index .Config.Labels "com.deepsafe.native-build.schema"}}')" = "deepsafe.ds91-native-build/v1"
test "$(docker image inspect "${IMAGE_TAG}" --format '{{index .Config.Labels "com.deepsafe.deepstream.base-digest"}}')" = "${BASE_DIGEST}"
test "$(docker image inspect "${IMAGE_TAG}" --format '{{index .Config.Labels "com.deepsafe.build.network"}}')" = "none"
test "$(docker image inspect "${IMAGE_TAG}" --format '{{index .Config.Labels "com.deepsafe.build.gpu-used"}}')" = "false"

printf '%s\n' "DeepStream 9.1 native image built and statically verified: ${IMAGE_TAG}"

