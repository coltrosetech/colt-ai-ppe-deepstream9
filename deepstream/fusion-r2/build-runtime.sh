#!/usr/bin/env bash
set -euo pipefail

umask 022

readonly FUSION_SOURCE=/build/src/deepstream/fusion
readonly FUSION_BUILD=/build/fusion-build
readonly APP_SOURCE=/build/src/upstream-app
readonly OUTPUT=/build/out
readonly PATCH=/build/src/deepstream/patches/deepsafe-fusion-ds9-app-r2.patch

test -d "${OUTPUT}"
test -f "${PATCH}"

cmake \
  -S "${FUSION_SOURCE}" \
  -B "${FUSION_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DDEEPSAFE_POSE_ROOT=/build/src/models/pose/postprocess \
  -DDEEPSAFE_PPE_ROOT=/build/src/models/ppe/postprocess
cmake --build "${FUSION_BUILD}" --parallel 2
(
  cd "${FUSION_BUILD}"
  ctest --output-on-failure
)

cp "${FUSION_SOURCE}/app-hook/deepstream_fusion_hook.cpp" "${APP_SOURCE}/"
cp "${FUSION_SOURCE}/app-hook/deepstream_fusion_hook.hpp" "${APP_SOURCE}/"
git apply --no-index --unsafe-paths --directory="${APP_SOURCE}" "${PATCH}"

if grep -Fq \
  "body_pose_gie_src_pad_buffer_probe, GST_PAD_PROBE_TYPE_BUFFER" \
  "${APP_SOURCE}/deepstream_parallel_infer_app.cpp"; then
  echo "legacy OpenPose probe remains registered" >&2
  exit 1
fi
test "$(grep -Fc 'deepsafe_fusion_app_hook_install(' \
  "${APP_SOURCE}/deepstream_parallel_infer_app.cpp")" -eq 1

make \
  -C "${APP_SOURCE}" \
  -j2 \
  DEEPSAFE_FUSION_INCLUDE_DIR="${FUSION_SOURCE}/include" \
  DEEPSAFE_FUSION_LIBRARY_DIR="${FUSION_BUILD}" \
  APP_INSTALL_DIR="${OUTPUT}/" \
  install

install -m 0555 \
  "${FUSION_BUILD}/libdeepsafe_fusion.so.1.0.0" \
  "${OUTPUT}/libdeepsafe_fusion.so.1"
install -m 0444 \
  "${FUSION_SOURCE}/default-runtime.conf" \
  "${OUTPUT}/fusion-runtime.conf"

test -x "${OUTPUT}/deepstream-parallel-infer"
test -x "${OUTPUT}/libdeepsafe_fusion.so.1"
test -s "${OUTPUT}/fusion-runtime.conf"
