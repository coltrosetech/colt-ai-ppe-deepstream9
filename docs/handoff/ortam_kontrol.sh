#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IMAGE="deepsafe-deepstream:9.0-control-refresh-20260725"
EXPECTED_IMAGE_ID="sha256:c0be08184405fff2161d7ca65e31601c309da7845e4e74453fc6b13fc328bf27"
HASH_FILE="${SCRIPT_DIR}/MODEL_DOSYALARI.sha256"
failures=0

pass() {
  printf '[PASS] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1"
  failures=$((failures + 1))
}

printf 'COLT AI PPE hedef ortam kontrolü\n'
printf 'Repo: %s\n\n' "${ROOT}"

if [[ "$(uname -m)" == "x86_64" ]]; then
  pass "Mimari x86_64"
else
  fail "Mimari x86_64 değil: $(uname -m)"
fi

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  printf '[INFO] İşletim sistemi: %s\n' "${PRETTY_NAME:-bilinmiyor}"
else
  warn "/etc/os-release okunamadı"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_row="$(nvidia-smi \
    --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  if [[ -n "${gpu_row}" ]]; then
    pass "NVIDIA GPU görünüyor: ${gpu_row}"
    compute_cap="$(printf '%s' "${gpu_row}" | awk -F',' '{gsub(/ /,"",$4); print $4}')"
    if [[ "${compute_cap}" == "8.6" ]]; then
      pass "Compute capability 8.6; mevcut parser ile aynı hedef"
    else
      warn "Compute capability ${compute_cap}; mevcut parser sm_86/compute_86 için derlendi"
    fi
  else
    fail "nvidia-smi GPU satırı üretmedi"
  fi
else
  fail "nvidia-smi bulunamadı"
fi

if command -v docker >/dev/null 2>&1; then
  if docker version >/dev/null 2>&1; then
    pass "Docker daemon erişilebilir"
  else
    fail "Docker kurulu fakat daemon erişilemiyor"
  fi
else
  fail "Docker bulunamadı"
fi

if docker compose version >/dev/null 2>&1; then
  pass "Docker Compose plugin erişilebilir"
else
  fail "Docker Compose plugin bulunamadı"
fi

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q 'nvidia'; then
  pass "Docker nvidia runtime kayıtlı"
else
  fail "Docker nvidia runtime görünmüyor"
fi

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  image_id="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
  pass "DeepStream image mevcut: ${image_id}"
  if [[ "${image_id}" == "${EXPECTED_IMAGE_ID}" ]]; then
    pass "Image ID referans teslim ile aynı"
  else
    warn "Image yeniden build edilmiş veya referanstan farklı"
  fi

  labels="$(docker image inspect "${IMAGE}" \
    --format '{{index .Config.Labels "com.deepsafe.deepstream.version"}} {{index .Config.Labels "com.deepsafe.cuda.version"}} {{index .Config.Labels "com.deepsafe.tensorrt.version"}}' 2>/dev/null)"
  if [[ "${labels}" == "9.0.0 13.1 10.14.1.48" ]]; then
    pass "Runtime label'ları doğru: ${labels}"
  else
    fail "Runtime label'ları beklenenden farklı: ${labels}"
  fi

  if docker run --rm --gpus all --pull=never "${IMAGE}" \
    nvidia-smi -L >/dev/null 2>&1; then
    pass "Container GPU erişimi çalışıyor"
  else
    fail "Container GPU erişimi başarısız"
  fi
else
  fail "DeepStream image yok: ${IMAGE}"
fi

if command -v python3 >/dev/null 2>&1; then
  python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)"
  if [[ "${python_version}" == "3.12" ]]; then
    pass "Python 3.12"
  else
    warn "Python ${python_version}; referans Python 3.12"
  fi
else
  fail "python3 bulunamadı"
fi

if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  pass "FFmpeg ve FFprobe mevcut"
  if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'h264_nvenc'; then
    pass "FFmpeg h264_nvenc encoder mevcut"
  else
    fail "FFmpeg h264_nvenc encoder bulunamadı"
  fi
else
  fail "FFmpeg veya FFprobe bulunamadı"
fi

if [[ -f "${HASH_FILE}" ]]; then
  if (cd "${ROOT}" && sha256sum -c "${HASH_FILE}" >/dev/null 2>&1); then
    pass "PPE/person model dosyalarının SHA-256 doğrulaması geçti"
  else
    fail "Model dosyası eksik veya SHA-256 doğrulaması başarısız"
  fi
else
  fail "Hash listesi bulunamadı: ${HASH_FILE}"
fi

engine_count="$(find \
  "${ROOT}/models/ppe/safetyvision-yolov8s-v2/640" \
  "${ROOT}/models/ppe/safetyvision-yolov8s-v2/960" \
  "${ROOT}/models/person/640" \
  "${ROOT}/models/person/960" \
  -maxdepth 1 -type f -name '*.engine' 2>/dev/null | wc -l)"

if [[ "${engine_count}" -eq 0 ]]; then
  pass "Taşınmış TensorRT engine yok; hedefte yeniden üretilebilir"
else
  warn "${engine_count} adet .engine bulundu; başka bilgisayardan geldiyse ilk koşudan önce silin"
fi

printf '\n'
if [[ "${failures}" -eq 0 ]]; then
  printf 'SONUÇ: temel ortam kontrolleri geçti.\n'
  exit 0
fi

printf 'SONUÇ: %d zorunlu kontrol başarısız.\n' "${failures}"
exit 1
