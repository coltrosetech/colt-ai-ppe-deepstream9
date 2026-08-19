# COLT AI PPE – hızlı kurulum

Tam açıklama için `README.md` dosyasını okuyun. Aşağıdaki akış, Ubuntu 24.04 x86_64 ve NVIDIA GPU bulunan yeni bilgisayarda ilk 5 saniyelik PPE koşusunu almak içindir.

## 1. Host kontrolü

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
docker version
docker compose version
docker info --format '{{json .Runtimes}}'
ffmpeg -hide_banner -encoders 2>/dev/null | grep h264_nvenc
```

Referans GPU compute capability `8.6` ve VRAM `16384 MiB`'dir.

## 2. DeepStream image'ını yükle

```bash
docker login ghcr.io
docker pull ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725
docker tag \
  ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725 \
  deepsafe-deepstream:9.0-control-refresh-20260725

docker image inspect deepsafe-deepstream:9.0-control-refresh-20260725 \
  --format '{{.Id}}'

docker run --rm --gpus all --pull=never \
  deepsafe-deepstream:9.0-control-refresh-20260725 nvidia-smi
```

Beklenen image ID:

```text
sha256:c0be08184405fff2161d7ca65e31601c309da7845e4e74453fc6b13fc328bf27
```

## 3. Projeyi hazırla

```bash
cd /hedef/yol/COLT-AI-PPE-RUNTIME-GITHUB-20260819

rm -f models/ppe/safetyvision-yolov8s-v2/640/*.engine
rm -f models/ppe/safetyvision-yolov8s-v2/960/*.engine
rm -f models/person/640/*.engine
rm -f models/person/960/*.engine

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r admin/requirements.txt
```

## 4. Dosyaları doğrula

Bu devir klasöründeki hash dosyasını repo köküne göre çalıştırın:

```bash
sha256sum -c teslimler/COLT-AI-PPE-TEKNIK-DEVIR-20260818/MODEL_DOSYALARI.sha256
```

## 5. Smoke videosunu ekle

GitHub teslimi video içermez. Kullanma yetkiniz bulunan ve en az bir PPE
tespiti içeren videoyu örneğin `data/samples/ppe-smoke.mp4` yoluna yerleştirin.

## 6. Önce plan, sonra GPU koşusu

```bash
.venv/bin/python -m validation.run_ppe_deepstream \
  --video data/samples/ppe-smoke.mp4 \
  --run-root validation/results/ppe/handoff-smoke-960 \
  --profiles 960 --gpu 0 \
  --start-seconds 0 --duration-seconds 5 \
  --threshold 0.10
```

Gerçek koşu:

```bash
.venv/bin/python -m validation.run_ppe_deepstream \
  --video data/samples/ppe-smoke.mp4 \
  --run-root validation/results/ppe/handoff-smoke-960 \
  --profiles 960 --gpu 0 \
  --start-seconds 0 --duration-seconds 5 \
  --threshold 0.10 --execute
```

İlk çalıştırma 960 TensorRT engine'ini hedef GPU üzerinde otomatik üretir.

## 7. Sonucu kontrol et

```bash
jq '{status, profiles: [.profiles[] | {
  profile,
  status,
  engine: .engine_load_attestation.status,
  detections: .conversion.statistics.ppe_detections
}]}' validation/results/ppe/handoff-smoke-960/manifest.json
```

Beklenen: ana ve profil `status=complete`, engine `pass`, detection sayısı sıfırdan büyük.

## 8. İşlenmiş MP4 gerekiyorsa

Önce video seçim/ROI arayüzünü açıp yeni kayıt oluşturun:

```bash
docker compose up --build video-selector
```

Arayüzün ürettiği 32 karakterlik seçim kimliğiyle çalıştırın:

```bash
.venv/bin/python -m video_selector.process_ppe_selection \
  --selection-id <secim-id> \
  --profiles 960 --gpu 0 --threshold 0.10 \
  --execute
```

Çıktılar `validation/results/content-deliveries/` altında oluşur.
