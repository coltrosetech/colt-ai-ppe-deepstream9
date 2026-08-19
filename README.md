# COLT AI – COLLBRAI PPE / DeepStream 9

NVIDIA DeepStream 9 üzerinde çalışan, insan takibiyle kişi bazlı PPE durumunu
birleştiren video analitik teslimidir. Hazır hat; kask ve yüksek görünürlüklü
yelek var/yok durumunu NvDCF `track_id` üzerinden zamansal olarak tutar, alarm
başlangıç/bitişlerini üretir, yürüyüş yolu ROI kuralını ve forklift sürücüsü
bastırma kuralını uygular.

## Teslim kapsamı

- SafetyVision YOLOv8s v2 PPE modeli için 640 ve 960 DeepStream-uyumlu ONNX
- YOLO11s kişi algılama için 640 ve 960 ONNX + external-data sidecar
- DeepStream 9 runner'ları, PPE/kişi füzyonu, ROI ve forklift kuralları
- Video seçim/ROI arayüzü ve işlenmiş video renderer'ı
- Kurulum, model provenance, SHA-256 makbuzları ve testler
- TensorRT engine'leri dahil değildir; hedef GPU'da yeniden üretilir

Tam teknik devir belgesi: [docs/handoff/README.md](docs/handoff/README.md)  
Hızlı kurulum: [docs/handoff/HIZLI_KURULUM.md](docs/handoff/HIZLI_KURULUM.md)

## Sabitlenen çalışma ortamı

| Bileşen | Referans |
|---|---|
| DeepStream | 9.0.0 |
| CUDA / TensorRT | 13.1 / 10.14.1.48 |
| GPU | RTX A5000 Laptop, 16 GiB, compute capability 8.6 |
| Profiller | 640×640 ve 960×960, FP16, dinamik batch 1–12 |
| Container | `ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725` |

Farklı GPU mimarisinde parser ve TensorRT engine'leri hedef sistemde yeniden
üretilmeli ve canlı smoke testi yapılmalıdır. Mevcut `.engine` dosyalarını başka
GPU/TensorRT sürümüne taşımayın.

## Tam paketi indirme

GitHub Release, kaynak/model ağacını video ve hedef-GPU TensorRT engine'i
içermeyen tek arşiv olarak sunmaktadır. Smoke testi için ekip kendi PPE videosunu
repo içine eklemelidir:

```bash
gh release download v2026.08.19-handoff \
  --repo coltrosetech/colt-ai-ppe-deepstream9
sha256sum -c COLT-AI-PPE-GITHUB-RELEASE-20260819.sha256
tar --zstd -xf COLT-AI-PPE-RUNTIME-GITHUB-20260819.tar.zst
cd COLT-AI-PPE-RUNTIME-GITHUB-20260819
```

Container yayınlandıktan sonra:

```bash
docker login ghcr.io
docker pull ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725
docker tag \
  ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725 \
  deepsafe-deepstream:9.0-control-refresh-20260725
```

Runner sabit yerel image adını ve `--pull=never` davranışını kullandığı için
ikinci `docker tag` adımı gereklidir.

## Host kontrolü

```bash
./docs/handoff/ortam_kontrol.sh
nvidia-smi
docker run --rm --gpus all --pull=never \
  deepsafe-deepstream:9.0-control-refresh-20260725 nvidia-smi
```

## Ham PPE smoke koşusu

CLI doğrudan dosya yolu ile değil, repo kökünden Python modülü olarak çağrılır:
`--video` zorunludur ve repo içindeki bir dosyayı göstermelidir.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r admin/requirements.txt

.venv/bin/python -m validation.run_ppe_deepstream \
  --video data/samples/<ppe-smoke-video>.mp4 \
  --run-root validation/results/ppe/handoff-smoke \
  --profiles 640 960 \
  --gpu 0 \
  --start-seconds 80 \
  --duration-seconds 10 \
  --threshold 0.10 \
  --execute
```

İlk gerçek koşu, eksik TensorRT engine'lerini hedef GPU'da ONNX'ten üretir.
Her farklı parametre seti için yeni ve boş bir `--run-root` kullanın.

## Takipli tam PPE video hattı

Önce seçim/ROI arayüzünde kayıt oluşturulur, ardından kayıt kimliğiyle işlem
başlatılır:

```bash
docker compose up --build video-selector

.venv/bin/python -m video_selector.process_ppe_selection \
  --selection-id <32-karakterlik-secim-id> \
  --profiles 640 960 \
  --gpu 0 \
  --threshold 0.10 \
  --execute
```

Bu yol SafetyVision PPE çıktısını YOLO11s + NvDCF kişi kimliğiyle birleştirir ve
COLT AI – COLLBRAI temalı işlenmiş H.264 MP4 üretir.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_run_ppe_deepstream.py \
  tests/test_process_ppe_selection.py \
  tests/test_person_ppe_fusion.py \
  tests/test_ppe_video.py \
  tests/test_forklift_driver_rules.py \
  tests/test_person_zone_rules.py \
  tests/test_run_person_deepstream_direct.py \
  tests/test_ppe_ds9_adapter_parity.py
```

GitHub-safe paket üzerinde bu kapsamda `131 passed, 1 skipped` sonucu alınmıştır.

## Model provenance ve kullanım durumu

PPE checkpoint'i Hugging Face üzerindeki
[`ayushgupta7777/safetyvision-yolov8`](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)
deposunun `56a71758b55f0e9f2b4b2d6b51a779a1f882da10` commit'inden alınmıştır.
Checkpoint SHA-256 değeri
`7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a` ve
lisansı AGPL-3.0'dır. Ayrıntılar
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) ve
[data/manifests](data/manifests) altındadır.

Bu teslimin mevcut kabul durumu `diagnostic_content_evidence_only`,
`production_ready=false` ve `commercially_cleared=false` olarak korunmuştur.
Çıktılar güvenlik personelinin incelemesine yardımcı olur; tek başına disiplin,
uygunluk veya iş güvenliği kararı vermek için kullanılmamalıdır.

## Hariç tutulan legacy alan

Hosta sabitlenmiş DeepStream 9.1 engine-builder deneyleri bu GitHub devrinin
dışında bırakılmıştır. Aktif ve doğrulanmış PPE lane'i DeepStream 9.0'dır.
