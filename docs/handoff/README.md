# COLT AI – PPE teknik devir ve kurulum rehberi

Tarih: 18 Ağustos 2026  
Kapsam: PPE modeli, DeepStream 9 çalıştırma hattı, kişi takibi ve işlenmiş video üretimi

Bu belge, mevcut PPE sistemini başka bir Ubuntu/NVIDIA bilgisayarda çalıştıracak ekip arkadaşı içindir. Yeni özellik geliştirme planı içermez; mevcut sistemi kurma, doğrulama ve çalıştırma adımlarını anlatır.

> GitHub tesliminde medya ve TensorRT engine'leri dağıtılmamaktadır. Güncel
> arşiv/container yolları ve hangi dosyaların bilerek çıkarıldığı için önce
> [`GITHUB_YAYINI.md`](GITHUB_YAYINI.md) dosyasını okuyun. Bu not, aşağıdaki
> eski yerel arşiv/video örneklerinin yerine geçer.

## 1. Kısa özet

Kullanılan PPE modeli **SafetyVision YOLOv8s v2**'dir. Model **Hugging Face** üzerindeki `ayushgupta7777/safetyvision-yolov8` deposundan, aşağıdaki kesin revizyondan alınmıştır:

- Model kartı: <https://huggingface.co/ayushgupta7777/safetyvision-yolov8>
- Commit: `56a71758b55f0e9f2b4b2d6b51a779a1f882da10`
- Kaynak checkpoint: `v2/best.pt`
- Model ailesi: Ultralytics YOLOv8s
- Lisans: AGPL-3.0

Çalışan ve doğrulanmış runtime:

| Bileşen | Kullanılan sürüm / yapı |
|---|---|
| İşletim sistemi | Ubuntu 24.04 x86_64 |
| DeepStream | 9.0.0 |
| CUDA – container içi | 13.1 |
| TensorRT | 10.14.1.48 |
| GStreamer | 1.24.2 |
| Çalışma hassasiyeti | TensorRT FP16, TF32 kapalı |
| Test edilen GPU | NVIDIA RTX A5000 Laptop, 16 GiB, compute capability 8.6 |
| Test edilen güncel driver | 595.84 |
| Model girişleri | 640×640 ve 960×960 |
| Engine profili | Dinamik batch 1–12; optimum/maksimum batch 12 |
| DeepStream parser | `NvDsInferParseYoloCuda` |
| Kişi izleme | YOLO11s person PGIE + NvDCF |

## 2. Bizim tamamladığımız işler

- Hugging Face checkpoint'i hash ile sabitlendi.
- Kaynak checkpoint'ten dinamik batch 1–12 destekleyen 640 ve 960 ONNX dosyaları üretildi.
- İki ONNX, DeepStream-Yolo CUDA parser'ına uygun `raw6` çıktısına adapte edildi.
- 640 ve 960 adaptör çıktıları gerçek kare üzerinde kaynak ONNX ile karşılaştırıldı; kayıtlı maksimum mutlak fark `0.0`.
- Her iki profil için RTX A5000 üzerinde FP16 TensorRT engine üretildi ve DeepStream 9 ile gerçek GPU koşusu yapıldı.
- PPE çıktıları, bağımsız YOLO11s kişi algılama ve NvDCF track ID'leriyle insan bazında birleştirildi.
- Kask ve yüksek görünürlüklü yelek var/yok durumu kişi bazında zamansal olarak tutuldu; her karede yeni kimlik oluşturulması engellendi.
- Alarm başlangıç/bitiş geçişleri üretildi; aynı ihlal her karede yeni alarm olarak yazılmıyor.
- Güvenli yürüyüş yolu ROI kuralı ve kişi alt-orta nokta kontrolü bağlandı.
- Forklift sürücüsü bastırma hattı eklendi. Mevcut uygulama özel forklift modeli kullanmıyor; YOLO11s COCO `truck` sınıfı `forklift_candidate` kanıtı olarak ele alınıyor.
- COLT AI – COLLBRAI görsel sözleşmesiyle 1920×1080 işlenmiş video çıktıları üretildi.
- S01, S02, S03 ve S04 örnekleri 960 profilde tamamlandı.
- Çekirdek PPE/füzyon/renderer test kümesi çalıştırıldı: `126 passed, 1 skipped`. Daha dar kurulum smoke kümesi: `62 passed`.

Projedeki durum alanları bu modeli hâlâ `diagnostic_content_evidence_only`, `production_ready=false` ve `commercially_cleared=false` olarak kaydeder. Bu alanlar çalışma hatası değildir; mevcut doğrulama seviyesini ifade eder.

## 3. Sistem nasıl çalışıyor?

Ham model ve tam video hattı iki farklı kullanım yoludur:

```text
Video ──> SafetyVision PPE / DeepStream 9 ──> PPE bbox + sınıf + skor / JSONL
   │
   └──> YOLO11s person / DeepStream 9 ──> NvDCF track_id
                                             │
PPE JSONL + kişi track'leri ──> kişi-PPE füzyonu
                               ├──> kask / yelek durumu
                               ├──> alarm başlangıç-bitişleri
                               ├──> yürüyüş yolu kuralı
                               ├──> forklift sürücüsü bastırma
                               └──> işlenmiş H.264 MP4
```

### 3.1 PPE sınıfları

Modelin 13 sınıfı:

```text
0 Fall-Detected
1 Gloves
2 Goggles
3 Hardhat
4 Mask
5 NO-Gloves
6 NO-Goggles
7 NO-Hardhat
8 NO-Mask
9 NO-Safety Vest
10 No_Harness
11 Person
12 Safety Vest
```

Mevcut runtime yalnız `3`, `7`, `9`, `11` ve `12` sınıflarını DeepStream çıktısında tutar. Uygulama eşlemesi:

| Model sınıfı | Uygulama alanı |
|---|---|
| Hardhat | `helmet` |
| NO-Hardhat | `no_helmet` |
| Safety Vest | `hi_vis` |
| NO-Safety Vest | `no_hi_vis` |
| Person | Ham PPE lane'de yardımcı kişi kanıtı |

Tam video hattında esas kişi kutusu PPE modelinin `Person` sınıfından değil, YOLO11s + NvDCF hattından gelir.

## 4. Gönderilmesi gerekenler

Yeni bilgisayara aşağıdaki üç parçayı aktarın:

1. Bu proje kaynak ağacı veya hazırlanan runtime kaynak/model arşivi.
2. `deepsafe-deepstream:9.0-control-refresh-20260725` Docker image arşivi ya da image'ı yeniden kurmak için NGC erişimi.
3. İşlenecek videolar ve varsa kayıtlı seçim/ROI JSON dosyaları.

`*.engine` dosyalarını hedef bilgisayarda kullanmayın. TensorRT engine donanım ve runtime'a bağlıdır; hedefte ONNX'ten yeniden üretilmelidir.

### 4.1 Ham PPE koşusu için zorunlu dosyalar

```text
models/ppe/safetyvision-yolov8s-v2/labels.txt
models/ppe/safetyvision-yolov8s-v2/ds9-raw6-real-frame-parity.json
models/ppe/safetyvision-yolov8s-v2/640/ds9-raw6-receipt.json
models/ppe/safetyvision-yolov8s-v2/640/safetyvision-yolov8s-v2-640-ds9-raw6.onnx
models/ppe/safetyvision-yolov8s-v2/960/ds9-raw6-receipt.json
models/ppe/safetyvision-yolov8s-v2/960/safetyvision-yolov8s-v2-960-ds9-raw6.onnx
validation/results/ppe/models/safetyvision-yolov8s-v2-cpu-export-r3/
  safetyvision-v2-cpu-export-r3-001/artifacts/
    safetyvision-yolov8s-v2-640-bdynamic-opset18.onnx
    safetyvision-yolov8s-v2-960-bdynamic-opset18.onnx
validation/run_ppe_deepstream.py
validation/run_caviar.py
```

Runner, adaptör makbuzlarını kaynak ONNX ve adapted ONNX hash'leriyle doğruladığı için hem kaynak hem adapted ONNX dosyaları gereklidir.

### 4.2 Tam takipli ve işlenmiş MP4 hattı için ek dosyalar

```text
models/person/640/yolo11s.onnx
models/person/640/yolo11s.onnx.data
models/person/640/labels.txt
models/person/960/yolo11s.onnx
models/person/960/yolo11s.onnx.data
models/person/960/labels.txt
validation/run_person_deepstream_direct.py
content/person_ppe_fusion.py
content/forklift_driver_rules.py
content/person_zone_rules.py
content/ppe_video.py
content/theme.py
video_selector/process_ppe_selection.py
```

`yolo11s.onnx.data` dosyaları ONNX external-data sidecar'ıdır ve zorunludur. NvDCF ayarı Docker image içindeki şu dosyadan kullanılır:

```text
/opt/nvidia/deepstream/deepstream-9.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml
```

Çekirdek dosyaların tam SHA-256 listesi bu klasördeki `MODEL_DOSYALARI.sha256` dosyasındadır.

### 4.3 Hazırlanan taşınabilir runtime arşivi

Teslimde ayrıca şu arşiv bulunur:

```text
COLT-AI-PPE-RUNTIME-KAYNAK-20260818.tar.zst
```

Arşiv kaynak kodu, iki PPE profili, iki kişi profili, source/adapted ONNX dosyaları, PPE `best.pt`, S01–S04 örnekleri ve testleri içerir. Hedefe özel üretilmesi gereken hiçbir `.engine` dosyası içermez.

Açma komutu:

```bash
tar --zstd -xf COLT-AI-PPE-RUNTIME-KAYNAK-20260818.tar.zst
cd COLT-AI-PPE-RUNTIME-KAYNAK-20260818
```

Arşiv ayrı bir geçici dizine açılarak doğrulandı: model hash kontrolü geçti, iki CLI açıldı ve genişletilmiş test kümesi `126 passed, 1 skipped` verdi.

## 5. Hedef bilgisayar gereksinimleri

Kesin tekrar için önerilen hedef:

- Ubuntu 24.04, x86_64/amd64
- NVIDIA dGPU
- DeepStream 9'un desteklediği NVIDIA driver; referans sistemde 595.84 çalışıyor
- Compute capability 8.6 GPU, çünkü mevcut özel parser `sm_86 + compute_86` olarak derlendi
- 16 GiB VRAM referans konfigürasyon; daha düşük VRAM için kabul ölçümü yapılmadı
- En az 50 GB boş disk alanı; DeepStream Triton image sanal boyutu yaklaşık 26.3 GB
- Docker Engine ve Docker Compose plugin
- NVIDIA Container Toolkit
- Python 3.12, `venv`, FFmpeg ve FFprobe
- FFmpeg içinde `h264_nvenc` encoder'ı

Host'a ayrı CUDA Toolkit veya TensorRT kurmak zorunlu değildir; bunlar Docker image içinde gelir. Host tarafında NVIDIA driver ve container runtime çalışmalıdır.

Mevcut Docker image `linux/amd64`'dir. ARM/Jetson üzerinde doğrudan çalışmaz.

## 6. Sıfırdan host kurulumu

Komutlar Ubuntu 24.04 içindir.

### 6.1 Temel paketler

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg2 git python3 python3-venv \
  ffmpeg jq zstd ubuntu-drivers-common
```

Ubuntu'nun hedef GPU için önerdiği NVIDIA driver'ı kurun:

```bash
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

Bilgisayar açıldıktan sonra kontrol edin:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
```

### 6.2 Docker Engine

Docker'ın resmî Ubuntu deposu:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

İsterseniz kullanıcıyı Docker grubuna ekleyin; yeniden oturum açılması gerekir:

```bash
sudo usermod -aG docker "$USER"
```

### 6.3 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o \
    /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Kurulumu doğrulayın:

```bash
docker info --format '{{json .Runtimes}}'
```

Çıktıda `nvidia` runtime görünmelidir.

Resmî güncel kurulum kaynakları:

- Docker Engine: <https://docs.docker.com/engine/install/ubuntu/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- DeepStream 9.0 kurulum matrisi: <https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_Installation.html>
- DeepStream 9.0 Docker: <https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_docker_containers.html>

## 7. DeepStream Docker image'ını aktarma

Runner image adını sabit kullanır:

```text
deepsafe-deepstream:9.0-control-refresh-20260725
```

Runner ayrıca `--pull=never` kullandığı için bu image hedef bilgisayarda önceden bulunmalıdır.

### 7.1 Önerilen yol: mevcut image'ı save/load ile taşıma

Kaynak bilgisayarda:

```bash
docker image inspect deepsafe-deepstream:9.0-control-refresh-20260725 \
  --format '{{.Id}}'

docker save deepsafe-deepstream:9.0-control-refresh-20260725 \
  | zstd -T0 -10 -o deepsafe-ds9-ppe-20260818.tar.zst
```

Beklenen image ID:

```text
sha256:c0be08184405fff2161d7ca65e31601c309da7845e4e74453fc6b13fc328bf27
```

Hedef bilgisayarda:

```bash
zstd -dc deepsafe-ds9-ppe-20260818.tar.zst | docker load

docker image inspect deepsafe-deepstream:9.0-control-refresh-20260725 \
  --format '{{.Id}}'
```

GPU'nun container içinde göründüğünü kontrol edin:

```bash
docker run --rm --gpus all --pull=never \
  deepsafe-deepstream:9.0-control-refresh-20260725 nvidia-smi
```

Runtime label kontrolü:

```bash
docker image inspect deepsafe-deepstream:9.0-control-refresh-20260725 \
  --format '{{index .Config.Labels "com.deepsafe.deepstream.version"}} {{index .Config.Labels "com.deepsafe.cuda.version"}} {{index .Config.Labels "com.deepsafe.tensorrt.version"}}'
```

Beklenen çıktı:

```text
9.0.0 13.1 10.14.1.48
```

### 7.2 Alternatif: image'ı yeniden kurma

NGC erişimi olan hedefte önce tabanı çekin:

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/deepstream:9.0-triton-multiarch@sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a
```

Repo kökünde:

```bash
docker build \
  -f deepstream/Dockerfile \
  --target runtime \
  --no-cache \
  --pull=false \
  -t deepsafe-deepstream:9.0-control-refresh-20260725 \
  --build-arg DEEPSTREAM_BASE_REF=nvcr.io/nvidia/deepstream:9.0-triton-multiarch@sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a \
  --build-arg DEEPSTREAM_BASE_DIGEST=sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a \
  --build-arg DEEPSTREAM_YOLO_PARSER_SHA256=2aa44a3395047ae371bee857476b1e78b438776c8a6b9643a055a16a0f15a7ae \
  --build-arg DEEPSAFE_RUNTIME_CONTROLLER_SHA256=$(sha256sum validation/ds9_runtime_compatibility.py | cut -d' ' -f1) \
  --build-arg DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256=$(sha256sum deepstream/runtime-control-manifest.json | cut -d' ' -f1) \
  --build-arg DEEPSAFE_DOCKERIGNORE_SHA256=$(sha256sum .dockerignore | cut -d' ' -f1) \
  .
```

Bu Dockerfile parser'ı `sm_86 + compute_86` için derler. Hedef compute capability 8.6 değilse mevcut parser binary'si exact hedef değildir; `deepstream/Dockerfile` içindeki CUDA gencode değerleri hedef SM'e göre yeniden derlenmelidir. Projedeki iki-geçişli parser runbook'u: `docs/ds9-parser-bootstrap-and-gpu-smoke.md`.

## 8. Proje ve Python ortamı

Arşivi açtıktan sonra daima proje kökünde çalışın:

```bash
cd /hedef/yol/ai-sistemleri
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r admin/requirements.txt
```

CLI'ları dosya yolu ile değil, Python modülü olarak çalıştırın:

```bash
.venv/bin/python -m validation.run_ppe_deepstream --help
.venv/bin/python -m video_selector.process_ppe_selection --help
```

Şu kullanım hatalıdır ve `ModuleNotFoundError: validation` üretebilir:

```text
python validation/run_ppe_deepstream.py
```

FFmpeg encoder kontrolü:

```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep h264_nvenc
```

## 9. Hedef bilgisayarda TensorRT engine üretimi

Projeyi kopyalarken engine dosyalarını hariç tutun. Kopyalandılarsa hedefte silin:

```bash
rm -f models/ppe/safetyvision-yolov8s-v2/640/*.engine
rm -f models/ppe/safetyvision-yolov8s-v2/960/*.engine
rm -f models/person/640/*.engine
rm -f models/person/960/*.engine
```

`--force` eski engine'i yeniden üretmez. Runner engine dosyasını yalnız yoksa ONNX'ten üretir.

İlk gerçek PPE koşusu sırasında runner otomatik olarak image içindeki `trtexec` ile şu profili üretir:

```text
precision: FP16
TF32: off
min shape: images:1x3xPxP
opt shape: images:12x3xPxP
max shape: images:12x3xPxP
workspace: 4096 MiB
P: 640 veya 960
```

Üretilen PPE engine yolları:

```text
models/ppe/safetyvision-yolov8s-v2/640/safetyvision_yolov8s_v2_ds9raw6_b12_gpu0_fp16.engine
models/ppe/safetyvision-yolov8s-v2/960/safetyvision_yolov8s_v2_ds9raw6_b12_gpu0_fp16.engine
```

Tam video hattı ayrıca kişi engine'lerini üretir:

```text
models/person/640/yolo11s_b12_gpu0_fp16.engine
models/person/960/yolo11s_b12_gpu0_fp16.engine
```

## 10. İlk smoke koşusu: ham PPE modeli

Bu yol yalnız SafetyVision PPE modelini çalıştırır. JSONL/KITTI üretir; işlenmiş MP4 üretmez.

Önce planı görün:

```bash
.venv/bin/python -m validation.run_ppe_deepstream \
  --video content/ppe-video-selector/media/S01.mp4 \
  --run-root validation/results/ppe/handoff-smoke-960 \
  --profiles 960 \
  --gpu 0 \
  --start-seconds 0 \
  --duration-seconds 5 \
  --threshold 0.10
```

Plan doğruysa gerçek GPU koşusu:

```bash
.venv/bin/python -m validation.run_ppe_deepstream \
  --video content/ppe-video-selector/media/S01.mp4 \
  --run-root validation/results/ppe/handoff-smoke-960 \
  --profiles 960 \
  --gpu 0 \
  --start-seconds 0 \
  --duration-seconds 5 \
  --threshold 0.10 \
  --execute
```

Yeni parametrelerle yeni bir `--run-root` kullanın. Aynı planın yarım kalan çıktısını tekrar etmek için `--force` eklenebilir.

Ham çıktı ağacı:

```text
validation/results/ppe/handoff-smoke-960/
├── plan.json
├── manifest.json
├── source-smoke.mp4
└── 960/
    ├── generated/config-infer-primary.txt
    ├── generated/deepstream-app.txt
    ├── engine-build.log
    ├── deepstream.log
    ├── kitti/*.txt
    ├── predictions.jsonl
    └── conversion.json
```

Başarı kontrolü:

```bash
jq '{status, profiles: [.profiles[] | {
  profile,
  status,
  engine: .engine_load_attestation.status,
  detections: .conversion.statistics.ppe_detections
}]}' validation/results/ppe/handoff-smoke-960/manifest.json
```

Beklenenler:

- ana `status`: `complete`
- profil `status`: `complete`
- engine attestation: `pass`
- `ppe_detections`: sıfırdan büyük

Smoke videosunda kask/yeleğe ilişkin en az bir tespit bulunmalıdır. Yalnız kişi görülen bir video, bu doğrulama runner'ında başarılı kabul edilmez.

## 11. Tam takipli ve işlenmiş video koşusu

Bu yol aşağıdakileri birlikte çalıştırır:

- SafetyVision PPE DeepStream inference
- YOLO11s kişi inference
- NvDCF tracking ve kalıcı `track_id`
- insan–PPE eşleştirme
- zamansal kask/yelek durumu
- varsa güvenli yürüyüş yolu ROI kuralı
- varsa forklift sürücüsü bastırma
- COLT AI – COLLBRAI işlenmiş H.264 MP4

Kayıtlı seçim ve queue dosyası şu dizinde olmalıdır:

```text
content/video-selector/state/queues/<selection-id>.json
```

Örnek plan:

```bash
.venv/bin/python -m video_selector.process_ppe_selection \
  --selection-id fe3a39d9d0f54719a320f170cebc15bc \
  --profiles 960 \
  --gpu 0 \
  --threshold 0.10
```

Gerçek koşu:

```bash
.venv/bin/python -m video_selector.process_ppe_selection \
  --selection-id fe3a39d9d0f54719a320f170cebc15bc \
  --profiles 960 \
  --gpu 0 \
  --threshold 0.10 \
  --execute
```

Tek video çalıştırmak için `--video-id` kullanın:

```bash
.venv/bin/python -m video_selector.process_ppe_selection \
  --selection-id fe3a39d9d0f54719a320f170cebc15bc \
  --video-id S01 \
  --profiles 960 \
  --gpu 0 \
  --threshold 0.10 \
  --execute
```

Tam çıktı kökü:

```text
validation/results/content-deliveries/
  video-selector-<selection-ilk-8>/ppe/<video-id>/
```

Başlıca çıktılar:

```text
deepstream-960/...                         ham PPE sonuçları
person-deepstream-960/...                  YOLO11s + NvDCF sonuçları
person-ppe-960.jsonl                       kişi bazlı PPE füzyonu
person-ppe-forklift-960.jsonl              forklift kuralı uygulanmış akış
COLT-AI-COLLBRAI-CAM-<ID>-PPE-960-*.mp4   işlenmiş video
ppe-manifest-960.json                      teslim makbuzu
```

## 12. Testler

Hızlı çekirdek test:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_run_ppe_deepstream.py \
  tests/test_process_ppe_selection.py \
  tests/test_person_ppe_fusion.py \
  tests/test_ppe_video.py
```

Mevcut doğrulanmış sonuç: `62 passed`.

ROI/forklift kurallarıyla genişletilmiş test:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_run_ppe_deepstream.py \
  tests/test_process_ppe_selection.py \
  tests/test_person_ppe_fusion.py \
  tests/test_ppe_video.py \
  tests/test_person_zone_rules.py \
  tests/test_forklift_driver_rules.py \
  tests/test_video_selector.py \
  tests/test_ppe_ds9_onnx_adapter.py
```

Mevcut doğrulanmış sonuç: `126 passed, 1 skipped`.

## 13. Sık karşılaşılan sorunlar

### `ModuleNotFoundError: validation`

Repo kökünde olun ve modül biçimini kullanın:

```bash
.venv/bin/python -m validation.run_ppe_deepstream --help
```

### `pull access denied` veya image bulunamadı

Runner registry'den çekmez. Önce `docker load` yapın ve tag'i kontrol edin:

```bash
docker image inspect deepsafe-deepstream:9.0-control-refresh-20260725
```

### TensorRT engine deserialize hatası

Kaynak bilgisayar engine'i kopyalanmıştır veya runtime/GPU değişmiştir. İlgili `*.engine` dosyasını silip aynı komutu tekrar çalıştırın. `--force` tek başına engine rebuild yapmaz.

### `no kernel image` / parser CUDA mimarisi hatası

Hedef GPU compute capability 8.6 değildir. Parser hedef SM için yeniden derlenmelidir; yalnız TensorRT engine'i silmek bunu çözmez.

### `h264_nvenc` bulunamadı

Host FFmpeg NVIDIA encoder desteği olmadan kurulmuştur veya driver görünmüyordur:

```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep h264_nvenc
nvidia-smi
```

### `required file is absent`

ONNX, receipt veya parity dosyası atlanmıştır. `MODEL_DOSYALARI.sha256` ile dosya listesini ve hash'leri doğrulayın. Person ONNX için `.onnx.data` sidecar'ını unutmayın.

### Koşu tamamlanıyor fakat işlenmiş MP4 yok

`validation.run_ppe_deepstream` ham inference runner'ıdır. MP4 için `video_selector.process_ppe_selection` tam hattını çalıştırın.

## 14. Devir kabul kontrol listesi

- [ ] `nvidia-smi` GPU'yu ve driver'ı gösteriyor.
- [ ] GPU compute capability kaydedildi.
- [ ] Docker çalışıyor ve `nvidia` runtime görünüyor.
- [ ] Custom DeepStream image doğru tag ile yüklü.
- [ ] Container içinden `nvidia-smi` çalışıyor.
- [ ] Image label'ları `9.0.0 / 13.1 / 10.14.1.48`.
- [ ] `MODEL_DOSYALARI.sha256` doğrulaması geçti.
- [ ] Kaynak makinenin `*.engine` dosyaları hedefte yok.
- [ ] Python 3.12 sanal ortamı ve requirements kuruldu.
- [ ] `h264_nvenc` host FFmpeg içinde görünüyor.
- [ ] 5 saniyelik ham 960 smoke koşusu `complete`.
- [ ] Engine load attestation `pass`.
- [ ] Çekirdek testler geçti.
- [ ] Tam hat gerekiyorsa tek S01 videosu işlenmiş MP4 üretti.

Bu maddeler tamamlandığında mevcut PPE modeli ve ona bağlı insan-bazlı video hattı yeni bilgisayarda devralınmış olur.
