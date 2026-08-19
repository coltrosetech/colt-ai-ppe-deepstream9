# DeepStream 9 parser bootstrap ve GPU smoke runbook

Bu runbook, upstream DeepStream-Yolo commit'inin DeepStream 9 uyumluluğunu
varsaymaz. Üretim image'ı ancak iki geçişli parser build, GPU'suz static/ABI
probe ve gerçek RTX A5000 Laptop GPU üzerinde beş raw-replayable smoke
kontrolünden sonra kullanılabilir.

Uygulama ve şemalar:

- `validation/ds9_parser_bootstrap.py`
- `validation/ds9_gpu_smoke.py`
- `validation/schemas/ds9-parser-bootstrap-v1.schema.json`
- `validation/schemas/ds9-gpu-smoke-authorization-v1.schema.json`
- `validation/schemas/ds9-gpu-smoke-evidence-v1.schema.json`
- `validation/schemas/ds9-cuda-kernel-proof-v1.schema.json`
- `deepstream/patches/deepstream-yolo-ds9-cuda-kernel-proof.patch`

## Değişmez güvenlik sırası

1. NGC/registry'den gerçek DeepStream 9 base manifest digest'i alınır.
2. Parser pass 1, beklenen parser SHA kabul etmeden ELF'i derler ve dışarı
   çıkarır. Upstream commit/tree yanında patch SHA, upstream CUDA kaynak SHA,
   patch sonrası kaynak SHA ve patch sonrası Git tree exact doğrulanır.
   Upstream Makefile değiştirilmez; SHA-256'sı pinlenir ve make command-line
   override'ı dört CUDA translation unit'i hem `sm_86` cubin hem `compute_86`
   PTX ile derler. Link sonrası ELF, SHA ölçümünden önce exact pinned GNU
   Binutils 2.42 `strip --strip-unneeded` ile canonicalize edilir. Controller
   SHA'yı doğrudan canonical binary'den ölçer.
3. Parser pass 2, yalnız mode `0440` ve tek hard-linkli pass-1 receipt'indeki
   SHA ile `--no-cache` tekrar derler. Aynı binary SHA üretilemezse image build
   durur.
4. Immutable production-candidate image ID üzerinde GPU'suz static probe
   tamamlanır; receipt `pending_gpu_smoke` kalır.
5. Operatör exact image ID, GPU UUID, contract SHA, static receipt SHA ve parser
   build receipt SHA için en fazla 24 saatlik tek-kullanımlık yetki verir.
6. `--execute`, re-entry evidence'i ve static receipt'i nonce claim'inden önce
   yeniden doğrular. Dört DeepStream işi ortak GPU guard arkasında çalışır.
7. Host ham DeepStream loglarını ve KITTI dosyalarını yeniden hesaplar. Beş
   check geçmezse `status=pass` üretilemez.
8. Passing smoke evidence ile static probe yeni, boş `current` dizininde tekrar
   çalıştırılır ve en fazla 24 saatlik production receipt üretir.

Bu sıra tersine çevrilemez. Static aday receipt'i genel Scene/CAVIAR/LOAF
işlerini açmaz; ortak guard'ın `static_candidate_smoke` modu yalnız exact smoke
worker komutunu kabul eder.

## Pass 0: gerçek base digest ve inert plan

`sha256:<64-hex>` değeri registry manifestinden gelmelidir. Controller tag-only,
`<registry-digest>`, all-zero/all-`f` ve düşük çeşitliliğe sahip placeholder
değerleri reddeder.

```bash
.venv/bin/python -m validation.ds9_parser_bootstrap \
  --dry-run \
  --base-ref 'nvcr.io/nvidia/deepstream:9.0-triton-multiarch@sha256:GERCEK_64_HEX' \
  --image-tag deepsafe-deepstream:9.0 \
  --session-root validation/results/ds9-runtime-compatibility/parser-bootstrap/SESSION_NONCE
```

Bu yol yalnız plan ve canlı source pinlerini yazar. Docker, GPU, `nvidia-smi`
ve inference çağrılmaz. Plan, `.dockerignore` SHA'sını da build input olarak
taşır; 24 GB veri/model/evidence dizinleri build context'e giremez.

## Pass 1: SHA keşfi

```bash
.venv/bin/python -m validation.ds9_parser_bootstrap \
  --execute-discovery \
  --base-ref 'nvcr.io/nvidia/deepstream:9.0-triton-multiarch@sha256:GERCEK_64_HEX' \
  --image-tag deepsafe-deepstream:9.0 \
  --session-root validation/results/ds9-runtime-compatibility/parser-bootstrap/SESSION_NONCE
```

Exact Docker target `parser-audit-export`'tur. Pass-1 komutunda
`DEEPSTREAM_YOLO_PARSER_SHA256` bulunmaz. Sabit commit/tree checkout'una
source-pinned kernel-proof patch'i uygulandıktan sonra exact komut şudur:

```text
nice -n 10 make -C nvdsinfer_custom_impl_Yolo -j2 CUDA_VER=13.1 'CUFLAGS=-I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-13.1/include -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86'
```

`CUDA_VER` ve `CUFLAGS` make command-line assignment olduğundan pinned
Makefile'daki `?=`/`:=` tanımlarını deterministik olarak override eder; iki
include yolu korunur. Makefile SHA
`fd2c03b810b8dae9d9d3a60b503616bbf6ed67a6f614843dd6a29f7f87ff8ad0`,
komut SHA ise
`e244df2d9c424fe7d027d62205ff21c820b58eb2cc00aa61df2d32cbfe329ac1`'dir.
Derlemeden sonra ve parser SHA hesaplanmadan önce çalışan exact post-link
komutu şudur:

```text
/usr/bin/x86_64-linux-gnu-strip --strip-unneeded nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

Tool binary SHA-256'sı
`4dad0d12aa5d6a49b117b4551b897175ad5b43b9525e8f9efd661133a1c8ea0d`,
exact version satırı `GNU strip (GNU Binutils for Ubuntu) 2.42`, komut SHA'sı
ise `5e19627d403e984d9e349f8c81332f7280a1ac8477d56056c9b27be63aeef7ca`'dır.
Build, `.symtab` ve `.strtab` kaldıysa veya `.dynsym` ve `.dynstr` tam birer
tane değilse fail-closed durur. Böylece geçici build dizini kaynaklı local
symbol farkları pass-1/pass-2 parser digest'ine giremez; dinamik export ve
loader tabloları korunur.

Export şu üç dosyayı içerir:

- parser ELF;
- ELF'ten ölçülmüş SHA-256;
- repository/commit/upstream tree, patch SHA, upstream ve patch sonrası kaynak
  SHA'ları, patch sonrası tree, değiştirilmemiş Makefile pini, exact cubin/PTX
  architecture ve codegen flag'leri ile exact build komutunu taşıyan
  pinned post-link tool/komut/section contract'i de dahil source-lineage v3
  JSON.

Parser build'in varsayılan ve güncel işletim politikası
`workstation_managed`'dır: donanım korumasının sahibi workstation BIOS/EC/GPU
driver zinciridir ve controller sabit bir sıcaklık eşiği uygulamaz. Controller
saniyede bir platform/GPU sıcaklığı ile GPU güç limiti, güç çekişi, pstate ve
slowdown flag'lerini kaydeder; yüksek sıcaklık, aktif slowdown veya güç-limit
farkı kalite teşhisidir ve tek başına build'i bloklamaz. Telemetry kaybı,
okunamayan/bozuk alan veya örnekler arasında GPU kimliği değişimi fail-closed
kalır: çalışan Docker CLI/build process group'una önce `SIGTERM`, gerekirse
`SIGKILL` gönderilir. GPU sorgusu salt okunur host telemetry sorgusudur; Docker
build'e GPU DeviceRequest verilmez ve inference başlatılmaz.

Başarılı çıktı:

```text
.../pass-1-discovery/discovery-receipt.json
```

Receipt exclusive mode `0440` oluşturulur; binary, digest, source lineage,
raw build log ve platform/GPU build-telemetry raporunu hash-pinler.

## Pass 2: ölçülen SHA ile deterministic production build

Parser SHA komut satırından verilmez:

```bash
.venv/bin/python -m validation.ds9_parser_bootstrap \
  --execute-production \
  --image-tag deepsafe-deepstream:9.0 \
  --discovery-receipt \
    validation/results/ds9-runtime-compatibility/parser-bootstrap/SESSION_NONCE/pass-1-discovery/discovery-receipt.json
```

Controller pass-1 receipt freshness/mode/link'i ile bütün raw pinleri yeniden
doğrular. `runtime` target'i `--no-cache` derlenir ve ölçülen SHA'yı build-time
exact karşılaştırır. Docker'ın `--iidfile` immutable image ID'si ile raw
`docker image inspect` çıktısı receipt'e bağlanır. Image lineage ayrıca
`.dockerignore`, runtime controller ve runtime-control-manifest SHA'larını
taşır.

Başarılı çıktı:

```text
.../pass-2-production/production-build-receipt.json
```

## GPU'suz static candidate

Yeni ve boş bir output dizini kullanılır:

```bash
.venv/bin/python -m validation.ds9_runtime_compatibility \
  --execute-static-probe \
  --image deepsafe-deepstream:9.0 \
  --output-dir validation/results/ds9-runtime-compatibility/candidate
```

Beklenen exit kodu `3`, status `pending_gpu_smoke` ve
`production_ready=false` değeridir. Static aday, generic tail capture
kullanmaz: tam stdout'u 16 KiB üst sınırla yakalanmış exact
`cuobjdump --list-elf` ve `--list-ptx` kayıtlarında indeksleri `1..4` olan dört
`sm_86` cubin ile dört `compute_86` PTX kaydının tamamını gerektirir. Bilinmeyen
nonblank satır, mixed/unknown architecture, eksik/fazla/duplicate indeks,
yanlış dosya adı, decode veya byte-limit sorunu fail-closed'dur. GPU smoke aynı
exact cubin koşulunu defense-in-depth olarak tekrar doğrular.

Static candidate nvinfer'i runtime'da yüklemeye çalışmaz: sürücü mount'u
olmayan container'da DS9 plugin'in `libcuda.so.1` bağı nedeniyle
`gst-inspect-1.0` geçerli bir GPU-free kontrol değildir. Bunun yerine exact
versioned path, pinned binary SHA, SDK version manifest, ELF kimliği/full
`DT_NEEDED` ve GStreamer descriptor sembolleri ham komut çıktısından yeniden
doğrulanır. Receipt `static_binary_metadata_only` kapsamını taşır; gerçek
plugin load/engine/CUDA kanıtı aşağıdaki guarded smoke'a aittir.

Static parser probe ayrıca tam ve bounded `readelf --section-headers --wide`
çıktısından post-link section contract'ini yeniden hesaplar: `.symtab` ile
`.strtab` yok, `.dynsym` ile `.dynstr` tam birer tane olmalıdır. Image label,
build-lineage v3 manifest'i ve raw ELF section tablosundan herhangi biri
farklıysa aday reddedilir.

Parser export kontrolü de bounded fakat truncation'sızdır: exact `nm` komutu
tam 221132-byte stdout üzerinden Python içinde parse edilir, original
byte/SHA-256 metadatası receipt'e bağlanır ve yalnız iki gerekli export satırı
saklanır. Son 128 KiB'ı alma yaklaşımı kullanılmaz; missing/duplicate sembol,
incomplete projection ve argv/path substitution static candidate'i reddeder.

## Smoke planı ve operatör yetkisi

64-hex nonce, final session dizininin basename'idir. GPU UUID açıkça yazılır:

```bash
.venv/bin/python -m validation.ds9_gpu_smoke \
  --session-root validation/results/ds9-runtime-compatibility/gpu-smoke/sessions/64_HEX_NONCE \
  --gpu-index 0 \
  --gpu-uuid GPU-8cbaba1c-2629-a732-f528-66f459089ef6 \
  --static-candidate-receipt validation/results/ds9-runtime-compatibility/candidate/receipt.json \
  --parser-build-receipt validation/results/ds9-runtime-compatibility/parser-bootstrap/SESSION_NONCE/pass-2-production/production-build-receipt.json
```

İlk planın tek authorization blocker'ı `operator_authorization_missing`
olmalıdır. Plan içindeki `definition_sha256` kullanılarak mode `0440`, tek
hard-linkli JSON hazırlanır:

```json
{
  "schema_version": "deepsafe.ds9-gpu-smoke-authorization/v1",
  "status": "approved",
  "operator_identity": "GERCEK_OPERATOR_KIMLIGI",
  "campaign_nonce": "64_HEX_NONCE",
  "session_id": "ds9-gpu-smoke-64_HEX_NONCE",
  "authorized_session_root": "validation/results/ds9-runtime-compatibility/gpu-smoke/sessions/64_HEX_NONCE",
  "issued_at_utc": "RFC3339",
  "expires_at_utc": "RFC3339_MAX_24_SAAT",
  "resolved_image_id": "sha256:ADAY_IMAGE_ID",
  "smoke_definition_sha256": "PLANDAKI_64_HEX",
  "static_candidate_receipt_sha256": "PLANDAKI_STATIC_RECEIPT_SHA",
  "parser_production_build_receipt_sha256": "PLANDAKI_BUILD_RECEIPT_SHA",
  "gpu_index": 0,
  "gpu_uuid": "GPU-8cbaba1c-2629-a732-f528-66f459089ef6",
  "approved_checks": [
    "cuda_parser_kernel_launch_sm86",
    "deepstream_640_engine_deserialize_no_fallback",
    "deepstream_960_engine_deserialize_no_fallback",
    "cpu_cuda_parser_parity_640",
    "cpu_cuda_parser_parity_960"
  ],
  "single_use": true
}
```

Authorization ile plan tekrar oluşturulur; status
`ready_for_authorized_execution` ve blocker listesi boş olmalıdır.

## Açık ve guarded GPU execution

```bash
.venv/bin/python -m validation.ds9_gpu_smoke \
  --execute \
  --session-root validation/results/ds9-runtime-compatibility/gpu-smoke/sessions/64_HEX_NONCE \
  --gpu-index 0 \
  --gpu-uuid GPU-8cbaba1c-2629-a732-f528-66f459089ef6 \
  --authorization validation/authorizations/ds9-gpu-smoke-64_HEX_NONCE.json \
  --reentry-evidence validation/results/gpu-reentry/current/evidence.json \
  --static-candidate-receipt validation/results/ds9-runtime-compatibility/candidate/receipt.json \
  --parser-build-receipt validation/results/ds9-runtime-compatibility/parser-bootstrap/SESSION_NONCE/pass-2-production/production-build-receipt.json
```

Docker komutu mutable tag'i çalıştırmaz; planın exact
`sha256:<immutable-image-id>` değerini `--pull=never --network=none` ile
çalıştırır. Preflight requested tag'i bağımsız çözer; tag başka image'a
retarget edilmişse, live container inspect ID farklıysa veya static receipt ID
eşleşmezse `Popen` öncesi/sonrası fail-closed olur.

Container root veya ek capability ile çalışmaz. Plan ve probe contract,
hostun effective UID/GID değerini `container_process_identity` alanına bağlar;
Docker komutu aynı kimliği exact `--user UID:GID` ile kullanırken
`--cap-drop ALL` korunur. Worker da kendi runtime UID/GID'sini contract ile
karşılaştırır. Böylece host sahibine ait mode `0440` contract/config dosyaları
okunabilir kalır; `CAP_DAC_OVERRIDE` eklenmez ve dosyalar world-readable yapılmaz.

Her 640/960 profilinde iki engine-only config çalışır:

- `NvDsInferParseYoloCuda`;
- `NvDsInferParseYolo`.

Config'lerde `onnx-file` yoktur; engine deserialize başarısızsa ONNX rebuild
fallback mümkün değildir. Dört DeepStream process'inin her birinde
`/dev/nvidiaN`, `/dev/nvidiactl` ve `/dev/nvidia-uvm` FD'leri canlı olarak
gözlenmelidir. Ham loglarda iki deserialize başarı izi, exact engine yolu,
sıfır fallback/build/CUDA error izi gerekir.

FD gözetimi her örneği process PID'si, exact `/proc/<pid>/fd` kökü, monotonik
başlangıç/bitiş zamanı, o örnekte görülen NVIDIA aygıtları ve yapılandırılmış
okuma hatalarıyla kaydeder. Bir FD'nin dizin snapshot'ı ile `readlink` arasında
kapanmasından doğan entry-level `ENOENT` olağan yarış olarak hiçbir FD kanıtı
üretmeden atlanır; diğer entry-level hatalar fail-closed'dur. Process kapanırken
kök `/proc/<aynı-pid>/fd` okumasının `EACCES` veya `ENOENT` vermesi ise yalnızca
zorunlu üç FD sınıfının daha önce eksiksiz görülmüş olması, daha önce hiçbir
okuma hatası bulunmaması, hatanın son örnek olması ve aynı process'in monotonik
olarak ölçülen exact `0.5 s` (`500000000 ns`) içinde `returncode=0` ile çıkması
halinde terminal teardown olarak kabul edilir. Root yolu/PID'si farklıysa,
grace aşılırsa, çıkış sıfır değilse veya başka bir errno/hata varsa FD gate'i
başarısız olur. Worker terminal örneği ve çıkış zamanını ham evidence'a yazar;
host replay örnek sırasını, zaman farkını, prior FD/error kümesini ve contract'a
bağlı terminal politikasını yeniden hesaplayıp exact eşitlik arar.

CUDA parser process'lerinde worker proof environment değerlerini yalnız
`640-cuda` ve `960-cuda` koşularına verir; CPU koşularından bunları siler.
Bu iki CUDA koşusunda `CUDA_DISABLE_PTX_JIT=1` de zorunludur; böylece marker
kanıtı embedded `sm_86` cubin yolundan gelir, yanında taşınan PTX JIT fallback
olarak kullanılamaz.
Patch, `decodeTensorYoloCuda<<<...>>>` satırının hemen ardından smoke modunda
sırasıyla `cudaGetLastError`, `cudaDeviceSynchronize` ve
`cudaFuncGetAttributes(decodeTensorYoloCuda)` çağırır. `binaryVersion` exact
`86` değilse parser başarısız olur; host replay hem `binaryVersion` hem
`ptxVersion` değerini exact `86` ister. Her CUDA process'i mutex ile korunan tek
bir `O_CREAT|O_EXCL|O_NOFOLLOW`, mode `0440`, tek-hard-link JSON marker üretir.
Marker campaign nonce, run ID, gerçek DeepStream PID, kernel adı, launch boyutu
ve üç CUDA dönüş koduna bağlıdır. Tekrarlanan veya paralel parser çağrıları
ikinci marker yazamaz.

CPU/CUDA KITTI çıktılarında frame seti ve detection sayısı eşit; her detection
aynı sınıfta, bbox mutlak farkı `<=0.25 px`, confidence farkı `<=1e-4` ve IoU
`>=0.999` olmalıdır. Her koşuda en az bir detection gerekir. Host bu değerleri
raw dosyalardan yeniden hesaplar. Kernel check'i ayrıca iki marker'ı no-follow
FD üzerinden boyut sınırlı okuyup pinler; eksik/fazla marker, CPU marker'ı,
duplicate JSON key, yanlış PID/nonce/run/binaryVersion/count veya mode/link
farkını reddeder. NVIDIA FD'leri, detection varlığı, parser config adı ve temiz
CUDA logu kernel lansman kanıtı sayılmaz. Raw loglar ile KITTI dosya/adet/ağaçları
da sabit üst sınırlarla replay edilir.

Passing evidence:

```text
.../gpu-smoke-evidence.json
```

Evidence; authorization, single-use claim, probe contract, static receipt,
parser build receipt, re-entry evidence, guard report/receipt, GPU identity,
dört raw log ve tüm KITTI dosyalarını pinler. Guard requested/executed command,
container name, requested tag metadata, exact image ID ve running-container ID
alanları yeniden çapraz doğrulanır.

## Production receipt

Passing evidence ile yeni, boş `current` dizini oluşturulur:

```bash
.venv/bin/python -m validation.ds9_runtime_compatibility \
  --execute-static-probe \
  --image deepsafe-deepstream:9.0 \
  --gpu-smoke-evidence validation/results/ds9-runtime-compatibility/gpu-smoke/sessions/64_HEX_NONCE/gpu-smoke-evidence.json \
  --output-dir validation/results/ds9-runtime-compatibility/current
```

Controller beş `pass` stringini güvenilir saymaz; raw evidence'i yeniden
oynatır. Yalnız tüm semantic doğrulamalar geçerse receipt
`production_ready=true` olur. Receipt en fazla 24 saat geçerlidir.
