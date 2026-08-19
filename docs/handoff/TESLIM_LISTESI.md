# Ekip arkadaşına gönderilecekler

## Zorunlu

1. Public GitHub reposu: `coltrosetech/colt-ai-ppe-deepstream9`
2. Release: `COLT-AI-PPE-RUNTIME-GITHUB-20260819.tar.zst`
3. Release: `COLT-AI-PPE-TEKNIK-DEVIR-GITHUB-20260819.zip`
4. Container: `ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725`
5. Ekip tarafından ayrıca sağlanacak, kullanma yetkili PPE videosu

Container'ı hedefte yükleme:

```bash
docker login ghcr.io
docker pull ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725
docker tag \
  ghcr.io/coltrosetech/colt-ai-deepstream9-ppe:9.0-control-refresh-20260725 \
  deepsafe-deepstream:9.0-control-refresh-20260725
```

Image'ın Docker'daki sanal boyutu yaklaşık 26.3 GB'dir.

## Proje klasörünü doğrudan kopyalıyorsanız

Kaynak bilgisayarın aşağıdaki engine dosyalarını göndermeyin veya hedefte silin:

```text
models/ppe/safetyvision-yolov8s-v2/640/*.engine
models/ppe/safetyvision-yolov8s-v2/960/*.engine
models/person/640/*.engine
models/person/960/*.engine
```

Hedefte model engine'leri ilk GPU koşusunda yeniden üretilecektir.

## Hedefte ilk okunacak dosya

`README.md`

Hızlı kurulum için `HIZLI_KURULUM.md`, otomatik kontrol için:

```bash
./docs/handoff/ortam_kontrol.sh
```
