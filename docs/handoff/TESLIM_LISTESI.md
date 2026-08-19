# Ekip arkadaşına gönderilecekler

## Zorunlu

1. Public GitHub reposu: `coltrosetech/colt-ai-ppe-deepstream9`
2. Release: `COLT-AI-PPE-RUNTIME-GITHUB-20260819.tar.zst`
3. Release: `COLT-AI-PPE-TEKNIK-DEVIR-GITHUB-20260819.zip`
4. NVIDIA NGC erişimi veya aynı workstation'dan doğrudan `docker save/load`
5. Ekip tarafından ayrıca sağlanacak, kullanma yetkili PPE videosu

Önerilen hedef kurulumu:

1. NGC tabanını çekin.
2. Teknik README'nin 7.2 bölümündeki exact komutla image'ı yerelde kurun.
3. Alternatif olarak 7.1'deki `docker save/load` yoluyla workstation'dan taşıyın.

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
