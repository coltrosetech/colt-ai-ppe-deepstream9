# SafetyVision YOLOv8s PPE checkpoint karantinası R2

Bu kayıt bir model kabulü veya çalıştırma makbuzu değildir. SafetyVision v2
deposunun exact `56a71758b55f0e9f2b4b2d6b51a779a1f882da10` commit'indeki
`best.pt`, sağlayıcının LFS SHA-256 değeriyle eşleşecek şekilde salt-okunur
karantinaya alındı. Checkpoint hiçbir Python model runtime'ı ile açılmadı;
`torch.load`, Ultralytics, export, inference, GPU, TensorRT ve DeepStream
çalıştırılmadı.

## Exact checkpoint

- Yol: `data/raw/ppe/models/safetyvision-yolov8-v2-56a7175/best.pt`
- Boyut: 22.547.434 byte
- SHA-256:
  `7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a`
- Biçim: PyTorch ZIP/pickle checkpoint
- Lisans kaydı: AGPL-3.0

## Statik arşiv incelemesi

Salt-okunur auditor, ZIP merkezi dizinini ve pickle opcode akışını model
nesnelerini oluşturmadan tekrar oynattı:

- 365 ZIP üyesi, 22.498.704 byte toplam açılmış içerik
- tekrar eden üye yok, güvensiz göreli/absolute üye yolu yok
- ZIP CRC testi başarılı
- `best/data.pkl`: 114.735 byte, 40.038 opcode
- 24 benzersiz `GLOBAL` referansı, 3.799 `REDUCE`, sıfır extension opcode

Global referansların gözlenen Torch ve Ultralytics YOLOv8 sınıflarıyla sınırlı
olması checkpoint'i güvenli ilan etmez. Pickle deserialization kod yürütme
yüzeyidir; bu envanter yalnızca bir sonraki izole çalıştırma planını daraltır.

## Mevcut ortam farkı

Checkpoint metadata'sı `ultralytics==8.4.51` bildirirken mevcut CPU export
ortamında `ultralytics==8.4.0` gözlendi. Bu nedenle mevcut ortam henüz exact
export runtime'ı olarak kabul edilmedi. Bir sonraki ardıl kayıt şunları birlikte
pinlemeden checkpoint açılmayacaktır:

- CPU-only container/image kimliği ve Python paket dosya envanteri
- network kapalı, salt-okunur root, GPU/device mount bulunmayan Docker argv'si
- exact checkpoint ve export entrypoint SHA-256 değerleri
- yalnız yeni ve boş bir output dizinine no-overwrite yayın
- 640 ve exact 960, dinamik batch `min=1`, `opt=12`, `max=12`
- opset 18, FP32, graph içinde NMS yok
- ONNX checker, tensor şekli/sınıf metadata replay'i ve terminal makbuz

CPU export başarıyla tamamlansa bile bu yalnızca aday ONNX üretir. Saha videosu
kalibrasyonu, parser/parite, TensorRT motoru, DeepStream 9.1 ve özellikle üst
kamera/oklüzyon testleri tamamlanmadan ürün veya PPE doğruluğu kabulü yoktur.
