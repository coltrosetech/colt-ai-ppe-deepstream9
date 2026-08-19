# GitHub teslim notu

Bu public GitHub teslimi, ekip arkadaşına kaynak kodu ve model artefaktlarını
taşımak için hazırlanmıştır.

## Release ile gelenler

- Kaynak kod, test, yapılandırma ve teknik dokümanlar
- SafetyVision PPE checkpoint'i ve 640/960 ONNX türevleri
- YOLO11s kişi modeli 640/960 ONNX dosyaları ve sidecar'ları
- Model/adaptör makbuzları ve SHA-256 listeleri

## Bilerek dışarıda bırakılanlar

- Kullanıcı tarafından sağlanmış veya yeniden dağıtım hakkı doğrulanmamış video
- Poster/kare görselleri ve kayıtlı kullanıcı seçim kuyrukları
- GPU/TensorRT'ye özgü `.engine` dosyaları
- Hosta sabitlenmiş deneysel DeepStream 9.1 engine-builder kanıtları
- Yerel `.env`, sanal ortamlar, cache ve üretilmiş çalışma sonuçları
- 26.3 GB sanal boyutlu NVIDIA DeepStream tabanlı Docker image; hedefte NGC
  tabanından yeniden kurulur veya workstation'dan doğrudan save/load ile taşınır

Bu nedenle eski teknik belgelerde örnek olarak geçen `S01`–`S04` veya
`data/samples/ppe-construction-2025-h264.mp4` yolları GitHub arşivinde hazır
gelmez. Hedef ekip, kullanma yetkisi bulunan videosunu repo içine yerleştirip
aynı CLI'larda `--video` değerini o yola göre değiştirmelidir.

Aktif çalışma hattı DeepStream 9.0'dır. TensorRT engine'leri ilk GPU koşusunda
hedef makinede yeniden üretilir.
