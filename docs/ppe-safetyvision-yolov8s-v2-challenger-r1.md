# SafetyVision YOLOv8s PPE challenger R1

Bu paket kabul edilmiş PPE modeli değildir. SafetyVision v2 deposunun exact
`56a71758b55f0e9f2b4b2d6b51a779a1f882da10` commit'inden iki ONNX dosyası
salt-okunur karantinaya alındı. Ağ, Docker, GPU, ONNX Runtime, TensorRT veya
DeepStream inference çalıştırılmadı.

## Neden challenger?

Modelin 13 sınıfı içinde ihtiyaç duyulan dört sinyal de açıkça bulunuyor:

| DeepSafe sinyali | Model sınıfı | ID |
|---|---|---:|
| `helmet` | `Hardhat` | 3 |
| `no_helmet` | `NO-Hardhat` | 7 |
| `hi_vis` | `Safety Vest` | 12 |
| `no_hi_vis` | `NO-Safety Vest` | 9 |

Model kartı, 640 ONNX için held-out test `mAP@50=0.738` ve
`mAP@50:95=0.463` bildiriyor. Bu sayılar yerelde yeniden üretilmedi. Model
yazarı `NO-Safety Vest` sınıfını en zayıf hedef olarak belgeliyor
(`mAP@50=0.386`, 217 test instance). Model kartı ayrıca drone/üst kamera
açılarını, yoğun kalabalık/oklüzyonu ve 50 pikselden küçük insanları
kapsam dışı sayıyor. Bu nedenle kullanıcının orta-yakın CCTV tipi için
erken aday olabilir, fakat üst açı kabulü için tek başına yeterli olamaz.

## Exact artefaktlar

- 640 ONNX: 44.764.727 byte,
  `ea18ae903a566e8fa76f3ee1c503075522dca269269315e9c862efa170430b35`
- 896 ONNX: 44.926.046 byte,
  `b250353639e01800f9cbe79c6002b8b041bdae7560328b8e18ad4a42dc3844e1`
- Model kartı: 15.629 byte,
  `da35935f220c0e348f9c4d770c4d2c78b7c6a1482fb7cdf51b5243d9444b2533`
- Karantina manifesti:
  `6b38eec658952fc5d2327625c1fc631df2ba160e1c64ca72cc2ce8a8e8254e30`

Her iki graph `onnx==1.22.0` ile statik `check_model` kontrolünden geçti;
opset 20, IR 9, 231 node, 144 initializer ve external-data sidecar yok. 640
girdisi sabit `[1,3,640,640]`, çıktısı `[1,17,8400]`; 896 girdisi sabit
`[1,3,896,896]`, çıktısı `[1,17,16464]`. Graph'larda NMS yoktur.

## Açık kapılar

- Lisans AGPL-3.0'dır; ticari/dağıtılan kullanım kararı ayrı kalır.
- Dataset provenance ve yazarın held-out split iddiası yerelde replay edilmedi.
- Dosyalar fixed batch 1'dir; 12-akış dynamic profile değildir.
- Exact 960 ONNX yoktur; 896 dosyası 960 kabulü yerine geçmez.
- YOLO raw-output parser/NMS paritesi, TensorRT ve DeepStream 9.1 çalışmadı.
- Construction-PPE tanısal karantinası ve kullanıcı saha videosu üzerinde
  hata analizi yapılmadan eşik veya ürün kabulü yoktur.

Bir sonraki adım, `.pt` checkpoint'i ayrı bir karantina kararıyla alıp
execution-closed CPU export hattında dynamic batch 1/12 için 640 ve 960 ONNX
üretmek veya önce fixed 640 dosyasını salt tanısal parser-parity challenger'ı
olarak sınamaktır.
