# LastNorm-iTransformer

**Makale:** *Normalizasyon Merkezini Son Gözleme Taşımak iTransformer'ı İyileştirir mi? Döviz Kurunda Kazanç, Diğer Veri Setlerinde Karışık Sonuç*. Yazar: Ufuk ASIL.

Bu depoda makaledeki deneylerin kodu, yapılandırmaları ve koşum kayıtları yer alır. Önerilen değişiklik **LastNorm** adını taşır. iTransformer'ın normalizasyonu merkez olarak pencere ortalamasını kullanır; LastNorm bunun yerine girdinin **son gözlemini** merkez alır. Ölçek (σ), omurga, eğitim ayarları ve seed'ler resmî iTransformer ile aynıdır.

## Klasörler

| Klasör | İçerik |
|---|---|
| `varyantlar/lastnorm/` | Önerilen yöntem: `model_lastnorm.py` (model), `kos.py` (koşum), `dogrula.py` (CPU sağlama testi) |
| `varyantlar/*` | Karşılaştırma ve ablasyon varyantları (`revin-afin`, `san`, `normsuz`, `lastnorm-esit`, `lastnorm-merkez`, `lastnorm-olcek`, …). Her birinin kendi `README.md` dosyası var |
| `scripts/` | Veri hazırlama (`veri_hazirla.py`, `veri_hazirla_ett.py`), rakip koşumu (`rakip_kos.py`), analiz ve makale tablo/figür betikleri (`makale_*.py`, `tahlil_tur*.py`) |
| `hipotezler/` | Koşumlardan önce yazılmış, ön-kayıtlı hipotez kartları |
| `kuyruk/` | Koşum yapılandırmaları (her satır bir koşum) ve beklenti dosyaları |
| `runs/` | Koşum kayıtları: `config.json`, `metrics.json`, loglar. Tahmin dosyaları (`preds/*.npz`) ve checkpoint'ler boyut nedeniyle depoda yok |
| `sonuclar/` | Toplu sonuç tabloları (CSV/MD). `sonuclar/makale/` makaledeki tablo ve şekillerin verisidir |
| `rakip/KURULUM.md` | Resmî iTransformer kodunun kurulumu (sabit commit) |
| `env/` | Ortam kurulumu ve sabitlenmiş paket listesi (`requirements.lock`) |

## Kurulum

1. **Python ortamı.** Python 3.11 ve PyTorch 2.4.1 (CUDA 12.1) kullanıldı. Ortam `env/.venv` altında beklenir:
   ```bash
   python -m venv env/.venv
   env/.venv/Scripts/python.exe -m pip install -r env/requirements.lock \
       --extra-index-url https://download.pytorch.org/whl/cu121          # Linux/macOS: env/.venv/bin/python
   ```
   `reformer-pytorch` paketini kurarken dikkat edilmesi gereken bir nokta var; ayrıntısı `rakip/KURULUM.md` §2'de.
2. **Resmî iTransformer kodu.** Kod hiç değiştirilmeden kullanılır:
   ```bash
   cd rakip && git clone https://github.com/thuml/iTransformer.git && cd iTransformer
   git checkout c2426e68ca13f74aaec08045c5c724d8ad328124
   ```
3. **Veri.** Veriler programatik olarak indirilir ve `data/` altına yazılır:
   ```bash
   env/.venv/Scripts/python.exe scripts/veri_hazirla.py       # Exchange-Rate
   env/.venv/Scripts/python.exe scripts/veri_hazirla_ett.py   # ETTh1/ETTh2/ETTm1/ETTm2
   ```

## Çalıştırma

Komutlar depo kökünden çalıştırılır:

```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/lastnorm/dogrula.py                                    # CPU sağlama testi (~10 s)
$PY varyantlar/lastnorm/kos.py --seed 2023 --out runs/lastnorm_s2023/metrics.json --num-workers 0
```

Makalede her yapılandırma 2023, 2024 ve 2025 seed'leriyle koşuldu. Diğer varyantlar da aynı biçimde çalışır; ayrıntılar her varyantın `README.md` dosyasında. `makale_*.py` betikleri `runs/*/preds/` altındaki tahmin dosyalarını okur. Bu dosyalar depoda bulunmadığından o betikleri çalıştırmak için önce koşumları yeniden üretmek gerekir.

## Veri ve lisans

- ETT (ETTh1, ETTh2, ETTm1, ETTm2): https://github.com/zhouhaoyi/ETDataset (CC BY-ND 4.0)
- Exchange-Rate: https://github.com/laiguokun/multivariate-time-series-data. Lisansı belirsiz olduğundan yalnız akademik amaçla kullanıldı.
- iTransformer: https://github.com/thuml/iTransformer (MIT)
