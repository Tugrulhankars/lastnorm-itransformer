# san — H01 (SAN: dilim düzeyi uyarlamalı normalizasyon)

- **Kart:** `hipotezler/H01.md` (A7; Z1/Z3).
- **Ebeveyn:** `baseline` (resmî iTransformer `c2426e6`, Exchange `.sh` birebir, h336 = 1 epoch). Koşumlar:
  `20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`.
- **Tek değişiklik:** `--use_norm 0`, yerine SAN geliyor (SAN@7e1ca66 resmî Exchange satırı: `period_len=6`,
  `station_lr=1e-3`, `station_type=adaptive`, 5 epoch istatistik ön-eğitimi, ardından omurga eğitimi). Omurga, veri,
  split, seed, lr, epoch, patience ve lradj ebeveynle aynı.
- **Dosyalar:**
  - `san_modul.py`: `Statistics_prediction` ve `MLP`. ustc-time-series/SAN@7e1ca66 `models/Statistics_prediction.py`
    dosyasından port edildi (Apache-2.0). Orijinaliyle `diff` alındı: hesap satırları birebir aynı, yalnız config
    geçirme biçimi farklı.
  - `model_san.py`: `SANiTransformer`. Sıra: normalize → resmî omurga (`use_norm=0`) → de_normalize. Önce omurga
    kurulur (aynı seed'de ilk ağırlıklar ebeveynle aynı olsun diye), sonra SAN. SAN'da sıra tersti; bu yalnız
    RNG sırasını değiştirir.
  - `exp_san.py`: `ExpSAN`. Eğitim mantığı SAN `exp/exp_main.py`'deki `train`/`vali`/`station_loss` ile aynı. İlk 5
    epoch'ta yalnız `station_optim` adım atar. Sonra en iyi istasyon checkpoint'i yüklenir ve **yalnız
    `model_optim` adım atar**: SAN kodu omurga fazında istatistik modülünü güncellemez (kartın kısa uygulama notunda
    "+ station_optim" yazıyor, ama üst tablo SAN@7e1ca66 `exp_main.py`'ye atıf yapıyor ve o kod bu şekilde
    çalışıyor). Ön-eğitim satırları resmî biçimde değil `[SAN-onegitim] ep N ...` biçiminde basılır. Böylece
    `progress.json`'a yalnız omurga epoch'ları girer ve kosum `--kural val_loss<0.01` ile çakışma olmaz.
  - `kos.py`: sarmalayıcı. `--kapi-test-metrics` reddedilir. `metrics.json`'a eklenenler: `san_onegitim_*_h{H}`,
    `san_param_sayisi_h{H}`, `omurga_param_sayisi_h{H}`.

## DURUM: ön-uçuş E (yön) sağlaması GEÇMEDİ → kuyruğa EKLENMEDİ (2026-10-14)
- Duman `20261014-025708_san-duman_s2023`: val_mse_h96 0.158678 (kart eşiği ≤ 0.1923, geçti).
- Ön-uçuş `20261014-025830_san-onucus_s2023`: val_mse_avg 0.582920 (kart aralığı [0.333, 0.60], geçti). Aynı seed ve
  aynı epoch-üst ile ebeveyn ön-uçuşuna göre fark **+30.9%**. Fark 4/4 ufukta ters yönde: +25.5 / +43.1 / +23.8 / +31.8%.
- **Teşhis:** SAN std başlığının son `ReLU`'su ölüyor. Ön-uçuş checkpoint'inde (h96, h336) train ve val
  pencerelerinin **%100**'ünde tahmin edilen std tam 0. Tarama (3 seed × 4 ufuk, yalnız train/val) aynı tabloyu
  veriyor: başlangıçta %38–60 sıfır, ön-eğitimin 1. epoch'undan sonra %89–100 sıfır. Tahmin std ortalaması
  0–0.006, gerçek dilim std'si ise ~0.04–0.05 (z-uzayı). Bu yüzden `de_normalize` = `y_n·(0+1e-5)+μ̂`: omurganın
  katkısı 1e-5 ile ölçekleniyor. Model fiilen yalnız SAN'ın ortalama-MLP'si ve dilim başına sabit tahmin üretiyor.
  Bunun izi omurganın train kaybında görülüyor: epoch 1→2 arasında 0.1181376 → 0.1181375.
  Port hesapça birebir, yani bu SAN@7e1ca66 ayarlarının bu veride verdiği davranış.
- Teşhis betikleri `teshis/` klasöründe. İkisi de yalnız train/val okur:
  - `teshis_checkpoint.py <WS> <run_dir>`: ön-uçuş checkpoint'inde std=0 oranını ve omurga gradyanını ölçer.
  - `olu_relu_tarama.py <WS>`: 3 seed × 4 ufuk için istasyon ön-eğitimini taklit eder (GPU'da ~1 dk).

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/san/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 96 --epoch-ust 1 --num-workers 0   # duman
$PY varyantlar/san/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0               # ön-uçuş
$PY varyantlar/san/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                              # tam (şef kararıyla)
```
