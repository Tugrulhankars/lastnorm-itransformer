# Replikasyon Kapısı — YEŞİL
*2026-10-14 02:19* · metrik `test_mse_avg` (düşük iyi)

| | değer |
|---|---|
| Literatür değeri/aralığı | 0.358 – 0.36 |
| Kaynak | Liu et al. 2024 iTransformer (ICLR, arXiv:2310.06625) Tab.10 ort. 0.360 + Tab.5 5-seed ort. 0.358 (hesap); Qiu et al. 2025 DUET (KDD, arXiv:2412.10859) Tab.3 TFB-reprod. 0.360 (hesap); Exchange, L=96, 7:1:2, StandardScaler(train) ölçekli MSE, test drop_last yok |
| Tolerans (mutlak) | ±0.008 → kabul aralığı [0.35, 0.368] |
| Bizim baseline (baseline) | **0.3646 ± 0.000766** (n=3 seed: 2023, 2024, 2025) |
| Seed değerleri | 0.3643, 0.3641, 0.3655 |

**Karar:** YEŞİL — Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir.

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).
