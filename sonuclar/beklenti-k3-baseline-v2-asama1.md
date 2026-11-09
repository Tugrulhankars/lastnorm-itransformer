# Beklenti Kapısı — kuyruk `k3-baseline-v2` aşama 1 → **GEÇTİ**
*2026-10-14 01:59* · metrik `val_mse_avg` (min) · beklenti: `k3-baseline-v2.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline-duman | 1 | 0.1362 | 0.1362 | 38.3 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | baseline-duman: ort 0.1362 ∈ [0.05, 0.26] (Yalnız h96 (val_mse_avg = val_mse_h96). Üst 0.26 ≈ 2 × Repeat-last val MSE h96 0.1282 (data/naive_val.csv): iTransformer use_norm=True ile pencere-ortalaması etrafında başlar; 1 epoch sonra naif tekrarın 2 katından kötüyse öğrenmiyor/kurulum hatası. Alt 0.05 ≈ 0.6 × literatür TEST h96 0.086 (Liu et al. 2024 Tab.10; val testten daha kaymalı, data_manifest.md) → altı sızıntı/protokol şüphesi.) |
