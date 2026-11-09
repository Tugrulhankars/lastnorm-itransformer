# Beklenti Kapısı — kuyruk `k4t1-h03` aşama 1 → **GEÇTİ**
*2026-10-14 03:09* · metrik `val_mse_h336` (min) · beklenti: `k4t1-h03.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline-ep336eq-duman | 1 | 0.4004 | 0.4004 | 38.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | baseline-ep336eq-duman: ort 0.4004 ∈ [0.4004357589950561, 0.4004377589950561] (H03 kartı Bütçe + Uygulama notu: duman val_mse_h336 == ebeveyn s2023 0.40043676, |fark| < 1e-6 (değer: 20261014-011230_baseline-onucus_s2023 val_mse_h336 = 0.40043675899505615; h336 resmî 1 epoch olduğundan ön-uçuş = tam koşum değeri; KURULUM §5: aynı seed tekrarı bit-bit aynı)) |
