# Beklenti Kapısı — kuyruk `k4t3-h08` aşama 1 → **GEÇTİ**
*2026-10-14 07:24* · metrik `val_mse_h336` (min) · beklenti: `k4t3-h08.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| lastnorm-ep336eq-duman | 1 | 0.395 | 0.395 | 35.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | lastnorm-ep336eq-duman: ort 0.395 ∈ [0.3949739767780304, 0.3949759767780304] (H08 kartı Bütçe + Mekanik DUR + uygulama notu: duman val_mse_h336 == 20261014-052508_lastnorm_s2023 val_mse_h336 0.3949749767780304, |fark| < 1e-6 (KURULUM §5: aynı seed tekrarı bit-bit aynı)) |
| 🟢 OK | ikincil_aralik | lastnorm-ep336eq-duman/s2023 param_sayisi=2.552e+05 ∈ [255184,255184] |
