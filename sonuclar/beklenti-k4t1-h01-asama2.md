# Beklenti Kapısı — kuyruk `k4t1-h01` aşama 2 → **GEÇTİ**
*2026-10-14 04:00* · metrik `val_mse_avg` (min) · beklenti: `k4t1-h01.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| san-onucus | 1 | 0.5829 | 0.5829 | 171.2 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | san-onucus: ort 0.5829 ∈ [0.333, 0.6] (H01 kartı uygulama notu: ön-uçuş val_mse_avg ∈ [0.333, 0.60]; Mekanik DUR 'ön-uçuş val_mse_avg > 0.60 (≈ 1.27 × Repeat 0.4730)') |
| 🟢 OK | ikincil_aralik | san-onucus/s2023 egitim_sure_sn=164.9 ∈ [10,700] |
