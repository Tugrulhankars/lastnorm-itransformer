# Beklenti Kapısı — kuyruk `k4t1-h02` aşama 2 → **GEÇTİ**
*2026-10-14 03:42* · metrik `val_mse_avg` (min) · beklenti: `k4t1-h02.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| revin-afin-onucus | 1 | 0.4438 | 0.4438 | 116.3 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | revin-afin-onucus: ort 0.4438 ∈ [0.4, 0.49] (H02 kartı Mekanik DUR + Uygulama notu: ön-uçuş val_mse_avg ∈ [0.40, 0.49] (ebeveyn ön-uçuşu 0.4453 ± %10)) |
| 🟢 OK | ikincil_aralik | revin-afin-onucus/s2023 revin_gamma_min=0.9721 ∈ [0.05,100] |
| 🟢 OK | ikincil_aralik | revin-afin-onucus/s2023 egitim_sure_sn=109.7 ∈ [10,900] |
