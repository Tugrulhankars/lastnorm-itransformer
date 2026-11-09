# Beklenti Kapısı — kuyruk `k4t2-h04` aşama 2 → **GEÇTİ**
*2026-10-14 05:25* · metrik `val_mse_avg` (min) · beklenti: `k4t2-h04.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| lastnorm-onucus | 1 | 0.4454 | 0.4454 | 163.5 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | lastnorm-onucus: ort 0.4454 ∈ [0.4, 0.49] (H04 kartı Mekanik DUR + uygulama notu: ön-uçuş val_mse_avg ∈ [0.40, 0.49] (ebeveyn ön-uçuşu 0.4453 ± %10)) |
| 🟢 OK | ikincil_aralik | lastnorm-onucus/s2023 egitim_sure_sn=153.8 ∈ [10,466] |
| 🟢 OK | ikincil_aralik | lastnorm-onucus/s2023 param_sayisi=3.047e+05 ∈ [304720,304720] |
