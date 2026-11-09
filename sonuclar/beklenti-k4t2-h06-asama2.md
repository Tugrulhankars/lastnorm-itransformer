# Beklenti Kapısı — kuyruk `k4t2-h06` aşama 2 → **GEÇTİ**
*2026-10-14 05:55* · metrik `val_mse_avg` (min) · beklenti: `k4t2-h06.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| normsuz-onucus | 1 | 2.318 | 2.318 | 217.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | normsuz-onucus: ort 2.318 ∈ [0.4, 10.0] (H06 kartı Mekanik DUR + uygulama notu: ön-uçuş val_mse_avg ∈ [0.40, 10.0]; alt sınır ters yönlü İNCELE (b) (normsuz ebeveyn ön-uçuşundan belirgin iyi), üst sınır sayısal patlama) |
| 🟢 OK | ikincil_aralik | normsuz-onucus/s2023 egitim_sure_sn=205.2 ∈ [10,466] |
| 🟢 OK | ikincil_aralik | normsuz-onucus/s2023 param_sayisi=3.047e+05 ∈ [304720,304720] |
