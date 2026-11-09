# Beklenti Kapısı — kuyruk `k4t1-h03` aşama 2 → **GEÇTİ**
*2026-10-14 03:12* · metrik `val_mse_h336` (min) · beklenti: `k4t1-h03.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline-ep336eq-onucus | 1 | 0.3797 | 0.3797 | 172.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | baseline-ep336eq-onucus: ort 0.3797 ∈ [0.3, 0.42] (H03 kartı Uygulama notu: ön-uçuş val_mse_h336 ∈ [0.30, 0.42]) |
| 🟢 OK | ikincil_aralik | baseline-ep336eq-onucus/s2023 val_mse_h96=0.1265 ∈ [0.1264655590057373,0.1264655590057373] |
| 🟢 OK | ikincil_aralik | baseline-ep336eq-onucus/s2023 val_mse_h192=0.2163 ∈ [0.21626509726047516,0.21626509726047516] |
| 🟢 OK | ikincil_aralik | baseline-ep336eq-onucus/s2023 val_mse_h720=1.038 ∈ [1.0379893779754639,1.0379893779754639] |
| 🟢 OK | ikincil_aralik | baseline-ep336eq-onucus/s2023 egitim_sure_sn=164.7 ∈ [10,900] |
