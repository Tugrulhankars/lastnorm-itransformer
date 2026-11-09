# Beklenti Kapısı — kuyruk `k4t2-h06` aşama 1 → **GEÇTİ**
*2026-10-14 05:51* · metrik `val_mse_avg` (min) · beklenti: `k4t2-h06.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| normsuz-duman | 1 | 2.184 | 2.184 | 65.3 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | normsuz-duman: ort 2.184 ∈ [0.05, 5.0] (üst: H06 kartı uygulama notu 'duman val_mse_h96 sonlu ve ≤ 5.0'; alt: K3 duman alt sınırı 0.05 ≈ 0.6 × lit. test h96 0.086 (Liu et al. 2024 Tab.10) → altı sızıntı/protokol şüphesi) |
