# Beklenti Kapısı — kuyruk `k4t1-h02` aşama 1 → **GEÇTİ**
*2026-10-14 03:40* · metrik `val_mse_avg` (min) · beklenti: `k4t1-h02.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| revin-afin-duman | 1 | 0.1363 | 0.1363 | 34.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | revin-afin-duman: ort 0.1363 ∈ [0.05, 0.1923] (üst: H02 kartı Mekanik DUR 'duman val_mse_h96 > 0.1923' (= 1.5 × Repeat val h96 0.1282); alt: K3 duman alt sınırı 0.05 ≈ 0.6 × lit. test h96 0.086 (Liu et al. 2024 Tab.10) → altı sızıntı/protokol şüphesi) |
