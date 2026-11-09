# Beklenti Kapısı — kuyruk `h16-ettm1` aşama 1 → **GEÇTİ**
*2026-10-15 06:47* · metrik `val_mse_avg` (min) · beklenti: `h16-ettm1.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| ettm1-baseline-h96-duman | 1 | 0.3944 | 0.3944 | 474.8 |
| ettm1-lastnorm-h96-duman | 1 | 0.4028 | 0.4028 | 483.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | ettm1-baseline-h96-duman: ort 0.3944 ∈ [0.1719, 1.8745] (alt = 0.5 x lit oran h96 0.2751 x Repeat val h96 1.2497; ust = 1.5 x en iyi trivial val h96 1.2497 (1 epoch kisaltmasi; P2 ETTh1 duman emsali)) |
| 🟢 OK | aralik | ettm1-lastnorm-h96-duman: ort 0.4028 ∈ [0.1719, 1.8745] (alt = 0.5 x lit oran h96 0.2751 x Repeat val h96 1.2497; ust = 1.5 x en iyi trivial val h96 1.2497 (1 epoch kisaltmasi; P2 ETTh1 duman emsali)) |
| 🟢 OK | ikincil_aralik | ettm1-baseline-h96-duman/s2023 param_sayisi=2.242e+05 ∈ [224224,224224] |
| 🟢 OK | ikincil_aralik | ettm1-lastnorm-h96-duman/s2023 param_sayisi=2.242e+05 ∈ [224224,224224] |
