# Beklenti Kapısı — kuyruk `k3-baseline-v2` aşama 2 → **GEÇTİ**
*2026-10-14 02:05* · metrik `val_mse_avg` (min) · beklenti: `k3-baseline-v2.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline-onucus | 2 | 0.4484 | 0.4453, 0.4516 | 174.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | farkli_seedler_farkli | baseline-onucus: seed'ler farklı |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | aralik | baseline-onucus: ort 0.4484 ∈ [0.36, 0.71] (Repeat-last val ort. 0.4730 (data/naive_val.csv) ölçeğinde. Üst 0.71 = 1.5 × 0.473: kısaltılmış eğitim (≤2 epoch) için gevşetildi; Tab.10'daki en kötü L=96 modeli (Stationary 0.461) / Repeat test (0.344, Zeng et al. 2023 Tab.2) = 1.34 oranının üstünde pay. Alt 0.36 ≈ 0.76 × 0.473: literatürde L=96/7:1:2'de Repeat'i en çok geçen bildirilen sonuç DUET lookback-aramalı 0.280/0.344 = 0.81 (Qiu et al. 2025 Tab.3, hesap, literatur-cipa.md §5.5); ≤2 epoch'lu iTransformer'ın bunun da altına inmesi sızıntı/protokol şüphesi.) |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2023 val_mse_h96=0.1265 ∈ [0.05,0.26] |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2024 val_mse_h96=0.1274 ∈ [0.05,0.26] |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2023 egitim_sure_sn=133.2 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2024 egitim_sure_sn=192 ∈ [10,900] |
