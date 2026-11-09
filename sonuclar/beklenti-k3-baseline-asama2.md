# Beklenti Kapısı — kuyruk `k3-baseline` aşama 2 → **İHLAL — BÜYÜK KOŞUMA GEÇİLMEZ**
*2026-10-14 01:58* · metrik `val_mse_avg` (min) · beklenti: `k3-baseline.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline-onucus | 2 | 0.4484 | 0.4453, 0.4516 | 187.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🔴 İHLAL | mekanik | 1 koşum rc≠0: 20261014-012200_baseline-onucus_s2023(rc=96) |
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | farkli_seedler_farkli | baseline-onucus: seed'ler farklı |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | aralik | baseline-onucus: ort 0.4484 ∈ [0.36, 0.71] (Repeat-last val ort. 0.4730 (data/naive_val.csv) ölçeğinde. Üst 0.71 = 1.5 × 0.473: kısaltılmış eğitim (≤2 epoch) için gevşetildi; Tab.10'daki en kötü L=96 modeli (Stationary 0.461) / Repeat test (0.344, Zeng et al. 2023 Tab.2) = 1.34 oranının üstünde pay. Alt 0.36 ≈ 0.76 × 0.473: literatürde L=96/7:1:2'de Repeat'i en çok geçen bildirilen sonuç DUET lookback-aramalı 0.280/0.344 = 0.81 (Qiu et al. 2025 Tab.3, hesap, literatur-cipa.md §5.5); ≤2 epoch'lu iTransformer'ın bunun da altına inmesi sızıntı/protokol şüphesi.) |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2023 val_mse_h96=0.1265 ∈ [0.05,0.26] |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2024 val_mse_h96=0.1274 ∈ [0.05,0.26] |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2023 egitim_sure_sn=183 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | baseline-onucus/s2024 egitim_sure_sn=171.9 ∈ [10,900] |

## Yapılacak
1. Koşum kuyruğu durdu; sıradaki aşama başlatılmadı.
2. Her İHLAL için hata hipotezi kur (beklenti-ihlali.md): kurulum/veri yolu/seed/metrik hesabı.
3. Düzeltme sonra duman aşamasından yeniden; aynı kuyruk adı tekrar koşulmaz (`<ad>-v2`).
4. Beklenti yanlışsa: kaynağıyla düzelt (eşiği sonucu geçirmek için gevşetme yasak).
