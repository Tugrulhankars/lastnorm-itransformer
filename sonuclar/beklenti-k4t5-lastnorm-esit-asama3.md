# Beklenti Kapısı — kuyruk `k4t5-lastnorm-esit` aşama 3 → **İHLAL — BÜYÜK KOŞUMA GEÇİLMEZ**
*2026-10-14 17:22* · metrik `val_mse_avg` (min) · beklenti: `k4t5-lastnorm-esit.beklenti.json` · 9 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| lastnorm-esit-k3 | 3 | 0.4477 | 0.4498, 0.4491, 0.4443 | 5.4 |
| lastnorm-esit-k2 | 3 | 1.141 | 1.14, 1.139, 1.143 | 5.3 |
| lastnorm-esit-k1 | 3 | 0.2209 | 0.2207, 0.2214, 0.2205 | 7.6 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | farkli_seedler_farkli | lastnorm-esit-k3: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | lastnorm-esit-k2: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | lastnorm-esit-k1: seed'ler farklı |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | min_kosum | 9 başarılı koşum (en az 9) |
| 🟢 OK | aralik | lastnorm-esit-k1: ort 0.2209 ∈ [0.1135, 0.222312] (alt: 0.5 × naif Repeat K1 0.227003 (sonuclar/naif-kat.csv, mekanik); üst: konvekslik 0.5·lastnorm-k1 Ā 0.217618 + 0.5·naif 0.227003 = 0.222311 (kart H14 Mekanizma (1) kimliği)) |
| 🟢 OK | aralik | lastnorm-esit-k2: ort 1.141 ∈ [0.5652, 1.142843] (alt: 0.5 × naif Repeat K2 1.130437 (mekanik); üst: konvekslik 0.5·lastnorm-k2 Ā 1.155247 + 0.5·naif 1.130437 = 1.142842) |
| 🟢 OK | aralik | lastnorm-esit-k3: ort 0.4477 ∈ [0.2365, 0.450418] (alt: 0.5 × naif Repeat K3 0.473029 (mekanik); üst: konvekslik 0.5·lastnorm K3 Ā 0.427804 + 0.5·naif 0.473029 = 0.450417) |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2023 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2024 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2025 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2023 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2024 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2025 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2023 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2024 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2025 saglama_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2023 oracle_ihlal_sayisi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2024 oracle_ihlal_sayisi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2025 oracle_ihlal_sayisi=0 ∈ [0,0] |
| 🔴 İHLAL | ikincil_aralik | lastnorm-esit-k2/s2023 oracle_ihlal_sayisi=3 ∉ [0,0] |
| 🔴 İHLAL | ikincil_aralik | lastnorm-esit-k2/s2024 oracle_ihlal_sayisi=2 ∉ [0,0] |
| 🔴 İHLAL | ikincil_aralik | lastnorm-esit-k2/s2025 oracle_ihlal_sayisi=3 ∉ [0,0] |
| 🔴 İHLAL | ikincil_aralik | lastnorm-esit-k1/s2023 oracle_ihlal_sayisi=2 ∉ [0,0] |
| 🔴 İHLAL | ikincil_aralik | lastnorm-esit-k1/s2024 oracle_ihlal_sayisi=2 ∉ [0,0] |
| 🔴 İHLAL | ikincil_aralik | lastnorm-esit-k1/s2025 oracle_ihlal_sayisi=2 ∉ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2023 kimlik_fark_maks=2.22e-16 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2024 kimlik_fark_maks=2.22e-16 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2025 kimlik_fark_maks=2.776e-16 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2023 kimlik_fark_maks=4.441e-16 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2024 kimlik_fark_maks=1.11e-15 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2025 kimlik_fark_maks=9.992e-16 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2023 kimlik_fark_maks=1.665e-16 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2024 kimlik_fark_maks=1.665e-16 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2025 kimlik_fark_maks=5.551e-17 ∈ [0,1e-06] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2023 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2024 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2025 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2023 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2024 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2025 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2023 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2024 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2025 w=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2023 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2024 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k3/s2025 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2023 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2024 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k2/s2025 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2023 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2024 egitim_sure_sn=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | lastnorm-esit-k1/s2025 egitim_sure_sn=0 ∈ [0,0] |

## Yapılacak
1. Koşum kuyruğu durdu; sıradaki aşama başlatılmadı.
2. Her İHLAL için hata hipotezi kur (beklenti-ihlali.md): kurulum/veri yolu/seed/metrik hesabı.
3. Düzeltme sonra duman aşamasından yeniden; aynı kuyruk adı tekrar koşulmaz (`<ad>-v2`).
4. Beklenti yanlışsa: kaynağıyla düzelt (eşiği sonucu geçirmek için gevşetme yasak).
