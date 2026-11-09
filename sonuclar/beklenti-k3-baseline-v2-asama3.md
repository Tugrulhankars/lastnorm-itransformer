# Beklenti Kapısı — kuyruk `k3-baseline-v2` aşama 3 → **GEÇTİ**
*2026-10-14 02:19* · metrik `val_mse_avg` (min) · beklenti: `k3-baseline-v2.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline | 3 | 0.4448 | 0.4428, 0.4341, 0.4577 | 274.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | farkli_seedler_farkli | baseline: seed'ler farklı |
| 🟢 OK | min_kosum | 3 başarılı koşum (en az 3) |
| 🟢 OK | aralik | baseline: ort 0.4448 ∈ [0.36, 0.634] (Tam eğitim: üst 0.634 = 1.34 × Repeat val 0.473 (1.34 = Tab.10 en kötü L=96 model Stationary 0.461 / Repeat test 0.344; Liu et al. 2024 Tab.10, Zeng et al. 2023 Tab.2); alt 0.36 ≈ 0.76 × 0.473 (DUET 0.280/0.344 = 0.81 oranının altı, Qiu et al. 2025 Tab.3) → altı İNCELE.) |
| 🟢 OK | ikincil_aralik | baseline/s2023 egitim_sure_sn=279.1 ∈ [60,3000] |
| 🟢 OK | ikincil_aralik | baseline/s2024 egitim_sure_sn=253.2 ∈ [60,3000] |
| 🟢 OK | ikincil_aralik | baseline/s2025 egitim_sure_sn=267.5 ∈ [60,3000] |
