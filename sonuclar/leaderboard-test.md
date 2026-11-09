# Leaderboard — TEST (kilit K5'te bir kez açıldı) · metrik `test_mse_avg` (düşük iyi)

*Açılış: 2026-10-14T15:42:23 → 2026-10-14T15:42:23 · betik `scripts/k5_test_ac.py --asama test` · okunan dosyalar ve md5: `sonuclar/test-acilis.log`*

Ön-kayıt: `orkestrasyon_durumu.md` 'K5 ŞEF ÖN-KAYDI'. Birincil karşılaştırmalar: **lastnorm vs baseline**, **lastnorm-ep336eq vs baseline-ep336eq**; diğer satırlar tablo/analiz. Test'e bakılarak seçim/arama yapılmadı.

## 1. Varyant özeti (n = 3 seed {2023, 2024, 2025}, eşleştirilmiş)

| # | varyant | test_mse_avg ort ± std | h96 | h192 | h336 | h720 | test_mae_avg | val_mse_avg (seçim) | Δtest vs baseline | p (eşl. t) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | san | 0.3007 ± 0.0048 | 0.0803 | 0.1589 | 0.2643 | 0.6996 | 0.3809 | 0.5848 ± 0.0044 | -17.52% | 0.0021 |
| 2 | lastnorm | 0.3465 ± 0.0019 | 0.0830 | 0.1728 | 0.3152 | 0.8151 | 0.3951 | 0.4278 ± 0.0059 | -4.96% | 0.0064 |
| 3 | lastnorm-ep336eq | 0.3473 ± 0.0027 | 0.0830 | 0.1728 | 0.3182 | 0.8151 | 0.3957 | 0.4191 ± 0.0051 | -4.75% | 0.0124 |
| 4 | lastnorm-dro | 0.3529 ± 0.0007 | 0.0835 | 0.1755 | 0.3190 | 0.8337 | 0.3980 | 0.4063 ± 0.0050 | -3.21% | 0.0021 |
| 5 | xmark-yok | 0.3606 ± 0.0018 | 0.0863 | 0.1771 | 0.3343 | 0.8446 | 0.4050 | 0.4436 ± 0.0098 | -1.10% | 0.0979 |
| 6 | baseline-ep336eq | 0.3646 ± 0.0003 | 0.0866 | 0.1794 | 0.3351 | 0.8574 | 0.4069 | 0.4376 ± 0.0126 | -0.00% | 0.9926 |
| 7 | baseline | 0.3646 ± 0.0008 | 0.0866 | 0.1794 | 0.3351 | 0.8574 | 0.4069 | 0.4448 ± 0.0119 | — | — |
| 8 | revin-afin | 0.3653 ± 0.0006 | 0.0867 | 0.1799 | 0.3354 | 0.8591 | 0.4072 | 0.4441 ± 0.0123 | +0.17% | 0.0271 |
| 9 | lastnorm-merkez | 0.3818 ± 0.0096 | 0.0850 | 0.1775 | 0.3311 | 0.9336 | 0.4121 | 0.4321 ± 0.0107 | +4.71% | 0.0999 |
| 10 | normsuz | 0.9644 ± 0.0162 | 0.3685 | 0.5991 | 1.4461 | 1.4439 | 0.7178 | 2.1892 ± 0.0401 | +164.49% | 0.0003 |

## 2. Karşılaştırmalar (test; eşleştirilmiş, n = 3)

| rol | varyant vs referans | ort (v / ref) | Δ | Δ% | çift Δ<0 | t | p (t) | %95 t-CI (Δ) | bootstrap %95 (Δ, n=3 dejenere) | Wilcoxon p (n=3 alt sınır 0.25) | ufuk yönü (Δ<0) | val Δ% (aynı çift) | yön korunuyor mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BIRINCIL-1 (resmi protokol) | lastnorm vs baseline | 0.3465 / 0.3646 | -0.0181 | -4.96% | 3/3 | -12.40 | 0.0064 | [-0.0243, -0.0118] | [-0.0206, -0.0155] | 0.25 | 4/4 | -3.83% | evet |
| BIRINCIL-2 (epoch-esit protokol) | lastnorm-ep336eq vs baseline-ep336eq | 0.3473 / 0.3646 | -0.0173 | -4.75% | 3/3 | -11.92 | 0.0070 | [-0.0236, -0.0111] | [-0.0197, -0.0147] | 0.25 | 4/4 | -4.24% | evet |
| protokol (H03) | baseline-ep336eq vs baseline | 0.3646 / 0.3646 | -0.0000 | -0.00% | 1/3 | -0.01 | 0.9926 | [-0.0022, +0.0021] | [-0.0009, +0.0008] | 1.00 | 1/4 | -1.62% | evet |
| protokol (H08 S1) | lastnorm-ep336eq vs lastnorm | 0.3473 / 0.3465 | +0.0007 | +0.22% | 0/3 | +1.51 | 0.2710 | [-0.0014, +0.0029] | [+0.0000, +0.0017] | 0.25 | 0/4 | -2.04% | HAYIR |
| ablasyon: lastnorm - sigma (H10) | lastnorm-merkez vs lastnorm | 0.3818 / 0.3465 | +0.0353 | +10.18% | 0/3 | +6.53 | 0.0227 | [+0.0120, +0.0585] | [+0.0279, +0.0458] | 0.25 | 0/4 | +1.01% | evet |
| basarisiz deneme (H09) | lastnorm-dro vs lastnorm | 0.3529 / 0.3465 | +0.0064 | +1.84% | 0/3 | +6.38 | 0.0237 | [+0.0021, +0.0106] | [+0.0049, +0.0083] | 0.25 | 0/4 | -5.02% | HAYIR |
| tablo (H02) | revin-afin vs baseline | 0.3653 / 0.3646 | +0.0006 | +0.17% | 0/3 | +5.95 | 0.0271 | [+0.0002, +0.0011] | [+0.0004, +0.0008] | 0.25 | 0/4 | -0.17% | HAYIR |
| tablo (H01, ariza notlu) | san vs baseline | 0.3007 / 0.3646 | -0.0639 | -17.52% | 3/3 | -21.78 | 0.0021 | [-0.0765, -0.0513] | [-0.0668, -0.0580] | 0.25 | 4/4 | +31.46% | HAYIR |
| tablo/ablasyon (H06) | normsuz vs baseline | 0.9644 / 0.3646 | +0.5998 | +164.49% | 0/3 | +61.95 | 0.0003 | [+0.5581, +0.6414] | [+0.5864, +0.6186] | 0.25 | 0/4 | +392.13% | evet |
| tablo (H07) | xmark-yok vs baseline | 0.3606 / 0.3646 | -0.0040 | -1.10% | 3/3 | -2.96 | 0.0979 | [-0.0099, +0.0018] | [-0.0061, -0.0015] | 0.25 | 4/4 | -0.29% | evet |
| tablo | lastnorm-merkez vs baseline | 0.3818 / 0.3646 | +0.0172 | +4.71% | 0/3 | +2.92 | 0.0999 | [-0.0081, +0.0425] | [+0.0073, +0.0277] | 0.25 | 3/4 | -2.86% | HAYIR |
| tablo | lastnorm-dro vs baseline | 0.3529 / 0.3646 | -0.0117 | -3.21% | 3/3 | -21.61 | 0.0021 | [-0.0141, -0.0094] | [-0.0123, -0.0106] | 0.25 | 4/4 | -8.66% | evet |
| tablo | lastnorm-ep336eq vs baseline | 0.3473 / 0.3646 | -0.0173 | -4.75% | 3/3 | -8.90 | 0.0124 | [-0.0257, -0.0090] | [-0.0206, -0.0138] | 0.25 | 4/4 | -5.79% | evet |

## 3. Ufuk bazında (birincil + protokol + ablasyon)

| varyant vs referans | ufuk | test v / ref | test Δ% | çift Δ<0 | p (t) | val Δ% | yön korunuyor mu |
|---|---|---|---|---|---|---|---|
| lastnorm vs baseline | h96 | 0.0830 / 0.0866 | -4.19% | 3/3 | 0.0284 | -3.12% | evet |
| lastnorm vs baseline | h192 | 0.1728 / 0.1794 | -3.65% | 3/3 | 0.0097 | -1.43% | evet |
| lastnorm vs baseline | h336 | 0.3152 / 0.3351 | -5.92% | 3/3 | 0.0002 | -2.08% | evet |
| lastnorm vs baseline | h720 | 0.8151 / 0.8574 | -4.93% | 3/3 | 0.0214 | -5.08% | evet |
| lastnorm-ep336eq vs baseline-ep336eq | h96 | 0.0830 / 0.0866 | -4.19% | 3/3 | 0.0284 | -3.12% | evet |
| lastnorm-ep336eq vs baseline-ep336eq | h192 | 0.1728 / 0.1794 | -3.65% | 3/3 | 0.0097 | -1.43% | evet |
| lastnorm-ep336eq vs baseline-ep336eq | h336 | 0.3182 / 0.3351 | -5.03% | 3/3 | 0.0019 | -3.85% | evet |
| lastnorm-ep336eq vs baseline-ep336eq | h720 | 0.8151 / 0.8574 | -4.93% | 3/3 | 0.0214 | -5.08% | evet |
| baseline-ep336eq vs baseline | h96 | 0.0866 / 0.0866 | +0.00% | 0/3 | 1.0000 | +0.00% | — |
| baseline-ep336eq vs baseline | h192 | 0.1794 / 0.1794 | +0.00% | 0/3 | 1.0000 | +0.00% | — |
| baseline-ep336eq vs baseline | h336 | 0.3351 / 0.3351 | -0.01% | 1/3 | 0.9926 | -7.21% | evet |
| baseline-ep336eq vs baseline | h720 | 0.8574 / 0.8574 | +0.00% | 0/3 | 1.0000 | +0.00% | — |
| lastnorm-ep336eq vs lastnorm | h96 | 0.0830 / 0.0830 | +0.00% | 0/3 | 1.0000 | +0.00% | — |
| lastnorm-ep336eq vs lastnorm | h192 | 0.1728 / 0.1728 | +0.00% | 0/3 | 1.0000 | +0.00% | — |
| lastnorm-ep336eq vs lastnorm | h336 | 0.3182 / 0.3152 | +0.95% | 0/3 | 0.2710 | -8.89% | HAYIR |
| lastnorm-ep336eq vs lastnorm | h720 | 0.8151 / 0.8151 | +0.00% | 0/3 | 1.0000 | +0.00% | — |
| lastnorm-merkez vs lastnorm | h96 | 0.0850 / 0.0830 | +2.51% | 0/3 | 0.1983 | +6.11% | evet |
| lastnorm-merkez vs lastnorm | h192 | 0.1775 / 0.1728 | +2.67% | 0/3 | 0.1645 | +5.94% | evet |
| lastnorm-merkez vs lastnorm | h336 | 0.3311 / 0.3152 | +5.04% | 0/3 | 0.1260 | -4.17% | HAYIR |
| lastnorm-merkez vs lastnorm | h720 | 0.9336 / 0.8151 | +14.53% | 0/3 | 0.0341 | +1.40% | evet |
| lastnorm-dro vs lastnorm | h96 | 0.0835 / 0.0830 | +0.60% | 0/3 | 0.0020 | +0.20% | evet |
| lastnorm-dro vs lastnorm | h192 | 0.1755 / 0.1728 | +1.54% | 0/3 | 0.2061 | +1.80% | evet |
| lastnorm-dro vs lastnorm | h336 | 0.3190 / 0.3152 | +1.20% | 0/3 | 0.0209 | +0.60% | evet |
| lastnorm-dro vs lastnorm | h720 | 0.8337 / 0.8151 | +2.27% | 0/3 | 0.0166 | -9.32% | HAYIR |

## 4. Notlar

- **Replikasyon sağlaması:** baseline `test_metrics.json` ile K3 istisnalı `metrics.json` test_mse_avg değerleri seed başına birebir aynı (|fark| < 1e-12); ortalama **0.3646** = replikasyon.md 0.3646 ✓ (0.3643, 0.3641, 0.3655).
- **Çoklu karşılaştırma (iki birincil, Holm):** lastnorm vs baseline p_Holm = 0.0129; lastnorm-ep336eq vs baseline-ep336eq p_Holm = 0.0129.
- **n = 3 notu:** bootstrap CI 27 yeniden-örnekten (tam numaralandırma) ve dejenere: yalnız çift Δ'ların min/max aralığını verir. Wilcoxon kesin testi n = 3'te p < 0.25 üretemez; anlamlılık kararı eşleştirilmiş t ile verilir (D2), Wilcoxon yalnız yön tutarlılığı bilgisidir.
- **Görülmemiş-iyi denetimi (H04 kartı):** lastnorm test_mse_avg < 0.317 → İNCELE. Ölçülen 0.3465 → tetiklenmedi.
- Validation leaderboard ayrı: `sonuclar/leaderboard.md` (leaderboard.py, yalnız val). Bu dosya yalnız test.
