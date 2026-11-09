# Tur 2 Tahlili (K4c): H04 lastnorm · H05 SAN-μ (kapı) · H06 normsuz · H07 xmark-yok (YALNIZ VALIDATION)

*Hazırlayan: deney-tahlilci (mod 2, tur-tahlili) · 2026-10-14 · tur 2 / 4*

- **Girdi:** `sonuclar/leaderboard.md` (06:37), `hipotezler/H04–H07.md`, `SIRALAMA.md` (Tur 2 bölümü, ortak ön-kayıtlı
  eklemeler 1–4), `varyantlar/san-mu/onkosul.log`, `sonuclar/h05-onkosul.csv`, `orkestrasyon_durumu.md` (seçim ve durma
  kuralı), `gunluk/deney-gunlugu.md`, `sonuclar/hata-analizi.md`, `sonuclar/tahlil-tur1.md`.
- **Tam koşumlar (aşama 3, n = 3, seed {2023, 2024, 2025}, eşleştirilmiş):**
  - baseline (ebeveyn): `020531_s2023`, `021024_s2024`, `021448_s2025`
  - lastnorm (H04): `052508_s2023`, `053601_s2024`, `054213_s2025`. Bellek temizliğiyle öldürülen `053238_s2024` (rc = 96) **dışlandı**.
  - normsuz (H06): `055518_s2023`, `060205_s2024`, `060859_s2025`
  - xmark-yok (H07): `061948_s2023`, `062637_s2024`, `063215_s2025`
  - Kıyas için (tur 1): baseline-ep336eq `031233/032853/033500`, revin-afin `034304/034735/035213`, san `040042/040547/041102`
- **Betikler (tekrar üretilebilir, WS kökünden):**
  - `scripts/tahlil_tur2.py` (tur-1 betiğinin uyarlaması) → `sonuclar/tahlil_tur2.log`, `tahlil-tur2-ufuk.csv`
    (varyant × ufuk × seed eşleştirilmiş farklar, h336 için ep336eq kıyası dahil), `tahlil-tur2-ikincil.csv` (7 varyant ×
    4 ufuk ikincil metrikler), **`normalizasyon-kiyas.csv`**.
  - `scripts/hata_analizi.py --runs … --etiket <ad> --cikti-ek _<ad>` → `hata-kirilim_{lastnorm,normsuz,xmark-yok}.csv`,
    `hata-adim_…csv`, `hata_analizi_…log`.
  - **Üzerine yazma yok:** baseline (`hata-kirilim.csv`, `hata-adim.csv`, `hata_analizi.log`) ve tur-1
    (`tahlil-tur1-*.csv`, `tahlil_tur1.log`) dosyalarının md5'i koşum öncesi ve sonrasında aynı (`md5sum -c` OK).
- **Sağlamalar:**
  - Hesapladığım ort ± std ve eşleştirilmiş p değerleri leaderboard.md ile aynı: lastnorm 0.4278 ± 0.00588, p = 0.228;
    xmark-yok 0.4436 ± 0.00978, p = 0.773; normsuz 2.189 ± 0.0401, p = 0.000185.
  - **M1:** kart mtime'ları H04 04:52, H05 04:54, H06 04:55, H07 04:56. Hepsi ilk ilgili ölçümden önce:
    `onkosul.log` 05:02, lastnorm-duman 05:03, normsuz-duman 05:08, xmark-yok-duman 05:12.

> **TEST KİLİDİ.** `leaderboard.csv`, `test_metrics.json`, `preds/test_kilitli/` ve `replikasyon.md` okunmadı.
> `metrics.json` dosyalarından `test` ile başlayan anahtarlar programatik olarak süzüldü. stdout.log'lardan yalnız
> `Train Loss` ve `Vali Loss` alanları regex ile çekildi. Tahminler yalnız `preds/val_h*.npz` dosyalarından alındı.

---

## 0. Özet kararlar

| Kart | Ölçüm (val, n = 3) | Kural (ön-kayıtlı) | Karar | Bayrak |
|---|---|---|---|---|
| **H04** lastnorm | avg 0.44485 → **0.42780** (Δ **−3.83%**, 3/3 çift, p = 0.228). Ufuk: −3.12 / −1.43 / −2.08 / −5.08%, yani **4/4**. Seed × ufuk çiftlerinin 12/12'si < 0. D1 M/R h96 1.453 → 1.018 | Δavg < 0 ∧ ≥ 3/4 ufuk. |Δavg| < 8.1% → "ayırt edilemez". Z2 özel kuralı tetiklenmedi (şekil düzeldi **ve** toplam iyileşti) | **TUT — ayırt edilemez** (Δ = 0.47 × MDE) | Sarı: h336 düşüş-pencere M/R 0.768 → 0.981 (+0.213) → "val trendine aşırı uyum" notu (ön-kayıtlı, karar değişmez, K5'e). h336 oyu bütçe-bağımlı (0.3920 > 0.3714) |
| **H05** SAN-μ | CPU kapısı: R_H = 1.219 / 1.257 / 1.333 / 1.136 (val), 0/4 < 1, ort. 1.236 → Ö1 BAŞARISIZ. Ö2 maks 0.0091 ≤ 0.10, GEÇER | Kapı = Ö1 ∧ Ö2 | **ÖN-KOŞUL BAŞARISIZ (koşulmadı)**, ÖLDÜR'e eşdeğer. Slot H07'ye geçti | Yok. Sonuç kartın ön-kayıtlı beklentisiyle aynı (P(geçer) ≈ 0.10) |
| **H06** normsuz | avg 0.44485 → **2.18924** (Δ **+392.1%**, 0/3, p = 0.00018). Ufuk: +613 / +566 / +721 / +204%, yani 0/4 | ÖLDÜR (i) ve (ii). Ters yönlü İNCELE tetiklenmedi | **ÖLDÜR (beklenen; ablasyon satırı)**. Aile kapanmaz | Sarı: büyüklük kartın üst ucunun (+210%) üstünde. Mekanizma ölçüldü, alarm kapandı (§4 #1, §5.2) |
| **H07** xmark-yok | avg 0.44485 → 0.44356 (Δ **−0.29%**, 2/3 çift, p = 0.773). Ufuk: −0.78 / −0.15 / +0.19 / −0.44%, yani 3/4 | Δavg < 0 ∧ ≥ 3/4. |Δavg| < 8.1%. İki yönlü İNCELE tetiklenmedi | **TUT — ayırt edilemez** (Δ = 0.04 × MDE). Kartın dilinde: **H0 desteklendi** (etki yok) | Sarı: "TUT" etiketi iyileşme diye okunmamalı. Δ seed gürültüsünün içinde |

- **Kırmızı bayrak: 0.** Görülmemiş-iyi eşiği aşılmadı. Bit-bit aynı varyant yok (parametreler eşit, tahminler farklı). n < 3 yok.
- **Sarı bayrak: 5** (§4).
- **Durma kuralı: TETİKLENMEDİ.** H04 bir TUT ve Δavg = −3.83% ≤ −1.0%. **Sayaç 1/2 → 0/2 sıfırlandı. Tur 3 açılır.** Ebeveyn `lastnorm` (§6).

---

## 1. Ölçüm okuması: ufuk bazında val MSE, Δ, eşleştirilmiş farklar

Biçim tur 1 ile aynı. Ebeveyn `baseline` (seed-ort. ± sd): h96 0.12486 ± 0.00105, h192 0.21139 ± 0.00246, h336 0.40027 ± 0.00196,
h720 1.04286 ± 0.04391, avg 0.44485 ± 0.01191. Seed başına baseline avg değerleri 0.44275 / 0.43412 / 0.45767.
"Çift Δ" sütunu s2023 / s2024 / s2025 sırasıyla varyant − ebeveyn farkıdır.
- **MDE (karar):** ön-kayıtlı 3v3 MDE (hata-analizi §6).
- **MDE_eş (bilgi):** eşleştirilmiş MDE = (t.975,2 + t.8,2) · sd_d / √3. SIRALAMA ortak ekleme 2 gereği raporlanır, kararı değiştirmez.

### H04 lastnorm
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 | sd_d · r | p (eşl.) | MDE · MDE_eş | Durum |
|---|---|---|---|---|---|---|---|---|
| h96 | 0.12096 ± 0.00039 | −0.00390 (**−3.12%**) | −0.0040 / −0.0031 / −0.0046 | 3/3 | 0.00078 · +0.79 | 0.013 | 2.6% · 1.9% | **MDE üstü** |
| h192 | 0.20837 ± 0.00045 | −0.00302 (−1.43%) | −0.0058 / −0.0006 / −0.0027 | 3/3 | 0.00260 · −0.22 | 0.18 | 3.5% · 3.8% | MDE altı |
| h336 | 0.39197 ± 0.00309 | −0.00831 (**−2.08%**) | −0.0055 / −0.0094 / −0.0100 | 3/3 | 0.00248 · +0.60 | 0.028 | 1.5% · 1.9% | **MDE üstü** |
| h720 | 0.98992 ± 0.02366 | −0.05294 (−5.08%) | −0.0267 / −0.0034 / **−0.1287** | 3/3 | 0.06664 · **−0.94** | 0.30 | 12.8% · 19.8% | MDE altı |
| **avg** | **0.42780 ± 0.00588** | **−0.01704 (−3.83%)** | −0.0105 / −0.0041 / −0.0365 | 3/3 | 0.01717 · **−0.84** | **0.228** | 8.1% · 12.0% | **ayırt edilemez** |

- 3/4 yön kuralı: **4/4.** h336 oyu düşülse bile 3/4 sağlanıyor, karar h336'ya bağlı değil.
- **h336 çift kıyas (ortak ekleme 1):**
  - vs `baseline` 0.40027: −2.08% (3/3).
  - vs `baseline-ep336eq` 0.37143: **+5.53%** (+0.0242 / +0.0215 / +0.0159; p = 0.014).
  - Varyant h336 > 0.3714 olduğundan oy **"eğitim-bütçesine bağımlı"** diye etiketlenir: epoch eşitleme lastnorm'dan 2.6 kat daha büyük h336 kazancı veriyor.

### H06 normsuz
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 | sd_d · r | p | Durum |
|---|---|---|---|---|---|---|---|
| h96 | 0.88979 ± 0.02882 | +0.76493 (+612.6%) | +0.787 / +0.777 / +0.732 | 0/3 | 0.029 · −0.43 | 0.0005 | MDE üstü, kötü |
| h192 | 1.40746 ± 0.01590 | +1.19606 (+565.8%) | +1.189 / +1.216 / +1.183 | 0/3 | 0.018 · −0.76 | 7e-5 | MDE üstü, kötü |
| h336 | 3.28495 ± 0.06660 | +2.88468 (+720.7%) | +2.958 / +2.829 / +2.867 | 0/3 | 0.066 · +0.38 | 0.0002 | MDE üstü, kötü |
| h720 | 3.17477 ± 0.23810 | +2.13191 (+204.4%) | +1.868 / +2.304 / +2.223 | 0/3 | 0.232 · +0.23 | 0.004 | MDE üstü, kötü |
| **avg** | **2.18924 ± 0.04012** | **+1.74440 (+392.1%)** | +1.700 / +1.782 / +1.751 | 0/3 | 0.041 · +0.07 | 0.00018 | **MDE üstü, kötü (48 × MDE)** |

- 3/4 kuralı **0/4**.
- h336 vs ep336eq: +784%. Etiket "bütçe-bağımlı değil", çünkü iyileşme yok.
- Repeat'e karşı M/R: 6.94 / 6.22 / 8.35 / 2.77.

### H07 xmark-yok
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 | sd_d · r | p | MDE · MDE_eş | Durum |
|---|---|---|---|---|---|---|---|---|
| h96 | 0.12389 ± 0.00041 | −0.00097 (−0.78%) | −0.0013 / −0.0002 / −0.0014 | 3/3 | 0.00065 · +1.00 | 0.12 | 2.6% · 1.6% | MDE altı |
| h192 | 0.21107 ± 0.00096 | −0.00032 (−0.15%) | −0.0019 / +0.0012 / −0.0003 | 2/3 | 0.00152 · +0.99 | 0.75 | 3.5% · 2.2% | MDE altı |
| h336 | 0.40102 ± 0.00106 | +0.00075 (+0.19%) | +0.0018 / +0.0020 / −0.0016 | 1/3 | 0.00200 · +0.23 | 0.58 | 1.5% · 1.6% | MDE altı |
| h720 | 1.03825 ± 0.03722 | −0.00461 (−0.44%) | +0.0243 / −0.0103 / −0.0278 | 2/3 | 0.02652 · +0.80 | 0.79 | 12.8% · 7.9% | MDE altı |
| **avg** | **0.44356 ± 0.00978** | **−0.00129 (−0.29%)** | +0.0057 / −0.0018 / −0.0078 | 2/3 | 0.00677 · +0.82 | 0.773 | 8.1% · 4.7% | **ayırt edilemez** |

- 3/4 kuralı **3/4** (h96, h192, h720). h336 oyu karşı yönde.
- h336 vs ep336eq: +7.97%, bütçe-bağımlı.
- İki yönlü İNCELE eşikleri tetiklenmedi:
  - (a) |Δavg| = 0.29% < 8.1%
  - (b) en büyük ufuk iyileşmesi −0.78%. 2 × MDE eşiği h96'da −5.2%.

### H05 SAN-μ (yalnız kapı, koşum yok)
| Ufuk | E_san | E_naif (son dilim μ) | E_son (son değer) | E_lb (μ_lb) | R_H = E_san/E_naif (val) | R_H train (bilgi) | σ_son-dilim < 1e-3 oranı |
|---|---|---|---|---|---|---|---|
| h96 | 0.1569 | 0.1287 | **0.1251** | 0.1901 | **1.219** | 0.920 | 0.0041 |
| h192 | 0.2847 | 0.2265 | **0.2230** | 0.3020 | **1.257** | 0.919 | 0.0048 |
| h336 | 0.5245 | 0.3934 | **0.3897** | 0.4936 | **1.333** | 0.897 | 0.0062 |
| h720 | 1.3527 | 1.1905 | **1.1412** | 1.2643 | **1.136** | 0.889 | 0.0091 |

- **Ö1:** #{R_H < 1} = 0/4 (gerekli ≥ 3), ort. R = 1.236 (gerekli < 1) → BAŞARISIZ.
- **Ö2:** maks 0.0091 ≤ 0.10 → GEÇER.
- **KAPI = BAŞARISIZ.** Kartın dilinde: "ÖN-KOŞUL BAŞARISIZ (koşulmadı)", ÖLDÜR'e eşdeğer. Aile tur 3'te yalnız yeni kaynaklı bir kartla (ör. SAN-softplus) açılabilir.
- **Bağlantılı bulgu (H04 ile tutarlılık):** val'de dört ufkun dördünde de seviye çapaları aynı sırada: **E_son < E_naif < E_lb**. E_san, h96/h192'de E_lb'nin altında, h336/h720'de üstünde.
  - Yani val'de gelecek dilim ortalamasının en iyi naif tahmincisi **son değer**.
  - Resmî NST'nin merkezi μ_lb ise naif üçlünün en kötüsü.
  - H04'ün kazanç yönünü bağımsız bir ölçüm önceden gösteriyordu (bileşen düzeyi, ölçüm koşumdan önce yapıldı).

### Eşleştirme notu (ortak ekleme 2)
- RNG akışı korunan kartlarda çift korelasyonu yüksek beklenmişti. **Gerçekleşen ufka bağlı:**
  - lastnorm: h96 +0.79, h336 +0.60, **h720 −0.94**, avg −0.84
  - xmark-yok: avg +0.82, h96 +1.00
  - normsuz: avg +0.07
- lastnorm'da h720 çiftleri **ters** korelasyonlu. Bu nedenle eşleştirilmiş MDE (avg %12.0) 3v3 MDE'den (%8.1) **büyük**: eşleştirme burada duyarlılık kaybettiriyor (§3). Karar eşiği değişmez (MDE 8.1%).

---

## 2. İkincil metrikler (val, seed-ort.; tanımlar hata-analizi §3.1 ve tahlil-tur1 §2 ile aynı)

Kaynak: `sonuclar/tahlil-tur2-ikincil.csv`. Baseline satırları tur 1 ile birebir aynı (0.974 / 0.2217 / 1.776 …).

| Ufuk · varyant | M/R | worst_block (blok) | worst_ratio | rho_shift | top10 M/R | bias P−Y | s_io oranı | std(P)/std(Y) | ufuk-içi trend | D1 M/R | \|P1−son\| | corr(P1−son, μ_lb−son) | β | Düşüş M/R (n) | Yükseliş M/R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h96 baseline | 0.974 | 0.2217 (B3) | 1.776 | 0.185 | 1.005 | −0.035 | 0.68 | 0.196 | +0.008 | 1.453 | 0.066 | +0.23 | 0.95 | 1.076 (218) | 0.890 |
| h96 **lastnorm** | **0.944** | 0.2166 (B3) | 1.790 | 0.237 | 0.957 | −0.043 | 0.66 | 0.169 | +0.033 | **1.018** | **0.011** | **+0.02** | **1.00** | **1.030** | **0.872** |
| h96 normsuz | **6.941** | 1.3972 (B3) | 1.571 | 0.010 | 5.69 | **−0.680** | 2.12 | 0.559 | −0.217 | **37.8** | 0.559 | +0.20 | 0.19 | 5.342 | 8.257 |
| h96 xmark-yok | 0.966 | 0.2170 (B3) | 1.752 | 0.271 | 1.031 | −0.037 | 0.65 | 0.203 | +0.007 | 1.488 | 0.068 | +0.32 | 0.92 | 1.027 | 0.916 |
| h192 baseline | 0.934 | 0.3136 (B3) | 1.484 | 0.418 | 0.821 | −0.069 | 0.57 | 0.158 | +0.027 | 1.235 | 0.079 | +0.30 | 0.93 | 1.059 (163) | 0.879 |
| h192 **lastnorm** | **0.921** | 0.3139 (B3) | 1.506 | 0.410 | 0.884 | −0.082 | 0.50 | 0.174 | +0.046 | **1.005** | **0.014** | **−0.01** | **1.01** | **1.001** | 0.885 |
| h192 normsuz | 6.219 | 1.9432 (B3) | 1.381 | 0.050 | 5.90 | −0.875 | 1.86 | 0.616 | −0.380 | 25.9 | 0.676 | +0.23 | −0.11 | 6.126 | 6.259 |
| h192 xmark-yok | 0.933 | 0.3144 (B3) | 1.489 | 0.411 | 0.848 | −0.053 | 0.57 | 0.157 | +0.031 | 1.288 | 0.085 | +0.38 | 0.91 | 1.070 | 0.872 |
| h336 baseline | 1.018 | 0.4606 (B2) | 1.151 | 0.810 | 1.152 | −0.272 | 0.30 | 0.166 | −0.037 | 1.194 | 0.089 | +0.68 | 0.72 | 0.768 (67) | 1.067 |
| h336 **lastnorm** | **0.997** | **0.4503 (B3)** | 1.149 | **0.661** | **0.966** | **−0.241** | **0.38** | 0.119 | −0.026 | **1.016** | **0.019** | **−0.18** | **1.01** | **0.981 (+0.213)** | **1.000** |
| h336 normsuz | 8.352 | 4.0868 (B3) | 1.244 | −0.601 | 4.17 | −1.392 | 2.87 | 0.464 | −0.171 | 55.6 | 1.261 | +0.47 | −2.95 | 8.295 | 8.363 |
| h336 xmark-yok | 1.020 | 0.4622 (B2) | 1.153 | 0.811 | 1.161 | −0.274 | 0.30 | 0.167 | −0.039 | 1.198 | 0.090 | +0.68 | 0.73 | 0.763 | 1.070 |
| h720 baseline | 0.911 | 1.6597 (B3) | 1.592 | 0.790 | 1.108 | −0.801 | 0.18 | 0.163 | −0.011 | 1.028 | 0.148 | +0.16 | 0.98 | n<10 (0) | 0.911 |
| h720 **lastnorm** | **0.865** | **1.5706 (B3)** | 1.587 | 0.768 | **0.968** | −0.785 | 0.19 | 0.177 | **+0.137** | **0.850** | **0.020** | −0.16 | 1.03 | n<10 | 0.865 |
| h720 normsuz | 2.775 | 4.5502 (B3) | 1.434 | 0.622 | 2.57 | −1.362 | 0.98 | 0.659 | −0.432 | 6.93 | 0.531 | +0.42 | −0.39 | n<10 | 2.775 |
| h720 xmark-yok | 0.907 | 1.6494 (B3) | 1.589 | 0.835 | 1.167 | −0.794 | 0.20 | 0.160 | +0.010 | 1.052 | 0.143 | +0.10 | 1.02 | n<10 | 0.907 |

### Ek göstergeler (CSV'de)

**Adım-H yön isabeti**

| | h96 | h192 | h336 | h720 |
|---|---|---|---|---|
| baseline | 0.58 | 0.63 | 0.47 | 0.64 |
| lastnorm | 0.63 | 0.65 | 0.45 | **0.85** |
| normsuz | 0.42 | 0.47 | 0.43 | **0.25** |
| xmark-yok | 0.59 | 0.65 | 0.47 | 0.66 |

**α\*** (sapmaların Repeat'e göre ölçeği)

| | h96 | h192 | h336 | h720 |
|---|---|---|---|---|
| baseline | 0.64 | 0.95 | 0.31 | 2.00 |
| lastnorm | 1.33 | 1.30 | 0.77 | 3.65 |
| normsuz | 0.03 | 0.02 | −0.07 | −0.81 |

**Değişken payı (Z4)**

| h720 | `mse_deg3` | `mse_deg0` | diğer-6 ort. |
|---|---|---|---|
| baseline | 2.607 | 1.884 | 0.642 |
| lastnorm | 2.408 | 1.855 | 0.609 |
| xmark-yok | 2.594 | 1.959 | 0.625 |

- lastnorm h336'da değişken `3` hafif kötüleşti (1.338 → 1.360), değişken `0` iyileşti (0.541 → 0.508).

**OOR kırılımı** (değişken 0/3/5/6; hedef hücrelerinin %78–85'i train aralığı dışında)

| | Tahminin train dışına çıkma oranı (h96 / h192 / h336 / h720) | OOR-4 MSE / iç-4 MSE (h96) |
|---|---|---|
| baseline | 0.75 / 0.74 / 0.58 / 0.00 | 0.187 / 0.062 |
| lastnorm | 0.75 / 0.73 / 0.61 / 0.00 | 0.180 / 0.062 |
| **normsuz** | **0.002 / 0.000 / 0.000 / 0.000** | **1.680 / 0.099** (oran 17×) |
| xmark-yok | ≈ baseline | ≈ baseline |

**Eğitim, süre ve parametre (yardımcı)**

| | Koşulan epoch (h96 / h192 / h336 / h720) | Eğitim süresi (s, 3 seed) | Parametre | Not |
|---|---|---|---|---|
| baseline | 10,9,8 / 10,10,10 / 1,1,1 / 6,8,5 | 279 / 253 / 268 | 304 720 | |
| lastnorm | 10,10,10 / 9,10,10 / 1,1,1 / 10,7,10 | 433 / 352 / 460 | 304 720 (eşit) | h96 train kaybı (s2023) 0.1190, baseline 0.1216 |
| normsuz | 10,10,10 / 10,10,10 / 1,1,1 / 7,5,6 | 384 / 395 / 407 | 304 720 (eşit) | h96 train 0.1399 · val 0.903 |
| xmark-yok | 6,9,8 / 10,10,10 / 1,1,1 / 9,8,10 | 386 / 319 / 294 | 304 720 (eşit) | erken durma ayrıştı (RNG, ön-kayıtlı) |

---

## 3. lastnorm'un p = 0.228'i: kaynak, MDE konumu, "TUT ama ayırt edilemez"

**Seed başına Δavg'nin ufuk ayrışımı** (Δ_H / 4; `tahlil_tur2.log`):

| Seed | Δavg (göreli) | h96 | h192 | h336 | h720 |
|---|---|---|---|---|---|
| s2023 | −0.01048 (−2.37%) | −0.00100 | −0.00144 | −0.00137 | −0.00668 |
| s2024 | −0.00413 (−0.95%) | −0.00077 | −0.00014 | −0.00236 | −0.00085 |
| s2025 | **−0.03652 (−7.98%)** | −0.00116 | −0.00068 | −0.00250 | **−0.03218** |

1. **p'nin kaynağı h720, özellikle s2025.**
   - Eşleştirilmiş Δavg varyansının **%94'ü** h720 katkısından geliyor: var(Δ_720/4) = 2.78e-4, var(Δavg) = 2.95e-4.
   - Bunun nedeni lastnorm'un seed'e özgü bir sıçraması değil. **Ebeveynin** s2025 h720 değeri aykırı: baseline 1.0903 (öteki iki seed 1.027 / 1.003). lastnorm'da aynı seed 0.9626.
   - lastnorm h720 seed sd'sini 0.044'ten 0.024'e indiriyor ve en kötü baseline seed'inde en iyi sonucu veriyor (çift r = −0.94).
   - Eşleştirme bu ters korelasyonda farkın sd'sini büyütüyor. Eşleştirmesiz Welch testi avg için p = 0.115 veriyor. Bu karar dışı bir karşılaştırmadır; yalnız mekanizmayı gösterir.
2. **h720 hariç üç ufuk:** çiftler −0.00507 / −0.00437 / −0.00579, ort. **−2.07%**, sd_d 0.0007, p = 0.0065.
   - Kısa ve orta ufuk kazancı küçük ama determinist bir yön etkisi.
   - **12/12** seed × ufuk çifti negatif. Çiftler bağımsız olmadığı için bu, karar dışı betimsel bir gözlemdir.
3. **MDE'ye göre konum:**
   - avg Δ −3.83% = **0.47 × MDE (8.1%)**.
   - Ufuk bazında h96 1.20 × MDE (2.6%), h336 1.38 × MDE (1.5%): bu ikisi **ayırt edilebilir**. h192 0.41 ×, h720 0.40 × MDE: ayırt edilemez.
   - Kartın beklentisi avg −3…+4% idi, sonuç alt uçta. Mekanizma beklentileri: D1 h96 ≤ 1.20 ✓ (1.018), h192 ≤ 1.10 ✓ (1.005), corr mutlakça azaldı ✓.
4. **Seçim kuralının harfiyen uygulanması:** Δavg < 0 ✓, ≥ 3/4 ufuk ✓ (4/4), |Δavg| < 8.1%.
   - **→ "TUT — ayırt edilemez".**
   - Durma kuralı bakımından Δavg = −3.83% ≤ −1% ✓. Karar n = 3 eşleştirilmiş seed üzerinden, ebeveyne göre göreli.
5. **Yorum (karardan ayrı):**
   - Sonuç, "son-değer çapası val'de küçük ama tutarlı yönlü bir iyileşme veriyor" cümlesini destekliyor. "iTransformer'ı anlamlı biçimde geçiyor" cümlesini desteklemiyor.
   - Kısa-ufuk bileşeni (h96 −3.1%, h336 −2.1%) ön-kayıtlı ufuk MDE'lerinin üstünde ve mekanizma düzeyinde açıklanıyor (§5.1). Avg'ye en büyük katkı ise en gürültülü ufuktan (h720) geliyor.
   - Testte A-çıkışı (anlamlı + pratik) için bu kanıt zayıf. Makalenin ağırlığı B-çıkışına (normalizasyon kıyası + mekanizma) kayıyor.
   - Eşiği gevşetmek (ör. lastnorm seed'lerinden yeni MDE %4.2 türetmek) **önerilmez** (çıpa ilkesi).

---

## 4. Kırmızı bayrak taraması

| # | Bayrak | Kart | Bulgu | Sonuç |
|---|---|---|---|---|
| 1 | **Beklenti ihlali (büyüklük)** | H06 | +392% > kartın üst ucu +210% (NST Tab.4 vanilla Transformer, hesap). Yön beklenen yönde, 4/4 ufuk kötü | **SARI. Mekanizma ölçüldü, alarm kapandı** (§5.2). Kod/sızıntı hatası işareti yok: (a) bayrak devrede: tahminler farklı (maks \|fark\| 1.8–4.4 z), parametre eşit; (b) model train'i öğrenmiş: h96 train kaybı 0.140, NST 0.122; (c) çöküş yalnız train-dışı seviyelerde: val/train kayıp oranı 6.4 (NST ≈ 1.0), OOR-4 / iç-4 MSE 17×, tahminlerin train aralığını aşma oranı %0–0.2 (gerçek %78–85). Fazlalığın nedenleri: val, literatürdeki test bölümlerinden daha kaymış (K2: val MMD² 1.00, test 0.58) ve h336'nın 1 epoch'u normsuz modeli daha çok cezalandırıyor (h336 +721% > h720 +204%). h336 hariç üç ufukta Δ +297%, yani yine üst ucun üstünde. Karar değişmez. Makaleye "val'de, en kaymış bölümde ölçüldü" notuyla girer |
| 2 | **Trend-asimetri (ön-kayıtlı not)** | H04 | Δavg < 0 iken h336 düşüş M/R **0.768 → 0.981 (+0.213 ≥ +0.05)**, yükseliş M/R 1.067 → 1.000. h96/h192'de ise ikisi birlikte düştü (1.076 → 1.030 / 0.890 → 0.872; 1.059 → 1.001 / 0.879 → 0.885) | **SARI: "val trendine aşırı uyum" notu** (kart ve SIRALAMA ortak ekleme 3). Karar değişmez. h336'daki kazanç yükseliş pencerelerinden geliyor; H03'teki örüntünün aynısı (0.77 → 0.92). K5 test blok kırılımı (Q1–Q4) ebeveyn, ep336eq ve lastnorm için yan yana raporlanmalı |
| 3 | Ayırt edilemezlik / tek-seed baskınlığı | H04 | p = 0.228, Δavg 0.47 × MDE. Varyansın %94'ü h720 s2025 çiftinden | **SARI (raporlama).** "TUT — ayırt edilemez" etiketi zorunlu. Makalede "anlamlı iyileşme" yazılamaz (D2) |
| 4 | "TUT" etiketinin yanlış okunması | H07 | Δavg −0.29% = 0.04 × MDE, 3/4 ufuk sınırda (h192 −0.15%, h720 −0.44%). Tahlilci ilkesi ("seed gürültüsü içinde Δ → ayrım yok") ile kartın ön-kayıtlı kuralı aynı sonuca varıyor: **etki yok** | **SARI.** Kart kuralı harfiyen "TUT — ayırt edilemez" verir. Kartın kendi dili "H0 desteklendi" der. Ebeveyn adayı değil (SIRALAMA: H07 yalnız ≤ −1% ise ebeveyn olurdu) |
| 5 | Süre ölçümü kirli (R6) | tümü | Parametresi ve FLOP'u ebeveynle aynı üç varyant da yavaş: lastnorm 352–460 s, normsuz 384–407 s, xmark-yok 294–386 s (baseline 253–279 s). Günlükte 3. bellek-temizliği öldürmesi var (053238) | **SARI.** Kart tavanı aşılmadı (seed başına > 540 s). Süreler hız/maliyet iddiası için kullanılamaz |
| 6 | Görülmemiş kadar iyi | H04, H07 | H04: avg 0.4278 > 0.3870; h96 −3.1% (eşik −8%), h192 −1.4% (−12%), h336 −2.1% (−15%), h720 −5.1% (−25%). H07: \|Δ\| eşiklerinin hepsinin altında. H06 ters yönlü eşiği (iyi çıkması) geçmedi | Temiz |
| 7 | Bit-bit aynı (G7, plasebo) | H04, H06, H07 | 3 × 4 × 3 hücrede `array_equal` = False. Pred maks \|fark\|: lastnorm 0.74–1.56, normsuz 1.8–4.4, xmark-yok 0.09–0.74 z. CPU doğrulamaları (`dogrula.log`) GEÇTİ: lastnorm x_son = `batch_x[:, -1]`, hedef penceresinden değil; f ≡ 0 → Repeat sağlandı | Temiz |
| 8 | n < 3 | hepsi | n = 3. lastnorm s2024 rc = 96 yarımı dışlandı; leaderboard bu dışlamayla aynı ort/sd'yi veriyor | Temiz |
| 9 | İkincil metrikte gizli bedel | H04 | worst_ratio h96 1.776 → 1.790, h192 1.484 → 1.506 (avg worst'ten hızlı düşüyor; worst_block mutlak olarak h96'da düştü, h192'de değişmedi). rho_shift h96 0.185 → 0.237. B3 M/R h96 1.036 → 1.011 (≥ 1 sürüyor). std(P)/std(Y) h336 0.166 → 0.119 (daha fazla yumuşatma, H03'teki gibi) | Bilgi. Belirgin bedel yok. Dayanıklılık metrikleri ya iyileşti ya da yerinde kaldı. Ödünleşim yalnız #2 |

**Kırmızı bayrak sayısı: 0. Sarı: 5** (#1–#5).

---

## 5. Mekanizma tahlili

### 5.1 H04: son-değer çapası Z2'yi kapattı, Z1'e küçük dokundu

**Z2 (ilk-adım basamağı): kapandı.**
- D1 M/R 1.453 / 1.235 / 1.194 / 1.028 → **1.018 / 1.005 / 1.016 / 0.850**.
- Adım-1 MSE h96'da 0.0117 → 0.0036 (Repeat 0.0033).
- |P1 − son| 0.066–0.148 → 0.011–0.020. Gerçek |Y1 − son| 0.030–0.037; model artık gerçek ilk adım değişiminden **daha küçük** bir sıçrama yapıyor.
- Seviye çapası β 0.72–0.98 → **1.00–1.03**. corr(P1 − son, μ_lb − son) +0.16…+0.68 → −0.18…+0.02.
- Yani P1 artık μ_lb'ye çekilmiyor. Kartın Z2 öngörüsü (D1 h96 ≤ 1.20, h192 ≤ 1.10) karşılandı.
- **Z2 özel kuralı:** D1 h96 düzelmesi 0.435 ≥ 0.10 ve toplam iyileşti. Kural (ÖLDÜR + aile kapanır) **tetiklenmedi**. "Mekanizma devrede değil" etiketi de yok (0.435 ≥ 0.05).

**h96/h192'de kazancın kaynağı: basamağın silinmesi.**
- h96 adım-dilimi M/R 1.46 / 1.12 / 1.06 / … → 1.02 / 1.00 / 1.00 / …: ilk %30'luk adım diliminin fazlası kalktı.
- Son dilimler zaten Repeat'in altındaydı (0.93–0.94 → 0.91).
- Düşüş ve yükseliş pencerelerinin ikisinde de M/R düştü. Bu, **trend uyumundan bağımsız** bir şekil düzeltmesine işaret ediyor.
- Tanısal ölçüm uyumu: hata-analizi'nin "yeniden-çapa" son-işlemi baseline'da h96'da +1.8% **kötüleştiriyordu**; lastnorm'da −0.3%. Naif yeniden-çapalamanın kaybettirdiği ofset bilgisi uçtan uca öğrenmeyle korunuyor (kart Risk 2 gerçekleşmedi).

**Z1 (uzun ufuk bias / trend körlüğü): kısmen, h720'de belirgin, h336'da az.**
- Bias h336 −0.272 → −0.241 (Repeat −0.240 düzeyi), h720 −0.801 → −0.785. **Seviye tavanı esasen sürüyor.**
- s_io oranı h336 0.30 → 0.38 (ep336eq 0.42), h720 0.18 → 0.19.
- Ufuk-içi tahmin trendi:
  - h336: −0.037 → −0.026. Hâlâ negatif. Kartın "0'a yaklaşır" öngörüsü kısmen sağlandı.
  - **h720: −0.011 → +0.137.** İşaret döndü (gerçek +0.42…+1.61).
- h720 adım-H yön isabeti 0.64 → **0.85**, α* 2.0 → 3.6 (sapmalar doğru yönlü ama hâlâ 3.6 kat temkinli).
- **h720 −5.1%'in kaynağı:** μ_lb'ye doğru ortalamaya-dönüş çekimi kalktı. Val tek bir yükseliş (K2) olduğundan μ_lb yükselen pencerelerde son değerin altında kalıyordu. Kaldırılan çekim bias'ı ve geç adımları iyileştiriyor: h720 adım-dilimi M/R 0.90 → 0.79.
- **Uyarı:** bu kazancın bir kısmı **val'in rejimine** bağlı. Testin geç dönemi düşük seviyede olduğundan (K2) h720 kazancı testte küçülebilir ya da tersine dönebilir.
- NLinear'ın Exchange'deki +6.0%'lık zararı (Zeng Tab.2) burada görülmedi. Kartın Risk 1'i val'de gerçekleşmedi; test sınaması K5'te yapılacak.

**h336: kazanç yükseliş pencerelerinden.**
- Yükseliş M/R 1.067 → 1.000, düşüş 0.768 → 0.981. H03'teki (ep336eq) örüntüyle aynı yönde.
- h336'da 1 epoch'luk model iki durumda da yakınsamamış. Son-değer çapası, bu yakınsamamış modelin μ_lb'ye çekilmesini kaldırarak ep336eq'nun kazancının yaklaşık üçte birini sağlıyor (−2.1% vs −7.2%).
- **İki etkinin toplanabilir olup olmadığı ölçülmedi.** Bunu lastnorm-ep336eq satırı gösterecek (§8, §9).

**Öğrenilen (D4):** "NST'nin merkezleme referansını pencere ortalamasından son girdi değerine taşımak (parametresiz, tek değişiklik) iTransformer'ın Exchange val'deki ilk-adım basamağını kaldırıyor (D1 M/R 1.45 → 1.02). val_mse_avg −3.8% iyileşiyor (4/4 ufuk, 12/12 seed × ufuk çifti; p = 0.23, MDE altı). Uzun ufuk seviye bias'ı ise neredeyse değişmiyor (h720 −0.80 → −0.79 z)."

### 5.2 H06: normalizasyonsuz iTransformer neden bu kadar kötü (seviye kayması + LayerNorm doyumu)

- **Train'i öğreniyor, val'e genellemiyor.**
  - h96 train kaybı 0.140 (NST 0.122, +15%), val kaybı 0.90 (NST 0.123).
  - Val/train oranı **6.4**; NST'de ≈ 1.0. h720'de train 0.49, val 2.9–3.3.
  - Optimizasyon arızası değil, **dağılım dışına genelleme arızası**.
- **Tahminler train aralığına hapsoluyor (kartın doyum öngörüsü doğrulandı).**
  - OOR değişkenlerinde (0, 3, 5, 6) gerçek hedef hücrelerinin %78–85'i train aralığı dışında.
  - normsuz tahminlerin train dışına çıkma oranı **%0.0–0.2**. NST'de %58–75: NST, μ_lb + σ_lb·f ile seviyeyi girdiden **taşıyabiliyor**.
  - Sonuç: OOR-4 değişken MSE'si h96'da 1.680, iç-4 değişkenlerde 0.099 (17×). NST'de 0.187 / 0.062 (3×).
  - Değişken başına M/R: OOR'de 6–19×, iç değişkenlerde 1.1–4.6×.
- **Seviye bilgisi çıkışa ulaşmıyor.**
  - D1 M/R 37.8 / 25.9 / 55.6 / 6.9. |P1 − son| 0.53–1.26 z: model ilk adımda son değeri bile yeniden üretemiyor.
  - Seviye çapası β 0.19 / −0.11 / −2.95 / −0.39. α* ≈ 0 ya da negatif.
  - Ufuk-içi tahmin trendi negatif (−0.17…−0.43): train ortalamasına doğru dönüş. h720 adım-H yön isabeti 0.25.
- Mekanizma kartın yazdığı gibi: token vektörü kodlayıcının son `LayerNorm`'undan geçiyor, çıkış genliği train'de öğrenilen projeksiyonla sınırlı.
- **h336'nın orantısız kötülüğü (+721%):** 1 epoch'luk resmî ayar normsuz modelde daha ağır. Epoch-1 train kaybı normsuzda 0.636, NST'de 0.470. NST'nin "kolay başlangıç" avantajı (f ≡ 0 → μ_lb) normsuzda yok.
- **Beklenti ihlali değerlendirmesi (beklenti-ihlali §4(5)):** (a) mekanizma devrede (bayrak uygulandı, tahminler farklı); (b) etki sürüyor (3/3 seed, sd küçük); (c) neden bu büyüklükte olduğu ölçümle açıklandı (OOR oranı, doyum, val/train oranı, h336 epoch etkisi). **Raporlanabilir.**
  - Kartın üst ucu (+210%) test bölümlerinde ve farklı omurgada ölçülmüş bir değerdi. Bizim val'imiz OOR hedef oranı bakımından en zor bölüm.
- **Öğrenilen (ablasyon satırı):** "use_norm = 0 ile iTransformer Exchange val'de 4/4 ufukta 3–8 kat kötüleşiyor (val_mse_avg 0.445 → 2.189, +392%). Model train'i öğreniyor (h96 train kaybı +15%), ama tahminleri train aralığından çıkamıyor (%0–0.2, gerçek %78–85). Örnek normalizasyonunun bu verideki değeri, train dışı seviyeleri girdiden çıkışa taşımasından geliyor."

### 5.3 H07: x_mark ablasyonu neden nötr (sentetik takvim)

- **Takvim sentetik: doğrulandı** (ham CSV `date` sütunu, yalnız tarih alanı okundu):
  - 7 588 satır, 1990-01-01 → 2010-10-10, ardışık günler arası fark **her yerde 1 gün** (7 587/7 587).
  - Haftanın günleri **tam eşit dağılmış** (her gün 1 084 kez), saat sabit 0.
  - Döviz piyasası hafta sonu kapalı olduğundan gerçek işlem günleri böyle dağılmaz. DayOfWeek ve DayOfMonth gözlemle ilişkisiz, belirlenimci birer testere dişi. HourOfDay sabit.
  - DayOfYear da sentetik takvimden türediği için gerçek mevsim fazı taşımıyor. Bu, K2 ve hata-analizi §7 şüphesinin doğrulanmasıdır.
- **Sonuç:**
  - Dört takvim token'ı bilgi taşımıyor. Kaldırılması 12 → 8 token ile dikkat ve LayerNorm istatistiklerini değiştiriyor: tahminler farklı (0.09–0.74 z), erken durma epoch'ları ayrışıyor (h96 6,9,8 vs 10,9,8).
  - Ama MSE etkisi seed gürültüsünün içinde: |Δ_H| ≤ 0.78%, Δavg 0.04 × MDE.
  - Mekanizma metrikleri ebeveynin ±0.05'i içinde: D1 M/R, bias, s_io, β, trend asimetrisi.
- beklenti-ihlali §4(5): (a) mekanizma devrede (`dogrula.log` (iii): x_mark'lı ve x_mark'sız çıkış farklı; tahminler farklı); (b) etki ≈ 0; (c) açıklama: token'lar sentetik ve belirlenimci, gözlem bilgisi yok. **Raporlanabilir.**
  - GLAFF'ın Traffic gözlemiyle tutarlı (iTransformer'da damga kaldırma "no significant decline"). Exchange'de bu nötrlüğün nedeni üstelik veri artefaktı.
- **Öğrenilen:** "Exchange'in tarih sütunu sentetik (hafta sonları dahil kesintisiz takvim). iTransformer'ın dört takvim token'ını kaldırmak val_mse_avg'yi değiştirmiyor (−0.29%, p = 0.77; ufuk |Δ| ≤ 0.8%). Resmî kıyaslarda bu token'lar etkisiz yük olarak taşınıyor."

### 5.4 H05: kapı neden başarısız oldu

- SAN'ın gelecek-dilim ortalama MLP'si **train'de** naif son-dilim ortalamasından %8–11 iyi (R 0.89–0.92), **val'de** %14–33 kötü (R 1.14–1.33).
- Yani train rejimindeki ortalama dinamiğini öğrenip val'in kaymış rejimine taşıyamıyor. Bu, Z1'in **seviye tahmini yoluyla** kapatılamayacağının bileşen düzeyinde kanıtı.
- H04'ün başarısıyla birlikte okunuyor: val'de **öğrenilmiş seviye tahmini < son değer** (E_son en düşük, 4/4 ufuk).
- **Öğrenilen:** "SAN'ın gelecek-dilim ortalama tahmincisi (H01 tam koşum checkpoint'leri, 3 seed) Exchange val'de son girdi diliminin ortalamasından 4/4 ufukta kötü (R = 1.14–1.33), train'de iyi (R = 0.89–0.92). Yalnız-ortalama SAN ön-kayıtlı kapıdan geçemedi ve koşulmadı."

---

## 6. Durma kuralı kararı

- **Kural (SIRALAMA Tur 2, orkestrasyon_durumu):** "Tur 2'de en az bir kart TUT alıp Δavg ≤ −1.0% (ebeveyn `baseline`'a göre göreli, n = 3 eşleştirilmiş) vermezse sayaç 2/2 olur ve K4 kapanır."
- **Tur 2 TUT'ları:**
  - H04 lastnorm: TUT (ayırt edilemez), Δavg **−3.83%**
  - H07 xmark-yok: TUT (ayırt edilemez), Δavg −0.29%. Δ > −1% olduğu için sayılmaz.
  - H06: ÖLDÜR. H05: kapı. İkisi de sayılmaz.
- **Karar: KOŞUL SAĞLANDI.**
  - **Durma sayacı 1/2 → 0/2 (sıfırlandı). K4 devam eder, tur 3 açılır.**
  - Tur 3 ebeveyni **`lastnorm`** (SIRALAMA koşullu ön-planı: "H04 TUT (≤ −1%) → tur 3 ebeveyni lastnorm"; birden çok TUT varsa en iyisi).
- Kuralın metni "TUT — ayırt edilemez" sonuçları dışarıda bırakmıyor. Yalnız "TUT ama Δavg > −1%" sonuçları dışarıda. Bu okuma eşik gevşetmesi değil, ön-kaydın düz metni.
- **Not (şef için):** Yeni zincir ebeveyninin val değeri 0.42780 ± 0.00588 (n = 3). Tur 3'te durma kuralı yeni ebeveyne göre işler: tur 3'te ≥%1'lik TUT yoksa sayaç 1/2 olur. Bütçenin 4. turu son tur olduğundan K4 en geç tur 4 sonunda kapanır.

---

## 7. Normalizasyon kıyas tablosu (B çıkışının çekirdeği; `sonuclar/normalizasyon-kiyas.csv`)

Val, n = 3, eşleştirilmiş, ebeveyn NST (resmî). h336 resmî 1 epoch.

| Normalizasyon (kart) | val_mse_avg | Δ vs NST (p) | h96 | h192 | h336 | h720 | Ufuk ↑ | M/R h96 / h720 | worst_block h336 / h720 | rho_shift h336 | bias h720 | D1 M/R h96 | Düşüş / yükseliş M/R h336 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yok, `use_norm 0` (H06) | 2.1892 ± 0.0401 | **+392.1%** (0.0002) | 0.8898 | 1.4075 | 3.2850 | 3.1748 | 0/4 | 6.94 / 2.77 | 4.087 / 4.550 | −0.60 | −1.362 | 37.8 | 8.29 / 8.36 |
| **NST μ/σ (resmî)** | **0.4449 ± 0.0119** | — | 0.1249 | 0.2114 | 0.4003 | 1.0429 | — | 0.974 / 0.911 | 0.461 / 1.660 | 0.81 | −0.801 | 1.453 | 0.768 / 1.067 |
| RevIN-afin (H02) | 0.4441 ± 0.0123 | −0.17% (0.13) | 0.1252 | 0.2116 | 0.3996 | 1.0400 | 2/4 | 0.977 / 0.909 | 0.459 / 1.655 | 0.81 | −0.800 | 1.455 | 0.769 / 1.065 |
| **son-değer çapası (H04)** | **0.4278 ± 0.0059** | **−3.83% (0.23)** | **0.1210** | **0.2084** | **0.3920** | **0.9899** | **4/4** | **0.944 / 0.865** | **0.450 / 1.571** | **0.66** | −0.785 | **1.018** | 0.981 / 1.000 |
| SAN resmî, std başlığı çökmüş (H01) | 0.5848 ± 0.0044 | +31.5% (0.004) | 0.1600 | 0.2885 | 0.5279 | 1.3627 | 0/4 | 1.248 / 1.191 | 0.633 / 2.046 | 0.84 | −0.957 | 2.559 | 0.616 / 1.485 |
| SAN-μ, yalnız-ortalama (H05) | — (koşulmadı) | ön-koşul başarısız | — | — | — | — | — | — | — | — | — | — | — |

- SAN-μ bileşen ölçümü (CPU kapısı, val): R_H = E_san / E_naif = 1.219 / 1.257 / 1.333 / 1.136.
- Referans (protokol, normalizasyon değil): NST + h336 epoch-eşit (H03) h336 0.3714, avg 0.4376.

**Tablodan okunan B bulguları (taslak, karar değil):**
1. Normalizasyonun **varlığı** belirleyici (yok → NST: ×4.9). **Biçimi** ikincil:
   - afin serbestlik etkisiz (−0.2%)
   - merkezleme referansı küçük ama tutarlı (−3.8%)
   - öğrenilmiş gelecek-istatistik tahmini bu veride zararlı ya da elendi.
2. Dayanıklılık (rho_shift, worst_block, top10) yalnız son-değer çapasında iyileşiyor: h336 rho 0.81 → 0.66, worst −2.3%, top10 M/R 1.15 → 0.97. Uzun ufuk bias'ı (Z1) hiçbir normalizasyon varyantıyla anlamlı biçimde kapanmıyor (−0.785…−0.801; Repeat −0.872).
3. Seviye çapalarının val sıralaması (H05 kapısı): son değer < son-dilim μ < μ_lb, ve öğrenilmiş μ̂ en kötü ya da kötüye yakın. Bu, tablodaki performans sıralamasıyla aynı. Mekanizma ile sonuç arasında bağımsız iki ölçüm tutarlı.

---

## 8. Tur 3 önerisi (durma kuralı açık; ebeveyn `lastnorm`)

**Bütçe görüşü.**
- Kalan ≈ 5.0 GPU-saat, K5 için ~1.0 ayrılı. Tur 3 + tur 4 için ≈ 4.0 saat. Kart başına 0.3–0.5 saat; tur başına ≤ 3 kart.
- **Bağlayıcı kısıt yine kart sayısı ve durma kuralı, GPU değil.**
- Bellek-temizliği öldürmeleri sürüyor (3 kez). Kuyruk başına boş RAM ölçümü önerilir.
- K5 GPU ihtiyacı (lastnorm kazanan kalırsa): epoch-eşit sürüm ≈ 0.3 ve ablasyon (1) ≈ 0.3. Test oturumu mevcut kilitli tahminlerden açılır, ≈ 0 GPU.

**Hata kütlesi hâlâ nerede (lastnorm ebeveyniyle):**
- h720 + h336 avg'nin %81'i (0.990 + 0.392)/4 / 0.428.
- Z1 seviye tavanı neredeyse yerinde (bias h720 −0.785, s_io oranı 0.19).
- h336'da 1-epoch yakınsamama sürüyor (vs ep336eq +5.5%).
- Z3 B3 aşınması: B3 M/R h96 1.011, h336 1.022.
- Z4: değişken `3`'ün payı %30–43.
- Z2 kapandı.

### Tur 3 aday kartları (sıralı; tasarımcı kaynaklayıp ön-kaydeder)

| Sıra | Kart (öneri) | Tür | Tek değişiklik (ebeveyn `lastnorm`) | Hedef | Beklenti (kaynak) | Maliyet | Not |
|---|---|---|---|---|---|---|---|
| **1** | **lastnorm-ep336eq** (H03'ün lastnorm üzerine uygulanması) | **PROTOKOL** (H03 tipi; ebeveyn olmaz, durma sayacına girmez) | h336 `--train_epochs 1 → 10` (patience 3) | Z5 × Z2 etkileşimi | h336 0.3920 → ≈ 0.36–0.37 (H03 −7.2% iç çıpa). Toplanabilirlik **ölçülmedi** | ≈ 0.3 (yalnız h336 × 3 seed + kontrol ufukları bit-bit) | K5'te zaten gerekli (tahlil-tur1 §6.5). Tur 3'te koşulursa K5 yükü azalır ve "lastnorm kazancı epoch eşitlemeyle toplanıyor mu?" sorusu testten önce yanıtlanır. Kontrol ufukları ebeveynle bit-bit aynı olmalı (plasebo testi) |
| **2** | **A5 zaman-grubu DRO** (Z3/Z4) | kazanç kartı | Kayıp: train'i K zaman grubuna bölüp en kötü grup kaybını ağırlıklandırma (group-DRO). Mimari ve normalizasyon aynı | worst_block, B3/B4 M/R, değişken `3` | avg 0…−8% (FOIL Tab.2, tahlil-tur1 §7). MDE altı beklenir. **Birincil karar avg**, worst_block ve B3/B4 M/R ön-kayıtlı ikincil | ≈ 0.4–0.5 | Tezin dayanıklılık eksenine doğrudan katkı. Risk: val tek yükseliş; grup tanımı yalnız train'den (sızıntı yok) |
| **3** | **lastnorm yalnız-merkezleme** (σ = 1; H04 Ablasyon 1) | ablasyon (K5'ten öne alınabilir) | σ_lb ölçeklemesi kapalı, merkez x_son | Kazancın çapadan mı, ölçek etkileşiminden mi geldiği | Yön belirsiz. normsuz sonucu σ olmadan OOR'nin çapa ile çözüldüğünü düşündürüyor, ama ölçek farkı (σ_lb ~0.16 z) omurganın giriş ölçeğini ×6 değiştirir | ≈ 0.3 | Kart değil ablasyon. Şef K5'e bırakmayı seçerse slot şu iki adaydan birine geçer |
| (alt.) | **Online ince ayar, gecikmeli geri bildirim** (Z3) | — | — | Z1/Z3 | Exchange için etki büyüklüğü **BULUNAMADI** (tur 1) | ≈ 0.5 + yüksek uygulama riski | **UYARI:** değerlendirme akışında geçmiş etiketleri kullanır, yani değerlendirme **protokolünü** değiştirir. Genel onay "veri/protokol değişikliği yapılmaz" sınırını içerir. **Kullanıcı onayı olmadan tur 3'e alınmamalı**; ayrı tablo/sınırlılık satırı olabilir |
| (alt.) | Z1-hedefli "drift çapası" (son değer + lookback eğimi) | — | — | Z1 | Kaynak gerekli (ör. Hyndman & Athanasopoulos drift yöntemi; Exchange'de rastgele yürüyüş drift'i zayıf). Val tek yükseliş olduğundan **trend-asimetri riski çok yüksek** | ≈ 0.35 | Önerilmez. Ancak tasarımcı kaynak bulur ve testte düşüş rejimine karşı ön-kayıtlı bir koruma yazarsa |

- **Önerilen tur 3 seti:** (1) lastnorm-ep336eq [protokol] + (2) A5 DRO [kazanç]. Üçüncü slot: şefin K5 bütçe tercihine göre (3) yalnız-merkezleme ya da boş.
- Tahmini maliyet ≈ 0.7–1.1 GPU-saat. Tur 3 sonrası kalan ≈ 3.9–4.3 saat (K5'in 1.0'ı dahil).
- **Tur 3 ortak ön-kayıtları:**
  - Ebeveyn değerleri `lastnorm` olur: h96 0.12096, h192 0.20837, h336 0.39197, h720 0.98992, avg 0.42780.
  - Trend-asimetri ebeveyn düşüş M/R: 1.030 / 1.001 / 0.981 / n<10.
  - h336 çift kıyası `lastnorm` ve `baseline-ep336eq`'e karşı (ve koşulduysa `lastnorm-ep336eq`'e karşı).
  - **MDE eşiği 8.1% olarak kalır.** lastnorm seed'lerinden türeyen %4.2 yalnız bilgi olarak yazılır; çıpa ilkesi gereği eşik değiştirilmez.
  - Resmî `baseline`'a karşı Δ da raporlanır (makale tablosu için).
- **x_mark (H07b):** H07 nötr çıktığı için yeni ebeveynde tekrarı **önerilmez** (bilgi değeri düşük).

**K5 ön-planı (tur 3 sonrası; test kilidini tahlilci açmaz):**
1. Kazananın epoch-eşit sürümü (yukarıda 1; tur 3'te yapılmadıysa).
2. lastnorm Ablasyon (1) yalnız-merkezleme.
3. Tek test oturumu, aynı seed'ler: resmî iTransformer, iTransformer epoch-eşit (H03), kazanan, kazanan epoch-eşit. Tablo bütünlüğü için şef kararıyla normsuz / RevIN-afin / SAN satırları (H06 kartı izin veriyor; ayrı arama yok, D1).
4. Test blok kırılımı (Q1–Q4) ve trend-asimetri (düşüş/yükseliş M/R) ebeveyn ve kazanan için yan yana (sarı #2). `hata_analizi.py`'nin test sürümü yalnız K5'te.
5. Görülmemiş-iyi denetimi: lastnorm `test_mse_avg` < 0.317 → İNCELE (H04 kartı).

---

## 9. Kartların "Sonuç" bölümleri için önerilen metin (şef yazar; tahlilci kartlara dokunmadı)

- **H04:**
  - Koşumlar: `20261014-052508_lastnorm_s2023`, `…053601_lastnorm_s2024`, `…054213_lastnorm_s2025` (`…053238_lastnorm_s2024` rc = 96, bellek temizliği, dışlandı).
  - Ölçülen: val_mse_avg 0.4278 ± 0.0059 (n = 3), Δ −0.0170 (−3.83%), p = 0.228. Eşleştirilmiş Δ sd 0.0172, r = −0.84. Varyansın %94'ü h720 s2025 çiftinden.
  - Ufuk: −3.12 / −1.43 / −2.08 / −5.08%, yani 4/4; seed × ufuk çiftlerinin 12/12'si < 0.
  - h336 0.3920: vs baseline −2.08%, vs ep336eq +5.53% (bütçe-bağımlı oy).
  - Mekanizma: D1 M/R h96 1.453 → 1.018.
  - Karar: **TUT — ayırt edilemez** (Δavg < 0, 4/4 ufuk, |Δ| < MDE 8.1%). Z2 özel kuralı tetiklenmedi (şekil düzeldi, toplam iyileşti).
  - Not: h336 düşüş-pencere M/R 0.768 → 0.981 → "val trendine aşırı uyum" notu, K5'e taşınır.
  - Öğrenilen: §5.1.
- **H05:**
  - Ön-koşul: R_H (val, 3-seed) = 1.219 / 1.257 / 1.333 / 1.136, #{R_H < 1} = 0/4, ort. 1.236. Ö2 maks 0.0091. KAPI = BAŞARISIZ (`varyantlar/san-mu/onkosul.log`, `sonuclar/h05-onkosul.csv`).
  - Koşumlar: koşulmadı (ÖN-KOŞUL BAŞARISIZ).
  - Karar: **ÖN-KOŞUL BAŞARISIZ** (ÖLDÜR'e eşdeğer). Slot H07'ye geçti. Aile tur 3'te yalnız yeni kaynaklı bir kartla açılabilir.
  - Öğrenilen: §5.4.
- **H06:**
  - Koşumlar: `20261014-055518_normsuz_s2023`, `…060205_normsuz_s2024`, `…060859_normsuz_s2025`.
  - Ölçülen: 2.1892 ± 0.0401, Δ +1.7444 (+392.1%), p = 0.00018. Eşleştirilmiş Δ sd 0.041, r = +0.07.
  - Ufuk: +613 / +566 / +721 / +204%, yani 0/4. h336 3.285 (vs ep336eq +784%).
  - OOR-4 / iç-4 MSE h96 1.680 / 0.099. Tahminlerin train dışı oranı %0–0.2 (gerçek %78–85).
  - Karar: **ÖLDÜR (beklenen; ablasyon satırı).** Büyüklük kartın üst ucunu (+210%) aşıyor. Neden: val'in OOR seviyesi ve h336 1-epoch etkisi; mekanizma ölçüldü (§5.2).
  - Öğrenilen: §5.2.
- **H07:**
  - Aktivasyon: H05 kapısı BAŞARISIZ → tur 2'de koşuldu.
  - Koşumlar: `20261014-061948_xmark-yok_s2023`, `…062637_xmark-yok_s2024`, `…063215_xmark-yok_s2025`.
  - Ölçülen: 0.4436 ± 0.0098, Δ −0.0013 (−0.29%), p = 0.773. Eşleştirilmiş Δ sd 0.0068, r = +0.82.
  - Ufuk: −0.78 / −0.15 / +0.19 / −0.44%, yani 3/4. h336 0.4010 (vs ep336eq +7.97%).
  - Karar: **TUT — ayırt edilemez** (kart kuralı harfiyen). Kartın diliyle **H0 desteklendi**: etki yok, |Δ| = 0.04 × MDE. Ebeveyn adayı değil.
  - Öğrenilen: §5.3 (tarih sütunu sentetik: 7 587 ardışık gün, haftanın her günü 1 084 kez).

## Sınırlılıklar
- Bütün değerler val'de (760 satır, iki yıllık tek yükseliş). h720'de 41 pencere var (≈ 1 bağımsız hedef parçası) ve düşüş penceresi yok.
- lastnorm'un h720 kazancı ve h336 düşüş-pencere bedeli rejim bağımlı olabilir.
- n = 3. lastnorm'un avg p değerine tek bir ebeveyn seed'inin (s2025 h720) aykırılığı hâkim.
- Süre ölçümleri paylaşımlı yük altında alındı (§4 #5).
- H05 kapısı H01 checkpoint'lerinin μ̂'sını kullandı (kartın ön-kayıtlı yöntemi). Yeniden eğitilmiş bir SAN-μ'nun μ̂'sı ortak faz nedeniyle farklı olmayacaktı (kart: SAN modülü ortak fazda güncellenmiyor).
