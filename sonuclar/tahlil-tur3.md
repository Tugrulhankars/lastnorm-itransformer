# Tur 3 Tahlili (K4c): H08 lastnorm-ep336eq (protokol) · H10 lastnorm-merkez · H09 lastnorm-dro (YALNIZ VALIDATION)

*Hazırlayan: deney-tahlilci (mod 2, tur-tahlili) · 2026-10-14 · tur 3 / 4*

- **Girdi:** `sonuclar/leaderboard.md` (15:17), `hipotezler/H08–H10.md`, `SIRALAMA.md` (Tur 3 bölümü, ortak ön-kayıtlı
  eklemeler 1–5), `orkestrasyon_durumu.md` (şef karar kuralı, tur-3 notları; yalnız okundu), `gunluk/deney-gunlugu.md`,
  `sonuclar/tahlil-tur2.md`, `runs/BILDIRIM.log`.
- **Tam koşumlar (aşama 3, n = 3, seed {2023, 2024, 2025}, eşleştirilmiş):**
  - lastnorm (ebeveyn, H04): `052508_s2023`, `053601_s2024`, `054213_s2025` (`053238_s2024` rc = 96 **dışlandı**)
  - lastnorm-ep336eq (H08, PROTOKOL): `140921_s2023`, `141641_s2024`, `142407_s2025` (`072750_s2023` rc = 96 **dışlandı**)
  - lastnorm-merkez (H10): `143314_s2023`, `143820_s2024`, `144341_s2025`
  - lastnorm-dro (H09): `144849_s2023`, `145213_s2024`, `151234_s2025` (`145536_s2025` rc = 96 **dışlandı**)
  - Kıyas: baseline `020531/021024/021448`, baseline-ep336eq (H03) `031233/032853/033500`
- **Betikler (tekrar üretilebilir, WS kökünden):**
  - `scripts/tahlil_tur3.py` (tur-2 betiğinin uyarlaması) → `sonuclar/tahlil_tur3.log`, `tahlil-tur3-ufuk.csv`
    (varyant × ufuk × seed eşleştirilmiş farklar; ebeveyn lastnorm, resmî baseline ve h336 için baseline-ep336eq /
    lastnorm-ep336eq), `tahlil-tur3-ikincil.csv` (6 varyant × 4 ufuk ikincil metrikler).
  - `scripts/hata_analizi.py --runs … --etiket <ad> --cikti-ek _<ad>` →
    `hata-kirilim_{lastnorm-ep336eq,lastnorm-merkez,lastnorm-dro}.csv`, `hata-adim_…csv`, `hata_analizi_…log`.
  - **Üzerine yazma yok:** baseline, tur-1 ve tur-2 çıktılarının (`hata-kirilim*.csv`, `hata-adim*.csv`,
    `hata_analizi*.log`, `tahlil*`, `normalizasyon-kiyas.csv`, 34 dosya) ve `tahlil_tur1/2.py`, `hata_analizi.py`'nin
    md5'i koşum öncesi ve sonrasında aynı (`md5sum -c` 34/34 OK).
- **Sağlamalar:**
  - Şefin ön sayıları (ebeveyn lastnorm'a göre eşleştirilmiş) yeniden hesaplandı ve **tuttu**: lastnorm-dro 0.40632 ±
    0.00502 (Δ −0.02149, p = 0.0492); lastnorm-ep336eq 0.41910 ± 0.00505 (Δ −0.00871, p = 0.0030); lastnorm-merkez
    0.43213 ± 0.01066 (Δ +0.00433, p = 0.687). `leaderboard.md`'deki p'ler **resmî baseline'a** göredir (0.0432 / 0.112 /
    0.190); betik bunları da aynen üretiyor.
  - **M1:** kart mtime'ları H08 06:55, H09 06:56, H10 06:57; ilk ilgili koşumlar 07:05 (ep336eq-duman), 07:18 (dro-duman),
    07:11 (merkez-duman). Hepsi ölçümden önce.
  - **Tekrar üretilebilirlik:** yarım kalan `145536_lastnorm-dro_s2025`'in yazılmış h96/h192/h336 val tahminleri tam
    koşum `151234_s2025` ile `array_equal` (maks. fark 0). rc = 96 dışlaması sonucu etkilemiyor, koşum determinist.
  - `lastnorm-merkez_s2023` tam koşumu ön-uçuşla (`071201`, ≤ 2 epoch) 4/4 ufukta bit-bit aynı. Neden: tam koşumda en iyi
    val epoch'u her ufukta ≤ 2 (h96 val 0.1305 → **0.1282** → 0.1289 → 0.1292 → 0.1283, patience 3 ile ep. 5'te durdu)
    ve en iyi checkpoint geri yükleniyor. Arıza değil, beklenen davranış.

> **TEST KİLİDİ.** `leaderboard.csv`, `test_metrics.json`, `preds/test_kilitli/` ve `replikasyon.md` okunmadı.
> `metrics.json` dosyalarından `test` ile başlayan anahtarlar programatik olarak süzüldü (betikte `assert`).
> Tahminler yalnız `preds/val_h*.npz`'den alındı. `progress.json` yalnız train/val kayıp geçmişi için okundu.

---

## 0. Özet kararlar

| Kart | Ölçüm (val, n = 3, ebeveyn lastnorm 0.42780) | Kural (ön-kayıtlı) | Karar | Bayrak |
|---|---|---|---|---|
| **H08** lastnorm-ep336eq (PROTOKOL) | h336 0.39197 → **0.35713** (Δ **−8.89%**, 3/3, p = 0.0030; 5.9 × MDE). vs baseline-ep336eq 0.37143: **−3.85%** (3/3, p = 0.0030). avg 0.4191 (−2.04%) | S1: Δ_rel ≤ −1.5% ∧ 3/3. S2: E_rel ≤ −1.5% ∧ 3/3 → TOPLANIR | **S1 DOĞRULANDI · S2 TOPLANIR** (zincir dışı, sayaca girmez) | Temiz. Kontrol ufukları 9/9 `array_equal`; İNCELE eşiği (≤ 0.3343) aşılmadı |
| **H10** lastnorm-merkez | avg **0.43213** (Δ **+1.01%**, 1/3 çift, p = 0.687). Ufuk: **+6.11 / +5.94** / −4.17 / +1.40%, yani **1/4** | ÖLDÜR (i) Δavg ≥ 0 **ve** (ii) < 3/4 ufuk | **ÖLDÜR** (ablasyon satırı "lastnorm − σ") | Sarı: kısa ufuk zararı MDE üstü (2.3× / 1.7×). "σ taşıyıcı" etiketi (≥ +50%) yok |
| **H09** lastnorm-dro | avg **0.40632** (Δ **−5.02%**, 3/3 çift, p = 0.049). Ufuk: +0.20 / **+1.80** / +0.60 / **−9.32%**, yani **1/4** | ÖLDÜR (ii): #{Δ_H < 0} = 1 < 3 | **ÖLDÜR** (ufuk kuralı) | Sarı: Δavg'nin **%107**'si h720'den; h720 hariç üç ufuk +0.88% (3/3 kötü, p = 0.005). Beklenti ihlali (iyi yönde, İNCELE eşiği altında). h192 "val trendine aşırı uyum" notu |

- **Kırmızı bayrak: 0.** Görülmemiş-iyi eşikleri aşılmadı. Plasebo yok (H09/H10 4/4 ufukta, H08 h336'da tahminler farklı;
  parametreler eşit). n < 3 yok.
- **Sarı bayrak: 5** (§4: #1–#4, #6).
- **Durma kuralı: sayaç 0/2 → 1/2.** Tur 3'te Δavg ≤ −1%'lik bir TUT yok. H09'un −5.02%'si ön-kayıtlı ufuk kuralıyla
  ÖLDÜR'e düştü, H08 sayaca girmez. Zincirin en iyi val'i `lastnorm` 0.42780 olarak kalıyor (§6).
- **Öneri: K5'e geçiş** (tur 4 açılmadan; gerekçe §8). Alternatif tur-4 seti §8'de.

---

## 1. Ölçüm okuması: ufuk bazında val MSE, Δ, eşleştirilmiş farklar

- Ebeveyn `lastnorm` (seed-ort. ± sd): h96 0.12096 ± 0.00039, h192 0.20837 ± 0.00045, h336 0.39197 ± 0.00309, h720 0.98992 ± 0.02366, avg 0.42780 ± 0.00588.
- "Çift Δ" sütunu s2023 / s2024 / s2025 sırasıyla varyant − ebeveyn farkıdır.
- **MDE (karar):** ön-kayıtlı 3v3 MDE (avg %8.1; ufuk 2.6 / 3.5 / 1.5 / 12.8%).
- **MDE_eş (bilgi):** eşleştirilmiş MDE = (t.975,2 + t.8,2) · sd_d / √3. Kararı değiştirmez.

### H08 lastnorm-ep336eq (PROTOKOL; yalnız h336 değişir)
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 | sd_d · r | p (eşl.) | MDE · MDE_eş | Durum |
|---|---|---|---|---|---|---|---|---|
| h96 / h192 / h720 | = ebeveyn | 0 | 0 / 0 / 0 | — | 0 | — | — | **bit-bit aynı (9/9 `array_equal`, alt_grup farkı 0)** |
| h336 | **0.35713 ± 0.00452** | −0.03484 (**−8.89%**) | −0.0369 / −0.0366 / −0.0310 | 3/3 | 0.00332 · +0.68 | **0.0030** | 1.5% · 2.6% | **MDE üstü (5.9×)** |
| avg | 0.41910 ± 0.00505 | −0.00871 (−2.04%) | −0.0092 / −0.0091 / −0.0078 | 3/3 | 0.00083 · +1.00 | 0.0030 | 8.1% · 0.6% | ayırt edilemez (avg kartta karar dışı) |

- **S1 (kartın kuralı):** Δ_rel = −8.89% ≤ −1.5% ve 3/3 → **DOĞRULANDI**. Beklenti h336 0.355–0.372 (−5.1…−9.4%): üst ucunda, içinde.
- **S2:** E_s = −0.0127 / −0.0151 / −0.0151, E_rel = **−3.85%** (p = 0.0030) ≤ −1.5% ve 3/3 → **TOPLANIR**.
  - Çarpımsal toplanırlık noktası 0.3637. Gözlenen 0.3571, yani noktanın **0.0066 altında**: iki etki çarpımsal toplamdan biraz fazlası kadar birleşiyor.
  - Ön-kayıtlı olasılıklar: TOPLANIR 0.40, İKAME 0.50. Sonuç azınlık senaryosu, ama kartın beklenti aralığının içinde (vs ep336eq −4.4…+0.2%).
- **h336 üçlü kıyas:** vs lastnorm −8.89%, vs baseline-ep336eq −3.85%, vs resmî baseline **−10.78%** (3/3, p = 0.0012). Etiket: **"eğitim-bütçesinden bağımsız"** (0.3571 ≤ 0.3714).
- **Epoch-eşit protokolde son-değer çapası vs NST** (lastnorm-ep336eq vs baseline-ep336eq, makale tablosu için):

  | | h96 | h192 | h336 | h720 | avg |
  |---|---|---|---|---|---|
  | Δ | −3.12% | −1.43% | −3.85% | −5.08% | **−4.24%** (3/3, p = 0.20) |

  4/4 ufuk. Resmî protokoldeki H04 etkisi (−3.83%) epoch eşitlemede kaybolmuyor, h336'da büyüyor (−2.08% → −3.85%).

### H10 lastnorm-merkez
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 | sd_d · r | p | MDE · MDE_eş | Durum |
|---|---|---|---|---|---|---|---|---|
| h96 | 0.12835 ± 0.00098 | +0.00739 (**+6.11%**) | +0.0063 / +0.0074 / +0.0085 | 0/3 | 0.00112 · −0.21 | 0.0076 | 2.6% · 2.9% | **MDE üstü, kötü** |
| h192 | 0.22074 ± 0.00100 | +0.01237 (**+5.94%**) | +0.0121 / +0.0120 / +0.0130 | 0/3 | 0.00057 · +0.97 | 0.0007 | 3.5% · 0.9% | **MDE üstü, kötü** |
| h336 | 0.37562 ± 0.00774 | −0.01635 (**−4.17%**) | −0.0178 / −0.0216 / −0.0097 | 3/3 | 0.00608 · +0.68 | 0.043 | 1.5% · 4.8% | MDE üstü, iyi |
| h720 | 1.00382 ± 0.04037 | +0.01391 (+1.40%) | −0.0470 / +0.0180 / +0.0707 | 1/3 | 0.05891 · −0.67 | 0.72 | 12.8% · 18.4% | MDE altı |
| **avg** | **0.43213 ± 0.01066** | **+0.00433 (+1.01%)** | −0.0116 / +0.0040 / +0.0206 | 1/3 | 0.01611 · −0.89 | **0.687** | 8.1% · 11.7% | ayırt edilemez, kötü yönde |

- 3/4 kuralı **1/4**. ÖLDÜR (i) ve (ii) birlikte.
- vs resmî baseline: avg −2.86% (p = 0.19). h96 +2.80%, h192 +4.42% (ikisi de baseline'dan bile kötü), h336 −6.16%, h720 −3.74%.
- h336 0.3756: vs ep336eq +1.13% (1/3), vs lastnorm-ep336eq +5.18% → **bütçe-bağımlı**.
- İNCELE eşikleri (avg ≤ 0.3722; ufuk −8 / −12 / −15 / −25%) tetiklenmedi. Ters yönlü bilgi notu (Δavg ≥ +50% → "σ taşıyıcı") **tetiklenmedi**.

### H09 lastnorm-dro
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 | sd_d · r | p | MDE · MDE_eş | Durum |
|---|---|---|---|---|---|---|---|---|
| h96 | 0.12120 ± 0.00082 | +0.00024 (+0.20%) | −0.0005 / +0.0000 / +0.0012 | 1/3 | 0.00086 · +0.13 | 0.67 | 2.6% · 2.2% | MDE altı |
| h192 | 0.21212 ± 0.00099 | +0.00375 (**+1.80%**) | +0.0030 / +0.0046 / +0.0037 | 0/3 | 0.00079 · +0.63 | **0.014** | 3.5% · 1.2% | MDE altı, tutarlı kötü |
| h336 | 0.39433 ± 0.00395 | +0.00236 (+0.60%) | +0.0038 / +0.0025 / +0.0007 | 0/3 | 0.00155 · +0.93 | 0.12 | 1.5% · 1.2% | MDE altı |
| h720 | 0.89762 ± 0.02303 | −0.09230 (**−9.32%**) | −0.1307 / −0.0823 / −0.0639 | 3/3 | 0.03449 · −0.09 | 0.044 | 12.8% · 10.8% | MDE altı (0.73×) |
| **avg** | **0.40632 ± 0.00502** | **−0.02149 (−5.02%)** | −0.0311 / −0.0188 / −0.0146 | 3/3 | 0.00858 · −0.23 | **0.049** | 8.1% · 6.2% | ayırt edilemez (0.62 × MDE) |

- 3/4 kuralı **1/4** → **ÖLDÜR (ii)**. Δavg < 0 ve p < 0.05 olsa da ön-kayıtlı kural iki koşulu birlikte istiyor.
- **Seed başına Δavg ayrışımı (Δ_H / 4):**

  | Seed | Δavg | h96 | h192 | h336 | h720 |
  |---|---|---|---|---|---|
  | s2023 | −0.03109 | −0.00012 | +0.00075 | +0.00096 | **−0.03267** |
  | s2024 | −0.01878 | +0.00001 | +0.00114 | +0.00063 | **−0.02057** |
  | s2025 | −0.01459 | +0.00030 | +0.00092 | +0.00018 | **−0.01599** |

  h720 katkısı Δavg'nin **%107**'si. h720 hariç üç ufuk: +0.00212 (**+0.88%**), 3/3 çift kötü, p = 0.0048.
- vs resmî baseline: avg −8.66% (p = 0.043), ama h192 +0.35% ve h720 −13.93%.
- h336: vs ep336eq +6.17%, vs lastnorm-ep336eq +10.41% → bütçe-bağımlı. Kartın "h336 ≈ 0 (|Δ| < 1.5%)" öngörüsü tuttu (+0.60%).
- İNCELE eşikleri tetiklenmedi: avg 0.4063 > 0.3850; h720 −9.3% > −25%; h336 +0.6% (eşik −5%).

### Eşleştirme notu (ortak ekleme 5)
- Çift korelasyonu: H08 h336 +0.68; H10 avg −0.89, h192 +0.97, h720 −0.67; H09 avg −0.23, h336 +0.93, h720 −0.09.
- H10'da eşleştirilmiş MDE (11.7%) 3v3 MDE'nin üstünde; H09'da altında (6.2%). Karar eşiği değişmez (MDE 8.1%).

---

## 2. İkincil metrikler (val, seed-ort.; tanımlar hata-analizi §3.1, tahlil-tur1/2 §2 ile aynı)

Kaynak: `sonuclar/tahlil-tur3-ikincil.csv`. lastnorm ve baseline satırları tur 2 ile birebir aynı.
H08'in h96/h192/h720 satırları lastnorm ile özdeş olduğundan yazılmadı.

| Ufuk · varyant | M/R | worst_block | worst_ratio | rho_shift | top10 M/R | bias P−Y | s_io oranı | std(P)/std(Y) | ufuk-içi trend | D1 M/R | adım-H yön | α* | Düşüş / yükseliş M/R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h96 lastnorm | 0.944 | 0.2166 | 1.790 | 0.237 | 0.957 | −0.043 | 0.66 | 0.169 | +0.033 | 1.018 | 0.63 | 1.33 | 1.030 / 0.872 |
| h96 **merkez** | **1.001** | 0.2226 | 1.734 | 0.223 | 0.946 | −0.043 | 0.74 | 0.210 | +0.043 | 1.073 | 0.61 | **0.45** | 1.076 / 0.939 |
| h96 **dro** | 0.945 | 0.2162 | 1.784 | 0.239 | 0.955 | −0.043 | 0.65 | 0.182 | +0.032 | 1.021 | 0.62 | 1.27 | 1.024 / 0.881 |
| h192 lastnorm | 0.921 | 0.3139 | 1.506 | 0.410 | 0.884 | −0.082 | 0.50 | 0.174 | +0.046 | 1.005 | 0.65 | 1.30 | 1.001 / 0.885 |
| h192 **merkez** | 0.975 | 0.3215 | 1.457 | 0.485 | 0.899 | −0.084 | 0.64 | 0.184 | +0.049 | 1.055 | 0.61 | 1.32 | 1.082 / 0.929 |
| h192 **dro** | 0.937 | 0.3152 | 1.486 | 0.392 | 0.879 | −0.075 | 0.51 | 0.190 | +0.051 | 1.014 | 0.63 | 1.20 | **1.054** / 0.886 |
| h336 lastnorm | 0.997 | 0.4503 | 1.149 | 0.661 | 0.966 | −0.241 | 0.38 | 0.119 | −0.026 | 1.016 | 0.45 | 0.77 | 0.981 / 1.000 |
| h336 baseline-ep336eq | 0.944 | 0.4307 | 1.160 | 0.631 | 0.996 | −0.217 | 0.42 | 0.124 | −0.020 | 1.040 | 0.55 | 1.51 | 0.920 / 0.949 |
| h336 **lastnorm-ep336eq** | **0.908** | **0.4300** | 1.204 | **0.610** | **0.815** | −0.213 | 0.39 | 0.132 | **+0.007** | 1.008 | **0.61** | 2.31 | **0.942 / 0.901** |
| h336 **merkez** | 0.955 | 0.4260 | 1.134 | 0.546 | 0.909 | **−0.196** | **0.49** | 0.181 | +0.080 | 1.054 | 0.63 | 2.01 | **1.101** / 0.926 |
| h336 **dro** | 1.003 | 0.4579 | 1.161 | 0.665 | 0.963 | −0.244 | 0.36 | 0.138 | −0.044 | 1.014 | 0.44 | 0.37 | 0.968 / 1.010 |
| h720 lastnorm | 0.865 | 1.5706 | 1.587 | 0.768 | 0.968 | −0.785 | 0.19 | 0.177 | +0.137 | 0.850 | 0.85 | 3.65 | n<10 / 0.865 |
| h720 **merkez** | 0.877 | 1.5967 | 1.591 | 0.868 | 0.887 | −0.792 | **0.26** | 0.159 | +0.130 | 0.941 | **1.00** | **7.36** | n<10 / 0.877 |
| h720 **dro** | **0.784** | **1.4351** | 1.599 | 0.838 | 0.866 | **−0.741** | 0.23 | 0.185 | **+0.204** | **0.750** | 0.96 | 4.40 | n<10 / **0.784** |

**Değişken payı ve OOR (seçili):**
- h720 değişken `3` / `0` MSE: lastnorm 2.408 / 1.855 → dro **2.164 / 1.711** → merkez 2.489 / 2.050.
- h720 dro değişken M/R'leri 0.63–0.95, yani kazanç sekiz değişkene yayılmış (`hata_analizi_lastnorm-dro.log` [2]).
- h336 lastnorm-ep336eq: `3` 1.360 → 1.251, `0` 0.508 → 0.438.
- Tahminlerin train aralığı dışına çıkma oranı (h96 / h192 / h336 / h720), gerçek 0.80 / 0.83 / 0.85 / 0.78:
  - lastnorm 0.75 / 0.73 / 0.61 / 0.00
  - merkez 0.74 / 0.73 / 0.64 / 0.00 (H06 tipi çöküş **yok**)
  - dro 0.75 / 0.73 / 0.60 / 0.00

**H09 DRO tanısı (metrics.json, son epoch, 3-seed ort.):**

| Ufuk | q (G1 en eski → G4 en yeni) | q_max (seed) | Grup train MSE | argmax |
|---|---|---|---|---|
| h96 | 0.30 / 0.14 / 0.14 / 0.43 | 0.40 / 0.44 / 0.44 | 0.145 / 0.079 / 0.079 / 0.173 | G4 (3/3) |
| h192 | 0.21 / 0.07 / 0.06 / **0.67** | 0.64 / 0.58 / 0.78 | 0.273 / 0.167 / 0.146 / 0.378 | G4 (3/3) |
| h336 | 0.25 / 0.21 / 0.20 / 0.35 | 0.35 / 0.34 / 0.35 | 0.445 / 0.322 / 0.295 / 0.660 | G4 (3/3) |
| h720 | **0.73** / 0.07 / 0.14 / 0.06 | 0.79 / 0.64 / 0.78 | 0.925 / 0.747 / 0.783 / 0.728 | **G1 (3/3)** |

- Mekanizma devrede: kartın "q_max ≥ 0.35" beklentisi h96/h192/h720'de karşılandı. "Uniform kaldı" notu (< 0.30) **yok**.
- Ağırlık h96–h336'da **en yeni** train grubuna, h720'de **en eski** gruba yığılıyor. Her iki durumda da yığıldığı grup, o ufukta train kaybı en yüksek grup.

**Eğitim, süre (yardımcı):**

| | Koşulan epoch (h96 / h192 / h336 / h720) | Süre (s) | Not |
|---|---|---|---|
| lastnorm | 10,10,10 / 9,10,10 / 1,1,1 / 10,7,10 | 433 / 352 / 460 | |
| lastnorm-ep336eq | = lastnorm / = / **10,10,9** / = | 424 / 430 / 532 | seed başına tavan 15 dk aşılmadı |
| merkez | **5,8,5 / 5,4,10** / 1,1,1 / 5,7,5 | 285 / 307 / 291 | Erken duruyor. h96 s2023 train kaybı ep. 2'de 0.136 (lastnorm 0.122) |
| dro | 6,8,8 / 6,5,9 / 1,1,1 / 10,7,10 | 189 / 189 / 259 | h96/h192 erken duruyor; h720 epoch sayısı lastnorm ile aynı |

---

## 3. H09'un −5.02%'si neden TUT değil (ve neden şef ön sayısıyla çelişmiyor)

1. **Şefin sayısı doğru, ama ön-kayıtlı kuralın yalnız ilk koşulunu gösteriyor.** Kural (orkestrasyon_durumu şef ön-kaydı,
   H09 kartı "Öldürme kuralı"): TUT = Δavg < 0 **VE** ≥ 3/4 ufukta Δ_H < 0. H09'da iyileşen ufuk yalnız h720. h192 ise 3/3 çiftte
   kötü (p = 0.014).
2. **Avg kazancı en gürültülü ufuktan.** h720 val'de 41 pencere var (≈ 1 bağımsız hedef parçası). Ön-kayıtlı MDE %12.8,
   gözlenen −9.3% = 0.73 × MDE. Kartın Risk 4'ü ("Δavg'nin büyük kısmı yine h720'den gelebilir; karar 3/4 ufuk kuralıyla
   korunur") harfiyen gerçekleşti. Kural tam bu durum için yazılmıştı.
3. **Beklenti ihlali (iyi yönde):** kart beklentisi avg −2…+3%, h720 −6…+6% idi. Gözlenen avg −5.0%, h720 −9.3%.
   - İNCELE eşiklerinin (−10% / h720 −25%) altında, kırmızı bayrak değil.
   - beklenti-ihlali §4(5) üç sorusu:
     - (a) Mekanizma devrede mi? Evet: q_max 0.64–0.79, tahminler farklı, CPU doğrulaması geçti.
     - (b) Etki sürüyor mu? Evet: 3/3 seed, sd ebeveynle aynı.
     - (c) Neden? h720'de q en eski grup G1'e (≈ 1990–1995, en yüksek train kaybı) yığılıyor. Model daha büyük ufuk-içi hareketler öğreniyor: trend +0.137 → +0.204, D1 M/R 0.85 → 0.75, adım-H yön isabeti 0.85 → 0.96, bias −0.785 → −0.741.
   - Val h720 bölümü **tek bir yükseliş** ve düşüş penceresi yok (n_düşüş = 0). Bu nedenle h720 kazancının trend-asimetri
     sınaması val'de **yapılamıyor**. Testin geç dönemi düşük seviyede (K2), kazanç orada tersine dönebilir.
4. **Sonuç:** "zaman-grubu GroupDRO Exchange'de ortalamayı h720 üzerinden düşürüyor, kısa/orta ufukları ise ya değiştirmiyor
   ya da kötüleştiriyor." Bu bir **ufuk ödünleşimi** bulgusu, bir yöntem kazancı değil. FOIL Tab.2'deki ≈ 0 etki
   (Informer, tek değişkenli) ile çelişmiyor: kısa ufuklar burada da ≈ 0.
5. **Eşik gevşetme önerilmez** (çıpa ilkesi). "h720'de DRO" gibi ufka özel bir uyarlama (H09b) val'e bakılarak seçilmiş
   olur (M3, çatallanan yollar). Önerilmez (§8).

---

## 4. Kırmızı / sarı bayrak taraması

| # | Bayrak | Kart | Bulgu | Sonuç |
|---|---|---|---|---|
| 1 | **Ufuk-baskın avg (tek-ufuk kazancı)** | H09 | Δavg'nin %107'si h720'den; h720 hariç +0.88% (3/3 kötü). Leaderboard sırasında dro 1. görünüyor (0.4063) | **SARI.** Kural ÖLDÜR verdi. Leaderboard'da "en iyi val" satırı makalede kazanan diye **sunulamaz**; "başarısız denemeler" tablosuna ufuk kırılımıyla girer |
| 2 | **Trend-asimetri (ön-kayıtlı not)** | H09 | Δavg < 0 iken h192 düşüş M/R 1.001 → **1.054 (+0.053 ≥ 0.05)**, yükseliş 0.885 → 0.886 | **SARI: "val trendine aşırı uyum" notu** (karar değişmez, K5'e) |
| 3 | Beklenti ihlali (iyi yönde) | H09 | avg −5.0% (beklenti −2…+3%), h720 −9.3% (−6…+6%) | **SARI.** Mekanizma ölçüldü (§3, q → G1). İNCELE eşiği altında |
| 4 | Kısa ufuk zararı + h336 kazancı yükseliş pencerelerinden | H10 | h96/h192 +6.1/+5.9% (MDE üstü). h336 −4.2%, ama düşüş M/R 0.981 → **1.101** (+0.12), yükseliş 1.000 → 0.926 | **SARI (bilgi).** Δavg > 0 olduğundan ön-kayıtlı not formel olarak uygulanmıyor. h336 kazancı val'in yükseliş rejimine bağlı |
| 5 | Dayanıklılık bedeli (ikincil) | H08 | h336 worst_block 0.4503 → 0.4300, ama worst_ratio 1.149 → **1.204** (avg en kötü bloktan hızlı düşüyor). Düşüş M/R 0.981 → 0.942 (iyileşti) | Bilgi. Belirgin bedel yok, "ödünleşim" notu gerekmez |
| 6 | Süre ölçümü kirli (R6) | tümü | H09 s2025 ön planda 259 s, arka planda 189 s; H08 s2025 532 s (eşit iş yükünde 424–430 s) | **SARI.** Süreler hız iddiası için kullanılamaz. Tavanlar aşılmadı |
| 7 | Görülmemiş kadar iyi | H08/H09/H10 | H08 h336 0.3571 > 0.3343; H09/H10 eşiklerin hepsinin üstünde | Temiz |
| 8 | Bit-bit / plasebo | H08/H09/H10 | H08 kontrol ufukları 9/9 aynı (beklenen), h336 farklı (0.70–0.97 z). H09 0.08–0.78 z, H10 0.38–1.05 z fark; parametreler eşit | Temiz |
| 9 | n < 3 / dışlama | H08, H09 | İki rc = 96 yarım koşum dışlandı. H09 yarım koşum ve tam koşum h96–h336'da bit-bit aynı (determinizm) | Temiz |
| 10 | Z2 koruma | H10 | D1 M/R h96 1.018 → 1.073 ≤ 1.10 | Not tetiklenmedi. |P1 − son| 0.011 → 0.025: basamak hafifçe geri geliyor, eşik altında |

**Kırmızı bayrak sayısı: 0. Sarı: 5** (#1–#4, #6). #5 ve #10 bilgi.

---

## 5. Mekanizma tahlili

### 5.1 H08: epoch eşitleme ile son-değer çapası toplanıyor
- **S1:** lastnorm'un h336'sı da 1 epoch'la eksik eğitilmiş. 9–10 epoch'la h336 −8.89% iyileşiyor (H03'te baseline üzerinde −7.21%).
- **Şekil:**
  - Ufuk-içi trend −0.026 → **+0.007**, yani işaret döndü.
  - Adım-H yön isabeti 0.45 → 0.61, α* 0.77 → 2.31, std(P)/std(Y) 0.119 → 0.132.
  - Yakınsamış model artık yönlü hareket tahmin ediyor.
  - Bias −0.241 → −0.213 ve s_io 0.375 → 0.391: seviye tavanı (Z1) **hâlâ yerinde**.
- **S2 (TOPLANIR):** Yakınsamış h336'da da son-değer çapası NST'ye göre −3.85% kazandırıyor (1-epoch'ta −2.08%). Kazanç büyüyor.
  - Yani H04'ün h336 etkisi bir eksik-eğitim telafisi değil.
  - Epoch-eşit protokolde lastnorm vs NST: avg **−4.24%**, 4/4 ufuk, 3/3 çift (p = 0.20, MDE altı).
- **Trend-asimetri:** düşüş 0.981 → 0.942 ve yükseliş 1.000 → 0.901, ikisi birlikte iyileşti. "Val trendine aşırı uyum" notu yok.
  - baseline-ep336eq düşüş M/R'si 0.920. lastnorm-ep336eq 0.942 ile bunun hafif üstünde (K5 test blok kırılımında izlenir).
- **Öğrenilen:** "Resmî betiğin h336 1-epoch ayarı son-değer çapalı iTransformer'ı da eksik eğitiyor. Epoch eşitlendiğinde
  h336 −8.9% iyileşiyor ve çapanın NST'ye üstünlüğü kaybolmuyor, büyüyor (h336 −2.1% → −3.9%; avg −4.2%, 4/4 ufuk).
  İki etki toplanıyor (çarpımsal toplamdan 0.007 daha iyi)."

### 5.2 H10: σ_lb ölçeklemesi kısa ufukta taşıyıcı, H06 çöküşünün kaynağı seviye
- **Kısa ufuk zararı (Risk 1 + Risk 2 birlikte):**
  - Optimizasyon yavaşlıyor: h96 s2023 epoch-2 train kaybı 0.136, lastnorm'da 0.122. En iyi val epoch'u 2'de kalıyor, erken durma 4–5. epoch'ta.
  - Genlik uyarlaması kayboluyor:
    - α* h96 1.33 → **0.45**: sapmalar Repeat'e göre ölçeksiz kalıyor.
    - Blok M/R B1 0.795 → 0.897: sakin dönemde aşırı tahmin.
    - h96 M/R 0.944 → 1.001: Repeat'i artık geçemiyor.
- **Uzun ufuk (Z1) mekanizması devrede ama MSE'ye dönüşmüyor:**
  - s_io oranı h336 0.375 → 0.494, h720 0.187 → **0.263**. Kartın |Δ| ≥ 0.05 öngörüsü karşılandı.
  - h720 adım-H yön isabeti 1.00, α* 7.36: yön her pencerede doğru, ama sapmalar çok temkinli.
  - h720 bias −0.785 → −0.792 değişmedi.
  - h336 −4.2%'lik kazanç yalnız yükseliş pencerelerinden (düşüş M/R +0.12 kötü).
- **H06 ayrışımı (ön-kayıtlı soru):** σ'yı kaldırmak yalnız **+1.0%** (H06 +392%), tahminlerin train dışı oranı korunuyor (0.74 / 0.73 / 0.64).
  - Normalizasyonun Exchange'deki değeri neredeyse tümüyle **merkezlemeden (seviye taşıma)** geliyor.
  - Ölçekleme kısa ufuklarda ~6% katkı veriyor, uzun ufukta nötr ya da hafif zararlı.
  - tahlil-tur2 §5.2'deki "seviye taşıma" cümlesi doğrulandı, değişiklik gerekmez.
- **Öğrenilen (ablasyon satırı "lastnorm − σ"):** "Son-değer çapalı iTransformer'da σ_lb ölçeklemesini kaldırmak
  (NLinear tarzı yalnız-merkezleme) val_mse_avg'yi +1.0% kötüleştiriyor (p = 0.69). Ufuk ödünleşimi var: h96/h192 +6%,
  h336 −4%, h720 +1%. Normalizasyonun bu verideki katkısı esasen seviye merkezlemesinden geliyor (normsuz +392%); ölçek
  kısa ufuk genliğini ayarlıyor."

### 5.3 H09: zaman-grubu GroupDRO, ufuk ödünleşimi
- §3'te. Kısaca: q en yüksek kayıplı zaman grubuna yığılıyor (h96–h336 en yeni, h720 en eski).
  - h720'de modelin daha büyük ve yönlü hareket tahmin etmesine yol açıyor (val'in tek yükselişinde ödüllendiriliyor).
  - h192'de düşüş pencerelerini kötüleştiriyor (+0.053).
  - worst_block ebeveynden yalnız 2/4 ufukta düşük (h96 marjinal 0.2166 → 0.2162, h720 1.571 → 1.435). "Dayanıklılık-ödünleşimi" notu (≥ 3/4) **tetiklenmedi**.
  - B4 M/R h192 1.066 → 1.140: Z3 aşınması kötüleşti.
- **Öğrenilen:** "Train'i 4 kronolojik gruba bölen GroupDRO (η = 0.01) val_mse_avg'yi −5.0% düşürüyor (p = 0.049), ama
  yalnız h720 üzerinden (−9.3%, MDE %12.8 altı). h96/h192/h336'da etki yok ya da kötüleşme var (+0.2 / +1.8 / +0.6%);
  ön-kayıtlı 3/4 ufuk kuralıyla ÖLDÜR. Hedeflenen Z3 aşınması (B3/B4 M/R) düzelmedi."

---

## 6. Durma kuralı kararı

- **Kural (SIRALAMA Tur 3, açık yazım):** "Tur 3'te en az bir kart TUT alıp Δavg ≤ −1.0% (ebeveyn `lastnorm`'a göre göreli,
  n = 3 eşleştirilmiş) vermezse sayaç 1/2 olur." Sayılanlar H09 ve H10. H08 sayılmaz.
- **Tur 3:** H10 ÖLDÜR (Δavg +1.01%), H09 ÖLDÜR (Δavg −5.02%, ama 1/4 ufuk; ÖLDÜR'ler sayılmaz). H08 protokol kartı.
- **Karar: sayaç 0/2 → 1/2.** Zincirin en iyi val'i değişmedi: `lastnorm` 0.42780 ± 0.00588.
- **Tur 4'ün yeri:** bütçenin 4. ve son turu. Tur 4'te ≥ %1'lik TUT çıkmazsa sayaç 2/2 olur. Her durumda K4 tur 4 sonunda kapanır.

---

## 7. Bütçe

- **Ölçülen tur-3 GPU süresi** (`runs/BILDIRIM.log`, betik §0): **4 128.8 s = 1.15 GPU-saat**. Kalemler:
  - duman + ön-uçuş: 0.21 sa (H08 iki kez koşuldu: 0.06 sa)
  - tam koşumlar 9 × 198–541 s: 0.83 sa
  - iki yarım koşum: 150 s + 211 s, 0.10 sa
- Şefin tahmini ≈ 1.25. Fark 0.1 sa, büyük olasılıkla koşumlar arası yükleme/boşta süreler ve CPU doğrulamaları. Sayaç için **muhafazakâr 1.25** önerilir.
- **Kümülatif:** ≈ 3.0 + 1.25 = **≈ 4.25 GPU-saat / 8** (ölçülenle 4.15). **Kalan ≈ 3.75 GPU-saat.** Tur 3 / 4.
- **K5 GPU ihtiyacı (lastnorm kazanan):** ≈ 0. Epoch-eşit sürüm (H08) ve Ablasyon 1 (H10) tur 3'te koşuldu. Test oturumu kilitli tahminlerden açılır.

---

## 8. Tur 4 / K5 önerisi

### Birincil öneri: **K5'e geçiş (tur 4 açılmadan)**

Gerekçe:
1. **Aday havuzu tükenmiş, P(≥ %1'lik TUT) düşük.**
   - A7 normalizasyon ailesi doygun: yedi biçim ölçüldü (yok / NST / RevIN-afin / SAN / SAN-μ kapısı / son-değer / son-değer−σ); tek TUT son-değer çapası.
   - A5 (DRO) ufuk-tutarsız çıktı. A11/A12 protokol soruları yanıtlandı. H07 (x_mark) nötr.
   - **A9 pilotu** (`tahlil_tur3.log`, CPU, mevcut val tahminleri, karar dışı): 3-seed topluluğu lastnorm'da val_mse_avg'yi yalnız **−0.34%** (baseline'da −0.61%) düşürüyor, yani ≥ %1 eşiğinin altında.
   - Bilinen hiçbir aday ≥ %1 için makul olasılık taşımıyor.
2. **H09 TUT almadığı için birleşim kartı açılamaz.** H08 ∘ H09 birleşimi ve H09'un epoch-eşit sürümü SIRALAMA M2 kuralı gereği ancak iki bağımsız TUT ile açılabilir. Bu koşul yok.
3. **Çoklu karşılaştırma riski.** Aynı 760 satırlık val bölümünde 10 kart ve 3 tur yapıldı. Ek bir tur, val'in tek yükseliş rejimine uyum riskini (sarı #1–#4) testten önce büyütür. Sınırlı kazanç beklentisiyle bu riski almak önerilmez.
4. **Makale B çıkışının malzemesi tamam:** normalizasyon tablosu + epoch-eşit sütunu (H08), ablasyon (H10, H06), başarısız denemeler (H01, H02, H05, H09). K5'in GPU yükü ≈ 0.
5. Kalan ≈ 3.75 GPU-saati harcamamak bütçe kuralını ihlal etmez (üst sınır).

### Alternatif (şef tur 4'ü açmayı seçerse; ebeveyn `lastnorm`, ≤ 2 kart, ≤ 1.0 GPU-saat)

| Aday | Durum | Not |
|---|---|---|
| H08 ∘ H09 birleşimi / dro-ep336eq | **Açılamaz** | H09 ÖLDÜR (M2) |
| H09b "DRO yalnız h720" | **Önerilmez** | Val'e bakılarak ufuk seçimi (M3, çatallanan yollar) |
| A9 seed-ensemble (rakibe aynı hakla) | Önerilmez | Pilot −0.34%. n = 3 bağımsız topluluk için varyant başına +6 seed (≈ 1.3 GPU-saat) |
| **H09-η0 (grup-dengeli ERM, q sabit)**, yalnız mekanizma kartı | Opsiyonel, ≈ 0.25 GPU-saat | Kazanç kartı değil, durma sayacına girmez. h720 kazancı uyarlanan q'dan mı yoksa sabit grup yeniden-ağırlıklandırmasından mı geliyor, bunu ayırır. Yalnız B çıkışının "başarısız denemeler" paragrafını zenginleştirir. Tasarımcı H09'un "TUT alırsa" ablasyonunu ÖLDÜR sonrası mekanizma kartı olarak yeniden ön-kaydetmeli |
| Dish-TS kıyas satırı | Düşük öncelik | Literatürde Exchange'de kötü; yalnız tablo bütünlüğü için |

### K5 ön-planı (test kilidini tahlilci açmaz)
1. **Tek test oturumu**, aynı seed'ler, kilitli tahminler:
   - resmî iTransformer, H03 ep336eq, **lastnorm (kazanan)**, **lastnorm-ep336eq (H08)**, lastnorm-merkez (H10, ablasyon −σ)
   - tablo bütünlüğü için şef kararıyla normsuz / RevIN-afin / SAN
   - lastnorm-dro yalnız "başarısız denemeler" satırı olarak (h720 kazancının testte sürüp sürmediği **raporlanır**; karar değiştirmez, ön-kayıt)
2. **Test blok kırılımı (Q1–Q4) ve trend-asimetri** yan yana: baseline, ep336eq, lastnorm, lastnorm-ep336eq, dro (sarı #1–#4). `hata_analizi.py` test sürümü yalnız K5'te.
3. **Görülmemiş-iyi denetimi:** lastnorm `test_mse_avg` < 0.317 → İNCELE (H04 kartı).
4. **İstatistik (D2):** baseline vs lastnorm ve ep336eq vs lastnorm-ep336eq eşleştirilmiş test + CI. Val → test düşüşü raporlanır.

---

## 9. Kartların "Sonuç" bölümleri
Tahlilci H08, H09 ve H10 kartlarının "Sonuç" bölümlerini bu tahlilin sayılarıyla doldurdu (kart üstü değiştirilmedi).
`gunluk/deney-gunlugu.md` tablosuna tur-3 satırları ve bütçe sayacı eklendi. `orkestrasyon_durumu.md`'ye dokunulmadı.

## Sınırlılıklar
- Bütün değerler val'de: 760 satır, tek yükseliş. h720'de 41 pencere var ve düşüş penceresi yok; H09'un h720 kazancı ve H10'un h336 kazancı val rejimine bağlı olabilir.
- n = 3 seed. H09'un p = 0.049'u tek ufkun (h720) varyansına dayanıyor.
- Süreler paylaşımlı yük ve ön/arka plan farkı altında ölçüldü (sarı #6).
- A9 pilotu tek topluluk (n = 1), yalnız yön/büyüklük bilgisi.
