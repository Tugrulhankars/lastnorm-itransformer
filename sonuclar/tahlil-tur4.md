# Tur 4 Tahlili (FAZ II, kat protokolü): H11 san-softplus · H12 lastnorm-birlesim · H13 lastnorm-olcek + kat protokolü bulgusu (YALNIZ VALIDATION)

*Hazırlayan: deney-tahlilci (mod 2, tur-tahlili) · 2026-10-14 ~17:10 · FAZ II tur 4 (K4 tur üst sınırı 7)*

- **Girdi:** `orkestrasyon_durumu.md` (FAZ II kat protokolü, yeni seçim kuralı, durma kuralı, K3 yeniden kullanım
  kararı, tur 4 kayıtları; yalnız okundu), `hipotezler/H11–H13.md`, `SIRALAMA.md` Tur 4 bölümü,
  `sonuclar/naif-kat.csv`, `naif-kat_k0.csv`, `varyantlar/san-softplus/kapi.log`, `sonuclar/h11-kapi.csv`,
  `varyantlar/lastnorm-{birlesim,olcek}/dogrula.log`, `gunluk/deney-gunlugu.md`, `runs/BILDIRIM.log`, `sonuclar/tahlil-tur3.md`.
- **Koşumlar (hepsi rc = 0; n = 3 seed × 3 kat, seed {2023, 2024, 2025}, eşleştirilmiş):**
  - baseline: K3 `020531/021024/021448` (resmî val, şef kararıyla yeniden kullanıldı) · K2 `162123/162312/162501` · K1 `162640/162755/162916`
  - lastnorm (ebeveyn, H04): K3 `052508/053601/054213` · K2 `163051/163203/163345` · K1 `163510/163633/163802` · K0 (iç model) `170039/170128/170216`
  - H12 lastnorm-birlesim: K3 `170302/170306/170310` · K2 `170314/170317/170321` · K1 `170325/170329/170332` (CPU son-işlem)
  - H13 lastnorm-olcek: K3 `170336/170340/170343` · K2 `170346/170350/170354` · K1 `170358/170401/170405` (CPU son-işlem)
  - H11 san-softplus: **koşum yok** (CPU kapısı BAŞARISIZ).
- **Betik (WS kökünden, tekrar üretilebilir):** `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/tahlil_tur4.py > sonuclar/tahlil_tur4.log`
  → `sonuclar/tahlil-tur4-kat.csv` (195 satır: varyant × kat × ufuk{96,192,336,720,avg} × seed; val MSE, naif MSE, M/R,
  ebeveyne ve baseline'a eşleştirilmiş mutlak/göreli farklar), `sonuclar/tahlil-tur4-mekanizma.csv` (72 satır: H12/H13
  kat × seed × ufuk ağırlık/α özetleri). Tahmin dosyası okunmaz; tüm sayılar `metrics.json` (val anahtarları) + naif CSV'lerden.
- **md5 (yeni dosyalar):** `scripts/tahlil_tur4.py` 7c594219f9cecaada9e5e8666da5dbb6 · `tahlil_tur4.log`
  7d653f6e7d344a3a7e689eb2eb1c38be · `tahlil-tur4-kat.csv` 9b909589c11ab57b6327b758815e722c · `tahlil-tur4-mekanizma.csv`
  e5e7953ba0fcece5413986db332afe03.
- **Üzerine yazma yok:** test-kilitli olmayan 72 `sonuclar/` dosyasının md5'i koşum öncesi/sonrası aynı (`md5sum -c` 72/72 OK).
  `scripts/` altında 9/11 aynı; **`scripts/rakip_kos.py` ve `scripts/kat_dogrula.py` 17:07'de değişti** — bu tahlil
  tarafından DEĞİL (betik yalnız okur; muhtemelen eşzamanlı P2/ETT işi). Şef bilgisine: tur-4 koşumları 17:04'te bittiği
  için bu değişiklik tur-4 sonuçlarını etkilemez, ama P2 için `--kat`'sız/K1–K3 bit-bit regresyonu yeniden doğrulanmalı.

> **TEST KİLİDİ.** `test_metrics.json`, `preds/test_kilitli/`, `leaderboard.csv`, `sonuclar/test-*`, `leaderboard-test.md`,
> `naif-test.csv`, `karar.md`, `ablasyon.*` okunmadı (md5 denetimi de bu dosyaları dışarıda bıraktı). `metrics.json`'dan
> `test` ile başlayan anahtarlar programatik süzüldü (`assert`). GPU kullanılmadı.
>
> **Test-maruziyeti beyanı (FAZ II):** Exchange test'i 15:42'de açıldı; tur 4'ün tüm kartları ve bu tahlil test sonrası
> yapılmıştır. Aşağıdaki her Exchange cümlesi makalede "test görüldükten sonra" etiketiyle yer alır.

---

## 0. Özet kararlar

| Kart | Ölçüm (3 kat × 3 seed; ebeveyn lastnorm Ā 0.60022) | Kural (FAZ II ön-kayıt) | Karar | Bayrak |
|---|---|---|---|---|
| **H11** san-softplus | CPU kapısı: Softplus ölü oranı K3 0.83–0.88, K1 0.55–0.68 (eşik ≤ 0.05); q_H K3 ≈ 1e-7…6e-6, K1 0.004–0.015 (eşik [0.5, 2]); G3 (ReLU K3 ölü ≥ 0.80) 4/4 GEÇER | KAPI = G1 ∧ G2 ∧ G3 | **ÖN-KOŞUL BAŞARISIZ (ÖLDÜR eşdeğeri)** — koşulmadı | Beklenti ihlali (P(GEÇER) 0.80 idi); mekanizma açıklandı (§4.1) → İNCELE değil |
| **H12** lastnorm-birlesim | Ā **0.60921** (Δ̄ **+1.50%**). Δ_k: K1 **+4.31%** (p 0.002), K2 **−1.38%** (p 0.002), K3 **+7.85%** (p 0.064). Δ_H: −1.12 / +0.13 / −0.78 / **+4.10%** | TUT ⇔ Δ̄ < 0 ∧ ≥ 2/3 kat ∧ ≥ 3/4 ufuk | **ÖLDÜR** (i) Δ̄ ≥ 0, (ii) 1/3 kat, (iii) 2/4 ufuk | Sarı: K1'de C = Repeat (w = 0, %97 dilim); K3 beklentisi (0…+0.3%) ihlal, mekanizma açık |
| **H13** lastnorm-olcek | Ā **0.61149** (Δ̄ **+1.88%**). Δ_k: K1 **+4.31%**, K2 **−0.67%** (p 0.047), K3 **+7.52%**. Δ_H: −1.12 / +0.13 / −0.64 / **+4.86%** | aynı | **ÖLDÜR** (i)(ii)(iii) | Sarı: H12 ile 9 çiftin 5'inde bit-bit aynı (α̂ > 1 yalnız K2'de dilimlerin %14'ü, K3'te %2'si) |

- **Kırmızı bayrak: 0.** İNCELE eşikleri (H12 Δ̄ ≤ −6%; H13 Δ̄ ≤ −15% / h720 ≤ −30%; oracle ihlali) aşılmadı; oracle
  (dilim) ihlali 0/72 hücre. n < 3 yok. Ebeveynle bit-bit aynı sonuç yok (aksine K1'de **naifle** bit-bit aynı; §4.2).
- **Sarı bayrak: 4** — (1) H11 kapı beklentisinin büyük ihlali; (2) H12/H13 K1'de Repeat'e tam çöküş; (3) H12/H13 kat
  yönleri tutarsız (K2 kazanç / K1–K3 kayıp); (4) H11–H13'ün hepsi test sonrası.
- **FAZ II durma sayacı: 0/2 → 1/2.** Tur 4'te Δ̄ ≤ −1% veren TUT yok. Tur 5 de vermezse Exchange araması kapanır (§6).
- **Kat protokolü bulgusu (§3; makalenin ana sonucu adayı):** lastnorm, iTransformer'ı **3/3 katta, 12/12 kat×ufuk
  hücresinde, 36/36 (kat, ufuk, seed) çiftinde** geçiyor: Ā −6.71% (9 çift eşleştirilmiş t p = 0.0125; log-oran p =
  1.3e-4; Wilcoxon p = 0.0039). Ama **naif Repeat'i sağlam biçimde geçmiyor**: K1, K3 evet; K2 hayır (M/R 1.022, 4/4
  ufukta > 1); bilgi katı K0'da da hayır (M/R 1.093). Kat-ort. ufuk M/R'si yalnız h720'de belirgin < 1 (0.952).
- **Öneri (§7):** Exchange aramasını **şimdi kapatıp ETT'ye geçmek** (birincil öneri); şef bir tur daha açmak isterse
  en fazla 2 düşük maliyetli yön (Y1 kısa geriye-bakış, Y2 sabit eşit ağırlık birleşimi).

---

## 1. Ölçüm okuması — kat × ufuk val MSE (3 seed ort. ± sd) ve naif oranı M/R

| Varyant | Kat | h96 | h192 | h336 | h720 | avg | M/R avg |
|---|---|---|---|---|---|---|---|
| naif Repeat | K1 / K2 / K3 | 0.07718 / 0.35543 / 0.12820 | 0.13598 / 0.87787 / 0.22632 | 0.23180 / 1.56229 / 0.39332 | 0.46305 / 1.72617 / 1.14427 | 0.22700 / 1.13044 / 0.47303 | 1 |
| baseline | K1 | 0.09156 ± 0.0008 | 0.14970 ± 0.0015 | 0.24445 ± 0.0013 | 0.45280 ± 0.0092 | 0.23463 ± 0.0025 | 1.034 |
| baseline | K2 | 0.39252 ± 0.0046 | 0.92986 ± 0.0031 | 1.70798 ± 0.0149 | 1.97221 ± 0.0176 | 1.25064 ± 0.0078 | 1.106 |
| baseline | K3 | 0.12486 ± 0.0011 | 0.21139 ± 0.0025 | 0.40027 ± 0.0020 | 1.04286 ± 0.0439 | 0.44485 ± 0.0119 | 0.940 |
| lastnorm | K1 | 0.08391 ± 0.0014 | 0.14081 ± 0.0006 | 0.23382 ± 0.0007 | 0.41192 ± 0.0031 | 0.21762 ± 0.0008 | 0.959 |
| lastnorm | K2 | 0.36047 ± 0.0017 | 0.88928 ± 0.0016 | 1.60014 ± 0.0029 | 1.77110 ± 0.0091 | 1.15525 ± 0.0028 | **1.022** |
| lastnorm | K3 | 0.12096 ± 0.0004 | 0.20837 ± 0.0005 | 0.39197 ± 0.0031 | 0.98992 ± 0.0237 | 0.42780 ± 0.0059 | 0.904 |
| H12 birlesim | K1 | 0.07718 ± 0 | 0.13598 ± 0 | 0.23180 ± 0 | 0.46306 ± 0.00001 | 0.22700 ± 0 | 1.000 |
| H12 birlesim | K2 | 0.35543 ± 0 | 0.88168 ± 0.0013 | 1.58380 ± 0.0029 | 1.73615 ± 0.0087 | 1.13926 ± 0.0030 | 1.008 |
| H12 birlesim | K3 | 0.12639 ± 0.0020 | 0.22240 ± 0.0005 | 0.39295 ± 0.0001 | 1.10376 ± 0.0512 | 0.46137 ± 0.0134 | 0.975 |
| H13 olcek | K1 | = H12 | = H12 | = H12 | = H12 | 0.22700 | 1.000 |
| H13 olcek | K2 | 0.35543 | 0.88168 | 1.58697 ± 0.0033 | 1.76588 ± 0.0058 | 1.14749 ± 0.0005 | 1.015 |
| H13 olcek | K3 | = H12 | = H12 | = H12 | 1.09813 ± 0.0608 | 0.45997 ± 0.0158 | 0.972 |
| lastnorm (bilgi) | K0 | 0.16807 | 0.37651 | 0.81648 | 2.85337 | 1.05361 | 1.093 (naif K0 0.96415) |

- Kat-ort. Ā: naif 0.61016 · baseline 0.64337 · lastnorm **0.60022** · H12 0.60921 · H13 0.61149.
- K3 değerleri resmî val koşumlarıdır (şef kararı: `--kat 3` resmî yolla bit-bit; tur-3 tahlilindeki değerlerle birebir).

## 2. Kart kararları (ebeveyn lastnorm; kural betikte, `tahlil_tur4.log` §2)

### H12 lastnorm-birlesim
- Δ_k: K1 +0.00939 (+4.31%, p 0.0022, 3/3 kötü) · K2 −0.01598 (−1.38%, p 0.0025, 3/3 iyi) · K3 +0.03357 (+7.85%, p 0.064, 3/3 kötü).
- Δ̄ +0.00899 (+1.50%); 9 (kat, seed) çiftinin 3'ü negatif; log-oran t p 0.041 (geo-ort. +3.51%, **kötü yönde anlamlı**).
- Δ_H (kat-ort.): h96 −1.12%, h192 +0.13%, h336 −0.78%, h720 **+4.10%** → 2/4.
- **Karar: ÖLDÜR** — üç koşulun üçü de tetiklendi.
- Kartın ön-kayıtlı ikincilleri: naif hedefi (≥ 11/12 hücre M/R ≤ 1) **5/12** (K1'de 4 hücre tam 1.000; K2'de 1); kat M/R
  K1 1.000 / K2 1.008 / K3 0.975 → 3/3 < 1 **sağlanmadı**. Rakip kuralı (karar dışı) baseline'a karşı TUT-biçimli
  (Ā −5.31%, 2/3 kat, 4/4 ufuk) ama bu lastnorm'un baseline üstünlüğünün kalıntısıdır (9 çift p 0.12), birleşimin değil.
- Mekanizma-devrede-değil kuralı (w ≡ 1 ≥ %90): tetiklenmedi; **tersine** w = 0 oranı K1 0.97 / K2 0.67 / K3 0.67.
- Beklenti (Δ̄ −1.0…+0.5%; K3 0…+0.3%): **ihlal** (Δ̄ +1.50%, K3 +7.85%). Mekanizma §4.2'de ölçülerek açıklandı;
  `dogrula.log` sağlamaları (i)–(v) + sızıntı testleri GEÇTİ → İNCELE değil, ÖLDÜR.

### H13 lastnorm-olcek
- Δ_k: K1 +4.31% (H12 ile bit-bit aynı) · K2 −0.67% (p 0.047) · K3 +7.52% (p 0.089). Δ̄ +0.01126 (+1.88%); log-oran t p 0.027 (kötü).
- Δ_H: −1.12 / +0.13 / −0.64 / **+4.86%** → 2/4. **Karar: ÖLDÜR.** K1 ∧ K2 alt-sonucu: hayır.
- H13 − H12: 9 çiftin 5'i bit-bit 0 (o çiftlerde tüm α̂ ≤ 1); fark yalnız K2 h336/h720 ve K3 h720'nin son dilimlerinde
  (α̂ > 1: K2'de dilimlerin %14'ü, K3'te %2'si). Bu dilimlerde büyütme **zarar** verdi (K2 H13 − H12 +0.72%). "Kısıtsız
  büyütme" sorusu yalnız dar bir dilim kümesinde sınandı ve olumsuz çıktı.
- Beklenti (Δ̄ −6…+6%; "α̂ > 1 dilim oranı h720'de ≥ %50"): Δ̄ aralıkta, **mekanizma beklentisi ihlal** (h720'de α̂ medyanı
  K1 −15.7, K3 −1.2).

### H11 san-softplus (kapı)
- Softplus kolunda σ̂ ortalaması K3'te 0.0010–0.0012 (gerçek dilim std 0.047–0.050), medyan oranı q ≈ 1e-7…6e-6; K1'de
  σ̂ 0.008–0.014, q 0.004–0.015. G1 0/4 + 0/4, G2 0/4 + 0/4. G3 (ReLU K3) 4/4 → ölçüm kodu sağlam.
- **Karar: ÖN-KOŞUL BAŞARISIZ (ÖLDÜR eşdeğeri); hipotez sınanamadı.** Bütçe harcanmadı (CPU 128 s).

## 3. Kat protokolü bulgusu — baseline vs lastnorm vs naif (makalenin ana sonucu adayı)

### 3.1 lastnorm vs iTransformer (rakip)
| | K1 | K2 | K3 | kat-ort. |
|---|---|---|---|---|
| Δ_k (lastnorm − baseline) | **−7.25%** (3/3, p 0.0065) | **−7.63%** (3/3, p 0.0030) | −3.83% (3/3, p 0.23) | **−6.71%** |
| Ufuk Δ (kat içi) | −8.4 / −5.9 / −4.4 / −9.0% | −8.2 / −4.4 / −6.3 / −10.2% | −3.1 / −1.4 / −2.1 / −5.1% | Δ_H −7.2 / −4.1 / −5.4 / −8.5% |

- **Eşleştirilmiş istatistik (kat × seed çiftleri):** 9/9 çift negatif; mutlak fark t p = 0.0125; log-oran t p = 1.3e-4
  (geo-ort. −6.24%); Wilcoxon (kesin) p = 0.0039. Kat × ufuk: **12/12 hücre**, her hücrede 3/3 seed; 36/36 (kat, ufuk,
  seed) log-oranı negatif (geo-ort. −5.73%).
- **Dürüst çerçeve:** seed'ler yalnız eğitim gürültüsünü örnekler; genelleme birimi kattır. Kat düzeyinde n = 3 →
  3/3 işaret testinin tek-yönlü p'si **0.125**. Yani "her katta, her ufukta aynı yön" çok güçlü bir **tutarlılık**
  kanıtıdır; "dönem-üstü anlamlılık" iddiası ise 3 dilimle istatistiksel olarak sınırlıdır. ETTh1/ETTh2 (test'i
  açılmamış) bu iddianın bağımsız sınamasıdır. 36-hücre Wilcoxon p = 2.9e-11 **raporlanmaz** (bağımlı çiftler; iyimser).
- Ön-kayıtlı FAZ II kuralı lastnorm'u baseline'a karşı **TUT** sayar (Δ̄ < 0, 3/3, 4/4). Bu, tur-2'deki tek-dilim
  kararının (−3.83%, p 0.23 — K3'te MDE altı) kat protokolüyle **güçlendiğini** gösterir: K1/K2'de etki iki kat büyük ve
  seed-düzeyinde anlamlı.

### 3.2 Naife karşı (M/R = model / Repeat; Repeat deterministik, p = seed'ler üzerinden tek-örneklem t)
| | K1 | K2 | K3 | Hücre M/R < 1 | Kat |
|---|---|---|---|---|---|
| baseline | 1.034 (p 0.033) — h96 1.186, h192 1.101, h336 1.055, h720 0.978 | **1.106** (p 0.001) — 4/4 > 1 | 0.940 (p 0.055) — h336 1.018 | 4/12 | 1/3 |
| lastnorm | **0.959** (p 0.002) — h96 1.087, h192 1.036, h336 1.009, **h720 0.890** | **1.022** (p 0.004) — 1.014 / 1.013 / 1.024 / 1.026, 4/4 > 1 | **0.904** (p 0.006) — h336 0.997 (p 0.53) | 5/12 | 2/3 |

- Kat-ort. ufuk M/R: baseline 1.086 / 1.041 / 1.076 / 1.040 (0/4 < 1) · lastnorm 1.008 / 0.999 / 1.018 / **0.952** (yalnız h720 belirgin).
- Kat-ort. Ā: lastnorm 0.60022 vs naif 0.61016 (M/R 0.984); kat-eşit geo-ort. M/R 0.960. Bu ortalama kazanç K3'ten ve
  K1 h720'den (41 pencere, ≈ 1 bağımsız parça) gelir.
- **Bilgi katı K0** (yalnız lastnorm, iç model olarak koşuldu, seçimde kullanılmaz): M/R 1.093, 4/4 ufukta > 1. Dört
  ardışık dilimde lastnorm naifi **2/4** dilimde geçiyor (K1, K3), K0/K2'de geçemiyor.
- **Sonuç cümlesi (taslak):** "Son-değer çapalı normalizasyon iTransformer'ı üç ardışık rolling-origin dilimde, her ufukta
  ve her seed'de iyileştirir (−6.7%, 12/12 hücre); ancak hiçbir model rastgele yürüyüşü (Repeat) dilimler boyunca tutarlı
  biçimde geçemez — kazanç uzun ufukta ve dilime bağlıdır." Bu, Meese–Rogoff (1983) bulgusunun derin tahminciler için
  dilim-düzeyinde yeniden üretimidir ve "naifi geçti" iddiası Exchange için **yazılamaz**.

## 4. Mekanizma

### 4.1 H11 — Softplus "yumuşak ölüm"e giriyor
- Softplus türevi sigmoid(z) > 0 olsa da istasyon ön-eğitiminin ilk epoch'unda çıkış ön-aktivasyonu çok negatife itiliyor:
  K3'te σ̂ medyanı ≈ 1e-7·σ_gerçek, yani z ≈ −16…−20; orada sigmoid(z) ≈ 1e-7…1e-9 — ReLU'nun sıfır gradyanından pratikte
  ayırt edilemez. K1'de çöküş daha sığ (ölü 0.55–0.68, q 0.004–0.015) ama yine kapı eşiğinin çok dışında.
- Ön-eğitim val kaybı iki kolda 1e-4 hassasiyetle aynı (ör. K3 h96 0.079733 vs 0.079710): kayıp σ̂'ya **duyarsız**; μ̂
  terimi baskın. Yani kök neden aktivasyonun gradyan sıfırı değil, **σ̂'yı sıfıra iten ve geri çekmeyen kayıp/ölçek
  dinamiği** (hedef dilim std'si ~0.05 z; ilk Adam adımları ön-aktivasyonu aşırı negatife taşıyor). Kartın "G1 neredeyse
  kesin" gerekçesi (σ̂ < 1e-3 için z < −6.9 gerekir) bu dinamiği öngörmedi.
- Öğrenilen: SAN'ın bu veride çalışması için aktivasyon değil σ parametrizasyonu/ilklendirmesi (ör. log-σ, hedef ölçeğe
  ilklenmiş bias) değişmeli — SAN'dan daha büyük sapma, ayrı kart ve kaynak ister; öncelik düşük (§7).

### 4.2 H12/H13 — neden Repeat'e çöktüler
1. **İç-holdout α̂ çoğunlukla negatif → w = clip(α̂) = 0 → C = R.** α̂ = Σ D(Y′ − R′)/Σ D², D = P′ − R′: iç modelin
   Repeat'ten sapması, iç-holdout'ta gerçekleşen sapmayla **ters işaretli**. Ağırlık = 0 oranı K1 0.97, K2 0.67, K3 0.67
   (dilim × ufuk × seed). K1'de ağırlıklar K0 modelinden geliyor; K0'da lastnorm naiften %9.3 kötü (M/R 1.145 / 1.175 /
   1.177 / 1.058) ve α̂ medyanı h720'de −15.7 → K1'in tamamı Repeat'e eşit (h96–h336 bit-bit, h720 1e-5).
2. **"Model vs Repeat" ilişkisi bitişik dilimler arasında kalıcı değil (rejim değişimi).** lastnorm M/R, iç-holdout katı →
   hedef kat: K0→K1 h720 1.058 → **0.890**; K1→K2 h720 0.890 → **1.026**; K2→K3 1.014–1.026 → **0.865–0.997**. Yani
   önceki 760 satırda (≈ 3 yıl günlük kur) modelin naife göre iyi/kötü olması, sonraki dilimde tersine dönebiliyor.
   Ölçüm: iç-holdout α̂ ile kat-val oracle α*_val arasındaki işaret uyumu K2'de 0.20–0.63, K3'te 0.27–0.40; Spearman
   12 hücrenin 8'inde negatif (K3 h720 α̂ medyanı −1.20, oracle +3.36).
3. **Sonuç kat bazında tam olarak bu tersinmeyi izliyor:** naife çekmek yalnız hedef katta modelin naiften kötü olduğu
   yerde kazandırır (K1 h96 +8.0%, K2 4/4 ufuk, K2 Δ_k −1.38%), modelin naiften iyi olduğu yerde kaybettirir (K1 h720
   −12.4%, K3 h720 −11.6%, K3 h192 −6.7%). K3'te kayıp, lastnorm'un Repeat'e üstünlüğünün (−9.6%) çoğunun geri verilmesidir.
4. **Oracle sınırı:** aynı hücrelerde örneklem-içi en iyi dilim ağırlığının kazancı K3'te ≈ 0–0.4%, K2'de 1.4–2.6% —
   yani birleşimin K3'te kazanabileceği bir şey zaten yoktu (lastnorm K3'te α* > 1 tarafında), K2'de ise küçük bir tavanı
   vardı ve H12 bunun ≈ %60–90'ını yakaladı. Kimlik (MSE(w) = MSE_R − 2wα*V + w²V) ile tutarlı; oracle ihlali 0.
5. **Öğrenilen (genel):** Exchange'de modelin naife göre göreli değeri **dönem-içi kalıcı değil**; önceki dönemden kestirilen
   herhangi bir son-işlem ağırlığı (konveks ya da regresyon) kat kuralını geçemez. Bu, combination puzzle'ın (Smith & Wallis
   2009) kayma altındaki uç biçimidir: kestirilen ağırlık işaret hatası yapıyor, yalnız varyans değil.

## 5. Kırmızı / sarı bayrak taraması

| # | Bayrak | Kart | Seviye | Not |
|---|---|---|---|---|
| — | Görülmemiş-iyi eşiği | H12/H13 | yok | Δ̄ > 0; oracle ihlali 0 |
| — | Ebeveynle bit-bit aynı | H12/H13 | yok | K1'de ebeveynle değil **naifle** bit-bit aynı (mekanizma: w = 0) |
| — | n < 3 | hepsi | yok | 3 seed × 3 kat tam; K0 iç modeli 3/3 |
| 1 | Beklenti ihlali (kötü yönde) | H11 | sarı | P(kapı) 0.80 → başarısız; §4.1 açıklama |
| 2 | Mekanizma tersinmesi (w ≡ 0) | H12/H13 | sarı | §4.2; kod sağlamaları GEÇTİ (sızıntı/hizalama) |
| 3 | Kat yönleri tutarsız | H12/H13 | sarı | yalnız K2 kazanç; kural 1/3 ile yakaladı |
| 4 | Test sonrası tasarım | H11–H13 | sarı | kartlarda beyanlı; makalede etiketlenir |
| 5 | Eşzamanlı betik değişikliği | — | bilgi | `rakip_kos.py`, `kat_dogrula.py` 17:07 (bu tahlil değil); P2 regresyonu önerilir |

## 6. Durma kuralı ve bütçe

- **FAZ II durma sayacı (şef ön-kaydı):** tur 4'te Δ̄ ≤ −1% veren TUT yok (H11 kapı, H12 +1.50%, H13 +1.88%) → **1/2**.
  Tur 5'te de yoksa Exchange araması kapanır ve kalan bütçe ek veri setlerine geçer.
- **Bütçe:** toplam 16 GPU-saat; kullanılan ≈ 4.75 (FAZ II ≈ 0.5: kat 0.30 + K0 0.04 + duman/ön-uçuş ≈ 0.15; H11 kapısı,
  H12/H13 CPU). Ek veri setleri için ≥ 4 ayrılmış → Exchange için teorik tavan ≈ 7.25 GPU-saat; bağlayıcı kısıt bütçe değil,
  **aday kalitesi ve durma kuralı**. Tur sayısı: 4/7.
- Zincir ucu değişmedi: **lastnorm** (Ā 0.60022).

## 7. Tur 5 önerisi

**Birincil öneri: Exchange aramasını şimdi kapat, ETT'ye geç (lastnorm ön-kayıtlı kazanan olarak, ayarsız aktarım).**
Gerekçe: (i) Makalenin savunulabilir ana sonucu zaten elde: lastnorm > iTransformer, 3/3 kat, 12/12 hücre (§3.1);
bağımsız sınaması ETT'dir. (ii) K2'de naifi geçmek ≈ −2.2%'lik K2 kazancı ister **ve** K1/K3'te kayıpsız olmalıdır; §4.2
model-vs-naif ilişkisinin bitişik dilimlerde tersine döndüğünü ölçtü → K2'ye özgü kazanç büyük olasılıkla K1/K3'te
kaybettirir (H12 tam bunu yaptı). (iii) Her yeni Exchange sonucu test-sonrası; temiz kanıt değeri düşük. (iv) P(Δ̄ ≤ −1%
TUT | aşağıdaki yönler) tahminim ≤ 0.15; bir tur daha durma sayacını büyük olasılıkla 2/2'ye taşır.

**Şef yine de tur 5 açarsa (≤ 2 kart, hipotez-tasarımcısı yazar; yalnız yön):**
- **Y1 — Kısa geriye-bakış (L 96 → 48), lastnorm üstünde, tek değişken.** Val gerekçesi: K2'de lastnorm 4/4 ufukta naiften
  kötü ve iç-holdout α̂ negatif (modelin sapması, geçmiş penceredeki eğilimi taşıyarak gerçekleşenle ters yönde); daha kısa
  pencere eski rejim bilgisini azaltır, son-değer çapası aynı kalır. Literatür: DLinear (Zeng et al., AAAI 2023) geriye-bakış
  analizinde Exchange'in uzun pencereden fayda görmeyen istisna olduğunu raporlar — **tasarımcı tablo/şekil künyesiyle
  doğrulamalı**. Maliyet ≈ 0.3 GPU-saat (9 koşum, daha kısa girdi). Not: L resmî betik hiperparametresidir; protokol
  değişikliği değil varyant ayarıdır, ama M3 (arama yok) gereği tek değer, ön-kayıtlı.
- **Y2 — Sabit eşit ağırlık birleşimi (w ≡ 0.5, lastnorm + Repeat), CPU, GPU 0.** H12'nin ön-kayıtlı Ablasyon 1'idir;
  kestirim yapılmadığı için §4.2'deki işaret hatası riski yok. Literatür: Smith & Wallis (2009), Bates & Granger (1969).
  Beklenti: K2 kazanır (lastnorm M/R > 1), K3/K1-h720 kaybeder (α* > 1 hücreleri) → kat kuralını geçme olasılığı düşük
  (≈ 0.15); yine de tanısal değeri yüksek ve bedava. Test sonrası fikir olarak beyan edilir.
- **Önerilmeyen:** H11b (σ parametrizasyonu) — SAN'dan büyük sapma, kaynaklı düzeltme yok, iki kart üst üste kapıda/ön-uçuşta
  öldü; ETT'den sonra, bütçe kalırsa. Online ince ayar — protokol dışı (şef tur-3 kararı).

## 8. Kart başına tek satır
- **H11** san-softplus: ÖN-KOŞUL BAŞARISIZ (Softplus σ̂ K3 ≈ 0.001, ölü %83–88; yumuşak ölüm) — hipotez sınanamadı.
- **H12** lastnorm-birlesim: ÖLDÜR (Δ̄ +1.50%, 1/3 kat, 2/4 ufuk; iç-holdout α̂ < 0 → Repeat'e çöküş; yalnız K2 −1.38%).
- **H13** lastnorm-olcek: ÖLDÜR (Δ̄ +1.88%, 1/3 kat, 2/4 ufuk; α̂ > 1 dilimlerin ≤ %14'ünde ve orada zararlı; 5/9 çiftte H12 ile aynı).
- **Kat protokolü:** lastnorm vs iTransformer 3/3 kat, 12/12 hücre, 9/9 çift (Wilcoxon p 0.0039; kat-düzeyi işaret p 0.125);
  naife karşı 2/3 kat (K2 M/R 1.022 kaybeder), 5/12 hücre.
