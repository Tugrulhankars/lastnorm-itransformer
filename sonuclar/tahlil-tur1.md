# Tur 1 Tahlili (K4c): H01 SAN · H02 RevIN-afin · H03 h336 epoch eşitleme (YALNIZ VALIDATION)

*Hazırlayan: deney-tahlilci (mod 2, tur-tahlili) · 2026-10-14 · tur 1 / 4*

- **Girdi:** `sonuclar/leaderboard.md` (val_mse_avg), `hipotezler/H01–H03.md`, `SIRALAMA.md`, `sonuclar/hata-analizi.md`,
  `orkestrasyon_durumu.md` (seçim kuralı), `gunluk/deney-gunlugu.md`, `varyantlar/san/teshis/`.
- **Tam koşumlar (aşama 3, n = 3, seed {2023, 2024, 2025}, eşleştirilmiş):**
  - baseline: `020531_s2023`, `021024_s2024`, `021448_s2025`
  - baseline-ep336eq (H03): `031233_s2023`, `032853_s2024`, `033500_s2025`. Yarım kalan `031950_s2024` (rc = 96, metrics yok) **dışlandı**.
  - revin-afin (H02): `034304_s2023`, `034735_s2024`, `035213_s2025`
  - san (H01): `040042_s2023`, `040547_s2024`, `041102_s2025`
- **Betikler (tekrar üretilebilir, WS kökünden):**
  - `scripts/tahlil_tur1.py` → `sonuclar/tahlil_tur1.log`, `tahlil-tur1-ufuk.csv` (varyant × ufuk × seed, eşleştirilmiş farklar), `tahlil-tur1-ikincil.csv` (varyant × ufuk ikincil metrikler, baseline dahil)
  - `scripts/hata_analizi.py --runs … --etiket <ad> --cikti-ek _<ad>` → `hata-kirilim_<ad>.csv`, `hata-adim_<ad>.csv`, `hata_analizi_<ad>.log`. Etiketler: baseline-ep336eq, revin-afin, san.
  - Baseline çıktıları **üzerine yazılmadı**: `hata-kirilim.csv`, `hata-adim.csv` ve `hata_analizi.log` md5'i koşum öncesi ve sonrasında aynı (48d995de…, 222ac641…, 966ff3b8…).
  - SAN teşhisi: `varyantlar/san/teshis/teshis_checkpoint.py` tam koşum checkpoint'lerinde CPU'da koşuldu (yalnız train/val) → `sonuclar/tahlil_tur1_san_teshis.log`.
- **Sağlamalar:**
  - Benim hesapladığım ort ± std ve eşleştirilmiş p değerleri leaderboard.md ile aynı: 0.4376 ± 0.0126, p = 0.00266; 0.4441 ± 0.0123, p = 0.134; 0.5848 ± 0.0044, p = 0.00435.
  - M1: kart mtime'ları 02:36–02:37, ilk varyant koşumları 02:45 / 02:50 / 02:57.

> **TEST KİLİDİ.** Bu tahlilde `leaderboard.csv`, `test_metrics.json`, `preds/test_kilitli/` ve `replikasyon.md` okunmadı.
> `metrics.json` dosyalarından yalnız `test` ile başlamayan anahtarlar programatik olarak süzüldü. Tahminler yalnız
> `preds/val_h*.npz` dosyalarından alındı.

---

## 0. Özet kararlar

| Kart | Ölçüm (val, n = 3) | Kural | Karar | Kırmızı/sarı bayrak |
|---|---|---|---|---|
| **H03** baseline-ep336eq (PROTOKOL) | h336 0.4003 → **0.3714** (Δ **−7.21%**, 3/3 çift < 0, p = 0.0027). Kontrol ufukları bit-bit aynı. avg Δ −1.62% (MDE altı) | Kartın kendi kuralı: Δ_rel ≤ −1.5% ve 3/3 çift | **DOĞRULANDI** | Sarı: düşüş pencerelerinde M/R 0.77 → 0.92 (val trendine uyum işareti; K5'te izlenmeli) |
| **H02** revin-afin (kontrol) | avg 0.4448 → 0.4441 (Δ **−0.17%**, p = 0.134). Ufuk: +0.28 / +0.09 / −0.18 / −0.27%, yani **2/4** | ÖLDÜR (ii): iyileşen ufuk sayısı < 3 | **ÖLDÜR** (etkisiz, MDE'nin çok altında) | Sarı: trend-asimetri bayrağı biçimsel olarak tetiklendi (h96/h192 düşüş M/R > 1.05), ama aynı değer ebeveynde de var |
| **H01** san | avg 0.4448 → **0.5848** (Δ **+31.46%**, 0/3 çift < 0, p = 0.0043). 4/4 ufukta kötü, Repeat'ten de kötü (+23.6%) | ÖLDÜR (i) ve (ii) | **ÖLDÜR — arıza notuyla** (sonuç SAN fikrinin değil, ölü-ReLU çöküşünün ölçümü) | **Kırmızı** (beklenti tersi): mekanizma tam koşum checkpoint'lerinde ölçüldü ve açıklandı. Std başlığı %95–100 sıfır, omurga devre dışı |

- Kırmızı bayrak sayısı **1**: H01 beklenti tersi. Arıza alarmı kapandı, mekanizma ölçüldü.
- Sarı bayrak sayısı **3**: H03 düşüş-penceresi bedeli, H02 biçimsel trend bayrağı, süre ölçümlerinin kirli olması (§4).
- Görülmemiş-iyi eşiği aşılmadı. n < 3 yok. Ebeveynle bit-bit aynı varyant yok.
- **Mevcut en iyi TUT: yok.** Tur 2'nin ebeveyni `baseline` olarak kalır (§6).

---

## 1. Ölçüm okuması: ufuk bazında val MSE, Δ, eşleştirilmiş farklar

Biçim: varyant seed-ort. ± sd (ebeveyn `baseline`: h96 0.12486 ± 0.00105, h192 0.21139 ± 0.00246,
h336 0.40027 ± 0.00196, h720 1.04286 ± 0.04391, avg 0.44485 ± 0.01191). "Çift Δ" sütunu seed 2023 / 2024 / 2025
sırasıyla varyant − ebeveyn farkıdır. MDE sütunu hata-analizi §6'dan alınan 3v3 göreli değerdir.

### H03 baseline-ep336eq (protokol kartı)
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 çift | p (eşl.) | MDE | Durum |
|---|---|---|---|---|---|---|---|
| h96 | 0.12486 ± 0.00105 | 0 (0.00%) | 0 / 0 / 0 | — | — | 2.6% | bit-bit aynı (tasarım gereği) |
| h192 | 0.21139 ± 0.00246 | 0 (0.00%) | 0 / 0 / 0 | — | — | 3.5% | bit-bit aynı |
| **h336** | **0.37143 ± 0.00447** | **−0.02885 (−7.21%)** | −0.0297 / −0.0309 / −0.0260 | **3/3** | **0.0027** | 1.5% | **MDE üstü** |
| h720 | 1.04286 ± 0.04391 | 0 (0.00%) | 0 / 0 / 0 | — | — | 12.8% | bit-bit aynı |
| avg | 0.43764 ± 0.01255 | −0.00721 (−1.62%) | −0.0074 / −0.0077 / −0.0065 | 3/3 | 0.0027 | 8.1% | ayırt edilemez (ikincil) |

### H02 revin-afin
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 çift | p (eşl.) | MDE | Durum |
|---|---|---|---|---|---|---|---|
| h96 | 0.12521 ± 0.00106 | +0.00035 (**+0.28%**) | +0.00035 / +0.00034 / +0.00035 | 0/3 | 0.0001 | 2.6% | MDE altı (ama tutarlı kötü) |
| h192 | 0.21158 ± 0.00237 | +0.00019 (+0.09%) | −0.00001 / +0.00021 / +0.00037 | 1/3 | 0.227 | 3.5% | MDE altı |
| h336 | 0.39956 ± 0.00206 | −0.00071 (−0.18%) | −0.00080 / −0.00077 / −0.00056 | 3/3 | 0.011 | 1.5% | MDE altı |
| h720 | 1.04003 ± 0.04519 | −0.00282 (−0.27%) | −0.0046 / −0.0029 / −0.0010 | 3/3 | 0.113 | 12.8% | MDE altı |
| **avg** | **0.44410 ± 0.01227** | **−0.00075 (−0.17%)** | −0.00126 / −0.00078 / −0.00021 | 3/3 | 0.134 | 8.1% | **ayırt edilemez** |

3/4 yön kuralı: seed-ort. olarak iyileşen ufuk sayısı **2/4** (h336, h720).

### H01 san
| Ufuk | Varyant | Δ (göreli) | Çift Δ | Δ<0 çift | p (eşl.) | MDE | Durum |
|---|---|---|---|---|---|---|---|
| h96 | 0.16003 ± 0.00288 | +0.03517 (**+28.2%**) | +0.033 / +0.040 / +0.032 | 0/3 | 0.0041 | 2.6% | MDE üstü, kötü |
| h192 | 0.28853 ± 0.01997 | +0.07714 (**+36.5%**) | +0.096 / +0.061 / +0.075 | 0/3 | 0.017 | 3.5% | MDE üstü, kötü |
| h336 | 0.52790 ± 0.03181 | +0.12763 (**+31.9%**) | +0.095 / +0.161 / +0.127 | 0/3 | 0.021 | 1.5% | MDE üstü, kötü |
| h720 | 1.36265 ± 0.01030 | +0.31979 (**+30.7%**) | +0.339 / +0.361 / +0.260 | 0/3 | 0.009 | 12.8% | MDE üstü, kötü |
| **avg** | **0.58478 ± 0.00436** | **+0.13993 (+31.5%)** | +0.141 / +0.155 / +0.123 | 0/3 | 0.0043 | 8.1% | **MDE üstü, kötü** |

3/4 yön kuralı: **0/4**. Repeat'e karşı: +24.8 / +27.5 / +34.2 / +19.1%, ortalamada +23.6%.

### Eşleştirme notu: yöntemsel bulgu, karar kuralını DEĞİŞTİRMEZ

- **H02'de seed çiftleri arası korelasyon r = +1.00.** γ ve β RNG çekmediği için ilk ağırlıklar ve veri sırası aynı.
  - Eşleştirilmiş Δavg'nin sd'si 0.00053, seed sd'si ise 0.0119.
  - Eşleştirilmiş MDE (df = 2, α = 0.05, güç 0.8): (4.30 + 1.06) · 0.00053 / √3 ≈ **0.0016**, yani **%0.37**. 3v3 MDE (%8.1) bunun ~20 katı.
  - Sonuç: RNG akışını koruyan küçük değişiklikler eşleştirilmiş tasarımda çok küçük farklarla da "anlamlı" çıkabilir. Örnek: H02 h96 +0.28%, p = 0.0001.
- Bu p değeri **pratik etki değil, determinizmin ölçüsüdür.** D2 gereği "anlamlı" kelimesi yalnız pratik marjla birlikte kullanılır. Ön-kayıtlı MDE (%8.1) eşiği aynen kalır (çıpa ilkesi).
- SAN'da çift korelasyonu bozuluyor (avg r = −0.92), çünkü RNG akışı değişiyor. Eşleştirme kazancı varyant türüne bağlı.
- Tur 2 kartları eşleştirilmiş Δ sd'sini ve çift korelasyonunu **raporlamalı**. Karar yine ön-kayıtlı kurala göre verilir.

---

## 2. İkincil metrikler (val, seed-ort.; tanımlar hata-analizi §3.1 ile aynı)

Kaynak: `sonuclar/tahlil-tur1-ikincil.csv`. Baseline satırları `hata-analizi.md` ile birebir aynı. Aşağıda varyant / ebeveyn
farkı gösteriliyor. H03'ün h96, h192 ve h720 satırları ebeveynle tıpatıp aynı olduğu için yalnız h336 verildi.

| Ufuk · varyant | M/R | worst_block_mse (blok) | worst_ratio | rho_shift | top10 M/R | bias P−Y | s_io oranı (tahmin/gerçek) | std(P)/std(Y) | D1 M/R | Düşüş-pencere M/R (n) | Yükseliş M/R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h96 baseline | 0.974 | 0.2217 (B3) | 1.776 | 0.185 | 1.01 | −0.035 | 0.68 | 0.196 | 1.45 | 1.076 (218) | 0.890 |
| h96 revin-afin | 0.977 | 0.2220 (B3) | 1.773 | 0.185 | 1.01 | −0.035 | 0.68 | 0.198 | 1.46 | 1.078 (218) | 0.894 |
| h96 san | **1.248** | 0.2367 (B3) | 1.480 | 0.387 | 1.24 | **−0.150** | 0.59 | 0.140 | **2.56** | 1.086 (218) | **1.382** |
| h192 baseline | 0.934 | 0.3136 (B3) | 1.484 | 0.418 | 0.82 | −0.069 | 0.57 | 0.158 | 1.24 | 1.059 (163) | 0.879 |
| h192 revin-afin | 0.935 | 0.3142 (B3) | 1.485 | 0.407 | 0.82 | −0.067 | 0.57 | 0.160 | 1.24 | 1.060 (163) | 0.880 |
| h192 san | **1.275** | 0.3685 (B3) | 1.279 | 0.626 | 1.44 | **−0.247** | 0.47 | 0.167 | **2.52** | 1.030 (163) | **1.383** |
| h336 baseline | 1.018 | 0.4606 (B2) | 1.151 | 0.810 | 1.15 | −0.272 | 0.30 | 0.166 | 1.19 | 0.768 (67) | 1.067 |
| **h336 baseline-ep336eq** | **0.944** | **0.4307 (B3)** | 1.160 | **0.631** | **1.00** | **−0.217** | **0.42** | **0.124** | **1.04** | **0.920 (67)** | **0.949** |
| h336 revin-afin | 1.016 | 0.4594 (B2) | 1.150 | 0.809 | 1.15 | −0.271 | 0.31 | 0.167 | 1.19 | 0.769 (67) | 1.065 |
| h336 san | **1.342** | 0.6331 (B2) | 1.199 | 0.842 | 1.35 | **−0.426** | 0.33 | 0.153 | **2.78** | 0.616 (67) | **1.485** |
| h720 baseline | 0.911 | 1.6597 (B3) | 1.592 | 0.790 | 1.11 | −0.801 | 0.18 | 0.163 | 1.03 | n<10 (0) | 0.911 |
| h720 revin-afin | 0.909 | 1.6555 (B3) | 1.592 | 0.789 | 1.10 | −0.800 | 0.18 | 0.164 | 1.03 | n<10 (0) | 0.909 |
| h720 san | **1.191** | 2.0458 (B3) | 1.501 | 0.675 | 1.16 | **−0.957** | 0.17 | 0.190 | **2.19** | n<10 (0) | 1.191 |

Değişken payı (Z4, `mse_deg3` / `mse_deg0` / diğer-6 ort., CSV'de):
- **H02:** değişmiyor (h720 2.599 / 1.887 / 0.639, baseline 2.607 / 1.884 / 0.642).
- **H03 h336:** her üç grupta düzgün azalma (1.273 / 0.495 / 0.201, baseline 1.338 / 0.541 / 0.220). Yani bedel yok.
- **SAN:** bütün gruplarda kötüleşme (h720 3.250 / 2.652 / 0.833).

Trend-asimetri bayrağı (kart ön-kaydı):
- Pencere yönü sign(mean_d(μ_horizon − μ_lookback)) ile belirlendi. "Düşüş" pencere sayısı n = 218 / 163 / 67 / 0.
- **h720'de düşüş penceresi yok** (n < 10). Bu ufukta raporlanmadı.

Eğitim, süre ve parametre (yardımcı):

| | Koşulan epoch (h96 / h192 / h336 / h720) | Eğitim süresi (s, 3 seed) | Parametre (en büyük ufuk) |
|---|---|---|---|
| baseline | 10,9,8 / 10,10,10 / 1,1,1 / 6,8,5 | 279 / 253 / 268 | 304 720 |
| ep336eq | aynı / aynı / **8,10,7** (en iyi ep 5 / 7 / 4) / aynı | 422 / 352 / 298 | 304 720 |
| revin-afin | aynı | 259 / 262 / 263 | 304 736 (+16) |
| san | **5,9,7 / 5,4,10** / 1,1,1 / 10,5,9 (+5 ön-eğitim epoch'u, en iyi 1–5) | 294 / 287 / 471 | 667 472 (×2.2) |

---

## 3. Bit-bit denetimleri (madde 3 ve 4)

- **H03 kontrol ufukları: DOĞRULANDI.**
  - h96, h192 ve h720'de seed başına `val_mse` farkı tam **0.0** (3 × 3).
  - `preds/val_h*.npz` için `np.array_equal` = True (9/9), maks |fark| = 0.
  - `alt_grup` farkı 0.
  - Yalnız h336 değişiyor: pred maks |fark| 0.73–0.89.
  - Ek tutarlılık: ep336eq'in h336 epoch-1 val kaybı ebeveynle aynı (s2023 0.4001631 = 0.4001631). Yama yalnız epoch sayısına dokunuyor.
- **H02 ebeveynden farklı: DOĞRULANDI** (G7 ve R5/plasebo kontrolü).
  - 4 ufkun ve 3 seed'in hepsinde `val_mse` farkı ≠ 0, `array_equal` = False.
  - Pred maks |fark| 0.008–0.061 z.
  - γ ve β eğitim sırasında hareket etti (§5.2). Değişiklik devrede.
- **H01 ebeveynden farklı.** Pred maks |fark| 0.74–1.56 z.

---

## 4. Kırmızı bayrak taraması

| # | Bayrak | Kart | Bulgu | Sonuç |
|---|---|---|---|---|
| 1 | **Beklentinin tersi** | H01 | +31.5%, 4/4 ufuk kötü, Repeat'ten kötü. Beklenti −5…−20% idi | **KIRMIZI → mekanizma ölçülerek kapatıldı** (§5.1). Kart ÖLDÜR. Sonuç makaleye "SAN çalışmaz" diye **yazılamaz** (beklenti-ihlali §4(5)a: mekanizma devrede değildi). "SAN resmî ayarı bu veride std başlığını öldürüyor" diye yazılabilir. |
| 2 | Görülmemiş kadar iyi | hepsi | H03 h336 0.3714, eşik ≤ 0.3042 → geçmedi. H02 avg 0.4441, eşik ≤ 0.4003 → geçmedi; ufuk eşikleri de geçilmedi. H01 kötü | Temiz |
| 3 | Ebeveynle bit-bit aynı | H02 | Farklı (§3) | Temiz |
| 4 | n < 3 | hepsi | n = 3 (rc = 96 koşum dışlandı; leaderboard bu dışlamayla aynı ort/sd'yi veriyor) | Temiz |
| 5 | İkincil metrikte gizli bedel | **H03** | h336 düşüş pencerelerinde M/R **0.768 → 0.920** (n = 67, +20%), yükseliş pencerelerinde 1.067 → 0.949. Pencere-içi std(P)/std(Y) 0.166 → 0.124: daha fazla yumuşatma. Kazanç val'in yükseliş pencerelerinden geliyor | **SARI.** Daha uzun eğitilen model val'in yükseliş trendine daha iyi uyuyor, düşüşlerde ise kaybediyor. Test geç dönemde düşük seviyede (K2) olduğu için K5'te test blok kırılımı (Q1–Q4) resmî ve epoch-eşit satırlar için yan yana raporlanmalı (H03 risk 2 doğrulandı). Karar değişmez, protokol kartı. |
| 6 | Trend-asimetri bayrağı (ön-kayıt) | H02 | Δavg < 0 ve düşüş M/R h96 1.078, h192 1.060 (> 1.05) → kural **biçimsel olarak tetiklendi** | **SARI.** Ebeveynde de aynı değer var (1.076 / 1.059), H02'ye özgü değil. Baseline'ın özelliği olarak K5'e taşınır. Kart zaten ÖLDÜR. |
| 7 | Süre ölçümü kirli (R6) | H03, H01 | ep336eq s2023 h96 eğitimi 110 s (aynı hesap, ebeveynde 96 s); san s2025 471 s. Paylaşımlı yük ya da bellek baskısı (günlükte iki bellek-temizliği öldürmesi var) | **SARI.** Süreler hız iddiası için kullanılamaz. Yalnız kaba maliyet: H03 h336 7.9 → 53–79 s. |
| 8 | Leaderboard etiketi | H03 | Kart, satırın `protokol-referansı` etiketiyle "en iyi TUT" hesabı dışında gösterilmesini istiyor. leaderboard.md'de "OK" görünüyor | Bilgi (şef): etiket eklenmeli. |

---

## 5. Mekanizma tahlili

### 5.1 H01 neden çöktü: ölü ReLU, tam koşumlarda ölçüldü

- **Tam koşum checkpoint'lerinde** (`teshis_checkpoint.py`, yalnız train/val, CPU), tahmin edilen dilim std'sinin sıfır olma oranı:
  - h96: **1.000 / 1.000 / 1.000** (s2023 / s2024 / s2025, val)
  - h336: **1.000 / 0.998 / 0.962**
  - Tahmin std ortalaması 0–0.0025. Gerçek dilim std'si 0.048–0.050 z.
  - Omurga gradyanı |∇| toplamı h96'da 0.8–1.2e-3 (3 seed). h336'da 1.1e-3 / 3.0e-2 / 1.23. En yüksek değer, std'nin %4'ünün canlı kaldığı s2025'te.
  - Sonuç: `ŷ = y_n · (σ̂ + 1e-5) + μ̂ ≈ μ̂`. iTransformer fiilen devre dışı. Tahmin, SAN ortalama-MLP'sinin dilim başına (6 adım) sabit seviyesi.
- **İzleri ikincil metriklerde görülüyor:**
  - D1 M/R 2.2–2.8: dilim ortalaması adım 1'de son değerden kopuk basamak üretiyor.
  - Tahminin Repeat'ten sapması gerçeğe **ters** işaretli: α* = −0.18 / −0.24 / −0.58 / −3.12 (baseline +0.64 / +0.95 / +0.31 / +2.00).
  - Adım-H yön isabeti 0.42 / 0.43 / 0.37 / 0.53.
  - h96/h192'de omurga epoch'ları 4–5'te erken durdu (kayıp düz).
- **Sistematik mi, seed kazası mı?** 3 seed × 4 ufuk tarama (günlük) + bu tahlilde SAN'ın varsayılan seed'i **2021** ile tekrar
  (`scratchpad/olu_relu_s2021.py`, yalnız istasyon ön-eğitimi, train/val): val'de std-çıktı==0 oranı 1. epoch'tan sonra
  1.00 / 0.98 / 0.97 / 0.91, 5. epoch'ta 1.00 / 0.92 / 0.91 / 0.88.
  - İstasyon ön-eğitimi omurgadan bağımsız. Dolayısıyla bu çöküş PatchTST omurgasıyla da beklenirdi.
  - **Çözülmemiş çelişki:** SAN Tab.10 (PatchTST → SAN, Exchange) −18.7% bildiriyor. Ön-uçuş `diff`'ine göre port hesapça birebir; ölçek (train StandardScaler, z uzayı) ve split (7:1:2) aynı görünüyor.
  - Olası açıklamalar ölçülmedi: (i) resmî SAN koşumunda veri ölçekleme ya da yükleyici farkı; (ii) resmî repoda sonraki bir commit'te aktivasyon düzeltmesi; (iii) bildirilen sonucun std'si sıfır bir "yalnız-ortalama" modelden gelmesi.
  - Bu çelişki çözülmeden makalede "SAN Exchange'de çalışmaz" yazılamaz. Yazılabilecek cümle: "SAN@7e1ca66 resmî Exchange ayarı, bizim hattımızda (seed 2021, 2023–2025) std başlığını ilk ön-eğitim epoch'unda öldürüyor."
- **Öğrenilen (D4 başarısız denemeler tablosu için):** "SAN, iTransformer omurgasında resmî Exchange ayarlarıyla (period_len 6, station_lr 1e-3) val_mse_avg'yi +31.5% kötüleştirdi (n = 3, 4/4 ufuk). Neden: std başlığının son ReLU'su ilk ön-eğitim epoch'unda ölüyor (tahmin σ̂ = 0, pencerelerin %96–100'ünde), de-normalizasyon omurga çıkışını 1e-5 ile ölçekliyor ve model dilim başına sabit bir ortalama tahmincisine iniyor. Ölçülen etki SAN'ın gelecek-istatistik fikrini sınamıyor."

### 5.2 H02 neden nötr kaldı: öğrenilen γ, β özdeşliğe yakın

- Öğrenilen değerler (3 seed × 4 ufuk × 8 değişken):
  - **γ ∈ [0.960, 1.033]**, ort. 0.989, |γ − 1| ort. 0.014.
  - **β ∈ [−0.026, +0.029]**, |β| ort. 0.010.
  - Kartın İNCELE eşiği (γ ≤ 0.05) uzak.
- β'nın işareti seed'ler arasında tutarlı (h192–h720'de 8/8 değişken, h96'da 6/8):
  - `1` ve `4` için β > 0, diğer değişkenler için β < 0.
  - De-normalizasyondaki ofset = −β·σ_lb/γ. Buna göre altı değişkende küçük bir **yukarı** ofset öğrenildi, yani train'in eğilim yönü.
- **Büyüklük hesabı (mekanizma düzeyi açıklama):**
  - Ofset ≈ |β| · σ_lb ≈ 0.010 × 0.15–0.19 ≈ **0.002 z**.
  - Kapatılması gereken bias −0.035 / −0.069 / −0.272 / −0.801 z, yani 1–3 mertebe büyük.
  - γ'nın ±%1–4 sapması pencere içi ölçeği aynı oranda değiştiriyor. Oysa std(P)/std(Y) zaten 0.16–0.20; bunu kapatacak büyüklükte değil.
- Bu yüzden ikincil metriklerin hepsi ebeveynin ±0.01'i içinde (bias, s_io oranı, D1, worst, rho).
- beklenti-ihlali §4(5) üç koşulu sağlanıyor: (a) mekanizma devrede (γ/β hareket etti, tahminler farklı); (b) etki sıfıra yakın; (c) neden sıfır olduğu hesapla açıklandı (σ_lb ile ölçeklenen ofset tavanı, kartın kendi mekanizma satırının öngördüğü gibi). Sonuç **raporlanabilir.**
- **Öğrenilen (D4):** "Afin RevIN, iTransformer'ın afinsiz örnek normalizasyonuna göre ayırt edilemez (−0.17%, 2/4 ufuk). Öğrenilen γ ≈ 1 ± 0.04, |β| ≈ 0.01. Değişken başına ofset σ_lb ile ölçeklendiğinden val'de ≈0.002 z'de kalıyor ve uzun ufuk bias'ını (−0.27…−0.80 z) kapatamıyor. Z1 tavanı normalizasyonun afin serbestliğiyle değil, seviye bilgisinin σ_lb birimine sıkışmasıyla ilgili."

### 5.3 H03 Z5'i ne kadar kapattı

| h336 göstergesi | baseline (1 ep) | ep336eq (8 / 10 / 7 ep) | Beklenen (kart) | Değerlendirme |
|---|---|---|---|---|
| val MSE | 0.4003 | **0.3714 (−7.2%)** | −5…−15% | aralıkta |
| **M/R** | **1.018** | **0.944** | 0.86–0.97 | **Z5 kapandı.** Repeat'ten 3/3 seed'de iyi (−5.7 / −6.6 / −4.4%), h192'nin −6.6%'sıyla aynı düzeyde |
| Adım-H yön isabeti | 0.470 | 0.549 | ↑ | yazı-turanın üstüne çıktı |
| Ufuk içi tahmin trendi | −0.037 (−0.066…−0.007) | −0.020 (−0.026…−0.004) | işaret değişimi | **Sağlanmadı.** Trend hâlâ negatif, yalnız yarıya indi. Gerçek +0.00…+0.54 |
| Ort. bias P−Y | −0.272 | −0.217 | ↓|·| | Repeat'in (−0.240) altına indi |
| s_io oranı | 0.30 | 0.42 | ↑ | kısmen; tavan (Z1) sürüyor |
| D1 M/R | 1.19 | 1.04 | — | Z2 basamağı h336'da büyük ölçüde kayboldu |
| worst_block_mse | 0.4606 (B2) | 0.4307 (B3) | ↓ | −6.5%; en kötü blok B2'den B3'e kaydı. worst_ratio 1.151 → 1.160 (değişmedi) |
| rho_shift / top10 M/R | 0.810 / 1.15 | 0.631 / 1.00 | — | Kayma–hata bağı zayıfladı |
| std(P)/std(Y) | 0.166 | 0.124 | — | daha fazla yumuşatma |
| α* | 0.31 | 1.51 | — | Sapmalar artık doğru yönlü ama küçük (fazla temkinli) |
| Düşüş / yükseliş pencere M/R | 0.768 / 1.067 | **0.920** / 0.949 | — | **Bedel:** kazanç yükseliş pencerelerinden geliyor |

- Val kaybında epoch-1 → en iyi epoch: −8.1 / −8.9 / −7.5%. Kartın iç çıpasının (−8.3…−15.9%) alt ucunda.
- **Yorum.** 1 epoch'luk resmî ayar h336'yı yakınsamamış ve ortalamaya-dönüş önyargılı bırakıyordu (Z5 doğrulandı). Epoch eşitleme bunu kaldırıyor.
- Uzun ufuk Z1 tavanı (trend körlüğü, s_io oranı < 0.5) epoch'la **çözülmüyor**. Hata kütlesinin %60'ı hâlâ h720'de: avg payı ep336eq'de %60.
- **Öğrenilen (protokol notu):** "Resmî betiğin h336 için `--train_epochs 1` ayarı, iTransformer'ı Exchange val'de Repeat'in altına itiyor (M/R 1.018). Öteki ufuklarla eşit 10 epoch'a (patience 3) çıkarmak h336 val MSE'yi −7.2% düşürüyor (3/3 seed, p = 0.003) ve M/R'yi 0.944'e indiriyor. Kazanç val'in yükseliş pencerelerinde yoğun (düşüşte M/R 0.77 → 0.92)."

---

## 6. H03'ün zincirdeki rolü ve tur 2 ebeveyni

- Kart ön-kaydı açık: H03 **ebeveyn olmaz**, "A7 kartlarının ebeveyni değişmez" (H03 Ablasyon planı (3), SIRALAMA "İstisna"). Bu üst tablo M1 gereği değiştirilemez.
- **Öneri: tur 2'nin resmî ebeveyni `baseline` kalsın. Her tur-2 kartı `baseline-ep336eq`'e karşı h336 karşılaştırmasını ZORUNLU ikincil olarak ön-kaydetsin.**

Gerekçe:
1. **Tek değişken (M2).** `baseline-ep336eq` ebeveyn olursa, resmî iTransformer'a göre her varyant iki karar taşır: epoch yaması + manipülasyon. Tek-değişken kuralı yalnız zincir içi geçerli kalır. Resmî satırla doğrudan kıyas ise (replikasyon ve literatür çıpası buna bağlı) birleşik bir değişikliğe dönüşür.
2. **Ön-kayıt bağlayıcı.** H03 kartı ve SIRALAMA "ebeveyn olmaz" diyor. Değiştirmek "kart sonucuna bakıp kuralı değiştirmek" olur. Yapılacaksa yeni ön-kayıtlı şef kararıyla ve gerekçe yazılarak yapılmalı.
3. **Karar duyarlılığı düşük.** H03'ün avg etkisi −1.62% ve MDE'nin (%8.1) çok altında. Ebeveyn değişikliği tur-2 kararlarını avg üzerinden pratikte etkilemez. Etkilediği yer **3/4 ufuk kuralının h336 oyu.**
4. **Adil karşılaştırma tarafındaki gerçek risk:** h336'da ek eğitim alan bir varyant, resmî ebeveyne karşı h336 oyunu "daha çok eğitim"le kazanabilir. SAN'da 5 ön-eğitim epoch'u, online ince ayar ve A5'te ek kayıp turları bu türdendir. Önlemler:
   - (a) Tur-2 varyantları h336'da resmî epoch'u (1) korur. Epoch bütçesi değiştiren kart (ön-eğitim, ince ayar) eklenen h336 epoch'larını raporlar.
   - (b) Her kart, h336 Δ'sını hem `baseline`'a hem `baseline-ep336eq`'e karşı verir.
   - (c) **h336 oyunu ancak varyantın h336 MSE'si ≤ 0.3714 (ep336eq) ise "eğitim-bütçesinden bağımsız iyileşme"** sayılır. Bu bir raporlama kuralıdır; karar kuralını değiştirmez ve kartlara **koşumdan önce** yazılmalıdır.
5. **K5'e not.** Kazanan çıkarsa makale tablosuna "iTransformer (resmî)" ve "iTransformer (epoch-eşit)" iki satır girer. Kazananın bir de **epoch-eşit sürümü** (varyant + h336 = 10 ep) tek ablasyon satırı olarak koşulmalı. Maliyet ≈ 0.25 GPU-saat.

**Ne zaman `baseline-ep336eq` daha iyi bir ebeveyn olurdu:**
- Kartın mekanizması doğrudan h336/h720 eğitim rejimini hedefliyorsa (A11 ailesi) ya da kazanan zincirin h336 değerlendirmesi yakınsamamış bir modele dayanıyorsa.
- Tur 2'de böyle bir kart önerilmiyor. Bu yüzden değişiklik gerekmiyor.

---

## 7. Tur 2 önerileri (SIRALAMA koşullu dallanmasının güncellemesi: "H01 ÖLDÜ" dalı)

**Bütçe görüşü.**
- Kalan ≈ 6.35 GPU-saat. K5 için ~1.0 ayrılırsa 3 tur için ≈ 5.35 saat kalıyor; tur başına ≤ 3 kart.
- Kart başına tam maliyet 0.25–0.5 saat (duman + ön-uçuş + 3 seed). En pahalı senaryo 9 kart × 0.5 = 4.5 saat.
- **Bağlayıcı kısıt GPU değil, kart sayısı (≤ 9).** Durma kuralına uzak.
- Tur 1'de bütün kartlar 3 seed koşuldu. Ölü-ReLU gibi E-ihlallerinde ön-uçuş sonrası tam koşum yerine öldürme kanıtının n = 1 ön-uçuşla kabulü şefin takdirinde. H01'de n = 3 kanıt alındı (~0.4 saat).

**Hata kütlesi hâlâ nerede:**
- h720 ve h336 avg'nin %80+'ı.
- Z1 tavanı hiçbir kartla değişmedi: s_io oranı 0.18 / 0.30 → en iyi 0.42, trend hâlâ ≈ 0.
- Z2 basamağı (D1 M/R 1.45 / 1.24) kısa ufuklarda olduğu gibi duruyor.
- Z3 (B3/B4 aşınması, rho_shift) değişmedi.

### Tur 2: önerilen 3 kart (sıralı)

| Sıra | Kart (öneri adı) | Aile / hedef | Tek değişiklik (ebeveyn `baseline`) | Beklenti (yön) | A katkısı | B katkısı | Maliyet |
|---|---|---|---|---|---|---|---|
| **1** | **H04 — son-değere çapalı normalizasyon** (`lastnorm`) | A7/A6 · Z2 + Z1 | `use_norm` bloğunda merkezleme μ_lb → x_son (son girdi değeri); σ_lb aynen; de-normalizasyon simetrik. NLinear'ın "son değeri çıkar" fikri (Zeng et al., AAAI 2023 — künye ve Exchange etkisi tasarımcı tarafından kaynaklanmalı) | D1 M/R h96/h192 1.45 / 1.24 → ~1.0; toplamda küçük (−0…−4%). Ölçülmüş mekanizma: β 0.72–0.98 ve corr(P1−son, μ_lb−son) +0.16…+0.68, yani seviye çapası μ_lb'ye çekiliyor | Orta: h96'nın Repeat'e karşı tüm marjı −2.6%, D1 düzeltmesi bunu büyütebilir | **Yüksek:** normalizasyon kıyas tablosunda merkezleme referansının etkisi (μ vs son). Z2'nin nedensel testi | düşük (H02 kancasıyla aynı) |
| **2** | **H05 — SAN-μ (H01b): std başlığı yerine girdi dilimi σ'sı** | A7 · Z1 | SAN normalize ve de-normalize aynen; **yalnız** de-normalizasyonda σ̂ yerine son girdi diliminin σ'sı (H01 Ablasyon planı (2) "yalnız-ortalama" tanımı, ön-kayıtlı). Ölü ReLU bileşeni devreden çıkıyor, gelecek-ortalama tahmini korunuyor | Belirsiz (SAN Tab.10 analoğu −5…−20%, ama tur-1 kanıtı μ̂'nın tek başına Repeat'ten +24% kötü olduğunu gösteriyor). **Ön koşul (CPU, dakikalar):** SAN ortalama-MLP'si val'de dilim μ'sünü naif "son girdi dilimi μ'sü"nden iyi tahmin ediyor mu? Etmiyorsa kart koşulmaz | Yüksek varyans: Z1'i doğrudan hedefleyen tek aile | **Yüksek:** tezin ana mekanizmasını (gelecek-seviye tahmini) ölü bileşen olmadan sınar. Olumsuz çıkarsa güçlü B bulgusu: "val'de gelecek-seviye tahmini Repeat'i geçemiyor" | orta (SAN kodu var; tek satır) |
| **3** | **H06 — `use_norm 0` (normsuz)** | A12 (analiz) · Z1 | resmî bayrak `--use_norm 1 → 0`; kod yok | Belirgin kötüleşme: val hedeflerinin %54–95'i 4 değişkende train aralığı dışında (hata-analizi §2). Yön kaynağı: Non-stationary Transformers (Liu et al., NeurIPS 2022) normalizasyon ablasyonu (tasarımcı künyeyi ve Exchange satırını doğrulamalı) | yok (kontrol) | **Yüksek:** normalizasyon tablosunun "yok" satırı. Kayma altında NST'nin katkısını ölçer. OOR değişkenlerinde (`3`, `0`, `6`) çöküş beklenir. Ucuz ve kodsuz | çok düşük |

Alternatif slotlar ve gerekçeleri:
- **H05 yerine SAN-softplus** (std başlığında ReLU → softplus). Daha "SAN'a sadık" ama kaynak gerekiyor. Softplus ile pozitif ölçek parametresi: DeepAR, Salinas et al., IJF 2020; ölü ReLU: Lu et al., 2019/2020, *Dying ReLU and Initialization*. Künyeler `atif-dogrulayici`dan geçmeli. Ayrıca yeni bir tasarım kararı ekliyor. SAN-μ ise H01'in ön-kayıtlı ablasyonundan geliyor, araştırmacı serbestliği daha az. **Öneri: SAN-μ önce.** Ön koşul geçer ve kazanç görülürse softplus tur 3'te "std tahmininin katkısı" ablasyonu olur.
- **x_mark ablasyonu** (hata-analizi §7) → **tur 3'e** ertelendi.
  - Ucuz: `x_mark_enc = None`. `DataEmbedding_inverted` aynı lineer gömmeyi paylaştığı için parametre sayısı ve RNG akışı büyük olasılıkla değişmez. Bu doğrulanmalı.
  - Bu yüzden §1'deki **eşleştirilmiş tasarımın yüksek duyarlılığı** (%0.4) burada işe yarar: küçük bir sahte-takvim etkisi bile ayırt edilir.
  - Yüksek B değeri (veri artefaktı bulgusu), ama A katkısı yok.
- **FAN** (A7-d) → tur 3, yalnız H04 veya H05'ten biri TUT alırsa ya da normalizasyon tablosu için gerekirse. Exchange periyodik değil ve trend baskın; FAN'ın frekans bileşeni ayıklaması burada zayıf bir mekanizma. Beklenti −8…0% (FAN Tab.3, farklı split; yalnız göreli).
- **Dish-TS** → yalnız kıyas satırı (literatürde Exchange'de hep kötü). Bütçe artarsa tur 4.
- **A5 kayma-dayanıklı kayıp** (zaman-grubu DRO, Z3/Z4) → tur 3. Ebeveyn: tur-2 TUT varsa o, yoksa `baseline`. Beklenti (0…−8%, FOIL Tab.2) MDE altında. Bu yüzden kartta **birincil karar avg'de kalır** ama worst_block_mse ve B3/B4 M/R ön-kayıtlı ikincil olarak öne çıkarılmalı. Tezin "dayanıklılık" eksenine doğrudan katkı (B).
- **Online ince ayar, gecikmeli geri bildirimle** (Z3) → tur 3 veya 4.
  - Tezin kayma-dayanıklılık eksenine en yakın aile. Ancak Exchange için literatürde etki büyüklüğü bulunamadı, beklenti yazılamaz.
  - Yalnız val'de, H-adım gecikmeli etiketle. Test kilidi açılmadan yalnız val akışında koşulur.
  - Uygulama riski yüksek. Ayrı tablo kartı.

### Önerilen tur planı (koşullu)
- **Tur 2:** H04 lastnorm, H05 SAN-μ (CPU ön koşulu geçerse; geçmezse slot x_mark'a), H06 use_norm 0. Tahmini ≈ 1.0–1.2 GPU-saat.
- **Tur 3:**
  - H04 veya H05 TUT ise → ebeveyn o; A5 zaman-grubu DRO + (H05 TUT ise) SAN-softplus std ablasyonu + x_mark.
  - İkisi de ÖLDÜR ise → x_mark, A5 DRO (ebeveyn `baseline`), online ince ayar.
- **Tur 4:** ≥ 2 bağımsız TUT varsa birleşim kartı. Yoksa online ince ayar ya da FAN. A9 seed-ensemble yalnız rakibe de aynı hakla.
- **K5 (~1 saat):** kazananın epoch-eşit sürümü (§6.5) + test oturumu, resmî/epoch-eşit/kazanan satırlar aynı oturumda.
- **B çıkışına en çok katan çekirdek:** normalizasyon kıyas tablosu (yok / NST / RevIN-afin / son-değer / SAN-μ / SAN-resmî-çöküş) × dayanıklılık metrikleri (worst_block, rho_shift, düşüş/yükseliş M/R) + H03 protokol notu + eşleştirme duyarlılığı bulgusu. Tur 2'deki üç kart bu tablonun eksik üç satırını doldurur.

### Tur 2 kartlarına ön-kayıt için önerilen ortak eklemeler (sonuçtan ÖNCE yazılmalı)
1. h336 Δ'sı hem `baseline`'a hem `baseline-ep336eq`'e karşı (§6.4c).
2. Eşleştirilmiş Δ sd'si ve çift korelasyonu raporlanır. Karar ön-kayıtlı MDE ile verilir.
3. Trend-asimetri bayrağında **ebeveyne göre** fark raporlanır (Δ düşüş-M/R). Mutlak > 1.05 eşiği baseline'da zaten tetikli; bu bir tespittir, eşik gevşetmesi değildir. Kartın kendi eşiği değiştirilmez, ebeveyn değeri yanına yazılır.
4. H05 için CPU ön koşulu, H04 için "D1 M/R düzelir ama toplam kötüleşirse ÖLDÜR" (Z2 öldürme kuralı adayı).
5. H06 için "görülmemiş kadar iyi" eşiği ters yönlüdür: normsuz model ebeveynden **iyi** çıkarsa İNCELE (ölçek/sızıntı şüphesi).

---

## 8. Kartların "Sonuç" bölümleri için önerilen metin (şef yazar; tahlilci kartlara dokunmadı)

- **H01:** Koşumlar `20261014-040042_san_s2023`, `…040547_san_s2024`, `…041102_san_s2025`. Ölçülen: val_mse_avg 0.5848 ± 0.0044 (n = 3), Δ +0.1399 (+31.5%), p = 0.0043; 0/4 ufuk iyileşti. Karar: **ÖLDÜR** (kural i ve ii). Arıza notu: std başlığı ölü ReLU (tam koşumlarda %96–100), omurga devre dışı, H01 hipotezinin mekanizması sınanamadı. Öğrenilen: §5.1.
- **H02:** Koşumlar `20261014-034304_revin-afin_s2023`, `…034735_s2024`, `…035213_s2025`. Ölçülen: 0.4441 ± 0.0123, Δ −0.00075 (−0.17%), p = 0.134; 2/4 ufuk. Karar: **ÖLDÜR** (kural ii; etki MDE'nin ~50 kat altında). Öğrenilen: §5.2.
- **H03:** Koşumlar `20261014-031233_baseline-ep336eq_s2023`, `…032853_s2024`, `…033500_s2025` (`…031950_s2024` rc = 96 dışlandı). Ölçülen: val_mse_h336 0.3714 ± 0.0045, Δ −0.0288 (−7.21%), 3/3 çift, p = 0.0027; kontrol ufukları bit-bit aynı; avg 0.4376 ± 0.0126 (−1.62%, ikincil). Karar: **DOĞRULANDI.** Öğrenilen: §5.3 (protokol notu + düşüş-penceresi bedeli).

## Sınırlılıklar
- Bütün değerler val'de (760 satır, iki yıllık tek yükseliş). h720'de düşüş penceresi yok. Trend-asimetri bayrağı uzun ufukta tanımsız.
- n = 3. Eşleştirilmiş p değerleri küçük değişikliklerde determinizmi ölçüyor (§1). Pencere örnekleme belirsizliği MDE'ye dahil değil.
- SAN–literatür çelişkisi (§5.1) çözülmedi. Seed 2021 tekrarı yalnız istasyon ön-eğitimini kapsıyor, uçtan uca SAN koşumu değil.
- Süre ve hız ölçümleri paylaşımlı yük altında alındı (§4 #7).
