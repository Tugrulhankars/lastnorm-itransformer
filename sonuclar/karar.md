# K5 Karar Taslağı: 5.3 Exchange-Rate / iTransformer

*deney-tahlilci (mod 3, karar) · 2026-10-14 · karar kullanıcınındır (genel onay kapsamında şef uygular).*

## 0. Durum beyanı
- **K4 erken kapandı.** Kullanıcı tur 4'ü açmadan K5'e geçti. Durma sayacı **1/2** iken (2/2 değil) çıkış kullanıcı kararıdır ve makalede beyan edilir. Kullanılan bütçe ≈ 4.25 / 8 GPU-saat, tur 3 / 4.
- **Test kilidi bir kez açıldı (D1).**
  - Betik: `scripts/k5_test_ac.py --asama test`. Zaman: 2026-10-14T15:42:23.
  - Ön-kayıtlı kümenin 30 `test_metrics.json` dosyası okundu (10 varyant × 3 seed). md5 listesi `sonuclar/test-acilis.log`'da.
  - Yeni test koşumu yapılmadı. Test değerleri koşum sırasında kilitli yazılmıştı.
  - Oturum sonrası tek teşhis okuması yapıldı (h336 dosya denetimi, §5 S2) ve log'a beyan edildi.
- **Test görüldükten sonra arama/seçim yapılmadı.** Kazanan (`lastnorm`) ve birincil karşılaştırmalar `orkestrasyon_durumu.md` "K5 ŞEF ÖN-KAYDI"nda test'ten önce sabitlendi.
- **Girdiler:**
  - Yalnız val: `leaderboard.md` (leaderboard.py)
  - Yalnız test: `leaderboard-test.md`, `test-sonuclari.csv`
  - `ablasyon.csv` / `ablasyon.md`, `replikasyon.md`, `tahlil-tur1/2/3.md`, `hata-analizi.md`, `gunluk/deney-gunlugu.md`, `hipotezler/H01–H10.md`

## 1. İstatistik (D2), ön-kayıtlı birincil karşılaştırmalar (test, n = 3 eşleştirilmiş)

| | lastnorm vs baseline (resmî protokol) | lastnorm-ep336eq vs baseline-ep336eq (epoch-eşit) |
|---|---|---|
| test_mse_avg (v / ref) | **0.3465 ± 0.0019 / 0.3646 ± 0.0008** | **0.3473 ± 0.0027 / 0.3646 ± 0.0003** |
| Δ (Δ%) | −0.0181 (**−4.96%**) | −0.0173 (**−4.75%**) |
| Çift Δ < 0 | 3/3 | 3/3 |
| p (eşl. t) / Holm (2 test) | **0.0064 / 0.0129** | **0.0070 / 0.0129** |
| %95 t-CI (Δ) | [−0.0243, −0.0118] | [−0.0236, −0.0111] |
| Bootstrap %95 (n = 3, dejenere: çift min–max) | [−0.0206, −0.0155] | [−0.0197, −0.0147] |
| Wilcoxon kesin p | 0.25 (n = 3 alt sınırı; yalnız yön bilgisi) | 0.25 |
| Ufuk yönü | **4/4**: h96 −4.19% (p .028), h192 −3.65% (.010), h336 −5.92% (.0002), h720 −4.93% (.021) | **4/4**: −4.19 / −3.65 / −5.03 / −4.93% |
| test_mae_avg | 0.3951 vs 0.4069 (−2.9%) | 0.3957 vs 0.4069 |
| val Δ% (aynı çift) | −3.83% (p = 0.228, 4/4; MDE altı) | −4.24% (p = 0.20, 4/4) |

- **Validation → test (aşırı-ayar işareti):** kazananın göreli kazancı val'de −3.83%, test'te −4.96%. **Düşüş yok**, yön ve ufuk deseni (4/4) korunuyor.
- **Neden val'de anlamsız, test'te anlamlı?**
  - Test'te seed varyansı çok düşük: sd 0.0008–0.0019, val'de 0.006–0.012.
  - Test bölümü büyük (1517 satır), val'in h720'si 41 pencere.
  - p yalnız başlatma (seed) gürültüsüne göre. Tek test dönemi örnekleme belirsizliği bu p'de yok (§6).
- **Görülmemiş-iyi denetimi (H04 kartı):** test_mse_avg < 0.317 → İNCELE. Ölçülen 0.3465, **tetiklenmedi**.
- **Literatür ölçeği:**
  - Kazanç parametresiz son-değer merkezlemesi için makul: NLinear ↔ Linear Exchange farkı ufukta 1–8.5% (Zeng 2023 Tab.2, H04 kartı).
  - 0.3465 literatür iTransformer'ından (0.360) ve DLinear'dan (0.354, `literatur-cipa.md` DUET/TFB tablosu) düşük.
  - Ama **Repeat düzeyinde** (Zeng 2023 Tab.2: 0.344, drop_last=True dönemi, kendi split'imizde hesaplanmadı). Bkz. sarı bayrak S1.

## 2. Replikasyon sağlaması
- baseline `test_metrics.json` = K3 istisnalı `metrics.json` test_mse_avg. Seed başına |fark| < 1e-12: 0.3643 / 0.3641 / 0.3655.
- Ortalama **0.3646** = `replikasyon.md` 0.3646 ✓. Betikte `assert`.
- K3 YEŞİL (lit 0.358–0.360, tol 0.008) yerinde kalıyor.

## 3. Ablasyon (özet; ayrıntı `ablasyon.md`)
lastnorm'a göre test:

| Değişiklik | test Δ |
|---|---|
| − çapa (= baseline) | +5.2% |
| − σ (H10) | **+10.2%** (0/4, p = 0.023) |
| − ikisi (normsuz) | +178% |
| + ep336eq | +0.2% (p = 0.27, nötr) |

- Her iki bileşen test'te gerekli. **Ablasyon tutarlı.**
- Val σ'nın katkısını küçük göstermişti (+1.0%). tahlil-tur3 §5.2'deki "değer esasen merkezlemeden" cümlesi makaleye **test sonucuyla düzeltilerek** girmeli.

## 4. Validation → test tutarlılığı (bütün ön-kayıtlı küme)

| Karşılaştırma | val Δ% | test Δ% | Yön | Yorum |
|---|---|---|---|---|
| **lastnorm vs baseline** | −3.83 | **−4.96** | korunuyor | kazanan; 4/4 ufukta korunuyor |
| **lastnorm-ep336eq vs baseline-ep336eq** | −4.24 | **−4.75** | korunuyor | protokolden bağımsız |
| baseline-ep336eq vs baseline (H03) | −1.62 (h336 −7.21) | −0.00 (h336 −0.01) | büyüklük kayboldu | protokol etkisi test'e **taşınmıyor** |
| lastnorm-ep336eq vs lastnorm (H08 S1) | −2.04 (h336 −8.89) | +0.22 (h336 +0.95) | **ters** | aynı |
| lastnorm-merkez vs lastnorm (H10) | +1.01 | +10.18 | korunuyor, büyüdü | ÖLDÜR doğru; σ test'te taşıyıcı |
| lastnorm-dro vs lastnorm (H09) | −5.02 (h720 −9.32) | +1.84 (h720 +2.27; 0/4) | **ters** | ön-kayıtlı 3/4 ufuk kuralı val'in h720 kazancını doğru reddetti |
| revin-afin vs baseline (H02) | −0.17 | +0.17 | ters (ikisi de ≈ 0) | nötr |
| xmark-yok vs baseline (H07) | −0.29 | −1.10 (p = 0.098) | korunuyor | nötr / hafif |
| normsuz vs baseline (H06) | +392 | +164 | korunuyor | çöküş |
| **san vs baseline (H01, arıza notlu)** | **+31.46** | **−17.52 (p = 0.002, 4/4)** | **ters** | **KIRMIZI bayrak (§5 K1)** |

- Yön 13 karşılaştırmanın 8'inde korunuyor (`leaderboard-test.md` §2).
- 10 varyantta val ↔ test sıralama korelasyonu Spearman ρ = 0.37 (p = 0.29). normsuz ve san hariç ρ = 0.62 (p = 0.10).
- **Sonuç:** val, kazananın yönünü doğru verdi. Ama val (760 satır, tek yükseliş rejimi) **genel olarak zayıf bir test vekili**: epoch/DRO/σ etkileri ve SAN sıralaması test'te değişiyor.

## 5. Bayraklar

**Kırmızı (kazananın değil, protokol/benchmark düzeyinde): 1**
- **K1: Val'de öldürülen, mekanizması çökmüş SAN test'te en iyi.**
  - Ölçüm: SAN 0.3007, kazanandan −13%, baseline'dan −17.5%, 4/4 ufuk, h720 0.6996.
  - Bu bir sızıntı belirtisi değil. Literatürde SAN + PatchTST Exchange 0.379 → 0.308 (−18.7%, h720 0.659; `literatur-cipa.md`), yani değer literatür ölçeğinde.
  - Ama bizim SAN'da std başlığı ölü ReLU (tahlil-tur1 §5.1). Tahmin fiilen "SAN ortalama-MLP'si": düzgün, düşük genlikli seviye tahmini.
  - Anlam: test dönemi (geç dönem düşük seviye, K2) ortalamaya dönüşü / düzgün tahmini ödüllendiriyor, val (tek yükseliş) cezalandırıyor. **"Tek şampiyon yok"** ve **rejim duyarlılığı** bulgusu.
  - Kural gereği SAN seçilmez: val'de öldürüldü, test'e bakıp seçim yasak.
  - Makalede gizlenmez. "Başarısız denemeler" ve "val–test rejim farkı" tablosunda raporlanır.
  - K6 öncesi önerilen kontrol (seçim değil, raporlama): kilitli test tahminleri üzerinde `hata_analizi.py` blok/trend kırılımı (Q1–Q4) ile san / lastnorm / baseline yan yana. h720'de "düz tahmin" hipotezi doğrulanmadan makale mekanizma iddia etmez.

**Sarı: 5**
- **S1: Kazanan ≈ naif düzey.**
  - lastnorm 0.3465, Repeat ise literatürde 0.344 (Zeng 2023 Tab.2, drop_last=True dönemi, ±küçük sapma).
  - Tez "iTransformer'ı Repeat düzeyine/altına indirir" diye abartılamaz. "Rakibi %5 iyileştirir, naif referansla aradaki farkı kapatır" sınırında kalınmalı.
  - Kendi split'imizde Repeat test MSE'si hesaplanmadı. Model içermez ve seçim sızıntısı yaratmaz; `literatur-cipa.md` satır 183 zaten öneriyor. K6 öncesi şef/kanit-cikarici tamamlar.
- **S2: Epoch protokolü (H03/H08) test'te sıfır.**
  - val h336'daki −7…−9% test'te ≈ 0.
  - Teşhis okuması (log'a beyan): dosya hatası yok, seed değerleri farklı. Test kaybı 10 epoch boyunca ≈ 0.336–0.338'de sabit.
  - "Resmî 1-epoch ayarı test'te kayıpsız" bulgusu. Ayarın test'e bakılarak seçilmiş olabileceği yalnız **hipotezdir**, kanıtı yok; makalede iddia edilmez.
- **S3: p yalnız seed gürültüsüne göre.** Tek test dönemi. Zaman-blok bootstrap ile dönem belirsizliği ölçülmedi; kilitli tahminlerle K6 öncesi yapılabilir, seçim değil.
- **S4: K4 sarı #2 (H09 trend-asimetri) ve #4 (H10 h336) test'te gerçekleşti.** İki "val trendine uyum" uyarısı doğru çıktı. Makalede yöntem dersi olarak yer alır.
- **S5: Erken K5 çıkışı + val'de seçim gücü düşük.** Kazanan val'de MDE altı (p = 0.23) seçildi. Test onayı ön-kayıtlı tek karşılaştırmaya dayanıyor (Holm ile 2 test).

**Temiz:**
- Plasebo/bit-bit yok. ep336eq'de kontrol ufuklarının aynı olması beklenen.
- n < 3 yok. 4 yetim rc = 96 koşum okunmadı.
- Görülmemiş-iyi eşiği tetiklenmedi. Replikasyon tutarlı.

## 6. Üç çıkış: kanıt tablosu

| Kriter | A. GEÇTİ | B. GEÇEMEDİ, ANLAŞILDI | C. TEKRARLANAMADI |
|---|---|---|---|
| Koşul | test p < 0.05 + pratik marj + bayraksız + ablasyon tutarlı | kazanç yok/anlamsız, mekanizma açıklandı | replikasyon SARI/KIRMIZI |
| Kanıt | p = 0.0064 / 0.0070 (Holm 0.013) ✓. Δ −4.96% / −4.75%, 4/4 ufuk, 3/3 çift ✓. Pratik: literatür iTransformer–DLinear farkının (≈ 1.7%) ~3 katı, ama Repeat düzeyinde (S1) ~. Kazanan için kırmızı yok ✓ (K1 protokol düzeyinde). Ablasyon tutarlı ✓ | Kazanç **var ve anlamlı** → B'nin ana koşulu sağlanmıyor. B malzemesi (val–test rejim farkı, normalizasyon ailesi doygunluğu, SAN tersine dönüşü) A'nın analiz bölümüne girer | Replikasyon **YEŞİL** (0.3646 ∈ [0.350, 0.368]) → **uygulanamaz** |
| Durum | **Karşılandı (sınırlılıklarla)** | Kısmen (ikincil anlatı) | Elendi |

### Öneri: **A. GEÇTİ** (★ önerilen), dar ve ölçülü tezle
B'nin analiz öğeleri katkı olarak eklenir (hibrit "iyileştirme + val/test rejim analizi" makalesi).

**Tez (A, önerilen):** "Exchange-Rate'te iTransformer'ın girdi normalizasyonunu pencere ortalaması yerine son gözleme
çapalamak (σ ölçeği korunarak), ilk-adım seviye basamağını ve uzun ufuk seviye yanlılığını azaltarak test MSE'sini
resmî ve epoch-eşit protokollerde sırasıyla %4.96 ve %4.75 düşürür (n = 3, p ≤ 0.007, 4/4 ufuk). Kazanç parametresizdir,
ancak modeli naif tekrar referansının ötesine taşımaz."

**Tez (B, alternatif):** "Tek-rejimli küçük bir validation dönemi Exchange-Rate'te normalizasyon/eğitim-bütçesi
manipülasyonlarının test etkisini güvenilir biçimde sıralayamaz. Val'de öldürülen çökmüş SAN test'te en iyi çıkıyor, epoch
eşitleme ve GroupDRO kazançları test'te kayboluyor ya da tersine dönüyor. Yalnız son-değer çapası iki dönemde de tutarlı."

**Tez (C):** uygulanamaz (replikasyon YEŞİL). Yazılırsa: "iTransformer'ın Exchange L = 96 sonucu (0.3646 ± 0.0008) resmî
betikle tolerans içinde tekrarlanır; h336 1-epoch ayarı val'de eksik eğitim gibi görünür ama test'te kayıpsızdır."

### Katkı listesi (A önerisi; her madde kanıt dosyasıyla)
1. **Replikasyon:** resmî iTransformer@c2426e6, 3 seed, test 0.3646 ± 0.0008, literatür 0.358–0.360 (`replikasyon.md`).
2. **Hata analizi:** rakibin Exchange'deki ölçülmüş zayıflıkları Z1–Z6 (seviye tavanı, ilk-adım basamağı, Repeat'e yakınsama), `hata-analizi.md`, `hata-kirilim.csv`.
3. **Ana sonuç:** son-değer çapalı normalizasyon, test −4.96% (p = 0.0064) / epoch-eşit −4.75% (p = 0.0070), 4/4 ufuk (`leaderboard-test.md`, `test-sonuclari.csv`, `hipotezler/H04.md`, `H08.md`).
4. **Ablasyon:** çapa ve σ ikisi de gerekli. −σ +10.2%, −çapa +5.2%, normsuz +178% (`ablasyon.csv`, `ablasyon.md`, `H06.md`, `H10.md`).
5. **Ön-kayıtlı negatif sonuçlar:** SAN (ölü ReLU arızası), RevIN-afin (γ ≈ 1), SAN-μ kapısı, x_mark, GroupDRO (ufuk ödünleşimi, test'te tersine) (`tahlil-tur1/2/3.md`, `gunluk/deney-gunlugu.md`).
6. **Val → test rejim analizi:**
   - Val, kazananın yönünü koruyor ama 13 karşılaştırmanın 5'inde yön tersine dönüyor.
   - SAN val +31% → test −17.5%. h336 epoch etkisi test'te sıfır.
   - Kaynak: `leaderboard-test.md` §2–3, bu dosya §4.
7. **Süreç şeffaflığı:** ön-kayıtlı kartlar (mtime < koşum), tek test açılışı + md5 logu (`test-acilis.log`), erken K5 çıkışı beyanı.

## 7. Sınırlılıklar (makaleye taslak)
- **3 seed.** Seed varyansı yalnız başlatma gürültüsü. Wilcoxon n = 3'te p < 0.25 veremez, bootstrap CI dejenere.
- **MDE %8.1 (val).** K4 seçimleri MDE altı farklarla yapıldı. Kazanan val'de anlamsızdı (p = 0.23).
- **Küçük val:** 760 satır, tek yükseliş rejimi, h720'de 41 pencere, düşüş penceresi yok. §4: val zayıf bir test vekili.
- **Tek test dönemi** (1517 satır). Dönem/rejim belirsizliği ölçülmedi.
- **Sentetik tarih / lisans:** Exchange-Rate tarih sütunu sentetik, kaynak lisansı "belirsiz-akademik" (K2 `data_manifest.md`). x_mark özelliği bu yüzden anlamsız olabilir (H07 nötr).
- **Erken K5 çıkışı:** durma sayacı 1/2 iken kullanıcı kararı. 4. tur ve ≈ 3.75 GPU-saat kullanılmadı.
- **Test bir kez açıldı** (15:42:23). Sonrasında yalnız teşhis okuması (beyanlı), arama yok.
- **Tek veri seti, tek omurga** (iTransformer). Son-değer çapasının başka veri/modellere genellenmesi iddia edilmez.
- **Süre ölçümleri kirli** (paylaşımlı yük, R6). Hız iddiası yok.
- **Repeat'in kendi split'imizde test değeri** henüz hesaplanmadı (S1).

## 8. Deney kanıt paketi kontrolü (D5)

| Öğe | Durum | Eksik / sahibi |
|---|---|---|
| Günlük | ✓ `gunluk/deney-gunlugu.md` (K5 satırı eklendi) | — |
| Replikasyon tablosu | ✓ `replikasyon.md` | — |
| Ablasyon | ✓ `ablasyon.csv` + `ablasyon.md` | — |
| ≥ 3 seed istatistik (val + test) | ✓ `leaderboard.md` (val), `leaderboard-test.md` (test) | — |
| Başarısız denemeler tablosu | ~ dağınık (tahlil-tur1/2/3, günlük) | Tek tablo, test sütunlu: kanit-cikarici (K6/Faz 2) |
| Test blok kırılımı (Q1–Q4), trend-asimetri | ✗ | `hata_analizi.py` test modu, kilitli tahminlerle (koşum yok). Şef onayıyla K6 öncesi, **yalnız raporlama** |
| Repeat test MSE (kendi split) | ✗ | Model içermez. Şef / kanit-cikarici, K6 öncesi |

## 9. Karar
- **Önerilen çıkış: A (GEÇTİ), dar tez ve B analiz katkılarıyla.**
- Seçimi kullanıcı yapar; genel onay kapsamında şef `karar.md` + `konu.md`'ye yazar.
- Bu taslak test'ten sonra yeni varyant/seçim önermez.

## Şef eki (2026-10-14, test açılışından SONRA, ayarsız referans)
Naif Repeat (son gözlem) TEST MSE, aynı split/scaler/pencereler (`scripts/k5_naif_test.py` → `sonuclar/naif-test.csv`; hedefler `data/hedefler_sl96/test_h*_true.npy` ile birebir doğrulandı): h96 0.0811 · h192 0.1671 · h336 0.3057 · h720 0.8101 · **ort. 0.3410** (MAE 0.3898).
Sonuç: Repeat 0.3410 < lastnorm 0.3465 < iTransformer 0.3646. Hiçbir öğrenilmiş varyant (çöken SAN hariç, 0.3007 — omurgası devre dışı, dilim-ortalaması tahmincisi) Repeat'i geçmiyor. Bu bulgu çıkış seçimini etkiler: A tezi "rakibi geçer" doğru ama "naif referansın altında kalır" ile birlikte yazılmak zorunda; şef önerisi B (analiz) çerçevesi + lastnorm'un anlamlı kazancı birincil bulgu.

## KULLANICI KARARI (2026-10-14)
**Seçilen çıkış: DURDUR — makale yazılmaz.** (Seçenekler A iyileştirme / B analiz / durdur sunuldu; şef önerisi B idi.) K6 açılmadı. Deney paketi olduğu gibi arşivde kalır.

---
## EK — FAZ II/III nihai çıkış kaydı (şef, 2026-10-15 ~06:10; hakem tur-1 B4 yanıtı)
**Süreç tarihçesi (makalede aynen beyan edilir):**
1. 2026-10-14 15:42 Exchange test tek açılış → tahlilci önerisi A (dar tez), şef önerisi B.
2. ~16:25 kullanıcı: **DURDUR** (makale yazılmayacak).
3. ~16:3x kullanıcı DURDUR'u geri aldı: "rakibi geçmeli ve literatürde geçerli olmalı, pozitif iş çıkarmalıyız" → FAZ II (+8 GPU-saat, +3 tur; rolling-origin kat protokolü, ETT aktarımı). FAZ II'nin tüm Exchange araması test görüldükten SONRA yapıldı.
4. FAZ II sonucu: tur 4–5'te TUT yok → Exchange araması kapandı; LastNorm ETT'ye aynen aktarıldı → ETT test (01:30) negatif. H15 VSEL (test-sonrası, keşifsel) ön-tanımlı ölçütü tutmadı.
5. 2026-10-15 01:08 kullanıcı: "bana sormadan ... makaleyi yaz; pozitif ve rakipleri geçmiş şekilde" → şef: dürüstlük sınırları korunarak yazım (K6).
6. Hakem tur-1 BÜYÜK REVİZYON → FAZ III: H16 ileriye dönük rejim sınaması (ön-kayıtlı), SAN kat koşumu, Exchange +7 seed.

**Seçilen çıkış: B (sistematik analiz) + A bileşeni.** Tez: "Son-değer çapası iTransformer'da güçlü kalıcılık (rastgele-yürüyüş benzeri) rejiminde ön-kayıtlı test kazancı verir, ortalamaya dönen/mevsimsel rejimde kaybettirir; rejim yalnız-train bir kalıcılık oranıyla (H16) öngörülebilir mi — ileriye dönük sınandı." H16 sonucu ne çıkarsa çıksın aynen raporlanır (öldürme kuralı H16.md'de).
**Makalede zorunlu beyanlar:** DURDUR→yeniden açılış ve hedef ifadesi; FAZ II'nin test-sonrası olması; R'nin 3 sette sonradan kalibrasyonu; VSEL keşifsel; Repeat'in Exchange test'te LastNorm'dan iyi olması.
