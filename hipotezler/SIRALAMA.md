# Hipotez Sıralaması — 5.3 Exchange / iTransformer (K4)

*Güncelleme: 2026-10-14 · K4 tur 1 / 4 · Hazırlayan: hipotez-tasarımcısı · Girdi: `sonuclar/hata-analizi.md`
(Z1–Z6, yalnız val), `notlar/literatur-cipa.md` §5.5, `manipulasyon-kutuphanesi.md`. Test metriği okunmadı.*

Mevcut en iyi TUT: **yok** → tüm tur-1 kartlarının ebeveyni `baseline` (resmî, h336 = 1 epoch;
`20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`;
val_mse_avg 0.4448 ± 0.0119). Öldürülmüş kart: **yok**.

Bütçe durumu: kalan ≈ 7.6 GPU-saat; K5 (ablasyon + test) için ~1.0 saat ayrılı → K4'e ≈ 6.6 saat. Tur 1 tahmini
toplam ≈ **1.3 GPU-saat** (duman + ön-uçuş + 3 seed tam, üç kart). Bağlayıcı kısıt GPU değil **kart sayısı**
(4 tur × ≤3 kart = ≤12): bu yüzden tur-1 kartları, sonraki turların hangi aileye gideceğini en çok belirleyen
(bilgi değeri yüksek) kartlar olarak seçildi.

## Tur 1 kartları

Skor = beklenen kazanç (val_mse_avg, mutlak %) × başarı olasılığı / maliyet (GPU-saat, uygulama zahmeti dahil
ağırlıklı). "Bilgi değeri" sütunu makalenin analiz (B) çıkışına katkıyı gösterir.

| Öncelik | Kart | Aile | Hedef | Beklenen etki (kaynak) | P(TUT / doğrulama) | Maliyet | Skor | Bilgi değeri |
|---|---|---|---|---|---|---|---|---|
| 1 | **H01** SAN | A7 | Z1, Z3 (hata kütlesinin %81'i h336+h720) | val avg −5…−20% (SAN Tab.10 PatchTST→SAN −18.7%) | ~0.5 (ayırt edilebilir TUT ~0.3) | 0.5 sa GPU + yüksek uygulama (iki aşamalı eğitim) ≈ 0.8 | 12.5 × 0.5 / 0.8 ≈ **7.8** | Yüksek: tezin ana iddiası (kayma altında normalizasyon), Z1 mekanizma testi |
| 2 | **H03** h336 epoch eşitleme (PROTOKOL) | A11/A12 | Z5 | h336 −5…−15%, avg −1…−3% (iç çıpa: ebeveyn eğitim eğrileri epoch-1→en iyi −8…−16%) | ~0.85 (doğrulama) | 0.45 sa GPU + çok düşük uygulama ≈ 0.45 | 2.2 × 0.85 / 0.45 ≈ **4.2** | Yüksek: rakibe adil kontrol; H01'in h336 kazancındaki eğitim-bütçesi karışıklığını ayırır; protokol notu |
| 3 | **H02** RevIN-afin (kontrol) | A7 | Z1/Z2 (kısmen) | val avg −3…+5% (literatur-cipa §5.5 A7-a) | ~0.35 (çoğunlukla "ayırt edilemez") | 0.35 sa GPU + düşük uygulama ≈ 0.4 | 1.0 × 0.35 / 0.4 ≈ **0.9** | Orta-yüksek: H01'in kontrol satırı; katalogdaki RevIN/Dish-TS kıyas tablosunun RevIN satırı |

### Neden bu sıra
1. **H01 önce (öncelik):** En büyük ölçülmüş hata kütlesini (Z1: uzun ufuk bias ve normalize-kayma tavanı 0.68→0.18)
   doğrudan hedefleyen tek aile; literatürde iTransformer'a en yakın analog (instance-norm'lu omurga, PatchTST) −18.7%.
   Başarı olasılığını düşüren iki şey açıkça kartta: (a) beklenen aralığın alt yarısı MDE (%8.1) altında, (b) val
   yükseliş trendine aşırı uyum riski.
2. **H03 ikinci:** Kazancı makale katkısı değil, ama (i) neredeyse kesin ve ucuz, (ii) H01'in h336 kazancını yorumlamak
   için gerekli: ebeveyn h336'da 1 epoch'la eksik eğitilmiş; SAN ise h336'da 5 ek istatistik epoch'u alıyor. H03
   olmadan H01'in h336 kazancı "daha çok eğitim" ile karışır. Ayrıca katalog konu kartının "rakibi adil koş" ilkesine
   hizmet eder. Resmî baseline değişmez (kart başlığında işaretli).
3. **H02 üçüncü:** Kazanç beklentisi nötr; değeri kontrol kolu olmasında. H01 TUT alırsa "SAN'ın kazancı normalizasyon
   katmanına öğrenilebilir serbestlik eklemekten mi?" sorusuna cevap verir. Uygulaması en basit kart (yalnız
   `_build_model`), ayrıca varyant sarmalayıcı altyapısını H01'den önce doğrular.

### Önerilen KOŞUM sırası (öncelikten farklı — gerekçe uygulama riski)
**H03 → H02 → H01.** H03 yalnız bir argv yaması (dakikalar içinde kuyruğa girer, GPU boş beklemez); H02 varyant
sarmalayıcı kancasını (`_build_model` override + `use_norm 0` yaması + metrics etiketleri) test eder; H01 en karmaşık
uygulama (iki aşamalı SAN eğitimi) — onun kodu yazılırken GPU H03/H02 ile dolu kalır. Kartların tamamı koşumdan önce
yazılı olduğu için koşum sırası ön-kaydı etkilemez (M1). Kuyruklar seri (`--paralel 1`, tek GPU).

## Ortak karar kuralı (şef ön-kaydı, orkestrasyon_durumu.md — kartlar bunu aynen kullanır)
- Birincil: `val_mse_avg` (3 seed ort., eşleştirilmiş seed {2023, 2024, 2025}).
- TUT = Δavg < 0 **VE** ≥3/4 ufukta Δ_H < 0. |Δavg| < %8.1 (MDE) → "TUT — ayırt edilemez".
- ÖLDÜR = Δavg ≥ 0 **veya** < 3/4 ufukta iyileşme.
- İNCELE = kartın "görülmemiş" eşiği aşıldı (H01: avg ≤ 0.3460 / −22.2%; H02: avg ≤ 0.4003 / −10%).
- İstisna: **H03 protokol kartıdır**, h336 dışı ufuklar tanım gereği Δ = 0 olduğundan 3/4 kuralı uygulanamaz; kendi
  ön-kayıtlı kuralı (h336 Δ ≤ −1.5% ve 3/3 seed çifti) ile DOĞRULANDI/DOĞRULANMADI/TERS raporlanır ve zincire
  (ebeveyn) girmez.
- Tüm kartlarda ikincil (raporlama, karar dışı): ufuk başına M/R, worst_block_mse/worst_ratio, blok M/R, rho_shift,
  top10_ratio, `alt_grup` (değişken `3`/`0` zorunlu, Z4), adım-1/D1 M/R (Z2), h336/h720 bias ve s_io oranı (Z1
  mekanizması), trend-asimetri bayrağı (düşüş pencerelerinde M/R), süre/parametre/VRAM.

## Ortak risk (her kartın risk bölümünde)
Val dönemi z uzayında tek bir yükseliş; K2'ye göre test geç dönemde (Q3/Q4) düşük seviyede. Val trendine uyan bir
varyant (özellikle seviye/ortalama tahmin eden H01, değişken başına ofset öğrenen H02) testte cezalandırılabilir.
Val'de bunu kısmen yakalamak için trend-asimetri bayrağı ön-kayıtlı; asıl kontrol K5 test blok kırılımı.

## Sonraki turlar için aday havuzu (kart DEĞİL — tur 1 sonuçlarına göre seçilecek)
Tur 1 sonucuna göre dallanma (ön-plan, sonuçtan önce yazıldı):
- **H01 TUT (ayırt edilebilir) ise →** tur 2 ebeveyni `san`; adaylar: A5 kayma-dayanıklı kayıp (zaman-grubu DRO,
  Z3/Z4; FOIL Tab.2 0…−8%), A6 son-değere çapalı artık başlık (Z2; öldürme kuralı: "D1 M/R düzelir ama toplam
  kötüleşirse ÖLDÜR"), Dish-TS kıyas satırı (ebeveyn baseline, kıyas amaçlı; beklenti kötüleşme +25…+104% RevIN'e göre).
- **H01 ÖLDÜR / ayırt edilemez ise →** tur 2: FAN (A7-d, −8…0%, FAN Tab.3 — farklı split, yalnız göreli etki) veya
  `use_norm 0` (normsuz; normalizasyonun kendisinin Exchange'deki değeri — analiz çıkışı), Dish-TS kıyas satırı,
  A6 artık başlık.
- **Her durumda (analiz çıkışı malzemesi):** x_mark (sentetik tarih) token ablasyonu (hata-analizi §7; beklenti
  |Δ| < MDE), online ince ayar gecikmeli geri bildirimle (Z3; literatürde Exchange etkisi BULUNAMADI → beklenti
  yazılamaz, yalnız "ayrı tablo" kartı; tur 3–4).
- **Son itme (tur 4):** yalnız ≥2 bağımsız TUT varsa birleşim kartı; A9 seed-ensemble yalnız rakibe de aynı hakla.

## Kapanmış kartlar
(yok)

---

## Tur 2 (eklendi 2026-10-14, K4 tur 2 / 4 — hiçbir tur-2 koşumu ve H05 ön-koşul ölçümü başlamadan)

*Hazırlayan: hipotez-tasarımcısı · Girdi: `sonuclar/tahlil-tur1.md` (§5 mekanizma, §6 ebeveyn, §7 öneriler),
`sonuclar/leaderboard.md`, `hipotezler/H01–H03.md` (Sonuç bölümleri), `sonuclar/hata-analizi.md`,
`notlar/literatur-cipa.md`, `orkestrasyon_durumu.md`, `gunluk/deney-gunlugu.md`. Test kilidi: `leaderboard.csv`,
`test_*` anahtarları, `test_metrics.json`, `preds/test_kilitli/`, `replikasyon.md` okunmadı. Yukarıdaki tur-1 bölümü
değiştirilmedi.*

### Tur 1 kapanışı (referans)
- **H01 öldü** (+31.5%, 0/4 ufuk, p = 0.0043), çünkü SAN std başlığının son ReLU'su ilk ön-eğitim epoch'unda öldü
  (σ̂ = 0, pencerelerin %96–100'ünde). Omurga devre dışı kaldı ve model dilim başına sabit μ̂'ya indi. **Hipotez
  sınanamadı.** → **H05** (H01b) aynı aileyi ölü bileşen olmadan, H01'in ön-kayıtlı Ablasyon (2) tanımıyla dener ve önce
  CPU kapısından geçer.
- **H02 öldü** (−0.17%, 2/4 ufuk), çünkü öğrenilen γ ≈ 1, |β| ≈ 0.01 oldu. Ofset σ_lb ile ölçeklendiği için ≈0.002 z'de
  kaldı. Afin serbestlik seviye çapasını taşıyamıyor. → **H04** serbestlik eklemez, **merkezleme referansını** (μ_lb →
  x_son) değiştirir.
- **H03 DOĞRULANDI** (protokol; h336 −7.21%, 3/3, M/R 1.018 → 0.944). Ön-kayıt gereği **ebeveyn olmaz**. Tur-2 kartları
  h336'yı ona karşı da raporlar (aşağıda ortak ekleme 1).
- Mevcut en iyi TUT: **yok** → bütün tur-2 kartlarının ebeveyni **`baseline`** (resmî, h336 = 1 epoch;
  `20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`;
  val_mse_avg 0.44485 ± 0.01191).

### Durma kuralı (açık yazım; orkestrasyon_durumu)
Ardışık 2 turda en iyi zincirin val_mse_avg Δ'sı < %1 ise K5'e geçilir. **Tur 1 sayacı 1/2** (TUT yok; H03 protokol
kartı olduğu için sayılmaz). **Tur 2'de en az bir kart TUT alıp Δavg ≤ −1.0% (ebeveyn `baseline`'a göre göreli, n = 3
eşleştirilmiş) vermezse sayaç 2/2 olur ve K4 kapanır → K5.** Sayılmayanlar: İNCELE açık kalan kartlar (temiz
kapanmadıkça), H05'in yalnız kapı ölçümü, ÖLDÜR'ler, "TUT ama Δavg > −1%" sonuçları. Tur-2 kartlarının toplam
P(≥%1'lik TUT) ≈ 1 − (1−0.20)(1−0.04)(1−0.12)(1−0.0) ≈ **0.32** (H04 0.20, H05 0.04, H07 0.12, H06 ≈ 0). Yani **tur 2'nin
K4'ün son turu olma olasılığı ≈ %68**. Bu yüzden tur-2 seti, K4 burada biterse makalenin B çıkışının çekirdeğini (normalizasyon
kıyas tablosu) tamamlayacak biçimde seçildi.

### Tur 2 kartları (koşulan kart sayısı ≤ 3)

Skor = beklenen kazanç (val_mse_avg, mutlak %, TUT koşulunda) × P(TUT) / beklenen maliyet (GPU-saat + uygulama).

| Öncelik | Kart | Aile | Hedef | Beklenen etki (kaynak) | P(TUT) | Maliyet | Skor | Bilgi değeri (B) |
|---|---|---|---|---|---|---|---|---|
| 1 | **H04** `lastnorm`: merkez μ_lb → x_son (σ aynen) | A7 (A6 fikri) | **Z2** (D1 M/R 1.45 / 1.23), Z1 | avg **−3…+4%**. D1 M/R h96 ≤ 1.20. Zarar tarafı NLinear Exchange +6.0% (Zeng Tab.2), kazanç tarafı Z2 tavanı (hata-analizi §4) | 0.30 (≥%1: 0.20) | 0.35 GPU + düşük uygulama (H02 kancası) ≈ 0.4 | 2.0 × 0.30 / 0.4 ≈ **1.5** | Yüksek: tablonun "son-değer" satırı; Z2'nin nedensel testi; H02'nin neden işlemediğinin tamamlayıcısı (serbestlik değil, çapa) |
| 2 | **H05** SAN-μ (H01b): σ̂ yerine son girdi diliminin σ'sı, **CPU kapılı** | A7 | **Z1** | Kapı geçerse avg **−3…−15%** (SAN Tab.10 −18.7%, iskontolu). Kapı **BAŞARISIZ beklenir**: P(geçer) ≈ 0.10 (H01'de μ̂ noktasal olarak Repeat'ten +23.6% kötüydü) | 0.04 (kapı dahil) | kapı ≈ 0.05 CPU + 0.10 × 0.6 ≈ **0.11** beklenen | 9 × 0.04 / 0.11 ≈ **3.3** | Yüksek, her iki dalda: kapı başarısızsa "SAN'ın gelecek-ortalama tahmincisi val'de naif dilim ortalamasını geçemiyor" bileşen bulgusu; geçerse tezin ana mekanizması ölü bileşensiz sınanır |
| 3 | **H06** `normsuz`: `--use_norm 0` | A7 ablasyonu / A12 | Z1 (OOR, normalizasyonun taban değeri) | **Kötüleşme +5…+210%** (SAN Tab.9 +5.6 / +15.2%; NST Tab.4 +212% hesap; karşı yön FAN DLinear −14.7%) | ≈ 0 (ÖLDÜR beklenir) | 0.30 GPU, kodsuz ≈ 0.3 | **0** (kazanç kartı değil) | **Çok yüksek:** normalizasyon tablosunun "yok" satırı (sıfır noktası). Ters yönlü İNCELE eşiği ön-kayıtlı |
| (yedek) | **H07** `xmark-yok`: x_mark → None (12 → 8 token) | A12 / A6 | §7 hipotez adayı (sentetik takvim) | **|Δavg| ≤ 2%** iki yönlü (GLAFF NeurIPS'24 Şekil 1a: iTransformer'da damga kaldırma "no significant decline") | 0.35 (≥%1: 0.12) | 0.30 GPU + düşük ≈ 0.35 | 1.0 × 0.35 / 0.35 ≈ **1.0** | Orta-yüksek: veri artefaktı bulgusu. **Yalnız H05 kapısı BAŞARISIZ ise tur 2'de koşulur**, aksi hâlde tur 3'e devreder |

**Kullanıcı onayı:** Hiçbir tur-2 kartı split, metrik ya da veri değiştirmiyor. H06 ve H07 rakibin model girdisini/bayrağını
değiştiriyor, değerlendirme protokolünü değil. **KULLANICI ONAYI GEREKEN KART YOK.**

### Neden bu sıra
1. **H04 önce:** Tur-2'de ≥%1'lik TUT için en yüksek olasılık (0.20) bu kartta. Ölçülmüş bir mekanizmayı (Z2: β 0.72–0.98,
   D1 M/R 1.45) parametre eklemeden hedefliyor. Uygulama H02 kancasının kopyası (en düşük risk). Durma kuralı açısından
   K4'ün devamı büyük ölçüde bu karta bağlı. Beklenti dürüstçe nötre yakın (−3…+4%), çünkü literatürdeki tek doğrudan
   kanıt (NLinear, Exchange) son-değer çapasının **zararlı** olabildiğini gösteriyor.
2. **H05 ikinci (skor yüksek, koşma olasılığı düşük):** Skoru, kapının ucuzluğundan geliyor. Kapı dakikalar süren bir CPU
   ölçümü, GPU harcamaz ve her iki sonucu da makaleye girer. Kapı başarısız beklendiği için tur-2'nin GPU planı fiilen
   H04 + H06 + H07'dir. Kapı geçerse H07'nin yerini alır.
3. **H06 üçüncü:** Kazanç beklentisi yok (skor 0). Ama B çıkışının çekirdeği olan normalizasyon kıyas tablosunun
   (yok / NST / RevIN-afin / son-değer / SAN-μ / SAN-resmî-çöküş) sıfır noktası. Kodsuz ve ucuz. K4 tur 2'de biterse
   (olasılık ≈ %68) bu satır olmadan tablo eksik kalır. Bu yüzden yedek H07'nin önünde.
4. **H07 yedek:** Ucuz, B değeri var, ama tablonun omurgası değil. Yalnız H05 slotu boşalırsa koşulur. Kart koşumdan
   önce (bugün) yazıldığı için aktivasyon dallanması M1'i bozmaz.
- **Ertelenenler ve gerekçe:** FAN (A7-d; Exchange periyodik değil, mekanizma zayıf, beklenti −8…0% farklı split; yalnız
  H04/H05 TUT ya da tablo gereği tur 3). Dish-TS (yalnız kıyas satırı, literatürde Exchange'de hep kötü; bütçe kalırsa
  tur 4). A5 zaman-grubu DRO (Z3/Z4; 0…−8%, FOIL Tab.2; MDE altında → tur 3, birincil karar avg, worst_block ve B3/B4 M/R
  ön-kayıtlı ikincil). Online ince ayar (Exchange için beklenti BULUNAMADI → yalnız ayrı tablo kartı, tur 3–4).
  SAN-softplus (yeni tasarım kararı, kaynak gerekiyor; yalnız H05 kapısı geçer ve TUT alırsa tur 3'te "std tahmininin
  katkısı" olarak).

### Önerilen KOŞUM sırası
**(0) H05 kapısı (CPU, ≤10 dk, GPU kuyruğu başlamadan) → H04 → H06 → [H07 | H05-tam].**
- Kapı önce çalışır, çünkü tur-2'nin üçüncü slotunu belirler. Ayrıca bellek-temizliği geçmişi (günlük arıza tablosu,
  boş RAM ~4 GB) nedeniyle CPU ölçümünün GPU kuyruğuyla eşzamanlı koşmaması önerilir.
- H04, öncelik ve durma kuralı açısından ilk GPU kuyruğudur. H06 kodsuz olduğundan H04 kodu doğrulanırken bile kuyruğa
  hazırdır (sıra değişebilir, ön-kayıt etkilenmez).
- Kuyruklar seri (`--paralel 1`), her biri `--kademeli` (duman → ön-uçuş → tam × 3), beklenti dosyaları kuyruktan önce.
  `--kapi-test-metrics` hiçbir kuyruğa verilmez.

### Bütçe (tur 2)
| Kalem | GPU-saat |
|---|---|
| H05 kapısı | 0 (CPU ≤ 0.15 sa) |
| H04 `lastnorm` | ≤ 0.35 |
| H06 `normsuz` | ≤ 0.30 |
| H07 `xmark-yok` (kapı başarısız, beklenen) **veya** H05 tam (kapı geçer) | ≤ 0.30 **veya** ≤ 0.50 |
| **Tur 2 toplam** | **≈ 0.95 (beklenen) – 1.15 (üst)** |

Kalan 6.35 → tur 2 sonrası ≈ 5.2–5.4 GPU-saat. K5 için ~1.0 ayrılı kalır. Tur 3–4'e ≈ 4.2+ saat (bağlayıcı kısıt yine kart
sayısı ve durma kuralı).

### Tur-2 kartlarının ortak ön-kayıtlı eklemeleri (her kartta yazılı)
1. **h336 çift kıyas:** h336 Δ hem `baseline`'a (0.4003) hem `baseline-ep336eq`'e (0.3714) karşı raporlanır. h336 oyu
   yalnız varyant h336 ≤ 0.3714 ise "eğitim-bütçesinden bağımsız iyileşme" diye etiketlenir. Raporlama kuralıdır, karar
   kuralını değiştirmez. Tur-2 kartları h336'da resmî 1 epoch'u korur. H05, eklenen 5 istasyon epoch'unu raporlar.
2. **Eşleştirme:** Eşleştirilmiş Δ sd'si ve çift korelasyonu r raporlanır. RNG akışını koruyan kartlarda (H04, H06;
   H07 kısmen) eşleştirilmiş p determinizmi ölçer. **Karar eşiği değişmez:** MDE %8.1 (avg), ufuk MDE'leri 2.6 / 3.5 /
   1.5 / 12.8%.
3. **Trend-asimetri:** Düşüş/yükseliş pencere M/R'si ebeveyn değerleri (1.076 / 1.059 / 0.768 / n<10) yanına yazılarak,
   Δ ebeveyne göre raporlanır. Δavg < 0 iken düşüş M/R ebeveynden ≥ +0.05 kötüleşirse "val trendine aşırı uyum" notu
   düşülür (karar değişmez, K5'e taşınır).
4. **Kart-özel:** H04 için Z2 kuralı ("D1 düzelir ama toplam kötüleşirse ÖLDÜR + aile kapanır"). H05 için CPU kapısı
   (Ö1 ∧ Ö2). H06 için ters yönlü İNCELE (normsuz ebeveynden iyi çıkarsa). H07 için iki yönlü İNCELE (|Δavg| ≥ MDE).

### Tur 3 için koşullu ön-plan (kart DEĞİL; tur-2 sonuçlarından önce yazıldı)
- **Durma kuralı tetiklenirse (≥%1'lik TUT yok) →** K4 kapanır, K5. Açık kalan yedek H07 (koşulmadıysa) ve tur-3 adayları
  "yapılmadı" diye raporlanır.
- **H04 TUT (≤ −1%) →** tur 3 ebeveyni `lastnorm`. Adaylar: A5 zaman-grubu DRO (Z3/Z4), H07b (x_mark, yeni ebeveynle;
  H07 tur 2'de koşulmadıysa), Ablasyon (1) yalnız-merkezleme K5'e.
- **H05 kapısı geçer ve H05 TUT →** ebeveyn `san-mu`. Adaylar: SAN-tahminsiz ablasyonu (K5), SAN-softplus (kaynaklı yeni
  kart), A5 DRO.
- **H07 TUT (≤ −1%) →** ebeveyn `xmark-yok` olur, ama bu bir veri artefaktı bulgusudur. Tur-3 kartları onu ebeveyn alırsa
  makale tablosunda "iTransformer (x_mark'sız)" referans satırı da verilir.
- **Birden fazla TUT →** tur 3'te yalnız en iyisi ebeveyn olur. Birleşim kartı yalnız ≥2 bağımsız TUT ile ve tur 4'te.

### Kapanmış kartlar (tur 2 itibarıyla)
| Kart | Durum | Gerekçe (tek satır) |
|---|---|---|
| H01 `san` | ÖLDÜR (arıza notlu) | std başlığı ölü ReLU → omurga devre dışı; mekanizma sınanamadı (tahlil-tur1 §5.1) |
| H02 `revin-afin` | ÖLDÜR | γ ≈ 1, |β| ≈ 0.01 → ≈0.002 z ofset; 2/4 ufuk; MDE'nin ~50 kat altında (§5.2) |
| H03 `baseline-ep336eq` | DOĞRULANDI (protokol, zincir dışı) | h336 −7.21%, 3/3; ebeveyn olmaz; düşüş-penceresi bedeli K5'te izlenecek (§5.3) |

---

## Tur 3 (eklendi 2026-10-14, K4 tur 3 / 4 — hiçbir tur-3 koşumu başlamadan)

*Hazırlayan: hipotez-tasarımcısı · Girdi: `sonuclar/tahlil-tur2.md` (§5 mekanizma, §7 normalizasyon tablosu, §8 tur 3 +
K5 ön-planı), `sonuclar/tahlil-tur1.md`, `sonuclar/normalizasyon-kiyas.csv`, `sonuclar/leaderboard.md`,
`hipotezler/H01–H07.md` (Sonuç bölümleri), `sonuclar/hata-analizi.md`, `notlar/literatur-cipa.md` §5.2–5.3,
`orkestrasyon_durumu.md` (seçim/durma kuralı, şef protokol kararları), `gunluk/deney-gunlugu.md`. Ek kaynak doğrulaması
(WebSearch/WebFetch 2026-10-14): Sagawa et al. group_DRO resmî kodu (`--robust_step_size` default 0.01), Wild-Time
(arXiv:2211.14238), RevIN bileşen ablasyonu arXiv:2603.11869 (Exchange yok). Test kilidi: `leaderboard.csv`, `test_*`
anahtarları, `test_metrics.json`, `preds/test_kilitli/`, `replikasyon.md` okunmadı. Yukarıdaki tur-1 ve tur-2
bölümleri değiştirilmedi.*

### Tur 2 kapanışı (referans)
- **H04 TUT — ayırt edilemez** (val_mse_avg 0.4278 ± 0.0059, Δ −3.83%, 4/4 ufuk, p = 0.228; D1 M/R h96 1.45 → 1.02).
  Δavg ≤ −1% olduğu için **durma sayacı 1/2 → 0/2**. **Tur-3 ebeveyni `lastnorm`** (`20261014-052508_lastnorm_s2023`,
  `…053601_lastnorm_s2024`, `…054213_lastnorm_s2025`). Sarı not: h336 düşüş-pencere M/R 0.768 → 0.981 (val trendine
  aşırı uyum notu, K5'e).
- **H05 kapıda öldü** (R_H = 1.14–1.33, 0/4 < 1), çünkü SAN'ın gelecek-dilim μ̂'sı val'de naif son-dilim ortalamasından
  kötü (train'de iyi). SAN ailesi bu workspace'te yalnız yeni kaynaklı bir kartla açılabilir. **Tur 3'te açılmıyor**:
  kapının ölçtüğü "öğrenilmiş seviye tahmini < son değer" bulgusu, Z1'in seviye tahmini yoluyla kapanmayacağını gösteriyor.
- **H06 öldü** (beklenen; +392%), çünkü normalizasyonsuz model train dışı seviyeyi çıkışa taşıyamadı. → **H10** bu
  çöküşün seviyeden mi ölçekten mi geldiğini, lastnorm'un merkezini koruyup yalnız σ'yı kaldırarak ayırır.
- **H07 TUT — ayırt edilemez, H0 desteklendi** (−0.29%; takvim sentetik). Ebeveyn adayı değil. Yeni ebeveynde tekrarı
  (H07b) **yapılmaz** (bilgi değeri düşük; tahlil-tur2 §8).

### Durma kuralı (tur 3; açık yazım)
- **Tur 3'te en az bir kart TUT alıp Δavg ≤ −1.0% (ebeveyn `lastnorm`'a göre göreli, n = 3 eşleştirilmiş) vermezse
  sayaç 1/2 olur.** Tur 4 bütçenin son turudur; K4 en geç tur 4 sonunda kapanır.
- **Sayılanlar:** H09, H10. **Sayılmayanlar:** H08 (protokol kartı, H03 emsali), İNCELE açık kalan kartlar, ÖLDÜR'ler,
  "TUT ama Δavg > −1%".
- P(≥%1'lik TUT, tur 3) ≈ 1 − (1 − 0.15)(1 − 0.10) ≈ **0.24** (H10 0.15, H09 0.10). Yani tur 3 sonunda sayacın 1/2
  olma olasılığı ≈ %76. Bu, tur 4'ün kesin son tur olmasını değiştirmez. Tur-3 seti bu yüzden K5'in iki ön-kayıtlı
  maddesini (epoch-eşit sürüm ve H04 Ablasyon 1) öne alacak biçimde seçildi. K4 hangi turda biterse bitsin makalenin
  normalizasyon tablosu ve ablasyonu GPU beklemeden hazır olur.
- **Tur-4 ebeveyni (ön-kayıt):** H09 ve H10'dan Δavg ≤ −1% ile TUT alanların en iyisi (Δavg'si en negatif). İkisi de
  alırsa tur 4'te açıkça işaretli bir **birleşim kartı** (lastnorm-merkez + DRO) açılabilir (M2: ikisi de tek başına TUT).
  Hiçbiri almazsa ebeveyn `lastnorm` kalır.

### Tur 3 kartları (≤ 3 kart)

Skor = beklenen kazanç (val_mse_avg, mutlak %, TUT koşulunda) × P(TUT) / maliyet (GPU-saat + uygulama). H08 için
"kazanç" protokol bulgusunun avg yansımasıdır ve zincire girmez.

| Öncelik | Kart | Aile | Hedef | Tek değişiklik (ebeveyn `lastnorm`) | Beklenen etki (kaynak) | P | Maliyet | Skor | Bilgi değeri (B) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **H08** `lastnorm-ep336eq` (**PROTOKOL**) | A11/A12 | Z5 × Z2 | h336 `--train_epochs` 1 → 10 | h336 **0.355–0.372** (vs lastnorm −5.1…−9.4%; vs ep336eq −4.4…+0.2%). İç çıpa H03 −7.21%, lastnorm'un 10-epoch ufuklardaki etkisi −1.4…−3.1% | P(DOĞRULANDI) 0.90; S2: TOPLANIR 0.40 / İKAME 0.50 / TERS 0.10 | ≤ 0.50 GPU, kodsuz (argv) ≈ 0.45 | 1.7 × 0.90 / 0.45 ≈ **3.4** (zincir dışı) | **Çok yüksek:** K5 ön-planı madde 1'i öne alır. "lastnorm kazancı epoch eşitlemeyle toplanıyor mu?" sorusunu testten önce yanıtlar. Normalizasyon tablosuna "epoch-eşit" sütunu |
| 2 | **H10** `lastnorm-merkez` (σ ≡ 1; H04 Ablasyon 1) | A7 | Z1 (s_io tavanı; σ_lb çarpımı) | de/normalizasyonda σ_lb → 1 | avg **−4…+8%** iki yönlü. NLinear Exchange +6.0% (Zeng Tab.2) zarar ucu; FAN-DLinear RevIN +17.3% (iskontolu) kazanç ucu | TUT 0.25 (≥%1: 0.15) | ≤ 0.40 GPU + çok düşük uygulama (tek satır forward) ≈ 0.4 | 3.0 × 0.25 / 0.4 ≈ **1.9** | **Yüksek:** K5 ablasyonunu öne alır; H06'nın çöküşünü seviye / ölçek olarak ayrıştırır; tabloya "son-değer, σ'sız (NLinear tarzı)" satırı |
| 3 | **H09** `lastnorm-dro` (zaman-grubu GroupDRO, K = 4, η = 0.01) | A5 | Z3 (aşınma, worst_block), Z4 | eğitim kaybı ERM → GroupDRO | avg **−2…+3%** (FOIL Tab.2 GroupDRO −0.1%; Wild-Time: ERM'yi tutarlı geçmiyor). h336 ≈ 0 (1 epoch'ta q ≈ uniform) | TUT 0.25 (≥%1: 0.10) | ≤ 0.45 GPU + orta uygulama (train döngüsü override) ≈ 0.55 | 1.5 × 0.25 / 0.55 ≈ **0.7** | **Orta-yüksek:** katalogdaki ikinci aile (A5) ve tezin dayanıklılık ekseni. Ön-kayıtlı ikincil worst_block / B3–B4 M/R ile "ortalama ↔ en kötü blok" ödünleşimi ölçülür |

**Kullanıcı onayı:** H09 ve H10 yalnız model içi normalizasyonu ve eğitim kaybını değiştiriyor; split, metrik, val
tanımı ve veri aynı. H08 rakibin bir eğitim ayarını (h336 epoch) H03 ile aynı gerekçeyle eşitliyor; değerlendirme
protokolü değişmiyor ve resmî satırlar korunuyor. Şef H03 teyidini bu kart için kayda geçirmeli (H08 başlığında
işaretli). **KULLANICI ONAYI GEREKEN KART YOK.** Şef kapsam kararı gereği **online ince ayar** (protokol değiştirir) ve
**drift çapası** (trend-asimetri riski) tur 3'e alınmadı.

### Neden bu sıra
1. **H08 önce:** Neredeyse kesin (H03 iç çıpası) ve kodsuz: `lastnorm` kos'una bir argv yaması. Ayrıca K5'te zaten
   gerekiyor. Tur 3'te koşulması K5 GPU yükünü düşürür ve S2 (toplanırlık) sorusunu test kilidi açılmadan yanıtlar.
   Durma sayacına girmediği için zincir kararlarını etkilemez. Ama makalede h336 kazancının hangi kısmının "eksik
   eğitim telafisi" olduğunu belirleyen tek ölçüm bu.
2. **H10 ikinci:** Tur 3'te ≥%1'lik TUT için en yüksek olasılık (0.15). Hata kütlesinin hâlâ en büyük parçasını
   (Z1, h336+h720 avg'nin %81'i; s_io oranı h720 0.19) ölçülmüş bir mekanizma (σ_lb çarpımı) üzerinden hedefliyor.
   Uygulaması tek satırlık bir `forward` değişikliği. Her iki sonucu da makaleye girer (K5 ablasyon tablosu), yani
   başarısız olsa da bilgi kaybı yok. Beklenti dürüstçe iki yönlü, çünkü merkez ve ölçeği ayrı ölçen bir kaynak yok ve
   en yakın iki kanıt (NLinear +6.0%, FAN-DLinear RevIN +17.3%) farklı yönleri destekliyor.
3. **H09 üçüncü:** Literatürdeki tek doğrudan Exchange ölçümü (FOIL Tab.2) GroupDRO etkisini ≈ 0 veriyor; Wild-Time da
   aynı yönde. Bu nedenle P(≥%1) en düşük kart. Yine de tur 3'te koşulur, çünkü (a) katalog 5.3'ün ikinci ailesi (A5)
   şimdiye kadar hiç denenmedi; (b) Z3 (aşınma) ve worst_block'u doğrudan hedefleyen tek kart; (c) "kaymaya dayanıklı
   kayıp Exchange'de ortalamayı değiştirmiyor, en kötü bloğu değiştiriyor mu?" sorusunun her iki yanıtı da B çıkışına
   girer. Uygulaması en karmaşık kart (train döngüsü override + indeks sarmalayıcı).
- **Alınmayanlar ve gerekçe:**
  - Online ince ayar: protokol değişikliği, kullanıcı onayı gerekir (şef kararı).
  - Drift çapası: trend-asimetri riski çok yüksek, Exchange'de drift zayıf (şef kararı).
  - SAN-softplus: H05 kapısı aile bileşeninin val'de naif çapadan kötü olduğunu gösterdi.
  - H07b: H0 desteklendi.
  - FAN: Exchange periyodik değil, mekanizma zayıf.
  - Dish-TS: literatürde Exchange'de hep kötü; yalnız kıyas satırı. Tur 4'te bütçe kalırsa.
  - A9 seed-ensemble: son itme, rakibe aynı hakla, tur 4.

### Önerilen KOŞUM sırası (öncelikle aynı; gerekçe uygulama riski)
**H08 → H10 → H09.**
- H08 kodsuzdur (argv yaması + mevcut `LastNormiTransformer`), ilk kuyruk olarak hemen girer.
- H10 tek satırlık model değişikliği ve CPU doğrulaması (< 1 dk) H08 koşarken hazırlanır.
- H09'un train döngüsü override'ı ve `dogrula.py`'si en uzun iştir; GPU H08/H10 ile doluyken yazılır.
- Kartların tamamı koşumdan önce yazılı olduğu için koşum sırası ön-kaydı etkilemez (M1).
- Kuyruklar seri (`--paralel 1`), `--kademeli` (duman → ön-uçuş → tam × 3), beklenti dosyaları kuyruktan önce.
  `--kapi-test-metrics` hiçbir kuyruğa verilmez.
- Bellek-temizliği öldürmeleri (3 kez) nedeniyle kuyruk başına boş RAM ölçümü önerilir. CPU `dogrula.py` betikleri GPU
  kuyruğuyla eşzamanlı koşmaz (`CUDA_VISIBLE_DEVICES=-1`, günlük arıza kaydı 05:02).

### Bütçe (tur 3)
| Kalem | GPU-saat |
|---|---|
| H08 `lastnorm-ep336eq` | ≤ 0.50 (beklenen ≈ 0.45) |
| H10 `lastnorm-merkez` | ≤ 0.40 |
| H09 `lastnorm-dro` | ≤ 0.45 |
| CPU doğrulamaları (H09, H10) | 0 (CPU ≤ 0.05 sa) |
| **Tur 3 toplam** | **≈ 1.2 (beklenen) – 1.35 (üst)** |

- Kalan ≈ 5.0 → tur 3 sonrası ≈ 3.65–3.8 GPU-saat.
- K5 için 1.0 ayrılı kalır. H08 ve H10 tur 3'te koşulduğu için K5'in GPU ihtiyacı lastnorm kazanan kalırsa ≈ 0'a iner
  (test oturumu kilitli tahminlerden). Tur 4'te ebeveyn değişirse kazananın epoch-eşit sürümü ≈ 0.4 olur.
- Tur 4'e ≈ 2.65+ saat kalır. Bağlayıcı kısıt yine kart sayısı ve durma kuralı.

### Tur-3 kartlarının ortak ön-kayıtlı eklemeleri (her kartta yazılı)
1. **Ebeveyn değerleri** `lastnorm`: h96 0.12096, h192 0.20837, h336 0.39197, h720 0.98992, avg 0.42780 (± 0.00588); seed
   başına avg 0.43227 / 0.42999 / 0.42115. Resmî `baseline`'a karşı Δ da raporlanır (makale tablosu için).
2. **h336 üçlü kıyas:** h336 Δ vs `lastnorm` (0.3920), vs `baseline-ep336eq` (0.3714) ve koşulduysa vs `lastnorm-ep336eq`
   (H08). "Eğitim-bütçesinden bağımsız" etiketi yalnız h336 ≤ 0.3714 ise. Raporlama kuralıdır, karar kuralı değişmez.
3. **Trend-asimetri:** ebeveyn `lastnorm` düşüş M/R 1.030 / 1.001 / 0.981 / n<10 (yükseliş 0.872 / 0.885 / 1.000 / 0.865)
   yanına yazılır. Δavg < 0 iken düşüş M/R ebeveynden ≥ +0.05 kötüleşirse "val trendine aşırı uyum" notu düşülür
   (karar değişmez, K5'e).
4. **Eşik:** MDE %8.1 (avg) sabittir; ufuk MDE'leri 2.6 / 3.5 / 1.5 / 12.8%. lastnorm seed'lerinden türeyen %4.2 yalnız
   bilgi olarak yazılır (çıpa ilkesi).
5. **Eşleştirme:** eşleştirilmiş Δ sd'si ve çift korelasyonu r raporlanır (H04'te h720 r = −0.94 eşleştirmeyi
   duyarsızlaştırdı; karar yine MDE kuralıyla).
6. **Kart-özel:** H08 S1 (h336 Δ ≤ −1.5%, 3/3) + S2 toplanırlık sınıfı (vs ep336eq). Kontrol ufukları lastnorm ile
   bit-bit aynı olmalı. H09 için mekanizma-devrede-değil notu (son-epoch q_max < 0.30) ve dayanıklılık-ödünleşimi notu.
   H10 için Z2 koruma notu (D1 M/R h96 > 1.10) ve ters yönlü "σ taşıyıcı" etiketi (Δavg ≥ +50%).

### K5 ön-planı (tur-3 sonuçlarından önce yazıldı; kart DEĞİL)
- **lastnorm kazanan kalırsa:**
  - K5 GPU ≈ 0: epoch-eşit satırı (H08) ve ablasyon (H10) hazır.
  - Test oturumunda satırlar: resmî, ep336eq, lastnorm, lastnorm-ep336eq. Şef kararıyla tablo bütünlüğü için H10 ve
    H09 satırları da eklenebilir (ayrı arama yok, D1).
- **H10 ya da H09 kazanır (tur 4 ebeveyni olur) ve tur 4 kazananı değiştirmezse:** kazananın epoch-eşit sürümü K5'te
  (≈ 0.4). H04 Ablasyon (1) zaten H10'dur. H10 kazandıysa onun ablasyonu yalnız ayırt edilebilir kazançta yapılır
  (H10 kartı).
- **Her durumda:** test blok kırılımı (Q1–Q4) ve trend-asimetri, ebeveyn ve kazanan için yan yana. Görülmemiş-iyi
  denetimi kartların K5 eşikleriyle yapılır (H04: `test_mse_avg` < 0.317).

### Kapanmış kartlar (tur 3 itibarıyla; tur-2 tablosuna ek)
| Kart | Durum | Gerekçe (tek satır) |
|---|---|---|
| H04 `lastnorm` | TUT — ayırt edilemez (açık; tur-3 ebeveyni) | Δ −3.83%, 4/4, p = 0.228; Z2 kapandı (D1 1.45 → 1.02), Z1 sürüyor (tahlil-tur2 §5.1) |
| H05 `san-mu` | ÖN-KOŞUL BAŞARISIZ (ÖLDÜR eşd.) | R_H 1.14–1.33 (val), 0/4 < 1: öğrenilmiş μ̂ val'de naif son-dilim ortalamasından kötü (§5.4) |
| H06 `normsuz` | ÖLDÜR (beklenen; tablo satırı) | +392%, 0/4: tahminler train aralığına hapsoluyor (%0–0.2 OOR; §5.2) |
| H07 `xmark-yok` | TUT — ayırt edilemez; H0 desteklendi (ebeveyn değil) | −0.29%, p = 0.77; takvim sentetik, token'lar bilgi taşımıyor (§5.3) |


<!-- EK DOSYA: Bu bölüm hipotezler/SIRALAMA.md'nin SONUNA (son satır "| H07 `xmark-yok` | ..." tablosundan sonra) aynen eklenmek üzere yazıldı. Tasarımcı oturumunda Edit aracı yoktu. SIRALAMA.md Write ile yeniden yazılmadı (orkestratör talimatı). Şef: Edit ile ekleyip bu dosyayı bırakabilir ya da silebilir. -->

---

## Tur 4 (FAZ II) (eklendi 2026-10-14, FAZ II ŞEF ÖN-KAYDI ~16:55'ten sonra, hiçbir tur-4 koşumu ve kapı ölçümü başlamadan)

*Hazırlayan: hipotez-tasarımcısı.*

**Girdi:** `orkestrasyon_durumu.md` (FAZ II kullanıcı kararları + şef ön-kaydı), `sonuclar/hata-analizi.md`,
`sonuclar/tahlil-tur1.md` (§5.1), `tahlil-tur2.md`, `tahlil-tur3.md`, `hipotezler/H01–H10.md`, `notlar/literatur-cipa.md`,
`gunluk/deney-gunlugu.md`, SAN portu `varyantlar/san/san_modul.py`.

**Ek kaynak doğrulaması (WebSearch 2026-10-14):**
- DeepAR'da softplus ile σ parametrelemesi (arXiv:1704.04110).
- M4 Comb/Naive2 sMAPE değerleri.
- SAN resmî reposunda std-aktivasyon düzeltmesi BULUNAMADI.
- Exchange'de model + Repeat birleşimi için nicel etki BULUNAMADI.

**Test kilidi:** şu dosyalar okunmadı: `sonuclar/karar.md`, `sonuclar/test-*`, `leaderboard-test.md`, `naif-test.csv`,
`ablasyon.*`, `leaderboard.csv`, `test_metrics.json`, `test_kilitli/`. Yukarıdaki tur 1–3 bölümleri değiştirilmedi.

> **Test maruziyeti beyanı.** Zorunlu okuma listesindeki `orkestrasyon_durumu.md` (K5 satırları) ve `deney-gunlugu.md`
> (K5 satırı), 15:42'de açılan Exchange test özetini içeriyor. Tasarımcı bu özeti gördü. Tur-4 kartlarının gerekçesi,
> beklentileri ve eşikleri yalnız val ve literatüre dayanır. Yine de **tur-4'ün tamamı makalede "Exchange test'i
> görüldükten sonra" diye beyan edilir.** Birincil temiz kanıt ETTh1/ETTh2'den gelir (şef ön-kaydı).
>
> **Belgelenebilen test-öncesi kayıtlar:**
> - Yalnız H11'in fikir sınıfı: günlük 03:00 "H01b ölü-ReLU'suz aktivasyon".
> - H12 ve H13 için orkestrasyon ya da günlükte 15:42 öncesi kayıt **yok**. `orkestrasyon_durumu.md` satır 91 birleşimi
>   "test'ten önce kayıtlı" sayıyor; şef bunu düzeltmeli ya da kaynağını göstermeli.

### Tur 3 kapanışı ve FAZ II geçişi (referans)
- **H09 öldü** (−5.0% avg, ama 1/4 ufuk; kazancın %107'si h720'den geldi). Zaman-grubu DRO, Z3 aşınmasını düzeltmedi.
- **H10 öldü** (+1.0%, 1/4 ufuk). σ_lb kısa ufuk genliğini taşıyor; normalizasyonun değeri seviye merkezlemesinde.
- **H08 DOĞRULANDI** (protokol). Zincire girmez.
- K5 ve FAZ II geçişi: kullanıcı kararlarıyla (orkestrasyon §FAZ II).
- **SAN ailesi** H05'te "yalnız yeni kaynaklı kartla" kapanmıştı. H11 bu koşulu karşılar: DeepAR softplus ve ölü-ReLU
  kaynakları var, ve aynı fikir test-öncesi kayıtlı (günlük 03:00).

### Ebeveyn ve karar kuralı (FAZ II, aynen)
- **Ebeveyn:** `lastnorm` kat koşumları `lastnorm-k{1,2,3}` × s{2023, 2024, 2025}.
  - K3 = resmî val. Resmî koşumlar 052508 / 053601 / 054213; kat koşumları bunlarla bit-bit aynıysa ikisi de kullanılabilir.
- **Kural:** TUT ⇔ Δ̄ < 0 ∧ ≥ 2/3 katta Δ_k < 0 ∧ ≥ 3/4 ufukta Δ_H < 0.
  - Δ_H: kat ortalaması.
  - Tanımlar H11 kartında.
- **İkincil kurallar:**
  - Kat × ufuk M/R ve kat başına M/R_k (naif hedefi: 3/3 kat < 1).
  - Aynı kuralın `baseline-k*`'a karşı uygulanışı ("rakibi geçer" cümlesi için zorunlu).
- **Koşullu (ön-kayıt):** `lastnorm` kat protokolünde `baseline`'a karşı TUT alamazsa, tur-4 kararları yine `lastnorm`'a
  karşı verilir (kartlar öyle kayıtlı). Rakip kıyası ise birincil rapor olur. Tur-5 ebeveyni o durumda aşağıdaki kurala bağlanır.

### Tur 4 kartları (≤ 3)

Skor = beklenen kazanç (Ā, mutlak %, TUT koşulunda) × P(TUT) / maliyet (GPU-saat + uygulama ≈ 0.05–0.2).

| Öncelik | Kart | Aile | Hedef | Tek değişiklik (ebeveyn `lastnorm`) | Beklenen etki (kaynak) | P(TUT) | Maliyet | Skor | Bilgi değeri |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **H13** `lastnorm-olcek` | Birleşim, Granger–Ramanathan (toplam = 1, α ≥ 0) | Z1: temkinlilik (K3 α* 1.33 / 1.30 / 0.77 / 3.65) | C = R + max(α̂, 0)·(P − R); α̂ iç-holdout (K_{k−1} modeli, train içi), 10 adım-dilimi | Δ̄ **−6…+6%**, h720 −15…+15% (kimlik + oracle iç çıpası; Smith–Wallis puzzle) | 0.20 | K0 paylaşımlı; tek başına 0.23 + 0.05 | 3 × 0.20 / 0.28 ≈ **2.1** (paylaşımla marjinal ≈ 12) | Yüksek: "MSE-eğitimli model kayma altında yapısal olarak mı temkinli, rejime özgü mü?" Her iki yanıt da B çıkışına girer |
| 2 | **H11** `san-softplus` (= H01b) | A7 | Z1, Z3 | lastnorm normalizasyonu → SAN (resmî Exchange ayarı), std başlığı ReLU → **Softplus**. `san` koduna göre tek satır. **CPU kapısı** (ölü oranı, ölçek oranı, ReLU yeniden üretimi) | Kapı geçerse Δ̄ **−12…+8%**, merkez 0…+3% (SAN Tab.10 −18.7%; FAN Tab.3 SAN vs RevIN −9.5%, vs normsuz +6.2%; H05 iç kanıtı) | 0.15 (kapı 0.80 dahil) | ≤ 1.2 GPU + 0.2 uygulama | 6 × 0.15 / 1.4 ≈ **0.64** | Çok yüksek: H01'in sınanamamış hipotezi ilk kez sınanır. Normalizasyon tablosundaki "SAN (resmî, çökmüş)" satırının düzeltilmiş eşi. SAN–literatür çelişkisine katkı |
| 3 | **H12** `lastnorm-birlesim` | Birleşim, Bates–Granger konveks | Z3 + FAZ II naif hedefi | C = R + clip(α̂, 0, 1)·(P − R); iç-holdout ve dilimler H13 ile aynı | Δ̄ **−1.0…+0.5%** (kazanç ≤ V/MSE ≈ %2–5; K3'te oracle w = 1, 3/4 ufuk). Naif hedefi P ≈ 0.6 | 0.25 | 0.23 GPU (K0, H13 ile ortak) + 0.05 | 0.5 × 0.25 / 0.28 ≈ **0.45** | Orta-yüksek: "naifi tutarlı geçme" güvencesi (kat × ufuk M/R ≤ 1). Combination puzzle ablasyonu (eşit ağırlık) bedava |

**Kullanıcı onayı:**
- Hiçbir kart split, metrik, val tanımı ya da veri değiştirmiyor.
- H12 ve H13'ün K0 iç modeli, kat protokolünün K_{k−1} tanımının bir adım geri uzantısıdır. Yalnız train satırlarını
  kullanır ve değerlendirme katı değildir. Yöntemin kendi eğitim maliyetidir ve rapor edilir.
- **KULLANICI ONAYI GEREKEN KART YOK.**

### Neden bu sıra
1. **H13 önce (skor).** Ölçülmüş bir val mekanizmasını doğrudan sınıyor: α* > 1 (3/4 ufuk), yön isabeti 0.63–0.85,
   std oranı 0.12–0.18. GPU maliyeti H12 ile ortak K0 koşumu kadar. Riski dürüstçe yazıldı (rejim bağımlılığı, H09
   emsali). K3 motivasyon katı olduğu için **K1 ∧ K2 alt-sonucu ön-kayıtlı** raporlanır.
2. **H11 ikinci.** Tek başına en büyük olası etki (SAN literatürü) ve en yüksek makale değeri onda. Ancak (a) H05 kapısı
   μ̂ başlığının K3'te naiften kötü olduğunu gösterdi ve (b) GPU açısından en pahalı kart. Kapı ucuz (CPU) ve
   başarısızlığı da raporlanabilir bir bileşen bulgusu.
3. **H12 üçüncü (zorunlu aday).** MSE kazancı matematiksel olarak küçük sınırlı (≤ V/MSE). K3'te büyük olasılıkla
   etkisiz ya da hafif zararlı (oracle w = 1). Değeri FAZ II'nin ikinci hedefinde: naif Repeat'i her kat ve ufukta
   geçmek ya da ona eşit kalmak. H13 ile aynı makineyi kullandığından ek GPU maliyeti 0.

**Çoklu karşılaştırma notu.**
- H12 ve H13 aynı iç-holdout kestiricisinin iki kısıt seçeneği. İkisi de ön-kayıtlı ve ikisi de raporlanır.
- Biri TUT alırsa makalede "iki birleşim kısıtından biri" diye yazılır.
- Ablasyon satırları (eşit ağırlık, tek ağırlık/ufuk, oracle) karar dışıdır; seçim için kullanılmaz.

### Önerilen KOŞUM sırası (öncelikten farklı; gerekçe: bağımlılık ve RAM)
**(0)** H11 CPU kapısı (≤ 20 dk; GPU kuyruğu boşken, `CUDA_VISIBLE_DEVICES=-1`) → **(1)** `lastnorm-k0` × 3 (GPU, ≈ 0.23)
→ **(2)** H12 + H13 CPU son-işlemi (ebeveyn kat koşumları bitince, < 10 dk) → **(3)** H11 GPU: duman → ön-uçuş →
3 kat × 3 seed (yalnız KAPI = GECTI ise, ≈ 1.1) → **(4)** koşullu `baseline-k0` × 3 (yalnız H12 ya da H13 TUT alırsa,
adil kıyas, ≈ 0.20).

Ek kurallar:
- CPU işleri GPU kuyruğuyla eşzamanlı koşmaz (bellek-temizliği geçmişi).
- Beklenti dosyaları kuyruktan önce yazılır.
- `--kapi-test-metrics` hiçbir kuyruğa verilmez.

### Bütçe (FAZ II; kalan ≈ 12 GPU-saat varsayımıyla)
| Kalem | GPU-saat |
|---|---|
| (Şef, P1) `lastnorm` + `baseline` kat koşumları, 2 × 3 kat × 3 seed (K3 resmî koşumlardan yeniden kullanılırsa 2 × 6 koşum) | ≈ 1.8 (≈ 1.2) |
| H11 kapı | 0 (CPU ≤ 0.35 sa) |
| H11 duman + ön-uçuş + 9 tam koşum (yalnız kapı geçerse) | ≤ 1.2 (beklenen 0.8 × 1.1 ≈ 0.9) |
| H12 + H13 ortak `lastnorm-k0` × 3 | ≈ 0.23 |
| Koşullu adil kıyas `baseline-k0` × 3 | ≤ 0.20 |
| **Tur 4 toplam (kartlar)** | **≈ 1.45 (üst 1.65)** |
| Ek veri setleri payı (ETTh1/ETTh2 doğrulama + replikasyon + kazanan, şef ön-kaydı) | ≥ 4.0 |
| Aktarım ekleri (yalnız kazanan için): H11 ETT × 3 seed ≈ 0.5 **veya** H12/H13 ETT K2 modeli ≈ 0.4 | ≤ 0.5 |
| **Tur 5–7 için kalan** | **≈ 12 − 1.8 − 1.65 − 4.5 ≈ 4.0** |

### FAZ II durma kuralı (şefe öneri, kart değil)
FAZ II ön-kaydında durma kuralı yazılı değil. Öneri: "Ardışık iki FAZ II turunda yeni kurala göre Δ̄ ≤ −1% veren bir
TUT yoksa Exchange araması kapanır ve ek veri setlerine geçilir." Bu, eski kuralın kat protokolüne taşınmış hâlidir.
Şef ön-kaydı olarak tur-4 sonucu görülmeden yazılmalıdır.

### Alınmayanlar ve gerekçe
- **lastnorm + SAN-μ seviye-trend çapası:** H05 kapısı μ̂'nın K3'te naif son-dilim ortalamasından 4/4 ufukta kötü
  olduğunu gösterdi (R_H 1.14–1.33). Ayrıca M2 gereği iki bileşen ancak ikisi de tek başına TUT ise birleşir.
- **Sönümlü trend (sabit φ):** şefin tur-3 kararı drift çapasını trend-asimetri riski nedeniyle almadı. Sabit φ
  (Gardner tarzı) seçimi bir hiperparametre serbestliği getirir (M3). H13 aynı "eksik ekstrapolasyon" sorusunu sabit φ
  olmadan, train içi kestirimle sınıyor.
- **A9 seed-ensemble:** pilot −0.34%; rakibe aynı hak; ≥ 3 bağımsız topluluk için varyant başına +6 seed.
- **H09b (yalnız h720 DRO):** val'e bakılarak ufuk seçimi olur (M3; tahlil-tur3 §8).
- **Dish-TS, FAN:** literatürde Exchange'de kötü ya da mekanizma zayıf (tur 2–3 gerekçeleri aynen).
- **Online ince ayar:** değerlendirme protokolünü değiştirir. **KULLANICI ONAYI GEREKİR**, otomatik koşmaz.
  Tur 5+ için yalnız kullanıcı isterse.

### Tur 5 için koşullu ön-plan (kart değil; tur-4 sonuçlarından önce yazıldı)
- **Ebeveyn:** tur-4'te kat kuralıyla TUT alanların en iyisi (Δ̄ en negatif).
  - Hiçbiri TUT almazsa `lastnorm` kalır, ancak yalnız `lastnorm` kat kuralıyla `baseline`'a karşı TUT ise. Değilse
    ebeveyn `baseline` olur.
- **H11 ve (H12 ∨ H13) ikisi de TUT ise:** açıkça işaretli bir birleşim kartı açılabilir (`san-softplus` + birleşim; M2).
- **Yalnız H11 TUT ise:** H11 Ablasyon (2) SAN-tahminsiz (≈ 1.0 GPU-saat) ve epoch-eşit sürüm (≈ 0.25).
- **Yalnız H13 TUT ise ve "yalnız motivasyon katında" notu varsa:** Exchange iddiası zayıf tutulur; aktarım kararı
  ETT K2/K3 kat sonucuna bırakılır.
- **Hiçbiri TUT almazsa:** önerilen durma kuralı tetiklenmeye yaklaşır. Kalan bütçe ek veri setlerine kaydırılır.

### Kapanmış kartlar (tur 3 itibarıyla; tur-3 tablosuna ek)
| Kart | Durum | Gerekçe (tek satır) |
|---|---|---|
| H08 `lastnorm-ep336eq` | DOĞRULANDI (protokol, zincir dışı) | h336 −8.89%, 3/3; epoch eşitleme ile son-değer çapası toplanıyor (tahlil-tur3 §5.1) |
| H09 `lastnorm-dro` | ÖLDÜR | 1/4 ufuk; Δavg'nin %107'si h720'den; Z3 aşınması düzelmedi (§3, §5.3) |
| H10 `lastnorm-merkez` | ÖLDÜR (ablasyon satırı) | +1.0%, 1/4 ufuk; σ_lb kısa ufuk genliğini taşıyor (§5.2) |


## Tur 5 (FAZ II, son Exchange turu) (eklendi 2026-10-14, tur-4 tahlili ve ŞEF KARARI tur 5'ten sonra, hiçbir H14 hesabı başlamadan)

*Hazırlayan: hipotez-tasarımcısı. Bu bölüm `SIRALAMA.md` sonuna eklenmek içindir; önceki bölümler değiştirilmedi.*

**Girdi:** `orkestrasyon_durumu.md` (FAZ II kuralları, durma kuralı, ŞEF KARARI tur 5), `sonuclar/tahlil-tur4.md`,
`hipotezler/H12.md`, `H13.md`, bu dosyanın Tur 4 bölümü, `notlar/literatur-cipa.md`.

**Test kilidi:** `karar.md`, `sonuclar/test-*`, `leaderboard-test.md`, `naif-test.csv`, `ablasyon.*`, `leaderboard.csv`,
`test_metrics.json`, `test_kilitli/` okunmadı. Val tahmin dosyaları da açılmadı. H14'ün val MSE'si önceden hesaplanmadı.

**Künye doğrulaması (WebSearch 2026-10-14):**
- Smith & Wallis 2009, OBES 71(3):331–355.
- Claeskens, Magnus, Vasnev & Wang 2016, IJF 32(3):754–762, doi:10.1016/j.ijforecast.2015.12.005.

> **Test maruziyeti beyanı.** Tur 5'in tek kartı (H14) test SONRASI bir fikirdir. Birleşim fikri 15:42 sonrasıdır;
> w ≡ 0.5 seçimi tur-4 val sonuçlarından sonra yapıldı. Makalede "Exchange test'i ve H12/H13 val sonuçları
> görüldükten sonra" diye yazılır.

### Tur 4 kapanışı (referans)
- **H11 öldü (ön-koşul):** Softplus σ başlığı "yumuşak ölüm"e girdi (K3 ölü oranı 0.83–0.88). Kök neden aktivasyon
  değil, σ parametrizasyonu.
- **H12 öldü:** Δ̄ +1.50%, 1/3 kat, 2/4 ufuk. İç-holdout α̂ < 0 → w = 0 → Repeat'e çöküş. "Model vs Repeat" ilişkisi
  bitişik dilimler arasında tersine dönüyor.
- **H13 öldü:** Δ̄ +1.88%. Temkinlilik (α* > 1) dönemler arası kararlı değil.
- **Kat bulgusu:** `lastnorm`, `baseline`'a karşı TUT (−6.71%, 3/3 kat, 12/12 hücre). Naife karşı 2/3 kat (K2 M/R 1.022).
- **FAZ II durma sayacı: 1/2.**

### Ebeveyn ve karar kuralı (aynen)
- **Ebeveyn:** `lastnorm` (tur-4 koşullu ön-planına göre: tur-4'te TUT yok, `lastnorm` ise `baseline`'a karşı TUT).
  Ā = 0.60022.
- **Kural:** TUT ⇔ Δ̄ < 0 ∧ ≥ 2/3 kat ∧ ≥ 3/4 ufuk.
- **Durma:** TUT ∧ Δ̄ ≤ −1% yoksa sayaç 2/2 olur. Exchange kapanır, `lastnorm` ile ETT'ye geçilir.

### Tur 5 kartı (tek kart; şef kararı)

Skor = beklenen kazanç (Ā, mutlak %, TUT koşulunda) × P(TUT) / maliyet (GPU-saat + uygulama).

| Öncelik | Kart | Aile | Hedef | Tek değişiklik (ebeveyn `lastnorm`) | Beklenen etki (kaynak) | P(TUT) | P(TUT ∧ Δ̄ ≤ −1%) | Maliyet | Skor | Bilgi değeri |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **H14** `lastnorm-esit` | Birleşim, sabit eşit ağırlık | Z3 + naif hedefi; H12'nin işaret hatası kanalını kapatmak | C = 0.5·P + 0.5·R; kestirim yok | Δ̄ **−1.0…+1.5%**, merkez ≈ 0. Kaynak: kimlik (w = 0.5 ebeveyni geçer ⇔ α* < 0.75; Repeat'i geçer ⇔ α* > 0.25), yayımlı oracle tavanı (K2 %1.4–2.6, K3 %0–0.4), Smith & Wallis 2009, Claeskens et al. 2016 | 0.15 | 0.05 | GPU 0 + uygulama 0.05 | 1 × 0.15 / 0.05 ≈ **3** (durma eşiğiyle 0.05 × 1.5 / 0.05 ≈ 1.5) | Orta. Puzzle sınaması (H14 ve H12, aynı seed). Naif hedefi (M/R ≤ 1 hücre sayısı). Adil kıyas `baseline-esit`. Hepsi bedava |

**Kullanıcı onayı:** H14 split, metrik, val tanımı ya da veri değiştirmiyor. **KULLANICI ONAYI GEREKEN KART YOK.**

### Neden yalnız bu kart ve nasıl okunmalı
1. **Şef kararı:** tur 5 yalnız Y2'dir. Y1 (L = 48) alınmadı, çünkü rakiple farklı geri bakış adil kıyası bozar.
2. **Maliyet sıfıra yakın:** GPU 0, CPU < 5 dk. K0 gerekmez, ETT'ye aktarım da kestirimsizdir.
3. **Dürüstlük notu (M1):** H14'ün sonucunun büyük kısmı yayımlı val bilgisinden kimlikle bellidir. K3 4/4 ufukta
   kayıp (α* 1.33 / 1.30 / 0.77 / 3.65, hepsi > 0.75). K2 4/4 ufukta kazanç (M/R > 1 ⇒ α* < 0.5). Açık kalan
   sorular: K1-h720'nin işareti (dolayısıyla K1 kat işareti), ufuk sayısı ve büyüklük. K1-h720'de kazanç için
   V/MSE_R ≳ %22 gerekir; hata-analizi §4'e göre ≈ %2–5. Bu yüzden P(TUT) düşük yazıldı. Kart ağırlıkla
   **tanısaldır**: puzzle ve naif hedefi. Şef "koşmaya değer mi" kararını bu bilgiyle verir. Tasarımcının önerisi
   koşmaktır, çünkü bedavadır ve H12'nin ön-kayıtlı Ablasyon 1'ini kapatır. Sonuç keşif diye raporlanmaz.
4. **Çoklu karşılaştırma:** birleşim ailesinde 3. deneme (H12, H13, H14). TUT alırsa "üç birleşim seçeneğinden biri,
   test ve val sonuçları görüldükten sonra" diye yazılır.

### Bütçe (tur 5)
| Kalem | GPU-saat |
|---|---|
| H14 (9 türev kayıt) + `baseline-esit` ablasyonu | 0 (CPU < 10 dk) |
| **Tur 5 toplam** | **0** |
| Kullanılan (tahlil-tur4 §6) | ≈ 4.75 / 16 |
| Ek veri setleri payı (ETTh1/ETTh2) | ≥ 4.0 (korunur) |

### Tur 5 sonrası koşullu ön-plan (kart değil; H14 sonucundan önce yazıldı)
- **H14 TUT ∧ Δ̄ ≤ −1%:** sayaç 1/2'de kalır. Zincir ucu `lastnorm-esit` olur. ETT'de birincil kazanan H14,
  ikincil `lastnorm`. Bir Exchange turu daha açılması şef kararıdır (tur 6/7).
- **H14 TUT ama Δ̄ > −1%:** sayaç 2/2 olur ve Exchange kapanır. Birincil kazanan `lastnorm` kalır. H14 ETT'de
  ikincil satır olarak ön-kayıtlı raporlanır (GPU 0).
- **H14 ÖLDÜR:** sayaç 2/2 olur ve Exchange kapanır. Kalan bütçe ETT'ye geçer, kazanan `lastnorm`. Puzzle ikincili
  (H14 ve H12) ile naif hedefi her durumda raporlanır.

### Alınmayanlar ve gerekçe (tur 5)
- **Y1 (L = 48):** şef kararı; adil kıyas (rakip L = 96) bozulur.
- **w ızgarası (0.25 / 0.75 …):** val'e bakarak w seçmek aramadır (M3). Tek değer 0.5 literatürün kanonik
  "eşit ağırlık" değeridir.
- **H11b (σ parametrizasyonu: log-σ, ilklendirme):** SAN'dan büyük sapma, kaynaklı düzeltme yok. Kart iki kez
  kapıda/ön-uçuşta öldü. ETT'den sonra, bütçe kalırsa.
- **H12b/H13b (başka iç-holdout, tek ağırlık/ufuk):** H12 öldü, çünkü dönemler arası ilişki işaret değiştiriyor. Bu
  bir kestirim-varyansı sorunu değil. Kestirimli hiçbir sürüm kat kuralını geçmeye aday değil (tahlil-tur4 §4.2.5).
- **Online ince ayar:** protokol değiştirir. **KULLANICI ONAYI GEREKİR**, otomatik koşmaz.

### Kapanmış kartlar (tur 4 itibarıyla; tur-3 tablosuna ek)
| Kart | Durum | Gerekçe (tek satır) |
|---|---|---|
| H11 `san-softplus` | ÖN-KOŞUL BAŞARISIZ (ÖLDÜR eşdeğeri) | Softplus σ̂ K3 ≈ 0.001, ölü %83–88; kök neden σ parametrizasyonu (tahlil-tur4 §4.1) |
| H12 `lastnorm-birlesim` | ÖLDÜR | Δ̄ +1.50%, 1/3 kat, 2/4 ufuk; α̂ < 0 → Repeat'e çöküş (§4.2) |
| H13 `lastnorm-olcek` | ÖLDÜR | Δ̄ +1.88%, 1/3 kat, 2/4 ufuk; temkinlilik dönemler arası kararsız (§4.2) |
