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
