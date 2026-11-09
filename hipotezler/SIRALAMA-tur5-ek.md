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
