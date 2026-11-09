# Hata Analizi (K3.5): iTransformer baseline, Exchange-Rate, YALNIZ VALIDATION

*Hazırlayan: deney-tahlilci (mod 1, hata-analizi) · 2026-10-14*

- **Girdi koşumları:** `runs/20261014-020531_baseline_s2023`, `runs/20261014-021024_baseline_s2024`,
  `runs/20261014-021448_baseline_s2025`. Resmî iTransformer `c2426e6`, Exchange `.sh` birebir, h336'da 1 epoch.
- **Veri:** `preds/val_h{H}.npz`, `data/hedefler_sl96/val_h{H}_true.npy`, ham CSV + train-scaler
  (`data/splits/exchange_rate_tslib_sl96.json`). Tüm değerler z uzayında (train StandardScaler).
- **Betik:** `scripts/hata_analizi.py`. Tekrar için:
  `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/hata_analizi.py`
  (K5'te `--runs <kazanan koşumlar> --etiket <ad> --cikti-ek _<ad>` ile aynı kırılım çıkar). Tam çıktı
  `sonuclar/hata_analizi.log` dosyasında. Kırılım tabloları `sonuclar/hata-kirilim.csv` (ufuk × değişken × blok,
  180 satır) ve `sonuclar/hata-adim.csv` (ufuk × adım-dilimi, 40 satır).
- **Sağlama:** Betiğin yeniden hesapladığı MSE'ler `metrics.json`'daki `val_mse_h*` ve `alt_grup` değerleriyle aynı
  (ör. s2023 h96 0.12534, d0 0.17231). Repeat-last değerleri `data/naive_val.csv` ile aynı (0.1282 / 0.2263 /
  0.3933 / 1.1443). Val hedefleri ham CSV'nin z satırlarıyla eşleşiyor (maks. fark < 1e-4).

> **TEST KİLİDİ.** Bu analizde `preds/test_kilitli/`, `test_metrics.json` ve `metrics.json` içindeki `test_*`
> anahtarları okunmadı, `sonuclar/replikasyon.md` açılmadı. **Test dönemine ait model-hata kırılımı K5'e
> ertelendi.** Test kilidi açılınca aynı betik, aynı blok ve skor tanımlarıyla test için koşulacak. Bağlam olarak
> yalnız K2'nin model içermeyen tanımlayıcı test ölçüleri anılmıştır.

Kısaltmalar: **M/R** = model MSE / Repeat-last MSE (<1 ise model daha iyi). **OOR** = train min–max aralığı dışı.
"Seed-ort." = 3 seed'in MSE'lerinin ortalaması.

---

## 1. Ufuk bazında: iTransformer ve Repeat-last

| H | Val pencere | ~bağımsız hedef parçası (760/H) | Model MSE seed (2023/2024/2025) | Ort. ± sd | Repeat | Δ (mutlak) | Δ % | Seed başına Δ % | Repeat'i yendiği pencere oranı |
|---|---|---|---|---|---|---|---|---|---|
| 96 | 665 | 7.9 | 0.1253 / 0.1237 / 0.1256 | 0.1249 ± 0.0011 | 0.1282 | −0.0033 | **−2.6%** | −2.2 / −3.5 / −2.0 | 0.565 |
| 192 | 569 | 4.0 | 0.2137 / 0.2088 / 0.2116 | 0.2114 ± 0.0025 | 0.2263 | −0.0149 | **−6.6%** | −5.6 / −7.7 / −6.5 | 0.575 |
| 336 | 425 | 2.3 | 0.4004 / 0.3982 / 0.4021 | 0.4003 ± 0.0020 | 0.3933 | +0.0070 | **+1.8%** | +1.8 / +1.3 / +2.2 | 0.485 |
| 720 | 41 | 1.06 | 1.0315 / 1.0058 / 1.0913 | 1.0429 ± 0.0439 | 1.1443 | −0.1014 | **−8.9%** | −9.9 / −12.1 / −4.6 | 0.780 |
| Ort. | | | 0.4428 / 0.4341 / 0.4577 | 0.4448 ± 0.0119 | 0.4730 | −0.0282 | −6.0% | | |

MAE'de de tablo aynı: −1.7% / −3.6% / **+2.0%** / −7.0%.

Şefin ilk gözlemi **doğrulandı**: h96 0.125 / 0.128, h192 0.211 / 0.226, h336 0.400 / 0.393 (Repeat daha iyi, 3
seed'in üçünde de), h720 1.04 / 1.14, seed aralığı 1.006–1.091.

Ek gözlemler:
- h720 ortalama val MSE'nin **%59'unu** oluşturuyor.
- h720'de 41 pencerenin hedefleri 760 val satırının neredeyse tamamını kaplıyor. Bağımsız hedef parçası sayısı
  yaklaşık 1. Pencere düzeyindeki "0.780 kazanma oranı" bu yüzden 41 bağımsız gözlem değil.
- h96'da model Repeat'i pencerelerin yalnız %56.5'inde yeniyor. Ortalama kazanç (−2.6%) küçük ve 3v3 seed ile
  ancak saptanabilir sınırda (MDE %2.6, §6).

## 2. Değişken bazında (8 kur)

Seed-ort. MSE, toplam hatadaki payı, M/R ve hedef OOR oranı (val):

| Değ. | h96 MSE (pay) | h192 MSE (pay) | h336 MSE (pay) | h720 MSE (pay) | M/R h96/192/336/720 | Val hedef OOR (h96) |
|---|---|---|---|---|---|---|
| 0 | 0.171 (17%) | 0.263 (16%) | 0.541 (17%) | 1.884 (23%) | 1.00 / 0.93 / 1.04 / 0.83 | 0.85 |
| 1 | 0.049 (5%) | 0.068 (4%) | 0.110 (3%) | 0.213 (3%) | 1.02 / 0.99 / 1.05 / 0.71 | 0.00 |
| 2 | 0.060 (6%) | 0.099 (6%) | 0.153 (5%) | 0.160 (2%) | 1.06 / 1.05 / 0.96 / 0.74 | 0.00 |
| **3** | **0.346 (35%)** | **0.706 (42%)** | **1.338 (42%)** | **2.607 (31%)** | 0.95 / 0.95 / 0.98 / 0.96 | 0.88 |
| 4 | 0.003 (0%) | 0.008 (0%) | 0.026 (1%) | 0.095 (1%) | 1.06 / 0.98 / 1.13 / 1.05 | 0.00 |
| 5 | 0.103 (10%) | 0.157 (9%) | 0.282 (9%) | 1.321 (16%) | 0.98 / 0.86 / 1.07 / 1.08 | 0.54 |
| 6 | 0.130 (13%) | 0.223 (13%) | 0.457 (14%) | 1.349 (16%) | 0.98 / 0.93 / 1.10 / 0.89 | 0.95 |
| OT | 0.138 (14%) | 0.167 (10%) | 0.295 (9%) | 0.713 (9%) | 0.94 / 0.85 / 0.98 / 0.85 | 0.24 |

(Val hedef OOR oranı ufka göre biraz değişiyor, tam değerler CSV'de. K2'nin val satır bazlı değerleri: 0.805 /
0 / 0 / 0.829 / 0 / 0.536 / 0.892 / 0.243.)

- `3` ve `0` birlikte hatanın **%51–59'unu** taşıyor. `4` ise neredeyse sıfır (val std'si train'in 0.15 katı).
- Değişkenler arası Spearman(OOR, MSE) = +0.78 / +0.88 / +0.88 / +0.90 (n = 8, p 0.002–0.022). Ama aynı ilişki
  Repeat'te de var. Spearman(OOR, M/R) = −0.68 / −0.56 / +0.12 / +0.34, yani tutarlı bir işaret yok. OOR değişkenleri
  **her iki yöntem için** zor. iTransformer'a özgü bir "aralık dışı çöküşü" görülmüyor.
- Değişken içinde, OOR hedef hücrelerinde hata belirgin biçimde yüksek (ör. h720 `3`: OOR 3.00 / iç 0.36;
  `6`: 1.47 / 0.04). Repeat'te de aynı desen var (3.15 / 0.34; 1.65 / 0.06). OOR hücrelerinde M/R iç hücrelere göre
  birkaç puan daha kötü (ör. h96 `6`: 0.98 / 0.80; h720 `0`: 0.83 / 0.78), ama fark küçük.
- Mekanizma notu: `use_norm` (afinsiz, NST tarzı) girdi penceresinin kendi ortalamasını çıkarıyor. Bu yüzden mutlak
  seviyenin train aralığı dışında olması modele doğrudan girmiyor. Bu, OOR'un M/R'yi açıklamamasıyla tutarlı.

## 3. Pencere ve zaman bazında: dayanıklılık metriklerinin val ön-uygulaması

### 3.1 Sabitlenmiş tanımlar (ön-kayıt adayı; val ve testte aynen)

- **Blok.** Dilimin **kendi** satırları `np.array_split` ile K = 4 bitişik kronolojik bloğa bölünür.
  - Val: [5311, 5501), [5501, 5691), [5691, 5881), [5881, 6071), her biri 190 satır.
  - Test: K2'nin `test_Q1..Q4` blokları (380 / 379 / 379 / 379 satır).
  - Her (pencere i, adım h, değişken d) hücresi **hedef satırına** (`hedef_ilk_satir[i] + h`) göre bloğa atanır.
    Pencere başlangıcına göre atama kullanılmaz, çünkü uzun ufuklarda geç blokları kapsamaz (K2 önerisi).
  - `MSE_k`: k bloğundaki hücrelerin ortalama karesel hatası (z uzayı).
  - **`worst_block_mse = max_k MSE_k`**, **`worst_ratio = worst_block_mse / tüm-hücre MSE`**.
  - Raporlama: ufuk başına, seed başına; seed-ort. ± sd. Repeat'in aynı değerleri yanında verilir.
  - Not: Kenar bloklar daha az hücre içerir (h96: 109 440 / 145 920 / 145 920 / 109 440), çünkü kenar satırları
    daha az pencere kapsıyor.
- **Kayma–hata korelasyonu** (literatur-cipa §6.2):
  - Kayma skoru: `s_io(i) = mean_d |μ(horizon_i,d) − μ(lookback_i,d)| / (σ(lookback_i,d) + 0.01)`. z uzayında
    hesaplanır, σ ddof = 0, ε = 0.01 z birimi.
  - **`rho_shift = Spearman(s_io, pencere MSE)`** (pencere MSE = 8 değişken × H adım ortalaması; seed başına
    hesaplanıp ortalanır).
  - **`top10_ratio`** = s_io'nun en üst %10'undaki pencerelerin MSE'si / tüm pencere MSE'si. Ek olarak bu
    pencerelerde M/R.
  - s_io gerçek horizon'u kullandığı için **yalnız değerlendirmede** hesaplanır. Eğitimde ya da seçimde girdi
    olamaz.
  - Uyarı: Pencereler örtüşüyor, bu yüzden Spearman p-değerleri geçersizdir ve raporlanmaz. Yalnız ρ ve yön kullanılır.

### 3.2 Val ön-uygulaması (seed-ort.)

| H | Blok MSE B1 / B2 / B3 / B4 | Repeat B1–B4 | M/R B1–B4 | worst_block_mse (sd) | worst_ratio | Repeat worst / ratio |
|---|---|---|---|---|---|---|
| 96 | 0.092 / 0.067 / **0.222** / 0.106 | 0.110 / 0.078 / 0.214 / 0.099 | 0.83 / 0.86 / **1.04 / 1.07** | 0.2217 (0.0013), B3 | 1.776 | 0.2141 / 1.670 |
| 192 | 0.154 / 0.155 / **0.314** / 0.177 | 0.195 / 0.188 / 0.316 / 0.156 | 0.79 / 0.82 / 0.99 / **1.14** | 0.3136 (0.0013), B3 | 1.484 | 0.3159 / 1.396 |
| 336 | 0.214 / **0.461** / 0.448 / 0.268 | 0.195 / 0.425 / 0.440 / 0.361 | **1.10 / 1.08 / 1.02** / 0.74 | 0.4606 (0.005), B2 | 1.151 | 0.4404 / 1.120 |
| 720 | 0.301 / 1.125 / **1.660** / 1.003 | 0.328 / 1.247 / 1.794 / 1.120 | 0.92 / 0.90 / 0.93 / 0.90 | 1.6597 (0.062), B3 | 1.592 | 1.7936 / 1.567 |

En kötü blok bütün seed'lerde aynı (h96, h192, h720'de B3; h336'da B2). Modelin worst_ratio değeri h96, h192 ve
h336'da Repeat'inkinden yüksek. Yani ortalamaya göre model, en kötü blokta Repeat'ten daha çok bozuluyor.

| H | rho_shift model (seed sd) | rho_shift Repeat | ρ(s_io, M/R) | top10_ratio model | top10_ratio Repeat | M/R üst %10 |
|---|---|---|---|---|---|---|
| 96 | +0.185 (0.031) | +0.370 | −0.27 | 1.30 | 1.26 | 1.01 |
| 192 | +0.418 (0.022) | +0.585 | −0.54 | 1.17 | 1.33 | 0.82 |
| 336 | +0.810 (0.005) | +0.664 | +0.45 | 1.84 | 1.63 | **1.15** |
| 720 | +0.790 (0.141) | +0.868 | +0.39 | 1.36 | 1.12 | **1.11** (5 pencere) |

Zaman seyri ve başka kayma skorları:
- h96 ve h192'de modelin Repeat'e üstünlüğü val dönemi boyunca **azalıyor**. Spearman(pencere sırası, M/R) =
  **+0.39 / +0.63**.
- Aynı yönde, girdi penceresinin train ortalamasına uzaklığı `s_in_mu = mean_d |μ_z(lookback)|` ile
  ρ(s_in_mu, M/R) = +0.38 / +0.57.
- Ancak val'de pencere sırası ile s_in_mu neredeyse aynı şeyi ölçüyor (ρ = 0.84 / 0.93; h336'da 0.997). **"Zaman"
  ile "seviye uzaklığı" val'de ayrıştırılamıyor.**
- h336 ve h720'de yön ters: ρ(sıra, M/R) = −0.50 / −0.40.
- Girdi oynaklığı `s_sd_lb`, M/R ile zayıf ilişkili (+0.02…+0.18).

## 4. Tahmin adımı bazında (adım 1..H)

M/R, ufku 10 eşit adım dilimine bölerek (`hata-adim.csv`):

| H | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 96 | **1.45** | 1.12 | 1.06 | 1.02 | 0.99 | 0.97 | 0.95 | 0.94 | 0.93 | 0.93 |
| 192 | **1.23** | 1.01 | 0.97 | 0.94 | 0.92 | 0.92 | 0.94 | 0.93 | 0.92 | 0.91 |
| 336 | **1.19** | 1.03 | 1.03 | 1.00 | 0.99 | 0.99 | 1.02 | 1.03 | 1.03 | 1.02 |
| 720 | 1.03 | 0.88 | 0.89 | 0.87 | 0.92 | 0.93 | 0.93 | 0.89 | 0.91 | 0.91 |

- **Adım 1** MSE: model 0.0117 / 0.0154 / 0.0170 / 0.0429, Repeat 0.0033 / 0.0036 / 0.0026 / 0.0022. Model
  3.5× / 4.3× / 6.5× / **19.5×** daha kötü.
- **Adım H** MSE: model 0.223 / 0.339 / 0.672 / 1.250, Repeat 0.238 / 0.375 / 0.662 / 1.405.
- **Model Repeat'e yakınsıyor mu, sapıyor mu?** Adım H'de modelin Repeat'ten uzaklığı ‖P−R‖² 0.012 / 0.019 /
  0.021 / 0.033. Gerçeğin Repeat'ten uzaklığı ‖Y−R‖² ise 0.238 / 0.375 / 0.662 / 1.405. Model, gerçek hareketin
  ancak %1–5'i büyüklüğünde Repeat'ten ayrılıyor. Pratikte **"Repeat + küçük sabit ofset"** gibi davranıyor.
  Kazancı da bu küçük ofsetin işaretinin çoğunlukla doğru olmasından geliyor.
- Adım H'de yön isabeti, sign(P−son) = sign(Y−son): 0.58 / 0.63 / **0.47** / 0.64. Yazı-turaya yakın, h336'da
  altında.
- Tanısal ölçek katsayısı α* (Y−R ≈ α*(P−R), val'de EKK): 0.64 / 0.95 / 0.31 / 2.00. Ufka göre tutarsız: h96'da
  ofset fazla büyük, h720'de fazla küçük, h336'da büyük ölçüde yanlış yönlü. Bu katsayı val'e fit edildiği için seçim
  aracı olarak **kullanılamaz**. Yalnız tanıdır.

## 5. Bias, seviye ve varyans (use_norm'un izi)

| H | Ort. bias P−Y (model) | Repeat bias | Ufuk içi gerçek trend, Y_H−Y_1 (değişken ort. aralığı) | Ufuk içi tahmin trendi | std(P)/std(Y) pencere içi | s_io medyan gerçek / tahmin (oran) |
|---|---|---|---|---|---|---|
| 96 | −0.035 | −0.063 | +0.04 … +0.20 | −0.003 … +0.016 | 0.196 | 1.67 / 1.22 (0.68) |
| 192 | −0.069 | −0.120 | +0.05 … +0.30 | +0.001 … +0.050 | 0.158 | 2.19 / 1.30 (0.57) |
| 336 | **−0.272** | −0.240 | +0.00 … +0.54 | **−0.066 … −0.007** | 0.166 | 2.99 / 0.89 (0.30) |
| 720 | −0.801 | −0.872 | +0.42 … +1.61 | −0.023 … +0.005 | 0.163 | 6.00 / 1.07 (**0.18**) |

- **Sistematik eksik tahmin.** Val döneminde kurlar z uzayında yükseliyor. Model de Repeat de geride kalıyor ve
  bias ufukla büyüyor. h336'da model Repeat'ten bile daha negatif: 1 epoch'luk model ufuk içinde **düşüş** (ortalamaya
  dönüş) tahmin ediyor, gerçek ise yükseliş.
- **Aşırı yumuşatma.** Pencere içi tahmin std'si gerçeğinkinin yalnız **0.16–0.20** katı (bütün değişkenlerde 0.02–0.36).
  Pencereler arasında std(P_H − son) / std(Y_H − son) = 0.22 / 0.21 / 0.18 / 0.34.
- **use_norm tavanı (mekanizma).** Çıkış `μ_lb + σ_lb · f(·)` biçiminde. Modelin horizon ortalamasını lookback
  ortalamasından σ_lb birimiyle ne kadar uzaklaştırabildiğine bakıldığında: gerçek kaymanın medyanı H ile 1.67'den
  6.00'ya çıkıyor, modelinki ise ufuktan bağımsız olarak **~0.9–1.3'te doyuyor**. Oran 0.68'den 0.18'e düşüyor.
  Val'de σ_lb küçük (medyan 0.15–0.19 z; train'de 1). Bu yüzden normalize uzayda gereken çıkış büyüklüğü, modelin
  eğitimde gördüğü dağılımın çok dışında kalıyor.
- **Seviye çapası.** Adım 1'de |P_1 − son| = 0.066 / 0.079 / 0.089 / 0.148 iken gerçek |Y_1 − son| = 0.030–0.037.
  Ofset lookback ortalamasına doğru çekiliyor: corr(P_1 − son, μ_lb − son) = +0.23 / +0.30 / **+0.68** / +0.16;
  β (P_1 − μ_lb ≈ β(son − μ_lb)) = 0.95 / 0.93 / **0.72** / 0.98. Ortalama işaretli ofset h96, h192 ve h720'de
  pozitif (+0.019 / +0.030 / +0.090), yani val'in yükseliş yönünde.
- **Tanısal kontrol (iddia değil).** Tahmini adım 1'de son değere yeniden çapalamak (P − P_1 + son) MSE'yi
  **kötüleştiriyor**: +2.0% / +5.4% / +1.3% / +14.4%. Yani ofset uzun adımlar için yararlı bilgi taşıyor. Sorun
  "yanlış seviye" değil, **yanlış şekil**: ufka yayılması gereken bir rampa yerine adım 1'den itibaren sabit bir
  basamak var. Repeat ile 0.5 karışım h96'da −1.0%, h336'da −2.0%, h192'de +1.6%, h720'de +4.1%.

## 6. Seed varyansı ve minimum saptanabilir etki

MDE: 3 seed'e karşı 3 seed, iki örneklem t-testi, α = 0.05 (iki yönlü), güç 0.8, df = 4:
MDE ≈ 3.03 · sd_seed. Eşleştirilmiş seed tasarımı (aynı seed çiftleri) bunu ancak varyantlar arası korelasyon
yüksekse düşürür. Bu korelasyon ölçülmedi.

| | h96 | h192 | h336 | h720 | val_mse_avg |
|---|---|---|---|---|---|
| sd_seed | 0.0011 | 0.0025 | 0.0020 | 0.0439 | 0.0119 |
| CV | 0.84% | 1.17% | 0.49% | **4.21%** | 2.7% |
| MDE (mutlak) | 0.0032 | 0.0075 | 0.0059 | **0.133** | 0.036 |
| MDE (göreli) | 2.6% | 3.5% | 1.5% | **12.8%** | **8.1%** |

- Bu MDE yalnız **eğitim rastlantısallığını** kapsıyor. Pencere örnekleme belirsizliği dahil değil: h720'de
  ~1 bağımsız hedef parçası var, h336'da ~2.3. Gerçek belirsizlik bundan büyük.
- **Sonuç:** Ortalama metrik (val_mse_avg) üzerinden 3 seed ile %8'in altındaki bir etki ayırt edilemiyor. Bunun
  nedeni h720'nin seed varyansı. Literatürdeki A7/A5 etkileri (%0–20, literatur-cipa §5.5) bu eşiğin altında ya da
  çevresinde. Kısa ufuklarda (h96–h336) %1.5–3.5'lik etkiler ayırt edilebilir.
- h720'de erken durdurma, resmî val kaybıyla 41 pencerenin 32'si üzerinde yapılıyor (drop_last + shuffle). En iyi
  epoch'lar 2–5 arası (koşulan epoch sayısı 6 / 8 / 5). Resmî val kaybı ile bizim tam-val MSE'miz arasındaki fark
  0.006–0.023. Seçimin kendisi gürültülü.

## 7. Sentetik tarih sütunu: yalnız HİPOTEZ ADAYI (ölçülmedi)

- Rakip kodu (okundu, değiştirilmedi): `layers/Embed.py: DataEmbedding_inverted`,
  `torch.cat([x, x_mark.permute(0,2,1)], 1)`. `timeF` + `freq='h'` ile 4 zaman özniteliği (HourOfDay, DayOfWeek,
  DayOfMonth, DayOfYear) **normalize edilmeden** 4 ek değişken token'ı olarak giriyor. Böylece 8 + 4 = 12 token
  oluyor. HourOfDay günlük veride sabit.
- K2'ye göre `date` büyük olasılıkla sentetik: hafta sonları dahil kesintisiz takvim. O hâlde DayOfWeek ve
  DayOfYear gerçek takvim bilgisi taşımıyor.
- **Hipotez adayı (A12/A6):** "x_mark token'larını kaldırmak (ya da sabitlemek) val MSE'yi değiştirmez (|Δ| < MDE)."
  Değişiklik çıkarsa iki açıklama var: ya sahte bir konum sinyali kullanılıyor ya da token sayısı dikkat
  dinamiğini etkiliyor. Bu bölüm **yorum değildir**. Etki ölçülmedi ve ölçülmeden makalede yorumlanmamalı.

## 8. Hız ve bellek profili (`.done` + `metrics.json`, `KURULUM.md` §6)

| | h96 | h192 | h336 | h720 | Seed toplamı |
|---|---|---|---|---|---|
| Eğitim süresi (s; 3 seed aralığı) | 65–96 | 67–82 | 7.7–8.2 (1 epoch) | 28–54 | 253–279 (`.done` sure_sn 260–288) |
| Koşulan epoch | 10 / 9 / 8 | 10 / 10 / 10 | 1 | 6 / 8 / 5 | |
| Tepe VRAM (torch) | 27 MB | 28 MB | 29 MB | 31 MB | |
| Parametre | 224 224 | 236 608 | 255 184 | 304 720 | |

- 3 seed'lik tam baseline yaklaşık **14 dk** GPU tutuyor (RTX 3060 Laptop, nw = 0).
- Bellek sınırlayıcı değil. Bütçe (≤ 8 GPU-saat, harcanan K3 dahil) kabaca ~30 adet "varyant × 3 seed" tam koşum setine yetiyor
  (üst sınır; değerlendirme ve duman/ön-uçuş payı hariç).
- h336'nın 8 s'lik eğitimi resmî 1-epoch ayarından kaynaklanıyor (bkz. Z5).

---

## Ölçülmüş zayıflıklar → manipülasyon aileleri

Etki aralıkları için çıpa `notlar/literatur-cipa.md` §5.5'tir. Aile kodları `manipulasyon-kutuphanesi.md`'den alındı.
Beklenen yönler **hipotezdir**. K4 kartları ön-kayıtla sınanmalı.

**Z1. Seviye kayması tavanı / trend körlüğü (en büyük hata kütlesi).**
- Ölçüm:
  - Ufuk içi tahmin trendi yaklaşık 0. Gerçek trend +0.04…+1.61 z.
  - Pencere içi std(P)/std(Y) = 0.16–0.20.
  - Normalize kayma oranı (tahmin / gerçek s_io) 0.68 → 0.57 → 0.30 → **0.18**.
  - Bias: −0.035 / −0.069 / −0.272 / −0.801.
  - Uzun ufuklar ortalamanın %59'u (h720) + %22'si (h336).
- Aile: **A7**. SAN, dilim düzeyinde gelecek ortalama ve std'yi tahmin ettiği için bu tavanı doğrudan hedefliyor.
  RevIN (afin) yalnız kısmen hedefliyor. Dish-TS horizon seviyesini tahmin ediyor, ama Exchange'de literatürde hep
  kötü (§5.2). FAN ikinci aday.
- Beklenen yön: h336/h720'de bias ve MSE azalır. Kısa ufuklarda etki nötre yakın.
- Risk: Val'in yükseliş trendine aşırı uyum. K2'nin tanımlayıcı ölçülerine göre test dönemi farklı bir seyir
  izliyor (Q3/Q4'te z ortalaması düşük). Val'de "trend yakalayan" varyant testte cezalandırılabilir. Kart
  "görülmemiş" eşiğini (avg < 0.280 ya da −25%) aynen taşımalı.

**Z2. Kısa adımlarda basamak hatası (seviye çapası lookback ortalamasına).**
- Ölçüm:
  - Adım-1 MSE Repeat'in 3.5× / 4.3× / 6.5× / 19.5× katı.
  - İlk adım dilimi M/R 1.45 / 1.23 / 1.19 / 1.03.
  - corr(P_1 − son, μ_lb − son) +0.16…+0.68; β 0.72–0.98.
- Aile: **A7** (son değere çapalı normalizasyon, NLinear tarzı "son değeri çıkar"; RevIN varyantı) ya da **A6**
  (Repeat-artık başlığı: P = son + g(·), g adım 1'de 0'dan başlar).
- Beklenen yön: h96 ve h192'de ilk %10–30 adımda kazanç, toplamda küçük (h96'nın Repeat'e karşı tüm marjı −2.6%).
- Uyarı: Naif yeniden-çapalama tanısal olarak MSE'yi +1.3…+14.4% **kötüleştiriyor**. Çözüm kaydırma değil, şekil
  (rampa). Bu bir öldürme kuralı adayı: "ilk dilim M/R düzelir ama toplam MSE kötüleşirse ÖLDÜR".

**Z3. Zamanla aşınan üstünlük ve kayma–hata bağı.**
- Ölçüm:
  - h96/h192 blok M/R B1→B4: 0.83→1.07 ve 0.79→1.14.
  - ρ(pencere sırası, M/R) = +0.39 / +0.63.
  - worst_ratio model 1.78 / 1.48 / 1.15, Repeat 1.67 / 1.40 / 1.12.
  - rho_shift model 0.19 / 0.42 / 0.81 / 0.79.
  - s_io üst %10'da M/R h336 1.15, h720 1.11.
  - Sınırlama: zaman ile seviye uzaklığı val'de karışık (ρ 0.84–0.93).
- Aile: **Online / sürekli öğrenme** (konu.md strateji 4). Gecikmeli geri bildirim (H adım) zorunlu, literatur §5.4.
  Exchange için etki büyüklüğü literatürde BULUNAMADI, beklenti yazılmamalı. İkinci aile: **A5** zaman-grubu DRO
  (train'i kronolojik gruplara bölüp en kötü grup kaybı). Literatürde 0 ile −8% (FOIL Tab.2).
- Beklenen yön: B3/B4 M/R düşer, worst_block_mse azalır. Ortalama MSE'de küçük değişim.

**Z4. Hata iki değişkende yoğunlaşmış.**
- Ölçüm:
  - `3` %31–42, `0` %16–23 (ikisi birlikte %51–59).
  - Değişkenler arası Spearman(OOR, MSE) = +0.78…+0.90.
  - Spearman(OOR, M/R) tutarsız (−0.68…+0.34). Bu zorluk modele özgü değil.
- Aile: **A5** (değişken ağırlıklı ya da robust kayıp; Huber veya değişken-grubu DRO). Etki literatürde küçük.
  Ayrı bir kart yerine Z1/Z3 kartlarında `alt_grup` (değişken) kırılımı **zorunlu ikincil metrik** olmalı.
- Beklenen yön: `3` ve `0`'da azalma. Diğer değişkenlerde bedel riski var (M7).

**Z5. h336 protokol artefaktı: Repeat'ten kötü tek ufuk.**
- Ölçüm:
  - M/R 1.018 (3/3 seed'de > 1).
  - Adım H yön isabeti 0.47, α* 0.31, β 0.72.
  - Ufuk içi tahmin trendi negatif, gerçek pozitif.
  - Train kaybı 1 epoch sonunda 0.470. Kıyas: h192 epoch-1 val 0.232'den 10 epoch'ta 0.213'e iniyor.
- Aile: **A11 / A12**. `train_epochs = 10` (öteki ufuklarla eşit) ayrı bir **protokol bulgusu** kartı olarak
  denenmeli. Resmî baseline değiştirilmez (Şef kaydı). Rapor "resmî betik vs eşitlenmiş epoch" iki satırıyla yapılır.
- Beklenen yön: h336 M/R ≤ 1 (h192'nin −6.6%'sına benzer bir aralık makul, ama kaynaklı değil).
- Risk: Bu kazanç bir manipülasyonun katkısıyla karışmamalı. Her A7 kartının ebeveyni resmî 1 epoch'tur.

**Z6. Seçim metriğinin gürültüsü (h720 baskın).**
- Ölçüm:
  - h720 seed sd 0.044 (CV %4.2), MDE %12.8.
  - val_mse_avg MDE %8.1.
  - h720 ortalamanın %59'u. ~1 bağımsız hedef parçası.
  - Erken durdurma 32/41 pencereyle.
- Aile: **A12** (ölçüm tasarımı) ve kapanışta **A9** (seed ensemble, adil kıyasla).
- Öneri (ön-kayda, sonuçtan önce): K4 kartlarında birincil seçim val_mse_avg kalır. Ek olarak
  **ufuk-başına M/R** ve **"ufukların en az 3'ünde aynı yön"** tutarlılık kuralı yazılsın. h720 tek başına karar
  verdirmesin. Bu, eşiği gevşetmek değil, gürültü yapısını kayda almaktır.

---

## Hipotez aileleri için öncelik önerisi

Gerekçe: hata kütlesi × başarı olasılığı / maliyet. Maliyet her koşum için ~14 dk × 3 seed.

1. **A7-SAN** (Z1, Z3). En büyük kütleyi (uzun ufuk bias ve trend tavanı) doğrudan hedefliyor. Literatürde en yakın
   analog PatchTST→SAN −18.7%. Kaymaya dayanıklılık metrikleriyle (worst_block, rho_shift) birlikte raporlanmalı.
2. **A7-RevIN (afin)** (Z1, Z2). Ucuz kontrol kolu. Beklenti nötre yakın (−3…+5%). SAN'ın kazancını "herhangi bir
   normalizasyon değişikliği" etkisinden ayırır.
3. **A11/A12: h336 epoch eşitleme** (Z5). Ayrı protokol kartı. Ucuz (~1.5 dk/seed). Kazanç neredeyse kesin
   görünüyor, ama makalenin katkısı değil, protokol bulgusu.
4. **A6/A7: son değere çapalı artık başlık** (Z2). Kısa ufuk ve ilk adım hatası için. Toplam etkisi küçük
   olabilir. Öldürme kuralı Z2'deki gibi.
5. **Online ince ayar, gecikmeli geri bildirimli** (Z3). Konu stratejisinde var, ama Exchange için literatür yok.
   Bütçe izin verirse 3. veya 4. turda. Ayrı tablo.
6. **A5 kayma-dayanıklı kayıp** (Z3, Z4). Beklenen etki küçük (0…−8%) ve 3 seed MDE'sinin (avg %8) altında. Bu
   yüzden kısa ufuk ya da blok metriğinde değerlendirilmeli.
7. **A7-Dish-TS.** Konu kartında var, ama literatürde Exchange'de hep kötü. Yalnız "kıyas satırı" olarak; beklenti
   kötüleşme.
8. **x_mark ablasyonu** (§7, hipotez adayı). Ucuz ve analiz makalesine (B çıkışı) malzeme.

## Sınırlılıklar

- Yalnız 3 seed. MDE ufka göre %1.5–12.8, ortalamada %8.1. Alan protokolü ≥ 5 seed öneriyor.
- h720 val yalnız 41 pencere (~1 bağımsız hedef parçası). Pencere düzeyi oranlar ve Spearman değerleri bu ufukta
  çok kırılgan. s_io üst %10 yalnız 5 pencere. h720'de bütün girdiler train aralığı dışında olduğu için s_oor_in
  korelasyonu tanımsız.
- Val, testten daha kaymış (K2: MMD² 1.00 / 0.58) ve iki yıllık tek bir yükseliş dönemi. Val'de görülen bias yönü ve
  trend örüntüsü teste genellenmeyebilir. Bulgular "val'de ölçüldü" olarak okunmalı.
- Pencereler örtüşüyor, bu yüzden korelasyonlarda p-değeri raporlanmadı. Blok sayısı 4, worst_block istatistiği
  kaba.
- Zaman ile seviye uzaklığı val'de karışık. Z3'ün "zaman" ve "kayma" yorumları ayrıştırılamıyor.
- Değişken ↔ ülke eşlemesi bir varsayım (K2). `3` "İsviçre" olarak adlandırılmamalı, yalnız sütun adıyla anılmalı.
- §4 ve §5'teki α*, yeniden-çapa ve 0.5-karışım değerleri val'de hesaplanan **tanısal** büyüklükler. Seçim ya da
  iddia için kullanılamaz.
- Test dönemi model-hata kırılımı **K5'e ertelendi** (test kilidi).
