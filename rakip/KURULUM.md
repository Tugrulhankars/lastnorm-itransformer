# Rakip Kurulumu (K3): iTransformer, Exchange-Rate

*Hazırlayan: rakip-kurucu · 2026-10-14 · Makine: Windows 11 Pro (10.0.26200), RTX 3060 Laptop 6 GB, sürücü 566.36*

## 1. Kod ve commit (R1)

| Kalem | Değer |
|---|---|
| Repo | https://github.com/thuml/iTransformer |
| Commit (pinli) | `c2426e68ca13f74aaec08045c5c724d8ad328124` (2025-07-17, "Update README.md"; son kod değişikliği e5b8223, 2025-03-13) |
| Yerel yol | `rakip/iTransformer/` |
| Resmî betik | `scripts/multivariate_forecasting/Exchange/iTransformer.sh` |
| Değişiklik | **Yok.** `git status --porcelain` boş. `YAMALAR.md` gerekmedi; Windows ve sürüm uyumu için gereken her şey sarmalayıcıda (`scripts/rakip_kos.py`) yapıldı (§4). |

```bash
# WS kökünden (Git Bash)
cd rakip && git clone https://github.com/thuml/iTransformer.git && cd iTransformer
git checkout c2426e68ca13f74aaec08045c5c724d8ad328124
git rev-parse HEAD        # c2426e68ca13f74aaec08045c5c724d8ad328124
git status --porcelain    # boş olmalı
```

## 2. Bağımlılıklar (deney ortamı `env/.venv`)

Repo `requirements.txt`: pandas==1.5.3, scikit-learn==1.2.2, numpy==1.23.5, matplotlib==3.7.0, torch==2.0.0,
reformer-pytorch==1.4.4. K1 ortamı: Python 3.11.4, torch 2.4.1+cu121 (CUDA 12.1, cuDNN 9.1.0), numpy 1.26.4,
pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2. Eklenenler:

```bash
PY=env/.venv/Scripts/python.exe
# reformer-pytorch: exp_basic.py tüm modelleri import ediyor (iTransformer kullanmasa da LSHSelfAttention import'u gerekli).
# DİKKAT: düz "pip install reformer-pytorch==1.4.4" yeni local-attention'ı çekip torch'u 2.14.1'e (CPU) yükseltiyor.
# Doğru sıra: --no-deps ve 2021 dönemi alt bağımlılıkları.
$PY -m pip install --no-deps "reformer-pytorch==1.4.4" "local-attention==1.4.4" "product-key-memory==0.1.10" "axial-positional-embedding==0.2.1"
$PY -m pip check                      # No broken requirements found.
$PY -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 2.4.1+cu121 True
$PY -m pip freeze > env/requirements.lock                                    # 35 paket
```

Kurulum sırasında olan: ilk denemede `pip install reformer-pytorch==1.4.4` torch'u 2.14.1'e yükseltti. Yeni alt
bağımlılıklar kaldırıldı, torch 2.4.1+cu121 resmî cu121 indeksinden geri kuruldu
(`pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`) ve yukarıdaki `--no-deps`
kurulumu yapıldı. Yan etki: setuptools 65.5.0 → 84.0.0 oldu (`pip freeze`'de görünmez; deneyi etkilemez).

**pandas 3.0.6 korundu (pandas<3 pinlenmedi).** Gerekçe ölçüldü: `Dataset_Custom` pandas 3 altında
(a) üç bölümün zaman özniteliklerini (`timeF`, freq='h' → HourOfDay, DayOfWeek, DayOfMonth, DayOfYear)
pandas'tan bağımsız saf-Python `datetime` formülüyle **birebir** (maks. mutlak fark 0.0) üretiyor;
(b) StandardScaler `mean_`/`scale_` değerleri K2'nin `data/splits/exchange_rate_tslib_sl96.json` dosyasındaki
yalnız-train değerleriyle aynı; (c) pencere sayıları (train/val/test, H=96: 5120/665/1422; H=720: 4496/41/798)
manifestle aynı. pandas'ın dokunduğu her çıktı pandas 1.5.3 tanımıyla özdeş, o yüzden sürüm düşürme gerekmedi.

## 3. Protokol: resmî betik birebir (R3)

Sarmalayıcı hiperparametreleri **elle kopyalamıyor, resmî dosyalardan okuyor**:
- `run.py` içindeki 50+ `parser.add_argument(...)` çağrısı AST ile çıkarılıyor. Varsayılanlar dosyanın kendisinden geliyor.
- `iTransformer.sh` ayrıştırılıyor ve her `python -u run.py` bloğu bir ufuk oluyor. Argümanlar aynen kullanılıyor.

Etkin ayar (her ufuk): `--is_training 1 --model iTransformer --data custom --features M --seq_len 96
--label_len 48 --e_layers 2 --enc_in 8 --dec_in 8 --c_out 8 --d_model 128 --d_ff 128 --n_heads 8 --dropout 0.1
--embed timeF --freq h --batch_size 32 --learning_rate 1e-4 --lradj type1 --patience 3 --loss MSE --use_norm 1
--des Exp --itr 1`. `train_epochs`: 96/192/720 → 10 (varsayılan), **336 → 1** (betikteki satır, kullanıcı kararıyla
korundu). Erken durdurma resmî `vali()` kaybıyla yapılıyor (bs32, shuffle, drop_last=True). En iyi checkpoint
yükleniyor. Test resmî `test()` ile yapılıyor (batch 1, örnek kaybı yok). Metrik ölçekli uzayda, `--inverse` kapalı.

Veri: `data/exchange_rate/exchange_rate.csv` (K2, sha256 `48b4d9d3…89842`; sarmalayıcı her koşumda hash'i doğruluyor,
yeniden indirmiyor). `--root_path` bu klasöre, `--checkpoints` ise `{RUN_DIR}/ckpt/` klasörüne yönlendiriliyor.

### Protokol farkları

| # | Fark | Resmî / makale | Bizde | Gerekçe ve etki |
|---|---|---|---|---|
| 1 | Seed | `run.py`: `fix_seed=2023` sabit; her ufuk ayrı süreç | Seed ∈ {2023, 2024, 2025}; sarmalayıcı her ufukta `Exp(args)`'tan hemen önce `random.seed(s); torch.manual_seed(s); np.random.seed(s)` çağırıyor (run.py'deki üç çağrının aynısı, aynı noktada) | Kullanıcı kararı (3 seed). s=2023 resmî süreçle aynı tohum durumunu veriyor. Rakip dosyası değişmedi. Makale Tab.5 5 seed kullanıyor; bizde 3 (Sınırlılık). |
| 2 | `num_workers` | 10 | **0** (`--num-workers 0`), **protokol dışı teknik uyum** | Windows'ta DataLoader işçileri `spawn` ile başlıyor ve her iterator'da 10 süreç torch'u yeniden import ediyor. Ölçüm (H=96, 1 epoch): nw=10 → 296 s (0.74 s/iter), nw=0 → 46 s. **Sonuç bit-bit aynı**: aynı seed'de train kaybı 0.1654488, resmî val kaybı 0.1358937, val_mse 0.13624180853366852; val tahminlerinde maks. fark 0.0, test MSE farkı 0.0. Nedeni: torch 2.4'te `_BaseDataLoaderIter` ve `RandomSampler` ana süreçteki RNG'yi işçi sayısından bağımsız tüketiyor, `Dataset_Custom.__getitem__` ise rastgelelik içermiyor. Boş RAM ~4 GB; 10 işçi bellek riski de taşıyordu. |
| 3 | Donanım | Makale: tek NVIDIA P100 16 GB (Ek A.2); betik `CUDA_VISIBLE_DEVICES=3` | RTX 3060 Laptop 6 GB, `cuda:0` | Tek GPU var. Bu küçük modelde bellek sınırı yok (tepe 31 MB). Donanımın sayısal etkisinin nicel kaynağı BULUNAMADI (literatur-cipa §3.1). |
| 4 | Yazılım | torch 2.0.0, pandas 1.5.3, numpy 1.23.5, sklearn 1.2.2 (requirements.txt); OS büyük olasılıkla Linux | torch 2.4.1+cu121, pandas 3.0.6, numpy 1.26.4, sklearn 1.9.1, Windows 11 | Veri hattının özdeşliği doğrulandı (§2). torch sürümünün kernel düzeyindeki sayısal farkı ölçülmedi; tolerans 0.008 (literatur-cipa §3.1) bu tür farka yer bırakıyor. |
| 5 | Ek değerlendirme | Resmî kod val'i yalnız erken durdurma için kullanıyor (bs32, drop_last=True, shuffle; H=720'de 41 pencerenin 32'si) | Ek olarak **bizim val metriğimiz**: en iyi checkpoint, val'in tüm pencereleri, batch 1, shuffle ve drop_last yok. Test ile aynı tanım. | K4 seçim metriği (`val_mse_avg`). Eğitime ve erken durdurmaya etkisi yok: değerlendirme `train()` bittikten sonra yapılıyor ve RNG kullanmıyor. |
| 6 | Çıktı yolları | `./checkpoints`, `./results`, `./test_results` (PDF'ler), `result_long_term_forecast.txt` cwd'ye yazılıyor | `{RUN_DIR}/ckpt/`; test çıktıları `{RUN_DIR}/test_kilitli/` (resmî `test()` çağrılmadan önce `chdir`). Resmî `pred.npy`/`true.npy` npz'ye alındıktan sonra siliniyor (`metrics.npy` kalıyor). | Test kilidi (K4 test görmez) ve disk. Hesaba etkisi yok. |
| 7 | Log | Resmî kod her epoch'ta `Test Loss` basıyor | Ham log `{RUN_DIR}/test_kilitli/rakip_stdout_h<H>.log` dosyasında. `stdout.log`'da `Test Loss: <kilitli>` ve `mse:` satırı maskeli. | Test körlüğü (M4). Rakip kodu çalışmaya devam ediyor, yalnız stdout sarmalanıyor. |
| 8 | Erken durdurma (canlı) | yok | `kosum.py --kural`: NaN/Inf, `val_loss`>3.0 veya `loss`>3.0 (min_adim 1), `val_loss`<0.01, 900 s ilerlemesiz | Güvenlik. Eşikler: train-ortalaması tahmincisinin val MSE'si 5.67–6.23, Repeat val 0.128–1.144 (`data/naive_val.csv`). Normal koşumda tetiklenmez (ölçülen en yüksek resmî val kaybı 1.20, H=720 epoch 1). |
| 9 | Epoch | 336 → 1 epoch | **Aynen 1** (tam koşumda) | Kullanıcı kararı (konu.md Şef kayıtları). Duman (≤1 epoch, yalnız H=96) ve ön-uçuş (≤2 epoch) koşumları `--epoch-ust` ile kısaltıldı. Bunlar `kisaltilmis=1` ve ayrı varyant adlarıyla (`baseline-duman`, `baseline-onucus`) işaretli; tam koşum olarak **sunulmaz**. |

Bilinen resmî tuhaflık (değiştirilmedi): `run.py`'nin `setting` biçim dizgesinde alan kayması var (ör.
`..._ft96_sl48_ll96_pl128_...`). Sarmalayıcı aynı dizgeyi üretiyor. Bu yalnız klasör adını etkiliyor.

## 4. Metrik köprüsü: `scripts/rakip_kos.py`

```bash
# tam koşum (kuyruk satırı; cwd=WS, {RUN_DIR} kosum.py tarafından dolduruluyor)
env/.venv/Scripts/python.exe scripts/rakip_kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0 --kapi-test-metrics
# duman / ön-uçuş
... --ufuklar 96 --epoch-ust 1 --num-workers 0          # duman
... --epoch-ust 2 --num-workers 0                        # ön-uçuş (4 ufuk)
```

Tek koşum 4 ufkun hepsini sırayla eğitiyor ve şu çıktıları üretiyor:
- `{RUN_DIR}/metrics.json`:
  - `val_mse_avg`, `val_mae_avg`, `val_mse_h{96,192,336,720}`, `val_mae_h*`
  - `egitim_sure_sn`, `cihaz`, `param_sayisi`
  - ufuk başına: `sure_sn_h*`, `egitim_sure_sn_h*`, `epoch_ayar_h*`, `epoch_resmi_h*`, `epoch_kosulan_h*`, `val_resmi_kayip_min_h*`, `vram_tepe_mb_h*`
  - `alt_grup`: ufuk × değişken val MSE
- `{RUN_DIR}/test_metrics.json`: `test_mse_avg`, `test_mae_avg`, `test_mse_h*`, `test_mae_h*`. Değerler resmî `test()`'in
  yazdığı `metrics.npy` dosyasından okunuyor. Kaydedilen pred/true'dan yeniden hesaplanan MSE ile tutarlılık
  doğrulanıyor (1e-6).
- **`--kapi-test-metrics`** (yalnız aşama 3 `baseline` satırları): `test_*` anahtarları `metrics.json`'a da yazılıyor.
  Nedeni: `replikasyon_kapisi.py` metriği `.done['metrics']` alanından okuyor ve `kosum.py` bu alanı yalnız
  `metrics.json`'dan dolduruyor. Anahtar sırası val→test olduğu için `BILDIRIM.log`/`izle` özetinde (ilk 4–6 anahtar)
  test görünmüyor. Duman ve ön-uçuş koşumlarında `metrics.json` test içermiyor.
- `{RUN_DIR}/preds/val_h{H}.npz` ve `{RUN_DIR}/preds/test_kilitli/test_h{H}.npz`:
  - `pred` (N,H,8) float32
  - `mse_pencere_degisken`, `mae_pencere_degisken` (N,8)
  - `mse_pencere_adim` (N,H)
  - `hedef_ilk_satir` (N,): ilk hedefin CSV satır indeksi. Pencere i, adım h → satır `hedef_ilk_satir[i]+h`. Bu, K2'nin çeyrek atamasıyla uyumlu.

  Gerçek değerler seed'den bağımsız olduğu için bir kez `data/hedefler_sl96/<split>_h<H>_true.npy` olarak yazılıyor.
  Sonraki her koşumda eşitlik doğrulanıyor; veri hattı değişirse koşum duruyor. Boyut yaklaşık 60 MB/koşum
  (4 ufuk), hedefler 53 MB (tek sefer).
- `{RUN_DIR}/progress.json`: `{"adim","ufuk","epoch","loss","val_loss","val_mse_h<H>","gecmis":[[H,ep,train,val],...]}`.
  Resmî log satırlarından ayrıştırılıyor.
- Cihaz: CUDA yoksa çıkış kodu 97 (resmî `use_gpu = cuda_available and ...` sessiz düşüşü devre dışı).

**Önemli (şef için):** `kuyruk.py`/`kosum.py` **venv yorumlayıcısıyla** başlatılmalı:
`env/.venv/Scripts/python.exe SK/kuyruk.py ...`. `kosum.py --gpu-zorunlu` torch'u kendi `sys.executable`'ında
import ediyor. Sistem Python'unda torch 2.2.2+cpu var ve orada rc=97 alınır.

## 5. Duman ve ön-uçuş (foreground, damgalı; seed başına yalnız val)

| Koşum | Ayar | val_mse_h96 | h192 | h336 | h720 | val_mse_avg | Süre |
|---|---|---|---|---|---|---|---|
| `20261014-010943_baseline-duman_s2023` | H=96, 1 epoch | 0.136242 | – | – | – | 0.136242 | 58 s |
| `20261014-011047_baseline-duman_s2024` | H=96, 1 epoch | 0.135278 | – | – | – | 0.135278 | 80 s |
| `20261014-011230_baseline-onucus_s2023` | 4 ufuk, ≤2 epoch (336: 1 = resmî) | 0.126466 | 0.216265 | 0.400437 | 1.037989 | 0.445289 | 247 s |

Referans (yalnız val, `data/naive_val.csv`): Repeat-last 0.1282 / 0.2263 / 0.3933 / 1.1443, ort. 0.4730.
Seed bağlama (R5): aynı ayarda 2023 → 0.13624181, 2024 → 0.13527797. Seed'ler farklı sonuç veriyor. Aynı seed'in
tekrarı (nw=10 ve nw=0) ise bit-bit aynı. Bu GPU'da koşum deterministik, yani seed farkı gerçekten tohumdan geliyor.

## 6. Süre ve donanım (R6)

Ön-uçuştan ölçülen epoch süresi, resmî `vali(val)+vali(test)` dahil: H=96 ~15–22 s, H=192 ~15 s, H=336 ~14 s,
H=720 ~9 s. Ufuk başına değerlendirme (bizim val + resmî test + kayıt) ~20–40 s.
Torch tepe VRAM (`max_memory_allocated`): 27 / 28 / 29 / 31 MB (H=96/192/336/720). CUDA bağlamı dahil süreç belleği
ölçülmedi, tipik olarak birkaç yüz MB. Parametre sayısı: 224,224 / 236,608 / 255,184 / 304,720.
**Tam koşum tahmini:** seed başına ≤ ~10 dk (10 epoch'un tamamı koşarsa; patience 3 ile erken durma olursa daha kısa).
3 seed ≈ 25–30 dk. Kuyruğun tamamı (duman + 2 ön-uçuş + 3 tam) ≈ 35–40 dk. GPU sınıfı: tek RTX 3060 Laptop, seri (`--paralel 1`).
Toplam K3 GPU süresi (bu ajan, foreground): ~13 dk (hata ayıklama: 5.1 + 0.9 dk; damgalı: 1.0 + 1.3 + 4.1 dk).

## 7. Kuyruk ve kapı

- Kuyruk: `kuyruk/k3-baseline.jsonl`, 6 satır:
  - aşama 1: `baseline-duman` s2023
  - aşama 2: `baseline-onucus` s2023, s2024
  - aşama 3: `baseline` s2023, s2024, s2025, `--kapi-test-metrics`
  - hepsi `--gpu-zorunlu`, `--kural`; zaman sınırı 10/15/40 dk.
- Beklenti (ön-kayıt): `kuyruk/k3-baseline.beklenti.json` (metrik `val_mse_avg`, yön min; test kuralı yok).
- Replikasyon kapısı (literatur-cipa §3.1, konu.md Şef kayıtları):
  `python SK/replikasyon_kapisi.py --kok WS --metrik test_mse_avg --yon min --lit 0.358 --lit-ust 0.360 --tolerans 0.008 --kaynak "<literatur-cipa §3.1>"`
  Kapı yalnız `varyant=baseline`, rc=0 ve `test_mse_avg` içeren koşumları sayıyor. Duman ve ön-uçuş koşumları farklı
  varyant adlı ve test içermediği için kapıya karışmıyor.

## 8. FAZ II P2 — ETTh1 / ETTh2 (rakip-kurucu, 2026-10-14 17:05–)

Rakip aynı: `rakip/iTransformer` @ `c2426e6` (`git status --porcelain` boş; **dokunulmadı**, `YAMALAR.md` yok). Ortam aynı
(`env/.venv`, yeni paket yok). Şef kararı (orkestrasyon FAZ II): ETT'de seçim yapılmaz, yalnız resmî split; rakip 3 seed +
Exchange'de seçilen `lastnorm` AYNEN; test sonda bir kez.

### 8.1 Kod (köprü) değişikliği
- `scripts/rakip_kos.py`: yalnız `VERILER["etth1"|"etth2"]["sha256"]` dolduruldu (`data/data_manifest_ett.json` ile aynı:
  ETTh1 `f18de3ad…e066`, ETTh2 `a3dc2c59…218b`; koşum uyuşmazlıkta durur) + açıklama yorumları. `--veri` yolu FAZ II P1'de
  yazılmıştı; ETT için ilk kez koşuldu. Exchange yolu değişmedi: `20261014-170816_baseline-p2-regresyon_s2023`
  (h96, 1 epoch) val npz'nin 5 dizisi `20261014-015848_baseline-duman_s2023` ile `np.array_equal`, `metrics.json` anahtar
  listesi ve süre-dışı tüm değerleri aynı.
- Hiperparametreler elle kopyalanmıyor: `resmi_sh_argumanlari()` `scripts/multivariate_forecasting/ETT/iTransformer_ETTh{1,2}.sh`
  dosyalarını `shlex` ile ayrıştırıyor, `run.py` varsayılanları AST ile okunuyor. Etkin (ayrıştırma çıktısı, CPU):
  ETTh1 d_model=d_ff 256/256/512/512 (h96/192/336/720), ETTh2 128 (4 ufuk); ikisinde de e_layers 2, n_heads 8, enc_in 7,
  train_epochs 10 (**h336 istisnası yok**), lr 1e-4, batch 32, patience 3, lradj type1, dropout 0.1, use_norm 1, timeF/h.
  Parametre (h96…h720): ETTh1 841 568 / 866 240 / 3 379 024 / 3 576 016; ETTh2 224 224 / 236 608 / 255 184 / 304 720.
- Veri sınıfı: resmî `data_dict['ETTh1'|'ETTh2'] = Dataset_ETT_hour` (`--kat`'sız yolda yamasız). Sınırlar 12/4/4 ay
  (train [0,8640), val hedef [8640,11520), test hedef [11520,14400)); CSV'nin 14400–17419 kuyruğu resmî kodda kullanılmaz.
- `scripts/kat_dogrula.py`: ETT için K3 naif Repeat değerleri `data/naive_val_<veri>.csv` (K3 satırları, 6 hane → tol 5e-7)
  ile karşılaştırılıyor (eklendi). Exchange varsayılan çıktıları (`kat-dogrula.log`, `naif-kat.csv`) yeniden üretildi ve
  önceki dosyalarla byte-aynı.

### 8.2 CPU sağlaması (yalnız resmî split = K3; ETT'de kat seçimi yapılmaz)
`env/.venv/Scripts/python.exe scripts/kat_dogrula.py --veri etth1 --katlar 3` (ve `etth2`) → `sonuclar/kat-dogrula_etth{1,2}_k3.log`,
`sonuclar/naif-kat_etth{1,2}_k3.csv`, ikisi de **GEÇTİ** (rc=0): 4 ufukta train/val `data_x`, `data_stamp`, `len`, scaler
resmî `Dataset_ETT_hour` ile bit-bit eşit; pencere sayıları 8449/8353/8209/7825 (train) ve 2785/2689/2545/2161 (val);
sızıntı yok; naif Repeat val ort. ETTh1 2.075811, ETTh2 0.479853 = veri-hazırlayıcı CSV'si.

### 8.3 Protokol farkları (ETT'ye özgü; §3 tablosundaki 1–8 aynen geçerli)
| # | Fark | Resmî | Bizde | Gerekçe / etki |
|---|---|---|---|---|
| E1 | GPU seçimi | betik `export CUDA_VISIBLE_DEVICES=1` | `cuda:0` (satır `run.py` bloğu değil, ayrıştırıcı atlar) | Tek GPU; hesaba etkisi yok |
| E2 | Veri yolu | `./dataset/ETT-small/` | `data/etth{1,2}/` (sha256 sabit; HF thuml@2b66e59e = ETDataset@1d16c8f4 bayt bayt) | Aynı dosya |
| E3 | d_model ETTh2 | makale metni {256, 512} (Ek A.2) | betik değeri 128 | Resmî betik esas; uyuşmazlık literatür çıpasında (§1.4) beyanlı |
| E4 | Test değerlendirme süresi | `test()` her 20 pencerede bir PDF çiziyor (2785 pencere → ~140 PDF/ufuk) | aynen (çıktı `test_kilitli/`) | Yalnız süre; ufuk başına ~70–110 s |
| E5 | Epoch | 10, patience 3 | tam koşumda aynen; duman ≤1 / ön-uçuş ≤2 (`--epoch-ust`, `kisaltilmis=1`, ayrı varyant adı) | Kısaltılmış koşumlar tam koşum diye sunulmaz |

Kozmetik not: `rakip_kos.py` başlık satırı `num_workers=resmi` yazıyor, `--num-workers 0` verildiğinde de (`0 or 'resmi'`);
uygulanan değer 0'dır (`args.num_workers` atanıyor). Exchange bit-bitliğini korumak için düzeltilmedi.
