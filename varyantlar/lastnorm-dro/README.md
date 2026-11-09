# lastnorm-dro — H09 (zaman-grubu GroupDRO, K = 4, η = 0.01)

- **Kart:** `hipotezler/H09.md` (A5; Z3, ikincil Z4).
- **Ebeveyn:** `lastnorm` (H04). Koşumlar: `20261014-052508_lastnorm_s2023`, `20261014-053601_lastnorm_s2024`,
  `20261014-054213_lastnorm_s2025`.
- **Ebeveynden tek fark:** eğitim kaybı ERM yerine GroupDRO (Sagawa et al. 2020, Alg. 1). Train pencereleri başlangıç
  indeksine göre 4 kronolojik gruba ayrılır. Her adımda `q ← q·exp(0.01·ℓ_g)` uygulanıp normalize edilir ve kayıp
  `Σ q_g ℓ_g` olur. Model, normalizasyon, optimizasyon, val kaybı (resmî MSE), erken durdurma, lr çizelgesi, h336
  epoch'u (1), veri ve seed aynıdır.
- **Dosyalar:**
  - `exp_dro.py`: `IndeksliVeri`, `grup_ata`, `dro_kayip` ve `dro_sinifi_kur(ExpLastNorm)` → `ExpLastNormDRO`. Yalnız
    `train` override edilir. Bu, resmî `train`'in (satır 82–184) birebir kopyasıdır; farklar `# [H09 a–e]` ile
    işaretlidir.
  - `kos.py`: `varyantlar/lastnorm/kos.py`'yi `importlib.util` ile `lastnorm_kos` adıyla yükler. Onun
    `yamali_sh_argumanlari` ve `exp_sinifi_kur` fonksiyonlarını değiştirmeden kullanır, sonra
    `elf.Exp_Long_Term_Forecast = ExpLastNormDRO` yapar. `metrics.json`'a şunlar eklenir: `varyant`, `kart: H09`,
    `ebeveyn`, `dro_eta`, `dro_K`, ufuk başına `dro_q_son_h{H}`, `dro_q_maks_son_h{H}`, `dro_grup_n_h{H}`,
    `dro_grup_mse_son_h{H}`. Epoch başına q izi stdout'a `[H09] DRO epoch …` satırı olarak yazılır.
    `--kapi-test-metrics` reddedilir, `test_` anahtarı denetlenir.
  - `dogrula.py` → `dogrula.log` (CPU, ~9 s), sonuç **GEÇTİ**:
    - (i) N_H 5120 / 5024 / 4880 / 4496; gruplar `np.array_split` ile aynı ve monoton; yalnız train kullanılıyor.
    - (ii) sarmalayıcı farkı 0; ilk 3 batch sırası ve RNG durumu orijinal DataLoader ile aynı.
    - (iii) q uniform iken DRO kaybı = MSELoss (fark ≤ 2.2e-16, float64).
    - (iv) q pozitif ve toplamı 1; boş grubun ağırlığı azalıyor.
    - (v) ilk ağırlıklar ve parametre sayısı lastnorm ile aynı; yalnız `train` override edilmiş.
- **Uygulama notu (e) — RNG:** Resmî döngüdeki `test_loss = self.vali(test_data, test_loader, …)` satırı hesaplanmaz
  ve yerine `nan` yazılır. Ancak torch DataLoader, shuffle'sız olsa bile her iterator açılışında global RNG'den bir
  int64 çeker (`torch/utils/data/dataloader.py:601`). Satır tamamen atlansaydı epoch ≥ 2'de train karıştırma sırası
  ebeveynden ayrışırdı; kart ise "eğitimi değiştirmez, RNG korunur" diyor. Bu yüzden yalnız `iter(test_loader)`
  oluşturulup atılır. num_workers = 0 olduğundan hiçbir test örneği okunmaz ve hiçbir test kaybı hesaplanmaz.
  `dogrula.py` (vi) bunu doğrular: tam tur ile yalnız `iter()` aynı RNG durumunu bırakır, hiç dokunmamak farklı bırakır.
- **Foreground (s2023, damgalı):**
  - Duman `20261014-071837_lastnorm-dro-duman_s2023`: val_mse_h96 0.127575 (eşik ≤ 0.1923; `lastnorm-duman`'a göre
    −0.05%).
  - Ön-uçuş `20261014-071929_lastnorm-dro-onucus_s2023`: val_mse_avg 0.440329 (eşik [0.40, 0.49]); `lastnorm-onucus`'a
    göre −1.13% (h96 −0.17 / h192 +0.61 / h336 +0.97 / h720 −2.38%). egitim_sure_sn 119.5 (eşik ≤ 328.6). Son-epoch
    q_max: 0.299 / 0.387 / 0.348 / 0.349. Val tahminleri bit-bit aynı değil (maks. fark 0.07–0.24), yani plasebo değil.
- **Kuyruk:** `kuyruk/k4t3-h09.jsonl` (5 satır: a1 duman, a2 ön-uçuş, a3 × {2023, 2024, 2025}), beklenti
  `kuyruk/k4t3-h09.beklenti.json`. Kuyruğu şef başlatır.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/lastnorm-dro/dogrula.py                                                                          # CPU sağlama, rc=0
$PY varyantlar/lastnorm-dro/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 96 --epoch-ust 1 --num-workers 0  # duman
$PY varyantlar/lastnorm-dro/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0              # ön-uçuş
$PY varyantlar/lastnorm-dro/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                             # tam (kuyruk)
```
