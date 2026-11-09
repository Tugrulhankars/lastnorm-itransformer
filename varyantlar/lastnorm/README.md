# lastnorm — H04 (son-değere çapalı normalizasyon)

- **Kart:** `hipotezler/H04.md` (A7; Z2, ikincil Z1).
- **Ebeveyn:** `baseline` (resmî iTransformer `c2426e6`, Exchange `.sh` birebir, h336 = 1 epoch). Koşumlar:
  `20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`.
- **Ebeveynden tek fark:** NST normalizasyonunun merkezi pencere ortalaması μ_lb yerine son girdi değeri
  `x_enc[:, -1:, :]` (detach) oluyor. σ_lb resmî formülle aynı, de-normalizasyon `σ_lb·f(·) + x_son`. Parametre,
  omurga, eğitim, veri, seed ve epoch'lar aynı.
- **Dosyalar:**
  - `model_lastnorm.py`: `LastNormiTransformer`. Önce resmî `Model(args)` kurulur (`use_norm=0`, aynı RNG tüketimi,
    yeni parametre yok). `istatistik()` μ/σ/x_son'u döndürür, `forward` kartın uygulama notundaki satırların birebir aynısı.
  - `kos.py`: her ufukta `--use_norm 0` argv yaması ve `ExpLastNorm` (yalnız `_build_model`; `__init__` →
    `Exp_Basic.__init__`). `metrics.json`'a `varyant`, `kart`, `ebeveyn` eklenir. `--kapi-test-metrics` reddedilir,
    çıkışta `test_` anahtarı denetlenir.
  - `dogrula.py` → `dogrula.log` (CPU, ~10 s): (i) ilk ağırlıklar resmî `use_norm=1` modelle özdeş (fark 0),
    parametre 224 224 / 236 608 / 255 184 / 304 720; (ii) σ_lb farkı 0; (iii) projektör sıfırken çıkış = Repeat
    (fark 0); (iv) bilgi: x_son = `batch_x[:, -1]` = `batch_y[:, label_len-1]`, ilk hedef adımı değil. **GEÇTİ.**
- **Foreground (s2023, damgalı):** duman `20261014-050354_lastnorm-duman_s2023` val_mse_h96 0.127633 (eşik ≤ 0.1923);
  ön-uçuş `20261014-050501_lastnorm-onucus_s2023` val_mse_avg 0.445366 ([0.40, 0.49] içinde), 164 s. Val tahminleri
  `baseline-onucus_s2023` ile bit-bit aynı değil (plasebo değil).
- **Kuyruk:** `kuyruk/k4t2-h04.jsonl` (5 satır: a1 duman, a2 ön-uçuş, a3 × {2023, 2024, 2025}), beklenti
  `kuyruk/k4t2-h04.beklenti.json`. Kuyruğu şef başlatır.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/lastnorm/dogrula.py                                                                          # CPU sağlama, rc=0
$PY varyantlar/lastnorm/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 96 --epoch-ust 1 --num-workers 0  # duman
$PY varyantlar/lastnorm/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0              # ön-uçuş
$PY varyantlar/lastnorm/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                             # tam (kuyruk)
```
