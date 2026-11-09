# revin-afin — H02 (RevIN-afin, kontrol kolu)

- **Kart:** `hipotezler/H02.md` (A7; Z1/Z2 kısmen). H01 (SAN) için kontrol satırı.
- **Ebeveyn:** `baseline` (resmî iTransformer `c2426e6`, Exchange `.sh` birebir, h336 = 1 epoch). Koşumlar:
  `20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`.
- **Tek değişiklik:** resmî afinsiz NST normalizasyonu (`use_norm=1`) yerine RevIN geliyor. μ/σ formülü aynı;
  üstüne değişken başına öğrenilebilir afin γ (başlangıç 1) ve β (başlangıç 0) ekleniyor, +16 parametre. Başka hiçbir şey değişmez.
- **Dosyalar:**
  - `model_revin.py`: `RevINiTransformer`. Önce resmî `model.iTransformer.Model(args)` kurulur (`use_norm=0`, aynı RNG
    tüketimi), sonra γ/β kurulur (RNG çekmez). İleri geçiş sırası: normalize (resmî μ/σ, sonra `γ·x+β`) → omurga →
    de-normalize (`(y−β)/(γ+1e-10)·σ+μ`).
  - `kos.py`: `rakip_kos.resmi_sh_argumanlari` sarılır ve her ufukta `--use_norm 0` verilir.
    `experiments.exp_long_term_forecasting.Exp_Long_Term_Forecast` modül özniteliği yerine `ExpRevIN` konur. Bu sınıfta
    yalnız `_build_model` değişir, `train` sonrasında γ/β kaydedilir. `__init__` ise `Exp_Basic.__init__`'e bağlanır,
    çünkü resmî `super(Exp_Long_Term_Forecast, self)` yamalı adla özyinelemeye girer. Eğitim döngüsü, optimizer,
    kayıp ve erken durdurma resmî kalır. `metrics.json`'a eklenenler: `varyant`, `kart`, `ebeveyn`,
    `revin_gamma_h{H}_d{j}`, `revin_beta_h{H}_d{j}`, `revin_gamma_min_h{H}`, `revin_gamma_min`.
    `--kapi-test-metrics` reddedilir.
  - `dogrula_esdeger.py` (+ `dogrula_esdeger.log`): CPU tek-değişken sağlaması. Aynı seed'de omurga ağırlıkları
    birebir aynı, parametre farkı 16. Başlangıç çıkışında resmî `use_norm=1` modeline göre maksimum fark eval'de ve
    train'de (dropout) 4 ufkun hepsinde **0.0** (eşik < 1e-6). Sonuç: GEÇTİ.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/revin-afin/dogrula_esdeger.py                         # CPU sağlama, rc=0 olmalı
$PY varyantlar/revin-afin/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 96 --epoch-ust 1 --num-workers 0   # duman
$PY varyantlar/revin-afin/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0               # ön-uçuş
$PY varyantlar/revin-afin/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                              # tam (kuyruk k4t1-h02)
```
