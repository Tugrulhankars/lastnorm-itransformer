# baseline-ep336eq — H03 (PROTOKOL KARTI)

- **Kart:** `hipotezler/H03.md` (A11/A12, Z5). Zincire girmez, ebeveyn olmaz. Leaderboard'da `protokol-referansı`.
- **Ebeveyn:** `baseline` (resmî iTransformer `c2426e6`, Exchange `.sh` birebir). Koşumlar:
  `20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`.
- **Tek değişiklik:** yalnız ufuk 336'da `--train_epochs` 1 → 10. Diğer üç ufuk ve öteki tüm ayarlar resmî
  betikle aynı (patience 3, lradj type1, lr 1e-4, batch 32).
- **Uygulama:** `kos.py`, `scripts/rakip_kos.py`'yi import eder. `rakip_kos.resmi_sh_argumanlari` fonksiyonunu
  sarar: resmî bloklar okunur, yalnız `bloklar[336]` içinde `argv_degistir(..., "--train_epochs", "10")` uygulanır.
  Sonra `rakip_kos.main()` çağrılır. Rakip koduna ve `rakip_kos.py`'ye dokunulmaz. `metrics.json`'a şu anahtarlar
  eklenir: `varyant`, `kart`, `ebeveyn`, `protokol_karti: 1`, `epoch_resmi_betik_h336: 1`.
  `--kapi-test-metrics` reddedilir.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
# duman: ebeveynle bit-bit aynı olmalı (epoch_ust 1 → min(10,1)=1). Beklenen val_mse_h336 = 0.40043676
$PY varyantlar/baseline-ep336eq/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 336 --epoch-ust 1 --num-workers 0
# ön-uçuş: h96/h192/h720 değerleri baseline-onucus_s2023 ile bit-bit aynı olmalı
$PY varyantlar/baseline-ep336eq/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0
# tam (kuyruk k4t1-h03, aşama 3)
$PY varyantlar/baseline-ep336eq/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0
```
