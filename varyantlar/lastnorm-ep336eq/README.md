# lastnorm-ep336eq — H08 (PROTOKOL KARTI)

- **Kart:** `hipotezler/H08.md` (A11/A12; Z5 × Z2). Zincire girmez, ebeveyn olmaz, durma sayacına girmez.
  Leaderboard'da `protokol-referansı`.
- **Ebeveyn:** `lastnorm` (H04). Koşumlar: `20261014-052508_lastnorm_s2023`, `20261014-053601_lastnorm_s2024`,
  `20261014-054213_lastnorm_s2025`.
- **Ebeveynden tek fark:** yalnız ufuk 336'da `--train_epochs` 1 → 10 (H03 yamasının lastnorm üzerinde tekrarı); h96/h192/h720
  `lastnorm` ile bit-bit aynı.
- **Uygulama:** `kos.py`, `varyantlar/lastnorm/kos.py`'yi `importlib.util` ile dosya yolundan `lastnorm_kos` adıyla yükler.
  Onun `yamali_sh_argumanlari` (dört blokta `--use_norm 0`) ve `exp_sinifi_kur` (`ExpLastNorm` →
  `LastNormiTransformer`, `varyantlar/lastnorm/model_lastnorm.py`) fonksiyonlarını değiştirmeden kullanır. Bunun üstüne
  yalnız `bloklar[336]` için `argv_degistir(..., "--train_epochs", "10")` uygular. Yama öncesi sağlamalar
  `baseline-ep336eq/kos.py` ile aynıdır: h336'da `--train_epochs 1` olmalı, öteki bloklarda `--train_epochs` olmamalı.
  `sys.dont_write_bytecode` açık olduğu için `varyantlar/lastnorm/` altına `__pycache__` yazılmaz.
  `metrics.json`'a şu alanlar eklenir: `varyant`, `kart: H08`, `ebeveyn: lastnorm`, `protokol_karti: 1`,
  `epoch_resmi_betik_h336: 1`. `--kapi-test-metrics` reddedilir ve `test_` anahtarı denetlenir.
- **CPU sağlaması (argv farkı):** yalnız h336 bloğunda `--train_epochs` '1' → '10'. Dört blokta `--use_norm 0` eklendi.
  Başka fark yok.
- **Foreground (s2023, damgalı):**
  - Duman `20261014-070520_lastnorm-ep336eq-duman_s2023`: val_mse_h336 0.3949749767780304, `lastnorm_s2023` ile bit-bit
    aynı (fark 0.0). 32 s sürdü.
  - Ön-uçuş `20261014-070604_lastnorm-ep336eq-onucus_s2023`: h96/h192/h720 val tahminleri `050501_lastnorm-onucus_s2023`
    ile `array_equal`, alt_grup farkı 0. h336 0.370900 (2 epoch; lastnorm-onucus 0.394975'e göre −6.09%). Yama devrede.
    163 s sürdü.
- **Kuyruk:** `kuyruk/k4t3-h08.jsonl` (5 satır: a1 duman, a2 ön-uçuş, a3 × {2023, 2024, 2025}), beklenti
  `kuyruk/k4t3-h08.beklenti.json`. Kuyruğu şef başlatır.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/lastnorm-ep336eq/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 336 --epoch-ust 1 --num-workers 0  # duman
$PY varyantlar/lastnorm-ep336eq/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0               # ön-uçuş
$PY varyantlar/lastnorm-ep336eq/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                              # tam (kuyruk)
```
