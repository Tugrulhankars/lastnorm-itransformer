# normsuz — H06 (normalizasyonsuz iTransformer; ablasyon satırı)

- **Kart:** `hipotezler/H06.md` (A7 ablasyonu / A12). Kazanç kartı değil; ÖLDÜR beklenen sonuç.
- **Ebeveyn:** `baseline` (resmî iTransformer `c2426e6`, Exchange `.sh` birebir, h336 = 1 epoch). Koşumlar:
  `20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`.
- **Ebeveynden tek fark:** resmî argv'de her ufukta `--use_norm 1 → 0`. Kod değişikliği yok, Exp sınıfı resmî.
- **Dosyalar:** `kos.py` (H03 `baseline-ep336eq/kos.py` kalıbı). `rakip_kos.resmi_sh_argumanlari` sarılıp her blokta
  `--use_norm 0` verilir, sonra `rakip_kos.main()` çağrılır. Resmî betik `--use_norm` vermiyor, run.py varsayılanı 1.
  `metrics.json`'a `varyant`, `kart`, `ebeveyn`, `use_norm: 0` eklenir. `--kapi-test-metrics` reddedilir.
- **Canlı kural:** `ust_sinir` val_loss/loss 10.0. Kartın mekanik DUR'u > 10.0. Tur 1'deki 3.0 bu varyantın meşru
  kötü val kaybını (ön-uçuş h336 epoch-1 val kaybı 3.36) öldürürdü.
- **Foreground (s2023, damgalı):** duman `20261014-050826_normsuz-duman_s2023` val_mse_h96 2.183849 (≤ 5.0;
  `baseline-duman` ile |fark| 2.05, plasebo değil; stdout `use_norm=0`). Ön-uçuş `20261014-050923_normsuz-onucus_s2023`
  val_mse_avg 2.317623 ([0.40, 10.0] içinde), 140 s.
- **Kuyruk:** `kuyruk/k4t2-h06.jsonl` (5 satır), beklenti `kuyruk/k4t2-h06.beklenti.json`. Kart gereği kötü sonuçta
  durdurulmaz, n = 3 tamamlanır.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/normsuz/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 96 --epoch-ust 1 --num-workers 0  # duman
$PY varyantlar/normsuz/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0              # ön-uçuş
$PY varyantlar/normsuz/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                             # tam (kuyruk)
```
