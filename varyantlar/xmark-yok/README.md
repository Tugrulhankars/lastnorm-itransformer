# xmark-yok — H07 (takvim token'ları kaldırıldı; YEDEK kart, aktif)

- **Kart:** `hipotezler/H07.md` (A12 / A6). Aktivasyon: H05 ön-koşul kapısı **BAŞARISIZ** oldu
  (`varyantlar/san-mu/onkosul.log`), bu yüzden tur 2'nin üçüncü kartı.
- **Ebeveyn:** `baseline` (resmî iTransformer `c2426e6`, Exchange `.sh` birebir, h336 = 1 epoch). Koşumlar:
  `20261014-020531_baseline_s2023`, `20261014-021024_baseline_s2024`, `20261014-021448_baseline_s2025`.
- **Ebeveynden tek fark:** kodlayıcıya `x_mark_enc` yerine `None` verilir. 4 takvim token'ı düşer, 12 token 8'e iner.
  `use_norm=1` ve geri kalan her şey aynı, argv yaması yok.
- **Dosyalar:**
  - `model_xmark.py`: `XmarkYokiTransformer`. Resmî `Model(args)` sarılır, `forward` = `backbone(x_enc, None, x_dec, x_mark_dec)`.
  - `kos.py`: `ExpXmarkYok` (yalnız `_build_model`; `__init__` → `Exp_Basic.__init__`). `metrics.json`'a
    `varyant`, `kart`, `ebeveyn`, `token_sayisi: 8` eklenir. `--kapi-test-metrics` reddedilir.
  - `dogrula.py` → `dogrula.log` (CPU): (i) ağırlıklar resmî modelle özdeş, parametre eşit; (ii) çıkış [B, H, 8],
    kodlayıcıda 8 token (resmîde 12); (iii) x_mark'lı ve x_mark'sız çıkışlar farklı (maks. fark 0.085–0.198). **GEÇTİ.**
- **Foreground (s2023, damgalı):** duman `20261014-051258_xmark-yok-duman_s2023` val_mse_h96 0.135984 (≤ 0.1923);
  ön-uçuş `20261014-051352_xmark-yok-onucus_s2023` val_mse_avg 0.447193 ([0.40, 0.49] içinde), 156 s. Ön-uçuş
  `baseline-onucus_s2023` ile bit-bit aynı değil.
- **Kuyruk:** `kuyruk/k4t2-h07.jsonl` (5 satır), beklenti `kuyruk/k4t2-h07.beklenti.json`.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/xmark-yok/dogrula.py
$PY varyantlar/xmark-yok/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 96 --epoch-ust 1 --num-workers 0  # duman
$PY varyantlar/xmark-yok/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0              # ön-uçuş
$PY varyantlar/xmark-yok/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                             # tam (kuyruk)
```
