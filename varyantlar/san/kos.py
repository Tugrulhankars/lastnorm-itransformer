#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/san/kos.py -- H01: iTransformer NST ornek normalizasyonu -> SAN (period_len 6, station_lr 1e-3,
adaptive, 5 epoch istatistik on-egitimi; SAN@7e1ca66).

Ebeveyn: `baseline` (scripts/rakip_kos.py; resmi iTransformer c2426e6 Exchange .sh birebir, h336 = 1 epoch).
Tek degisiklik: her ufukta `--use_norm 0` + SANiTransformer (model_san.py) + SAN iki asamali egitim (exp_san.py).
Omurga, veri, split, seed, omurga lr/epoch/patience/lradj aynen ebeveyn.

Rakip koduna ve scripts/rakip_kos.py'ye DOKUNULMAZ: `rakip_kos.resmi_sh_argumanlari` sarilir ve
`experiments.exp_long_term_forecasting.Exp_Long_Term_Forecast` modul ozniteligi ExpSAN ile degistirilir
(rakip_kos.main() bu adi yerel import ediyor). `--kapi-test-metrics` REDDEDILIR (K4 test kilidi).

Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/san/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
sys.path.insert(0, str(HERE))

VARYANT = "san"
KART = "H01"
EBEVEYN = "baseline"

_orijinal_sh = rakip_kos.resmi_sh_argumanlari


def yamali_sh_argumanlari():
    bloklar = _orijinal_sh()
    return {h: rakip_kos.argv_degistir(a, "--use_norm", "0") for h, a in bloklar.items()}


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H01] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    bilinen, _ = ap.parse_known_args()

    import experiments.exp_long_term_forecasting as elf
    import exp_san
    rakip_kos.resmi_sh_argumanlari = yamali_sh_argumanlari
    elf.Exp_Long_Term_Forecast = exp_san.ExpSAN
    print(f"[H01] varyant={VARYANT}: --use_norm 0 + SAN (period_len={exp_san.PERIOD_LEN}, "
          f"station_lr={exp_san.STATION_LR}, {exp_san.STATION_TYPE}, on-egitim {exp_san.STATION_PRETRAIN_EPOCH} ep)",
          flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H01] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN,
              "san_period_len": exp_san.PERIOD_LEN, "san_station_lr": exp_san.STATION_LR,
              "san_onegitim_epoch_ayar": exp_san.STATION_PRETRAIN_EPOCH})
    for H, d in sorted(exp_san.KAYIT.items()):
        m.update(d)
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
