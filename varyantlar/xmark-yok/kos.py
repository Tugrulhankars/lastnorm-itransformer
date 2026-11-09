#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/xmark-yok/kos.py -- H07 (YEDEK; H05 on-kosul kapisi BASARISIZ oldugu icin aktif): kodlayiciya takvim
token'lari (x_mark_enc) verilmez (12 -> 8 token).

Ebeveyn: `baseline` (scripts/rakip_kos.py; resmi iTransformer c2426e6 Exchange .sh birebir, h336 = 1 epoch).
Tek degisiklik: model XmarkYokiTransformer (model_xmark.py) -> backbone(x_enc, None, ...). `use_norm=1` (NST) dahil
diger her sey ebeveynle ayni; ARGV YAMASI YOK. Egitim dongusu resmi Exp_Long_Term_Forecast; alt sinifta yalniz
_build_model degisir; __init__ -> Exp_Basic.__init__ (H02 RecursionError duzeltmesi).

Rakip koduna ve scripts/rakip_kos.py'ye DOKUNULMAZ. `--kapi-test-metrics` REDDEDILIR (K4 test kilidi).
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/xmark-yok/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
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

VARYANT = "xmark-yok"
KART = "H07"
EBEVEYN = "baseline"
TOKEN_SAYISI = 8


def exp_sinifi_kur():
    import experiments.exp_long_term_forecasting as elf
    from experiments.exp_basic import Exp_Basic
    from model_xmark import XmarkYokiTransformer

    Resmi = elf.Exp_Long_Term_Forecast

    class ExpXmarkYok(Resmi):
        def __init__(self, args):
            # resmi __init__ = super(Exp_Long_Term_Forecast, self).__init__(args); modul adi yamalandigi icin
            # dogrudan Exp_Basic (esdeger, ozyineleme yok)
            Exp_Basic.__init__(self, args)

        def _build_model(self):
            return XmarkYokiTransformer(self.args).float()

    elf.Exp_Long_Term_Forecast = ExpXmarkYok  # rakip_kos.main() bu adi yerel import ediyor
    return ExpXmarkYok


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H07] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    bilinen, _ = ap.parse_known_args()

    exp_sinifi_kur()
    print(f"[H07] varyant={VARYANT}: kodlayiciya x_mark_enc=None (token {TOKEN_SAYISI}); use_norm resmi (1)", flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H07] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN, "token_sayisi": TOKEN_SAYISI})
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
