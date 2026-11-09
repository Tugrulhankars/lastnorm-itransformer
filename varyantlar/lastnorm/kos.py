#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/lastnorm/kos.py -- H04: NST normalizasyonunun merkezi mu_lb -> son girdi degeri x_son (sigma aynen).

Ebeveyn: `baseline` (scripts/rakip_kos.py; resmi iTransformer c2426e6 Exchange .sh birebir, h336 = 1 epoch).
Tek degisiklik: her ufukta `--use_norm 0` (resmi NST blogu kapanir) + model LastNormiTransformer (model_lastnorm.py):
ayni sigma_lb formulu, merkez x_enc[:, -1:, :]. Parametre, omurga, egitim dongusu (resmi Exp_Long_Term_Forecast),
veri, seed ve h336 epoch'u (1) ebeveynle ayni. Alt sinifta yalniz _build_model degisir; __init__ ->
Exp_Basic.__init__ (H02 RecursionError duzeltmesi, gunluk ariza tablosu).

Rakip koduna ve scripts/rakip_kos.py'ye DOKUNULMAZ. `--kapi-test-metrics` REDDEDILIR (K4 test kilidi).
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/lastnorm/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
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

VARYANT = "lastnorm"
KART = "H04"
EBEVEYN = "baseline"

_orijinal_sh = rakip_kos.resmi_sh_argumanlari


def yamali_sh_argumanlari():
    bloklar = _orijinal_sh()
    return {h: rakip_kos.argv_degistir(a, "--use_norm", "0") for h, a in bloklar.items()}


def exp_sinifi_kur():
    import experiments.exp_long_term_forecasting as elf
    from experiments.exp_basic import Exp_Basic
    from model_lastnorm import LastNormiTransformer

    Resmi = elf.Exp_Long_Term_Forecast

    class ExpLastNorm(Resmi):
        def __init__(self, args):
            # resmi __init__ = super(Exp_Long_Term_Forecast, self).__init__(args); modul adi yamalandigi icin
            # dogrudan Exp_Basic (esdeger, ozyineleme yok)
            Exp_Basic.__init__(self, args)

        def _build_model(self):
            return LastNormiTransformer(self.args).float()

    elf.Exp_Long_Term_Forecast = ExpLastNorm  # rakip_kos.main() bu adi yerel import ediyor
    return ExpLastNorm


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H04] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    bilinen, _ = ap.parse_known_args()

    rakip_kos.resmi_sh_argumanlari = yamali_sh_argumanlari
    exp_sinifi_kur()
    print(f"[H04] varyant={VARYANT}: --use_norm 0 + LastNorm (merkez x_son, olcek sigma_lb) her ufukta", flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H04] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN})
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
