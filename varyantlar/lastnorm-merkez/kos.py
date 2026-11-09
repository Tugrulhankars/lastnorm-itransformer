#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/lastnorm-merkez/kos.py -- H10: lastnorm'da sigma_lb -> 1 (yalniz merkezleme x_son; H04 Ablasyon (1)).

Ebeveyn: `lastnorm` (H04; varyantlar/lastnorm/kos.py). Kalip birebir: her ufukta `--use_norm 0` argv yamasi,
Exp alt sinifinda yalniz _build_model degisir (-> MerkezLastNormiTransformer), __init__ -> Exp_Basic.__init__
(RecursionError duzeltmesi). Omurga, parametreler, egitim dongusu (resmi), veri, seed ve h336 epoch'u (1) ebeveynle ayni.

Rakip koduna, scripts/rakip_kos.py'ye ve varyantlar/lastnorm/ dosyalarina DOKUNULMAZ (LastNormiTransformer
sys.path uzerinden import edilir; sys.dont_write_bytecode -> o klasore __pycache__ yazilmaz).
`--kapi-test-metrics` REDDEDILIR (K4 test kilidi); cikista test_ anahtari denetlenir.
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/lastnorm-merkez/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
sys.path.insert(0, str(WS / "varyantlar" / "lastnorm"))  # model_lastnorm (ebeveyn modeli)
sys.path.insert(0, str(HERE))

VARYANT = "lastnorm-merkez"
KART = "H10"
EBEVEYN = "lastnorm"

_orijinal_sh = rakip_kos.resmi_sh_argumanlari


def yamali_sh_argumanlari():
    bloklar = _orijinal_sh()
    return {h: rakip_kos.argv_degistir(a, "--use_norm", "0") for h, a in bloklar.items()}


def exp_sinifi_kur():
    import experiments.exp_long_term_forecasting as elf
    from experiments.exp_basic import Exp_Basic
    from model_merkez import MerkezLastNormiTransformer

    Resmi = elf.Exp_Long_Term_Forecast

    class ExpMerkez(Resmi):
        def __init__(self, args):
            # resmi __init__ = super(Exp_Long_Term_Forecast, self).__init__(args); modul adi yamalandigi icin
            # dogrudan Exp_Basic (esdeger, ozyineleme yok) -- lastnorm/kos.py ile ayni
            Exp_Basic.__init__(self, args)

        def _build_model(self):
            return MerkezLastNormiTransformer(self.args).float()

    elf.Exp_Long_Term_Forecast = ExpMerkez  # rakip_kos.main() bu adi yerel import ediyor
    return ExpMerkez


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H10] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    bilinen, _ = ap.parse_known_args()

    rakip_kos.resmi_sh_argumanlari = yamali_sh_argumanlari
    exp_sinifi_kur()
    print(f"[H10] varyant={VARYANT}: --use_norm 0 + LastNorm merkez x_son, sigma = 1 (olcekleme yok) her ufukta",
          flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H10] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN, "ablasyon_of": "H04-1"})
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
