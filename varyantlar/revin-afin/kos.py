#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/revin-afin/kos.py -- H02: afinsiz NST normalizasyonu -> RevIN (afin, +16 parametre).

Ebeveyn: `baseline` (scripts/rakip_kos.py; resmi iTransformer c2426e6 Exchange .sh birebir, h336 = 1 epoch).
Tek degisiklik: her ufukta `--use_norm 0` (resmi NST kapanir) + model RevINiTransformer (model_revin.py):
ayni mu/sigma formulu, ustune ogrenilebilir degisken basina gamma (1) / beta (0).
Egitim dongusu, optimizer (Adam, tum parametreler), kayip, erken durdurma, lr cizelgesi RESMI
(Exp_Long_Term_Forecast); alt sinifta yalniz _build_model degisir (+ egitim sonrasi gamma/beta kaydi).

Rakip koduna ve scripts/rakip_kos.py'ye DOKUNULMAZ. `--kapi-test-metrics` REDDEDILIR (K4 test kilidi).
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/revin-afin/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
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

VARYANT = "revin-afin"
KART = "H02"
EBEVEYN = "baseline"
GAMMA_BETA = {}  # pred_len -> (gamma list, beta list)  (en iyi checkpoint yuklendikten sonra)

_orijinal_sh = rakip_kos.resmi_sh_argumanlari


def yamali_sh_argumanlari():
    bloklar = _orijinal_sh()
    return {h: rakip_kos.argv_degistir(a, "--use_norm", "0") for h, a in bloklar.items()}


def exp_sinifi_kur():
    import experiments.exp_long_term_forecasting as elf
    from experiments.exp_basic import Exp_Basic
    from model_revin import RevINiTransformer

    Resmi = elf.Exp_Long_Term_Forecast

    class ExpRevIN(Resmi):
        def __init__(self, args):
            # Resmi __init__ yalniz `super(Exp_Long_Term_Forecast, self).__init__(args)` cagirir; modul adi
            # bu alt sinifa yamalandigi icin o satir ozyinelemeye girer -> Exp_Basic.__init__ dogrudan (esdeger).
            Exp_Basic.__init__(self, args)

        def _build_model(self):
            return RevINiTransformer(self.args).float()

        def train(self, setting):
            m = super().train(setting)  # resmi train: erken durdurma + en iyi checkpoint yuklenir
            GAMMA_BETA[int(self.args.pred_len)] = (
                [float(x) for x in self.model.gamma.detach().cpu()],
                [float(x) for x in self.model.beta.detach().cpu()])
            return m

    elf.Exp_Long_Term_Forecast = ExpRevIN  # rakip_kos.main() bu adi yerel import ediyor
    return ExpRevIN


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H02] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    bilinen, _ = ap.parse_known_args()

    rakip_kos.resmi_sh_argumanlari = yamali_sh_argumanlari
    exp_sinifi_kur()
    print(f"[H02] varyant={VARYANT}: --use_norm 0 + RevIN (afin gamma/beta) her ufukta", flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H02] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN})
    gmin = []
    for H, (g, b) in sorted(GAMMA_BETA.items()):
        for j, (gj, bj) in enumerate(zip(g, b)):
            m[f"revin_gamma_h{H}_d{j}"] = gj
            m[f"revin_beta_h{H}_d{j}"] = bj
        m[f"revin_gamma_min_h{H}"] = min(g)
        gmin.append(min(g))
    if gmin:
        m["revin_gamma_min"] = min(gmin)  # kart: herhangi bir degiskende <= 0.05 -> INCELE
    rakip_kos.atomik_json(out, m)
    if gmin and min(gmin) <= 0.05:
        print(f"[H02] UYARI: ogrenilen gamma <= 0.05 (min {min(gmin):.4f}) -> kart: INCELE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
