#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/lastnorm-dro/kos.py -- H09: lastnorm + zaman-grubu GroupDRO egitim kaybi (K = 4, eta = 0.01).

Ebeveyn: `lastnorm` (H04). Kalip: `varyantlar/lastnorm/kos.py` importlib.util ile DOSYA YOLUNDAN `lastnorm_kos` adiyla
yuklenir; onun `yamali_sh_argumanlari` (dort blokta --use_norm 0) ve `exp_sinifi_kur` (ExpLastNorm) BIREBIR kullanilir.
Ustune exp_dro.ExpLastNormDRO(ExpLastNorm) kurulur (yalniz train override) ve
elf.Exp_Long_Term_Forecast = ExpLastNormDRO yapilir. Model, normalizasyon, optimizasyon, val kaybi, erken durdurma,
lr cizelgesi, h336 epoch'u (1), veri ve seed ebeveynle ayni.

Rakip koduna, scripts/rakip_kos.py'ye ve varyantlar/lastnorm/ dosyalarina DOKUNULMAZ (sys.dont_write_bytecode).
`--kapi-test-metrics` REDDEDILIR (K4 test kilidi); cikista test_ anahtari denetlenir.
metrics.json ekleri: varyant, kart, ebeveyn, dro_eta, dro_K, dro_q_son_h{H} (liste), dro_grup_n_h{H},
dro_grup_mse_son_h{H} (grup basina son-epoch train MSE), dro_q_maks_son_h{H}.
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/lastnorm-dro/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

LASTNORM_KOS = WS / "varyantlar" / "lastnorm" / "kos.py"

VARYANT = "lastnorm-dro"
KART = "H09"
EBEVEYN = "lastnorm"


def lastnorm_kos_yukle():
    spec = importlib.util.spec_from_file_location("lastnorm_kos", str(LASTNORM_KOS))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lastnorm_kos"] = mod
    spec.loader.exec_module(mod)
    return mod


LK = lastnorm_kos_yukle()  # rakip_kos.resmi_sh_argumanlari'nin ORIJINALINI yakalar
sys.path.insert(0, str(HERE))
import exp_dro  # noqa: E402


def exp_sinifi_kur():
    import experiments.exp_long_term_forecasting as elf
    ExpLastNorm = LK.exp_sinifi_kur()  # lastnorm ile birebir (elf adini da yamar)
    ExpLastNormDRO = exp_dro.dro_sinifi_kur(ExpLastNorm)
    elf.Exp_Long_Term_Forecast = ExpLastNormDRO  # rakip_kos.main() bu adi yerel import ediyor
    return ExpLastNormDRO


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H09] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    bilinen, _ = ap.parse_known_args()

    rakip_kos.resmi_sh_argumanlari = LK.yamali_sh_argumanlari
    exp_sinifi_kur()
    print(f"[H09] varyant={VARYANT}: lastnorm (--use_norm 0 + LastNorm) + GroupDRO egitim kaybi "
          f"(K={exp_dro.DRO_K} kronolojik train grubu, eta={exp_dro.DRO_ETA}); val/erken durdurma resmi MSE", flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H09] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN, "dro_eta": exp_dro.DRO_ETA, "dro_K": exp_dro.DRO_K})
    for H, k in sorted(exp_dro.DRO_KAYIT.items()):
        m[f"dro_q_son_h{H}"] = k["q_son"]
        m[f"dro_q_maks_son_h{H}"] = max(k["q_son"])
        m[f"dro_grup_n_h{H}"] = k["grup_n"]
        m[f"dro_grup_mse_son_h{H}"] = k["grup_mse_son"]
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
