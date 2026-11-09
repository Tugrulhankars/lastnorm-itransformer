#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/normsuz/kos.py -- H06: resmi iTransformer `--use_norm 1 -> 0` (normalizasyonsuz; ablasyon satiri).

Ebeveyn: `baseline` (scripts/rakip_kos.py; resmi iTransformer c2426e6 Exchange .sh birebir, h336 = 1 epoch).
Tek degisiklik: resmi .sh'nin HER ufuk blogunda `--use_norm 0`. Kod degisikligi yok: resmi Model.forecast
icindeki NST normalize/de-normalize bloklari atlanir. Exp sinifi DEGISMEZ (resmi Exp_Long_Term_Forecast).
Parametre sayisi, RNG tuketimi, veri, egitim ebeveynle ayni. (H03 baseline-ep336eq/kos.py kalibi.)

Rakip koduna ve scripts/rakip_kos.py'ye DOKUNULMAZ. `--kapi-test-metrics` REDDEDILIR (K4 test kilidi).
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/normsuz/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

VARYANT = "normsuz"
KART = "H06"
EBEVEYN = "baseline"

_orijinal_sh = rakip_kos.resmi_sh_argumanlari


def yamali_sh_argumanlari():
    bloklar = _orijinal_sh()
    # yama-oncesi saglama: resmi betik her ufukta --use_norm vermiyorsa run.py varsayilani (1) gecerli
    yeni = {}
    for h, a in bloklar.items():
        if "--use_norm" in a and a[a.index("--use_norm") + 1] != "1":
            raise SystemExit(f"[H06] beklenmeyen resmi betik: h{h} blogunda --use_norm {a[a.index('--use_norm') + 1]}")
        yeni[h] = rakip_kos.argv_degistir(a, "--use_norm", "0")
    return yeni


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H06] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    bilinen, _ = ap.parse_known_args()

    rakip_kos.resmi_sh_argumanlari = yamali_sh_argumanlari
    print(f"[H06] varyant={VARYANT}: her ufukta --use_norm 0 (resmi NST kapali); Exp/model resmi", flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H06] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN, "use_norm": 0})
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
