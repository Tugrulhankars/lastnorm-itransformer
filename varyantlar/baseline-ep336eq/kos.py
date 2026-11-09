#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/baseline-ep336eq/kos.py -- H03 (PROTOKOL KARTI): h336'da resmi `--train_epochs 1` -> 10.

Ebeveyn: `baseline` (scripts/rakip_kos.py, resmi iTransformer c2426e6 Exchange .sh birebir).
Tek degisiklik: resmi .sh bloklarindan YALNIZ pred_len=336 blogunda `--train_epochs` 1 -> 10.
h96/h192/h720 bloklari dokunulmadan doner (ebeveynle bit-bit ayni olmali).

Rakip koduna ve scripts/rakip_kos.py'ye DOKUNULMAZ: rakip_kos import edilir, modul ozniteligi
`resmi_sh_argumanlari` sarilir, sonra rakip_kos.main() cagrilir (ayni argumanlar: --seed, --out,
--ufuklar, --epoch-ust, --num-workers). `--kapi-test-metrics` REDDEDILIR (K4 test kilidi).

Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/baseline-ep336eq/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

VARYANT = "baseline-ep336eq"
KART = "H03"
EBEVEYN = "baseline"
EPOCH_YENI_H336 = "10"

_orijinal_sh = rakip_kos.resmi_sh_argumanlari


def yamali_sh_argumanlari():
    bloklar = _orijinal_sh()
    a336 = bloklar[336]
    # yama-oncesi saglama: resmi betikte yalniz h336 blogunda --train_epochs 1 var
    if "--train_epochs" not in a336 or a336[a336.index("--train_epochs") + 1] != "1":
        raise SystemExit("[H03] beklenmeyen resmi betik: h336 blogunda '--train_epochs 1' yok")
    for h in (96, 192, 720):
        if "--train_epochs" in bloklar[h]:
            raise SystemExit(f"[H03] beklenmeyen resmi betik: h{h} blogunda --train_epochs var")
    yeni = dict(bloklar)
    yeni[336] = rakip_kos.argv_degistir(a336, "--train_epochs", EPOCH_YENI_H336)
    return yeni


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H03] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ufuklar", default="96,192,336,720")
    bilinen, _ = ap.parse_known_args()

    rakip_kos.resmi_sh_argumanlari = yamali_sh_argumanlari
    print(f"[H03] varyant={VARYANT}: yalniz h336 --train_epochs 1 -> {EPOCH_YENI_H336}; diger ufuklar resmi", flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H03] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN, "protokol_karti": 1})
    if "336" in bilinen.ufuklar.split(","):
        m["epoch_resmi_betik_h336"] = 1  # epoch_resmi_h336 yamadan sonra 10 okur; resmi betik degeri 1
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
