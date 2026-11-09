#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
varyantlar/lastnorm-ep336eq/kos.py -- H08 (PROTOKOL KARTI): `lastnorm` uzerinde h336 `--train_epochs` 1 -> 10.

Ebeveyn: `lastnorm` (H04; varyantlar/lastnorm/kos.py: her ufukta --use_norm 0 + LastNormiTransformer).
Tek degisiklik: resmi .sh bloklarindan YALNIZ pred_len=336 blogunda `--train_epochs` 1 -> 10 (H03 ile ayni yama).
h96/h192/h720 bloklari `lastnorm` ile bit-bit ayni olmali.

Uygulama (kart uygulama notu):
  * `varyantlar/lastnorm/kos.py` importlib.util ile DOSYA YOLUNDAN yuklenir (modul adi `lastnorm_kos`; `kos` ad
    cakismasi yok). Onun `yamali_sh_argumanlari` (dort blokta --use_norm 0) ve `exp_sinifi_kur` (ExpLastNorm:
    _build_model -> LastNormiTransformer, __init__ -> Exp_Basic.__init__) BIREBIR kullanilir; LastNormiTransformer
    `varyantlar/lastnorm/model_lastnorm.py`'den sys.path ile import edilir (kopya yok).
  * Yama-oncesi saglamalar `varyantlar/baseline-ep336eq/kos.py`'den aynen: h336 blogunda `--train_epochs 1`,
    h96/h192/h720 bloklarinda `--train_epochs` yok.
  * sys.dont_write_bytecode: varyantlar/lastnorm/ altina __pycache__ yazilmaz (o klasore dokunulmaz).

Rakip koduna, scripts/rakip_kos.py'ye ve varyantlar/lastnorm/ dosyalarina DOKUNULMAZ.
`--kapi-test-metrics` REDDEDILIR (K4 test kilidi); cikista test_ anahtari denetlenir.
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/lastnorm-ep336eq/kos.py --seed 2023 --out {RUN_DIR}/metrics.json --num-workers 0
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

VARYANT = "lastnorm-ep336eq"
KART = "H08"
EBEVEYN = "lastnorm"
EPOCH_YENI_H336 = "10"


def lastnorm_kos_yukle():
    spec = importlib.util.spec_from_file_location("lastnorm_kos", str(LASTNORM_KOS))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lastnorm_kos"] = mod
    spec.loader.exec_module(mod)
    return mod


LK = lastnorm_kos_yukle()  # rakip_kos.resmi_sh_argumanlari'nin ORIJINALINI yakalar (yamadan once yuklenmeli)


def yamali_sh_argumanlari():
    bloklar = LK.yamali_sh_argumanlari()  # lastnorm: dort blokta --use_norm 0
    a336 = bloklar[336]
    # yama-oncesi saglama (baseline-ep336eq/kos.py ile ayni)
    if "--train_epochs" not in a336 or a336[a336.index("--train_epochs") + 1] != "1":
        raise SystemExit("[H08] beklenmeyen resmi betik: h336 blogunda '--train_epochs 1' yok")
    for h in (96, 192, 720):
        if "--train_epochs" in bloklar[h]:
            raise SystemExit(f"[H08] beklenmeyen resmi betik: h{h} blogunda --train_epochs var")
    for h, a in bloklar.items():
        if a[a.index("--use_norm") + 1] != "0":
            raise SystemExit(f"[H08] lastnorm yamasi devrede degil: h{h} --use_norm != 0")
    yeni = dict(bloklar)
    yeni[336] = rakip_kos.argv_degistir(a336, "--train_epochs", EPOCH_YENI_H336)
    return yeni


def main():
    if "--kapi-test-metrics" in sys.argv:
        print("[H08] HATA: --kapi-test-metrics K4 varyantlarinda yasak (test kilidi)", flush=True)
        return 2
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ufuklar", default="96,192,336,720")
    bilinen, _ = ap.parse_known_args()

    rakip_kos.resmi_sh_argumanlari = yamali_sh_argumanlari
    LK.exp_sinifi_kur()  # elf.Exp_Long_Term_Forecast = ExpLastNorm (lastnorm ile birebir)
    print(f"[H08] varyant={VARYANT}: lastnorm (--use_norm 0 + LastNorm, her ufuk) + yalniz h336 --train_epochs 1 -> "
          f"{EPOCH_YENI_H336}; diger ufuklar lastnorm ile ayni", flush=True)
    rc = rakip_kos.main()
    if rc != 0:
        return rc
    out = Path(bilinen.out).resolve()
    m = json.loads(out.read_text(encoding="utf-8"))
    if any(k.startswith("test_") for k in m):
        raise SystemExit("[H08] HATA: metrics.json'da test_ anahtari var (K4 kilidi)")
    m.update({"varyant": VARYANT, "kart": KART, "ebeveyn": EBEVEYN, "protokol_karti": 1})
    if "336" in bilinen.ufuklar.split(","):
        m["epoch_resmi_betik_h336"] = 1  # epoch_resmi_h336 yamadan sonra 10 okur; resmi betik degeri 1
    rakip_kos.atomik_json(out, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
