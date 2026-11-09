#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula.py -- H14 CPU saglamalari (kart H14 "Uygulama notu" (i)-(v)), 3 kat x 3 seed (kuyruktaki 9 kaynak kosum).
Varyant MSE'si BASILMAZ (yalniz saglama farklari).
  (i)   w=1 -> C == P bit-bit, MSE == ebeveyn metrics (<1e-6)
  (ii)  w=0 -> C == Repeat bit-bit, MSE == sonuclar/naif-kat.csv (<1e-6)
  (iii) kimlik |MSE_C - (0.25 MSE_P + 0.25 MSE_R + 0.5 E[e_P e_R])| <= 1e-6
  (iv)  test_kilitli/, test_* dosyasi, test_ anahtarli metrics.json -> SystemExit (birlesim.guvenli/metrics_oku)
  (v)   K0 / ic model / ic-holdout dosyasi acilmaz: builtins.open + np.load izlenir; runs/ altindan okunan her
        preds/ ve metrics.json yolu yalniz hedef kosuma ait; '_lastnorm-k0_' iceren yol yok; test_ yolu yok.
  (vi)  agirlik kestirimi cagrilmaz: birlesim.agirlik_kestir / tum_agirliklar cagrilirsa hata.
Kullanim (WS kokunden): env/.venv/Scripts/python.exe varyantlar/lastnorm-esit/dogrula.py
Cikti: varyantlar/lastnorm-esit/dogrula.log
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import builtins  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esit as S  # noqa: E402
import numpy as np  # noqa: E402

B = S.B
KAYNAK = {3: {2023: "20261014-052508_lastnorm_s2023", 2024: "20261014-053601_lastnorm_s2024",
              2025: "20261014-054213_lastnorm_s2025"},
          2: {2023: "20261014-163051_lastnorm-k2_s2023", 2024: "20261014-163203_lastnorm-k2_s2024",
              2025: "20261014-163345_lastnorm-k2_s2025"},
          1: {2023: "20261014-163510_lastnorm-k1_s2023", 2024: "20261014-163633_lastnorm-k1_s2024",
              2025: "20261014-163802_lastnorm-k1_s2025"}}

ACILAN = []
_open, _npload, _ptread = builtins.open, np.load, Path.read_text


def _izle_open(f, *a, **k):
    ACILAN.append(str(f).replace("\\", "/"))
    return _open(f, *a, **k)


def _izle_npload(f, *a, **k):
    ACILAN.append(str(f).replace("\\", "/"))
    return _npload(f, *a, **k)


def _izle_read_text(self, *a, **k):
    ACILAN.append(str(self).replace("\\", "/"))
    return _ptread(self, *a, **k)


def _yasak(*a, **k):
    raise AssertionError("(vi) agirlik kestirimi cagrildi")


def main():
    L = _open(HERE / "dogrula.log", "w", encoding="utf-8")

    def yaz(s=""):
        print(s, flush=True)
        L.write(s + "\n")
        L.flush()

    builtins.open, np.load, Path.read_text = _izle_open, _izle_npload, _izle_read_text
    B.agirlik_kestir = _yasak
    B.tum_agirliklar = _yasak
    ok = True
    yaz(f"[dogrula H14] {time.strftime('%Y-%m-%d %H:%M:%S')} varyant=lastnorm-esit w={S.W_SABIT} sabit | CPU "
        f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    ham = B.ham_yukle()
    for kat in (3, 2, 1):
        for seed in (2023, 2024, 2025):
            ACILAN.clear()
            S.OKUNAN.clear()
            r = S.hesapla(kat, seed, ham=ham, log=yaz)
            beklenen = KAYNAK[kat][seed]
            hd_ok = r["hedef"] == beklenen
            runs_yol = [p for p in ACILAN if "/runs/" in p]
            preds_metrics = [p for p in runs_yol if "/preds/" in p or p.endswith("/metrics.json")]
            v_ok = (all(f"/runs/{beklenen}/" in p for p in preds_metrics)
                    and not any("_lastnorm-k0_" in p for p in ACILAN)
                    and not any("test_kilitli" in p or Path(p).name.startswith("test_") for p in ACILAN)
                    and len(set(preds_metrics)) == 5)
            sg = r["saglama"]
            yaz(f"  => K{kat} s{seed} hedef={r['hedef']} (beklenen {beklenen}: {hd_ok}) | (i) {sg['i']} (ii) {sg['ii']} "
                f"(iii) {sg['iii']} (v-esit) {sg['v']} (v-izleme: {len(set(preds_metrics))} ayri runs/ preds+metrics dosyasi ({len(preds_metrics)} acilis), "
                f"hepsi hedef, K0 yok) {v_ok}")
            ok &= all(sg.values()) and hd_ok and v_ok
    builtins.open, np.load, Path.read_text = _open, _npload, _ptread
    # (iv) test kilidi
    d_ok = True
    for p in ("runs/x/preds/test_kilitli/test_h96.npz", "runs/x/test_metrics.json"):
        try:
            B.guvenli(Path(p))
            d_ok = False
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "metrics.json").write_text(json.dumps({"val_mse_avg": 1.0, "test_mse_avg": 1.0}), encoding="utf-8")
        try:
            B.metrics_oku(Path(td))
            d_ok = False
        except SystemExit:
            pass
    yaz(f"\n(iv) test_kilitli/, test_* dosyasi, test_ anahtarli metrics.json -> SystemExit: {d_ok}")
    yaz("(vi) birlesim.agirlik_kestir / tum_agirliklar yasaklanarak 9 hesap kosuldu -> cagri yok: True")
    ok &= d_ok
    yaz(f"\nSONUC: {'GECTI' if ok else 'KALDI'}")
    L.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
