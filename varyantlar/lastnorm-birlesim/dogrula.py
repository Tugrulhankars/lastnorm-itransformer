#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula.py -- H12/H13 CPU saglamalari (kart H12 "Uygulama notu" (i)-(v); H13: ayni + alpha=1 -> C == P) ve
SIZINTI testi: agirliklar yalniz katin train'inden mi?

Mevcut tam kosumlarla (seed 2023): K2 (ic model lastnorm-k1, hedef lastnorm-k2) ve K3 (ic model lastnorm-k2, hedef
resmi lastnorm 052508). K1 icin ic model lastnorm-k0 kuyrukta; K1 saglamalari kuyruk satirinin kendisinde kosar
(birlesim.py saglama basarisizsa rc=1, metrics yazmaz). Varyant MSE'si BASILMAZ (yalniz saglama ve agirliklar).
  (A) (i) w=1 -> C == P bit-bit ve MSE == hedef metrics; (ii) w=0 -> C == Repeat, MSE == naif-kat.csv;
      (iii) ic-holdout satirlari < b_k (assert, birlesim.agirlik_kestir); (iv) P' MSE == K_{k-1} metrics (<1e-6).
  (B) SIZINTI: ham veride b_k ve sonrasindaki TUM satirlar NaN yapilinca agirliklar bit-bit ayni (kat val'i, test
      ve sonrasi kestirime girmiyor); duyarlilik kontrolu: b_k-1 satiri (ic-holdout'un son hedef satiri) bozulunca
      agirliklar degisiyor (test kor degil).
  (C) Hedef (kat val) tahminleri rastgele gurultuyle degistirilince agirliklar bit-bit ayni.
  (D) (v) test_kilitli/ yolu, test_* dosyasi ve test_ anahtarli metrics.json okunmaya calisilinca SystemExit.
  (E) donusum: H12 clip(a,0,1), H13 max(a,0).
Kullanim (WS kokunden): env/.venv/Scripts/python.exe varyantlar/lastnorm-birlesim/dogrula.py --varyant birlesim|olcek
Cikti: varyantlar/lastnorm-birlesim/dogrula.log veya varyantlar/lastnorm-olcek/dogrula.log
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lastnorm-olcek"))
import birlesim as B  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--varyant", choices=["birlesim", "olcek"], default="birlesim")
    a = ap.parse_args()
    if a.varyant == "birlesim":
        cfg, log_yol = B.VARSAYILAN, HERE / "dogrula.log"
    else:
        import olcek
        cfg, log_yol = olcek.CFG, HERE.parent / "lastnorm-olcek" / "dogrula.log"
    L = open(log_yol, "w", encoding="utf-8")

    def yaz(s=""):
        print(s, flush=True)
        L.write(s + "\n")
        L.flush()

    ok = True
    yaz(f"[dogrula {cfg['kart']}] {time.strftime('%Y-%m-%d %H:%M:%S')} varyant={cfg['varyant']} "
        f"donusum={cfg['donusum_adi']} | CPU")
    ham = B.ham_yukle()
    rng = np.random.default_rng(0)
    for kat in (2, 3):
        T, V, b_k, vs = B.b_kat(kat, ham.shape[0])
        yaz(f"\n=== K{kat} s2023: agirlik verisi [0,{b_k}) | ic-holdout [{b_k - V},{b_k}) | kat val [{b_k},{vs})")
        r = B.hesapla(kat, 2023, cfg, ham=ham, log=yaz)
        sg = r["saglama"]
        yaz(f"  (A) ic={r['ic']} hedef={r['hedef']} | (i) {sg['i']} (ii) {sg['ii']} (iii) assert gecti "
            f"(iv) {sg['iv']} (v) {sg['v']}")
        ok &= all(sg.values())
        # (B) sizinti
        AG0 = B.tum_agirliklar(kat, 2023, cfg, ham)
        ham_nan = ham.copy()
        ham_nan[b_k:] = np.nan
        AG1 = B.tum_agirliklar(kat, 2023, cfg, ham_nan)
        es_B = all(np.array_equal(AG0[H][0], AG1[H][0], equal_nan=True) and AG0[H][1] == AG1[H][1] for H in B.UFUKLAR)
        ham_boz = ham.copy()
        ham_boz[b_k - 1] += 0.5 * ham[:b_k].std(0)
        AG2 = B.tum_agirliklar(kat, 2023, cfg, ham_boz)
        duyarli = any(not np.array_equal(AG0[H][0], AG2[H][0]) for H in B.UFUKLAR)
        yaz(f"  (B) satir >= {b_k} NaN -> agirliklar bit-bit ayni: {es_B} | satir {b_k - 1} bozuldu -> alpha_hat "
            f"degisti: {duyarli}")
        ok &= es_B and duyarli
        # (C) hedef tahminleri rastgele
        r2 = B.hesapla(kat, 2023, cfg, ham=ham,
                       hedef_pred_degistir=lambda P: rng.normal(size=P.shape).astype(np.float32), log=lambda *_: None)
        es_C = all(r["H"][H]["w"] == r2["H"][H]["w"] and
                   np.array_equal(r["H"][H]["alfa"], r2["H"][H]["alfa"], equal_nan=True) for H in B.UFUKLAR)
        yaz(f"  (C) kat val tahminleri rastgele -> agirliklar bit-bit ayni: {es_C}")
        ok &= es_C
        for H in B.UFUKLAR:
            yaz(f"  bilgi h{H}: alpha_hat=[" + ", ".join(f"{x:.3f}" for x in r["H"][H]["alfa"]) + "]")
    # (D) test kilidi
    yaz("")
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
    yaz(f"(D) (v) test_kilitli/, test_* dosyasi, test_ anahtarli metrics.json -> SystemExit: {d_ok}")
    ok &= d_ok
    # (E) donusum
    f = cfg["donusum"]
    e_ok = (f(-0.3), f(0.4), f(2.5)) == ((0.0, 0.4, 1.0) if a.varyant == "birlesim" else (0.0, 0.4, 2.5))
    yaz(f"(E) donusum({cfg['donusum_adi']}): f(-0.3)={f(-0.3)} f(0.4)={f(0.4)} f(2.5)={f(2.5)} -> {e_ok}")
    ok &= e_ok
    yaz(f"\nSONUC: {'GECTI' if ok else 'KALDI'}")
    L.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
