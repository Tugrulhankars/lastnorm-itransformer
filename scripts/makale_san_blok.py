# -*- coding: utf-8 -*-
"""Makale -- SAN (H01) test ceyrek (blok) kirilimi + ortalamaya-donus katsayisi (hakem tur-1 B1(a); BETIMSEL).
(makale icin sonradan eklendi -- kanit-cikarici, FAZ III, 2026-10-15; CPU son-islem, egitim/GPU yok)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_san_blok.py

TEST NOTU: Exchange test 2026-10-14 15:42:23'te tek resmi acilisla acildi; SAN kilitli test tahminleri o acilista
zaten okunmustu. Bu betik YALNIZ RAPORLAMA yapar; secim/ayar uretmez, yeni kosum yoktur.

Kaynak kosumlar (rc=0, 3 seed): SAN runs/20261014-{040042,040547,041102}_san_s{2023,2024,2025};
iTransformer / LastNorm: makale_ortak.RUNS['exchange'] (test-sonuclari.csv ile ayni).

(1) BLOK MSE -- makale_hata_kirilim.py ile AYNI tanim: test dilimin kendi satirlari np.array_split K=4 (Q1..Q4);
    her (pencere, adim, degisken) hucresi hedef satirina gore bloga atanir; MSE z uzayi; model seed basina, sonra
    3 seed ort./sd(ddof=1). Repeat seed'siz. 'avg' = 4 ufkun blok MSE'lerinin esit agirlikli ortalamasi.
(2) ORTALAMAYA-DONUS KATSAYISI beta (ufuk x blok; blok = pencerenin ILK hedef satirinin dustugu ceyrek):
    a_ic = xbar_L,ic - x_L,ic  (girdi penceresi ortalamasi - son gozlem), d_ic = mean_h yhat_ic,h - x_L,ic;
    beta = sum(a*d)/sum(a^2) (sabitsiz, pencere x degisken havuzlu). Repeat beta=0, pencere-ortalamasi tahmincisi
    beta=1 (tanim geregi). 'gercek' satiri ayni hesabi gerceklesen hedefle yapar (donemin gerceklesen donusu).
    Model beta'si seed basina hesaplanir, 3 seed ort./sd verilir.
DOGRULAMA (assert): (a) TUM blok MSE == test_metrics.json test_mse_hH (1e-5) her kosum-ufuk;
  (b) iTr/LastNorm/Repeat blok MSE == sonuclar/makale/hata-kirilim_exchange.csv (degisken=TUM) (1.5e-6);
  (c) Repeat beta == 0 ve pencere-ortalamasi beta == 1 (1e-9).
Cikti: sonuclar/makale/san-blok.csv, sonuclar/makale/san-donus.csv
"""
import csv
import sys

import numpy as np

import makale_ortak as O

sys.stdout.reconfigure(encoding="utf-8")
SAN = ["20261014-040042_san_s2023", "20261014-040547_san_s2024", "20261014-041102_san_s2025"]
MODELLER = {"itr": O.RUNS["exchange"]["itr"], "lastnorm": O.RUNS["exchange"]["lastnorm"], "san": SAN}


def hk_oku():
    p = O.OUT / "hata-kirilim_exchange.csv"
    with open(p, encoding="utf-8") as f:
        sat = [l for l in f if not l.startswith("#")]
    return {(int(r["ufuk"]), r["blok"]): r for r in csv.DictReader(sat)
            if r["bolum"] == "test" and r["degisken"] == "TUM"}


def beta(a, d):
    if a.size == 0:  # blokta pencere baslangici yok (ör. H=720'de Q4): tanimsiz -> NaN, CSV'de bos
        return float("nan")
    return float((a * d).sum() / (a * a).sum())


def main():
    O.kosumlari_dogrula()
    import json
    for sd, rid in zip(O.SEEDS, SAN):
        done = json.loads((O.WS / "runs" / rid / ".done").read_text(encoding="utf-8"))
        assert done["rc"] == 0 and rid.endswith(f"_s{sd}"), rid
    z, cols = O.yukle_z("exchange")
    bl = O.blok_sinirlari("exchange", "test")
    bad = O.blok_adlari("test")
    hk = hk_oku()
    blok_sat, donus_sat = [], []
    dog = {"a": 0, "b": 0, "c": 0}
    blok_mse = {}  # (m, H) -> (S, K+1)
    for H in O.UFUKLAR:
        Y = O.yukle_true("exchange", "test", H)
        N = Y.shape[0]
        ilk = O.yukle_pred(MODELLER["itr"][0], "test", H)["hedef_ilk_satir"].astype(np.int64)
        trow = ilk[:, None] + np.arange(H)[None, :]
        maskeler = [(trow >= b0) & (trow < b1) for (b0, b1) in bl] + [np.ones((N, H), bool)]
        # pencere blogu (beta icin): ilk hedef satiri
        pb = [((ilk >= b0) & (ilk < b1)) for (b0, b1) in bl] + [np.ones(N, bool)]
        R, X, _ = O.repeat_ve_girdi(z, ilk, H)
        xL = X[:, -1, :]                       # (N,C)
        a = X.mean(1) - xL                     # (N,C)
        # dogrulama (c)
        assert abs(beta(a, R.mean(1) - xL)) < 1e-9
        assert abs(beta(a, X.mean(1) - xL) - 1.0) < 1e-9
        dog["c"] += 1
        for m, ids in MODELLER.items():
            ms, bs = [], []
            for rid in ids:
                f = O.yukle_pred(rid, "test", H)
                assert np.array_equal(f["hedef_ilk_satir"], ilk), rid
                P = f["pred"].astype(np.float64)
                E = (P - Y) ** 2
                mm = np.array([E[msk].mean() for msk in maskeler])
                assert abs(mm[-1] - O.kayitli_mse(rid, "test", H)) < 1e-5, (rid, H)
                dog["a"] += 1
                d = P.mean(1) - xL
                bs.append([beta(a[w], d[w]) for w in pb])
                ms.append(mm)
                del P, E
            blok_mse[(m, H)] = np.array(ms)
            bsa = np.array(bs)
            for k in range(len(pb)):
                donus_sat.append(dict(ufuk=H, blok=("TUM" if k == len(pb) - 1 else bad[k]), seri=m,
                                      n_pencere=int(pb[k].sum()), beta_ort=O.r6(bsa[:, k].mean()),
                                      beta_sd=O.r6(bsa[:, k].std(ddof=1)),
                                      **{f"beta_s{s}": O.r6(bsa[i, k]) for i, s in enumerate(O.SEEDS)}))
        ER = (R - Y) ** 2
        rm = np.array([ER[msk].mean() for msk in maskeler])
        blok_mse[("repeat", H)] = rm[None, :]
        dg = Y.mean(1) - xL
        for k in range(len(pb)):
            donus_sat.append(dict(ufuk=H, blok=("TUM" if k == len(pb) - 1 else bad[k]), seri="gercek",
                                  n_pencere=int(pb[k].sum()), beta_ort=O.r6(beta(a[pb[k]], dg[pb[k]])), beta_sd="",
                                  **{f"beta_s{s}": "" for s in O.SEEDS}))
        # dogrulama (b)
        for k in range(len(maskeler)):
            b_ad = "TUM" if k == len(maskeler) - 1 else bad[k]
            r = hk[(H, b_ad)]
            assert abs(blok_mse[("itr", H)][:, k].mean() - float(r["itr_ort"])) < 1.5e-6
            assert abs(blok_mse[("lastnorm", H)][:, k].mean() - float(r["lastnorm_ort"])) < 1.5e-6
            assert abs(rm[k] - float(r["repeat"])) < 1.5e-6
            dog["b"] += 1
    # blok satirlari (ufuk + avg)
    K1 = len(bl) + 1
    for u in O.UFUKLAR + ["avg"]:
        for k in range(K1):
            b_ad = "TUM" if k == K1 - 1 else bad[k]
            v = {}
            for m in ("itr", "lastnorm", "san", "repeat"):
                arr = (blok_mse[(m, u)][:, k] if u != "avg"
                       else np.mean([blok_mse[(m, H)][:, k] for H in O.UFUKLAR], axis=0))
                v[m] = arr
            s, rep = v["san"], float(v["repeat"][0])
            blok_sat.append(dict(
                ufuk=u, blok=b_ad, blok_satir=("" if b_ad == "TUM" else f"[{bl[k][0]},{bl[k][1]})"),
                itr_ort=O.r6(v["itr"].mean()), lastnorm_ort=O.r6(v["lastnorm"].mean()),
                san_ort=O.r6(s.mean()), san_sd=O.r6(s.std(ddof=1)),
                **{f"san_s{sd}": O.r6(s[i]) for i, sd in enumerate(O.SEEDS)},
                repeat=O.r6(rep),
                san_M_R=round(float(s.mean() / rep), 4), lastnorm_M_R=round(float(v["lastnorm"].mean() / rep), 4),
                itr_M_R=round(float(v["itr"].mean() / rep), 4),
                d_san_itr_pct=round(float((s.mean() / v["itr"].mean() - 1) * 100), 3),
                d_san_ln_pct=round(float((s.mean() / v["lastnorm"].mean() - 1) * 100), 3),
                d_san_rep_pct=round(float((s.mean() / rep - 1) * 100), 3),
                san_iyi_rep_seed=int((s < rep).sum())))
    run_ids = MODELLER["itr"] + MODELLER["lastnorm"] + SAN
    meta = O.meta_satirlari(
        "makale_san_blok.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_san_blok.py",
        ["HAKEM TUR-1 B1(a) icin betimsel kirilim (FAZ III); test zaten acikti; secim yok",
         "tanim: makale_hata_kirilim.py ile ayni blok atamasi (hedef satiri), z uzayi MSE; avg = 4 ufkun esit agirlikli ort.",
         "bloklar: " + "; ".join(f"{b}={x}" for b, x in zip(bad, bl)),
         "san_M_R = san_ort/repeat; d_* = 100*(san_ort/ref-1); san_iyi_rep_seed = SAN<Repeat olan seed sayisi",
         f"dogrulama: (a) {dog['a']} kosum-ufuk TUM==test_metrics (1e-5); (b) {dog['b']} iTr/LN/Repeat blok "
         f"satiri==hata-kirilim_exchange.csv (1.5e-6); (c) {dog['c']} beta tanim sinamasi (Repeat 0, pencere-ort 1)"],
        run_ids)
    O.csv_yaz(O.OUT / "san-blok.csv", blok_sat, meta)
    meta2 = O.meta_satirlari(
        "makale_san_blok.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_san_blok.py",
        ["HAKEM TUR-1 B1(a): ortalamaya-donus katsayisi beta = sum(a*d)/sum(a^2); a = xbar_L - x_L, "
         "d = mean_h(yhat_h) - x_L; pencere x degisken havuzlu, sabitsiz; Repeat=0, pencere-ortalamasi=1 (tanim)",
         "blok = pencerenin ilk hedef satirinin dustugu test ceyregi; seri=gercek: ayni hesap gerceklesen hedefle",
         f"dogrulama: (c) {dog['c']} ufukta beta(Repeat)=0 ve beta(pencere-ort)=1 (1e-9)"],
        run_ids)
    for r in donus_sat:  # NaN (pencere yok) -> bos hucre
        for k2, v2 in list(r.items()):
            if isinstance(v2, float) and np.isnan(v2):
                r[k2] = ""
    O.csv_yaz(O.OUT / "san-donus.csv", donus_sat, meta2)
    print("DOGRULAMA GECTI:", dog)
    for r in blok_sat:
        if r["ufuk"] == "avg":
            print(r["blok"], "iTr", r["itr_ort"], "LN", r["lastnorm_ort"], "SAN", r["san_ort"], "Rep", r["repeat"],
                  "SAN/Rep", r["san_M_R"])
    for r in donus_sat:
        if r["blok"] == "TUM":
            print("beta", r["ufuk"], r["seri"], r["beta_ort"])


if __name__ == "__main__":
    main()
