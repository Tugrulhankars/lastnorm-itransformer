# -*- coding: utf-8 -*-
"""Makale -- hata kirilimi (zaman blogu x degisken), val VE test: iTransformer vs LastNorm vs Repeat.
(makale icin sonradan eklendi -- deney-gelistirici, 2026-10-15; CPU son-islem, egitim/GPU yok)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_hata_kirilim.py [--veri exchange etth1 etth2]

TANIM (scripts/hata_analizi.py ile AYNI, sabit): dilimin KENDI satirlari [o0,o1) np.array_split ile K=4 bitisik
kronolojik bloga bolunur (val: B1..B4; test: Q1..Q4 = splits JSON 'test_ceyrekleri_kendi_satirlari', assert'li).
Her (pencere i, adim h, degisken d) hucresi HEDEF SATIRINA (hedef_ilk_satir[i]+h) gore bloga atanir.
MSE = o hucrelerin karesel hata ortalamasi (z uzayi). Model: seed basina MSE, sonra 3 seed ort./sd (ddof=1).
bias = isaretli ortalama hata (tahmin - gercek; seed-ort.). Repeat seed'siz.

Cikti: sonuclar/makale/hata-kirilim_<veri>.csv
  veri,bolum,ufuk,degisken,blok,blok_satir,n_hucre,itr_ort,itr_sd,itr_s2023,itr_s2024,itr_s2025,lastnorm_ort,lastnorm_sd,
  lastnorm_s2023,lastnorm_s2024,lastnorm_s2025,repeat,d_ln_itr_pct,ln_iyi_seed,itr_M_R,lastnorm_M_R,itr_bias,
  lastnorm_bias,repeat_bias
  (degisken = ad | TUM; blok = B1..B4/Q1..Q4 | TUM; ln_iyi_seed = LastNorm<iTr olan seed sayisi, eslestirilmis)

DOGRULAMA (assert):
  (a) TUM/TUM hucresi her kosum icin == metrics.json val_mse_hH / test_metrics.json test_mse_hH (1e-5)
  (b) Exchange val: her (ufuk, degisken, blok) satiri sonuclar/hata-kirilim.csv (baseline) ve hata-kirilim_lastnorm.csv
      model_mse_ort/model_mse_s*/repeat_mse ile esit (6 ondalik; 1.5e-6)
  (c) Repeat TUM/TUM == naif-kat*.csv (val) / naif-test.csv, vsel-test.csv (test) (1e-6)
"""
import argparse
import csv
import sys

import numpy as np

import makale_ortak as O
from makale_adim_egrisi import naif_test, naif_val

sys.stdout.reconfigure(encoding="utf-8")


def eski_kirilim(etiket):
    ad = "hata-kirilim.csv" if etiket == "itr" else "hata-kirilim_lastnorm.csv"
    with open(O.WS / "sonuclar" / ad, encoding="utf-8") as f:
        return {(int(r["ufuk"]), r["degisken"], r["blok"]): r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", nargs="+", default=["exchange", "etth1", "etth2"])
    a = ap.parse_args()
    O.kosumlari_dogrula()
    for veri in a.veri:
        z, cols = O.yukle_z(veri)
        C = len(cols)
        nv, nt = naif_val(veri), naif_test(veri)
        eski = {m: eski_kirilim(m) for m in ("itr", "lastnorm")} if veri == "exchange" else None
        satirlar, dog = [], {"a": 0, "b": 0, "c": 0}
        for bolum in ("val", "test"):
            bl = O.blok_sinirlari(veri, bolum)
            bad = O.blok_adlari(bolum)
            for H in O.UFUKLAR:
                Y = O.yukle_true(veri, bolum, H)
                N = Y.shape[0]
                ilk = O.yukle_pred(O.RUNS[veri]["itr"][0], bolum, H)["hedef_ilk_satir"].astype(np.int64)
                trow = ilk[:, None] + np.arange(H)[None, :]
                maskeler = [(trow >= b0) & (trow < b1) for (b0, b1) in bl] + [np.ones((N, H), bool)]
                # mse[m] (S, K+1, C+1), bias[m] (S, K+1, C+1)
                sonuc = {}
                for m in ("itr", "lastnorm"):
                    ms, bs = [], []
                    for rid in O.RUNS[veri][m]:
                        f = O.yukle_pred(rid, bolum, H)
                        assert np.array_equal(f["hedef_ilk_satir"], ilk)
                        D = f["pred"].astype(np.float64) - Y
                        E = D ** 2
                        mm = np.zeros((len(maskeler), C + 1)); bb = np.zeros_like(mm)
                        for k, msk in enumerate(maskeler):
                            ek, dk = E[msk], D[msk]  # (n,C)
                            mm[k, :C] = ek.mean(0); mm[k, C] = ek.mean()
                            bb[k, :C] = dk.mean(0); bb[k, C] = dk.mean()
                        assert abs(mm[-1, -1] - O.kayitli_mse(rid, bolum, H)) < 1e-5, (rid, bolum, H)
                        dog["a"] += 1
                        ms.append(mm); bs.append(bb)
                        del D, E
                    sonuc[m] = (np.stack(ms), np.stack(bs))
                R, _, _ = O.repeat_ve_girdi(z, ilk, H)
                DR = R - Y
                ER = DR ** 2
                rm = np.zeros((len(maskeler), C + 1)); rb = np.zeros_like(rm)
                for k, msk in enumerate(maskeler):
                    rm[k, :C] = ER[msk].mean(0); rm[k, C] = ER[msk].mean()
                    rb[k, :C] = DR[msk].mean(0); rb[k, C] = DR[msk].mean()
                assert abs(rm[-1, -1] - (nv if bolum == "val" else nt)[H]) < 1.5e-6, (veri, bolum, H)
                dog["c"] += 1
                for k in range(len(maskeler)):
                    b_ad = "TUM" if k == K_SON(maskeler) else bad[k]
                    b_sat = "" if b_ad == "TUM" else f"[{bl[k][0]},{bl[k][1]})"
                    n_h = int(maskeler[k].sum())
                    for j in range(C + 1):
                        d_ad = "TUM" if j == C else cols[j]
                        it = sonuc["itr"][0][:, k, j]; ln = sonuc["lastnorm"][0][:, k, j]
                        satirlar.append(dict(
                            veri=veri, bolum=bolum, ufuk=H, degisken=d_ad, blok=b_ad, blok_satir=b_sat,
                            n_hucre=n_h * (C if j == C else 1),
                            itr_ort=O.r6(it.mean()), itr_sd=O.r6(it.std(ddof=1)),
                            **{f"itr_s{sd}": O.r6(it[i]) for i, sd in enumerate(O.SEEDS)},
                            lastnorm_ort=O.r6(ln.mean()), lastnorm_sd=O.r6(ln.std(ddof=1)),
                            **{f"lastnorm_s{sd}": O.r6(ln[i]) for i, sd in enumerate(O.SEEDS)},
                            repeat=O.r6(rm[k, j]),
                            d_ln_itr_pct=round(float((ln.mean() / it.mean() - 1) * 100), 3),
                            ln_iyi_seed=int((ln < it).sum()),
                            itr_M_R=round(float(it.mean() / rm[k, j]), 4),
                            lastnorm_M_R=round(float(ln.mean() / rm[k, j]), 4),
                            itr_bias=round(float(sonuc["itr"][1][:, k, j].mean()), 5),
                            lastnorm_bias=round(float(sonuc["lastnorm"][1][:, k, j].mean()), 5),
                            repeat_bias=round(float(rb[k, j]), 5)))
                        if eski is not None and bolum == "val":
                            eb = "TUM" if b_ad == "TUM" else b_ad
                            for m, arr in (("itr", it), ("lastnorm", ln)):
                                r = eski[m][(H, d_ad, eb)]
                                assert abs(arr.mean() - float(r["model_mse_ort"])) < 1.5e-6, (m, H, d_ad, eb)
                                for i in range(3):
                                    assert abs(arr[i] - float(r[f"model_mse_s{i}"])) < 1.5e-6
                                assert abs(rm[k, j] - float(r["repeat_mse"])) < 1.5e-6
                                assert int(r["n_hucre"]) == n_h * (C if j == C else 1)
                                dog["b"] += 1
                tb = sonuc["itr"][0][:, :K_BLOK, C].mean(0); lb = sonuc["lastnorm"][0][:, :K_BLOK, C].mean(0)
                print(f"{veri} {bolum} H={H}: blok itr {np.round(tb, 4)} ln {np.round(lb, 4)} rep {np.round(rm[:K_BLOK, C], 4)}"
                      f" | d% {np.round((lb / tb - 1) * 100, 1)}")
        meta = O.meta_satirlari(
            "makale_hata_kirilim.py",
            f"PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_hata_kirilim.py --veri {veri}",
            ["tanim (hata_analizi.py ile ayni): dilimin kendi satirlari np.array_split K=4 (val B1-B4, test Q1-Q4); hucre "
             "hedef satirina gore bloga atanir; MSE z uzayi; model seed basina sonra 3 seed ort./sd(ddof=1); repeat seed'siz",
             "bloklar: " + "; ".join(f"{b}={x}" for bol in ("val", "test")
                                     for b, x in zip(O.blok_adlari(bol), O.blok_sinirlari(veri, bol))),
             "d_ln_itr_pct = 100*(lastnorm_ort/itr_ort-1); ln_iyi_seed = eslestirilmis seed'lerde lastnorm<itr sayisi; "
             "*_bias = isaretli ort. hata (tahmin-gercek)",
             f"dogrulama: (a) {dog['a']} kosum-ufuk TUM/TUM==kayitli MSE (1e-5); (b) {dog['b']} Exchange val satiri=="
             f"hata-kirilim*.csv; (c) {dog['c']} Repeat==naif dosyalari"],
            O.RUNS[veri]["itr"] + O.RUNS[veri]["lastnorm"])
        O.csv_yaz(O.OUT / f"hata-kirilim_{veri}.csv", satirlar, meta)
        print(f"DOGRULAMA GECTI ({veri}):", dog)


def K_SON(maskeler):
    return len(maskeler) - 1


K_BLOK = O.K_BLOK

if __name__ == "__main__":
    main()
