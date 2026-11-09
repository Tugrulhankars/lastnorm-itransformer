# -*- coding: utf-8 -*-
"""Makale -- kayma dayaniklilik metrikleri, TEST: iTransformer vs LastNorm vs Repeat.
(makale icin sonradan eklendi -- 2026-10-29; CPU son-islem, egitim/GPU yok)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_dayaniklilik.py [--veri exchange etth1 etth2]

TANIMLAR: sonuclar/hata-analizi.md s3.1'de test acilmadan ONCE sabitlenen ve val'de on-uygulanan tanimlar, aynen:
  - Blok: dilimin kendi satirlari np.array_split K=4 (test: Q1..Q4); hucre HEDEF satirina gore bloga atanir.
    worst_block_mse = max_k MSE_k ; worst_ratio = worst_block_mse / tum-hucre MSE.
  - s_io(i) = mean_d |mu(horizon_i,d) - mu(lookback_i,d)| / (sd(lookback_i,d) + 0.01)  (z uzayi, ddof=0).
    rho_shift = Spearman(s_io, pencere MSE) (seed basina, sonra ortalama); top10_ratio = s_io ust %10 pencere
    MSE'si / tum pencere MSE'si. Pencereler ortustugunden Spearman p-degeri RAPORLANMAZ.
  - Ek (betimsel): Exchange test ceyrekleri icin train->ceyrek kayma olcusu MMD (data/kayma_mmd.csv,
    mmd_b_z_sigmaMed sutunu; K2'de, test acilmadan once hesaplandi) ve ceyrek basina LastNorm-iTransformer farki.

Cikti: sonuclar/makale/dayaniklilik.csv, sonuclar/makale/dayaniklilik-ceyrek.csv

DOGRULAMA (assert):
  (a) tum-hucre MSE her kosum icin == test_metrics.json test_mse_hH (1e-5)
  (b) ceyrek MSE (seed-ort.) == sonuclar/makale/hata-kirilim_<veri>.csv TUM satirlari (itr_ort/lastnorm_ort/repeat; 2e-6)
"""
import argparse
import csv
import sys

import numpy as np
from scipy import stats

import makale_ortak as O

sys.stdout.reconfigure(encoding="utf-8")
EPS = 0.01
MODELLER = ("itr", "lastnorm", "repeat")


def kirilim_tum(veri):
    with open(O.OUT / f"hata-kirilim_{veri}.csv", encoding="utf-8") as f:
        satir = [x for x in f if not x.startswith("#")]
    return {(int(r["ufuk"]), r["blok"]): r for r in csv.DictReader(satir)
            if r["bolum"] == "test" and r["degisken"] == "TUM"}


def mmd_ceyrek():
    with open(O.WS / "data/kayma_mmd.csv", encoding="utf-8") as f:
        return {r["dilim"]: float(r["mmd_b_z_sigmaMed"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", nargs="+", default=["exchange", "etth1", "etth2"])
    a = ap.parse_args()
    O.kosumlari_dogrula()
    ozet, ceyrek = [], []
    dog = {"a": 0, "b": 0}
    for veri in a.veri:
        z, _ = O.yukle_z(veri)
        sinir = O.blok_sinirlari(veri, "test")
        ref = kirilim_tum(veri)
        for H in O.UFUKLAR:
            ilk = O.yukle_pred(O.RUNS[veri]["itr"][0], "test", H)["hedef_ilk_satir"].astype(np.int64)
            trow = ilk[:, None] + np.arange(H)[None, :]
            blok_id = np.full(trow.shape, -1)
            for k, (b0, b1) in enumerate(sinir):
                blok_id[(trow >= b0) & (trow < b1)] = k
            assert (blok_id >= 0).all(), (veri, H)
            R, X, Y = O.repeat_ve_girdi(z, ilk, H)
            Yt = O.yukle_true(veri, "test", H)
            assert np.allclose(Yt, Y, atol=1e-5), (veri, H)
            mu_lb, sd_lb = X.mean(1), X.std(1)
            s_io = np.mean(np.abs(Y.mean(1) - mu_lb) / (sd_lb + EPS), axis=1)
            ust = s_io >= np.quantile(s_io, 0.9)
            hata = {}  # model -> list of (N,H,C) kare hata, seed basina
            for m in ("itr", "lastnorm"):
                hata[m] = []
                for rid in O.RUNS[veri][m]:
                    f = O.yukle_pred(rid, "test", H)
                    assert np.array_equal(f["hedef_ilk_satir"], ilk)
                    e2 = (f["pred"].astype(np.float64) - Y) ** 2
                    assert abs(e2.mean() - O.kayitli_mse(rid, "test", H)) < 1e-5, (rid, H)
                    dog["a"] += 1
                    hata[m].append(e2)
            hata["repeat"] = [(R - Y) ** 2]
            olc = {}
            for m in MODELLER:
                tum = np.array([e.mean() for e in hata[m]])
                blok = np.array([[e.mean(2)[blok_id == k].mean() for k in range(O.K_BLOK)] for e in hata[m]])
                pen = [e.mean(axis=(1, 2)) for e in hata[m]]
                rho = np.array([stats.spearmanr(s_io, p)[0] for p in pen])
                top = np.array([p[ust].mean() / p.mean() for p in pen])
                olc[m] = dict(tum=tum, blok=blok, worst=blok.max(1), ratio=blok.max(1) / tum, rho=rho, top=top,
                              top_mse=np.array([p[ust].mean() for p in pen]))
                for k, ad in enumerate(O.blok_adlari("test")):
                    r = ref[(H, ad)]
                    beklenen = float(r["repeat"] if m == "repeat" else r[f"{m}_ort"])
                    assert abs(blok[:, k].mean() - beklenen) < 2e-6, (veri, H, m, ad, blok[:, k].mean(), beklenen)
                    dog["b"] += 1
            for m in MODELLER:
                o = olc[m]
                ozet.append(dict(
                    veri=veri, ufuk=H, model=m, n_seed=len(o["tum"]), n_pencere=len(s_io), n_ust10=int(ust.sum()),
                    mse=O.r6(o["tum"].mean()),
                    worst_block_mse=O.r6(o["worst"].mean()), worst_block_mse_sd=O.r6(o["worst"].std(ddof=1)) if len(o["tum"]) > 1 else "",
                    worst_blok=O.blok_adlari("test")[int(o["blok"].mean(0).argmax())],
                    worst_ratio=round(float(o["ratio"].mean()), 4),
                    rho_shift=round(float(o["rho"].mean()), 4), rho_shift_sd=round(float(o["rho"].std(ddof=1)), 4) if len(o["tum"]) > 1 else "",
                    top10_ratio=round(float(o["top"].mean()), 4), top10_mse=O.r6(o["top_mse"].mean()),
                    top10_M_R=round(float(o["top_mse"].mean() / olc["repeat"]["top_mse"].mean()), 4),
                    worst_M_R=round(float(o["worst"].mean() / olc["repeat"]["worst"].mean()), 4),
                    d_worst_ln_itr_pct="" if m != "lastnorm" else round(100 * (o["worst"].mean() / olc["itr"]["worst"].mean() - 1), 2),
                    d_top10_ln_itr_pct="" if m != "lastnorm" else round(100 * (o["top_mse"].mean() / olc["itr"]["top_mse"].mean() - 1), 2),
                    worst_ln_iyi_seed="" if m != "lastnorm" else int((o["worst"] < olc["itr"]["worst"]).sum()),
                    top10_ln_iyi_seed="" if m != "lastnorm" else int((o["top_mse"] < olc["itr"]["top_mse"]).sum()),
                ))
            if veri == "exchange":
                mmd = mmd_ceyrek()
                for k, ad in enumerate(O.blok_adlari("test")):
                    it, ln, rp = (olc[m]["blok"][:, k].mean() for m in MODELLER)
                    ceyrek.append(dict(veri=veri, ufuk=H, ceyrek=ad, mmd_train_ceyrek=round(mmd[f"test_{ad}"], 4),
                                       itr=O.r6(it), lastnorm=O.r6(ln), repeat=O.r6(rp),
                                       d_ln_itr_pct=round(100 * (ln / it - 1), 2),
                                       ln_M_R=round(float(ln / rp), 4), itr_M_R=round(float(it / rp), 4)))
        print(f"{veri}: tamam")
    meta = O.meta_satirlari("makale_dayaniklilik.py",
                            "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_dayaniklilik.py",
                            ["tanim: sonuclar/hata-analizi.md s3.1 (test acilmadan once sabitlendi; val on-uygulamasi orada)",
                             "Spearman p-degeri ortusen pencereler nedeniyle verilmez",
                             f"dogrulama: (a) {dog['a']} kosum x ufuk MSE == test_metrics.json; (b) {dog['b']} ceyrek MSE == hata-kirilim_<veri>.csv"],
                            O.tum_run_ids(a.veri))
    O.csv_yaz(O.OUT / "dayaniklilik.csv", ozet, meta)
    if ceyrek:
        O.csv_yaz(O.OUT / "dayaniklilik-ceyrek.csv", ceyrek,
                  meta + ["# mmd_train_ceyrek: data/kayma_mmd.csv mmd_b_z_sigmaMed (train -> test ceyregi, RBF, medyan bant)"])
    print("dogrulama:", dog)


if __name__ == "__main__":
    main()
