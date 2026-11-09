# -*- coding: utf-8 -*-
"""Makale F2 -- ufuk-adim hata egrisi (val VE test): iTransformer vs LastNorm vs Repeat, her H icin adim basina MSE.
(makale icin sonradan eklendi -- deney-gelistirici, 2026-10-15; CPU son-islem, egitim/GPU yok)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_adim_egrisi.py [--veri exchange etth1 etth2]

Tanim: adim h (1..H) icin MSE_h = tum pencereler ve tum degiskenler uzerinden (pred - gercek)^2 ortalamasi (z uzayi,
train-only scaler). Model: kosumun kaydettigi preds/*.npz 'mse_pencere_adim' (N,H) -> pencere ortalamasi; seed basina,
sonra 3 seed ort. ve sd (ddof=1). Repeat: ham CSV + splits JSON scaler'i ile yeniden kurulur (seed'den bagimsiz).

Cikti: sonuclar/makale/adim-egrisi.csv
  veri,bolum,ufuk,adim,itr_ort,itr_sd,itr_s2023,itr_s2024,itr_s2025,lastnorm_ort,lastnorm_sd,lastnorm_s2023,
  lastnorm_s2024,lastnorm_s2025,repeat,itr_M_R,lastnorm_M_R,d_ln_itr_pct

DOGRULAMA (assert; biri tutmazsa betik durur):
  (a) her kosum/ufuk: adim ortalamalarinin ortalamasi == test_metrics.json test_mse_hH / metrics.json val_mse_hH (1e-5)
  (b) mse_pencere_adim == pred ve hedef .npy'den yeniden hesap (1e-5); hedef .npy == ham CSV z satirlari (1e-4)
  (c) Repeat val ufuk-ort. == sonuclar/naif-kat*.csv K3 (6 ondalik; 1e-6); Repeat test Exchange == sonuclar/naif-test.csv
      (1e-6); Repeat test ETT == sonuclar/vsel-test.csv test_repeat (1e-6)
  (d) Exchange val 10 adim-dilimi seed-ort. model MSE == sonuclar/hata-adim.csv (baseline) ve hata-adim_lastnorm.csv (1e-6)
"""
import argparse
import csv
import sys

import numpy as np

import makale_ortak as O

sys.stdout.reconfigure(encoding="utf-8")


def naif_val(veri):
    ad = {"exchange": "naif-kat.csv", "etth1": "naif-kat_etth1_k3.csv", "etth2": "naif-kat_etth2_k3.csv"}[veri]
    with open(O.WS / "sonuclar" / ad, encoding="utf-8") as f:
        return {int(r["ufuk"]): float(r["mse"]) for r in csv.DictReader(f) if r["kat"] == "3"}


def naif_test(veri):
    if veri == "exchange":
        with open(O.WS / "sonuclar/naif-test.csv", encoding="utf-8") as f:
            return {int(r["ufuk"]): float(r["test_mse"]) for r in csv.DictReader(f)}
    with open(O.WS / "sonuclar/vsel-test.csv", encoding="utf-8") as f:
        rr = [r for r in csv.DictReader(f) if r["veri"] == veri]
    out = {}
    for r in rr:
        out.setdefault(int(r["H"]), set()).add(round(float(r["test_repeat"]), 9))
    assert all(len(v) == 1 for v in out.values())
    return {k: v.pop() for k, v in out.items()}


def hata_adim(etiket):
    ad = "hata-adim.csv" if etiket == "itr" else "hata-adim_lastnorm.csv"
    with open(O.WS / "sonuclar" / ad, encoding="utf-8") as f:
        return [(int(r["ufuk"]), int(r["adim_bas"]), int(r["adim_bit"]), float(r["model_mse"]), float(r["repeat_mse"]))
                for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", nargs="+", default=["exchange", "etth1", "etth2"])
    a = ap.parse_args()
    O.kosumlari_dogrula()
    satirlar, dog = [], {"a": 0, "b": 0, "c": 0, "d": 0}
    for veri in a.veri:
        z, _ = O.yukle_z(veri)
        nv, nt = naif_val(veri), naif_test(veri)
        for bolum in ("val", "test"):
            for H in O.UFUKLAR:
                Y = O.yukle_true(veri, bolum, H)
                adim = {}
                ilk0 = None
                for m in ("itr", "lastnorm"):
                    per = []
                    for rid in O.RUNS[veri][m]:
                        f = O.yukle_pred(rid, bolum, H)
                        ilk = f["hedef_ilk_satir"].astype(np.int64)
                        if ilk0 is None:
                            ilk0 = ilk
                        assert np.array_equal(ilk, ilk0)
                        mpa = f["mse_pencere_adim"].astype(np.float64)
                        e = ((f["pred"].astype(np.float64) - Y) ** 2).mean(axis=2)  # (N,H)
                        assert np.abs(e - mpa).max() < 1e-5, (rid, H, "mse_pencere_adim")
                        dog["b"] += 1
                        s = mpa.mean(0)  # (H,)
                        kay = O.kayitli_mse(rid, bolum, H)
                        assert abs(s.mean() - kay) < 1e-5, (rid, bolum, H, s.mean(), kay)
                        dog["a"] += 1
                        per.append(s)
                    adim[m] = np.stack(per)  # (3,H)
                R, _, Yz = O.repeat_ve_girdi(z, ilk0, H)
                assert np.abs(Yz - Y).max() < 1e-4, (veri, bolum, H, "hedef/CSV")
                r_step = ((R - Y) ** 2).mean(axis=(0, 2))
                ref = (nv if bolum == "val" else nt)[H]
                assert abs(r_step.mean() - ref) < 1e-6 + 5e-7, (veri, bolum, H, r_step.mean(), ref)
                dog["c"] += 1
                if veri == "exchange" and bolum == "val":
                    for m in ("itr", "lastnorm"):
                        sm = adim[m].mean(0)
                        for (u, b0, b1, mm, rm) in hata_adim(m):
                            if u != H:
                                continue
                            assert abs(sm[b0 - 1:b1].mean() - mm) < 1.5e-6, (m, H, b0, sm[b0 - 1:b1].mean(), mm)
                            assert abs(r_step[b0 - 1:b1].mean() - rm) < 1.5e-6, (H, b0)
                            dog["d"] += 1
                for h in range(H):
                    it, ln = adim["itr"][:, h], adim["lastnorm"][:, h]
                    satirlar.append(dict(
                        veri=veri, bolum=bolum, ufuk=H, adim=h + 1,
                        itr_ort=O.r6(it.mean()), itr_sd=O.r6(it.std(ddof=1)),
                        **{f"itr_s{sd}": O.r6(it[i]) for i, sd in enumerate(O.SEEDS)},
                        lastnorm_ort=O.r6(ln.mean()), lastnorm_sd=O.r6(ln.std(ddof=1)),
                        **{f"lastnorm_s{sd}": O.r6(ln[i]) for i, sd in enumerate(O.SEEDS)},
                        repeat=O.r6(r_step[h]), itr_M_R=round(float(it.mean() / r_step[h]), 4),
                        lastnorm_M_R=round(float(ln.mean() / r_step[h]), 4),
                        d_ln_itr_pct=round(float((ln.mean() / it.mean() - 1) * 100), 3)))
                print(f"{veri} {bolum} H={H}: itr {adim['itr'].mean():.4f} ln {adim['lastnorm'].mean():.4f} "
                      f"rep {r_step.mean():.4f} | adim1 itr/ln/rep {adim['itr'][:, 0].mean():.4f}/"
                      f"{adim['lastnorm'][:, 0].mean():.4f}/{r_step[0]:.4f} | adimH {adim['itr'][:, -1].mean():.4f}/"
                      f"{adim['lastnorm'][:, -1].mean():.4f}/{r_step[-1]:.4f}")
    meta = O.meta_satirlari(
        "makale_adim_egrisi.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_adim_egrisi.py",
        ["tanim: adim h MSE = pencereler x degiskenler uzerinden ortalama karesel hata (z uzayi, train-only scaler); "
         "model = seed basina mse_pencere_adim pencere-ort., *_ort/*_sd = 3 seed ort./sd(ddof=1); repeat seed'siz",
         "d_ln_itr_pct = 100*(lastnorm_ort/itr_ort - 1); *_M_R = model_ort/repeat",
         f"dogrulama: (a) {dog['a']} kosum-ufuk adim-ort==kayitli MSE (1e-5); (b) {dog['b']} mse_pencere_adim==yeniden hesap; "
         f"(c) {dog['c']} Repeat ufuk-ort.==naif-kat*/naif-test/vsel-test; (d) {dog['d']} Exchange val dilim==hata-adim*.csv"],
        O.tum_run_ids(a.veri))
    O.csv_yaz(O.OUT / "adim-egrisi.csv", satirlar, meta)
    print("DOGRULAMA GECTI:", dog)


if __name__ == "__main__":
    main()
