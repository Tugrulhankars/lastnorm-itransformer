# -*- coding: utf-8 -*-
"""Makale -- nitel tahmin paneli VERISI (cizim degil): ornek pencerelerde girdi + gercek + iTransformer + LastNorm + Repeat.
(makale icin sonradan eklendi -- deney-gelistirici, 2026-10-15; CPU son-islem, egitim/GPU yok)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_nitel_panel.py

SECIM OLCUTU (koddan ONCE sabitlendi; sonuca bakilarak degistirilmez; saman-adam/bilgisiz panel yasagi):
  - Kaynak: TEST dilimi, seed 2023 kosumlari (iTr 020531/230036.., LastNorm 052508/..), H in {96, 336}.
  - Exchange: bloklar Q1..Q4 (splits JSON test ceyrekleri). Pencere i, tahmin KOKENI (ilk hedef satiri hedef_ilk_satir[i])
    Qk araligina dusuyorsa Qk'ya aittir.
  - ETTh2: tek panel / ufuk, tum test pencereleri (blok = TUM).
  - Pencere-farki d_i = MSE_LastNorm(i) - MSE_iTr(i) (pencere icindeki tum adim x degisken ortalamasi, z uzayi,
    preds npz 'mse_pencere_degisken' ortalamasi). Secilen pencere = d_i'si blok medyanina EN YAKIN pencere
    (bagda kucuk indeks). Boylece panel ne LastNorm'un en iyi ne en kotu oldugu orneklerdir -- tipik orneklerdir.
  - Degisken = secilen pencerenin girdi+hedef araliginda (L+H satir) z-varyansi EN BUYUK degisken.
  - Degerler standardize (train-only z), ham olcek DEGIL.

Cikti: sonuclar/makale/nitel-panel.csv (uzun bicim; '#' basliginda secim olcutu + secilen pencereler tablosu)
  veri,ufuk,blok,pencere,hedef_ilk_satir,degisken,d_secilen,d_blok_medyan,n_blok_pencere,t,faz,gercek,itr,lastnorm,repeat
  (t = -95..0 girdi [0 = son gozlem], 1..H hedef; girdi satirlarinda model sutunlari bos)
DOGRULAMA (assert): secilen pencerede npz'den yeniden hesaplanan MSE'ler mse_pencere_degisken ile esit (1e-5); hedef
  .npy == CSV z satirlari; Repeat == girdi son satiri.
"""
import sys

import numpy as np

import makale_ortak as O

sys.stdout.reconfigure(encoding="utf-8")
SEED_I = 0  # seed 2023
PANEL_UFUK = [96, 336]
PANEL_VERI = {"exchange": "bloklar", "etth2": "tum"}


def main():
    O.kosumlari_dogrula()
    satirlar, ozet = [], []
    for veri, mod in PANEL_VERI.items():
        z, cols = O.yukle_z(veri)
        r_itr, r_ln = O.RUNS[veri]["itr"][SEED_I], O.RUNS[veri]["lastnorm"][SEED_I]
        for H in PANEL_UFUK:
            fi, fl = O.yukle_pred(r_itr, "test", H), O.yukle_pred(r_ln, "test", H)
            ilk = fi["hedef_ilk_satir"].astype(np.int64)
            assert np.array_equal(ilk, fl["hedef_ilk_satir"])
            Y = O.yukle_true(veri, "test", H)
            R, X, Yz = O.repeat_ve_girdi(z, ilk, H)
            assert np.abs(Yz - Y).max() < 1e-4
            mi = fi["mse_pencere_degisken"].astype(np.float64).mean(1)
            ml = fl["mse_pencere_degisken"].astype(np.float64).mean(1)
            d = ml - mi
            if mod == "bloklar":
                gruplar = [(ad, (ilk >= b0) & (ilk < b1)) for ad, (b0, b1) in
                           zip(O.blok_adlari("test"), O.blok_sinirlari(veri, "test"))]
            else:
                gruplar = [("TUM", np.ones_like(ilk, bool))]
            for ad, msk in gruplar:
                idx = np.where(msk)[0]
                if len(idx) == 0:
                    continue
                med = float(np.median(d[idx]))
                i = int(idx[np.argmin(np.abs(d[idx] - med))])  # argmin ilk eslesmeyi verir -> bagda kucuk indeks
                seg = np.concatenate([X[i], Y[i]], 0)  # (L+H, C)
                j = int(np.argmax(seg.var(0)))
                pi, pl = fi["pred"][i].astype(np.float64), fl["pred"][i].astype(np.float64)
                assert abs(((pi - Y[i]) ** 2).mean() - mi[i]) < 1e-5 and abs(((pl - Y[i]) ** 2).mean() - ml[i]) < 1e-5
                assert np.allclose(R[i], X[i, -1][None, :])
                ozet.append(f"{veri} H={H} blok={ad} n_pencere={len(idx)} secilen_pencere={i} hedef_ilk_satir={ilk[i]} "
                            f"degisken={cols[j]} d={d[i]:+.5f} blok_medyan_d={med:+.5f} | pencere MSE (tum degisken) "
                            f"itr={mi[i]:.4f} ln={ml[i]:.4f} rep={((R[i] - Y[i]) ** 2).mean():.4f}")
                ortak = dict(veri=veri, ufuk=H, blok=ad, pencere=i, hedef_ilk_satir=int(ilk[i]), degisken=cols[j],
                             d_secilen=round(float(d[i]), 6), d_blok_medyan=round(med, 6), n_blok_pencere=len(idx))
                for t in range(O.SEQ):
                    satirlar.append(dict(**ortak, t=t - O.SEQ + 1, faz="girdi", gercek=O.r6(X[i, t, j]),
                                         itr="", lastnorm="", repeat=""))
                for h in range(H):
                    satirlar.append(dict(**ortak, t=h + 1, faz="hedef", gercek=O.r6(Y[i, h, j]), itr=O.r6(pi[h, j]),
                                         lastnorm=O.r6(pl[h, j]), repeat=O.r6(R[i, h, j])))
    for s in ozet:
        print(s)
    meta = O.meta_satirlari(
        "makale_nitel_panel.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_nitel_panel.py",
        ["SECIM OLCUTU (onceden sabit): TEST, seed 2023, H in {96,336}. Exchange: tahmin kokeni (ilk hedef satiri) Qk'da "
         "olan pencereler arasinda, d_i = MSE_LastNorm(i) - MSE_iTr(i) (pencere icindeki tum adim x degisken ort.) "
         "blok MEDYANINA en yakin pencere (bagda kucuk indeks); ETTh2: tum test pencereleri icinde ayni kural. "
         "Degisken = secilen pencerenin girdi+hedef (L+H satir) araliginda z-varyansi en buyuk degisken. "
         "Degerler standardize (train-only z), ham olcek degil. En iyi/en kotu ornek SECILMEDI.",
         "t: -95..0 girdi (0 = son gozlem = Repeat seviyesi), 1..H hedef; girdi satirlarinda model sutunlari bos"]
        + ["secilen: " + s for s in ozet],
        [O.RUNS[v][m][SEED_I] for v in PANEL_VERI for m in ("itr", "lastnorm")])
    O.csv_yaz(O.OUT / "nitel-panel.csv", satirlar, meta)
    print(f"DOGRULAMA GECTI: {len(ozet)} panel")


if __name__ == "__main__":
    main()
