#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kat_dogrula.py -- FAZ II P1 kat protokolunun CPU saglamasi + naif Repeat val referansi
(makale icin sonradan eklendi -- deney-gelistirici, 2026-10-14).

Her veri seti (--veri; varsayilan exchange) x kat {1,2,3} x ufuk {96,192,336,720}, resmi .sh argumanlariyla:
 (a) k3 == resmi: rakip_kos.kat_veri_sinifi(Dataset_*, k=3) train/val data_x, data_stamp, len, scaler mean/scale
     resmi Dataset_Custom / Dataset_ETT_hour ile np.array_equal (bit-bit). Biri KALIRSA rc=1.
 (b) satir araliklari, pencere sayilari (train/val), scaler istatistikleri (kat train'inden; bagimsiz numpy
     yeniden hesabi ile esitlik), olcekli veri == (ham - mu)/sd (bagimsiz).
 (c) sizinti: train pencerelerinin tum satirlari < val_bas; scaler satirlari < val_bas; val hedef satirlari
     [val_bas, val_son) icinde; test dilimi 0 pencere ve __getitem__ hata veriyor; kullanilan en buyuk satir < resmi
     test baslangici (val_son <= T+V).
 (d) naif Repeat (son girdi degeri ufuk boyunca) val MSE/MAE (olcekli uzay, tum pencereler, resmi metrik tanimi)
     -> sonuclar/naif-kat.csv; exchange k3 degerleri data/naive_val.csv (K2 veri hazirlayici) ile |fark| < 1e-6.
Test verisi kullanilmaz (test bayragi yalniz 'bos mu' diye kurulur). Cikti: stdout + sonuclar/kat-dogrula[_<veri>].log
Kullanim (WS kokunden): env/.venv/Scripts/python.exe scripts/kat_dogrula.py [--veri exchange] [--katlar 1,2,3]
  --katlar (tur 4 eki, deney-yurutucu 2026-10-14): varsayilan 1,2,3 = eski davranis ve eski cikti adlari (birebir).
  Baska bir kume (orn. 0 = H12/H13 ic modeli K0) verilirse cikti adlarina _k<katlar> eki gelir
  (sonuclar/kat-dogrula[_<veri>]_k0.log, naif-kat[_<veri>]_k0.csv); mevcut dosyalar degismez.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
WS = HERE.parent
sys.path.insert(0, str(HERE))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TABAN_ANAHTAR = {"exchange": "custom", "etth1": "ETTh1", "etth2": "ETTh2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", choices=sorted(rakip_kos.VERILER), default="exchange")
    ap.add_argument("--katlar", default="1,2,3")
    a = ap.parse_args()
    katlar = tuple(int(x) for x in a.katlar.split(","))
    rakip_kos.AKTIF = V = rakip_kos.VERILER[a.veri]
    csv_yol = V["dizin"] / V["csv"]
    if not csv_yol.exists():
        print(f"[kat_dogrula] {a.veri}: veri yok ({csv_yol}) -- atlandi", flush=True)
        return 3
    ek = "" if a.veri == "exchange" else f"_{a.veri}"
    if katlar != (1, 2, 3):
        ek += "_k" + "".join(str(k) for k in katlar)
    log = open(WS / "sonuclar" / f"kat-dogrula{ek}.log", "w", encoding="utf-8")

    def yaz(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    from data_provider.data_factory import data_dict
    Taban = data_dict[TABAN_ANAHTAR[a.veri]]
    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    sha = rakip_kos.sha256(csv_yol)
    yaz(f"[kat_dogrula] veri={a.veri} csv={csv_yol} sha256={sha} (sabit: {V['sha256']}) taban={Taban.__name__}")
    if V["sha256"] is not None and sha != V["sha256"]:
        yaz("HATA: hash uyusmuyor"); return 1

    # bagimsiz ham veri (sutun duzeni resmi siniftaki gibi)
    df = pd.read_csv(csv_yol)
    if V["bolme"] == "custom":
        cols = [c for c in df.columns if c not in ("date", "OT")] + ["OT"]
    else:
        cols = list(df.columns[1:])
    ham = df[cols].values.astype(np.float64)
    n = len(df)

    ok = True
    satirlar = []
    naif_ref = {}
    if a.veri == "exchange":
        nv = pd.read_csv(WS / "data" / "naive_val.csv")
        for _, r in nv[nv.yontem == "repeat_last"].iterrows():
            naif_ref[int(r.ufuk)] = (float(r.mse), float(r.mae), int(r.pencere))
    else:  # P2 (rakip-kurucu, 2026-10-14): ETT K3 (= resmi val) naif degerleri veri-hazirlayici CSV'siyle (6 hane)
        nv = pd.read_csv(WS / "data" / f"naive_val_{a.veri}.csv")
        for _, r in nv[(nv.yontem == "repeat_last") & (nv.kat == "K3")].iterrows():
            naif_ref[int(r.ufuk)] = (float(r.mse), float(r.mae), int(r.pencere))
    naif_tol = 1e-6 if a.veri == "exchange" else 5e-7 + 1e-9  # ETT CSV'si 6 ondalik haneye yuvarli

    for kat in katlar:
        T, Vlen, vb, vs = rakip_kos.kat_sinirlari(V["bolme"], n, kat)
        Kat = rakip_kos.kat_veri_sinifi(Taban, V["bolme"], kat)
        yaz(f"\n=== kat K{kat}: n={n} T_resmi={T} V={Vlen} | train satir [0,{vb}) | val hedef satir [{vb},{vs}) | "
            f"val girdi satir [{vb - 96},{vs}) | resmi test bas={T + Vlen}")
        mu = ham[:vb].mean(0); sd = ham[:vb].std(0)  # StandardScaler: ddof=0
        for H in rakip_kos.UFUKLAR:
            argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(V["dizin"]) + os.sep)
            argv = rakip_kos.argv_degistir(argv, "--data_path", V["csv"])
            g = parser.parse_args(argv)
            kw = dict(root_path=g.root_path, data_path=g.data_path, size=[g.seq_len, g.label_len, g.pred_len],
                      features=g.features, target=g.target, timeenc=0 if g.embed != 'timeF' else 1, freq=g.freq)
            L = g.seq_len
            tr, va, te = (Kat(flag=f, **kw) for f in ("train", "val", "test"))
            # (a) k3 == resmi
            if kat == 3:
                r_tr, r_va = Taban(flag="train", **kw), Taban(flag="val", **kw)
                es = all([np.array_equal(tr.data_x, r_tr.data_x), np.array_equal(tr.data_stamp, r_tr.data_stamp),
                          np.array_equal(va.data_x, r_va.data_x), np.array_equal(va.data_stamp, r_va.data_stamp),
                          len(tr) == len(r_tr), len(va) == len(r_va),
                          np.array_equal(tr.scaler.mean_, r_tr.scaler.mean_),
                          np.array_equal(tr.scaler.scale_, r_tr.scaler.scale_)])
                yaz(f"  h{H} (a) K3 == resmi {Taban.__name__} (train/val data_x, data_stamp, len, scaler): "
                    f"{'BIT-BIT ESIT' if es else 'FARKLI'}")
                ok &= es
            # (b) satir/pencere/scaler
            n_tr, n_va = len(tr), len(va)
            n_tr_bek, n_va_bek = vb - L - H + 1, Vlen - H + 1
            sc_ok = np.allclose(tr.scaler.mean_, mu, rtol=0, atol=1e-12) and \
                np.allclose(tr.scaler.scale_, sd, rtol=1e-12, atol=0) and \
                np.array_equal(tr.scaler.mean_, va.scaler.mean_) and np.array_equal(tr.scaler.scale_, va.scaler.scale_)
            x_ok = np.allclose(va.data_x, (ham[vb - L:vs] - mu) / sd, rtol=0, atol=1e-10) and \
                np.allclose(tr.data_x, (ham[:vb] - mu) / sd, rtol=0, atol=1e-10)
            b_ok = (n_tr == n_tr_bek and n_va == n_va_bek and tr.kat_bilgi["satir"] == [0, vb]
                    and va.kat_bilgi["satir"] == [vb - L, vs] and len(va.data_x) == Vlen + L)
            # (c) sizinti
            tr_maks = (n_tr - 1) + L + H - 1           # son train penceresinin son hedef satiri
            va_hedef = (vb - L) + L + np.arange(n_va)  # her val penceresinin ilk hedef satiri
            va_hedef_son = va_hedef[-1] + H - 1
            try:
                te[0]; te_kilit = False
            except RuntimeError:
                te_kilit = True
            s_ok = (tr_maks < vb and va_hedef[0] == vb and va_hedef_son == vs - 1 and len(te) == 0 and te_kilit
                    and vs <= T + Vlen)
            # (d) naif Repeat
            X = np.stack([va[i][0] for i in range(n_va)]); Y = np.stack([va[i][1][-H:] for i in range(n_va)])
            P = np.repeat(X[:, -1:, :], H, axis=1)
            mse = float(np.mean((P - Y) ** 2)); mae = float(np.mean(np.abs(P - Y)))
            ref_s = ""
            if kat == 3 and H in naif_ref:
                rm, ra, rp = naif_ref[H]
                r_ok = abs(mse - rm) < naif_tol and abs(mae - ra) < naif_tol and rp == n_va
                nv_ad = "naive_val.csv" if a.veri == "exchange" else f"naive_val_{a.veri}.csv"
                ref_s = f" | data/{nv_ad} {rm:.6f}/{ra:.6f} -> {'ESIT' if r_ok else 'FARKLI'}"
                ok &= r_ok
            yaz(f"  h{H}: pencere train {n_tr} (bek {n_tr_bek}) val {n_va} (bek {n_va_bek}) test {len(te)} | "
                f"satir/pencere {'OK' if b_ok else 'HATA'} | scaler==numpy(train) {'OK' if sc_ok else 'HATA'} | "
                f"olcek {'OK' if x_ok else 'HATA'} | sizinti yok (train son hedef {tr_maks} < {vb}; val hedef "
                f"[{va_hedef[0]},{va_hedef_son}]; test 0 pencere, okuma hatasi={te_kilit}) {'OK' if s_ok else 'HATA'}"
                f" | naif Repeat val mse {mse:.6f} mae {mae:.6f}{ref_s}")
            ok &= bool(b_ok and sc_ok and x_ok and s_ok)
            satirlar.append({"veri": a.veri, "kat": kat, "ufuk": H, "seq_len": L, "train_satir_son": vb,
                             "val_satir_bas": vb, "val_satir_son": vs, "pencere_train": n_tr, "pencere_val": n_va,
                             "yontem": "repeat_last", "mse": f"{mse:.6f}", "mae": f"{mae:.6f}"})
        yaz(f"  scaler ort (kat train) = {np.array2string(mu, precision=5, max_line_width=200)}")
        yaz(f"  scaler std (kat train) = {np.array2string(sd, precision=5, max_line_width=200)}")

    # kat basina 4-ufuk ortalama
    yaz("\nnaif Repeat val MSE (4 ufuk ort.): " + ", ".join(
        f"K{k}={np.mean([float(r['mse']) for r in satirlar if r['kat'] == k]):.6f}" for k in katlar))
    cikti = WS / "sonuclar" / f"naif-kat{ek}.csv"
    with open(cikti, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader(); w.writerows(satirlar)
    yaz(f"yazildi: {cikti}")
    yaz(f"SONUC: {'GECTI' if ok else 'KALDI'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
