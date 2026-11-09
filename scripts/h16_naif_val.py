#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""h16_naif_val.py -- FAZ III/H16 beklenti araliklari icin VAL naif referanslari (CPU; test OKUNMAZ).

Her veri (ettm1, ettm2, weather) ve H in {96,192,336,720}: resmi Dataset sinifi (data_dict[--data], flag='val', .sh
argumanlari) -> data_y (train-z olcekli). Pencere i: girdi son satiri data_y[i+L-1], hedef data_y[i+L : i+L+H].
  Repeat MSE   = ort (hedef - girdi_son)^2           (Zeng 2023 'Repeat' tanimi)
  Train-ort MSE = ort hedef^2                          (z uzayinda train ortalamasi = 0)
Hedefler data/hedefler_sl96_<veri>/val_h<H>_true.npy ile tolerans 0 karsilastirilir (hizalama saglamasi).
Cikti: data/naive_val_<veri>.csv (ufuk, repeat_mse, trainort_mse, pencere). Ayni bicim: data/naive_val_etth1.csv emsali.
Kullanim (WS kokunden): env/.venv/Scripts/python.exe -I scripts/h16_naif_val.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos as rk  # noqa: E402

sys.path.insert(0, str(rk.REPO))


def main():
    from data_provider.data_factory import data_dict
    for veri in ("ettm1", "ettm2", "weather"):
        rk.AKTIF = rk.VERILER[veri]
        assert rk.sha256(rk.AKTIF["dizin"] / rk.AKTIF["csv"]) == rk.AKTIF["sha256"]
        parser, _ = rk.resmi_parser()
        sh = rk.resmi_sh_argumanlari()
        satirlar = []
        for H in rk.UFUKLAR:
            a = parser.parse_args(rk.argv_degistir(rk.argv_degistir(sh[H], "--root_path", str(rk.AKTIF["dizin"]) + "/"),
                                                   "--data_path", rk.AKTIF["csv"]))
            ds = data_dict[a.data](root_path=a.root_path, data_path=a.data_path, flag="val",
                                   size=[a.seq_len, a.label_len, a.pred_len], features=a.features, target=a.target,
                                   timeenc=0 if a.embed != "timeF" else 1, freq=a.freq)
            y = np.asarray(ds.data_y, dtype=np.float64)
            L, N = a.seq_len, len(ds)
            ref = np.load(rk.AKTIF["hedef"] / f"val_h{H}_true.npy", mmap_mode="r")
            assert ref.shape[0] == N, (ref.shape, N)
            rep = tro = 0.0
            for i in range(N):
                t = y[i + L:i + L + H]
                if i % 997 == 0 or i == N - 1:  # hizalama: resmi float32 hedefle birebir
                    assert np.array_equal(t.astype(np.float32), np.asarray(ref[i])), (veri, H, i)
                rep += float(((t - y[i + L - 1]) ** 2).sum())
                tro += float((t ** 2).sum())
            n_el = N * H * y.shape[1]
            satirlar.append({"ufuk": H, "repeat_mse": rep / n_el, "trainort_mse": tro / n_el, "pencere": N})
            print(f"[naif_val] {veri} h{H}: Repeat {rep / n_el:.4f} train-ort {tro / n_el:.4f} (N={N})", flush=True)
        ort = {"ufuk": "ort", "repeat_mse": float(np.mean([s["repeat_mse"] for s in satirlar])),
               "trainort_mse": float(np.mean([s["trainort_mse"] for s in satirlar])), "pencere": ""}
        print(f"[naif_val] {veri} ort: Repeat {ort['repeat_mse']:.4f} train-ort {ort['trainort_mse']:.4f}", flush=True)
        with open(WS / "data" / f"naive_val_{veri}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ufuk", "repeat_mse", "trainort_mse", "pencere"])
            w.writeheader()
            w.writerows(satirlar + [ort])
    return 0


if __name__ == "__main__":
    sys.exit(main())
