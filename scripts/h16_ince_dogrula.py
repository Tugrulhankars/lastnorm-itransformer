#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""h16_ince_dogrula.py -- FAZ III/H16 'ince kayit' yolunun eski yolla bit-bit esdegerligi (CPU, GPU yok).

Gercek hedef dizilerinin (data/hedefler_sl96_<veri>/val_h<H>_true.npy) bir alt kumesi + sabit tohumlu sentetik tahmin
(true + gurultu, float32) uzerinde:
  (1) pencere_kaydet (eski, tam dizi) ile pencere_kaydet_ince (parca parca, pred'siz): ortak anahtarlar np.array_equal;
      girdi hem bellek ici hem np.load(mmap_mode='r') (test yolu) ile; alt kume PARCA'nin kati olmayan uzunlukta.
  (2) hedef_dogrula_yaz ile hedef_dogrula_yaz_ince: ayni dosyaya karsi ayni karar (gecer / 1e-5 bozulmada durur).
  (3) metrik ifadesi (np.mean((p-t)**2), float32) mmap girdide bellek ici ile ayni sayi.
Kullanim (WS kokunden): env/.venv/Scripts/python.exe -I scripts/h16_ince_dogrula.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos as rk  # noqa: E402

N_ALT = 2 * rk.PARCA + 77  # parca sinirini iki kez gecer, son parca eksik


def bir_durum(veri, H, tmp: Path):
    f = rk.VERILER[veri]["hedef"] / f"val_h{H}_true.npy"
    t_full = np.load(f, mmap_mode="r")
    t = np.array(t_full[:N_ALT], dtype=np.float32)
    rng = np.random.default_rng(20261015 + H)
    p = (t + rng.normal(0, 0.3, size=t.shape)).astype(np.float32)
    np.save(tmp / "p.npy", p); np.save(tmp / "t.npy", t)
    pm = np.load(tmp / "p.npy", mmap_mode="r"); tm = np.load(tmp / "t.npy", mmap_mode="r")

    # (1)
    rk.pencere_kaydet(tmp / "eski.npz", p, t, 1234, np)
    rk.pencere_kaydet_ince(tmp / "ince_bellek.npz", p, t, 1234, np)
    rk.pencere_kaydet_ince(tmp / "ince_mmap.npz", pm, tm, 1234, np)
    e = np.load(tmp / "eski.npz")
    for ad in ("ince_bellek", "ince_mmap"):
        y = np.load(tmp / f"{ad}.npz")
        assert set(y.files) == set(e.files) - {"pred"}, (y.files, e.files)
        for k in y.files:
            assert y[k].dtype == e[k].dtype and np.array_equal(y[k], e[k]), (veri, H, ad, k)
        y.close()
    e.close()

    # (2) ayni dosyaya karsi karar: gecer
    d = tmp / "hedef"; d.mkdir(exist_ok=True)
    np.save(d / f"val_h{H}_true.npy", t)
    rk.hedef_dogrula_yaz("val", H, t, np, d)
    rk.hedef_dogrula_yaz_ince("val", H, tm, np, d)
    # bozulmus true (son pencerede 1e-5) -> ikisi de durur
    tb = t.copy(); tb[-1, -1, -1] += 1e-5
    for fn in (rk.hedef_dogrula_yaz, rk.hedef_dogrula_yaz_ince):
        try:
            fn("val", H, tb, np, d)
            raise AssertionError(f"{fn.__name__} bozulmayi yakalamadi")
        except SystemExit:
            pass

    # (3) metrik ifadesi mmap'te ayni
    a = float(np.mean((p - t) ** 2)); b = float(np.mean((pm - tm) ** 2))
    assert a == b, (a, b)
    del pm, tm
    return f"{veri} h{H}: N={N_ALT} C={t.shape[2]} -> pencere 4 anahtar bit-bit (bellek+mmap), hedef karari ayni, mse {a:.6f} ayni"


def main():
    satirlar = []
    with tempfile.TemporaryDirectory() as td:
        for veri in ("ettm1", "weather"):
            for H in (96, 720):
                sd = Path(td) / f"{veri}_{H}"; sd.mkdir()
                satirlar.append(bir_durum(veri, H, sd))
    for s in satirlar:
        print("[h16_ince_dogrula] GECTI", s)
    print("[h16_ince_dogrula] TUMU GECTI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
