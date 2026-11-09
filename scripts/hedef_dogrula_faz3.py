#!/usr/bin/env python
"""FAZ III hedef dosyalari icin rakip_kos.py yolunun bagimsiz saglamasi (veri-hazirlayici, 2026-10-15).

rakip_kos.py'nin val/test 'true' uretimi: resmi Dataset -> DataLoader(batch 1, shuffle yok) -> batch_y.float()
-> batch_y[:, -pred_len:, :]. Bu betik ayni zinciri (resmi data_dict[...] Dataset sinifi + DataLoader(batch 1,
shuffle yok, drop_last=False) + .float(); val ve test icin) orneklenmis pencerelerde calistirir
ve data/hedefler_sl96_<ad>/*.npy ile BIREBIR (tolerans 0) karsilastirir. Model yok, GPU yok; test'te yalniz hedef
dizisi okunur (tahmin/hata hesabi yok).
Kullanim: PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I -X utf8 scripts/hedef_dogrula_faz3.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WS = Path(__file__).resolve().parents[1]
REPO = WS / "rakip" / "iTransformer"
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from data_provider.data_factory import data_dict  # noqa: E402

SETLER = {"ettm1": ("ETTm1", "ETTm1.csv"), "ettm2": ("ETTm2", "ETTm2.csv"),
          "weather": ("custom", "weather.csv"), "solar": ("Solar", "solar_AL.txt")}
ORNEK = 64


def main():
    rng = np.random.default_rng(0)
    toplam = 0
    for ad, (anahtar, csv) in SETLER.items():
        hd = WS / "data" / f"hedefler_sl96_{ad}"
        for flag in ("val", "test"):
            for H in (96, 192, 336, 720):
                args = SimpleNamespace(root_path=str(WS / "data" / ad) + os.sep, data_path=csv, seq_len=96,
                                       label_len=48, pred_len=H, features="M", target="OT", embed="timeF", freq="h")
                ds = data_dict[anahtar](root_path=args.root_path, data_path=args.data_path, flag=flag,
                                        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                                        target=args.target, timeenc=1, freq=args.freq)
                ref = np.load(hd / f"{flag}_h{H}_true.npy", mmap_mode="r")
                assert ref.shape[0] == len(ds), (ad, flag, H, ref.shape, len(ds))
                idx = np.unique(np.concatenate([[0, len(ds) - 1], rng.integers(0, len(ds), ORNEK)]))
                dl = DataLoader(Subset(ds, idx.tolist()), batch_size=1, shuffle=False, num_workers=0, drop_last=False)
                for i, (bx, by, bxm, bym) in zip(idx, dl):
                    t = by.float()[:, -H:, :].numpy()[0]
                    if not np.array_equal(t, ref[i]):
                        raise SystemExit(f"UYUSMAZ: {ad} {flag} H{H} pencere {i}")
                toplam += len(idx)
                print(f"[dogrula] {ad:8s} {flag:4s} H{H:3d}: {len(idx)} pencere birebir (N={len(ds)}, sekil {ref.shape})",
                      flush=True)
                del ref, ds
    print(f"[dogrula] TAMAM: {toplam} orneklenmis pencere, tolerans 0")


if __name__ == "__main__":
    main()
