#!/usr/bin/env python
"""H16 olcutu R -- hipotezler/H16.md'deki tanimin BIREBIR uygulamasi (FAZ III, veri-hazirlayici, 2026-10-15).

Tanim (H16.md, "Olcut (sabit)"): train bolumunde (resmi split, train-z olcekli), L=96, her H in {96,192,336,720}
icin adim 7 ile kayan pencereler;
    R(H) = ortalama MSE(son degeri H adim tekrarla) / ortalama MSE(pencere ortalamasini H adim tekrarla)
    R_bar = 4 ufuk ortalamasi. Yalniz train; val/test kullanilmaz.
Uygulama ayrintilari (tanimdan dogrudan; serbest secim yok):
  * train = resmi split'in train dilimi [0, num_train) (data/splits/<ad>_sl96.json border2s[0]; kodla capraz
    denetlenir). CSV okunduktan HEMEN sonra yalniz bu satirlar tutulur; val/test satirlari hicbir hesaba girmez.
  * olcek = StandardScaler, yalniz bu train satirlarina fit (resmi iTransformer scaler'i ile ayni).
  * pencere baslangiclari s = 0, 7, 14, ... ; girdi Z[s:s+96], hedef Z[s+96:s+96+H]; s+96+H <= num_train.
  * MSE = pencere icinde H adim x tum degiskenler ortalamasi (features=M); "ortalama MSE" = pencereler uzerinden
    ortalama. Son deger = girdinin son satiri; pencere ortalamasi = girdinin 96 satir ortalamasi (degisken bazinda).
  * Kalibrasyon (H16.md beyani): exchange 0.779, etth1 1.540, etth2 1.177 -> |R_bar - beklenen| <= 0.002 assert.
Tahmin kurali (H16.md): R_bar < 1 -> LastNorm test_mse_avg < iTransformer (iyilesir); R_bar > 1 -> LastNorm >=
iTransformer (kotulesir); |R_bar - 1| < 0.05 -> belirsiz (sinamadan sayilmaz).

Kullanim (WS kokunden): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I -X utf8 scripts/h16_R.py
Ciktilar: sonuclar/h16-R.csv, sonuclar/h16-tahmin.md
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WS = Path(__file__).resolve().parents[1]
L, ADIM, HORIZONS = 96, 7, [96, 192, 336, 720]
BELIRSIZ_BANT = 0.05
KALIBRASYON = {"exchange": 0.779, "etth1": 1.540, "etth2": 1.177}
TOL = 0.002
# ad: (csv, split json, okuma bicimi, num_train kodu)
SETLER = {
    "exchange": ("data/exchange_rate/exchange_rate.csv", "data/splits/exchange_rate_tslib_sl96.json", "custom"),
    "etth1": ("data/etth1/ETTh1.csv", "data/splits/etth1_sl96.json", "ett"),
    "etth2": ("data/etth2/ETTh2.csv", "data/splits/etth2_sl96.json", "ett"),
    "ettm1": ("data/ettm1/ETTm1.csv", "data/splits/ettm1_sl96.json", "ett"),
    "ettm2": ("data/ettm2/ETTm2.csv", "data/splits/ettm2_sl96.json", "ett"),
    "weather": ("data/weather/weather.csv", "data/splits/weather_sl96.json", "custom"),
    "solar": ("data/solar/solar_AL.txt", "data/splits/solar_sl96.json", "solar"),
}
NUM_TRAIN_KOD = {  # data_provider/data_loader.py
    "ett_h": 12 * 30 * 24, "ett_m": 12 * 30 * 24 * 4,
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_satirlari(ad):
    csv, sj, bicim = SETLER[ad]
    split = json.loads((WS / sj).read_text(encoding="utf-8"))
    T = int(split["border2s"][0])
    if bicim == "solar":      # Dataset_Solar okuma bicimi
        rows = []
        with open(WS / csv, "r", encoding="utf-8") as f:
            for line in f.readlines():
                rows.append(np.stack([float(i) for i in line.strip("\n").split(",")]))
        N = len(rows)
        X = np.stack(rows[:T], 0)            # yalniz train satirlari tutulur
        del rows
        T_kod = int(N * 0.7)
    else:
        df = pd.read_csv(WS / csv)
        N = len(df)
        if bicim == "custom":     # Dataset_Custom sutun duzeni (MSE degisken sirasindan bagimsiz; birebirlik icin)
            cols = [c for c in df.columns if c not in ("date", "OT")]
            df = df[["date"] + cols + ["OT"]]
            T_kod = int(N * 0.7)
        else:
            T_kod = NUM_TRAIN_KOD["ett_m" if "ettm" in ad else "ett_h"]
        X = df[df.columns[1:]].iloc[:T].to_numpy(dtype=np.float64)   # yalniz train satirlari
        del df
    assert T == T_kod and N == split["N"], (ad, T, T_kod, N, split["N"])
    assert X.shape[0] == T
    return X, T, N, sha256(WS / csv)


def R_hesapla(X):
    Z = StandardScaler().fit(X).transform(X)      # train-z: scaler yalniz train'e fit
    T, C = Z.shape
    cs = np.vstack([np.zeros((1, C)), np.cumsum(Z, axis=0)])
    satir = []
    for H in HORIZONS:
        s = np.arange(0, T - L - H + 1, ADIM)
        son = Z[s + L - 1]                                     # (n, C)
        ort = (cs[s + L] - cs[s]) / L                          # (n, C) girdi penceresi ortalamasi
        mse_son = np.empty(len(s)); mse_ort = np.empty(len(s))
        for k in range(0, len(s), 256):
            ss = s[k:k + 256]
            Y = Z[ss[:, None] + L + np.arange(H)[None, :]]      # (b, H, C)
            mse_son[k:k + 256] = ((Y - son[k:k + 256, None, :]) ** 2).mean(axis=(1, 2))
            mse_ort[k:k + 256] = ((Y - ort[k:k + 256, None, :]) ** 2).mean(axis=(1, 2))
        assert s[-1] + L + H <= T
        satir.append({"H": H, "pencere": int(len(s)), "ilk_s": int(s[0]), "son_s": int(s[-1]),
                      "son_hedef_satiri": int(s[-1] + L + H - 1),
                      "mse_son_deger": float(mse_son.mean()), "mse_pencere_ort": float(mse_ort.mean()),
                      "R": float(mse_son.mean() / mse_ort.mean())})
    return satir


def tahmin(Rb):
    if abs(Rb - 1) < BELIRSIZ_BANT:
        return "belirsiz"
    return "iyilesir" if Rb < 1 else "kotulesir"


def main():
    t0 = time.time()
    rows, ozet = [], {}
    for ad in SETLER:
        X, T, N, h = train_satirlari(ad)
        sat = R_hesapla(X)
        Rb = float(np.mean([r["R"] for r in sat]))
        rol = "kalibrasyon" if ad in KALIBRASYON else "sinama"
        for r in sat:
            rows.append({"veri": ad, "rol": rol, "num_train": T, "N_csv": N, "degisken": X.shape[1], **r})
        rows.append({"veri": ad, "rol": rol, "num_train": T, "N_csv": N, "degisken": X.shape[1], "H": "ort",
                     "R": Rb})
        ozet[ad] = {"rol": rol, "R_bar": Rb, "R_H": [r["R"] for r in sat], "num_train": T, "C": X.shape[1],
                    "csv_sha256": h, "tahmin": tahmin(Rb)}
        print(f"[h16] {ad:9s} {rol:11s} T={T:6d} C={X.shape[1]:3d} R(H)=" +
              " ".join(f"{r['R']:.4f}" for r in sat) + f"  R_bar={Rb:.4f}  -> {tahmin(Rb)}", flush=True)
        if ad in KALIBRASYON:
            assert abs(Rb - KALIBRASYON[ad]) <= TOL, f"KALIBRASYON TUTMADI: {ad} {Rb:.4f} vs {KALIBRASYON[ad]}"
            print(f"[h16]   kalibrasyon OK: |{Rb:.4f} - {KALIBRASYON[ad]}| = {abs(Rb - KALIBRASYON[ad]):.4f} <= {TOL}")
    out = WS / "sonuclar"
    pd.DataFrame(rows).to_csv(out / "h16-R.csv", index=False, float_format="%.6f")
    (out / "h16-R.json").write_text(json.dumps({"olusturma": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                "komut": "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I -X utf8 scripts/h16_R.py",
                                                "L": L, "adim": ADIM, "ufuklar": HORIZONS, "belirsiz_bant": BELIRSIZ_BANT,
                                                "kalibrasyon_beklenen": KALIBRASYON, "tol": TOL, "sonuc": ozet},
                                               indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[h16] yazildi: {out / 'h16-R.csv'}, {out / 'h16-R.json'} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    sys.exit(main())
