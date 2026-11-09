#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
birlesim.py -- H12 (`lastnorm-birlesim`): lastnorm val tahmini P ile Repeat R'nin adim-dilimi basina konveks
birlesimi C = R + w_{H,j}(P - R), w = clip(alpha_hat, 0, 1); alpha_hat YALNIZ katin train'i icindeki ic-holdout'tan.
Kart: hipotezler/H12.md ("Degisen tek sey", "Uygulama notu"). H13 (`lastnorm-olcek`) ayni makineyi
varyantlar/lastnorm-olcek/olcek.py uzerinden yalniz donusum = max(alpha_hat, 0) ile kullanir.

CPU (CUDA_VISIBLE_DEVICES=-1; torch import edilmez). GPU suresi 0.

Tanimlar (kart):
  * Kat k'nin train'i [0, b_k), b_k = T - (3-k)V (Exchange T=5311, V=760; rakip_kos.kat_sinirlari ile ayni).
  * Ic-holdout = [b_k - V, b_k) = K_{k-1} val. Ic model = K_{k-1} kat kosumu (ayni seed, lastnorm):
      k=1 -> `lastnorm-k0` (kuyruk k4t4-lastnorm-k0, --kat 0), k=2 -> `lastnorm-k1`, k=3 -> `lastnorm-k2`.
    Hedef (kat k) kosumu: k=1 `lastnorm-k1`, k=2 `lastnorm-k2`, k=3 = resmi lastnorm tam kosumlari
    (052508/053601/054213; sef K3 yeniden kullanim karari, K3 kodu resmi yolla bit-bit).
  * P' (ic modelin ic-holdout tahmini) K_{k-1} scaler uzayindan kat-k scaler uzayina:
      x_k = (x_{k-1} s_{k-1} + m_{k-1} - m_k) / s_k   (degisken basina; scaler = StandardScaler, rakip veri sinifi gibi)
    R', Y' ham CSV + kat-k scaler'indan yeniden kurulur (R' = son girdi satiri, hedef_ilk_satir - 1).
  * Her (H, j), j = np.array_split(range(H), 10): pencereler x dilim adimlari x 8 degisken havuzlanir:
      alpha_hat = sum D(Y'-R') / sum D^2, D = P' - R';  sum D^2 < 1e-12 -> w = 1.
    H12: w = clip(alpha_hat, 0, 1).  H13: w = max(alpha_hat, 0).
  * Kat k val'ine ayni seed'in w'leri uygulanir: C = w P + (1-w) R (= R + w(P-R); w=1 -> C == P bit-bit,
    w=0 -> C == R bit-bit).
  * Agirlik kestirimi `agirlik_kestir()` ham veriyi YALNIZ [0, b_k) dilimiyle alir (ham[:b_k] kopyasi); kat val'i,
    test ve sonraki satirlar bu fonksiyona hic girmez. dogrula.py bunu pertubasyon testiyle de gosterir.
Saglamalar (her kosumda; biri kalirsa rc=1, metrics yazilmaz): (i) w=1 -> C == P bit-bit; (ii) w=0 -> C == Repeat ve
MSE == sonuclar/naif-kat.csv (<1e-6); (iii) ic-holdout satirlari < b_k (assert); (iv) P' ile yeniden kurulan Y'
(K_{k-1} uzayi) MSE'si K_{k-1} metrics.json val_mse_h* ile esit (<1e-6); (v) test_* anahtari / test_kilitli/ yolu
okunmaz (yol filtresi + anahtar assert'i).
Cikti: {RUN_DIR}/metrics.json (standart val sozlugu + varyant/kart/ebeveyn/kat/turev_kaynak/agirliklar_h{H}...) ve
{RUN_DIR}/preds/val_h{H}.npz (birlesik tahmin; rakip_kos.pencere_kaydet semasi).
Kullanim (WS kokunden; kuyrukta kosum.py ile):
  env/.venv/Scripts/python.exe varyantlar/lastnorm-birlesim/birlesim.py --kat 1 --seed 2023 --out {RUN_DIR}/metrics.json
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402  (yalniz kat_sinirlari / pencere_kaydet / atomik_json; torch import etmez)

UFUKLAR = [96, 192, 336, 720]
SEQ = 96
N_DILIM = 10
DEN_ESIK = 1e-12
CSV_YOL = rakip_kos.VERI_DIZIN / "exchange_rate.csv"
HEDEF = rakip_kos.HEDEF_DIZIN
NAIF_CSV = WS / "sonuclar" / "naif-kat.csv"
RESMI_K3 = {2023: "20261014-052508_lastnorm_s2023", 2024: "20261014-053601_lastnorm_s2024",
            2025: "20261014-054213_lastnorm_s2025"}


def konveks(a):
    return float(min(max(a, 0.0), 1.0))


VARSAYILAN = {"varyant": "lastnorm-birlesim", "kart": "H12", "ebeveyn": "lastnorm", "donusum": konveks,
              "donusum_adi": "clip(alpha_hat, 0, 1)", "alfa_anahtari": "agirliklar"}


# ------------------------------------------------------------------ guvenli okuma (v)
def guvenli(p):
    s = str(p).replace("\\", "/")
    if "test_kilitli" in s or Path(s).name.startswith("test_"):
        raise SystemExit(f"(v) TEST KILIDI: {s} okunamaz")
    return Path(p)


def metrics_oku(run_dir):
    m = json.loads(guvenli(run_dir / "metrics.json").read_text(encoding="utf-8"))
    yasak = [k for k in m if k.startswith("test_")]
    if yasak:
        raise SystemExit(f"(v) {run_dir.name}/metrics.json test_ anahtari iceriyor -> kullanilmaz")
    return m


def npz_oku(run_dir, H):
    f = np.load(guvenli(run_dir / "preds" / f"val_h{H}.npz"))
    return f["pred"], f["hedef_ilk_satir"]


# ------------------------------------------------------------------ veri
def ham_yukle():
    df = pd.read_csv(guvenli(CSV_YOL))
    cols = [c for c in df.columns if c not in ("date", "OT")] + ["OT"]  # Dataset_Custom sutun duzeni
    return df[cols].values  # float64


def b_kat(k, n):
    T, V, vb, vs = rakip_kos.kat_sinirlari("custom", n, k)
    return T, V, vb, vs


def olcek(ham_parca):
    sc = StandardScaler()
    sc.fit(ham_parca)
    return sc.mean_.copy(), sc.scale_.copy()


def kosum_bul(varyant, seed):
    if varyant == "lastnorm-k3":
        return WS / "runs" / RESMI_K3[int(seed)]
    aday = []
    for d in sorted((WS / "runs").glob(f"*_{varyant}_s{seed}")):
        cfg, done = d / "config.json", d / ".done"
        if not (cfg.exists() and done.exists()):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8"))
        j = json.loads(done.read_text(encoding="utf-8"))
        if c.get("varyant") == varyant and str(c.get("seed")) == str(seed) and j.get("rc") == 0 \
                and j.get("metrics_var") and int(c.get("asama", 3)) == 3:
            aday.append(d)
    if len(aday) != 1:
        raise SystemExit(f"{varyant} s{seed}: tam (asama 3, rc=0) kosum sayisi {len(aday)} != 1 "
                         f"({[a.name for a in aday]}) -> YETERSIZ/BELIRSIZ")
    return aday[0]


def dilimler(H):
    return np.array_split(np.arange(H), N_DILIM)


# ------------------------------------------------------------------ agirlik kestirimi (YALNIZ [0, b_k))
def agirlik_kestir(ham_train, b_k, V, ic_pred, ic_ilk, H, donusum):
    """ham_train = ham[:b_k] (kopya). Kat val'i / test bu fonksiyona girmez. Donus: alpha_hat[10], w[10], ek."""
    assert ham_train.shape[0] == b_k, (ham_train.shape, b_k)
    N = ic_pred.shape[0]
    # (iii) ic-holdout: hedef satirlari [b_k - V, b_k) icinde, girdi satirlari >= 0
    assert np.array_equal(ic_ilk, (b_k - V) + np.arange(N)), "ic-holdout hedef_ilk_satir beklenen degil"
    idx = ic_ilk[:, None] + np.arange(H)[None, :]
    assert idx.max() < b_k and idx.min() >= b_k - V and int(ic_ilk.min()) - SEQ >= 0, "(iii) ic-holdout sinir ihlali"
    m_ic, s_ic = olcek(ham_train[:b_k - V])     # K_{k-1} train scaler'i
    m_k, s_k = olcek(ham_train)                 # kat-k train scaler'i
    z_ic = (ham_train - m_ic) / s_ic
    z_k = (ham_train - m_k) / s_k
    Y_ic = z_ic[idx]                                                  # (iv) icin, K_{k-1} uzayi
    P_k = (ic_pred.astype(np.float64) * s_ic + m_ic - m_k) / s_k      # P' -> kat-k uzayi
    Y_k = z_k[idx]
    R_k = np.repeat(z_k[ic_ilk - 1][:, None, :], H, axis=1)          # son girdi satiri
    D = P_k - R_k
    E = Y_k - R_k
    alfa, w = [], []
    for st in dilimler(H):
        den = float(np.sum(D[:, st, :] ** 2))
        if den < DEN_ESIK:
            alfa.append(float("nan"))
            w.append(1.0)
        else:
            a = float(np.sum(D[:, st, :] * E[:, st, :]) / den)
            alfa.append(a)
            w.append(donusum(a))
    ek = {"ic_mse_yeniden_K_km1": float(np.mean((ic_pred.astype(np.float32) - Y_ic.astype(np.float32)) ** 2)),
          "m_ic": m_ic, "s_ic": s_ic, "m_k": m_k, "s_k": s_k}
    return alfa, w, ek


def birlestir(P, R, w, H):
    """C = w P + (1-w) R, dilim basina (float64; w=1 -> P, w=0 -> R bit-bit)."""
    C = np.empty_like(R)
    P64 = P.astype(np.float64)
    for wj, st in zip(w, dilimler(H)):
        C[:, st, :] = wj * P64[:, st, :] + (1.0 - wj) * R[:, st, :]
    return C


def naif_ref(k):
    out = {}
    with open(guvenli(NAIF_CSV), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["veri"] == "exchange" and int(r["kat"]) == k:
                out[int(r["ufuk"])] = float(r["mse"])
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ ana hesap
def tum_agirliklar(kat, seed, cfg, ham):
    """Kat k, seed s icin 4 ufkun agirliklari. Ham veriden YALNIZ ham[:b_k] kopyasi kullanilir (dogrula.py bunu
    b_k ve sonrasindaki satirlari bozarak sinar). Donus: {H: (alfa_hat[10], w[10], ek)}."""
    T, V, b_k, vs = b_kat(kat, ham.shape[0])
    ic_dir = kosum_bul(f"lastnorm-k{kat - 1}", seed)
    ham_train = ham[:b_k].copy()
    return {H: agirlik_kestir(ham_train, b_k, V, *npz_oku(ic_dir, H), H, cfg["donusum"]) for H in UFUKLAR}


def hesapla(kat, seed, cfg, ham=None, hedef_pred_degistir=None, log=print):
    """Bir (kat, seed) icin agirliklar + birlesik val tahmini + saglamalar. Kayit yazmaz."""
    ham = ham_yukle() if ham is None else ham
    n = ham.shape[0]
    T, V, b_k, vs = b_kat(kat, n)
    ic_var = f"lastnorm-k{kat - 1}"
    hd_var = f"lastnorm-k{kat}"
    ic_dir, hd_dir = kosum_bul(ic_var, seed), kosum_bul(hd_var, seed)
    m_ic_json, m_hd_json = metrics_oku(ic_dir), metrics_oku(hd_dir)
    for nm, m in ((ic_dir.name, m_ic_json), (hd_dir.name, m_hd_json)):
        if int(m.get("kisaltilmis", 1)) != 0 or int(m.get("ufuk_sayisi", 0)) != 4:
            raise SystemExit(f"{nm}: tam kosum degil (kisaltilmis/ufuk)")
        if int(m.get("seed")) != int(seed):
            raise SystemExit(f"{nm}: seed uyusmuyor")
    if int(m_ic_json.get("kat", -1)) != kat - 1:
        raise SystemExit(f"{ic_dir.name}: kat {m_ic_json.get('kat')} != {kat - 1}")
    if kat != 3 and int(m_hd_json.get("kat", -1)) != kat:
        raise SystemExit(f"{hd_dir.name}: kat {m_hd_json.get('kat')} != {kat}")
    naif = naif_ref(kat)
    AG = tum_agirliklar(kat, seed, cfg, ham)  # kat val'i / hedef tahminleri GORMEZ
    m_k_v, s_k_v = olcek(ham[:b_k])
    z_v = (ham[:vs] - m_k_v) / s_k_v  # yalniz uygulama/degerlendirme (kat val) icin
    sonuc = {"kat": kat, "seed": seed, "ic": ic_dir.name, "hedef": hd_dir.name, "b_k": b_k, "V": V, "val_son": vs,
             "H": {}}
    sag = {"i": True, "ii": True, "iii": True, "iv": True, "v": True}
    for H in UFUKLAR:
        alfa, w, ek = AG[H]
        # scaler capraz kontrolu (ic kosumun kendi kaydi)
        if "kat_scaler_ort" in m_ic_json:
            assert np.allclose(ek["m_ic"], m_ic_json["kat_scaler_ort"], rtol=0, atol=1e-12) and \
                np.allclose(ek["s_ic"], m_ic_json["kat_scaler_std"], rtol=1e-12, atol=0), "ic scaler uyusmuyor"
        if "kat_scaler_ort" in m_hd_json:
            assert np.allclose(ek["m_k"], m_hd_json["kat_scaler_ort"], rtol=0, atol=1e-12), "kat scaler uyusmuyor"
        # (iv)
        d_iv = abs(ek["ic_mse_yeniden_K_km1"] - float(m_ic_json[f"val_mse_h{H}"]))
        sag["iv"] &= d_iv < 1e-6
        # --- kat val (uygulama)
        P, ilk = npz_oku(hd_dir, H)
        if hedef_pred_degistir is not None:
            P = hedef_pred_degistir(P)
        N = P.shape[0]
        assert np.array_equal(ilk, b_k + np.arange(N)), "kat val hedef_ilk_satir beklenen degil"
        idx = ilk[:, None] + np.arange(H)[None, :]
        assert idx.max() < vs
        Y64 = z_v[idx]
        tf = HEDEF / (f"val_h{H}_true.npy" if kat == 3 else f"val_k{kat}_h{H}_true.npy")
        Yt = np.load(guvenli(tf))
        assert Yt.shape == Y64.shape and float(np.abs(Yt - Y64).max()) < 1e-5, f"hedef yeniden kurulumu uyusmuyor {tf}"
        R = np.repeat(z_v[ilk - 1][:, None, :], H, axis=1)
        C = birlestir(P, R, w, H)
        C32 = C.astype(np.float32)
        # (i) w=1 -> C == P ; (ii) w=0 -> C == R ve naif MSE
        C1 = birlestir(P, R, [1.0] * N_DILIM, H)
        C0 = birlestir(P, R, [0.0] * N_DILIM, H)
        s_i = bool(np.array_equal(C1.astype(np.float32), P)) and \
            abs(float(np.mean((C1.astype(np.float32) - Yt) ** 2)) - float(m_hd_json[f"val_mse_h{H}"])) < 1e-6
        s_ii = bool(np.array_equal(C0, R)) and abs(float(np.mean((R - Y64) ** 2)) - naif[H]) < 1e-6
        if hedef_pred_degistir is None:
            sag["i"] &= s_i
        sag["ii"] &= s_ii
        mse = float(np.mean((C32 - Yt) ** 2))
        mae = float(np.mean(np.abs(C32 - Yt)))
        # --- ikinciller (agirliklar dondurulduktan SONRA; kat val yalniz raporlama / INCELE (a))
        P64 = P.astype(np.float64)
        Dv, Ev = P64 - R, Y64 - R
        alfa_val, Vj, gain_sl, nj = [], [], 0.0, []
        for st in dilimler(H):
            Vd = float(np.mean(Dv[:, st, :] ** 2))
            av = float(np.mean(Dv[:, st, :] * Ev[:, st, :]) / Vd) if Vd > 0 else float("nan")
            alfa_val.append(av)
            Vj.append(Vd)
            nj.append(len(st))
            if Vd > 0:
                a_opt = cfg["donusum"](av)
                mse_p = float(np.mean((P64[:, st, :] - Y64[:, st, :]) ** 2))
                mse_opt = float(np.mean((R[:, st, :] - Y64[:, st, :]) ** 2)) - 2 * a_opt * av * Vd + a_opt ** 2 * Vd
                gain_sl += len(st) * (mse_p - mse_opt)
        gain_sl /= H
        mse_P = float(np.mean((P64 - Y64) ** 2))
        Vh = float(np.mean(Dv ** 2))
        a_h = float(np.mean(Dv * Ev) / Vh) if Vh > 0 else float("nan")
        a_opt_h = cfg["donusum"](a_h)
        mse_opt_h = float(np.mean((R - Y64) ** 2)) - 2 * a_opt_h * a_h * Vh + a_opt_h ** 2 * Vh
        gain_obs = mse_P - float(np.mean((C - Y64) ** 2))
        sonuc["H"][H] = {
            "mse": mse, "mae": mae, "alfa": alfa, "w": w, "C32": C32, "Yt": Yt, "ilk": ilk,
            "mse_P": float(m_hd_json[f"val_mse_h{H}"]), "mse_R": float(np.mean((R - Y64) ** 2)),
            "alfa_val": alfa_val, "V_bolu_MSEP": Vh / mse_P, "spearman": spearman(alfa, alfa_val),
            "oracle_kazanc_dilim": gain_sl / mse_P, "oracle_kazanc_tek_alfa": (mse_P - mse_opt_h) / mse_P,
            "gozlenen_kazanc": gain_obs / mse_P, "d_iv": d_iv, "s_i": s_i, "s_ii": s_ii,
        }
        log(f"  K{kat} s{seed} h{H}: w=" + "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
            f" | (i) {'OK' if s_i else 'HATA'} (ii) {'OK' if s_ii else 'HATA'} (iv) |d|={d_iv:.2e} "
            f"{'OK' if d_iv < 1e-6 else 'HATA'}")
    sonuc["saglama"] = sag
    sonuc["ic_metrics"] = m_ic_json
    sonuc["hd_metrics"] = m_hd_json
    return sonuc


def main(cfg=VARSAYILAN, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if any(x.startswith("--kapi-test") for x in sys.argv):
        print("HATA: test metrigi yasak (K4 kilidi)")
        return 2
    t0 = time.time()
    print(f"[{cfg['kart']}] varyant={cfg['varyant']} kat={a.kat} seed={a.seed} donusum={cfg['donusum_adi']} (CPU)",
          flush=True)
    s = hesapla(a.kat, a.seed, cfg)
    if not all(s["saglama"].values()):
        print(f"[{cfg['kart']}] SAGLAMA BASARISIZ: {s['saglama']} -> metrics yazilmaz", flush=True)
        return 1
    out = Path(a.out).resolve()
    rd = out.parent
    metrik, ek, alt = {}, {}, {}
    for H in UFUKLAR:
        h = s["H"][H]
        rakip_kos.pencere_kaydet(rd / "preds" / f"val_h{H}.npz", h["C32"], h["Yt"], int(h["ilk"][0]), np)
        metrik[f"val_mse_h{H}"] = h["mse"]
        metrik[f"val_mae_h{H}"] = h["mae"]
        for j in range(h["C32"].shape[2]):
            alt[f"val_mse_h{H}_d{j}"] = float(np.mean((h["C32"][:, :, j] - h["Yt"][:, :, j]) ** 2))
        ek[f"{cfg['alfa_anahtari']}_h{H}"] = h["w"]
        ek[f"alfa_ham_h{H}"] = h["alfa"]
        ek[f"alfa_val_oracle_h{H}"] = h["alfa_val"]
        ek[f"spearman_alfa_ic_val_h{H}"] = h["spearman"]
        ek[f"ebeveyn_val_mse_h{H}"] = h["mse_P"]
        ek[f"repeat_val_mse_h{H}"] = h["mse_R"]
        ek[f"V_bolu_MSEP_h{H}"] = h["V_bolu_MSEP"]
        ek[f"gozlenen_kazanc_orani_h{H}"] = h["gozlenen_kazanc"]
        ek[f"oracle_kazanc_orani_dilim_h{H}"] = h["oracle_kazanc_dilim"]
        ek[f"oracle_kazanc_orani_tek_alfa_h{H}"] = h["oracle_kazanc_tek_alfa"]
    tum_w = [x for H in UFUKLAR for x in s["H"][H]["w"]]
    tum_a = [x for H in UFUKLAR for x in s["H"][H]["alfa"] if np.isfinite(x)]
    ek["w_bir_orani"] = float(np.mean([x == 1.0 for x in tum_w]))
    ek["w_sifir_orani"] = float(np.mean([x == 0.0 for x in tum_w]))
    ek["w_095_105_orani"] = float(np.mean([0.95 <= x <= 1.05 for x in tum_w]))
    ek["kirpma_orani"] = float(np.mean([w_ != a_ for H in UFUKLAR for w_, a_ in zip(s["H"][H]["w"], s["H"][H]["alfa"])]))
    ek["alfa_maks"] = float(max(tum_a)) if tum_a else float("nan")
    ek["alfa_min"] = float(min(tum_a)) if tum_a else float("nan")
    # INCELE (a): dilim-duzeyi oracle siniri matematiksel olarak asilamaz (1e-9 sayisal pay)
    ek["oracle_ihlal_sayisi"] = int(sum(s["H"][H]["gozlenen_kazanc"] > s["H"][H]["oracle_kazanc_dilim"] + 1e-9
                                        for H in UFUKLAR))
    ek["oracle_ihlal_sayisi_tek_alfa_hucre"] = int(sum(
        s["H"][H]["gozlenen_kazanc"] > s["H"][H]["oracle_kazanc_tek_alfa"] + 1e-9 for H in UFUKLAR))
    pm = s["hd_metrics"]
    sonuc = {
        "val_mse_avg": float(np.mean([metrik[f"val_mse_h{H}"] for H in UFUKLAR])),
        "val_mae_avg": float(np.mean([metrik[f"val_mae_h{H}"] for H in UFUKLAR])),
        **{f"val_mse_h{H}": metrik[f"val_mse_h{H}"] for H in UFUKLAR},
        **{f"val_mae_h{H}": metrik[f"val_mae_h{H}"] for H in UFUKLAR},
        "egitim_sure_sn": 0.0,
        "cpu_sure_sn": round(time.time() - t0, 2),
        "cihaz": "cpu (son-islem; GPU 0 -- egitim maliyeti ic model + kat kosumu)",
        "seed": a.seed, "ufuk_sayisi": 4, "kisaltilmis": 0,
        "param_sayisi": pm.get("param_sayisi"),
        "varyant": cfg["varyant"], "kart": cfg["kart"], "ebeveyn": cfg["ebeveyn"], "kat": a.kat,
        "turev_kaynak": [s["ic"], s["hedef"]],
        "donusum": cfg["donusum_adi"], "dilim_sayisi": N_DILIM,
        "kat_val_satir": [s["b_k"], s["val_son"]], "ic_holdout_satir": [s["b_k"] - s["V"], s["b_k"]],
        "agirlik_veri_satir": [0, s["b_k"]],
        "saglama_gecti": 1, "saglamalar": {k: bool(v) for k, v in s["saglama"].items()},
        **ek,
        "alt_grup": alt,
    }
    assert not any(k.startswith("test_") for k in sonuc)
    rakip_kos.atomik_json(out, sonuc)
    print(f"[{cfg['kart']}] bitti: val_mse_avg={sonuc['val_mse_avg']:.6f} (ebeveyn "
          f"{float(pm['val_mse_avg']):.6f}) w=1 orani {ek['w_bir_orani']:.2f} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
