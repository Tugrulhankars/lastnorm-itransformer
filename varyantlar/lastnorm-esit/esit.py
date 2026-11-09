#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
esit.py -- H14 (`lastnorm-esit`): ebeveyn kat-val tahmini P ile Repeat R'nin SABIT esit agirlikli birlesimi
C = 0.5 P + 0.5 R (tum kat/ufuk/adim/degiskende ayni w = 0.5; HICBIR agirlik kestirilmez). Kart: hipotezler/H14.md.

H12 makinesi (varyantlar/lastnorm-birlesim/birlesim.py) import edilir: ham_yukle, b_kat, olcek (R/Y yeniden kurma),
kosum_bul, npz_oku, metrics_oku, guvenli (test kilidi yol filtresi), naif_ref, dilimler, birlestir ve
rakip_kos.pencere_kaydet. Agirlik kestirimi (`agirlik_kestir` / `tum_agirliklar`) CAGRILMAZ; ic model (K_{k-1},
K0 dahil) ve ic-holdout tahminleri ACILMAZ (yol kaydi + assert, sagama (v)).

CPU (CUDA_VISIBLE_DEVICES=-1; torch import edilmez). GPU suresi 0.

Saglamalar (her kosumda; biri kalirsa rc=1, metrics yazilmaz):
  (i)   w = 1 -> C == P bit-bit ve MSE == ebeveyn metrics val_mse_h (<1e-6)
  (ii)  w = 0 -> C == R bit-bit ve MSE == sonuclar/naif-kat.csv (<1e-6)
  (iii) kimlik (kart INCELE (c)): |MSE_C - (0.25 MSE_P + 0.25 MSE_R + 0.5 E[e_P e_R])| <= 1e-6 (float64)
  (iv)  test_* anahtari / test_kilitli/ / test_* dosyasi okunmaz (birlesim.guvenli + anahtar assert'i)
  (v)   okunan her kosum dizini yalniz hedef (ebeveyn kat) kosumudur; K0 / ic model / ic-holdout dosyasi acilmaz
Ebeveyn: --ebeveyn lastnorm (varsayilan; kart H14) | baseline (kart Ablasyon 1 `baseline-esit`, kart "H14-abl1").
K3 lastnorm = resmi 052508/053601/054213 (birlesim.RESMI_K3). baseline K3 icin --kaynak <run_id> zorunlu.
Kullanim (WS kokunden; kuyrukta kosum.py ile):
  env/.venv/Scripts/python.exe varyantlar/lastnorm-esit/esit.py --kat 1 --seed 2023 --out {RUN_DIR}/metrics.json
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lastnorm-birlesim"))
import birlesim as B  # noqa: E402  (rakip_kos'u da import eder; torch yok)

W_SABIT = 0.5
UFUKLAR = B.UFUKLAR
KIMLIK_ESIK = 1e-6

# (v) okunan kosum dizinleri kaydi
OKUNAN = []


def _kayitli_npz(run_dir, H):
    OKUNAN.append(str(run_dir))
    return B.npz_oku(run_dir, H)


def _kayitli_metrics(run_dir):
    OKUNAN.append(str(run_dir))
    return B.metrics_oku(run_dir)


def hedef_kosum(ebeveyn, kat, seed, kaynak=None):
    if kaynak:
        d = B.WS / "runs" / kaynak
        if not d.is_dir():
            raise SystemExit(f"--kaynak {kaynak} bulunamadi")
        return d
    if ebeveyn == "baseline" and kat == 3:
        raise SystemExit("baseline K3 icin --kaynak <run_id> gerekli")
    return B.kosum_bul(f"{ebeveyn}-k{kat}", seed)


def hesapla(kat, seed, ebeveyn="lastnorm", kaynak=None, ham=None, log=print):
    """Bir (kat, seed) icin C = 0.5 P + 0.5 R + saglamalar. Kayit yazmaz."""
    ham = B.ham_yukle() if ham is None else ham
    n = ham.shape[0]
    T, V, b_k, vs = B.b_kat(kat, n)
    hd_dir = hedef_kosum(ebeveyn, kat, seed, kaynak)
    m_hd = _kayitli_metrics(hd_dir)
    if int(m_hd.get("kisaltilmis", 1)) != 0 or int(m_hd.get("ufuk_sayisi", 0)) != 4:
        raise SystemExit(f"{hd_dir.name}: tam kosum degil (kisaltilmis/ufuk)")
    if int(m_hd.get("seed")) != int(seed):
        raise SystemExit(f"{hd_dir.name}: seed uyusmuyor")
    if kat != 3 and int(m_hd.get("kat", -1)) != kat:
        raise SystemExit(f"{hd_dir.name}: kat {m_hd.get('kat')} != {kat}")
    naif = B.naif_ref(kat)
    m_k, s_k = B.olcek(ham[:b_k])                     # kat-k train scaler'i (rakip veri sinifi gibi)
    if "kat_scaler_ort" in m_hd:
        assert np.allclose(m_k, m_hd["kat_scaler_ort"], rtol=0, atol=1e-12), "kat scaler uyusmuyor"
    z_v = (ham[:vs] - m_k) / s_k
    sonuc = {"kat": kat, "seed": seed, "hedef": hd_dir.name, "b_k": b_k, "V": V, "val_son": vs, "H": {}}
    sag = {"i": True, "ii": True, "iii": True, "iv": True, "v": True}
    for H in UFUKLAR:
        P, ilk = _kayitli_npz(hd_dir, H)
        N = P.shape[0]
        assert np.array_equal(ilk, b_k + np.arange(N)), "kat val hedef_ilk_satir beklenen degil"
        idx = ilk[:, None] + np.arange(H)[None, :]
        assert idx.max() < vs
        Y64 = z_v[idx]
        tf = B.HEDEF / (f"val_h{H}_true.npy" if kat == 3 else f"val_k{kat}_h{H}_true.npy")
        Yt = np.load(B.guvenli(tf))
        assert Yt.shape == Y64.shape and float(np.abs(Yt - Y64).max()) < 1e-5, f"hedef yeniden kurulumu uyusmuyor {tf}"
        R = np.repeat(z_v[ilk - 1][:, None, :], H, axis=1)
        w = [W_SABIT] * B.N_DILIM
        C = B.birlestir(P, R, w, H)                    # float64, C = 0.5 P + 0.5 R
        C32 = C.astype(np.float32)
        # (i) / (ii)
        C1 = B.birlestir(P, R, [1.0] * B.N_DILIM, H)
        C0 = B.birlestir(P, R, [0.0] * B.N_DILIM, H)
        d_i = abs(float(np.mean((C1.astype(np.float32) - Yt) ** 2)) - float(m_hd[f"val_mse_h{H}"]))
        s_i = bool(np.array_equal(C1.astype(np.float32), P)) and d_i < 1e-6
        d_ii = abs(float(np.mean((R - Y64) ** 2)) - naif[H])
        s_ii = bool(np.array_equal(C0, R)) and d_ii < 1e-6
        # (iii) kimlik
        P64 = P.astype(np.float64)
        eP, eR = P64 - Y64, R - Y64
        mse_P64, mse_R64 = float(np.mean(eP ** 2)), float(np.mean(eR ** 2))
        mse_C64 = float(np.mean((C - Y64) ** 2))
        d_iii = abs(mse_C64 - (0.25 * mse_P64 + 0.25 * mse_R64 + 0.5 * float(np.mean(eP * eR))))
        s_iii = d_iii <= KIMLIK_ESIK
        sag["i"] &= s_i
        sag["ii"] &= s_ii
        sag["iii"] &= s_iii
        mse = float(np.mean((C32 - Yt) ** 2))
        mae = float(np.mean(np.abs(C32 - Yt)))
        # ikinciller (tanisal): hucre tek-alfa oracle siniri (kart INCELE (a))
        D, E = P64 - R, Y64 - R
        Vh = float(np.mean(D ** 2))
        a_h = float(np.mean(D * E) / Vh) if Vh > 0 else float("nan")
        a_c = min(max(a_h, 0.0), 1.0) if np.isfinite(a_h) else 1.0
        oracle_kazanc = Vh * (1.0 - a_c) ** 2 / mse_P64
        gozlenen = (mse_P64 - mse_C64) / mse_P64
        sonuc["H"][H] = {"mse": mse, "mae": mae, "C32": C32, "Yt": Yt, "ilk": ilk,
                         "mse_P": float(m_hd[f"val_mse_h{H}"]), "mse_R": mse_R64, "alfa_val": a_h,
                         "V_bolu_MSEP": Vh / mse_P64, "oracle": oracle_kazanc, "gozlenen": gozlenen,
                         "d_i": d_i, "d_ii": d_ii, "d_iii": d_iii, "s_i": s_i, "s_ii": s_ii, "s_iii": s_iii}
        log(f"  K{kat} s{seed} h{H}: (i) w=1 C==P bit-bit, |dMSE|={d_i:.1e} {'OK' if s_i else 'HATA'} | "
            f"(ii) w=0 C==R, |dMSE naif|={d_ii:.1e} {'OK' if s_ii else 'HATA'} | (iii) kimlik |d|={d_iii:.1e} "
            f"{'OK' if s_iii else 'HATA'}")
    # (v) yalniz hedef kosum dizini okundu; K0 / ic model yok
    izinli = {str(hd_dir)}
    disarida = sorted(set(OKUNAN) - izinli)
    yasak = [p for p in OKUNAN if "_lastnorm-k0_" in p or "_baseline-k0_" in p]
    sag["v"] = not disarida and not yasak
    OKUNAN.clear()
    sonuc["saglama"] = sag
    sonuc["hd_metrics"] = m_hd
    return sonuc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ebeveyn", choices=["lastnorm", "baseline"], default="lastnorm")
    ap.add_argument("--kaynak", default=None, help="hedef kosum run_id (varsayilan: kosum_bul / resmi K3)")
    a = ap.parse_args(argv)
    if any(x.startswith("--kapi-test") for x in sys.argv):
        print("HATA: test metrigi yasak (K4 kilidi)")
        return 2
    kart = "H14" if a.ebeveyn == "lastnorm" else "H14-abl1"
    varyant = f"{a.ebeveyn}-esit"
    t0 = time.time()
    print(f"[{kart}] varyant={varyant} kat={a.kat} seed={a.seed} w={W_SABIT} sabit (kestirim yok, CPU)", flush=True)
    s = hesapla(a.kat, a.seed, a.ebeveyn, a.kaynak)
    if not all(s["saglama"].values()):
        print(f"[{kart}] SAGLAMA BASARISIZ: {s['saglama']} -> metrics yazilmaz", flush=True)
        return 1
    out = Path(a.out).resolve()
    rd = out.parent
    metrik, ek, alt = {}, {}, {}
    for H in UFUKLAR:
        h = s["H"][H]
        B.rakip_kos.pencere_kaydet(rd / "preds" / f"val_h{H}.npz", h["C32"], h["Yt"], int(h["ilk"][0]), np)
        metrik[f"val_mse_h{H}"] = h["mse"]
        metrik[f"val_mae_h{H}"] = h["mae"]
        for j in range(h["C32"].shape[2]):
            alt[f"val_mse_h{H}_d{j}"] = float(np.mean((h["C32"][:, :, j] - h["Yt"][:, :, j]) ** 2))
        ek[f"ebeveyn_val_mse_h{H}"] = h["mse_P"]
        ek[f"repeat_val_mse_h{H}"] = h["mse_R"]
        ek[f"alfa_val_oracle_h{H}"] = h["alfa_val"]
        ek[f"V_bolu_MSEP_h{H}"] = h["V_bolu_MSEP"]
        ek[f"gozlenen_kazanc_orani_h{H}"] = h["gozlenen"]
        ek[f"oracle_kazanc_orani_tek_alfa_h{H}"] = h["oracle"]
        ek[f"kimlik_fark_h{H}"] = h["d_iii"]
    # INCELE (a): hucre tek-alfa oracle siniri (konveks [0,1]) matematiksel olarak asilamaz (1e-9 sayisal pay)
    ek["oracle_ihlal_sayisi"] = int(sum(s["H"][H]["gozlenen"] > s["H"][H]["oracle"] + 1e-9 for H in UFUKLAR))
    ek["kimlik_fark_maks"] = float(max(s["H"][H]["d_iii"] for H in UFUKLAR))
    pm = s["hd_metrics"]
    sonuc = {
        "val_mse_avg": float(np.mean([metrik[f"val_mse_h{H}"] for H in UFUKLAR])),
        "val_mae_avg": float(np.mean([metrik[f"val_mae_h{H}"] for H in UFUKLAR])),
        **{f"val_mse_h{H}": metrik[f"val_mse_h{H}"] for H in UFUKLAR},
        **{f"val_mae_h{H}": metrik[f"val_mae_h{H}"] for H in UFUKLAR},
        "egitim_sure_sn": 0.0,
        "cpu_sure_sn": round(time.time() - t0, 2),
        "cihaz": "cpu (son-islem; GPU 0 -- egitim maliyeti ebeveyn kat kosumu)",
        "seed": a.seed, "ufuk_sayisi": 4, "kisaltilmis": 0,
        "param_sayisi": pm.get("param_sayisi"),
        "varyant": varyant, "kart": kart, "ebeveyn": a.ebeveyn, "kat": a.kat,
        "turev_kaynak": [s["hedef"]], "w": W_SABIT, "agirlik_kestirimi": "yok (sabit)",
        "kat_val_satir": [s["b_k"], s["val_son"]],
        "saglama_gecti": 1, "saglamalar": {k: bool(v) for k, v in s["saglama"].items()},
        **ek,
        "alt_grup": alt,
    }
    assert not any(k.startswith("test_") for k in sonuc)
    B.rakip_kos.atomik_json(out, sonuc)
    print(f"[{kart}] bitti: val_mse_avg={sonuc['val_mse_avg']:.6f} (ebeveyn {float(pm['val_mse_avg']):.6f}) -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
