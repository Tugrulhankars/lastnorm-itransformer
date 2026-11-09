# -*- coding: utf-8 -*-
"""K4c tur-3 tahlili (YALNIZ VALIDATION) -- deney-tahlilci mod 2. tahlil_tur2.py'nin uyarlamasi
(tur 1/2 ciktilarinin USTUNE YAZMAZ).

Calistirma (WS kokunden):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/tahlil_tur3.py > sonuclar/tahlil_tur3.log

TEST KILIDI: metrics.json'dan yalniz test ile baslamayan anahtarlar suzulur (test_* hic bir degiskene
alinmaz). preds/test_kilitli/, test_metrics.json, leaderboard.csv, replikasyon.md OKUNMAZ.
Tahminler yalniz preds/val_h*.npz'den.

Ebeveyn: lastnorm (H04). Tur 3 kartlari: H08 lastnorm-ep336eq (PROTOKOL), H10 lastnorm-merkez, H09 lastnorm-dro.
Cikti: sonuclar/tahlil-tur3-ufuk.csv     (varyant x ufuk x seed eslestirilmis farklar vs lastnorm; h336 icin
                                          baseline-ep336eq ve lastnorm-ep336eq kiyaslari; resmi baseline'a karsi da)
       sonuclar/tahlil-tur3-ikincil.csv  (varyant x ufuk ikincil metrikler, seed-ort.; 6 varyant)
Tanimlar hata_analizi.py / tahlil_tur1.py / tahlil_tur2.py ile ayni (blok, s_io, rho_shift, EPS=0.01, trend-asimetri,
D1 = adimlarin ilk %10'u, beta/corr(P1-son, mu_lb-son), alpha*, OOR).
"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

WS = Path(__file__).resolve().parents[1]
UFUKLAR = [96, 192, 336, 720]
SEEDS = [2023, 2024, 2025]
EPS = 0.01
SEQ = 96
VAL_O = (5311, 6071)
DEG = ["0", "1", "2", "3", "4", "5", "6", "OT"]
OOR_DEG = [0, 3, 5, 6]
KOSUM = {
    "baseline": ["20261014-020531_baseline_s2023", "20261014-021024_baseline_s2024", "20261014-021448_baseline_s2025"],
    "baseline-ep336eq": ["20261014-031233_baseline-ep336eq_s2023", "20261014-032853_baseline-ep336eq_s2024",
                         "20261014-033500_baseline-ep336eq_s2025"],
    "lastnorm": ["20261014-052508_lastnorm_s2023", "20261014-053601_lastnorm_s2024",
                 "20261014-054213_lastnorm_s2025"],  # 053238_s2024 rc=96 DISLANDI
    "lastnorm-ep336eq": ["20261014-140921_lastnorm-ep336eq_s2023", "20261014-141641_lastnorm-ep336eq_s2024",
                         "20261014-142407_lastnorm-ep336eq_s2025"],  # 072750_s2023 rc=96 DISLANDI
    "lastnorm-merkez": ["20261014-143314_lastnorm-merkez_s2023", "20261014-143820_lastnorm-merkez_s2024",
                        "20261014-144341_lastnorm-merkez_s2025"],
    "lastnorm-dro": ["20261014-144849_lastnorm-dro_s2023", "20261014-145213_lastnorm-dro_s2024",
                     "20261014-151234_lastnorm-dro_s2025"],  # 145536_s2025 rc=96 DISLANDI
}
EBEVEYN = "lastnorm"
TUR3 = ["lastnorm-ep336eq", "lastnorm-merkez", "lastnorm-dro"]
KART = {"lastnorm-ep336eq": "H08", "lastnorm-merkez": "H10", "lastnorm-dro": "H09"}
MDE_REL = {96: 0.026, 192: 0.035, 336: 0.015, 720: 0.128, "avg": 0.081}  # hata-analizi.md s6 (on-kayitli)
T_MDE = stats.t.ppf(0.975, 2) + stats.t.ppf(0.8, 2)
# gorulmemis-iyi esikleri (kartlardan, goreli vs lastnorm; avg mutlak)
INCELE = {"lastnorm-merkez": dict(avg=0.3722, h={96: -0.08, 192: -0.12, 336: -0.15, 720: -0.25}),
          "lastnorm-dro": dict(avg=0.3850, h={96: -0.08, 192: -0.12, 336: -0.05, 720: -0.25})}


def val_metrik(run):
    d = json.loads((WS / "runs" / run / "metrics.json").read_text(encoding="utf-8"))
    out = {k: v for k, v in d.items() if not k.lower().startswith("test") and k != "alt_grup"}
    out["alt_grup"] = {k: v for k, v in d.get("alt_grup", {}).items() if k.startswith("val_")}
    return out


def yukle_z():
    sp = json.loads((WS / "data/splits/exchange_rate_tslib_sl96.json").read_text(encoding="utf-8"))
    mu = np.array([sp["scaler_train_only"]["mean"][d] for d in DEG])
    sc = np.array([sp["scaler_train_only"]["scale"][d] for d in DEG])
    ham = np.genfromtxt(WS / "data/exchange_rate/exchange_rate.csv", delimiter=",", skip_header=1, usecols=range(1, 9))
    return (ham - mu) / sc


def eslestir(x, b):
    d = x - b
    p = float("nan") if np.all(d == 0) else stats.ttest_rel(x, b).pvalue
    r = np.corrcoef(x, b)[0, 1] if x.std() > 0 and b.std() > 0 else float("nan")
    sd_d = d.std(ddof=1)
    return d, p, r, sd_d, T_MDE * sd_d / np.sqrt(len(d))


def pred(run, H):
    return np.load(WS / "runs" / run / "preds" / f"val_h{H}.npz")["pred"]


def butce():
    """runs/BILDIRIM.log'dan tur-3 GPU suresi (duman + on-ucus + tam + yarim kalanlar)."""
    sat = (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8").splitlines()
    bas, bit = {}, {}
    for s in sat:
        m = re.match(r"(\S+) (BAŞLADI|BİTTİ) (\S+)", s)
        if not m:
            continue
        t, tur, rid = datetime.fromisoformat(m.group(1)), m.group(2), m.group(3)
        if not re.search(r"lastnorm-(ep336eq|merkez|dro)", rid):
            continue
        if tur == "BAŞLADI":
            bas[rid] = t
        else:
            sm = re.search(r"sure=([\d.]+)s", s)
            bit[rid] = float(sm.group(1))
    top = 0.0
    print("\n# 0. butce (runs/BILDIRIM.log; tur-3 koşumları)")
    for rid, t0 in sorted(bas.items()):
        if rid in bit:
            sn = bit[rid]; etk = "rc=0"
        else:  # yarim: son heartbeat/progress zamanina kadar
            hb = WS / "runs" / rid / "progress.json"
            t1 = datetime.fromisoformat(json.loads(hb.read_text(encoding="utf-8"))["zaman"])
            sn = (t1 - t0).total_seconds(); etk = "YARIM (rc=96), progress.json son zaman"
        top += sn
        print(f"  {rid:45s} {sn:7.1f} s  {etk}")
    print(f"  TOPLAM tur 3: {top:.1f} s = {top / 3600:.3f} GPU-saat")


def main():
    butce()
    M = {v: [val_metrik(r) for r in rs] for v, rs in KOSUM.items()}
    for v, ms in M.items():
        assert [m["seed"] for m in ms] == SEEDS, v
        assert all(k[:4].lower() != "test" for m in ms for k in m), v
    ufuk_rows = []
    ozet = {}
    # ---------------- 1. ufuk bazinda olcum + eslestirilmis fark (ebeveyn lastnorm)
    print("\n# 1. val MSE (seed 2023/2024/2025), Δ vs lastnorm (ebeveyn), eslestirilmis")
    for v in TUR3:
        print(f"\n== {v} ({KART[v]})")
        ozet[v] = {}
        for H in UFUKLAR + ["avg"]:
            key = f"val_mse_h{H}" if H != "avg" else "val_mse_avg"
            b = np.array([m[key] for m in M[EBEVEYN]])
            x = np.array([m[key] for m in M[v]])
            d, p, r, sd_d, mde_e = eslestir(x, b)
            rel = d.mean() / b.mean()
            neg = int((d < 0).sum())
            mde = MDE_REL[H]
            durum = "MDE-ustu" if abs(rel) >= mde else "MDE-alti (ayirt edilemez)"
            ozet[v][H] = dict(x=x, b=b, d=d, rel=rel, p=p, r=r, sd_d=sd_d)
            print(f"  {key:12s} var {np.round(x, 5)} ort {x.mean():.5f}±{x.std(ddof=1):.5f} | ebv ort {b.mean():.5f}±{b.std(ddof=1):.5f}"
                  f" | Δ {d.mean():+.5f} ({rel * 100:+.2f}%) cift Δ {np.round(d, 5)} sd_d {sd_d:.5f}"
                  f" | Δ<0 cift {neg}/3 | p_eslesik {p:.4g} | r(cift) {r:+.2f} | MDE {mde * 100:.1f}% -> {durum}"
                  f" | MDE_eslesik {mde_e:.5f} ({mde_e / b.mean() * 100:.2f}%) | Δ/MDE {abs(rel) / mde:.2f}")
            for i, s in enumerate(SEEDS):
                ufuk_rows.append(dict(varyant=v, kart=KART[v], ebeveyn=EBEVEYN, ufuk=H, seed=s,
                                      val_mse_varyant=round(float(x[i]), 8), val_mse_ebeveyn=round(float(b[i]), 8),
                                      fark=float(d[i]), fark_goreli=round(float(d[i] / b[i]), 6),
                                      fark_avg_katkisi=round(float(d[i] / 4), 6) if H != "avg" else ""))
        kural = sum(ozet[v][H]["x"].mean() < ozet[v][H]["b"].mean() for H in UFUKLAR)
        ozet[v]["kural"] = kural
        print(f"  3/4 yon kurali: iyilesen ufuk sayisi (seed-ort.) = {kural}/4 | Δavg<0: {ozet[v]['avg']['d'].mean() < 0}")
        # resmi baseline'a karsi (makale tablosu)
        for H in UFUKLAR + ["avg"]:
            key = f"val_mse_h{H}" if H != "avg" else "val_mse_avg"
            b = np.array([m[key] for m in M["baseline"]]); x = ozet[v][H]["x"]
            d, p, r, _, _ = eslestir(x, b)
            print(f"    vs baseline {key:12s} Δ {d.mean():+.5f} ({d.mean() / b.mean() * 100:+.2f}%) cift<0 {(d < 0).sum()}/3 p {p:.4g}")
            for i, s in enumerate(SEEDS):
                ufuk_rows.append(dict(varyant=v, kart=KART[v], ebeveyn="baseline", ufuk=H, seed=s,
                                      val_mse_varyant=round(float(x[i]), 8), val_mse_ebeveyn=round(float(b[i]), 8),
                                      fark=float(d[i]), fark_goreli=round(float(d[i] / b[i]), 6), fark_avg_katkisi=""))
        # h336 uclu kiyas
        x = ozet[v][336]["x"]
        for ref in ["baseline-ep336eq"] + (["lastnorm-ep336eq"] if v != "lastnorm-ep336eq" else []):
            e = np.array([m["val_mse_h336"] for m in M[ref]])
            d, p, r, sd_d, _ = eslestir(x, e)
            print(f"  h336 vs {ref}: var {x.mean():.5f} vs {e.mean():.5f} Δ {d.mean():+.5f} ({d.mean() / e.mean() * 100:+.2f}%)"
                  f" cift {np.round(d, 5)} cift<0 {(d < 0).sum()}/3 p {p:.4g}")
            for i, s in enumerate(SEEDS):
                ufuk_rows.append(dict(varyant=v, kart=KART[v], ebeveyn=ref, ufuk=336, seed=s,
                                      val_mse_varyant=round(float(x[i]), 8), val_mse_ebeveyn=round(float(e[i]), 8),
                                      fark=float(d[i]), fark_goreli=round(float(d[i] / e[i]), 6), fark_avg_katkisi=""))
        print(f"  h336 etiketi: " + ("egitim-butcesinden bagimsiz (<= 0.3714)" if x.mean() <= 0.3714 else "butce-bagimli (> 0.3714)"))
        print("  seed basina Δavg ayrisimi (Δ_H/4):")
        for i, s in enumerate(SEEDS):
            parca = {H: ozet[v][H]["d"][i] / 4 for H in UFUKLAR}
            print(f"    s{s}: Δavg {ozet[v]['avg']['d'][i]:+.5f} = " + " + ".join(f"h{H} {parca[H]:+.5f}" for H in UFUKLAR))
        d_avg = ozet[v]["avg"]["d"]
        d3 = np.mean([ozet[v][H]["d"] for H in [96, 192, 336]], axis=0)
        b3 = np.mean([ozet[v][H]["b"] for H in [96, 192, 336]], axis=0)
        print(f"  h720 haric 3-ufuk ort Δ: cift {np.round(d3, 5)} ort {d3.mean():+.5f} ({d3.mean() / b3.mean() * 100:+.2f}%)"
              f" p {stats.ttest_1samp(d3, 0).pvalue:.4g}")
        print(f"  h720 payi: Σ Δ_720/4 = {np.mean(ozet[v][720]['d']) / 4:+.5f} / Δavg {d_avg.mean():+.5f}"
              f" ({np.mean(ozet[v][720]['d']) / 4 / d_avg.mean() * 100:.0f}%) | var(Δ_720/4)={np.var(ozet[v][720]['d'] / 4, ddof=1):.3g}"
              f" / var(Δavg)={np.var(d_avg, ddof=1):.3g}")
        # Welch (karar disi bilgi)
        print(f"  Welch (esl.siz, bilgi) avg p {stats.ttest_ind(ozet[v]['avg']['x'], ozet[v]['avg']['b'], equal_var=False).pvalue:.4g}")
        if v in INCELE:
            t = INCELE[v]
            ihl = [f"avg {ozet[v]['avg']['x'].mean():.4f} <= {t['avg']}"] if ozet[v]["avg"]["x"].mean() <= t["avg"] else []
            ihl += [f"h{H} {ozet[v][H]['rel'] * 100:+.2f}% <= {t['h'][H] * 100:.0f}%" for H in UFUKLAR if ozet[v][H]["rel"] <= t["h"][H]]
            print(f"  gorulmemis-iyi (INCELE) esikleri: {'TETIKLENDI ' + str(ihl) if ihl else 'tetiklenmedi'}")

    # ---------------- H08 ozel: S1 / S2 + kontrol ufuklari bit-bit
    print("\n# H08 ozel (S1/S2, kontrol ufuklari)")
    v = "lastnorm-ep336eq"
    x = ozet[v][336]["x"]; b = ozet[v][336]["b"]
    ds = x - b
    s1rel = ds.mean() / 0.39197
    s1 = "DOGRULANDI" if (s1rel <= -0.015 and (ds < 0).all()) else (
        "TERS" if (s1rel >= 0.015 and (ds > 0).all()) else "DOGRULANMADI")
    e = np.array([m["val_mse_h336"] for m in M["baseline-ep336eq"]])
    es = x - e
    erel = es.mean() / 0.37143
    s2 = "TOPLANIR" if (erel <= -0.015 and (es < 0).all()) else ("TERS ETKILESIM" if (erel >= 0.015 and (es > 0).all()) else "IKAME")
    print(f"  S1: Δ_s {np.round(ds, 5)} Δ_rel {s1rel * 100:+.2f}% -> {s1}")
    print(f"  S2: E_s {np.round(es, 5)} E_rel {erel * 100:+.2f}% p {stats.ttest_rel(x, e).pvalue:.4g} -> {s2}")
    print(f"  gozlenen h336 {x.mean():.5f} - carpimsal toplanirlik noktasi 0.3637 = {x.mean() - 0.3637:+.5f}")
    print(f"  INCELE (a) h336 <= 0.3343: {x.mean() <= 0.3343}")
    for H in [96, 192, 720]:
        mf = [mv[f"val_mse_h{H}"] - mb[f"val_mse_h{H}"] for mv, mb in zip(M[v], M[EBEVEYN])]
        eq = [bool(np.array_equal(pred(rv, H), pred(rb, H))) for rv, rb in zip(KOSUM[v], KOSUM[EBEVEYN])]
        ag = [abs(M[v][i]["alt_grup"].get(k, 0) - M[EBEVEYN][i]["alt_grup"].get(k, 0))
              for i in range(3) for k in M[v][i]["alt_grup"] if f"h{H}" in k]
        print(f"  kontrol h{H}: metrik fark {mf} | pred array_equal {eq} | alt_grup maks|fark| {max(ag) if ag else 'yok'}"
              f" | INCELE(b) |Δ|>1e-6: {any(abs(f) > 1e-6 for f in mf)}")

    # ---------------- H08 ek: epoch-esit protokolde son-deger capasi vs NST (lastnorm-ep336eq vs baseline-ep336eq)
    print("\n# H08 ek: epoch-esit protokolde lastnorm etkisi (lastnorm-ep336eq vs baseline-ep336eq; eslestirilmis)")
    for H in UFUKLAR + ["avg"]:
        key = f"val_mse_h{H}" if H != "avg" else "val_mse_avg"
        x = np.array([m[key] for m in M["lastnorm-ep336eq"]]); e = np.array([m[key] for m in M["baseline-ep336eq"]])
        d, p, r, _, _ = eslestir(x, e)
        print(f"  {key:12s} {x.mean():.5f}±{x.std(ddof=1):.5f} vs {e.mean():.5f}±{e.std(ddof=1):.5f} Δ {d.mean():+.5f}"
              f" ({d.mean() / e.mean() * 100:+.2f}%) cift<0 {(d < 0).sum()}/3 p {p:.4g}")

    # ---------------- A9 pilotu (bilgi, karar disi): 3-seed ortalama tahmin (topluluk) vs tek-model ortalamasi
    print("\n# A9 pilotu (bilgi): 3 seed tahmin ortalamasi MSE vs tek-model MSE ortalamasi (val; n=1 topluluk)")
    for v in ["baseline", "lastnorm", "lastnorm-dro"]:
        parca = []
        for H in UFUKLAR:
            true = np.load(WS / f"data/hedefler_sl96/val_h{H}_true.npy").astype(np.float64)
            Pv = np.stack([pred(r, H).astype(np.float64) for r in KOSUM[v]])
            tek = np.mean([((p_ - true) ** 2).mean() for p_ in Pv]); ens = ((Pv.mean(0) - true) ** 2).mean()
            parca.append((H, tek, ens))
        print(f"  {v:13s} " + " | ".join(f"h{H} tek {t:.5f} ens {en:.5f} ({(en / t - 1) * 100:+.2f}%)" for H, t, en in parca)
              + f" | avg tek {np.mean([t for _, t, _ in parca]):.5f} ens {np.mean([en for _, _, en in parca]):.5f}"
              f" ({(np.mean([en for _, _, en in parca]) / np.mean([t for _, t, _ in parca]) - 1) * 100:+.2f}%)")

    # ---------------- 3. bit-bit / plasebo denetimi + tekrar uretilebilirlik
    print("\n# 3. bit-bit denetimi (preds/val_h*.npz) vs lastnorm; parametre sayisi")
    for v in TUR3:
        for H in UFUKLAR:
            pf, eq = [], []
            for rb, rv in zip(KOSUM[EBEVEYN], KOSUM[v]):
                pb, pv = pred(rb, H).astype(np.float64), pred(rv, H).astype(np.float64)
                pf.append(float(np.abs(pv - pb).max())); eq.append(bool(np.array_equal(pv, pb)))
            par = [m.get(f"param_sayisi_h{H}") for m in M[v]]
            parb = [m.get(f"param_sayisi_h{H}") for m in M[EBEVEYN]]
            print(f"  {v:17s} h{H}: pred maks|fark| {np.round(pf, 4)} | array_equal {eq} | param {par[0]} (lastnorm {parb[0]})")
    print("\n# 3b. tekrar uretilebilirlik: yarim kalan 145536_lastnorm-dro_s2025 (rc=96) vs tam 151234 (ayni seed)")
    for H in [96, 192, 336]:
        f = WS / "runs/20261014-145536_lastnorm-dro_s2025/preds" / f"val_h{H}.npz"
        if f.exists():
            a = np.load(f)["pred"]; c = pred("20261014-151234_lastnorm-dro_s2025", H)
            print(f"  h{H}: array_equal {np.array_equal(a, c)} maks|fark| {np.abs(a.astype(np.float64) - c).max():.3g}")
    print("# 3c. lastnorm-merkez s2023 tam == on-ucus? (on-ucus epoch<=2; tam koşumda en iyi epoch <= 2 ise ayni checkpoint)")
    for H in UFUKLAR:
        a = pred("20261014-071201_lastnorm-merkez-onucus_s2023", H); c = pred("20261014-143314_lastnorm-merkez_s2023", H)
        print(f"  h{H}: array_equal {np.array_equal(a, c)}")
    for r in ["20261014-071201_lastnorm-merkez-onucus_s2023", "20261014-143314_lastnorm-merkez_s2023"]:
        g = json.loads((WS / "runs" / r / "progress.json").read_text(encoding="utf-8")).get("gecmis", [])
        print(f"  {r}: (ufuk, epoch, train, val) {[(u, e_, round(t, 4), round(vv, 4)) for u, e_, t, vv in g]}")

    # ---------------- H09 DRO tani
    print("\n# H09 DRO tani (metrics.json): son-epoch q, q_max, grup train MSE")
    for H in UFUKLAR:
        qs = np.array([m[f"dro_q_son_h{H}"] for m in M["lastnorm-dro"]])
        gm = np.array([m[f"dro_grup_mse_son_h{H}"] for m in M["lastnorm-dro"]])
        print(f"  h{H}: q ort {np.round(qs.mean(0), 3)} (q_max seed {np.round(qs.max(1), 3)}) | grup train MSE ort {np.round(gm.mean(0), 4)}"
              f" | argmax q {qs.argmax(1).tolist()} | grup_n {M['lastnorm-dro'][0][f'dro_grup_n_h{H}']}")
    qm = [np.mean([m[f"dro_q_maks_son_h{H}"] for m in M["lastnorm-dro"]]) for H in (96, 192, 720)]
    print(f"  mekanizma-devrede-degil notu (q_max < 0.30 her ucunde h96/h192/h720): {all(q < 0.30 for q in qm)} ({np.round(qm, 3)})")

    # ---------------- 2. ikincil metrikler (val) + trend-asimetri
    z = yukle_z()
    tr_min, tr_max = z[:5311].min(0), z[:5311].max(0)
    sat = np.arange(*VAL_O)
    bloklar = [(int(p[0]), int(p[-1]) + 1) for p in np.array_split(sat, 4)]
    ik_rows = []
    print("\n# 2. ikincil metrikler (seed-ort.)")
    for H in UFUKLAR:
        true = np.load(WS / f"data/hedefler_sl96/val_h{H}_true.npy").astype(np.float64)
        ilk = None
        P = {}
        for v, rs in KOSUM.items():
            arr = []
            for r in rs:
                f = np.load(WS / "runs" / r / "preds" / f"val_h{H}.npz")
                arr.append(f["pred"].astype(np.float64))
                ilk = f["hedef_ilk_satir"] if ilk is None else ilk
                assert np.array_equal(ilk, f["hedef_ilk_satir"])
            P[v] = np.stack(arr)
        idx = ilk[:, None] + np.arange(H)[None, :]
        assert np.abs(z[idx] - true).max() < 1e-4
        X = z[(ilk - SEQ)[:, None] + np.arange(SEQ)[None, :]]
        last = X[:, -1, :]
        R = np.repeat(last[:, None, :], H, axis=1)
        r_win = ((R - true) ** 2).mean(axis=(1, 2))
        m_r = ((R - true) ** 2).mean()
        blok_id = np.zeros_like(idx)
        for k, (b0, b1) in enumerate(bloklar):
            blok_id[(idx >= b0) & (idx < b1)] = k
        r2 = (R - true) ** 2
        bm_r = np.array([r2[blok_id == k].mean() for k in range(4)])
        mu_lb, sd_lb = X.mean(1), X.std(1)
        mu_hz = true.mean(1)
        s_io = np.mean(np.abs(mu_hz - mu_lb) / (sd_lb + EPS), axis=1)
        yon = np.sign(np.mean(mu_hz - mu_lb, axis=1))
        dus = yon < 0
        q = s_io >= np.quantile(s_io, 0.9)
        d1 = np.array_split(np.arange(H), 10)[0]
        oor_true = (true > tr_max) | (true < tr_min)
        for v in KOSUM:
            Pv = P[v]
            S = Pv.shape[0]
            e = (Pv - true[None]) ** 2
            m_s = e.mean(axis=(1, 2, 3))
            e_win_s = e.mean(axis=(2, 3))
            bm_s = np.array([[e[s][blok_id == k].mean() for k in range(4)] for s in range(S)])
            worst = bm_s.max(1)
            rho = np.mean([stats.spearmanr(s_io, e_win_s[s])[0] for s in range(S)])
            bias = (Pv - true[None]).mean()
            s_io_P = np.array([np.mean(np.abs(Pv[s].mean(1) - mu_lb) / (sd_lb + EPS), axis=1) for s in range(S)]).mean(0)
            oran_sio = float(np.median(s_io_P / s_io))
            d1_mr = e[:, :, d1, :].mean() / r2[:, d1, :].mean()
            dus_mr = e_win_s[:, dus].mean() / r_win[dus].mean() if dus.sum() >= 10 else float("nan")
            yuk_mr = e_win_s[:, ~dus].mean() / r_win[~dus].mean() if (~dus).sum() >= 10 else float("nan")
            top_mr = e_win_s[:, q].mean() / r_win[q].mean()
            vr = np.mean([Pv[s].std(1).mean() / true.std(1).mean() for s in range(S)])
            trend_p = (Pv[:, :, -1, :] - Pv[:, :, 0, :]).mean()
            dv = e.mean(axis=(0, 1, 2))
            betas = [((last - mu_lb).ravel() @ (Pv[s][:, 0, :] - mu_lb).ravel()) / ((last - mu_lb).ravel() @ (last - mu_lb).ravel())
                     for s in range(S)]
            cc = [np.corrcoef((Pv[s][:, 0, :] - last).ravel(), (mu_lb - last).ravel())[0, 1] for s in range(S)]
            gap1 = np.mean(np.abs(Pv[:, :, 0, :] - last[None]))
            sgn = np.mean([np.mean(np.sign(Pv[s][:, -1, :] - last) == np.sign(true[:, -1, :] - last)) for s in range(S)])
            alpha = np.mean([((Pv[s] - R).ravel() @ (true - R).ravel()) / ((Pv[s] - R).ravel() @ (Pv[s] - R).ravel())
                             for s in range(S)])
            msk_o = oor_true[:, :, OOR_DEG]
            e_o = e[:, :, :, OOR_DEG]
            mse_oor_hucre = e_o[:, msk_o].mean()
            mse_ic_hucre = e_o[:, ~msk_o].mean() if (~msk_o).sum() > 0 else float("nan")
            p_oor = ((Pv[:, :, :, OOR_DEG] > tr_max[OOR_DEG]) | (Pv[:, :, :, OOR_DEG] < tr_min[OOR_DEG])).mean()
            row = dict(varyant=v, ufuk=H, n_seed=S, val_mse_ort=round(float(m_s.mean()), 6), val_mse_sd=round(float(m_s.std(ddof=1)), 6),
                       M_R=round(float(m_s.mean() / m_r), 4), repeat_mse=round(float(m_r), 6),
                       blok_MR_B1=round(float(bm_s[:, 0].mean() / bm_r[0]), 3), blok_MR_B2=round(float(bm_s[:, 1].mean() / bm_r[1]), 3),
                       blok_MR_B3=round(float(bm_s[:, 2].mean() / bm_r[2]), 3), blok_MR_B4=round(float(bm_s[:, 3].mean() / bm_r[3]), 3),
                       worst_block_mse=round(float(worst.mean()), 5), worst_block_sd=round(float(worst.std(ddof=1)), 5),
                       worst_blok=f"B{int(np.bincount(bm_s.argmax(1)).argmax()) + 1}",
                       worst_ratio=round(float((worst / m_s).mean()), 4), rho_shift=round(float(rho), 4),
                       top10_MR=round(float(top_mr), 3), bias_P_Y=round(float(bias), 4), s_io_oran_tahmin_gercek=round(oran_sio, 3),
                       std_P_std_Y=round(float(vr), 3), ufuk_ici_tahmin_trendi=round(float(trend_p), 4),
                       D1_MR=round(float(d1_mr), 3), P1_son_abs=round(float(gap1), 4), corr_P1son_mulbson=round(float(np.mean(cc)), 3),
                       beta_capa=round(float(np.mean(betas)), 3), adimH_yon_isabeti=round(float(sgn), 3), alpha_yildiz=round(float(alpha), 3),
                       n_dusus=int(dus.sum()), n_yukselis=int((~dus).sum()),
                       dusus_MR=round(float(dus_mr), 4) if dus_mr == dus_mr else "n<10",
                       yukselis_MR=round(float(yuk_mr), 4) if yuk_mr == yuk_mr else "n<10",
                       mse_deg0=round(float(dv[0]), 4), mse_deg3=round(float(dv[3]), 4),
                       mse_diger6=round(float(np.delete(dv, [0, 3]).mean()), 5),
                       mse_oor4deg=round(float(dv[OOR_DEG].mean()), 4), mse_ic4deg=round(float(np.delete(dv, OOR_DEG).mean()), 5),
                       mse_oor_hucre=round(float(mse_oor_hucre), 4), mse_ic_hucre_oor_deg=round(float(mse_ic_hucre), 4),
                       tahmin_train_disi_oran=round(float(p_oor), 3), gercek_train_disi_oran=round(float(msk_o.mean()), 3))
            ik_rows.append(row)
            print(f"  h{H} {v:17s} " + " ".join(f"{k}={row[k]}" for k in list(row.keys())[3:]))

    ik = {(r["varyant"], r["ufuk"]): r for r in ik_rows}
    print("\n# trend-asimetri (Δ dusus-M/R ve yukselis-M/R, ebeveyn lastnorm)")
    for v in TUR3:
        for H in [96, 192, 336]:
            a, b = ik[(v, H)], ik[(EBEVEYN, H)]
            dd = a["dusus_MR"] - b["dusus_MR"]; dy = a["yukselis_MR"] - b["yukselis_MR"]
            not_ = "VAL TRENDINE ASIRI UYUM notu" if (ozet[v]["avg"]["d"].mean() < 0 and dd >= 0.05) else "-"
            print(f"  {v:17s} h{H}: dusus {b['dusus_MR']} -> {a['dusus_MR']} (Δ {dd:+.4f}) | yukselis {b['yukselis_MR']} -> {a['yukselis_MR']}"
                  f" (Δ {dy:+.4f}) | {not_}")
    print("\n# dayaniklilik (worst_block_mse vs lastnorm; H09 odunlesim notu)")
    for v in TUR3:
        low = [H for H in UFUKLAR if ik[(v, H)]["worst_block_mse"] < ik[(EBEVEYN, H)]["worst_block_mse"]]
        print(f"  {v:17s} worst_block " + " / ".join(f"h{H} {ik[(EBEVEYN, H)]['worst_block_mse']}->{ik[(v, H)]['worst_block_mse']}"
                                                   for H in UFUKLAR) + f" | ebeveynden dusuk: {len(low)}/4 {low}"
              + " | worst_ratio " + " / ".join(f"{ik[(v, H)]['worst_ratio']}" for H in UFUKLAR)
              + " | B3/B4 M/R h96 " + f"{ik[(v, 96)]['blok_MR_B3']}/{ik[(v, 96)]['blok_MR_B4']} h192 {ik[(v, 192)]['blok_MR_B3']}/{ik[(v, 192)]['blok_MR_B4']}")

    print("\n# yardimci: epoch_kosulan, egitim_sure_sn, param, alt_grup degisken 3/0 (val, h720)")
    for v, ms in M.items():
        ep = {H: [m.get(f"epoch_kosulan_h{H}") for m in ms] for H in UFUKLAR}
        print(f"  {v:17s} epoch {ep} | sure {[m['egitim_sure_sn'] for m in ms]} | param {ms[0].get('param_sayisi')}"
              f" | val_resmi_kayip_min {[[round(m.get(f'val_resmi_kayip_min_h{H}', float('nan')), 4) for H in UFUKLAR] for m in ms]}")

    for ad, rows in [("tahlil-tur3-ufuk.csv", ufuk_rows), ("tahlil-tur3-ikincil.csv", ik_rows)]:
        with open(WS / "sonuclar" / ad, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"yazildi: sonuclar/{ad} ({len(rows)} satir)")


if __name__ == "__main__":
    main()
