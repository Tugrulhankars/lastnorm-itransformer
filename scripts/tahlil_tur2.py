# -*- coding: utf-8 -*-
"""K4c tur-2 tahlili (YALNIZ VALIDATION) -- deney-tahlilci mod 2. tahlil_tur1.py'nin uyarlamasi
(tur 1 ciktilarinin USTUNE YAZMAZ).

Calistirma (WS kokunden):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/tahlil_tur2.py > sonuclar/tahlil_tur2.log

TEST KILIDI: metrics.json'dan yalniz test ile baslamayan anahtarlar suzulur (test_* hic bir degiskene
alinmaz). preds/test_kilitli/, test_metrics.json, leaderboard.csv, replikasyon.md OKUNMAZ.
Tahminler yalniz preds/val_h*.npz'den.

Cikti: sonuclar/tahlil-tur2-ufuk.csv     (varyant x ufuk x seed, eslestirilmis farklar; h336 icin ep336eq kiyasi da)
       sonuclar/tahlil-tur2-ikincil.csv  (varyant x ufuk ikincil metrikler, seed-ort.; 7 varyant)
       sonuclar/normalizasyon-kiyas.csv  (normalizasyon kiyas tablosu: yok / NST / RevIN-afin / son-deger /
                                          SAN-cokus / SAN-mu(kapida elendi))
Tanimlar hata_analizi.py / tahlil_tur1.py ile ayni (blok, s_io, rho_shift, EPS=0.01, trend-asimetri,
D1 = adimlarin ilk %10'u, beta/corr(P1-son, mu_lb-son), alpha*).
"""
import csv
import json
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
OOR_DEG = [0, 3, 5, 6]          # hata-analizi s2: val hedeflerinin %54-95'i train araligi disinda
KOSUM = {
    "baseline": ["20261014-020531_baseline_s2023", "20261014-021024_baseline_s2024", "20261014-021448_baseline_s2025"],
    "baseline-ep336eq": ["20261014-031233_baseline-ep336eq_s2023", "20261014-032853_baseline-ep336eq_s2024",
                         "20261014-033500_baseline-ep336eq_s2025"],
    "revin-afin": ["20261014-034304_revin-afin_s2023", "20261014-034735_revin-afin_s2024", "20261014-035213_revin-afin_s2025"],
    "san": ["20261014-040042_san_s2023", "20261014-040547_san_s2024", "20261014-041102_san_s2025"],
    "lastnorm": ["20261014-052508_lastnorm_s2023", "20261014-053601_lastnorm_s2024",
                 "20261014-054213_lastnorm_s2025"],  # 053238_s2024 rc=96 DISLANDI
    "normsuz": ["20261014-055518_normsuz_s2023", "20261014-060205_normsuz_s2024", "20261014-060859_normsuz_s2025"],
    "xmark-yok": ["20261014-061948_xmark-yok_s2023", "20261014-062637_xmark-yok_s2024", "20261014-063215_xmark-yok_s2025"],
}
TUR2 = ["lastnorm", "normsuz", "xmark-yok"]
KART = {"lastnorm": "H04", "normsuz": "H06", "xmark-yok": "H07"}
MDE_REL = {96: 0.026, 192: 0.035, 336: 0.015, 720: 0.128, "avg": 0.081}  # hata-analizi.md s6 (on-kayitli)
T_MDE = stats.t.ppf(0.975, 2) + stats.t.ppf(0.8, 2)  # eslestirilmis MDE katsayisi (df=2)


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


def main():
    M = {v: [val_metrik(r) for r in rs] for v, rs in KOSUM.items()}
    for v, ms in M.items():
        assert [m["seed"] for m in ms] == SEEDS, v
    ufuk_rows = []
    ozet = {}
    # ---------------- 1. ufuk bazinda olcum + eslestirilmis fark (ebeveyn baseline) + h336 vs ep336eq
    print("# 1. val MSE (seed 2023/2024/2025), Δ vs baseline, eslestirilmis")
    for v in TUR2:
        print(f"\n== {v} ({KART[v]})")
        ozet[v] = {}
        for H in UFUKLAR + ["avg"]:
            key = f"val_mse_h{H}" if H != "avg" else "val_mse_avg"
            b = np.array([m[key] for m in M["baseline"]])
            x = np.array([m[key] for m in M[v]])
            d, p, r, sd_d, mde_e = eslestir(x, b)
            rel = d.mean() / b.mean()
            neg = int((d < 0).sum())
            mde = MDE_REL[H]
            durum = "MDE-ustu" if abs(rel) >= mde else "MDE-alti (ayirt edilemez)"
            ozet[v][H] = dict(x=x, b=b, d=d, rel=rel, p=p)
            print(f"  {key:12s} var {np.round(x, 5)} ort {x.mean():.5f}±{x.std(ddof=1):.5f} | base ort {b.mean():.5f}±{b.std(ddof=1):.5f}"
                  f" | Δ {d.mean():+.5f} ({rel * 100:+.2f}%) cift Δ {np.round(d, 5)} sd_d {sd_d:.5f}"
                  f" | Δ<0 cift {neg}/3 | p_eslesik {p:.4g} | r(cift) {r:+.2f} | MDE3v3 {mde * 100:.1f}% -> {durum}"
                  f" | MDE_eslesik {mde_e:.5f} ({mde_e / b.mean() * 100:.2f}%) | Δ/MDE3v3 {abs(rel) / mde:.2f}")
            for i, s in enumerate(SEEDS):
                ufuk_rows.append(dict(varyant=v, kart=KART[v], ebeveyn="baseline", ufuk=H, seed=s,
                                      val_mse_varyant=round(float(x[i]), 8), val_mse_ebeveyn=round(float(b[i]), 8),
                                      fark=float(d[i]), fark_goreli=round(float(d[i] / b[i]), 6),
                                      fark_avg_katkisi=round(float(d[i] / 4), 6) if H != "avg" else ""))
        kural = sum(ozet[v][H]["x"].mean() < ozet[v][H]["b"].mean() for H in UFUKLAR)
        print(f"  3/4 yon kurali: iyilesen ufuk sayisi (seed-ort.) = {kural}/4")
        # h336 vs ep336eq (zorunlu ikincil)
        e = np.array([m["val_mse_h336"] for m in M["baseline-ep336eq"]])
        x = ozet[v][336]["x"]
        d, p, r, sd_d, _ = eslestir(x, e)
        print(f"  h336 vs baseline-ep336eq: var {x.mean():.5f} vs ep {e.mean():.5f} Δ {d.mean():+.5f} ({d.mean() / e.mean() * 100:+.2f}%)"
              f" cift {np.round(d, 5)} p {p:.4g} | etiket: "
              + ("egitim-butcesinden bagimsiz iyilesme" if x.mean() <= 0.3714 else "h336 <= 0.3714 DEGIL -> oy 'butce-bagimli'"))
        for i, s in enumerate(SEEDS):
            ufuk_rows.append(dict(varyant=v, kart=KART[v], ebeveyn="baseline-ep336eq", ufuk=336, seed=s,
                                  val_mse_varyant=round(float(x[i]), 8), val_mse_ebeveyn=round(float(e[i]), 8),
                                  fark=float(d[i]), fark_goreli=round(float(d[i] / e[i]), 6), fark_avg_katkisi=""))
        # seed basina avg farkinin ufuk ayrisimi
        print("  seed basina Δavg ayrisimi (Δ_H/4):")
        for i, s in enumerate(SEEDS):
            parca = {H: ozet[v][H]["d"][i] / 4 for H in UFUKLAR}
            print(f"    s{s}: Δavg {ozet[v]['avg']['d'][i]:+.5f} = " + " + ".join(f"h{H} {parca[H]:+.5f}" for H in UFUKLAR))
        # leave-one-out / h720 haric
        d_avg = ozet[v]["avg"]["d"]
        d3 = np.mean([ozet[v][H]["d"] for H in [96, 192, 336]], axis=0)
        b3 = np.mean([ozet[v][H]["b"] for H in [96, 192, 336]], axis=0)
        print(f"  h720 haric 3-ufuk ort Δ: cift {np.round(d3, 5)} ort {d3.mean():+.5f} ({d3.mean() / b3.mean() * 100:+.2f}%)"
              f" p {stats.ttest_1samp(d3, 0).pvalue:.4g} | sd_d {d3.std(ddof=1):.5f}")
        print(f"  avg cift farki sd {d_avg.std(ddof=1):.5f}, katki varyansi h720 payi:"
              f" var(Δ_720/4)={np.var(ozet[v][720]['d'] / 4, ddof=1):.3g} / var(Δavg)={np.var(d_avg, ddof=1):.3g}")

    # ---------------- 3. bit-bit / plasebo denetimi
    print("\n# 3. bit-bit denetimi (metrics + preds/val_h*.npz) vs baseline; parametre sayisi")
    for v in TUR2:
        for H in UFUKLAR:
            mfark, pfark, esit = [], [], []
            for rb, rv, mb, mv in zip(KOSUM["baseline"], KOSUM[v], M["baseline"], M[v]):
                mfark.append(mv[f"val_mse_h{H}"] - mb[f"val_mse_h{H}"])
                pb = np.load(WS / "runs" / rb / "preds" / f"val_h{H}.npz")["pred"]
                pv = np.load(WS / "runs" / rv / "preds" / f"val_h{H}.npz")["pred"]
                pfark.append(float(np.abs(pv.astype(np.float64) - pb).max()))
                esit.append(bool(np.array_equal(pv, pb)))
            par = [m.get(f"param_sayisi_h{H}") for m in M[v]]
            parb = [m.get(f"param_sayisi_h{H}") for m in M["baseline"]]
            print(f"  {v:10s} h{H}: metrik fark {np.round(mfark, 6)} | pred maks|fark| {np.round(pfark, 4)} | array_equal {esit}"
                  f" | param {par[0]} (baseline {parb[0]})")

    # ---------------- 2. ikincil metrikler (val) + trend-asimetri + Z2/Z1 mekanizma
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
            # OOR grubu: hedefi train disi olan hucreler vs ic hucreler (OOR degiskenlerinde)
            msk_o = oor_true[:, :, OOR_DEG]
            e_o = e[:, :, :, OOR_DEG]
            mse_oor_hucre = e_o[:, msk_o].mean()
            mse_ic_hucre = e_o[:, ~msk_o].mean() if (~msk_o).sum() > 0 else float("nan")
            # tahminin train araligi disina cikma orani (OOR degiskenleri; gercegin orani yaninda)
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

    # ---------------- trend-asimetri: Δ vs ebeveyn
    print("\n# trend-asimetri (Δ dusus-M/R ve yukselis-M/R, ebeveyn baseline)")
    ik = {(r["varyant"], r["ufuk"]): r for r in ik_rows}
    for v in TUR2:
        for H in [96, 192, 336]:
            a, b = ik[(v, H)], ik[("baseline", H)]
            dd = a["dusus_MR"] - b["dusus_MR"]; dy = a["yukselis_MR"] - b["yukselis_MR"]
            not_ = "VAL TRENDINE ASIRI UYUM notu" if (ozet[v]["avg"]["d"].mean() < 0 and dd >= 0.05) else "-"
            print(f"  {v:10s} h{H}: dusus {b['dusus_MR']} -> {a['dusus_MR']} (Δ {dd:+.4f}) | yukselis {b['yukselis_MR']} -> {a['yukselis_MR']}"
                  f" (Δ {dy:+.4f}) | {not_}")

    # ---------------- yardimci: epoch / sure / param
    print("\n# yardimci: epoch_kosulan, egitim_sure_sn, param")
    for v, ms in M.items():
        ep = {H: [m.get(f"epoch_kosulan_h{H}") for m in ms] for H in UFUKLAR}
        print(f"  {v:17s} epoch {ep} | sure {[m['egitim_sure_sn'] for m in ms]} | param {ms[0].get('param_sayisi')}"
              f" | val_resmi_kayip_min {[[round(m.get(f'val_resmi_kayip_min_h{H}', float('nan')), 4) for H in UFUKLAR] for m in ms]}")

    # ---------------- normalizasyon kiyas tablosu
    print("\n# normalizasyon kiyas tablosu")
    NORM = [("yok (use_norm 0)", "normsuz", "H06"), ("NST ortalama/std (resmi)", "baseline", "rakip"),
            ("RevIN-afin", "revin-afin", "H02"), ("son-deger capasi", "lastnorm", "H04"),
            ("SAN resmi (std basligi cokmus)", "san", "H01")]
    nk_rows = []
    for ad, v, kart in NORM:
        avg = np.array([m["val_mse_avg"] for m in M[v]])
        b = np.array([m["val_mse_avg"] for m in M["baseline"]])
        d = avg - b
        p = float("nan") if np.all(d == 0) else stats.ttest_rel(avg, b).pvalue
        row = dict(normalizasyon=ad, varyant=v, kart=kart, n_seed=len(avg), val_mse_avg_ort=round(float(avg.mean()), 5),
                   val_mse_avg_sd=round(float(avg.std(ddof=1)), 5), delta_vs_NST_goreli=round(float(d.mean() / b.mean()), 4),
                   p_eslesik=round(p, 5) if p == p else "", iyilesen_ufuk=sum(
                       np.mean([m[f"val_mse_h{H}"] for m in M[v]]) < np.mean([m[f"val_mse_h{H}"] for m in M["baseline"]])
                       for H in UFUKLAR) if v != "baseline" else "")
        for H in UFUKLAR:
            row[f"val_mse_h{H}"] = round(float(np.mean([m[f"val_mse_h{H}"] for m in M[v]])), 5)
        for H in UFUKLAR:
            row[f"M_R_h{H}"] = ik[(v, H)]["M_R"]
        for H in UFUKLAR:
            row[f"worst_block_mse_h{H}"] = ik[(v, H)]["worst_block_mse"]
        for H in UFUKLAR:
            row[f"rho_shift_h{H}"] = ik[(v, H)]["rho_shift"]
        for H in [336, 720]:
            row[f"bias_h{H}"] = ik[(v, H)]["bias_P_Y"]
            row[f"s_io_oran_h{H}"] = ik[(v, H)]["s_io_oran_tahmin_gercek"]
        row["D1_MR_h96"] = ik[(v, 96)]["D1_MR"]
        row["D1_MR_h192"] = ik[(v, 192)]["D1_MR"]
        for H in [96, 192, 336]:
            row[f"dusus_MR_h{H}"] = ik[(v, H)]["dusus_MR"]
            row[f"yukselis_MR_h{H}"] = ik[(v, H)]["yukselis_MR"]
        row["mse_oor4deg_h720"] = ik[(v, 720)]["mse_oor4deg"]
        row["kosumlar"] = " ".join(KOSUM[v])
        row["not"] = ""
        nk_rows.append(row)
    # SAN-mu: CPU kapisinda elendi (koşulmadi) -> yalniz bilesen olcumu
    kap = list(csv.DictReader(open(WS / "sonuclar/h05-onkosul.csv", encoding="utf-8")))
    R_H = {int(r["ufuk"]): float(r["R"]) for r in kap if r["split"] == "val" and r["seed"] == "ort"}
    sm = {k: "" for k in nk_rows[0]}
    sm.update(normalizasyon="SAN-mu (yalniz-ortalama SAN)", varyant="san-mu", kart="H05", n_seed=0,
              kosumlar="kosulmadi (ON-KOSUL BASARISIZ)",
              **{"not": "CPU kapisi: R_H=E_san/E_naif val " + " / ".join(f"h{H} {R_H[H]:.3f}" for H in UFUKLAR)
                 + " (0/4 < 1) -> Ö1 BASARISIZ; Ö2 GECER; kart kosulmadi"})
    nk_rows.append(sm)
    for r in nk_rows:
        print("  " + " | ".join(f"{k}={r[k]}" for k in list(r.keys())[:9] + ["M_R_h96", "M_R_h720", "worst_block_mse_h720",
                                                                            "bias_h720", "D1_MR_h96", "dusus_MR_h96", "not"]))

    for ad, rows in [("tahlil-tur2-ufuk.csv", ufuk_rows), ("tahlil-tur2-ikincil.csv", ik_rows), ("normalizasyon-kiyas.csv", nk_rows)]:
        with open(WS / "sonuclar" / ad, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"yazildi: sonuclar/{ad} ({len(rows)} satir)")


if __name__ == "__main__":
    main()
