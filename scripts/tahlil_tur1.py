# -*- coding: utf-8 -*-
"""K4c tur-1 tahlili (YALNIZ VALIDATION) -- deney-tahlilci mod 2.

Calistirma (WS kokunden):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/tahlil_tur1.py > sonuclar/tahlil_tur1.log

TEST KILIDI: metrics.json'dan yalniz 'val_*' ve test ile baslamayan yardimci anahtarlar suzulur
(test_* anahtarlari hic bir degiskene alinmaz). preds/test_kilitli/, test_metrics.json OKUNMAZ.
Tahminler yalniz preds/val_h*.npz'den.

Cikti: sonuclar/tahlil-tur1-ufuk.csv (varyant x ufuk x seed val MSE ve eslestirilmis farklar)
       sonuclar/tahlil-tur1-ikincil.csv (varyant x ufuk ikincil metrikler, seed-ort.)
Tanimlar hata_analizi.py ile ayni (blok, s_io, rho_shift, EPS=0.01); ek: trend-asimetri bayragi
(H01/H02 kart on-kaydi): pencere yonu = sign(mean_d(mu_horizon - mu_lookback)); 'dusus' = < 0;
n<10 ise raporlanmaz.
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
KOSUM = {
    "baseline": ["20261014-020531_baseline_s2023", "20261014-021024_baseline_s2024", "20261014-021448_baseline_s2025"],
    "baseline-ep336eq": ["20261014-031233_baseline-ep336eq_s2023", "20261014-032853_baseline-ep336eq_s2024",
                         "20261014-033500_baseline-ep336eq_s2025"],  # 031950_s2024 rc=96 DISLANDI
    "revin-afin": ["20261014-034304_revin-afin_s2023", "20261014-034735_revin-afin_s2024", "20261014-035213_revin-afin_s2025"],
    "san": ["20261014-040042_san_s2023", "20261014-040547_san_s2024", "20261014-041102_san_s2025"],
}
MDE_REL = {96: 0.026, 192: 0.035, 336: 0.015, 720: 0.128, "avg": 0.081}  # hata-analizi.md s6


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


def main():
    M = {v: [val_metrik(r) for r in rs] for v, rs in KOSUM.items()}
    for v, ms in M.items():
        assert [m["seed"] for m in ms] == SEEDS, v
    # ---------------- 1. ufuk bazinda olcum + eslestirilmis fark
    ufuk_rows = []
    print("# 1. val MSE (seed 2023/2024/2025), Δ vs baseline, eslestirilmis")
    for v in ["baseline-ep336eq", "revin-afin", "san"]:
        print(f"\n== {v}")
        for H in UFUKLAR + ["avg"]:
            key = f"val_mse_h{H}" if H != "avg" else "val_mse_avg"
            b = np.array([m[key] for m in M["baseline"]])
            x = np.array([m[key] for m in M[v]])
            d = x - b
            rel = d.mean() / b.mean()
            if np.all(d == 0):
                p = float("nan")
            else:
                p = stats.ttest_rel(x, b).pvalue
            r_pair = np.corrcoef(x, b)[0, 1] if x.std() > 0 and b.std() > 0 else float("nan")
            neg = int((d < 0).sum())
            mde = MDE_REL[H]
            durum = "MDE-ustu" if abs(rel) >= mde else "MDE-alti (ayirt edilemez)"
            print(f"  {key:12s} var {np.round(x, 5)} ort {x.mean():.5f}±{x.std(ddof=1):.5f} | base ort {b.mean():.5f}±{b.std(ddof=1):.5f}"
                  f" | Δ {d.mean():+.5f} ({rel * 100:+.2f}%) cift Δ {np.round(d, 5)} sd {d.std(ddof=1):.5f}"
                  f" | Δ<0 cift {neg}/3 | p_eslesik {p:.4f} | r(cift) {r_pair:+.2f} | MDE {mde * 100:.1f}% -> {durum}")
            for i, s in enumerate(SEEDS):
                ufuk_rows.append(dict(varyant=v, ebeveyn="baseline", ufuk=H, seed=s, val_mse_varyant=round(float(x[i]), 8),
                                      val_mse_ebeveyn=round(float(b[i]), 8), fark=float(d[i]),
                                      fark_goreli=round(float(d[i] / b[i]), 6)))
        kural = sum(np.mean([m[f"val_mse_h{H}"] for m in M[v]]) < np.mean([m[f"val_mse_h{H}"] for m in M["baseline"]])
                    for H in UFUKLAR)
        print(f"  3/4 yon kurali: iyilesen ufuk sayisi (seed-ort.) = {kural}/4")

    # ---------------- 3. H03 kontrol ufuklari bit-bit + H02 farklilik (pred duzeyinde)
    print("\n# 3. bit-bit denetimi (metrics + preds/val_h*.npz)")
    for v in ["baseline-ep336eq", "revin-afin", "san"]:
        for H in UFUKLAR:
            mfark, pfark, esit = [], [], []
            for rb, rv, mb, mv in zip(KOSUM["baseline"], KOSUM[v], M["baseline"], M[v]):
                mfark.append(mv[f"val_mse_h{H}"] - mb[f"val_mse_h{H}"])
                pb = np.load(WS / "runs" / rb / "preds" / f"val_h{H}.npz")["pred"]
                pv = np.load(WS / "runs" / rv / "preds" / f"val_h{H}.npz")["pred"]
                pfark.append(float(np.abs(pv.astype(np.float64) - pb).max()))
                esit.append(bool(np.array_equal(pv, pb)))
            agf = [mv["alt_grup"][k] - mb["alt_grup"][k] for mb, mv in zip(M["baseline"], M[v])
                   for k in mb["alt_grup"] if k.startswith(f"val_mse_h{H}_")]
            print(f"  {v:17s} h{H}: metrik fark seed {mfark} | pred maks|fark| {np.round(pfark, 8)} | array_equal {esit}"
                  f" | alt_grup maks|fark| {max(abs(a) for a in agf):.3g}")

    # ---------------- 2. ikincil metrikler (val) + trend-asimetri
    z = yukle_z()
    sat = np.arange(*VAL_O)
    bloklar = [(int(p[0]), int(p[-1]) + 1) for p in np.array_split(sat, 4)]
    ik_rows = []
    print("\n# 2. ikincil metrikler (seed-ort.)")
    for H in UFUKLAR:
        true = np.load(WS / f"data/hedefler_sl96/val_h{H}_true.npy").astype(np.float64)
        N = true.shape[0]
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
            dv = e.mean(axis=(0, 1, 2))  # degisken, seed-ort
            row = dict(varyant=v, ufuk=H, n_seed=S, val_mse_ort=round(float(m_s.mean()), 6), val_mse_sd=round(float(m_s.std(ddof=1)), 6),
                       M_R=round(float(m_s.mean() / m_r), 4), repeat_mse=round(float(m_r), 6),
                       blok_mse_B1=round(float(bm_s[:, 0].mean()), 5), blok_mse_B2=round(float(bm_s[:, 1].mean()), 5),
                       blok_mse_B3=round(float(bm_s[:, 2].mean()), 5), blok_mse_B4=round(float(bm_s[:, 3].mean()), 5),
                       blok_MR_B1=round(float(bm_s[:, 0].mean() / bm_r[0]), 3), blok_MR_B2=round(float(bm_s[:, 1].mean() / bm_r[1]), 3),
                       blok_MR_B3=round(float(bm_s[:, 2].mean() / bm_r[2]), 3), blok_MR_B4=round(float(bm_s[:, 3].mean() / bm_r[3]), 3),
                       worst_block_mse=round(float(worst.mean()), 5), worst_block_sd=round(float(worst.std(ddof=1)), 5),
                       worst_blok=f"B{int(np.bincount(bm_s.argmax(1)).argmax()) + 1}",
                       worst_ratio=round(float((worst / m_s).mean()), 4), rho_shift=round(float(rho), 4),
                       top10_MR=round(float(top_mr), 3), bias_P_Y=round(float(bias), 4), s_io_oran_tahmin_gercek=round(oran_sio, 3),
                       std_P_std_Y=round(float(vr), 3), ufuk_ici_tahmin_trendi=round(float(trend_p), 4),
                       D1_MR=round(float(d1_mr), 3), n_dusus=int(dus.sum()), n_yukselis=int((~dus).sum()),
                       dusus_MR=round(float(dus_mr), 4) if dus_mr == dus_mr else "n<10",
                       yukselis_MR=round(float(yuk_mr), 4) if yuk_mr == yuk_mr else "n<10",
                       mse_deg0=round(float(dv[0]), 4), mse_deg3=round(float(dv[3]), 4),
                       mse_diger6=round(float(np.delete(dv, [0, 3]).mean()), 5))
            ik_rows.append(row)
            print(f"  h{H} {v:17s} " + " ".join(f"{k}={row[k]}" for k in ["val_mse_ort", "M_R", "worst_block_mse", "worst_blok", "worst_ratio",
                                                                         "rho_shift", "top10_MR", "bias_P_Y", "s_io_oran_tahmin_gercek",
                                                                         "std_P_std_Y", "ufuk_ici_tahmin_trendi", "D1_MR", "n_dusus", "dusus_MR",
                                                                         "yukselis_MR", "mse_deg3", "mse_deg0", "mse_diger6"]))
    # ---------------- epoch / sure / parametre (val disi yardimci)
    print("\n# yardimci: epoch_kosulan, egitim_sure_sn, param")
    for v, ms in M.items():
        ep = {H: [m.get(f"epoch_kosulan_h{H}") for m in ms] for H in UFUKLAR}
        print(f"  {v:17s} epoch {ep} | sure {[m['egitim_sure_sn'] for m in ms]} | param {ms[0].get('param_sayisi')}")
    # SAN on-egitim, RevIN gamma/beta ozet
    for m in M["san"]:
        print("  san s%d on-egitim en iyi ep: %s  val_min: %s" % (m["seed"], [m[f"san_onegitim_en_iyi_ep_h{H}"] for H in UFUKLAR],
                                                                  [round(m[f"san_onegitim_val_min_h{H}"], 4) for H in UFUKLAR]))
    gam = np.array([[[m[f"revin_gamma_h{H}_d{j}"] for j in range(8)] for H in UFUKLAR] for m in M["revin-afin"]])
    bet = np.array([[[m[f"revin_beta_h{H}_d{j}"] for j in range(8)] for H in UFUKLAR] for m in M["revin-afin"]])
    print(f"  revin gamma: min {gam.min():.4f} max {gam.max():.4f} ort {gam.mean():.4f} | |gamma-1| ort {np.abs(gam - 1).mean():.4f}")
    print(f"  revin beta : min {bet.min():.4f} max {bet.max():.4f} ort {bet.mean():+.4f} | |beta| ort {np.abs(bet).mean():.4f}")
    for i, H in enumerate(UFUKLAR):
        print(f"    h{H}: gamma seed-ort deg {np.round(gam[:, i].mean(0), 3)} beta seed-ort deg {np.round(bet[:, i].mean(0), 4)}"
              f" | beta isaret tutarliligi (3 seed ayni isaret) {int(np.sum(np.abs(np.sign(bet[:, i]).sum(0)) == 3))}/8")

    for ad, rows in [("tahlil-tur1-ufuk.csv", ufuk_rows), ("tahlil-tur1-ikincil.csv", ik_rows)]:
        with open(WS / "sonuclar" / ad, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"yazildi: sonuclar/{ad} ({len(rows)} satir)")


if __name__ == "__main__":
    main()
