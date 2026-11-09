# -*- coding: utf-8 -*-
"""K3.5 hata analizi (YALNIZ VALIDATION) -- iTransformer baseline vs Repeat-last, Exchange-Rate.

Calistirma (WS kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/hata_analizi.py \
      [--runs runs/<id1> runs/<id2> ...] [--etiket baseline]

Varsayilan --runs: uc baseline kosumu (s2023/s2024/s2025). K5'te kazanan varyant icin ayni betik
--runs ile kosulur ("once/sonra"). Ciktilar:
  sonuclar/hata-kirilim.csv     ufuk x degisken x blok (val)
  sonuclar/hata-adim.csv        ufuk x adim-dilimi (val) hata buyume egrisi
  stdout                        hata-analizi.md'deki tum sayilar

TEST KILIDI: betik preds/test_kilitli/, test_metrics.json ve metrics.json'daki test_* anahtarlarini
OKUMAZ. Yalniz preds/val_h*.npz, data/hedefler_sl96/val_*, ham CSV ve split JSON'u kullanir.

SABITLENMIS DAYANIKLILIK METRIKLERI (on-kayit adayi; val ve test icin ayni tanim):
  Blok: dilimin KENDI satirlari [o0,o1) np.array_split ile K=4 bitisik kronolojik bloga bolunur
        (val: [5311,6071) -> 4x190 satir; test: K2'nin test_Q1..Q4'u). Her (pencere i, adim h, degisken d)
        hucresi HEDEF SATIRINA (hedef_ilk_satir[i]+h) gore bloga atanir. MSE_k = o bloktaki hucrelerin
        karesel hata ortalamasi (z uzayi). worst_block_mse = max_k MSE_k; worst_ratio = worst / tum-hucre MSE.
  Kayma skoru (literatur-cipa s6.2): s_io(i) = mean_d |mu(horizon_i,d) - mu(lookback_i,d)| / (sd(lookback_i,d) + EPS),
        EPS = 0.01 (z birimi). Gercek horizon'u kullanir -> YALNIZ degerlendirme. rho = Spearman(s_io, pencere MSE).
        top10_mse = s_io'nun en ust %10'undaki pencerelerin MSE'si / tum pencere MSE'si.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", message="An input array is constant")  # H=720: tum girdiler OOR -> rho tanimsiz (nan)
WS = Path(__file__).resolve().parents[1]
UFUKLAR = [96, 192, 336, 720]
DEG = ["0", "1", "2", "3", "4", "5", "6", "OT"]
K_BLOK = 4
EPS = 0.01
SEQ = 96
VAL_B1 = 5215          # val dataset border1
VAL_O = (5311, 6071)   # val kendi satirlari
VARSAYILAN_RUNS = ["runs/20261014-020531_baseline_s2023", "runs/20261014-021024_baseline_s2024",
                   "runs/20261014-021448_baseline_s2025"]


def yukle_z():
    sp = json.loads((WS / "data/splits/exchange_rate_tslib_sl96.json").read_text(encoding="utf-8"))
    mu = np.array([sp["scaler_train_only"]["mean"][d] for d in DEG])
    sc = np.array([sp["scaler_train_only"]["scale"][d] for d in DEG])
    ham = np.genfromtxt(WS / "data/exchange_rate/exchange_rate.csv", delimiter=",", skip_header=1,
                        usecols=range(1, 9))
    z = (ham - mu) / sc
    return z, z[:5311].min(0), z[:5311].max(0)


def blok_sinirlari():
    satir = np.arange(*VAL_O)
    parcalar = np.array_split(satir, K_BLOK)
    return [(int(p[0]), int(p[-1]) + 1) for p in parcalar]


def mse(a, b, axis=None):
    return np.mean((a - b) ** 2, axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=VARSAYILAN_RUNS)
    ap.add_argument("--etiket", default="baseline")
    ap.add_argument("--cikti-ek", default="", help="ciktilara sonek (K5 'sonra' kosusu icin)")
    a = ap.parse_args()
    runs = [WS / r for r in a.runs]
    z, tr_min, tr_max = yukle_z()
    bloklar = blok_sinirlari()
    print(f"# hata_analizi.py  etiket={a.etiket}  runs={[r.name for r in runs]}")
    print(f"bloklar (val kendi satirlari, hedef-satiri bazli): {bloklar}")

    kirilim, adim_satir = [], []
    ozet = {}
    for H in UFUKLAR:
        true = np.load(WS / f"data/hedefler_sl96/val_h{H}_true.npy").astype(np.float64)
        N = true.shape[0]
        preds, ilk = [], None
        for r in runs:
            f = np.load(r / "preds" / f"val_h{H}.npz")
            preds.append(f["pred"].astype(np.float64))
            ilk = f["hedef_ilk_satir"] if ilk is None else ilk
            assert np.array_equal(ilk, f["hedef_ilk_satir"])
        P = np.stack(preds)  # (S,N,H,8)
        S = P.shape[0]
        # --- veri hatti denetimi: hedefler == ham CSV z satirlari
        idx = ilk[:, None] + np.arange(H)[None, :]          # (N,H) hedef satirlari
        assert np.abs(z[idx] - true).max() < 1e-4, "hedef/CSV uyusmazligi"
        lb_idx = (ilk - SEQ)[:, None] + np.arange(SEQ)[None, :]  # (N,96) girdi satirlari
        X = z[lb_idx]                                        # (N,96,8)
        last = X[:, -1, :]                                   # (N,8)
        R = np.repeat(last[:, None, :], H, axis=1)           # Repeat-last (N,H,8)
        Pm = P.mean(0)                                       # seed-ortalama tahmin (yalniz tanisal)

        # ===== 1. ufuk duzeyi
        m_s = np.array([mse(P[s], true) for s in range(S)])
        m_r = mse(R, true)
        mae_s = np.array([np.mean(np.abs(P[s] - true)) for s in range(S)])
        mae_r = np.mean(np.abs(R - true))
        # pencere bazli: model (seed-ort hata) vs repeat
        e_win_s = np.array([mse(P[s], true, axis=(1, 2)) for s in range(S)])  # (S,N)
        e_win = e_win_s.mean(0)
        r_win = mse(R, true, axis=(1, 2))
        kazan = float(np.mean(e_win < r_win))
        n_eff = (N + H - 1) / H
        # MDE (3 vs 3 seed, iki-orneklem t, alfa .05 iki-yonlu, guc .8, df=2S-2)
        sd = m_s.std(ddof=1)
        dfm = 2 * S - 2
        mde = (stats.t.ppf(0.975, dfm) + stats.t.ppf(0.8, dfm)) * sd * np.sqrt(2 / S)
        print(f"\n===== H={H}  N_pencere={N}  bagimsiz-hedef-parcasi~{n_eff:.2f}")
        print(f"[1] MSE seed={np.round(m_s, 4)} ort={m_s.mean():.4f} sd={sd:.4f} min={m_s.min():.4f} max={m_s.max():.4f}"
              f" | Repeat={m_r:.4f}  D={m_s.mean() - m_r:+.4f} ({(m_s.mean() / m_r - 1) * 100:+.1f}%)"
              f" | seed-bazli D%={np.round((m_s / m_r - 1) * 100, 1)}")
        print(f"    MAE ort={mae_s.mean():.4f} Repeat={mae_r:.4f} ({(mae_s.mean() / mae_r - 1) * 100:+.1f}%)"
              f" | modelin Repeat'i yendigi pencere orani={kazan:.3f}")
        print(f"[6] seed sd={sd:.4f} (CV {sd / m_s.mean() * 100:.2f}%)  MDE(3v3, a=.05, guc=.8)={mde:.4f}"
              f" ({mde / m_s.mean() * 100:.1f}% goreli)")

        # ===== 2. degisken
        oor_cell = ((true < tr_min) | (true > tr_max))      # (N,H,8) hedef train araligi disi
        oor_in = ((X < tr_min) | (X > tr_max))              # girdi
        m_dv_s = np.array([mse(P[s], true, axis=(0, 1)) for s in range(S)])  # (S,8)
        m_dv = m_dv_s.mean(0)
        r_dv = mse(R, true, axis=(0, 1))
        pay = m_dv / m_dv.sum()
        oor_d = oor_cell.mean(axis=(0, 1))
        rho_v, p_v = stats.spearmanr(oor_d, m_dv)
        rho_v2, _ = stats.spearmanr(oor_d, m_dv / r_dv)
        print("[2] degisken: " + " | ".join(
            f"{d}: {m_dv[j]:.3f} (pay {pay[j] * 100:.0f}%, Rep {r_dv[j]:.3f}, M/R {m_dv[j] / r_dv[j]:.2f}, OOR {oor_d[j]:.2f})"
            for j, d in enumerate(DEG)))
        print(f"    Spearman(OOR, MSE)={rho_v:+.2f} (p={p_v:.3f}, n=8)  Spearman(OOR, M/R)={rho_v2:+.2f}")
        # hucre duzeyi: OOR olan vs olmayan hedef hucrelerinde hata (degisken icinde)
        e2 = ((Pm - true) ** 2)
        r2 = ((R - true) ** 2)
        ic = []
        for j in range(8):
            o = oor_cell[..., j]
            if 0.02 < o.mean() < 0.98:
                ic.append(f"{DEG[j]}: OOR {e2[..., j][o].mean():.3f} vs ic {e2[..., j][~o].mean():.3f}"
                          f" (Rep {r2[..., j][o].mean():.3f} vs {r2[..., j][~o].mean():.3f})")
        print("    degisken-ici OOR vs ic hucre MSE (seed-ort tahmin): " + "; ".join(ic))

        # ===== 3. blok ve kayma skoru
        trow = idx  # (N,H)
        blok_id = np.zeros_like(trow)
        for k, (b0, b1) in enumerate(bloklar):
            blok_id[(trow >= b0) & (trow < b1)] = k
        e_cell_s = (P - true[None]) ** 2  # (S,N,H,8)
        bm_s = np.array([[e_cell_s[s][blok_id == k].mean() for k in range(K_BLOK)] for s in range(S)])
        bm_r = np.array([r2[blok_id == k].mean() for k in range(K_BLOK)])
        worst_s = bm_s.max(1)
        ratio_s = worst_s / m_s
        print(f"[3] blok MSE (seed-ort): {np.round(bm_s.mean(0), 4)}  Repeat: {np.round(bm_r, 4)}"
              f"  M/R: {np.round(bm_s.mean(0) / bm_r, 2)}")
        print(f"    worst_block_mse seed={np.round(worst_s, 4)} ort={worst_s.mean():.4f}"
              f" (en kotu blok idx seed: {bm_s.argmax(1)}) | worst/avg ort={ratio_s.mean():.3f}"
              f" | Repeat worst={bm_r.max():.4f} (B{bm_r.argmax() + 1}) worst/avg={bm_r.max() / m_r:.3f}")
        hucre_say = [int((blok_id == k).sum()) * 8 for k in range(K_BLOK)]
        print(f"    blok hucre sayisi: {hucre_say}")
        # kayma skorlari
        mu_lb, sd_lb = X.mean(1), X.std(1)
        mu_hz = true.mean(1)
        s_io = np.mean(np.abs(mu_hz - mu_lb) / (sd_lb + EPS), axis=1)
        s_in = np.mean(np.abs(mu_lb), axis=1)                 # girdinin train ortalamasindan uzakligi (z)
        s_oor_in = oor_in.mean(axis=(1, 2))                   # girdinin train-araligi-disi orani
        s_sd = np.mean(sd_lb, axis=1)                         # girdi oynakligi (train=1)
        rho = {}
        for ad, sk in [("s_io", s_io), ("s_in_mu", s_in), ("s_oor_in", s_oor_in), ("s_sd_lb", s_sd)]:
            rm = [stats.spearmanr(sk, e_win_s[s])[0] for s in range(S)]
            rr = stats.spearmanr(sk, r_win)[0]
            rmr = stats.spearmanr(sk, e_win / r_win)[0]
            rho[ad] = (np.mean(rm), np.std(rm, ddof=1), rr, rmr)
            print(f"    Spearman({ad}, pencere MSE): model ort={np.mean(rm):+.3f} (seed sd {np.std(rm, ddof=1):.3f})"
                  f" Repeat={rr:+.3f}  | ({ad}, M/R oran)={rmr:+.3f}")
        # use_norm cikisi = mu_lb + sd_lb * f(.) -> modelin normalize birimde "ulastigi" seviye kaymasi
        s_io_P = np.array([np.mean(np.abs(P[s].mean(1) - mu_lb) / (sd_lb + EPS), axis=1) for s in range(S)]).mean(0)
        print(f"    s_io medyan: gercek={np.median(s_io):.2f}  tahmin(|mu(P)-mu_lb|/sd_lb)={np.median(s_io_P):.2f}"
              f"  (oran {np.median(s_io_P / s_io):.2f})  girdi sd_lb medyan (z)={np.median(sd_lb):.3f}")
        q = s_io >= np.quantile(s_io, 0.9)
        top_m = np.array([e_win_s[s][q].mean() for s in range(S)])
        print(f"    s_io ust %10 ({q.sum()} pencere): model MSE={top_m.mean():.4f} (x{top_m.mean() / m_s.mean():.2f} ort)"
              f" Repeat={r_win[q].mean():.4f} (x{r_win[q].mean() / m_r:.2f}) M/R={top_m.mean() / r_win[q].mean():.2f}")
        widx = np.arange(N)
        print(f"    Spearman(pencere sirasi, M/R)={stats.spearmanr(widx, e_win / r_win)[0]:+.3f}"
              f"  Spearman(pencere sirasi, s_in_mu)={stats.spearmanr(widx, s_in)[0]:+.3f}"
              f"  Spearman(pencere sirasi, s_sd_lb)={stats.spearmanr(widx, s_sd)[0]:+.3f}")
        # zaman seyri: pencere baslangicina gore 8 dilim
        dil = np.array_split(np.arange(N), min(8, N))
        print("    zaman seyri (pencere sirasi 8 dilim) model/Repeat MSE: " + ", ".join(
            f"{e_win[d_].mean():.3f}/{r_win[d_].mean():.3f}" for d_ in dil))

        # ===== 4. adim bazli
        e_step = ((Pm - true) ** 2).mean(axis=(0, 2))           # seed-ort tahmin (tanisal)
        e_step_s = e_cell_s.mean(axis=(1, 3)).mean(0)           # seed-ortalama hata
        r_step = r2.mean(axis=(0, 2))
        pr_step = ((P - R[None]) ** 2).mean(axis=(1, 3)).mean(0)  # modelin Repeat'ten uzakligi
        tr_step = ((true - R) ** 2).mean(axis=(0, 2))             # gercegin Repeat'ten uzakligi (=r_step)
        dilimler = np.array_split(np.arange(H), 10)
        for k, d_ in enumerate(dilimler):
            adim_satir.append(dict(ufuk=H, adim_dilim=k + 1, adim_bas=int(d_[0]) + 1, adim_bit=int(d_[-1]) + 1,
                                   model_mse=round(float(e_step_s[d_].mean()), 6),
                                   repeat_mse=round(float(r_step[d_].mean()), 6),
                                   oran_M_R=round(float(e_step_s[d_].mean() / r_step[d_].mean()), 4),
                                   model_repeat_uzaklik_mse=round(float(pr_step[d_].mean()), 6),
                                   bias_model=round(float((P - true[None]).mean(axis=(0, 1, 3))[d_].mean()), 5),
                                   bias_repeat=round(float((R - true).mean(axis=(0, 2))[d_].mean()), 5)))
        print("[4] adim-dilimi M/R: " + " ".join(f"{e_step_s[d_].mean() / r_step[d_].mean():.2f}" for d_ in dilimler))
        print(f"    adim1: model {e_step_s[0]:.4f} Repeat {r_step[0]:.4f} | adimH: model {e_step_s[-1]:.4f}"
              f" Repeat {r_step[-1]:.4f} | ||P-R||^2 adimH {pr_step[-1]:.4f} vs ||Y-R||^2 {tr_step[-1]:.4f}")
        # tahmin edilen surukleme vs gercek surukleme (adim H, seed bazli)
        cos_, slope_, sgn_, alpha_ = [], [], [], []
        for s in range(S):
            dp = (P[s][:, -1, :] - last).ravel()
            dt = (true[:, -1, :] - last).ravel()
            sgn_.append(np.mean(np.sign(dp) == np.sign(dt)))
            slope_.append(np.polyfit(dp, dt, 1)[0])
            # tum adimlar: Y-R ~ alpha (P-R) en kucuk kareler (tanisal, val'de fit -> secim icin KULLANILMAZ)
            dP = (P[s] - R).ravel(); dT = (true - R).ravel()
            alpha_.append(dP @ dT / (dP @ dP))
        print(f"    adim H isaret uyumu(P-son, Y-son)={np.mean(sgn_):.3f}  egim(Y-son ~ P-son)={np.mean(slope_):+.3f}"
              f"  alpha*(tum adimlar)={np.mean(alpha_):+.3f} (seed {np.round(alpha_, 3)})")
        # ===== 5. bias / seviye / varyans
        bias_dv = (P - true[None]).mean(axis=(0, 1, 2))           # (8,)
        bias_r = (R - true).mean(axis=(0, 1))
        trend_true = (true[:, -1, :] - true[:, 0, :]).mean(0)
        trend_pred = (P[:, :, -1, :] - P[:, :, 0, :]).mean(axis=(0, 1))
        print("[5] bias (P-Y) degisken: " + " ".join(f"{d}:{bias_dv[j]:+.3f}" for j, d in enumerate(DEG))
              + f" | ort {bias_dv.mean():+.3f}  Repeat bias ort {bias_r.mean():+.3f}")
        print("    ufuk-ici trend (adimH-adim1) gercek/tahmin: " + " ".join(
            f"{d}:{trend_true[j]:+.3f}/{trend_pred[j]:+.3f}" for j, d in enumerate(DEG)))
        # seviye capasi: P adim1 - mu_lb = beta (son - mu_lb)
        betas, gap1 = [], []
        for s in range(S):
            xx = (last - mu_lb).ravel(); yy = (P[s][:, 0, :] - mu_lb).ravel()
            betas.append(xx @ yy / (xx @ xx))
            gap1.append(np.mean(np.abs(P[s][:, 0, :] - last)))
        print(f"    seviye capasi beta (P1-mu_lb ~ beta*(son-mu_lb))={np.mean(betas):.3f}"
              f"  |P1-son| ort={np.mean(gap1):.4f}  |Y1-son| ort={np.mean(np.abs(true[:, 0, :] - last)):.4f}")
        # P1-son kaymasi ortalamaya dogru mu? corr(P1-son, mu_lb-son)
        cc = [np.corrcoef((P[s][:, 0, :] - last).ravel(), (mu_lb - last).ravel())[0, 1] for s in range(S)]
        print(f"    corr(P1-son, mu_lb-son)={np.mean(cc):+.3f}  isaretli ort(P1-son)={np.mean(P[:, :, 0, :] - last):+.4f}")
        # TANISAL son-islem (yalniz girdi bilgisi kullanir; val'de olculur, K4 karti olmadan iddia degil)
        Pa = P - P[:, :, :1, :] + last[None, :, None, :]       # adim-1'i son degere yeniden capala
        Pb = 0.5 * (P + R[None])                               # Repeat ile yari-yariya karisim
        ma = np.array([mse(Pa[s], true) for s in range(S)]); mb = np.array([mse(Pb[s], true) for s in range(S)])
        print(f"    TANISAL: yeniden-capa MSE={ma.mean():.4f} ({(ma.mean() / m_s.mean() - 1) * 100:+.1f}% vs model,"
              f" {(ma.mean() / m_r - 1) * 100:+.1f}% vs Rep)  0.5-karisim MSE={mb.mean():.4f}"
              f" ({(mb.mean() / m_s.mean() - 1) * 100:+.1f}% vs model)")
        # varyans orani: pencere-ici (adimlar boyunca) std
        vr_in = np.array([np.mean(P[s].std(1)) / np.mean(true.std(1)) for s in range(S)])
        vr_in_dv = P.std(2).mean(axis=(0, 1)) / true.std(1).mean(0)
        # pencereler-arasi: adim H'de tahmin degisiminin (P-son) std'si / gercek (Y-son) std'si
        vr_cross = np.array([(P[s][:, -1, :] - last).std() / (true[:, -1, :] - last).std() for s in range(S)])
        print(f"    varyans orani pencere-ici std(P)/std(Y)={vr_in.mean():.3f}  degisken: "
              + " ".join(f"{d}:{vr_in_dv[j]:.2f}" for j, d in enumerate(DEG))
              + f" | std(P_H-son)/std(Y_H-son)={vr_cross.mean():.3f}")

        ozet[H] = dict(m=m_s.mean(), sd=sd, r=m_r, ms=m_s)
        # ===== CSV: ufuk x degisken(+TUM) x blok(+TUM)
        for j in list(range(8)) + [None]:
            for k in list(range(K_BLOK)) + [None]:
                msk = np.ones((N, H), bool) if k is None else (blok_id == k)
                sl = slice(None) if j is None else j
                es = np.array([e_cell_s[s][msk][..., sl].mean() for s in range(S)])
                er = r2[msk][..., sl].mean()
                bi = float((P - true[None])[:, msk][..., sl].mean())
                br = float((R - true)[msk][..., sl].mean())
                oo = float(oor_cell[msk][..., sl].mean())
                n_c = int(msk.sum()) * (8 if j is None else 1)
                b_ad = "TUM" if k is None else f"B{k + 1}"
                row = dict(etiket=a.etiket, bolum="val", ufuk=H, degisken="TUM" if j is None else DEG[j], blok=b_ad,
                           blok_satir="" if k is None else f"[{bloklar[k][0]},{bloklar[k][1]})",
                           n_hucre=n_c, model_mse_ort=round(float(es.mean()), 6),
                           model_mse_sd=round(float(es.std(ddof=1)), 6) if S > 1 else "",
                           **{f"model_mse_s{i}": round(float(es[i]), 6) for i in range(S)},
                           repeat_mse=round(float(er), 6),
                           delta_M_R=round(float(es.mean() - er), 6),
                           oran_M_R=round(float(es.mean() / er), 4),
                           bias_model=round(bi, 5), bias_repeat=round(br, 5),
                           hedef_train_araligi_disi_oran=round(oo, 4))
                kirilim.append(row)

    ort_m = np.mean([ozet[H]["m"] for H in UFUKLAR]); ort_r = np.mean([ozet[H]["r"] for H in UFUKLAR])
    print(f"\n4-ufuk ort: model {ort_m:.4f} Repeat {ort_r:.4f} ({(ort_m / ort_r - 1) * 100:+.1f}%)"
          f"  H720 payi model {ozet[720]['m'] / (4 * ort_m) * 100:.0f}%")
    avg_s = np.mean([ozet[H]["ms"] for H in UFUKLAR], axis=0)
    S = len(avg_s)
    if S > 1:
        sd_a = avg_s.std(ddof=1)
        mde_a = (stats.t.ppf(0.975, 2 * S - 2) + stats.t.ppf(0.8, 2 * S - 2)) * sd_a * np.sqrt(2 / S)
        print(f"val_mse_avg seed={np.round(avg_s, 4)} sd={sd_a:.4f} MDE(3v3)={mde_a:.4f} ({mde_a / avg_s.mean() * 100:.1f}%)")

    out = WS / "sonuclar"
    for ad, rows in [(f"hata-kirilim{a.cikti_ek}.csv", kirilim), (f"hata-adim{a.cikti_ek}.csv", adim_satir)]:
        with open(out / ad, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"yazildi: {out / ad} ({len(rows)} satir)")


if __name__ == "__main__":
    sys.exit(main())
