# -*- coding: utf-8 -*-
"""K4c tur-4 (FAZ II) tahlili -- YALNIZ VALIDATION, kat protokolu. deney-tahlilci mod 2.
Onceki tahlil ciktilarinin USTUNE YAZMAZ (yalniz *tur4* dosyalari uretir).

Calistirma (WS kokunden):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/tahlil_tur4.py > sonuclar/tahlil_tur4.log

TEST KILIDI: metrics.json'dan yalniz 'test' ile baslamayan anahtarlar alinir (assert). preds/test_kilitli/,
test_metrics.json, leaderboard.csv, sonuclar/test-*, naif-test.csv, karar.md, ablasyon.* OKUNMAZ.
Tahmin dosyasi okunmaz; tum sayilar metrics.json (val) + sonuclar/naif-kat*.csv + sonuclar/h11-kapi.csv'den.

Karar kurali (FAZ II sef on-kaydi; H11-H13 kartlari):
  A(V,k) = ort_s ort_H m ; Abar = ort_k A ; Dbar = Abar(V) - Abar(lastnorm)
  D_k = A(V,k) - A(lastnorm,k) ; D_H = ort_k ort_s [m(V) - m(lastnorm)]_H
  TUT <=> Dbar < 0 & #{k: D_k<0} >= 2 & #{H: D_H<0} >= 3
Ciktilar: sonuclar/tahlil-tur4-kat.csv (varyant x kat x ufuk x seed), sonuclar/tahlil-tur4-mekanizma.csv
          (H12/H13 kat x ufuk x seed agirlik/alfa ozetleri).
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

WS = Path(__file__).resolve().parents[1]
H_ = [96, 192, 336, 720]
SEEDS = [2023, 2024, 2025]
KATLAR = [1, 2, 3]
R = "20261014-"
KOSUM = {
    ("baseline", 3): ["020531_baseline_s2023", "021024_baseline_s2024", "021448_baseline_s2025"],
    ("baseline", 2): ["162123_baseline-k2_s2023", "162312_baseline-k2_s2024", "162501_baseline-k2_s2025"],
    ("baseline", 1): ["162640_baseline-k1_s2023", "162755_baseline-k1_s2024", "162916_baseline-k1_s2025"],
    ("lastnorm", 3): ["052508_lastnorm_s2023", "053601_lastnorm_s2024", "054213_lastnorm_s2025"],
    ("lastnorm", 2): ["163051_lastnorm-k2_s2023", "163203_lastnorm-k2_s2024", "163345_lastnorm-k2_s2025"],
    ("lastnorm", 1): ["163510_lastnorm-k1_s2023", "163633_lastnorm-k1_s2024", "163802_lastnorm-k1_s2025"],
    ("lastnorm", 0): ["170039_lastnorm-k0_s2023", "170128_lastnorm-k0_s2024", "170216_lastnorm-k0_s2025"],
    ("lastnorm-birlesim", 3): ["170302_lastnorm-birlesim-k3_s2023", "170306_lastnorm-birlesim-k3_s2024",
                               "170310_lastnorm-birlesim-k3_s2025"],
    ("lastnorm-birlesim", 2): ["170314_lastnorm-birlesim-k2_s2023", "170317_lastnorm-birlesim-k2_s2024",
                               "170321_lastnorm-birlesim-k2_s2025"],
    ("lastnorm-birlesim", 1): ["170325_lastnorm-birlesim-k1_s2023", "170329_lastnorm-birlesim-k1_s2024",
                               "170332_lastnorm-birlesim-k1_s2025"],
    ("lastnorm-olcek", 3): ["170336_lastnorm-olcek-k3_s2023", "170340_lastnorm-olcek-k3_s2024",
                            "170343_lastnorm-olcek-k3_s2025"],
    ("lastnorm-olcek", 2): ["170346_lastnorm-olcek-k2_s2023", "170350_lastnorm-olcek-k2_s2024",
                            "170354_lastnorm-olcek-k2_s2025"],
    ("lastnorm-olcek", 1): ["170358_lastnorm-olcek-k1_s2023", "170401_lastnorm-olcek-k1_s2024",
                            "170405_lastnorm-olcek-k1_s2025"],
}
KOSUM = {k: [R + r for r in v] for k, v in KOSUM.items()}
KART = {"lastnorm-birlesim": "H12", "lastnorm-olcek": "H13", "lastnorm": "H04", "baseline": "rakip"}
EBEVEYN = "lastnorm"
INCELE = {"lastnorm-birlesim": dict(avg=-0.06), "lastnorm-olcek": dict(avg=-0.15, h720=-0.30)}


def val_metrik(run):
    d = json.loads((WS / "runs" / run / "metrics.json").read_text(encoding="utf-8"))
    out = {k: v for k, v in d.items() if not k.lower().startswith("test")}
    assert all(not k.lower().startswith("test") for k in out)
    out.pop("alt_grup", None)
    return out


def naif():
    n = {}
    for f in ["naif-kat.csv", "naif-kat_k0.csv"]:
        for r in csv.DictReader(open(WS / "sonuclar" / f, encoding="utf-8")):
            n[(int(r["kat"]), int(r["ufuk"]))] = float(r["mse"])
    for k in [0, 1, 2, 3]:
        n[(k, "avg")] = np.mean([n[(k, H)] for H in H_])
    return n


def m_arr(M, v, k, H):
    key = "val_mse_avg" if H == "avg" else f"val_mse_h{H}"
    return np.array([M[(v, k)][i][key] for i in range(3)])


def ptt(x, y):
    d = x - y
    return float("nan") if np.allclose(d, 0) else float(stats.ttest_rel(x, y).pvalue)


def wil(d):
    d = np.asarray(d)
    if np.allclose(d, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


def karar(M, v, ref):
    Ak = {k: m_arr(M, v, k, "avg").mean() for k in KATLAR}
    Ar = {k: m_arr(M, ref, k, "avg").mean() for k in KATLAR}
    Abar, Abar_r = np.mean(list(Ak.values())), np.mean(list(Ar.values()))
    Dk = {k: Ak[k] - Ar[k] for k in KATLAR}
    DH = {H: np.mean([(m_arr(M, v, k, H) - m_arr(M, ref, k, H)).mean() for k in KATLAR]) for H in H_}
    DHrel = {H: DH[H] / np.mean([m_arr(M, ref, k, H).mean() for k in KATLAR]) for H in H_}
    nk = sum(d < 0 for d in Dk.values())
    nh = sum(d < 0 for d in DH.values())
    tut = (Abar - Abar_r < 0) and nk >= 2 and nh >= 3
    # 9 (kat, seed) cifti
    x = np.concatenate([m_arr(M, v, k, "avg") for k in KATLAR])
    y = np.concatenate([m_arr(M, ref, k, "avg") for k in KATLAR])
    lr = np.log(x / y)
    return dict(Ak=Ak, Ar=Ar, Abar=Abar, Abar_r=Abar_r, Dbar=Abar - Abar_r, Dbar_rel=(Abar - Abar_r) / Abar_r,
                Dk=Dk, Dk_rel={k: Dk[k] / Ar[k] for k in KATLAR}, DH=DH, DHrel=DHrel, nk=nk, nh=nh, tut=tut,
                p9=ptt(x, y), p9_log=float(stats.ttest_1samp(lr, 0).pvalue) if not np.allclose(lr, 0) else float("nan"),
                w9=wil(x - y), n9neg=int((x - y < 0).sum()), lr_mean=float(np.exp(lr.mean()) - 1),
                pk={k: ptt(m_arr(M, v, k, "avg"), m_arr(M, ref, k, "avg")) for k in KATLAR})


def yaz_karar(ad, r, ref):
    print(f"\n### {ad} vs {ref}")
    for k in KATLAR:
        print(f"  K{k}: A {r['Ak'][k]:.5f} vs {r['Ar'][k]:.5f}  D_k {r['Dk'][k]:+.5f} ({r['Dk_rel'][k] * 100:+.2f}%)"
              f"  p(3 seed, esl.) {r['pk'][k]:.4g}")
    print(f"  Abar {r['Abar']:.5f} vs {r['Abar_r']:.5f}  Dbar {r['Dbar']:+.5f} ({r['Dbar_rel'] * 100:+.2f}%)")
    print("  D_H (kat-ort.): " + " ".join(f"h{H} {r['DH'][H]:+.5f} ({r['DHrel'][H] * 100:+.2f}%)" for H in H_))
    print(f"  #{{k: D_k<0}} = {r['nk']}/3   #{{H: D_H<0}} = {r['nh']}/4   -> {'TUT' if r['tut'] else 'OLDUR'}")
    print(f"  9 cift (kat x seed): {r['n9neg']}/9 negatif; esl. t p {r['p9']:.4g}; log-oran t p {r['p9_log']:.4g}"
          f" (geo-ort. oran {r['lr_mean'] * 100:+.2f}%); Wilcoxon p {r['w9']:.4g}")


def main():
    M = {key: [val_metrik(r) for r in runs] for key, runs in KOSUM.items()}
    for key, ms in M.items():
        assert [m["seed"] for m in ms] == SEEDS, key
        if "kat" in ms[0] and ms[0]["kat"] is not None:
            assert all(m["kat"] == key[1] for m in ms), key
    N = naif()
    print("# Tur 4 tahlili (FAZ II, kat protokolu) -- yalniz val\n")
    print("## Kosum listesi")
    for key, runs in KOSUM.items():
        print(f"  {key[0]:18s} K{key[1]}: {', '.join(r[9:] for r in runs)}")

    # ---------- 1. tablo: varyant x kat x ufuk
    print("\n## 1. Kat x ufuk val MSE (3 seed ort. +- sd) ve naif orani M/R")
    for v in ["baseline", "lastnorm", "lastnorm-birlesim", "lastnorm-olcek"]:
        print(f"\n[{v}]")
        for k in KATLAR:
            s = []
            for H in H_ + ["avg"]:
                a = m_arr(M, v, k, H)
                s.append(f"h{H} {a.mean():.5f}+-{a.std(ddof=1):.5f} (M/R {a.mean() / N[(k, H)]:.3f})")
            print(f"  K{k}: " + " | ".join(s))
    a0 = {H: m_arr(M, "lastnorm", 0, H) for H in H_ + ["avg"]}
    print("\n[lastnorm K0 (yalniz ic model; karar disi bilgi)]  " + " | ".join(
        f"h{H} {a0[H].mean():.5f} (M/R {a0[H].mean() / N[(0, H)]:.3f})" for H in H_ + ["avg"]))
    print("[naif Repeat]  " + " ; ".join(
        f"K{k}: " + " ".join(f"{N[(k, H)]:.5f}" for H in H_ + ["avg"]) for k in [0, 1, 2, 3]))

    # ---------- 2. kart kararlari
    print("\n## 2. Kart kararlari (ebeveyn lastnorm; kural on-kayitli)")
    sonuc = {}
    for v in ["lastnorm-birlesim", "lastnorm-olcek"]:
        r = karar(M, v, EBEVEYN)
        sonuc[v] = r
        yaz_karar(f"{KART[v]} {v}", r, EBEVEYN)
        rb = karar(M, v, "baseline")
        yaz_karar(f"{KART[v]} {v} (rakip iddiasi, karar disi)", rb, "baseline")
        inc = INCELE[v]
        bay = r["Dbar_rel"] <= inc["avg"] or ("h720" in inc and r["DHrel"][720] <= inc["h720"])
        print(f"  INCELE esigi (Dbar<= {inc['avg'] * 100:.0f}%{', h720<=-30%' if 'h720' in inc else ''}): "
              f"{'ASILDI' if bay else 'asilmadi'}")
        print(f"  K1 ^ K2 alt-sonucu (D_1<0 ^ D_2<0): {r['Dk'][1] < 0 and r['Dk'][2] < 0}")
        mr_cells = [(k, H, m_arr(M, v, k, H).mean() / N[(k, H)]) for k in KATLAR for H in H_]
        print(f"  naif hedefi: {sum(c[2] <= 1.0 for c in mr_cells)}/12 hucre M/R<=1; kat M/R "
              + " ".join(f"K{k} {r['Ak'][k] / N[(k, 'avg')]:.4f}" for k in KATLAR))
    # H12 vs H13 farki
    d = np.concatenate([m_arr(M, "lastnorm-olcek", k, "avg") - m_arr(M, "lastnorm-birlesim", k, "avg") for k in KATLAR])
    print(f"\n  H13 - H12 (9 cift avg): {np.round(d, 5)}  bit-bit esit cift: {int((d == 0).sum())}/9")
    for k in KATLAR:
        for H in H_:
            eq = np.array_equal(m_arr(M, "lastnorm-olcek", k, H), m_arr(M, "lastnorm-birlesim", k, H))
            nv = np.allclose(m_arr(M, "lastnorm-birlesim", k, H), N[(k, H)], rtol=2e-5)
            print(f"    K{k} h{H}: H13==H12 {eq}  H12~=Repeat {nv}", end=";")
        print()

    # ---------- 3. kat protokolu bulgusu
    print("\n## 3. Kat protokolu bulgusu: baseline vs lastnorm vs naif")
    r = karar(M, "lastnorm", "baseline")
    yaz_karar("lastnorm (H04)", r, "baseline")
    print("  kat x ufuk 12 hucre (lastnorm - baseline, 3 seed esl.):")
    hucre_neg = 0
    for k in KATLAR:
        s = []
        for H in H_:
            x, y = m_arr(M, "lastnorm", k, H), m_arr(M, "baseline", k, H)
            hucre_neg += (x.mean() < y.mean())
            s.append(f"h{H} {(x.mean() / y.mean() - 1) * 100:+.2f}% ({int((x < y).sum())}/3, p {ptt(x, y):.3g})")
        print(f"    K{k}: " + " | ".join(s))
    print(f"  hucre: {hucre_neg}/12 lastnorm < baseline")
    # 36 (kat, ufuk, seed) cifti, goreli
    lr36 = np.array([np.log(m_arr(M, "lastnorm", k, H) / m_arr(M, "baseline", k, H)) for k in KATLAR for H in H_]).ravel()
    print(f"  36 (kat x ufuk x seed) log-oran: {int((lr36 < 0).sum())}/36 negatif; geo-ort {np.expm1(lr36.mean()) * 100:+.2f}%;"
          f" Wilcoxon p {wil(lr36):.3g} (bagimsizlik varsayimi iyimser: seed ve ufuk ic ice)")
    print(f"  kat birim (n=3) isaret testi: 3/3 -> tek-yonlu binom p = {0.5 ** 3:.3f}")

    print("\n  Naife karsi (M/R = model/Repeat; Repeat deterministik):")
    for v in ["baseline", "lastnorm"]:
        for k in KATLAR:
            s = []
            for H in H_ + ["avg"]:
                a = m_arr(M, v, k, H)
                t = stats.ttest_1samp(a, N[(k, H)])
                s.append(f"h{H} {a.mean() / N[(k, H)]:.3f} [{(a / N[(k, H)]).min():.3f},{(a / N[(k, H)]).max():.3f}] p {t.pvalue:.2g}")
            print(f"    {v:9s} K{k}: " + " | ".join(s))
        cells = sum(m_arr(M, v, k, H).mean() < N[(k, H)] for k in KATLAR for H in H_)
        kat_ok = sum(m_arr(M, v, k, "avg").mean() < N[(k, "avg")] for k in KATLAR)
        Abar = np.mean([m_arr(M, v, k, "avg").mean() for k in KATLAR])
        Nbar = np.mean([N[(k, "avg")] for k in KATLAR])
        lrN = np.array([np.log(m_arr(M, v, k, "avg") / N[(k, "avg")]) for k in KATLAR]).ravel()
        print(f"    {v}: naifi gecen hucre {cells}/12, kat {kat_ok}/3; Abar {Abar:.5f} vs naif {Nbar:.5f} "
              f"(M/R {Abar / Nbar:.4f}); kat-esit agirlikli geo-ort M/R {np.exp(lrN.mean()):.4f}")
    # ufuk bazinda (kat-ort.) M/R
    for v in ["baseline", "lastnorm"]:
        print(f"    {v} ufuk M/R (kat-ort. MSE / kat-ort. Repeat): " + " ".join(
            f"h{H} {np.mean([m_arr(M, v, k, H).mean() for k in KATLAR]) / np.mean([N[(k, H)] for k in KATLAR]):.3f}"
            for H in H_))

    # ---------- 4. H12/H13 mekanizma
    print("\n## 4. H12/H13 mekanizma (ic-holdout alfa vs kat-val oracle)")
    mek = []
    for v, kw in [("lastnorm-birlesim", "agirliklar"), ("lastnorm-olcek", "alfa")]:
        for k in KATLAR:
            for i, s in enumerate(SEEDS):
                m = M[(v, k)][i]
                for H in H_:
                    w = np.array(m[f"{kw}_h{H}"])
                    ah = np.array(m[f"alfa_ham_h{H}"])
                    ao = np.array(m[f"alfa_val_oracle_h{H}"])
                    mek.append(dict(varyant=v, kart=KART[v], kat=k, seed=s, ufuk=H, ic_model=m["turev_kaynak"][0],
                                    w_ort=w.mean(), w_sifir_orani=float((w == 0).mean()),
                                    alfa_ic_medyan=float(np.median(ah)), alfa_val_medyan=float(np.median(ao)),
                                    isaret_uyum=float((np.sign(ah) == np.sign(ao)).mean()),
                                    spearman=m[f"spearman_alfa_ic_val_h{H}"], V_MSEP=m[f"V_bolu_MSEP_h{H}"],
                                    gozlenen_kazanc=m[f"gozlenen_kazanc_orani_h{H}"],
                                    oracle_dilim=m[f"oracle_kazanc_orani_dilim_h{H}"],
                                    oracle_tek=m[f"oracle_kazanc_orani_tek_alfa_h{H}"]))
        print(f"  [{KART[v]} {v}] oracle ihlal (dilim) toplam: "
              f"{sum(M[(v, k)][i]['oracle_ihlal_sayisi'] for k in KATLAR for i in range(3))}; "
              f"w=0 orani (kat): " + " ".join(f"K{k} {np.mean([M[(v, k)][i]['w_sifir_orani'] for i in range(3)]):.2f}" for k in KATLAR)
              + "; w=1 orani: " + " ".join(f"K{k} {np.mean([M[(v, k)][i]['w_bir_orani'] for i in range(3)]):.2f}" for k in KATLAR)
              + "; alfa in [0.95,1.05]: " + " ".join(f"K{k} {np.mean([M[(v, k)][i]['w_095_105_orani'] for i in range(3)]):.2f}" for k in KATLAR))
    print("  kat x ufuk (3 seed ort.; birlesim): w_ort | w=0 | alfa_ic med | alfa_val med | isaret uyumu | Spearman | gozlenen | oracle(dilim)")
    for k in KATLAR:
        for H in H_:
            rr = [x for x in mek if x["varyant"] == "lastnorm-birlesim" and x["kat"] == k and x["ufuk"] == H]
            f = lambda c: np.mean([x[c] for x in rr])
            print(f"    K{k} h{H}: {f('w_ort'):.3f} | {f('w_sifir_orani'):.2f} | {f('alfa_ic_medyan'):+.2f} | "
                  f"{f('alfa_val_medyan'):+.2f} | {f('isaret_uyum'):.2f} | {f('spearman'):+.2f} | "
                  f"{f('gozlenen_kazanc') * 100:+.2f}% | {f('oracle_dilim') * 100:+.2f}%")
    # ic modelin kendi val'indeki M/R vs hedef kattaki M/R (rejim degisimi)
    print("\n  Rejim: lastnorm M/R ic-holdout katinda (K_{k-1}) -> hedef katta (K_k):")
    for k in KATLAR:
        print(f"    K{k - 1}->K{k}: " + " | ".join(
            f"h{H} {m_arr(M, 'lastnorm', k - 1, H).mean() / N[(k - 1, H)]:.3f} -> {m_arr(M, 'lastnorm', k, H).mean() / N[(k, H)]:.3f}"
            for H in H_))
    with open(WS / "sonuclar/tahlil-tur4-mekanizma.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mek[0].keys()))
        w.writeheader()
        for x in mek:
            w.writerow({kk: (f"{vv:.6g}" if isinstance(vv, float) else vv) for kk, vv in x.items()})

    # ---------- 5. H11 kapi
    print("\n## 5. H11 kapi (sonuclar/h11-kapi.csv)")
    for rw in csv.DictReader(open(WS / "sonuclar/h11-kapi.csv", encoding="utf-8")):
        if rw["kat"]:
            print(f"  K{rw['kat']} h{rw['ufuk']} {rw['kol']:8s} olu {float(rw['olu_orani']):.3f} q {float(rw['q']):.2e} "
                  f"sigma_hat {float(rw['sigma_hat_ort']):.4f} gercek {float(rw['sigma_gercek_ort']):.4f} R {float(rw['R']):.3f}")
        else:
            print("  " + rw["kapi"])

    # ---------- 6. uzun CSV
    rows = []
    for (v, k), runs in KOSUM.items():
        for i, s in enumerate(SEEDS):
            for H in H_ + ["avg"]:
                key = "val_mse_avg" if H == "avg" else f"val_mse_h{H}"
                x = M[(v, k)][i][key]
                row = dict(varyant=v, kart=KART[v], kat=k, ufuk=H, seed=s, run_id=runs[i], val_mse=x,
                           naif_mse=N[(k, H)], M_R=x / N[(k, H)])
                if (EBEVEYN, k) in M and v != EBEVEYN:
                    e = M[(EBEVEYN, k)][i][key]
                    row.update(ebeveyn_mse=e, fark_ebeveyn=x - e, fark_ebeveyn_rel=x / e - 1)
                if ("baseline", k) in M and v != "baseline":
                    b = M[("baseline", k)][i][key]
                    row.update(baseline_mse=b, fark_baseline=x - b, fark_baseline_rel=x / b - 1)
                rows.append(row)
    alan = ["varyant", "kart", "kat", "ufuk", "seed", "run_id", "val_mse", "naif_mse", "M_R", "ebeveyn_mse",
            "fark_ebeveyn", "fark_ebeveyn_rel", "baseline_mse", "fark_baseline", "fark_baseline_rel"]
    with open(WS / "sonuclar/tahlil-tur4-kat.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=alan)
        w.writeheader()
        for x in rows:
            w.writerow({kk: (f"{vv:.6g}" if isinstance(vv, float) else vv) for kk, vv in x.items()})
    print(f"\nCSV: sonuclar/tahlil-tur4-kat.csv ({len(rows)} satir), sonuclar/tahlil-tur4-mekanizma.csv ({len(mek)} satir)")


if __name__ == "__main__":
    main()
