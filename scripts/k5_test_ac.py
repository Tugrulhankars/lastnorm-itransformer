#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
k5_test_ac.py — K5 ablasyon + TEK SEFERLIK test kilidi acilisi (yeni koşum YOK).

Iki asama (sira zorunlu):
  python scripts/k5_test_ac.py --asama val    -> sonuclar/ablasyon.csv (yalniz validation sutunlari).
                                                 test_metrics.json / preds/test_kilitli OKUNMAZ; metrics.json'dan
                                                 'test' ile baslayan anahtarlar suzulur.
  python scripts/k5_test_ac.py --asama test   -> TEST KILIDI ACILIR (bir kez): on-kayitli kumenin (orkestrasyon_durumu
                                                 'K5 SEF ON-KAYDI') tam koşumlarinin test_metrics.json dosyalari tek
                                                 oturumda okunur -> sonuclar/test-sonuclari.csv, leaderboard-test.md,
                                                 ablasyon.csv (test sutunlari eklenir), test-acilis.log (zaman + md5).
                                                 sonuclar/test-acilis.log varsa REDDEDER (ikinci acilis yasak, D1).

Koşum listesi tahlil-tur1/2/3.md'deki tam koşum listeleridir (asama 3, rc=0; yetim rc=96 haric). Liste test
gorulmeden once sabitlendi; betik her koşumun .done rc=0 ve config varyant/seed tutarliligini assert eder.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
KOK = Path(__file__).resolve().parents[1]
RUNS = KOK / "runs"
SON = KOK / "sonuclar"
UFUK = [96, 192, 336, 720]
SEEDS = [2023, 2024, 2025]

# On-kayitli okunacak kume (test gorulmeden sabit). Kaynak: tahlil-tur1/2/3.md "Tam koşumlar".
KOSUMLAR = {
    "baseline":         ["20261014-020531_baseline_s2023", "20261014-021024_baseline_s2024", "20261014-021448_baseline_s2025"],
    "baseline-ep336eq": ["20261014-031233_baseline-ep336eq_s2023", "20261014-032853_baseline-ep336eq_s2024", "20261014-033500_baseline-ep336eq_s2025"],
    "revin-afin":       ["20261014-034304_revin-afin_s2023", "20261014-034735_revin-afin_s2024", "20261014-035213_revin-afin_s2025"],
    "san":              ["20261014-040042_san_s2023", "20261014-040547_san_s2024", "20261014-041102_san_s2025"],
    "lastnorm":         ["20261014-052508_lastnorm_s2023", "20261014-053601_lastnorm_s2024", "20261014-054213_lastnorm_s2025"],
    "normsuz":          ["20261014-055518_normsuz_s2023", "20261014-060205_normsuz_s2024", "20261014-060859_normsuz_s2025"],
    "xmark-yok":        ["20261014-061948_xmark-yok_s2023", "20261014-062637_xmark-yok_s2024", "20261014-063215_xmark-yok_s2025"],
    "lastnorm-ep336eq": ["20261014-140921_lastnorm-ep336eq_s2023", "20261014-141641_lastnorm-ep336eq_s2024", "20261014-142407_lastnorm-ep336eq_s2025"],
    "lastnorm-merkez":  ["20261014-143314_lastnorm-merkez_s2023", "20261014-143820_lastnorm-merkez_s2024", "20261014-144341_lastnorm-merkez_s2025"],
    "lastnorm-dro":     ["20261014-144849_lastnorm-dro_s2023", "20261014-145213_lastnorm-dro_s2024", "20261014-151234_lastnorm-dro_s2025"],
}
# (varyant, referans, rol) — birincil iki karsilastirma on-kayitli; digerleri tablo/analiz satiri
KARSILASTIRMA = [
    ("lastnorm", "baseline", "BIRINCIL-1 (resmi protokol)"),
    ("lastnorm-ep336eq", "baseline-ep336eq", "BIRINCIL-2 (epoch-esit protokol)"),
    ("baseline-ep336eq", "baseline", "protokol (H03)"),
    ("lastnorm-ep336eq", "lastnorm", "protokol (H08 S1)"),
    ("lastnorm-merkez", "lastnorm", "ablasyon: lastnorm - sigma (H10)"),
    ("lastnorm-dro", "lastnorm", "basarisiz deneme (H09)"),
    ("revin-afin", "baseline", "tablo (H02)"),
    ("san", "baseline", "tablo (H01, ariza notlu)"),
    ("normsuz", "baseline", "tablo/ablasyon (H06)"),
    ("xmark-yok", "baseline", "tablo (H07)"),
    ("lastnorm-merkez", "baseline", "tablo"),
    ("lastnorm-dro", "baseline", "tablo"),
    ("lastnorm-ep336eq", "baseline", "tablo"),
]
# Ablasyon satirlari (kazanan lastnorm; zincir: baseline -> lastnorm; protokol eki ep336eq)
ABLASYON = [
    ("baseline", "baseline (NST, mu+sigma; resmi)", "baseline"),
    ("lastnorm", "lastnorm = KAZANAN (son-deger capa + sigma)", "baseline"),
    ("lastnorm-merkez", "lastnorm - sigma (yalniz son-deger merkezleme, H10)", "lastnorm"),
    ("lastnorm-ep336eq", "lastnorm + ep336eq (h336 10 epoch, H08)", "lastnorm"),
    ("baseline-ep336eq", "baseline + ep336eq (H03)", "baseline"),
    ("normsuz", "normalizasyon yok (H06; capa ve sigma ikisi de geri alinmis)", "baseline"),
    ("revin-afin", "RevIN-afin (mu+sigma + gamma/beta, H02)", "baseline"),
]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def denetle(varyant, rid):
    d = RUNS / rid
    done = json.loads((d / ".done").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert done.get("rc") == 0 and done.get("metrics_var"), f"{rid}: rc!=0 / metrik yok"
    assert cfg["varyant"] == varyant, f"{rid}: varyant {cfg['varyant']} != {varyant}"
    assert f"_s{cfg['seed']}" in rid
    return d, int(cfg["seed"])


def val_oku(d: Path):
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    m = {k: v for k, v in m.items() if not k.startswith("test")}  # test kilidi (K3 istisnasi dahil suzulur)
    assert not any(k.startswith("test") for k in m)
    out = {"val_mse_avg": m["val_mse_avg"], "val_mae_avg": m["val_mae_avg"]}
    for h in UFUK:
        out[f"val_mse_h{h}"] = m[f"val_mse_h{h}"]
        out[f"val_mae_h{h}"] = m[f"val_mae_h{h}"]
    return out


# ---------- istatistik ----------
def t_p(diffs):
    """eslestirilmis t (iki yonlu). scipy varsa onu kullanir."""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    sd = stdev(diffs)
    if sd == 0:
        return (float("inf") if mean(diffs) else float("nan")), (0.0 if mean(diffs) else 1.0)
    t = mean(diffs) / (sd / math.sqrt(n))
    try:
        from scipy import stats
        return t, float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        return t, float("nan")


def t_ci(diffs, alpha=0.05):
    n = len(diffs)
    try:
        from scipy import stats
        q = stats.t.ppf(1 - alpha / 2, n - 1)
    except Exception:
        q = {2: 4.303}.get(n - 1, 1.96)
    h = q * stdev(diffs) / math.sqrt(n)
    return mean(diffs) - h, mean(diffs) + h


def boot_ci(diffs):
    """tam bootstrap (n=3 -> 27 esit olasilikli yeniden ornek); %2.5/%97.5 yuzdelik. n=3'te dejenere: bilgi amacli."""
    ms = sorted(mean(c) for c in itertools.product(diffs, repeat=len(diffs)))
    k = len(ms)
    return ms[max(0, int(math.floor(0.025 * k)))], ms[min(k - 1, int(math.ceil(0.975 * k)) - 1)]


def wilcoxon_p(diffs):
    """kesin isaretli-sira (iki yonlu). n=3'te ulasilabilecek en kucuk p = 0.25."""
    n = len(diffs)
    ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
    r = [0] * n
    for k, i in enumerate(ranks):
        r[i] = k + 1
    wp = sum(r[i] for i in range(n) if diffs[i] > 0)
    dist = [sum(rr for rr, s in zip(range(1, n + 1), signs) if s) for signs in itertools.product([0, 1], repeat=n)]
    lo = sum(1 for x in dist if x <= wp) / len(dist)
    hi = sum(1 for x in dist if x >= wp) / len(dist)
    return min(1.0, 2 * min(lo, hi))


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ---------- asamalar ----------
def asama_val():
    rows = {}
    for v, ids in KOSUMLAR.items():
        rows[v] = {}
        for rid in ids:
            d, s = denetle(v, rid)
            rows[v][s] = val_oku(d)
    return rows


def ozet(rows, v, key):
    xs = [rows[v][s][key] for s in SEEDS]
    return mean(xs), stdev(xs), xs


def ablasyon_yaz(val, test=None):
    alanlar = ["satir", "varyant", "aciklama", "referans", "n", "kosum_idler"]
    metr = ["mse_avg"] + [f"mse_h{h}" for h in UFUK]
    for sp, src in (("val", val), ("test", test)):
        if src is None:
            continue
        for m in metr:
            alanlar += [f"{sp}_{m}_ort", f"{sp}_{m}_std"]
        alanlar += [f"{sp}_mae_avg_ort", f"{sp}_d_avg_vs_ref_%", f"{sp}_p_vs_ref", f"{sp}_ufuk_iyi_vs_ref"]
    out = []
    for i, (v, acik, ref) in enumerate(ABLASYON):
        r = {"satir": i, "varyant": v, "aciklama": acik, "referans": ref, "n": 3,
             "kosum_idler": " ".join(x.split("_")[0] for x in KOSUMLAR[v])}
        for sp, src in (("val", val), ("test", test)):
            if src is None:
                continue
            for m in metr:
                o, sd, _ = ozet(src, v, f"{sp}_{m}")
                r[f"{sp}_{m}_ort"] = round(o, 6); r[f"{sp}_{m}_std"] = round(sd, 6)
            r[f"{sp}_mae_avg_ort"] = round(ozet(src, v, f"{sp}_mae_avg")[0], 6)
            if v == ref:
                r[f"{sp}_d_avg_vs_ref_%"] = ""; r[f"{sp}_p_vs_ref"] = ""; r[f"{sp}_ufuk_iyi_vs_ref"] = ""
            else:
                a = [src[v][s][f"{sp}_mse_avg"] for s in SEEDS]; b = [src[ref][s][f"{sp}_mse_avg"] for s in SEEDS]
                r[f"{sp}_d_avg_vs_ref_%"] = round(100 * (mean(a) - mean(b)) / mean(b), 3)
                r[f"{sp}_p_vs_ref"] = round(t_p([x - y for x, y in zip(a, b)])[1], 5)
                iyi = sum(1 for h in UFUK if ozet(src, v, f"{sp}_mse_h{h}")[0] < ozet(src, ref, f"{sp}_mse_h{h}")[0])
                r[f"{sp}_ufuk_iyi_vs_ref"] = f"{iyi}/4"
        out.append(r)
    with open(SON / "ablasyon.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar); w.writeheader(); w.writerows(out)
    print(f"[k5] ablasyon.csv yazildi ({'val+test' if test else 'yalniz val'})")


def karsilastir(src, sp, v, ref):
    a = [src[v][s][f"{sp}_mse_avg"] for s in SEEDS]; b = [src[ref][s][f"{sp}_mse_avg"] for s in SEEDS]
    d = [x - y for x, y in zip(a, b)]
    t, p = t_p(d)
    res = {"v": v, "ref": ref, "ort_v": mean(a), "sd_v": stdev(a), "ort_r": mean(b), "sd_r": stdev(b),
           "d": mean(d), "drel": 100 * mean(d) / mean(b), "t": t, "p": p, "ci": t_ci(d), "boot": boot_ci(d),
           "wil": wilcoxon_p(d), "cift": d, "cift_iyi": sum(1 for x in d if x < 0), "ufuk": {}}
    for h in UFUK:
        ah = [src[v][s][f"{sp}_mse_h{h}"] for s in SEEDS]; bh = [src[ref][s][f"{sp}_mse_h{h}"] for s in SEEDS]
        dh = [x - y for x, y in zip(ah, bh)]
        res["ufuk"][h] = {"ort_v": mean(ah), "ort_r": mean(bh), "drel": 100 * mean(dh) / mean(bh),
                          "p": t_p(dh)[1], "cift_iyi": sum(1 for x in dh if x < 0)}
    res["ufuk_iyi"] = sum(1 for h in UFUK if res["ufuk"][h]["drel"] < 0)
    return res


def asama_test():
    log = SON / "test-acilis.log"
    if log.exists() and "TAMAM" in log.read_text(encoding="utf-8"):
        sys.exit("[k5] RED: test-acilis.log TAMAM -> test kilidi zaten acildi (D1: bir kez).")
    t0 = datetime.now().isoformat(timespec="seconds")
    onceki = log.read_text(encoding="utf-8") if log.exists() else ""
    with open(log, "a", encoding="utf-8") as f:  # acilis girisimi okumadan ONCE kaydedilir (yarim kalirsa beyan edilir)
        f.write(f"GIRISIM baslangic {t0}\n")
    val = asama_val()
    okunan = []  # (yol, md5)
    test = {}
    for v, ids in KOSUMLAR.items():
        test[v] = {}
        for rid in ids:
            d, s = denetle(v, rid)
            p = d / "test_metrics.json"
            okunan.append((p.relative_to(KOK).as_posix(), md5(p)))
            tm = json.loads(p.read_text(encoding="utf-8"))
            assert int(tm["seed"]) == s and int(tm.get("kisaltilmis", 0)) == 0 and int(tm["ufuk_sayisi"]) == 4, rid
            test[v][s] = {k: float(tm[k]) for k in tm if k.startswith("test_")}
    # K3 replikasyon tutarliligi: baseline metrics.json'daki (K3 istisnasi) test_mse_avg ile test_metrics.json ayni olmali
    rep = []
    for rid in KOSUMLAR["baseline"]:
        mj = json.loads((RUNS / rid / "metrics.json").read_text(encoding="utf-8"))
        s = int(rid[-4:])
        rep.append((rid, mj.get("test_mse_avg"), test["baseline"][s]["test_mse_avg"]))
        assert mj.get("test_mse_avg") is not None and abs(mj["test_mse_avg"] - test["baseline"][s]["test_mse_avg"]) < 1e-12, rid
    rep_ort = mean(x[2] for x in rep)
    assert round(rep_ort, 4) == 0.3646, f"replikasyon uyusmazligi {rep_ort}"
    t1 = datetime.now().isoformat(timespec="seconds")

    # ---- test-sonuclari.csv (koşum bazinda) ----
    alan = ["varyant", "seed", "run_id", "test_mse_avg", "test_mae_avg"] + [f"test_mse_h{h}" for h in UFUK] + \
           [f"test_mae_h{h}" for h in UFUK] + ["val_mse_avg"] + [f"val_mse_h{h}" for h in UFUK]
    with open(SON / "test-sonuclari.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alan); w.writeheader()
        for v, ids in KOSUMLAR.items():
            for rid in ids:
                s = int(rid[-4:])
                w.writerow({"varyant": v, "seed": s, "run_id": rid, **{k: test[v][s][k] for k in alan if k.startswith("test_")},
                            **{k: val[v][s][k] for k in alan if k.startswith("val_")}})

    # ---- leaderboard-test.md ----
    L = ["# Leaderboard — TEST (kilit K5'te bir kez açıldı) · metrik `test_mse_avg` (düşük iyi)", "",
         f"*Açılış: {t0} → {t1} · betik `scripts/k5_test_ac.py --asama test` · okunan dosyalar ve md5: `sonuclar/test-acilis.log`*",
         "", "Ön-kayıt: `orkestrasyon_durumu.md` 'K5 ŞEF ÖN-KAYDI'. Birincil karşılaştırmalar: **lastnorm vs baseline**, "
         "**lastnorm-ep336eq vs baseline-ep336eq**; diğer satırlar tablo/analiz. Test'e bakılarak seçim/arama yapılmadı.", "",
         "## 1. Varyant özeti (n = 3 seed {2023, 2024, 2025}, eşleştirilmiş)", "",
         "| # | varyant | test_mse_avg ort ± std | h96 | h192 | h336 | h720 | test_mae_avg | val_mse_avg (seçim) | Δtest vs baseline | p (eşl. t) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    sira = sorted(KOSUMLAR, key=lambda v: mean(test[v][s]["test_mse_avg"] for s in SEEDS))
    for i, v in enumerate(sira, 1):
        o, sd, _ = ozet(test, v, "test_mse_avg")
        hs = " | ".join(fmt(ozet(test, v, f"test_mse_h{h}")[0]) for h in UFUK)
        vo, vsd, _ = ozet(val, v, "val_mse_avg")
        if v == "baseline":
            dd, pp = "—", "—"
        else:
            k = karsilastir(test, "test", v, "baseline"); dd = f"{k['drel']:+.2f}%"; pp = fmt(k["p"], 4)
        L.append(f"| {i} | {v} | {o:.4f} ± {sd:.4f} | {hs} | {fmt(ozet(test, v, 'test_mae_avg')[0])} | {vo:.4f} ± {vsd:.4f} | {dd} | {pp} |")
    L += ["", "## 2. Karşılaştırmalar (test; eşleştirilmiş, n = 3)", "",
          "| rol | varyant vs referans | ort (v / ref) | Δ | Δ% | çift Δ<0 | t | p (t) | %95 t-CI (Δ) | bootstrap %95 (Δ, n=3 dejenere) | Wilcoxon p (n=3 alt sınır 0.25) | ufuk yönü (Δ<0) | val Δ% (aynı çift) | yön korunuyor mu |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    kayit = []
    for v, ref, rol in KARSILASTIRMA:
        kt = karsilastir(test, "test", v, ref); kv = karsilastir(val, "val", v, ref)
        korunuyor = "evet" if (kt["d"] < 0) == (kv["d"] < 0) else "HAYIR"
        L.append(f"| {rol} | {v} vs {ref} | {kt['ort_v']:.4f} / {kt['ort_r']:.4f} | {kt['d']:+.4f} | {kt['drel']:+.2f}% | "
                 f"{kt['cift_iyi']}/3 | {kt['t']:+.2f} | {fmt(kt['p'])} | [{kt['ci'][0]:+.4f}, {kt['ci'][1]:+.4f}] | "
                 f"[{kt['boot'][0]:+.4f}, {kt['boot'][1]:+.4f}] | {kt['wil']:.2f} | {kt['ufuk_iyi']}/4 | {kv['drel']:+.2f}% | {korunuyor} |")
        kayit.append((rol, kt, kv))
    L += ["", "## 3. Ufuk bazında (birincil + protokol + ablasyon)", "",
          "| varyant vs referans | ufuk | test v / ref | test Δ% | çift Δ<0 | p (t) | val Δ% | yön korunuyor mu |",
          "|---|---|---|---|---|---|---|---|"]
    for rol, kt, kv in kayit[:6]:
        for h in UFUK:
            u = kt["ufuk"][h]; uv = kv["ufuk"][h]
            ayni = "—" if u["drel"] == 0 and uv["drel"] == 0 else ("evet" if (u["drel"] < 0) == (uv["drel"] < 0) else "HAYIR")
            L.append(f"| {kt['v']} vs {kt['ref']} | h{h} | {u['ort_v']:.4f} / {u['ort_r']:.4f} | {u['drel']:+.2f}% | {u['cift_iyi']}/3 | "
                     f"{fmt(u['p'])} | {uv['drel']:+.2f}% | {ayni} |")
    # Holm (iki birincil)
    ps = sorted([(kayit[0][1]["p"], "lastnorm vs baseline"), (kayit[1][1]["p"], "lastnorm-ep336eq vs baseline-ep336eq")])
    holm = [(ad, min(1.0, p * (2 - i))) for i, (p, ad) in enumerate(ps)]
    holm[1] = (holm[1][0], max(holm[1][1], holm[0][1]))
    L += ["", "## 4. Notlar", "",
          f"- **Replikasyon sağlaması:** baseline `test_metrics.json` ile K3 istisnalı `metrics.json` test_mse_avg değerleri seed başına "
          f"birebir aynı (|fark| < 1e-12); ortalama **{rep_ort:.4f}** = replikasyon.md 0.3646 ✓ "
          f"({', '.join(f'{x[2]:.4f}' for x in rep)}).",
          "- **Çoklu karşılaştırma (iki birincil, Holm):** " + "; ".join(f"{a} p_Holm = {b:.4f}" for a, b in holm) + ".",
          "- **n = 3 notu:** bootstrap CI 27 yeniden-örnekten (tam numaralandırma) ve dejenere: yalnız çift Δ'ların min/max aralığını verir. "
          "Wilcoxon kesin testi n = 3'te p < 0.25 üretemez; anlamlılık kararı eşleştirilmiş t ile verilir (D2), Wilcoxon yalnız yön tutarlılığı bilgisidir.",
          "- **Görülmemiş-iyi denetimi (H04 kartı):** lastnorm test_mse_avg < 0.317 → İNCELE. "
          f"Ölçülen {ozet(test, 'lastnorm', 'test_mse_avg')[0]:.4f} → {'İNCELE' if ozet(test, 'lastnorm', 'test_mse_avg')[0] < 0.317 else 'tetiklenmedi'}.",
          "- Validation leaderboard ayrı: `sonuclar/leaderboard.md` (leaderboard.py, yalnız val). Bu dosya yalnız test.", ""]
    (SON / "leaderboard-test.md").write_text("\n".join(L), encoding="utf-8")

    # ---- test-acilis.log ----
    G = [onceki + f"GIRISIM baslangic {t0}", "", f"K5 TEST KILIDI ACILISI (tek oturum, D1) — TAMAM", f"baslangic: {t0}", f"bitis:     {t1}",
         f"betik: scripts/k5_test_ac.py md5={md5(Path(__file__))}",
         f"okunan test dosyasi sayisi: {len(okunan)} (10 varyant x 3 seed; yalniz asama-3 rc=0)",
         "yetim/dislanan (OKUNMADI): 20261014-031950_baseline-ep336eq_s2024 (rc=96), 20261014-053238_lastnorm_s2024 (rc=96), "
         "20261014-072750_lastnorm-ep336eq_s2023 (.done yok/rc=96), 20261014-145536_lastnorm-dro_s2025 (.done yok/rc=96)",
         "preds/test_kilitli/ ve test_kilitli/ altindaki ham dosyalar bu oturumda OKUNMADI (yalniz test_metrics.json).",
         "replikasyon sagmasi: baseline metrics.json test_mse_avg == test_metrics.json (3/3), ort %.6f" % rep_ort, "",
         "md5  yol"]
    G += [f"{h}  {p}" for p, h in okunan]
    log.write_text("\n".join(G) + "\n", encoding="utf-8")
    ablasyon_yaz(val, test)
    print(f"[k5] test acildi {t0}; {len(okunan)} dosya; replikasyon ort {rep_ort:.4f}")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", choices=["val", "test"], required=True)
    a = ap.parse_args()
    if a.asama == "val":
        ablasyon_yaz(asama_val())
    else:
        asama_test()
