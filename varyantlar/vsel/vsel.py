"""H15 VSEL — val-seçimli değişken-düzeyi aday seçimi {iTransformer-RevIN, iTransformer-LastNorm, Repeat}. CPU son-işlem.
Kural hipotezler/H15.md'de sabit. --dogrula: yalniz val (Repeat val'in naif kayitlariyla, model val'in metrics.json ile eşleşmesi).
--test: TEK hesap; sonuclar/vsel-test.md yazar, ikinci kez reddeder."""
import argparse, csv, json, re, sys
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
WS = Path(__file__).resolve().parents[2]
UF = [96, 192, 336, 720]
VERI = {"exchange": ("data/exchange_rate/exchange_rate.csv", 5311,
                     {"baseline": ["20261014-020531_baseline_s2023", "20261014-021024_baseline_s2024", "20261014-021448_baseline_s2025"],
                      "lastnorm": ["20261014-052508_lastnorm_s2023", "20261014-053601_lastnorm_s2024", "20261014-054213_lastnorm_s2025"]}),
        "etth1": ("data/etth1/ETTh1.csv", 8640, None), "etth2": ("data/etth2/ETTh2.csv", 8640, None)}
SEEDS = ["2023", "2024", "2025"]

def kosumlar(d):
    if VERI[d][2]: return VERI[d][2]
    log = (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8")
    out = {}
    for y in ["baseline", "lastnorm"]:
        ids = re.findall(rf"BİTTİ (\d{{8}}-\d{{6}}_{d}-{y}_s(\d+)) rc=0", log)
        out[y] = [next(i for i, s in ids if s == sd) for sd in SEEDS]
    return out

def Z(d):
    yol, ntr, _ = VERI[d]
    with open(WS / yol, encoding="utf-8") as f:
        r = list(csv.reader(f))
    X = np.array([[float(v) for v in row[1:]] for row in r[1:]])
    tr = X[:ntr]
    return (X - tr.mean(0)) / tr.std(0, ddof=0)

def repeat_mse(z, ilk, H):  # (pencere, değişken)
    idx = ilk[:, None] + np.arange(H)[None, :]
    return ((z[idx] - z[ilk - 1][:, None, :]) ** 2).mean(1)

def yukle(rid, bolum, H):
    p = WS / "runs" / rid / "preds" / (f"test_kilitli/test_h{H}.npz" if bolum == "test" else f"val_h{H}.npz")
    f = np.load(p); return f["mse_pencere_degisken"].astype(np.float64), f["hedef_ilk_satir"]

def hesap(d, bolum_list):
    z = Z(d); K = kosumlar(d); sonuc = {}
    for si, sd in enumerate(SEEDS):
        for H in UF:
            v = {}; t = {}
            for y in ["baseline", "lastnorm"]:
                for b in bolum_list:
                    m, ilk = yukle(K[y][si], b, H)
                    (v if b == "val" else t)[y] = (m, ilk)
            for b, dct in (("val", v), ("test", t)):
                if dct:
                    ilk = dct["baseline"][1]; assert (ilk == dct["lastnorm"][1]).all()
                    dct["repeat"] = (repeat_mse(z, ilk, H), ilk)
            adaylar = ["baseline", "lastnorm", "repeat"]
            vm = np.stack([v[a][0].mean(0) for a in adaylar])  # (3, C)
            sec = vm.argmin(0)  # bağda ilk = baseline
            sonuc[(sd, H)] = {"sec": [adaylar[i] for i in sec],
                              "val": {a: float(v[a][0].mean()) for a in adaylar},
                              "val_vsel": float(np.mean([vm[sec[c], c] for c in range(vm.shape[1])]))}
            if t:
                tm = np.stack([t[a][0].mean(0) for a in adaylar])
                sonuc[(sd, H)]["test"] = {a: float(t[a][0].mean()) for a in adaylar}
                sonuc[(sd, H)]["test"]["vsel"] = float(np.mean([tm[sec[c], c] for c in range(tm.shape[1])]))
    return sonuc

ap = argparse.ArgumentParser(); ap.add_argument("--asama", choices=["dogrula", "test"], required=True); a = ap.parse_args()
if a.asama == "dogrula":
    for d in VERI:
        s = hesap(d, ["val"])
        for H in UF:
            print(d, H, "repeat_val", round(s[("2023", H)]["val"]["repeat"], 4), "base", round(mean(s[(x, H)]["val"]["baseline"] for x in SEEDS), 4),
                  "vsel", round(mean(s[(x, H)]["val_vsel"] for x in SEEDS), 4), "seçim(s2023)", s[("2023", H)]["sec"])
else:
    OUT = WS / "sonuclar/vsel-test.md"
    if OUT.exists(): sys.exit("RED: H15 test zaten hesaplandı")
    L = ["# H15 VSEL — test (tek hesap; test-sonrası tasarım, keşifsel)", "",
         "| veri | iTransformer | LastNorm | Repeat | VSEL | Δ VSEL vs iTr | p (eşli t) | VSEL/Repeat |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for d in VERI:
        s = hesap(d, ["val", "test"])
        per = {k: [mean(s[(x, H)]["test"][k] for H in UF) for x in SEEDS] for k in ["baseline", "lastnorm", "repeat", "vsel"]}
        p = stats.ttest_rel(per["vsel"], per["baseline"]).pvalue
        L.append(f"| {d} | {mean(per['baseline']):.4f}±{stdev(per['baseline']):.4f} | {mean(per['lastnorm']):.4f}±{stdev(per['lastnorm']):.4f} | "
                 f"{mean(per['repeat']):.4f} | **{mean(per['vsel']):.4f}±{stdev(per['vsel']):.4f}** | {100*(mean(per['vsel'])/mean(per['baseline'])-1):+.2f}% | {p:.3g} | {mean(per['vsel'])/mean(per['repeat']):.3f} |")
        for x in SEEDS:
            for H in UF:
                rows.append({"veri": d, "seed": x, "H": H, **{f"test_{k}": s[(x, H)]["test"][k] for k in s[(x, H)]["test"]},
                             **{f"val_{k}": s[(x, H)]["val"][k] for k in s[(x, H)]["val"]}, "val_vsel": s[(x, H)]["val_vsel"],
                             "secim": "/".join(s[(x, H)]["sec"])})
    with open(WS / "sonuclar/vsel-test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
