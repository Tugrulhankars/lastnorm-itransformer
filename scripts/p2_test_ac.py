"""p2_test_ac.py — P2 (ETTh1/ETTh2) TEK SEFERLIK test kilidi acilisi. On-kayitli kume: p2 kuyruklarinin asama-3 rc=0 tam
kosumlari (etthX-baseline, etthX-lastnorm; 3 seed). Seçim yok; yalniz raporlama. Ikinci acilisi reddeder."""
import json, re, sys, hashlib
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
WS = Path(__file__).resolve().parents[1]
LOG = WS / "sonuclar/test-acilis-p2.log"
if LOG.exists(): sys.exit("RED: P2 test zaten acildi")
bild = (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8")
runs = re.findall(r"BİTTİ (\d{8}-\d{6}_(etth[12]-(?:baseline|lastnorm))_s(\d+)) rc=0", bild)
ufuk = ["h96", "h192", "h336", "h720"]
R = {}; satir = []
for rid, v, s in runs:
    f = WS / "runs" / rid / "test_metrics.json"; b = f.read_bytes()
    m = json.loads(b); R.setdefault(v, {})[s] = m
    satir.append(f"{rid} md5={hashlib.md5(b).hexdigest()}")
out = ["# P2 Test Sonuçları (ETTh1/ETTh2) — tek açılış " + datetime.now().isoformat(timespec="seconds"), "",
       "| veri | yöntem | test_mse_avg (ort±std, n=3) | " + " | ".join(ufuk) + " | test_mae_avg |", "|---|---|---|---|---|---|---|---|"]
csv = ["veri,yontem,seed,test_mse_avg,test_mae_avg," + ",".join("test_mse_" + u for u in ufuk)]
for d in ["etth1", "etth2"]:
    for y in ["baseline", "lastnorm"]:
        ms = R[f"{d}-{y}"]; seeds = sorted(ms)
        a = [ms[s]["test_mse_avg"] for s in seeds]
        out.append(f"| {d} | {y} | {mean(a):.4f}±{stdev(a):.4f} | " + " | ".join(f"{mean(ms[s]['test_mse_'+u] for s in seeds):.4f}" for u in ufuk)
                   + f" | {mean(ms[s]['test_mae_avg'] for s in seeds):.4f} |")
        for s in seeds:
            csv.append(f"{d},{y},{s},{ms[s]['test_mse_avg']},{ms[s]['test_mae_avg']}," + ",".join(str(ms[s]['test_mse_'+u]) for u in ufuk))
    b = [R[f"{d}-baseline"][s]["test_mse_avg"] for s in sorted(R[f"{d}-baseline"])]
    l = [R[f"{d}-lastnorm"][s]["test_mse_avg"] for s in sorted(R[f"{d}-lastnorm"])]
    t = stats.ttest_rel(l, b)
    out.append(f"\n{d}: lastnorm vs baseline Δ={100*(mean(l)/mean(b)-1):+.2f}% (eşli t p={t.pvalue:.3g})")
(WS / "sonuclar/test-sonuclari-p2.md").write_text("\n".join(out) + "\n", encoding="utf-8")
(WS / "sonuclar/test-sonuclari-p2.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
LOG.write_text(datetime.now().isoformat() + "\n" + "\n".join(satir) + "\n", encoding="utf-8")
print("\n".join(out))
