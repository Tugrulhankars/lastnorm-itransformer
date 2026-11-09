"""f3_seed_test_ac.py — E2 (Exchange +7 seed) test_metrics.json TEK okuma; ön-kayıtlı orijinal 3 seed ile birleşik 10-seed raporu.
Betimsel (Exchange test 2026-10-14 15:42'de açılmıştı). İkinci çalıştırmayı reddeder."""
import json, re, sys, hashlib
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
WS = Path(__file__).resolve().parents[1]; OUT = WS / "sonuclar/f3-seed-test.md"
if OUT.exists(): sys.exit("RED: zaten açıldı")
log = (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8")
orj = {"baseline": ["20261014-020531_baseline_s2023", "20261014-021024_baseline_s2024", "20261014-021448_baseline_s2025"],
       "lastnorm": ["20261014-052508_lastnorm_s2023", "20261014-053601_lastnorm_s2024", "20261014-054213_lastnorm_s2025"]}
ek = {y: [r for r, s in sorted(set(re.findall(rf"BİTTİ (\d{{8}}-\d{{6}}_{y}-ekseed_s(\d+)) rc=0", log)), key=lambda t: t[1])] for y in orj}
U = ["avg", "h96", "h192", "h336", "h720"]; md5 = []
def oku(rid, bolum):
    f = WS / "runs" / rid / ("test_metrics.json" if bolum == "test" else "metrics.json"); b = f.read_bytes()
    if bolum == "test": md5.append(f"{rid} {hashlib.md5(b).hexdigest()}")
    m = json.loads(b); return {u: m[f"{bolum}_mse_{u}"] for u in U}
L = [f"# E2 — Exchange 10-seed (3 ön-kayıtlı + 7 ek) — tek okuma {datetime.now().isoformat(timespec='seconds')} (betimsel)", ""]
for bolum in ["val", "test"]:
    for grup, sec in [("ek7", ek), ("tum10", {y: orj[y] + ek[y] for y in orj})]:
        b = [oku(r, bolum) for r in sec["baseline"]]; l = [oku(r, bolum) for r in sec["lastnorm"]]
        L.append(f"## {bolum} · {grup} (n={len(b)})")
        for u in U:
            bb = [x[u] for x in b]; ll = [x[u] for x in l]; t = stats.ttest_rel(ll, bb); w = stats.wilcoxon(ll, bb)
            L.append(f"- {u}: iTr {mean(bb):.4f}±{stdev(bb):.4f} · LN {mean(ll):.4f}±{stdev(ll):.4f} · Δ {100*(mean(ll)/mean(bb)-1):+.2f}% · eşli t p={t.pvalue:.3g} · Wilcoxon p={w.pvalue:.3g} · LN<iTr {sum(a<c for a,c in zip(ll,bb))}/{len(bb)}")
        L.append("")
(WS / "sonuclar/f3-seed-test.log").write_text("\n".join(md5) + "\n", encoding="utf-8")
OUT.write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
