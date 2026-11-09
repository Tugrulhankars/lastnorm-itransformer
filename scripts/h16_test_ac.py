"""h16_test_ac.py — H16 TEK SEFERLİK test açılışı (ETTm1, ETTm2; 4 ufuk × 3 seed × {baseline, lastnorm}).
Kural hipotezler/H16.md: R̄<1 ⇒ LN<iTr; R̄>1 ⇒ LN≥iTr. Replikasyon kapısı lit ±tol. İkinci açılışı reddeder."""
import json, re, sys, hashlib
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
WS = Path(__file__).resolve().parents[1]; LOG = WS / "sonuclar/test-acilis-h16.log"
if LOG.exists(): sys.exit("RED: H16 test zaten açıldı")
bild = (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8")
R = {"exchange": 0.7790, "etth1": 1.5399, "etth2": 1.1770, "ettm1": 1.5360, "ettm2": 1.0967}
LIT = {"ettm1": (0.407, 0.009), "ettm2": (0.288, 0.008)}
DPCT_KAL = {"exchange": -4.96, "etth1": 3.82, "etth2": 1.88}  # kalibrasyon (sonuclar/karar.md, test-sonuclari-p2.md)
U = [96, 192, 336, 720]; SEEDS = ["2023", "2024", "2025"]; md5 = []
def run(d, kol, H, s):
    ids = re.findall(rf"BİTTİ (\d{{8}}-\d{{6}}_{d}-{kol}-h{H}_s{s}) rc=0", bild); assert len(ids) == 1, (d, kol, H, s, ids); return ids[0]
def oku(rid, H, bolum):
    f = WS / "runs" / rid / ("test_metrics.json" if bolum == "test" else "metrics.json"); b = f.read_bytes()
    if bolum == "test": md5.append(f"{rid} {hashlib.md5(b).hexdigest()}")
    m = json.loads(b); k = f"{bolum}_mse_h{H}"
    return m[k] if k in m else m[f"{bolum}_mse_avg"]
L = [f"# H16 — ileriye dönük rejim sınaması: TEST tek açılış {datetime.now().isoformat(timespec='seconds')}", "",
     "| veri | R̄ | tahmin | iTr test (ort±sd) | LN test | Δ% | eşli t p | LN<iTr seed | ufuk LN<iTr | val Δ% | lit iTr ±tol | replikasyon | tahmin doğru? |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
dpct = dict(DPCT_KAL); sonuc = {}
for d in ["ettm1", "ettm2"]:
    T = {k: {s: {H: oku(run(d, k, H, s), H, "test") for H in U} for s in SEEDS} for k in ["baseline", "lastnorm"]}
    V = {k: {s: {H: oku(run(d, k, H, s), H, "val") for H in U} for s in SEEDS} for k in ["baseline", "lastnorm"]}
    b = [mean(T["baseline"][s].values()) for s in SEEDS]; l = [mean(T["lastnorm"][s].values()) for s in SEEDS]
    vb = mean(mean(V["baseline"][s].values()) for s in SEEDS); vl = mean(mean(V["lastnorm"][s].values()) for s in SEEDS)
    dp = 100 * (mean(l) / mean(b) - 1); dpct[d] = dp; p = stats.ttest_rel(l, b).pvalue
    uf = sum(mean(T["lastnorm"][s][H] for s in SEEDS) < mean(T["baseline"][s][H] for s in SEEDS) for H in U)
    tah = "kötüleşir" if R[d] > 1.05 else ("iyileşir" if R[d] < 0.95 else "belirsiz")
    dogru = (dp >= 0) if tah == "kötüleşir" else ((dp < 0) if tah == "iyileşir" else None)
    lv, tol = LIT[d]; rep = "YEŞİL" if abs(mean(b) - lv) <= tol else "KIRMIZI"
    sonuc[d] = dogru
    L.append(f"| {d} | {R[d]:.3f} | {tah} | {mean(b):.4f}±{stdev(b):.4f} | {mean(l):.4f}±{stdev(l):.4f} | {dp:+.2f} | {p:.3g} | {sum(x<y for x,y in zip(l,b))}/3 | {uf}/4 | {100*(vl/vb-1):+.2f} | {lv}±{tol} | {rep} | {'EVET' if dogru else ('—' if dogru is None else 'HAYIR')} |")
    for H in U:
        L.append(f"|  | h{H} |  | {mean(T['baseline'][s][H] for s in SEEDS):.4f} | {mean(T['lastnorm'][s][H] for s in SEEDS):.4f} | {100*(mean(T['lastnorm'][s][H] for s in SEEDS)/mean(T['baseline'][s][H] for s in SEEDS)-1):+.2f} |  |  |  |  |  |  |  |")
ks = list(R); rho = stats.spearmanr([R[k] for k in ks], [dpct[k] for k in ks])
L += ["", f"**Önceden tanımlı başarı (belirsiz-olmayan tüm yeni setlerde işaret doğru):** {'TUTTU' if all(v for v in sonuc.values() if v is not None) else 'TUTMADI'} ({sum(1 for v in sonuc.values() if v)}/{len(sonuc)})",
      f"**İkincil:** Spearman(R̄, Δ%) 5 set (3 kalibrasyon + 2 sınama) ρ={rho.correlation:.3f} (p={rho.pvalue:.3g}); değerler: " + ", ".join(f"{k} R̄={R[k]:.3f} Δ={dpct[k]:+.2f}%" for k in ks),
      "**Beyan:** sınama tek yönlü (yeni setlerin hepsi R̄>1); kalibrasyon setleri sınama değildir; Solar/Weather sonuç görülmeden kesildi."]
LOG.write_text(datetime.now().isoformat() + "\n" + "\n".join(md5) + "\n", encoding="utf-8")
(WS / "sonuclar/h16-test.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
