"""zincir.py — verilen kuyrukların bitmemiş satırlarını sırayla satir_kos.py ile koşar (BILDIRIM.log'da rc=0 BİTTİ olanları atlar).
--tek: yalnız sıradaki 1 satırı koşar (ön plan, bellek temizliğine karşı). Kullanım: python scripts/zincir.py [--tek] kuyruk1 kuyruk2 ..."""
import json, re, subprocess, sys
from pathlib import Path
WS = Path(__file__).resolve().parents[1]; PY = str(WS / "env/.venv/Scripts/python.exe")
tek = "--tek" in sys.argv; adlar = [a for a in sys.argv[1:] if a != "--tek"]
def bitti():
    return set(re.findall(r"BİTTİ \d{8}-\d{6}_(\S+?)_s(\d+) rc=0", (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8")))
kalan = 0
for ad in adlar:
    for i, s in enumerate((WS / "kuyruk" / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not s.strip(): continue
        j = json.loads(s)
        if (j["varyant"], str(j["seed"])) in bitti(): continue
        kalan += 1
        if tek and kalan > 1: continue
        r = subprocess.run([PY, str(WS / "scripts/satir_kos.py"), ad, str(i)], cwd=str(WS))
        print(f"[zincir] {ad} {i} {j['varyant']} s{j['seed']} rc={r.returncode}", flush=True)
        if r.returncode != 0: sys.exit(r.returncode)
print(f"[zincir] kalan_satir_sonra={max(kalan-1,0) if tek else 0}", flush=True)
