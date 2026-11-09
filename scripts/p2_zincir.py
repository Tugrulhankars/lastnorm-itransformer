"""p2_zincir.py — P2 tam koşumlarını (ETTh1/ETTh2 baseline+lastnorm, 3 seed) sırayla satir_kos.py ile koşar.
BILDIRIM.log'da 'BİTTİ <ts>_<varyant>_s<seed> rc=0' olan satırları atlar → kesintide güvenle yeniden başlatılır."""
import json, re, subprocess, sys
from pathlib import Path
WS = Path(__file__).resolve().parents[1]
PY = str(WS / "env/.venv/Scripts/python.exe")
SIRA = ["p2-etth1-baseline", "p2-etth1-lastnorm", "p2-etth2-baseline", "p2-etth2-lastnorm"]

def bitti():
    log = (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8")
    return set(re.findall(r"BİTTİ \d{8}-\d{6}_(\S+?)_s(\d+) rc=0", log))

for ad in SIRA:
    satirlar = [json.loads(s) for s in (WS / "kuyruk" / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    for i, j in enumerate(satirlar, 1):
        if (j["varyant"], str(j["seed"])) in bitti():
            print(f"[zincir] atla {ad} {i} (bitti)", flush=True); continue
        r = subprocess.run([PY, str(WS / "scripts/satir_kos.py"), ad, str(i)], cwd=str(WS))
        print(f"[zincir] {ad} {i} rc={r.returncode}", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)
print("[zincir] TAMAM", flush=True)
