#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
olcek.py -- H13 (`lastnorm-olcek`): C = R + alpha_{H,j}(P - R), alpha = max(alpha_hat, 0) (Granger-Ramanathan, toplam=1,
sabitsiz, model agirligi >= 0; ust kirpma YOK). Kart: hipotezler/H13.md.
alpha_hat kestiricisi, ic-holdout, dilimler, saglamalar ve cikti semasi H12 ile BIREBIR ayni:
varyantlar/lastnorm-birlesim/birlesim.py import edilir; tek fark agirlik donusumu (H12: clip(alpha_hat, 0, 1)).
metrics.json'da agirliklar `alfa_h{H}` anahtariyla yazilir (kart uygulama notu). CPU, GPU suresi 0.
Kullanim (WS kokunden):
  env/.venv/Scripts/python.exe varyantlar/lastnorm-olcek/olcek.py --kat 1 --seed 2023 --out {RUN_DIR}/metrics.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lastnorm-birlesim"))
import birlesim  # noqa: E402


def olcek_donusum(a):
    return float(max(a, 0.0))


CFG = {"varyant": "lastnorm-olcek", "kart": "H13", "ebeveyn": "lastnorm", "donusum": olcek_donusum,
       "donusum_adi": "max(alpha_hat, 0)", "alfa_anahtari": "alfa"}

if __name__ == "__main__":
    sys.exit(birlesim.main(CFG))
