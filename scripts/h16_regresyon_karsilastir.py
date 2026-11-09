#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""h16_regresyon_karsilastir.py <ref_run_dir> <yeni_dir>: rakip_kos degisikligi sonrasi exchange/etth yolu bit-bit ayni mi.
preds/val_h*.npz ve preds/test_kilitli/test_h*.npz tum anahtarlar np.array_equal; metrics.json anahtar kumesi ayni ve
sure/vram/cihaz disindaki tum degerler ayni; test_metrics.json AYNI mi yalniz evet/hayir (sayi basilmaz, kilit)."""
import json, sys
from pathlib import Path
import numpy as np
ref, yeni = Path(sys.argv[1]), Path(sys.argv[2])
DEGISKEN = ("sure", "vram", "egitim_sure_sn", "cihaz", "ram_tepe", "hizlandirma")  # ram: olcum; hizlandirma: A/B bayragi
ok = True
for f in sorted(ref.glob("preds/**/*.npz")):
    g = yeni / f.relative_to(ref)
    a, b = np.load(f), np.load(g)
    esit = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    print(f"[regresyon] {f.relative_to(ref)} anahtarlar={sorted(a.files)} bit-bit={'EVET' if esit else 'HAYIR'}")
    ok &= esit
ma = json.loads((ref / "metrics.json").read_text(encoding="utf-8")); mb = json.loads((yeni / "metrics.json").read_text(encoding="utf-8"))
anahtar = set(ma) == set(mb)
farkli = [k for k in ma if k in mb and not any(d in k for d in DEGISKEN) and ma[k] != mb[k]]
print(f"[regresyon] metrics.json anahtar kumesi ayni={'EVET' if anahtar else 'HAYIR'} ({len(ma)} anahtar); "
      f"sure/vram/cihaz disi farkli deger: {farkli or 'yok'}")
ta = json.loads((ref / "test_metrics.json").read_text(encoding="utf-8")); tb = json.loads((yeni / "test_metrics.json").read_text(encoding="utf-8"))
print(f"[regresyon] test_metrics.json ayni (sayi basilmadi)={'EVET' if ta == tb else 'HAYIR'}")
ok &= anahtar and not farkli and ta == tb
print("[regresyon] SONUC", "GECTI (bit-bit)" if ok else "KALDI")
sys.exit(0 if ok else 1)
