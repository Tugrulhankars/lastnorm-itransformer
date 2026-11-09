# -*- coding: utf-8 -*-
"""Makale T1 -- replikasyon birlesik tablosu (Exchange, ETTh1, ETTh2): literatur degeri + kaynak + tolerans vs bizim
3-seed resmi iTransformer test MSE'miz. (makale icin sonradan eklendi -- deney-gelistirici, 2026-10-15; CPU, yalniz okur)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_replikasyon.py

Literatur sabitleri (kaynak: literatur) AYNEN su dosyalardan aktarilir (betik yeni tolerans belirlemez):
  - notlar/literatur-cipa.md s3.1: Exchange ort. [0.358, 0.360], tol 0.008 (YESIL [0.350, 0.368]); ufuk bantlari
    96 0.086-0.088/0.005 · 192 0.177-0.181/0.005 · 336 0.331-0.334/0.005 · 720 0.829-0.847/0.024 (bilgi amacli)
  - notlar/literatur-cipa-ett.md s1.1, s5: ETTh1 0.454 tol 0.010 (YESIL [0.444, 0.464]); ETTh2 0.383 tol 0.008
    ([0.375, 0.391]); ufuk (Tablo 10) ETTh1 0.386/0.441/0.487/0.503, ETTh2 0.297/0.380/0.428/0.427;
    ufuk tol: ETTh2 0.005 (720: 0.008), ETTh1 max(0.005, 0.022 x deger) (bilgi amacli)
Bizim degerler: runs/<id>/test_metrics.json (kosum basina, kilitli dosya; acilislar 15:42 / 01:30'da yapildi).
Karar: YESIL <=> bizim_ort in [lit_alt - tol, lit_ust + tol]; SARI <=> 2*tol icinde; aksi KIRMIZI.
DOGRULAMA (assert): Exchange ort. == sonuclar/replikasyon.md 0.3646 (seed 0.3643/0.3641/0.3655); ETT ort. ==
  sonuclar/test-sonuclari-p2.csv baseline satirlari (1e-9).
Cikti: sonuclar/makale/replikasyon-birlesik.csv/.md
  veri,ufuk,lit_alt,lit_ust,tol,yesil_alt,yesil_ust,bizim_ort,bizim_sd,s2023,s2024,s2025,fark_lit_orta,karar,rol,kaynak
"""
import csv
import json
import sys

import numpy as np

import makale_ortak as O

sys.stdout.reconfigure(encoding="utf-8")

KAYNAK_EX = ("Liu et al. 2024 iTransformer (ICLR, arXiv:2310.06625) Tab.10 ort. 0.360 + Tab.5 5-seed ort. 0.358 (hesap); "
             "Qiu et al. 2025 DUET (KDD, arXiv:2412.10859) Tab.3 TFB-reprod. 0.360; notlar/literatur-cipa.md s3.1")
KAYNAK_ETT = "Liu et al. 2024 iTransformer (ICLR, arXiv:2310.06625) Tab.10 (L=96); notlar/literatur-cipa-ett.md s1.1, s5"
LIT = {  # veri -> ufuk -> (lit_alt, lit_ust, tol, rol)
    "exchange": {"avg": (0.358, 0.360, 0.008, "kapi"), 96: (0.086, 0.088, 0.005, "bilgi"), 192: (0.177, 0.181, 0.005, "bilgi"),
                 336: (0.331, 0.334, 0.005, "bilgi"), 720: (0.829, 0.847, 0.024, "bilgi")},
    "etth1": {"avg": (0.454, 0.454, 0.010, "kapi"), 96: (0.386, 0.386, None, "bilgi"), 192: (0.441, 0.441, None, "bilgi"),
              336: (0.487, 0.487, None, "bilgi"), 720: (0.503, 0.503, None, "bilgi")},
    "etth2": {"avg": (0.383, 0.383, 0.008, "kapi"), 96: (0.297, 0.297, 0.005, "bilgi"), 192: (0.380, 0.380, 0.005, "bilgi"),
              336: (0.428, 0.428, 0.005, "bilgi"), 720: (0.427, 0.427, 0.008, "bilgi")},
}


def tol_ett1(x):
    return max(0.005, round(0.022 * x, 3))


def main():
    O.kosumlari_dogrula()
    with open(O.WS / "sonuclar/test-sonuclari-p2.csv", encoding="utf-8") as f:
        p2 = [r for r in csv.DictReader(f) if r["yontem"] == "baseline"]
    satirlar = []
    for veri, lit in LIT.items():
        tm = [json.loads((O.WS / "runs" / rid / "test_metrics.json").read_text(encoding="utf-8"))
              for rid in O.RUNS[veri]["itr"]]
        for u, (lo, hi, tol, rol) in lit.items():
            key = "test_mse_avg" if u == "avg" else f"test_mse_h{u}"
            v = np.array([float(t[key]) for t in tm])
            if tol is None:
                tol = tol_ett1(lo)
            yl, yu = lo - tol, hi + tol
            m = v.mean()
            karar = "YESIL" if yl <= m <= yu else ("SARI" if lo - 2 * tol <= m <= hi + 2 * tol else "KIRMIZI")
            if veri != "exchange":
                ref = np.array([float(r[key]) for r in sorted(p2, key=lambda r: r["seed"]) if r["veri"] == veri])
                assert np.allclose(ref, v, atol=1e-9), (veri, key)
            satirlar.append(dict(veri=veri, ufuk=u, lit_alt=lo, lit_ust=hi, tol=tol, yesil_alt=round(yl, 4),
                                 yesil_ust=round(yu, 4), bizim_ort=round(m, 4), bizim_sd=round(v.std(ddof=1), 4),
                                 **{f"s{sd}": round(float(v[i]), 4) for i, sd in enumerate(O.SEEDS)},
                                 fark_lit_orta=round(m - (lo + hi) / 2, 4), karar=karar, rol=rol,
                                 kaynak=KAYNAK_EX if veri == "exchange" else KAYNAK_ETT))
    ex = next(s for s in satirlar if s["veri"] == "exchange" and s["ufuk"] == "avg")
    assert ex["bizim_ort"] == 0.3646 and (ex["s2023"], ex["s2024"], ex["s2025"]) == (0.3643, 0.3641, 0.3655), ex
    assert all(s["karar"] == "YESIL" for s in satirlar if s["rol"] == "kapi")
    meta = O.meta_satirlari(
        "makale_replikasyon.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_replikasyon.py",
        ["lit_* ve tol: literatur-cipa*.md'den aynen (kaynak: literatur); rol=kapi satiri resmi replikasyon olcutudur, "
         "rol=bilgi ufuk bantlari yalniz bilgi amaclidir",
         "Exchange kapisi 2026-10-14 02:19 resmi replikasyon_kapisi.py ile YESIL (sonuclar/replikasyon.md); ETT icin resmi "
         "kapi dosyasi yok -> bu tablo ayni kuralla elle-degil betikle birlestirir (manifest B-11)",
         "ETTh2 d_model: makale metni {256,512}, resmi betik 128 (betik esas)",
         "dogrulama: Exchange ort.==replikasyon.md 0.3646; ETT==test-sonuclari-p2.csv; tum kapi satirlari YESIL"],
        [rid for v in LIT for rid in O.RUNS[v]["itr"]])
    O.csv_yaz(O.OUT / "replikasyon-birlesik.csv", satirlar, meta)
    sut = ["veri", "ufuk", "lit_alt", "lit_ust", "tol", "yesil_alt", "yesil_ust", "bizim_ort", "bizim_sd", "s2023",
           "s2024", "s2025", "karar", "rol"]
    L = ["# T1 -- Replikasyon birlesik tablosu (resmi iTransformer, test MSE, z uzayi)", ""] + \
        [f"> {m.lstrip('# ')}" for m in meta] + ["", "| " + " | ".join(sut) + " |", "|" + "---|" * len(sut)]
    for s in satirlar:
        L.append("| " + " | ".join(str(s[c]) for c in sut) + " |")
    L += ["", f"Kaynak (Exchange): {KAYNAK_EX}", f"Kaynak (ETT): {KAYNAK_ETT}"]
    (O.OUT / "replikasyon-birlesik.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for s in satirlar:
        print(s["veri"], s["ufuk"], s["bizim_ort"], "+-", s["bizim_sd"], f"[{s['yesil_alt']}, {s['yesil_ust']}]", s["karar"], s["rol"])
    print("DOGRULAMA GECTI")


if __name__ == "__main__":
    main()
