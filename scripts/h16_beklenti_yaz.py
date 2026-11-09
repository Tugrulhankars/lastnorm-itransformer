#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""h16_beklenti_yaz.py -- FAZ III/H16 on-kayitli beklenti dosyalari (kuyruk/h16-<veri>.beklenti.json).

Calisma: tam kosumlardan ve ETTm1 dumanindan ONCE (2026-10-15). Kaynaklar:
  * lit oran_h = iTransformer test MSE_h (Liu 2024 Tab.10 s.23) / Repeat test MSE_h (Zeng 2023 Tab.2 s.5)
    (notlar/literatur-cipa-faz3.md §1.1, §2);
  * VAL naif: data/naive_val_<veri>.csv (scripts/h16_naif_val.py; resmi Dataset val, hedefler tolerans 0 hizali);
  * param: resmi iTransformer.Model(.sh argumanlari) CPU sayimi (LastNorm esit; yeni parametre yok).
Aralik (her varyant-ufuk satiri, val_mse_avg = o ufkun val MSE'si):
  alt_h = 0.5 x lit oran_h x Repeat_val_h   (P2 ETTh1 emsali: literaturdeki model/Repeat oraninin yarisinin altina
                                             inmek sizinti/protokol suphesi -> INCELE)
  ust_h = min(Repeat_val_h, train-ort_val_h) (P2 ETTh1 emsali: tam egitimli modelin en iyi trivial tahminciden kotu
                                             olmasi kurulum hatasi)
Duman (asama 1, yalniz ETTm1 h96, 1 epoch): alt ayni, ust = 1.5 x en iyi trivial (1 epoch kisaltmasi icin gevsetme;
P2 ETTh1 duman emsali).
Test: hicbir kurala konmadi (kilitli). Replikasyon capasi (lit +-tol, test_mse_avg; literatur-cipa-faz3.md §3) yalniz
kilit acilisinda 4 ufuk birlestirilerek sinanir (aciklamada).
"""
import csv
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
LIT_ITR = {"ettm1": [0.334, 0.377, 0.426, 0.491], "ettm2": [0.180, 0.250, 0.311, 0.412],
           "weather": [0.174, 0.221, 0.278, 0.358]}
LIT_REP = {"ettm1": [1.214, 1.261, 1.283, 1.319], "ettm2": [0.266, 0.340, 0.412, 0.521],
           "weather": [0.259, 0.309, 0.377, 0.465]}
CAPA = {"ettm1": (0.407, 0.009, "[0.398, 0.416]", "[0.389, 0.425]"),
        "ettm2": (0.288, 0.008, "[0.280, 0.296]", "[0.272, 0.304]"),
        "weather": (0.258, 0.008, "[0.250, 0.266]", "[0.242, 0.274]")}
PARAM = {"ettm1": (224224, 304720), "ettm2": (224224, 304720), "weather": (4833888, 5154000)}
UF = [96, 192, 336, 720]
KOLLAR = ("baseline", "lastnorm")


def naif(veri):
    with open(WS / "data" / f"naive_val_{veri}.csv", encoding="utf-8") as f:
        return {int(r["ufuk"]): (float(r["repeat_mse"]), float(r["trainort_mse"])) for r in csv.DictReader(f)
                if r["ufuk"] != "ort"}


def yaz(veri):
    nv = naif(veri)
    lit, cap = CAPA[veri][0], CAPA[veri]
    kurallar3 = [{"tur": "sonlu"}, {"tur": "farkli_seedler_farkli"}, {"tur": "kollar_farkli"},
                 {"tur": "min_kosum", "n": 24}]
    for i, H in enumerate(UF):
        oran = LIT_ITR[veri][i] / LIT_REP[veri][i]
        rep, tro = nv[H]
        alt, ust = round(0.5 * oran * rep, 4), round(min(rep, tro), 4)
        for kol in KOLLAR:
            kurallar3.append({
                "tur": "aralik", "varyant": f"{veri}-{kol}-h{H}", "alt": alt, "ust": ust,
                "kaynak": (f"alt = 0.5 x lit oran {oran:.4f} (iTr {LIT_ITR[veri][i]} Liu 2024 Tab.10 s.23 / Repeat "
                           f"{LIT_REP[veri][i]} Zeng 2023 Tab.2 s.5) x Repeat val h{H} {rep:.4f}; ust = en iyi trivial "
                           f"val h{H} min(Repeat {rep:.4f}, train-ort {tro:.4f}) (data/naive_val_{veri}.csv; P2 ETTh1 emsali)"
                           + ("" if kol == "baseline" else "; lastnorm: ayni makullük araligi (yon kurali yok, secim yok)"))})
    kurallar3.append({"tur": "ikincil_aralik", "metrik": "param_sayisi", "alt": PARAM[veri][0], "ust": PARAM[veri][1],
                      "kaynak": "resmi Model(.sh h96..h720) CPU sayimi; ikincil_aralik varyanta gore suzulmez -> ufuk min..max; "
                                "LastNorm esit"})
    B = {
        "_aciklama": (
            f"ON-KAYIT (deney-yurutucu, FAZ III H16, 2026-10-15): {veri} tam kosumlarindan ONCE yazildi"
            + (" (asama 1 ETTm1 duman kurallari dumandan ONCE)" if veri == "ettm1" else "")
            + ". Satir = (kol, ufuk, seed); varyant adi '<veri>-<kol>-h<H>', metrik val_mse_avg = o ufkun val MSE'si "
              "(--ufuk-basi; notlar/h16-kuyruk.md sema). Kollar: baseline = resmi iTransformer .sh birebir "
              "(scripts/rakip_kos.py), lastnorm = H04 AYNEN (varyantlar/lastnorm/kos.py; hiperparametre aktarma/ayar yok). "
              "Yon kurali YOK: H16 tahmini TEST uzerinde ve test kilitli; val'de secim yapilmaz. "
              f"TEST hicbir kurala konmadi. Replikasyon capasi (yalniz kilit acilisinda, 4 ufuk birlestirilmis seed basina "
              f"test_mse_avg ortalamasi): lit {lit} (Liu 2024 Tab.10 Avg), tol {cap[1]}, YESIL {cap[2]}, SARI {cap[3]} "
              "(notlar/literatur-cipa-faz3.md §3). Son asamadan sonra kapi otomatik kosmaz -> sef beklenti_kontrol.py "
              "--asama 3 ile elle."),
        "metrik": "val_mse_avg",
        "yon": "min",
        "asama": {"3": {"aciklama": f"TAM -- {{baseline, lastnorm}} x h{{96,192,336,720}} x seed {{2023,2024,2025}} "
                                    f"= 24 satir; resmi {veri} .sh birebir (10 epoch, patience 3), ufuk basina ayri surec.",
                        "kurallar": kurallar3}},
    }
    if veri == "ettm1":
        rep, tro = nv[96]
        oran = LIT_ITR[veri][0] / LIT_REP[veri][0]
        k1 = [{"tur": "sonlu"}, {"tur": "min_kosum", "n": 2}, {"tur": "kollar_farkli"}]
        for kol in KOLLAR:
            k1.append({"tur": "aralik", "varyant": f"ettm1-{kol}-h96-duman", "alt": round(0.5 * oran * rep, 4),
                       "ust": round(1.5 * min(rep, tro), 4),
                       "kaynak": f"alt = 0.5 x lit oran h96 {oran:.4f} x Repeat val h96 {rep:.4f}; ust = 1.5 x en iyi "
                                 f"trivial val h96 {min(rep, tro):.4f} (1 epoch kisaltmasi; P2 ETTh1 duman emsali)"})
        k1.append({"tur": "ikincil_aralik", "metrik": "param_sayisi", "alt": 224224, "ust": 224224,
                   "kaynak": "resmi Model(.sh h96) CPU sayimi; LastNorm esit"})
        B["asama"]["1"] = {"aciklama": "DUMAN -- ettm1-{baseline,lastnorm}-h96-duman: s2023, --ufuk-basi --ufuklar 96 "
                                       "--epoch-ust 1 (foreground, deney-yurutucu; sure olcumu -> tam satir tahmini).",
                           "kurallar": k1}
        B["asama"] = {k: B["asama"][k] for k in sorted(B["asama"])}
    p = WS / "kuyruk" / f"h16-{veri}.beklenti.json"
    if p.exists():
        raise SystemExit(f"{p} zaten var (on-kayit uzerine yazilmaz)")
    p.write_text(json.dumps(B, ensure_ascii=False, indent=2), encoding="utf-8")
    print("yazildi", p)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["ettm1", "ettm2", "weather"]:
        yaz(v)
