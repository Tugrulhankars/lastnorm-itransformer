# -*- coding: utf-8 -*-
"""Makale F3/T6 -- (1) Delta-vs-tur birlesik tablosu (H01-H15) ve (2) Basarisiz denemeler tablosu.
(makale icin sonradan eklendi -- deney-gelistirici, 2026-10-15; CPU, yalniz mevcut sonuc dosyalarini okur)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_kart_tablolari.py

Sayisal girdiler (hepsi diskten; hicbiri elle yazilmaz):
  - tur 1-3 (resmi val_mse_avg, 3 seed): sonuclar/test-sonuclari.csv val_mse_* sutunlari (run_id'li) -> Delta% ebeveyne,
    iyilesen ufuk sayisi (3-seed ort. Delta_H<0), eslestirilmis t p; test_mse_avg (yalniz rapor) ayni dosyadan.
  - tur 4 (FAZ II 3-kat A-bar = 3 kat x 3 seed ort. val_mse_avg): sonuclar/tahlil-tur4-kat.csv
  - tur 5 (H14): runs/*_lastnorm-esit-k{1,2,3}_s*/metrics.json (val anahtarlari)
  - kapilar: sonuclar/h05-onkosul.csv (H05), sonuclar/h11-kapi.csv (H11)
  - H15: sonuclar/vsel-test.csv (val secim-ici ve test)
Kararlar ve "neden oldu" metinleri tahlil dosyalarindan aktarilir (karar_kaynagi sutunu); betik karar VERMEZ.
DOGRULAMA (assert): hesaplanan Delta'lar tahlil dosyalarindaki raporlanmis degerlerle (+-0.01 puan) ve ufuk/kat sayilari
  birebir; test Delta'lari leaderboard-test.md ile (+-0.01 puan); VSEL test Delta'lari vsel-test.md ile.

Ciktilar: sonuclar/makale/delta-vs-tur.csv/.md, sonuclar/makale/basarisiz-denemeler.csv/.md
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import makale_ortak as O

sys.stdout.reconfigure(encoding="utf-8")
WS = O.WS
UF = O.UFUKLAR

# kart -> (varyant, tur, ebeveyn, metrik tipi)
KART = {
    "H01": ("san", "1", "baseline", "resmi"), "H02": ("revin-afin", "1", "baseline", "resmi"),
    "H03": ("baseline-ep336eq", "1", "baseline", "resmi"),
    "H04": ("lastnorm", "2", "baseline", "resmi"), "H05": ("san-mu", "2", "baseline", "kapi"),
    "H06": ("normsuz", "2", "baseline", "resmi"), "H07": ("xmark-yok", "2", "baseline", "resmi"),
    "H08": ("lastnorm-ep336eq", "3", "lastnorm", "resmi"), "H09": ("lastnorm-dro", "3", "lastnorm", "resmi"),
    "H10": ("lastnorm-merkez", "3", "lastnorm", "resmi"),
    "H04-kat": ("lastnorm", "4 (P1 kat protokolu)", "baseline", "kat3"),
    "H11": ("san-softplus", "4", "lastnorm", "kapi"), "H12": ("lastnorm-birlesim", "4", "lastnorm", "kat3"),
    "H13": ("lastnorm-olcek", "4", "lastnorm", "kat3"), "H14": ("lastnorm-esit", "5", "lastnorm", "kat3"),
    "H15": ("vsel", "test-sonrasi (kesifsel)", "baseline", "vsel"),
}
# Kararlar: tahlil dosyalarindan AKTARIM (betik karar vermez)
KARAR = {
    "H01": ("OLDUR (ariza notlu)", "sonuclar/tahlil-tur1.md s0"),
    "H02": ("OLDUR", "sonuclar/tahlil-tur1.md s0"),
    "H03": ("DOGRULANDI (protokol karti; zincir disi)", "sonuclar/tahlil-tur1.md s0"),
    "H04": ("TUT (ayirt edilemez: abs(D) < MDE %8.1)", "sonuclar/tahlil-tur2.md s0"),
    "H05": ("KAPI BASARISIZ (kosulmadi)", "sonuclar/tahlil-tur2.md s0"),
    "H06": ("OLDUR (beklenen; ablasyon satiri)", "sonuclar/tahlil-tur2.md s0"),
    "H07": ("TUT-ayirt edilemez (H0 desteklendi; zincire alinmadi)", "sonuclar/tahlil-tur2.md s0"),
    "H08": ("DOGRULANDI (S1; protokol, zincir disi)", "sonuclar/tahlil-tur3.md s0"),
    "H09": ("OLDUR (ufuk kurali 1/4)", "sonuclar/tahlil-tur3.md s0"),
    "H10": ("OLDUR (ablasyon -sigma)", "sonuclar/tahlil-tur3.md s0"),
    "H04-kat": ("DOGRULANDI (3/3 kat, 12/12 hucre; test-sonrasi tasarlanmis protokol)", "sonuclar/tahlil-tur4.md s3.1"),
    "H11": ("KAPI BASARISIZ (kosulmadi)", "sonuclar/tahlil-tur4.md s0, s2"),
    "H12": ("OLDUR", "sonuclar/tahlil-tur4.md s0, s2"),
    "H13": ("OLDUR", "sonuclar/tahlil-tur4.md s0, s2"),
    "H14": ("OLDUR (beklenti kapisi asama 3 ihlal)", "orkestrasyon_durumu.md 'Oturum devami' (tahlil dosyasi yok; hipotezler/H14.md Sonuc bos)"),
    "H15": ("KESIFSEL -- on-tanimli basari (3/3 veri) TUTMADI", "sonuclar/vsel-test.md; hipotezler/H15.md"),
}
# Tahlil dosyalarinda raporlanan degerler (dogrulama icin): kart -> (delta_pct, ufuk_iyi, kat_iyi)
RAPOR = {"H01": (31.46, 0, None), "H02": (-0.17, 2, None), "H03": (-1.62, None, None), "H04": (-3.83, 4, None),
         "H06": (392.1, 0, None), "H07": (-0.29, 3, None), "H08": (-2.04, None, None), "H09": (-5.02, 1, None),
         "H10": (1.01, 1, None), "H04-kat": (-6.71, 4, 3), "H12": (1.50, 2, 1), "H13": (1.88, 2, None), "H14": (0.5, None, None)}
TEST_RAPOR = {"san": -17.52, "lastnorm": -4.96, "lastnorm-ep336eq": -4.75, "lastnorm-dro": -3.21, "xmark-yok": -1.10,
              "baseline-ep336eq": -0.00, "revin-afin": 0.17, "lastnorm-merkez": 4.71, "normsuz": 164.49}
VSEL_RAPOR = {"exchange": -3.75, "etth1": 0.40, "etth2": -0.14}

MEKANIZMA = {
    "H01": "SAN (Liu et al. 2023): use_norm=0 + dilim-duzeyi istasyon mu/sigma tahmini (period_len 6, 5 ep. on-egitim)",
    "H02": "RevIN-afin: resmi mu/sigma + ogrenilebilir degisken-basina gamma/beta (+16 param)",
    "H05": "SAN-mu: yalniz SAN mu-dali (on-kosul kapisi, CPU)",
    "H06": "Normalizasyon yok (use_norm=0): ne capa ne sigma",
    "H07": "x_mark_enc=None: zaman-ozniteligi tokenlari kaldirildi (12 -> 8 token)",
    "H09": "GroupDRO (K=4 kronolojik grup, eta=0.01) ile egitim (LastNorm uzerine)",
    "H10": "LastNorm - sigma: x - x_son (olcekleme yok)",
    "H11": "SAN std basligi ReLU -> Softplus (olu-ReLU duzeltmesi; on-kosul kapisi)",
    "H12": "Model-Repeat birlesimi C = R + clip(alpha,0,1)(P-R), alpha ic-holdout'tan",
    "H13": "Model-Repeat olcekleme C = R + max(alpha,0)(P-R), alpha ic-holdout'tan",
    "H14": "Sabit esit agirlikli birlesim C = 0.5P + 0.5R",
}
NEDEN = {
    "H01": "Val +31.5%, 4/4 ufuk kotu, Repeat'ten de kotu; std basligi son ReLU ilk on-egitim epoch'unda oluyor "
           "(tahmin std %89-100 sifir) -> omurga katkisi ~0, model capa+sabit tahminciye cokuyor",
    "H02": "Etkisiz: val -0.17% (MDE'nin cok altinda), iyilesen ufuk 2/4 < 3",
    "H05": "On-kosul kapisi O1: R_H (SAN-mu/naif) 0/4 ufukta < 1 (ort. 1.236) -> kosulmadi",
    "H06": "Val +392%, 0/4 ufuk; ablasyon satiri ('-capa -sigma') olarak tutuldu",
    "H07": "Etki yok: val -0.29% (2/3 seed), MDE'nin 0.04 kati; H0 desteklendi (zincir ebeveyni lastnorm oldu)",
    "H09": "Val -5.02% ama kazancin %107'si h720'den; diger 3 ufuk +0.88% (3/3 kotu) -> ufuk kurali 1/4",
    "H10": "Val +1.01%, 1/4 ufuk (kisa ufuklarda +6%); test'te sigma tasiyici cikti (+10.2%)",
    "H11": "Softplus olu orani K3 0.83-0.88 (esik <=0.05); sigma-hat ~1e-3 vs gercek ~0.05 -> hipotez sinanamadi",
    "H12": "3-kat A +1.50%; K1'de Repeat'e tam cokus (ic-holdout alpha<=0), kat yonleri tutarsiz (1/3 kat, 2/4 ufuk)",
    "H13": "3-kat A +1.88%; H12 ile 9 ciftin 5'inde bit-bit ayni, alpha>1 dilimlerinde zarar",
    "H14": "3-kat A ebeveynden kotu (+0.5%); 'esit agirlik >= en iyi tekil' beklentisi K1/K2'de ihlal",
}


def oku_csv(yol):
    with open(WS / yol, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resmi_val_test():
    r = oku_csv("sonuclar/test-sonuclari.csv")
    d = {}
    for x in r:
        d.setdefault(x["varyant"], []).append(x)
    for v in d.values():
        v.sort(key=lambda x: x["seed"])
    return d


def kat_ozet(r_kat, varyant, kat_list=(1, 2, 3)):
    """(kat,seed,ufuk)->val_mse; A-bar, kat-ort (seed-ort), ufuk-ort (kat x seed)."""
    a = {}
    for x in r_kat:
        if x["varyant"] == varyant and int(x["kat"]) in kat_list:
            a[(int(x["kat"]), int(x["seed"]), x["ufuk"])] = float(x["val_mse"])
    return a


def esit_ozet():
    a, ids = {}, []
    for p in sorted(glob.glob(str(WS / "runs" / "*_lastnorm-esit-k*_s*"))):
        m = json.loads((Path(p) / "metrics.json").read_text(encoding="utf-8"))
        done = json.loads((Path(p) / ".done").read_text(encoding="utf-8"))
        assert done["rc"] == 0
        k, s = int(m["kat"]), int(m["seed"])
        for H in UF:
            a[(k, s, str(H))] = float(m[f"val_mse_h{H}"])
        a[(k, s, "avg")] = float(m["val_mse_avg"])
        ids.append(Path(p).name)
    assert len(ids) == 9
    return a, ids


def kat_istatistik(v, e):
    """v,e: dict (kat,seed,ufuk)->mse. Doner: A_v, A_e, delta%, kat_iyi, ufuk_iyi, p (9 cift eslestirilmis t)."""
    katlar = sorted({k for k, _, _ in v}); seeds = sorted({s for _, s, _ in v})
    Av = np.mean([v[(k, s, "avg")] for k in katlar for s in seeds])
    Ae = np.mean([e[(k, s, "avg")] for k in katlar for s in seeds])
    kat_iyi = sum(np.mean([v[(k, s, "avg")] for s in seeds]) < np.mean([e[(k, s, "avg")] for s in seeds]) for k in katlar)
    ufuk_iyi = sum(np.mean([v[(k, s, str(H))] for k in katlar for s in seeds]) <
                   np.mean([e[(k, s, str(H))] for k in katlar for s in seeds]) for H in UF)
    p = stats.ttest_rel([v[(k, s, "avg")] for k in katlar for s in seeds],
                        [e[(k, s, "avg")] for k in katlar for s in seeds]).pvalue
    return Av, Ae, (Av / Ae - 1) * 100, int(kat_iyi), int(ufuk_iyi), float(p), len(katlar)


def yaz_md(yol, baslik, satirlar, sutunlar, notlar):
    L = [f"# {baslik}", "", *[f"> {n}" for n in notlar], "", "| " + " | ".join(sutunlar) + " |",
         "|" + "---|" * len(sutunlar)]
    for s in satirlar:
        L.append("| " + " | ".join("" if s.get(c) in (None, "") else str(s.get(c)) for c in sutunlar) + " |")
    (O.OUT / yol).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"yazildi: sonuclar/makale/{yol}")


def main():
    rv = resmi_val_test()
    r_kat = oku_csv("sonuclar/tahlil-tur4-kat.csv")
    esit, esit_ids = esit_ozet()
    vsel = oku_csv("sonuclar/vsel-test.csv")
    h05 = oku_csv("sonuclar/h05-onkosul.csv")
    h11 = oku_csv("sonuclar/h11-kapi.csv")
    base_test = np.mean([float(x["test_mse_avg"]) for x in rv["baseline"]])
    satirlar, run_ids, dog = [], [], 0

    for kart, (var, tur, eb, tip) in KART.items():
        s = dict(kart=kart, varyant=var, tur=tur, ebeveyn=eb, metrik_tanimi="", deger="", ebeveyn_deger="",
                 delta_pct="", ufuk_iyi="", kat_iyi="", p_eslesik="", karar=KARAR[kart][0], test_mse_avg="",
                 test_delta_ebeveyn_pct="", test_delta_baseline_pct="", karar_kaynagi=KARAR[kart][1])
        if tip == "resmi":
            v, e = rv[var], rv[eb]
            assert [x["seed"] for x in v] == [x["seed"] for x in e] == ["2023", "2024", "2025"]
            va = np.array([float(x["val_mse_avg"]) for x in v]); ea = np.array([float(x["val_mse_avg"]) for x in e])
            dp = (va.mean() / ea.mean() - 1) * 100
            ui = sum(np.mean([float(x[f"val_mse_h{H}"]) for x in v]) < np.mean([float(x[f"val_mse_h{H}"]) for x in e])
                     for H in UF)
            ta = np.array([float(x["test_mse_avg"]) for x in v]); te = np.array([float(x["test_mse_avg"]) for x in e])
            s.update(metrik_tanimi="resmi val_mse_avg (4 ufuk ort., 3 seed ort.)", deger=round(va.mean(), 5),
                     ebeveyn_deger=round(ea.mean(), 5), delta_pct=round(dp, 2), ufuk_iyi=f"{int(ui)}/4",
                     p_eslesik=round(float(stats.ttest_rel(va, ea).pvalue), 4),
                     test_mse_avg=f"{ta.mean():.4f}+-{ta.std(ddof=1):.4f}",
                     test_delta_ebeveyn_pct=round((ta.mean() / te.mean() - 1) * 100, 2),
                     test_delta_baseline_pct=round((ta.mean() / base_test - 1) * 100, 2))
            run_ids += [x["run_id"] for x in v]
            rd, ru, _ = RAPOR[kart]
            assert abs(dp - rd) < 0.011 * max(1, abs(rd) / 10), (kart, dp, rd)
            if ru is not None:
                assert ui == ru, (kart, ui, ru)
            assert abs(s["test_delta_baseline_pct"] - TEST_RAPOR[var]) < 0.011, (kart, s["test_delta_baseline_pct"])
            dog += 1
        elif tip == "kat3":
            vv = esit if var == "lastnorm-esit" else kat_ozet(r_kat, var)
            ee = kat_ozet(r_kat, eb)
            Av, Ae, dp, ki, ui, p, nk = kat_istatistik(vv, ee)
            s.update(metrik_tanimi="FAZ II 3-kat A-bar (K1-K3 x 3 seed ort. val_mse_avg)", deger=round(Av, 5),
                     ebeveyn_deger=round(Ae, 5), delta_pct=round(dp, 2), ufuk_iyi=f"{ui}/4", kat_iyi=f"{ki}/{nk}",
                     p_eslesik=round(p, 4), test_mse_avg="YOK (test hesaplanmadi)" if var != "lastnorm" else "")
            if var == "lastnorm":  # H04-kat: test degeri H04 satiriyla ayni kosumlar (K3)
                ta = np.array([float(x["test_mse_avg"]) for x in rv["lastnorm"]])
                s.update(test_mse_avg=f"{ta.mean():.4f}+-{ta.std(ddof=1):.4f}",
                         test_delta_ebeveyn_pct=round((ta.mean() / base_test - 1) * 100, 2),
                         test_delta_baseline_pct=round((ta.mean() / base_test - 1) * 100, 2))
            run_ids += sorted({x["run_id"] for x in r_kat if x["varyant"] in (var, eb) and x["kat"] in "123"})
            if var == "lastnorm-esit":
                run_ids += esit_ids
            rd, ru, rk = RAPOR[kart]
            assert abs(dp - rd) < (0.06 if kart == "H14" else 0.011), (kart, dp, rd)
            if ru is not None:
                assert ui == ru, (kart, ui, ru)
            if rk is not None:
                assert ki == rk, (kart, ki, rk)
            dog += 1
        elif kart == "H05":
            kapi = [x for x in h05 if x["split"] == "KAPI"][0]
            ort = [x for x in h05 if x["split"] == "val" and x["seed"] == "ort"]
            s.update(metrik_tanimi="on-kosul kapisi (val, CPU): R_H = E_SAN-mu / E_naif, O1: R_H<1 ufuk sayisi",
                     deger="R_H " + " / ".join(f"{float(x['R']):.3f}" for x in ort) + f" (ort. {float(kapi['R']):.3f})",
                     ufuk_iyi=f"{sum(float(x['R']) < 1 for x in ort)}/4", test_mse_avg="YOK (kosulmadi)")
            assert abs(float(kapi["R"]) - 1.2363) < 1e-3
            run_ids += sorted({x["kosum"] for x in h05 if x["kosum"].startswith("2026")})
            dog += 1
        elif kart == "H11":
            sp = [x for x in h11 if x["kol"] == "softplus"]
            k3 = [float(x["olu_orani"]) for x in sp if x["kat"] == "3"]
            k1 = [float(x["olu_orani"]) for x in sp if x["kat"] == "1"]
            s.update(metrik_tanimi="on-kosul kapisi (CPU): Softplus olu orani (esik <=0.05), G1/G2/G3",
                     deger=f"olu orani K3 {min(k3):.2f}-{max(k3):.2f}, K1 {min(k1):.2f}-{max(k1):.2f}; "
                           + [x for x in h11 if x["kol"] == "KAPI"][0]["kapi"],
                     test_mse_avg="YOK (kosulmadi)")
            assert 0.82 < min(k3) and max(k3) < 0.88
            dog += 1
        elif kart == "H15":
            per = {}
            for veri in ("exchange", "etth1", "etth2"):
                rr = [x for x in vsel if x["veri"] == veri]
                t = {k: [np.mean([float(x[f"test_{k}"]) for x in rr if x["seed"] == sd]) for sd in ("2023", "2024", "2025")]
                     for k in ("vsel", "baseline")}
                per[veri] = (np.mean(t["vsel"]) / np.mean(t["baseline"]) - 1) * 100
                assert abs(per[veri] - VSEL_RAPOR[veri]) < 0.011, (veri, per[veri])
            rx = [x for x in vsel if x["veri"] == "exchange"]
            vv = np.mean([float(x["val_vsel"]) for x in rx]); vb = np.mean([float(x["val_baseline"]) for x in rx])
            s.update(metrik_tanimi="val-secim-ici VSEL (Exchange; secim val'de yapildigi icin iyimser) + test 3 veri",
                     deger=round(vv, 5), ebeveyn_deger=round(vb, 5), delta_pct=round((vv / vb - 1) * 100, 2),
                     test_mse_avg="; ".join(f"{k} {v:+.2f}%" for k, v in per.items()),
                     test_delta_baseline_pct=round(per["exchange"], 2))
            dog += 1
        satirlar.append(s)
    run_ids = list(dict.fromkeys(run_ids))

    meta = O.meta_satirlari(
        "makale_kart_tablolari.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_kart_tablolari.py",
        ["metrik_tanimi heterojen: tur 1-3 resmi val_mse_avg; tur 4-5 FAZ II 3-kat A-bar (Exchange test 15:42'den SONRA "
         "tasarlandi); H05/H11 kapi (Delta yok); H15 test-sonrasi kesifsel. F3 figurunde iki panel/isaret ile ayrilmali.",
         "delta_pct = 100*(deger/ebeveyn_deger-1); ufuk_iyi = 3-seed (kat3: kat x seed) ort. Delta_H<0 ufuk sayisi; "
         "p_eslesik: tur1-3 3 seed, kat3 9 (kat,seed) cifti eslestirilmis t",
         "test_* sutunlari YALNIZ RAPOR (secimde kullanilmadi); test_delta_baseline_pct baseline 3-seed test ort.'ya gore",
         "karar ve karar_kaynagi tahlil dosyalarindan aktarim; betik karar vermez",
         f"dogrulama: {dog} kartta hesaplanan Delta/ufuk/kat sayilari tahlil ve leaderboard-test degerleriyle eslesti"],
        run_ids)
    O.csv_yaz(O.OUT / "delta-vs-tur.csv", satirlar, meta)
    sut = ["kart", "varyant", "tur", "ebeveyn", "metrik_tanimi", "deger", "ebeveyn_deger", "delta_pct", "ufuk_iyi",
           "kat_iyi", "p_eslesik", "karar", "test_mse_avg", "test_delta_ebeveyn_pct", "test_delta_baseline_pct"]
    yaz_md("delta-vs-tur.md", "Delta-vs-tur birlesik tablosu (H01-H15)", satirlar, sut,
           [m.lstrip("# ") for m in meta])

    # ---- T6 basarisiz denemeler
    t6 = []
    for kart in ["H01", "H02", "H05", "H06", "H07", "H09", "H10", "H11", "H12", "H13", "H14"]:
        s = next(x for x in satirlar if x["kart"] == kart)
        t6.append(dict(kart=kart, varyant=s["varyant"], tur=s["tur"], mekanizma=MEKANIZMA[kart], ebeveyn=s["ebeveyn"],
                       val_metrik=s["metrik_tanimi"],
                       val_delta_pct=s["delta_pct"] if s["delta_pct"] != "" else s["deger"],
                       ufuk_iyi=s["ufuk_iyi"], karar=s["karar"], neden_oldu=NEDEN[kart], test_mse_avg=s["test_mse_avg"],
                       test_delta_ebeveyn_pct=s["test_delta_ebeveyn_pct"],
                       test_delta_baseline_pct=s["test_delta_baseline_pct"], kaynak=s["karar_kaynagi"]))
    meta6 = O.meta_satirlari(
        "makale_kart_tablolari.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_kart_tablolari.py",
        ["T6 Basarisiz denemeler: secim val'de (on-kayitli kural) yapildi; test sutunu YALNIZ RAPOR.",
         "DURUSTLUK: H01 SAN test'te en iyi (0.3007, iTr'den -17.5%, Repeat 0.3410'dan da iyi) ama val'de +31.5% kotuydu "
         "ve cokmus (capa+sabit) bir tahminciydi; test'e bakip secmek yasak oldugundan secilmedi. Tartisma'da val-test "
         "uyumsuzlugu (Spearman 0.37) ve Repeat'in gucuyle ayni olgu olarak raporlanir.",
         "H06 ve H10 ayni zamanda ablasyon satirlaridir (T4); H07 oldurulmedi (TUT-ayirt edilemez) ama zincire alinmadi.",
         "neden_oldu metinleri tahlil-tur1..4.md / orkestrasyon_durumu.md'den ozetlenmistir (kaynak sutunu)"],
        run_ids)
    O.csv_yaz(O.OUT / "basarisiz-denemeler.csv", t6, meta6)
    yaz_md("basarisiz-denemeler.md", "T6 -- Basarisiz denemeler", t6,
           ["kart", "varyant", "tur", "mekanizma", "ebeveyn", "val_delta_pct", "ufuk_iyi", "karar", "neden_oldu",
            "test_mse_avg", "test_delta_ebeveyn_pct", "test_delta_baseline_pct"], [m.lstrip("# ") for m in meta6])
    for s in satirlar:
        print(f"{s['kart']:8s} tur {s['tur'][:6]:6s} D {str(s['delta_pct']):>8s}  ufuk {s['ufuk_iyi']:5s} kat {s['kat_iyi']:4s}"
              f" | {s['karar'][:40]:40s} | test {s['test_mse_avg']} ({s['test_delta_baseline_pct']})")
    print(f"DOGRULAMA GECTI: {dog} kart")


if __name__ == "__main__":
    main()
