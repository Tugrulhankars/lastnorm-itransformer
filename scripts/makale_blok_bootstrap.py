# -*- coding: utf-8 -*-
"""Makale -- donem belirsizligi: Exchange TEST icin hareketli-blok bootstrap (MBB) %95 GA.
(makale icin sonradan eklendi -- deney-gelistirici, 2026-10-15; CPU son-islem, egitim/GPU yok)

Calistirma (WS_D kokunden, Git Bash):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_blok_bootstrap.py

Neden: seed-eslestirilmis t testi (leaderboard-test.md) yalniz BASLATMA gurultusunu olcer; tek test donemi (1517 satir)
icin donem/orneklem belirsizligi olculmemisti (manifest A-6, karar.md S3). Bu betik onu yaklasik olarak olcer.

TANIM (onceden sabit):
  - Pencere-duzeyi kayip: L_m(i) = pencere i'nin tum adim x degisken MSE'si (npz 'mse_pencere_degisken' ortalamasi);
    modeller icin 3 seed ORTALAMASI (seed gurultusu bu GA'ya dahil DEGIL); Repeat seed'siz, CSV + train-only scaler.
  - Fark serisi d(i) = L_A(i) - L_B(i), i = 0..N-1 (pencereler tahmin kokenine gore kronolojik; ardisik pencereler H-1
    hedef satirini paylasir -> guclu otokorelasyon).
  - MBB: blok uzunlugu b = H pencere; her tekrarda ceil(N/b) blok, baslangiclar U{0..N-b}'den iadeli; birlestirilip ilk
    N tutulur; istatistik = ort. d (ve 100*ort.d/ort.L_B). B = 1000 tekrar, numpy default_rng(20261015). GA = yuzdelik
    [2.5, 97.5]. p_bir_yonlu = P*(ort. d >= 0) (bootstrap dagiliminda; kaba gosterge, test degil).
  - "avg" satiri: 4 ufuk ortalamasi; her tekrarda ufuklar AYRI MBB ile orneklenir ama ayni tekrar indeksi icin ayni
    goreli baslangic konumlari (u ~ U[0,1), baslangic = floor(u*(N_H-b_H+1))) kullanilir -> ufuklar arasi donem
    eslestirmesi yaklasiktir (beyan).
  - Karsilastirmalar: LastNorm - iTr ; LastNorm - Repeat ; iTr - Repeat.
SINIR: H=720'de N=798, b=720 -> yalniz 79 farkli blok baslangici; GA kaba. Bu bir donem-duyarlilik analizidir,
  bagimsiz test donemi yerine gecmez.
DOGRULAMA (assert): nokta tahminleri test_metrics.json ort. (iTr, LastNorm) ve naif-test.csv (Repeat) ile 1e-5.
Cikti: sonuclar/makale/blok-bootstrap.csv
  ufuk,karsilastirma,n_pencere,blok,n_tekrar,delta_nokta,delta_pct_nokta,ga_alt,ga_ust,ga_pct_alt,ga_pct_ust,
  p_bir_yonlu_delta_ge0,ayri_blok_baslangici,boot_ort,uyari
  (boot_ort = bootstrap dagiliminin ortalamasi; b/N > 0.5 ise MBB kenar pencerelerini az temsil eder -> uyari)
"""
import sys

import numpy as np

import makale_ortak as O
from makale_adim_egrisi import naif_test

sys.stdout.reconfigure(encoding="utf-8")
B = 1000
RNG_SEED = 20261015
KARS = [("lastnorm", "itr"), ("lastnorm", "repeat"), ("itr", "repeat")]


def main():
    O.kosumlari_dogrula()
    veri = "exchange"
    z, _ = O.yukle_z(veri)
    nt = naif_test(veri)
    L = {}
    for H in O.UFUKLAR:
        Y = O.yukle_true(veri, "test", H)
        d = {}
        ilk = None
        for m in ("itr", "lastnorm"):
            ser = []
            for rid in O.RUNS[veri][m]:
                f = O.yukle_pred(rid, "test", H)
                ilk = f["hedef_ilk_satir"].astype(np.int64) if ilk is None else ilk
                assert np.array_equal(ilk, f["hedef_ilk_satir"])
                s = f["mse_pencere_degisken"].astype(np.float64).mean(1)
                assert abs(s.mean() - O.kayitli_mse(rid, "test", H)) < 1e-5
                ser.append(s)
            d[m] = np.mean(ser, 0)
        R, _, _ = O.repeat_ve_girdi(z, ilk, H)
        d["repeat"] = ((R - Y) ** 2).mean(axis=(1, 2))
        assert abs(d["repeat"].mean() - nt[H]) < 1e-6
        L[H] = d
    rng = np.random.default_rng(RNG_SEED)
    nb_max = max(int(np.ceil(len(L[H]["itr"]) / H)) for H in O.UFUKLAR)
    U = rng.random((B, nb_max))  # ortak goreli baslangiclar
    idx = {}
    for H in O.UFUKLAR:
        N = len(L[H]["itr"]); b = H; nb = int(np.ceil(N / b))
        st = np.floor(U[:, :nb] * (N - b + 1)).astype(np.int64)  # (B,nb)
        ii = (st[:, :, None] + np.arange(b)[None, None, :]).reshape(B, -1)[:, :N]
        idx[H] = ii
    satirlar = []
    for a_, b_ in KARS:
        boot_d, boot_ref, nok_d, nok_ref = [], [], [], []
        for H in O.UFUKLAR:
            dd = L[H][a_] - L[H][b_]
            ref = L[H][b_]
            bd = dd[idx[H]].mean(1); br = ref[idx[H]].mean(1)
            boot_d.append(bd); boot_ref.append(br); nok_d.append(dd.mean()); nok_ref.append(ref.mean())
            satirlar.append(_satir(H, f"{a_}-{b_}", len(dd), H, dd.mean(), ref.mean(), bd, br, len(dd) - H + 1))
        bd = np.mean(boot_d, 0); br = np.mean(boot_ref, 0)
        satirlar.append(_satir("avg", f"{a_}-{b_}", "", "H (ufuk basina)", np.mean(nok_d), np.mean(nok_ref), bd, br, ""))
    # nokta tahmini avg denetimi (Exchange test raporlanan degerler)
    ln_itr = next(s for s in satirlar if s["ufuk"] == "avg" and s["karsilastirma"] == "lastnorm-itr")
    assert abs(ln_itr["delta_pct_nokta"] - (-4.96)) < 0.02, ln_itr
    for s in satirlar:
        print(f"{s['karsilastirma']:16s} H={str(s['ufuk']):4s} d={s['delta_nokta']:+.5f} ({s['delta_pct_nokta']:+.2f}%)"
              f"  %95 GA [{s['ga_alt']:+.5f}, {s['ga_ust']:+.5f}] = [{s['ga_pct_alt']:+.2f}%, {s['ga_pct_ust']:+.2f}%]"
              f"  P*(d>=0)={s['p_bir_yonlu_delta_ge0']}")
    meta = O.meta_satirlari(
        "makale_blok_bootstrap.py", "PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_blok_bootstrap.py",
        ["Exchange TEST; pencere kaybi = adim x degisken MSE, modeller 3 seed ortalamasi (seed gurultusu GA'ya DAHIL DEGIL)",
         f"hareketli-blok bootstrap: blok = H pencere, {B} tekrar, rng {RNG_SEED}, yuzdelik %95 GA; avg = 4 ufuk ort. "
         "(ortak goreli blok konumlari; ufuklar arasi donem eslestirmesi yaklasik)",
         "delta = A - B (karsilastirma 'A-B'); delta_pct = 100*delta/ort.L_B; p_bir_yonlu_delta_ge0 = bootstrap "
         "dagiliminda delta>=0 orani (kaba gosterge)",
         "SINIR: H=720'de yalniz 79 farkli blok baslangici ve b/N=0.90 -> MBB kenar yanliligi (boot_ort != nokta; uyari "
         "sutunu); tek test donemi -> donem-duyarlilik analizi, bagimsiz kanit degil",
         "dogrulama: nokta tahminleri test_metrics.json / naif-test.csv ile 1e-5; avg LN-iTr = -4.96%"],
        O.RUNS[veri]["itr"] + O.RUNS[veri]["lastnorm"])
    O.csv_yaz(O.OUT / "blok-bootstrap.csv", satirlar, meta)
    print("DOGRULAMA GECTI")


def _satir(H, ad, n, blok, d0, r0, bd, br, n_bas):
    pct = 100 * bd / br
    return dict(ufuk=H, karsilastirma=ad, n_pencere=n, blok=blok, n_tekrar=B, delta_nokta=O.r6(d0),
                delta_pct_nokta=round(float(100 * d0 / r0), 3),
                ga_alt=O.r6(np.percentile(bd, 2.5)), ga_ust=O.r6(np.percentile(bd, 97.5)),
                ga_pct_alt=round(float(np.percentile(pct, 2.5)), 3), ga_pct_ust=round(float(np.percentile(pct, 97.5)), 3),
                p_bir_yonlu_delta_ge0=round(float(np.mean(bd >= 0)), 4), ayri_blok_baslangici=n_bas,
                boot_ort=O.r6(bd.mean()),
                uyari=("b/N=%.2f: MBB kenar yanliligi, bootstrap dagilimi nokta tahminine gore merkezsiz; GA kaba" % (H / n))
                if isinstance(H, int) and H / n > 0.5 else ("H=720 bileseni kenar yanlili" if H == "avg" else ""))


if __name__ == "__main__":
    main()
