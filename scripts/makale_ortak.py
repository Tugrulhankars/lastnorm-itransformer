# -*- coding: utf-8 -*-
"""Makale son-islem betikleri icin ortak yardimcilar (makale icin sonradan eklendi -- deney-gelistirici, 2026-10-15).

Dogrudan calistirilmaz; scripts/makale_*.py tarafindan ice aktarilir. Oz-sinama:
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/makale_ortak.py

YALNIZ OKUR: runs/<id>/{.done,metrics.json,test_metrics.json,preds/val_h*.npz,preds/test_kilitli/test_h*.npz},
data/ (ham CSV, splits JSON, hedef .npy). Hicbir mevcut dosyayi degistirmez; ciktilar sonuclar/makale/ altina.
Test dosyalari 2026-10-14 15:42 (Exchange) ve 2026-10-15 01:30 (ETT) resmi acilislarda zaten acilmisti; bu betikler
yalniz RAPORLAMA yapar, hicbir secim/ayar uretmez.
"""
import csv
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
OUT = WS / "sonuclar" / "makale"
UFUKLAR = [96, 192, 336, 720]
SEQ = 96
SEEDS = [2023, 2024, 2025]
K_BLOK = 4

# Kaynak kosumlar (rc=0, tam 4-ufuk, n=3 seed). Exchange: sonuclar/test-sonuclari.csv run_id sutunu ile ayni;
# ETT: runs/BILDIRIM.log "BITTI ... rc=0" satirlari (varyantlar/vsel/vsel.py ile ayni secim; yetim rc=96 kosumlar haric).
RUNS = {
    "exchange": {"itr": ["20261014-020531_baseline_s2023", "20261014-021024_baseline_s2024", "20261014-021448_baseline_s2025"],
                 "lastnorm": ["20261014-052508_lastnorm_s2023", "20261014-053601_lastnorm_s2024", "20261014-054213_lastnorm_s2025"]},
    "etth1": {"itr": ["20261014-230036_etth1-baseline_s2023", "20261014-231547_etth1-baseline_s2024", "20261014-233004_etth1-baseline_s2025"],
              "lastnorm": ["20261014-233943_etth1-lastnorm_s2023", "20261014-234744_etth1-lastnorm_s2024", "20261015-001831_etth1-lastnorm_s2025"]},
    "etth2": {"itr": ["20261015-002817_etth2-baseline_s2023", "20261015-003636_etth2-baseline_s2024", "20261015-010142_etth2-baseline_s2025"],
              "lastnorm": ["20261015-010801_etth2-lastnorm_s2023", "20261015-011429_etth2-lastnorm_s2024", "20261015-012039_etth2-lastnorm_s2025"]},
}
VERI = {
    "exchange": dict(csv="data/exchange_rate/exchange_rate.csv", split="data/splits/exchange_rate_tslib_sl96.json",
                     hedef="data/hedefler_sl96"),
    "etth1": dict(csv="data/etth1/ETTh1.csv", split="data/splits/etth1_sl96.json", hedef="data/hedefler_sl96_etth1"),
    "etth2": dict(csv="data/etth2/ETTh2.csv", split="data/splits/etth2_sl96.json", hedef="data/hedefler_sl96_etth2"),
}
MODEL_AD = {"itr": "iTransformer (resmi, RevIN-tarzi use_norm=1)", "lastnorm": "iTransformer + LastNorm (H04)",
            "repeat": "Naif Repeat (son gozlem)"}


def kosumlari_dogrula():
    """Her kaynak kosum rc=0 .done damgali, seed dogru; ETT kimlikleri BILDIRIM.log rc=0 satirlariyla ayni."""
    log = (WS / "runs/BILDIRIM.log").read_text(encoding="utf-8")
    for veri, d in RUNS.items():
        for model, ids in d.items():
            for sd, rid in zip(SEEDS, ids):
                done = json.loads((WS / "runs" / rid / ".done").read_text(encoding="utf-8"))
                assert done["rc"] == 0, (rid, done["rc"])
                assert rid.endswith(f"_s{sd}"), rid
                assert (WS / "runs" / rid / "test_metrics.json").exists(), rid
            if veri != "exchange":
                y = "baseline" if model == "itr" else "lastnorm"
                ids_log = re.findall(rf"BİTTİ (\d{{8}}-\d{{6}}_{veri}-{y}_s(\d+)) rc=0", log)
                beklenen = [next(i for i, s in ids_log if s == str(sd)) for sd in SEEDS]
                assert beklenen == ids, (veri, model, beklenen, ids)
    # Exchange: test-sonuclari.csv ile ayni kimlikler
    with open(WS / "sonuclar/test-sonuclari.csv", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    for model, var in (("itr", "baseline"), ("lastnorm", "lastnorm")):
        ids = [x["run_id"] for x in r if x["varyant"] == var]
        assert ids == RUNS["exchange"][model], (model, ids)
    return True


def split(veri):
    return json.loads((WS / VERI[veri]["split"]).read_text(encoding="utf-8"))


def yukle_z(veri):
    """Ham CSV -> train-only StandardScaler (splits JSON) ile z uzayi. Doner: z (T,C), degisken adlari."""
    sp = split(veri)
    with open(WS / VERI[veri]["csv"], encoding="utf-8") as f:
        r = list(csv.reader(f))
    cols = r[0][1:]
    X = np.array([[float(v) for v in row[1:]] for row in r[1:]], dtype=np.float64)
    mu = np.array([sp["scaler_train_only"]["mean"][c] for c in cols])
    sc = np.array([sp["scaler_train_only"]["scale"][c] for c in cols])
    return (X - mu) / sc, cols


def kendi_satirlari(veri, bolum):
    o0, o1 = split(veri)["dilim"][bolum]["kendi_satirlari[o0,o1)"]
    return int(o0), int(o1)


def blok_sinirlari(veri, bolum):
    """hata_analizi.py ile ayni tanim: dilimin KENDI satirlari np.array_split ile K=4 bitisik kronolojik bloga."""
    o0, o1 = kendi_satirlari(veri, bolum)
    parcalar = np.array_split(np.arange(o0, o1), K_BLOK)
    b = [(int(p[0]), int(p[-1]) + 1) for p in parcalar]
    if bolum == "test":  # splits JSON'daki test ceyrekleriyle birebir olmali
        q = split(veri)["test_ceyrekleri_kendi_satirlari"]
        assert b == [tuple(q[f"test_Q{k + 1}"]["satir[o0,o1)"]) for k in range(K_BLOK)], (veri, b)
    return b


def blok_adlari(bolum):
    return [f"Q{k + 1}" for k in range(K_BLOK)] if bolum == "test" else [f"B{k + 1}" for k in range(K_BLOK)]


def npz_yolu(rid, bolum, H):
    p = WS / "runs" / rid / "preds"
    return p / "test_kilitli" / f"test_h{H}.npz" if bolum == "test" else p / f"val_h{H}.npz"


def yukle_pred(rid, bolum, H):
    return np.load(npz_yolu(rid, bolum, H))


def yukle_true(veri, bolum, H):
    return np.load(WS / VERI[veri]["hedef"] / f"{bolum}_h{H}_true.npy").astype(np.float64)


def kayitli_mse(rid, bolum, H):
    """Kosumun kendi kaydettigi MSE: val -> metrics.json val_mse_hH; test -> test_metrics.json test_mse_hH."""
    if bolum == "test":
        return float(json.loads((WS / "runs" / rid / "test_metrics.json").read_text(encoding="utf-8"))[f"test_mse_h{H}"])
    return float(json.loads((WS / "runs" / rid / "metrics.json").read_text(encoding="utf-8"))[f"val_mse_h{H}"])


def repeat_ve_girdi(z, ilk, H):
    """Repeat tahmini (N,H,C) = girdi penceresinin son satiri; girdi penceresi (N,SEQ,C); hedef (N,H,C) ham z'den."""
    lb = (ilk - SEQ)[:, None] + np.arange(SEQ)[None, :]
    X = z[lb]
    R = np.repeat(X[:, -1:, :], H, axis=1)
    Y = z[ilk[:, None] + np.arange(H)[None, :]]
    return R, X, Y


def meta_satirlari(betik, komut, aciklama, run_ids):
    """CSV basina yazilan '#' yorum satirlari (okuma: pandas.read_csv(f, comment='#'))."""
    s = [f"# betik: scripts/{betik}  (makale icin sonradan eklendi -- deney-gelistirici)",
         f"# komut (WS_D kokunden): {komut}",
         f"# uretim: {_dt.datetime.now().isoformat(timespec='seconds')}",
         "# TEST NOTU: test tahminleri resmi acilislarda (Exchange 2026-10-14 15:42:23; ETT 2026-10-15 01:30:32) zaten "
         "acilmisti; bu dosya yalniz raporlamadir, hicbir secim/ayar icin kullanilmadi."]
    s += [f"# {a}" for a in aciklama]
    s.append("# kaynak_run_id: " + " ".join(run_ids))
    return s


def csv_yaz(yol, satirlar, meta, alanlar=None):
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    alanlar = alanlar or list(satirlar[0].keys())
    with open(yol, "w", newline="", encoding="utf-8") as f:
        for m in meta:
            f.write(m.replace("\n", " ") + "\n")
        w = csv.DictWriter(f, fieldnames=alanlar)
        w.writeheader()
        w.writerows(satirlar)
    print(f"yazildi: {yol.relative_to(WS)} ({len(satirlar)} satir)")


def tum_run_ids(veriler=None):
    veriler = veriler or list(RUNS)
    return [rid for v in veriler for m in ("itr", "lastnorm") for rid in RUNS[v][m]]


def r6(x):
    return round(float(x), 6)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("kosum dogrulama:", kosumlari_dogrula())
    for v in RUNS:
        print(v, "val bloklari", blok_sinirlari(v, "val"), "test bloklari", blok_sinirlari(v, "test"))
