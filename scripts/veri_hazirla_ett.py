#!/usr/bin/env python
"""FAZ II P2 veri hazirligi -- ETTh1 ve ETTh2, iTransformer/TSLib Dataset_ETT_hour protokolu.

Exchange betigi (scripts/veri_hazirla.py) ile ayni kalip; ona DOKUNMAZ. Idempotent: dosya varsa yeniden
indirmez, hash'i yine dogrular (--force yeniden indirir).
Kullanim (WS kokunden, Git Bash):
    PYTHONIOENCODING=utf-8 HF_HOME=data/hf_cache env/.venv/Scripts/python.exe -I -X utf8 scripts/veri_hazirla_ett.py [--force]

Ciktilar (WS/data altinda), <ad> = etth1 | etth2:
    <ad>/ETTh?.csv                               -- TSLib'in kullandigi dosya (HF thuml, sabit revizyon)
    <ad>/kaynak_etdataset/ETTh?.csv              -- ETDataset orijinali (zhouhaoyi, sabit commit) -- karsilastirma
    splits/<ad>_sl96.json                        -- resmi sabit sinirlar + pencere sayilari + 3-kat dilimleri
    kayma_istatistikleri_<ad>.csv                -- dilim x degisken dagilim/kayma tablosu
    kayma_mmd_<ad>.csv                           -- MMD tanim varyantlari (Exchange kayma_mmd.csv ile ayni tanim)
    naive_val_<ad>.csv                           -- Repeat / sifir baseline, YALNIZ val dilimleri (K1,K2,K3=resmi val)
    data_manifest_ett.json                       -- tum sayilar (iki veri seti)

TEST DILIMI UZERINDE HICBIR TAHMIN/HATA HESAPLANMAZ; testte yalniz betimsel dagilim istatistigi.

FAZ III genislemesi (2026-10-15, veri-hazirlayici; H16 on-kaydi): --veri ettm1 ettm2 weather solar
    PYTHONIOENCODING=utf-8 HF_HOME=data/hf_cache env/.venv/Scripts/python.exe -I -X utf8 scripts/veri_hazirla_ett.py \
        --veri ettm1 ettm2 weather solar
  ETT yolu (etth1/etth2, varsayilan) DEGISMEDI: ayni fonksiyon, ayni ciktilar; FAZ III setleri ayri fonksiyonda
  (hazirla_faz3) islenir ve data_manifest_faz3.json'a yazilir (data_manifest_ett.json'a dokunulmaz).
  FAZ III ciktilari, <ad> = ettm1 | ettm2 | weather | solar:
    <ad>/<dosya>                                 -- iTransformer/TSLib dosyasi (kaynak tablosu FAZ3 sozlugunde)
    splits/<ad>_sl96.json                        -- resmi sinirlar (data_loader.py ile birebir) + pencere sayilari
    hedefler_sl96_<ad>/{val,test}_h{H}_true.npy  -- RESMI Dataset sinifinin data_y'sinden (rakip_kos.py
                                                    hedef_dogrula_yaz bicimi: float32 (N,H,C), olcekli uzay)
    kayma_istatistikleri_<ad>.csv                -- degisken x dilim betimsel istatistik (MMD/naif YOK)
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- sabitler
HF_REPO = "thuml/Time-Series-Library"
HF_REV = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"  # Exchange ile ayni HF dataset revizyonu (2025-11-12)
ETD_REPO = "zhouhaoyi/ETDataset"
ETD_REV = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"  # ETDataset main, 2021-06-04 "Update cite"
# HF tree API (rev 2b66e59e) 'oid' ve 'size'; ETDataset GitHub contents API 'sha'/'size' ayni degerleri verir
RESMI = {
    "etth1": {"dosya": "ETTh1.csv", "git_blob_sha1": "a52c4925778c07c1ef1a2cf6fd01594919717d9e", "boyut": 2589657},
    "etth2": {"dosya": "ETTh2.csv", "git_blob_sha1": "6e3760778598fdee5208213d3fce122637307eec", "boyut": 2417960},
}

SEQ_LEN = 96
LABEL_LEN = 48          # run.py varsayilani (iTransformer kullanmaz)
HORIZONS = [96, 192, 336, 720]
BATCH = 32              # run.py varsayilani (train/val; test bs=1 iTransformer data_factory)
TARGET = "OT"
AY = 30 * 24            # Dataset_ETT_hour: 1 "ay" = 720 saat
N_TRAIN, N_VAL, N_TEST = 12 * AY, 4 * AY, 4 * AY
V = N_VAL               # 3-kat protokolu dilim uzunlugu (resmi val uzunlugu)


def log(*a):
    print(*a, flush=True)


def hashes(b: bytes) -> dict:
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "md5": hashlib.md5(b).hexdigest(),
            "git_blob_sha1": hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()}


def download(url: str, dest: Path, force: bool):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest.read_bytes(), False
    req = urllib.request.Request(url, headers={"User-Agent": "veri_hazirla_ett.py/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        b = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(b)
    tmp.replace(dest)
    return b, True


# ---------------------------------------------------------------- MMD (Exchange betigi ile ayni tanim)
def _sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def _ksum(A, B, gamma, chunk=1024):
    s = 0.0
    for i in range(0, len(A), chunk):
        s += np.exp(-gamma * _sqdist(A[i:i + chunk], B)).sum()
    return s


def mmd2(X, Y, sigma):
    """RBF k(x,y)=exp(-||x-y||^2/(2 sigma^2)). Biased (V) ve unbiased (U) MMD^2."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X, Y = X[:, None], Y[:, None]
    g = 1.0 / (2.0 * sigma ** 2)
    m, n = len(X), len(Y)
    kxx, kyy, kxy = _ksum(X, X, g), _ksum(Y, Y, g), _ksum(X, Y, g)
    biased = kxx / m ** 2 + kyy / n ** 2 - 2 * kxy / (m * n)
    unbiased = (kxx - m) / (m * (m - 1)) + (kyy - n) / (n * (n - 1)) - 2 * kxy / (m * n)
    return float(biased), float(unbiased)


def _ust_ucgen_parcalari(P, chunk=384):
    """pdist(P) degerlerini satir-parcalari halinde uretir (cdist = pdist ile ayni Oklid rutini)."""
    n = len(P)
    for i in range(0, n - 1, chunk):
        A = P[i:i + chunk]
        D = cdist(A, P[i:])
        ii = np.arange(len(A))[:, None]
        jj = np.arange(D.shape[1])[None, :]
        yield D[jj > ii]


def median_dist(P, bins=1 << 16):
    """np.median(pdist(P)) ile ayni deger; tam pdist dizisini bellekte tutmadan (iki gecisli kesin secim)."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P[:, None]
    n = len(P)
    if n <= 3000:
        return float(np.median(pdist(P)))
    m = n * (n - 1) // 2
    ranks = [m // 2] if m % 2 else [m // 2 - 1, m // 2]
    ust = 2.0 * float(np.sqrt(((P - P.mean(0)) ** 2).sum(1)).max()) * (1 + 1e-9) + 1e-300  # ucgen esitsizligi
    edges = np.linspace(0.0, ust, bins + 1)
    cnt = np.zeros(bins, dtype=np.int64)
    for v in _ust_ucgen_parcalari(P):
        cnt += np.histogram(v, bins=edges)[0]
    assert cnt.sum() == m
    cum = np.cumsum(cnt)
    hedef_bin = sorted({int(np.searchsorted(cum, r, side="right")) for r in ranks})
    lo, hi = edges[hedef_bin[0]], edges[hedef_bin[-1] + 1]
    oncesi = int(cum[hedef_bin[0] - 1]) if hedef_bin[0] > 0 else 0
    sec = []
    for v in _ust_ucgen_parcalari(P):
        b = np.searchsorted(edges, v, side="right") - 1      # np.histogram ile ayni atama (son kenar dahil)
        b = np.minimum(b, bins - 1)
        sec.append(v[(b >= hedef_bin[0]) & (b <= hedef_bin[-1])])
    sec = np.sort(np.concatenate(sec))
    vals = [sec[r - oncesi] for r in ranks]
    return float(np.mean(vals))


def blok_hash(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).digest()


# ---------------------------------------------------------------- tek veri seti
def hazirla(ad: str, ws: Path, force: bool) -> dict:
    data_dir = ws / "data"
    r = RESMI[ad]
    fn = r["dosya"]
    hf_url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/ETT-small/{fn}"
    etd_url = f"https://raw.githubusercontent.com/{ETD_REPO}/{ETD_REV}/ETT-small/{fn}"
    csv_path = data_dir / ad / fn
    etd_path = data_dir / ad / "kaynak_etdataset" / fn
    M = {}

    # ---- V1/V2
    b, yeni = download(hf_url, csv_path, force)
    h = hashes(b)
    log(f"[{ad}][V1] TSLib CSV:", hf_url, "yeni indirildi" if yeni else "(mevcut dosya)")
    log(f"[{ad}][V2]", h)
    if h["bytes"] != r["boyut"] or h["git_blob_sha1"] != r["git_blob_sha1"]:
        raise SystemExit(f"{ad}: HASH/BOYUT UYUSMAZ: beklenen {r['boyut']} / {r['git_blob_sha1']}")
    M["kaynak_tslib"] = {"url": hf_url, "hf_repo": HF_REPO, "hf_revizyon": HF_REV,
                         "resmi_git_blob_oid": r["git_blob_sha1"], "resmi_boyut": r["boyut"],
                         "olculen": h, "resmi_eslesme": True, "yerel_yol": str(csv_path)}
    be, yeni_e = download(etd_url, etd_path, force)
    he = hashes(be)
    log(f"[{ad}][V1] ETDataset orijinali:", etd_url, "yeni indirildi" if yeni_e else "(mevcut dosya)", he["sha256"])
    if he["git_blob_sha1"] != r["git_blob_sha1"]:
        raise SystemExit(f"{ad}: ETDataset git blob {he['git_blob_sha1']} != {r['git_blob_sha1']}")

    # ---- okuma ve temel denetimler
    df = pd.read_csv(csv_path)
    dfe = pd.read_csv(etd_path)
    cols = list(df.columns)
    feat = cols[1:]                       # Dataset_ETT_hour features=M: df_raw.columns[1:] (yeniden siralama YOK)
    N = len(df)
    dates = pd.to_datetime(df["date"])
    X = df[feat].to_numpy(dtype=np.float64)
    Xe = dfe[list(dfe.columns)[1:]].to_numpy(dtype=np.float64)
    dd = dates.diff().dropna().value_counts()
    vals_str = df[feat].astype(str).agg("|".join, axis=1)
    M["orijinal_karsilastirma"] = {
        "etdataset_url": etd_url, "etdataset_revizyon": ETD_REV, "etdataset_hash": he,
        "bayt_bayt_ayni": bool(b == be), "sha256_ayni": he["sha256"] == h["sha256"],
        "sutunlar_ayni": list(dfe.columns) == cols, "sekil_ayni": list(Xe.shape) == list(X.shape),
        "tarih_sutunu_ayni": bool((dfe["date"].astype(str) == df["date"].astype(str)).all()),
        "max_mutlak_deger_farki": float(np.abs(Xe - X).max()) if Xe.shape == X.shape else None}
    M["dosya"] = {
        "satir": N, "sutun": len(cols), "sutunlar": cols, "tslib_sutun_sirasi": cols,
        "degisken_sayisi": len(feat), "hedef": TARGET,
        "tarih_ilk": str(dates.iloc[0]), "tarih_son": str(dates.iloc[-1]),
        "tarih_monoton_artan": bool(dates.is_monotonic_increasing),
        "ardisik_tarih_farki_dagilimi": {str(k): int(v) for k, v in dd.items()},
        "eksik_deger_toplam": int(df.isna().sum().sum()),
        "tekrar_eden_tarih": int(dates.duplicated().sum()),
        "tekrar_eden_tam_satir_tarih_dahil": int(df.duplicated().sum()),
        "tekrar_eden_deger_vektoru_tarih_haric": int(vals_str.duplicated().sum()),
        "ardisik_ozdes_deger_satiri": int((np.abs(np.diff(X, axis=0)).sum(1) == 0).sum()),
        "sonlu_olmayan_deger": int((~np.isfinite(X)).sum()),
        "kullanilan_satir": N_TRAIN + N_VAL + N_TEST,
        "kullanilmayan_kuyruk_satir": N - (N_TRAIN + N_VAL + N_TEST),
    }
    log(f"[{ad}][dosya]", json.dumps(M["dosya"], ensure_ascii=False, default=str))
    log(f"[{ad}][orijinal]", {k: v for k, v in M["orijinal_karsilastirma"].items() if k != "etdataset_hash"})

    # ---- V3 split (Dataset_ETT_hour ile birebir)
    b1 = [0, N_TRAIN - SEQ_LEN, N_TRAIN + N_VAL - SEQ_LEN]
    b2 = [N_TRAIN, N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST]
    own = {"train": (0, N_TRAIN), "val": (N_TRAIN, N_TRAIN + N_VAL), "test": (N_TRAIN + N_VAL, b2[2])}
    names = ["train", "val", "test"]
    d = lambda i: str(dates.iloc[i])
    split = {
        "veri": fn, "protokol": "iTransformer/TSLib Dataset_ETT_hour (data_provider/data_loader.py)",
        "seed": None, "not": "deterministik sabit sinirlar (12/4/4 ay x 30 gun x 24 saat); rastgelelik yok, seed gerekmez",
        "oran": "train 12*30*24 / val 4*30*24 / test 4*30*24 satir; kalan kuyruk kullanilmaz", "N": N,
        "num_train": N_TRAIN, "num_vali": N_VAL, "num_test": N_TEST,
        "kullanilmayan_kuyruk[o0,o1)": [b2[2], N],
        "kullanilmayan_kuyruk_tarih": [d(b2[2]), d(N - 1)],
        "seq_len": SEQ_LEN, "label_len": LABEL_LEN,
        "border1s": b1, "border2s": b2, "dilim": {}, "pencere": {},
    }
    for k, nm in enumerate(names):
        o0, o1 = own[nm]
        split["dilim"][nm] = {
            "dataset_satir_araligi[b1,b2)": [b1[k], b2[k]], "dataset_satir_sayisi": b2[k] - b1[k],
            "kendi_satirlari[o0,o1)": [o0, o1], "kendi_satir_sayisi": o1 - o0,
            "girdi_ilk_tarih": d(b1[k]), "kendi_ilk_tarih": d(o0), "son_tarih": d(o1 - 1),
            "geri_kaydirma_ortusme_satiri": o0 - b1[k]}
    drop_tab = []
    for H in HORIZONS:
        split["pencere"][str(H)] = {}
        for k, nm in enumerate(names):
            n = (b2[k] - b1[k]) - SEQ_LEN - H + 1
            first_t, last_t = b1[k] + SEQ_LEN, b2[k] - 1
            rem = n % BATCH
            e = {"pencere": n, "ilk_hedef_satiri": first_t, "son_hedef_satiri": last_t,
                 "ilk_hedef_tarihi": d(first_t), "son_hedef_tarihi": d(last_t),
                 "batch32_tam_batch": n // BATCH, "drop_last_True_atilan": rem,
                 "drop_last_True_kalan": n - rem}
            if nm == "test":
                last_kept_t = b1[k] + (n - rem - 1) + SEQ_LEN + H - 1
                e["itransformer_test_bs1_kayip"] = 0
                e["tslib_hatali_donem_bs32_drop_last_hic_degerlendirilmeyen_son_satir"] = last_t - last_kept_t
            split["pencere"][str(H)][nm] = e
            drop_tab.append({"ufuk": H, "dilim": nm, "pencere": n, "atilan_bs32": rem})
    log(f"[{ad}][V3] border1s", b1, "border2s", b2)
    for row in drop_tab:
        log("   ", row)

    # ---- scaler (yalniz train)
    sc = StandardScaler().fit(X[b1[0]:b2[0]])
    Z = sc.transform(X)
    sc_all = StandardScaler().fit(X[:b2[2]])
    tr = X[:N_TRAIN]
    scal = {
        "fit_araligi": [b1[0], b2[0]], "fit_satir": b2[0] - b1[0],
        "mean_train_ile_ayni": bool(np.allclose(sc.mean_, tr.mean(0))),
        "scale_train_std_ddof0_ile_ayni": bool(np.allclose(sc.scale_, tr.std(0, ddof=0))),
        "train_z_ort_max_mutlak": float(np.abs(Z[:N_TRAIN].mean(0)).max()),
        "train_z_std_min_max": [float(Z[:N_TRAIN].std(0).min()), float(Z[:N_TRAIN].std(0).max())],
        "mean": dict(zip(feat, map(float, sc.mean_))), "scale": dict(zip(feat, map(float, sc.scale_))),
        "kullanilan_tum_satirlara_fit_edilseydi_ort_farki_z": dict(zip(feat, map(float, (sc_all.mean_ - sc.mean_) / sc.scale_))),
        "kullanilan_tum_satirlara_fit_edilseydi_scale_orani": dict(zip(feat, map(float, sc_all.scale_ / sc.scale_))),
        "tslib_kodu": "train_data = df_data[border1s[0]:border2s[0]]; self.scaler.fit(train_data.values)",
    }
    split["scaler_train_only"] = {"mean": scal["mean"], "scale": scal["scale"]}
    M["scaler"] = scal

    # ---- 3-kat (rolling-origin) protokolu: K3 = resmi val, K2/K1 = onceki V satirlik dilimler
    kat = {}
    for kn, v0 in (("K1", N_TRAIN - 2 * V), ("K2", N_TRAIN - V), ("K3", N_TRAIN)):
        v1 = v0 + V
        fsc = StandardScaler().fit(X[0:v0])
        e = {"train_kendi_satirlari[o0,o1)": [0, v0], "train_satir": v0,
             "val_kendi_satirlari[o0,o1)": [v0, v1], "val_satir": V,
             "val_dataset_satir_araligi[b1,b2)": [v0 - SEQ_LEN, v1],
             "train_tarih": [d(0), d(v0 - 1)], "val_tarih": [d(v0), d(v1 - 1)],
             "scaler_fit_araligi": [0, v0],
             "scaler_mean": dict(zip(feat, map(float, fsc.mean_))),
             "scaler_scale": dict(zip(feat, map(float, fsc.scale_))),
             "pencere": {}}
        for H in HORIZONS:
            e["pencere"][str(H)] = {"train": v0 - SEQ_LEN - H + 1, "val": (V + SEQ_LEN) - SEQ_LEN - H + 1,
                                    "train_drop_last_bs32_atilan": (v0 - SEQ_LEN - H + 1) % BATCH,
                                    "val_drop_last_bs32_atilan": (V - H + 1) % BATCH}
        if kn == "K3":
            e["resmi_val_ile_ayni"] = bool([v0, v1] == list(own["val"]) and [v0 - SEQ_LEN, v1] == [b1[1], b2[1]])
        kat[kn] = e
    split["kat_protokolu"] = {
        "tanim": "FAZ II P1 sef on-kaydi: resmi train+val icinde uzunlugu V=resmi val olan 3 ardisik dogrulama dilimi; "
                 "her katta train = dilimden onceki tum satirlar; scaler yalniz o katin train'i; L=96; test dokunulmaz",
        "V": V, "katlar": kat}
    log(f"[{ad}][kat]", {k: (v["train_kendi_satirlari[o0,o1)"], v["val_kendi_satirlari[o0,o1)"]) for k, v in kat.items()})

    # ---- V5 sizinti denetimi
    leak = {"hedef_sinirlari": {}}
    ok_all = True
    for H in HORIZONS:
        P = split["pencere"][str(H)]
        ok = (P["train"]["son_hedef_satiri"] < P["val"]["ilk_hedef_satiri"]
              and P["val"]["son_hedef_satiri"] < P["test"]["ilk_hedef_satiri"]
              and dates.iloc[P["train"]["son_hedef_satiri"]] < dates.iloc[P["val"]["ilk_hedef_satiri"]]
              and dates.iloc[P["val"]["son_hedef_satiri"]] < dates.iloc[P["test"]["ilk_hedef_satiri"]])
        ok_all &= ok
        leak["hedef_sinirlari"][str(H)] = {
            "train_son_hedef": P["train"]["son_hedef_tarihi"], "val_ilk_hedef": P["val"]["ilk_hedef_tarihi"],
            "val_son_hedef": P["val"]["son_hedef_tarihi"], "test_ilk_hedef": P["test"]["ilk_hedef_tarihi"],
            "kronolojik_ve_ayrik": bool(ok),
            "hedef_satiri_ortusmesi_train_val": max(0, P["train"]["son_hedef_satiri"] - P["val"]["ilk_hedef_satiri"] + 1),
            "hedef_satiri_ortusmesi_val_test": max(0, P["val"]["son_hedef_satiri"] - P["test"]["ilk_hedef_satiri"] + 1)}
    leak["tum_ufuklarda_hedef_ayrik"] = bool(ok_all)
    leak["girdi_geri_kaydirma_ortusmesi"] = {
        "val_girdisinde_train_satiri": N_TRAIN - b1[1], "test_girdisinde_val_satiri": b2[1] - b1[2],
        "not": "mesru: girdi penceresi gecmise bakar"}
    keys = [row.tobytes() for row in X]
    tr_set = set(keys[0:N_TRAIN])
    va_keys = keys[own["val"][0]:own["val"][1]]
    va_set = set(va_keys)
    te_keys = keys[own["test"][0]:own["test"][1]]
    leak["satir_deger_cakismasi"] = {
        "test_satiri_train_icinde": int(sum(k in tr_set for k in te_keys)),
        "test_satiri_val_icinde": int(sum(k in va_set for k in te_keys)),
        "val_satiri_train_icinde": int(sum(k in tr_set for k in va_keys)),
        "tarih_cakismasi_train_test": int(len(set(dates.iloc[:N_TRAIN]) & set(dates.iloc[own["test"][0]:own["test"][1]]))),
    }
    win_overlap = {}
    for H in HORIZONS:
        def blocks(k):
            n = (b2[k] - b1[k]) - SEQ_LEN - H + 1
            return {blok_hash(X[b1[k] + s + SEQ_LEN: b1[k] + s + SEQ_LEN + H]) for s in range(n)}
        win_overlap[str(H)] = int(len(blocks(0) & blocks(2)))
    leak["hedef_blogu_cakismasi_train_test"] = win_overlap
    leak["hedef_blogu_karsilastirma_yontemi"] = "SHA-1(ham float64 blok baytlari); birebir esitlik"
    leak["scaler_sizintisi"] = "yok: fit yalniz [0, 8640)"
    leak["kullanilmayan_kuyruk_not"] = "satir [14400, N) hicbir dilimde yok; test'ten sonra gelir, sizinti yaratmaz"
    M["sizinti"] = leak
    log(f"[{ad}][V5]", json.dumps({k: leak[k] for k in ("tum_ufuklarda_hedef_ayrik", "satir_deger_cakismasi",
                                                         "hedef_blogu_cakismasi_train_test")}, ensure_ascii=False))

    # ---- dagilim kaymasi (yalniz satir dagilim istatistigi; test'te tahmin/hata YOK)
    te0, te1 = own["test"]
    q_idx = np.array_split(np.arange(te0, te1), 4)
    slices = {"train": (0, N_TRAIN), "val": own["val"], "test": own["test"]}
    for i, q in enumerate(q_idx, 1):
        slices[f"test_Q{i}"] = (int(q[0]), int(q[-1]) + 1)
    split["test_ceyrekleri_kendi_satirlari"] = {
        k: {"satir[o0,o1)": list(v), "ilk_tarih": d(v[0]), "son_tarih": d(v[1] - 1)}
        for k, v in slices.items() if k.startswith("test_Q")}
    Ztr = Z[:N_TRAIN]
    sig_z = median_dist(np.vstack([Ztr, Z[te0:te1]]))
    sig_raw = median_dist(np.vstack([X[:N_TRAIN], X[te0:te1]]))
    sig_1d = [median_dist(np.concatenate([Ztr[:, j], Z[te0:te1, j]])) for j in range(len(feat))]
    rows, mmd_rows = [], []
    for nm, (a, bnd) in slices.items():
        R, Zs = X[a:bnd], Z[a:bnd]
        for j, c in enumerate(feat):
            ks = stats.ks_2samp(Ztr[:, j], Zs[:, j]) if nm != "train" else None
            m1 = mmd2(Ztr[:, j], Zs[:, j], sig_1d[j])[0] if nm != "train" else 0.0
            rows.append({
                "dilim": nm, "satir_bas": a, "satir_bit_haric": bnd, "satir_sayisi": bnd - a,
                "ilk_tarih": d(a), "son_tarih": d(bnd - 1), "degisken": c,
                "ham_ort": R[:, j].mean(), "ham_std": R[:, j].std(), "ham_min": R[:, j].min(), "ham_max": R[:, j].max(),
                "z_ort": Zs[:, j].mean(), "z_std": Zs[:, j].std(), "z_min": Zs[:, j].min(), "z_max": Zs[:, j].max(),
                "kayma_ort_farki_z": Zs[:, j].mean() - Ztr[:, j].mean(),
                "kayma_std_orani": Zs[:, j].std() / Ztr[:, j].std(),
                "z_train_araligi_disi_oran": float(((Zs[:, j] < Ztr[:, j].min()) | (Zs[:, j] > Ztr[:, j].max())).mean()),
                "ks_D": ks.statistic if ks else 0.0, "mmd2_1d_rbf_medyan": m1})
        if nm == "train":
            continue
        bz, uz = mmd2(Ztr, Zs, sig_z)
        b1z, _ = mmd2(Ztr, Zs, 1.0)
        br, _ = mmd2(X[:N_TRAIN], R, sig_raw)
        sig_own = median_dist(np.vstack([Ztr, Zs]))
        bo, _ = mmd2(Ztr, Zs, sig_own)
        sub = [rr for rr in rows if rr["dilim"] == nm]
        mr = {"dilim": nm, "referans": "resmi train [0,8640), resmi scaler", "satir_sayisi": bnd - a,
              "ilk_tarih": d(a), "son_tarih": d(bnd - 1),
              "ort_mutlak_ort_farki_z": float(np.mean([abs(rr["kayma_ort_farki_z"]) for rr in sub])),
              "ort_std_orani": float(np.mean([rr["kayma_std_orani"] for rr in sub])),
              "ort_ks_D": float(np.mean([rr["ks_D"] for rr in sub])),
              "ort_mmd2_1d": float(np.mean([rr["mmd2_1d_rbf_medyan"] for rr in sub])),
              "sigma_z_medyan_train+test": sig_z,
              "mmd2_b_z_sigmaMed": bz, "mmd_b_z_sigmaMed": float(np.sqrt(max(bz, 0))), "mmd2_u_z_sigmaMed": uz,
              "mmd2_b_z_sigma1": b1z, "mmd_b_z_sigma1": float(np.sqrt(max(b1z, 0))),
              "sigma_z_medyan_train+dilim": sig_own,
              "mmd2_b_z_sigmaMedDilim": bo, "mmd_b_z_sigmaMedDilim": float(np.sqrt(max(bo, 0))),
              "sigma_ham_medyan_train+test": sig_raw,
              "mmd2_b_ham_sigmaMed": br, "mmd_b_ham_sigmaMed": float(np.sqrt(max(br, 0)))}
        mmd_rows.append(mr)
        rows.append({"dilim": nm, "satir_bas": a, "satir_bit_haric": bnd, "satir_sayisi": bnd - a,
                     "ilk_tarih": d(a), "son_tarih": d(bnd - 1), "degisken": f"TUM({len(feat)})",
                     "kayma_ort_farki_z": mr["ort_mutlak_ort_farki_z"], "kayma_std_orani": mr["ort_std_orani"],
                     "ks_D": mr["ort_ks_D"], "mmd2_1d_rbf_medyan": mr["ort_mmd2_1d"],
                     "mmd2_7d_z_rbf_medyan": bz, "mmd_7d_z_rbf_medyan": mr["mmd_b_z_sigmaMed"]})
        log(f"[{ad}][kayma] {nm:8s} |dmu|={mr['ort_mutlak_ort_farki_z']:.3f} std_or={mr['ort_std_orani']:.3f} "
            f"KS={mr['ort_ks_D']:.3f} MMD2(z,med)={bz:.4f} MMD(z,med)={mr['mmd_b_z_sigmaMed']:.4f} "
            f"MMD(z,s1)={mr['mmd_b_z_sigma1']:.4f} MMD(ham)={mr['mmd_b_ham_sigmaMed']:.4f}")
    # kat bazinda kayma: kat-train <-> kat-val, kat scaler'i (yalniz train+val satirlari)
    for kn, e in kat.items():
        v0, v1 = e["val_kendi_satirlari[o0,o1)"]
        fsc = StandardScaler().fit(X[0:v0])
        Zf = fsc.transform(X[:v1])
        A, B = Zf[:v0], Zf[v0:v1]
        sg = median_dist(np.vstack([A, B]))
        bb, uu = mmd2(A, B, sg)
        b1b, _ = mmd2(A, B, 1.0)
        mr = {"dilim": f"kat_{kn}_val", "referans": f"kat train [0,{v0}), kat scaler", "satir_sayisi": V,
              "ilk_tarih": d(v0), "son_tarih": d(v1 - 1),
              "ort_mutlak_ort_farki_z": float(np.mean(np.abs(B.mean(0) - A.mean(0)))),
              "ort_std_orani": float(np.mean(B.std(0) / A.std(0))),
              "ort_ks_D": float(np.mean([stats.ks_2samp(A[:, j], B[:, j]).statistic for j in range(len(feat))])),
              "sigma_z_medyan_train+dilim": sg, "mmd2_b_z_sigmaMedDilim": bb,
              "mmd_b_z_sigmaMedDilim": float(np.sqrt(max(bb, 0))), "mmd2_u_z_sigmaMedDilim": uu,
              "mmd2_b_z_sigma1": b1b, "mmd_b_z_sigma1": float(np.sqrt(max(b1b, 0)))}
        mmd_rows.append(mr)
        log(f"[{ad}][kayma-kat] {kn} |dmu|={mr['ort_mutlak_ort_farki_z']:.3f} std_or={mr['ort_std_orani']:.3f} "
            f"KS={mr['ort_ks_D']:.3f} MMD(z,medDilim)={mr['mmd_b_z_sigmaMedDilim']:.4f} MMD(z,s1)={mr['mmd_b_z_sigma1']:.4f}")
    pd.DataFrame(rows).to_csv(data_dir / f"kayma_istatistikleri_{ad}.csv", index=False, float_format="%.6g")
    pd.DataFrame(mmd_rows).to_csv(data_dir / f"kayma_mmd_{ad}.csv", index=False, float_format="%.6g")
    M["kayma"] = {"mmd_tanimlari": {
        "cekirdek": "RBF exp(-||x-y||^2/(2 sigma^2)); X=train kendi satirlari, Y=dilim satirlari (pencere degil, satir)",
        "sigmaMed": "train+test birlesik ornekte ciftler arasi Oklid uzakligi medyani (z-uzayi), tum dilimlerde sabit",
        "sigmaMedDilim": "train+ilgili dilim birlesik medyani", "sigma1": "sigma=1 (z-uzayi)",
        "kat_satirlari": "kat-train <-> kat-val, kat scaler'i; sigmaMedDilim = kat-train+kat-val medyani",
        "b/u": "biased V / unbiased U; mmd_* = sqrt(mmd2_b)"}, "ozet": mmd_rows}

    # ---- naive: YALNIZ val dilimleri (K1, K2, K3=resmi val); her katta o katin scaler'i
    nv = []
    for kn, e in kat.items():
        v0, v1 = e["val_kendi_satirlari[o0,o1)"]
        fsc = StandardScaler().fit(X[0:v0])
        Zf = fsc.transform(X[:v1])
        vb1 = v0 - SEQ_LEN
        for H in HORIZONS:
            n = (v1 - vb1) - SEQ_LEN - H + 1
            s = np.arange(n)
            last_in = vb1 + s + SEQ_LEN - 1
            tgt = last_in[:, None] + 1 + np.arange(H)[None, :]
            assert tgt.max() == v1 - 1 and tgt.min() == v0 and tgt.max() < own["test"][0]
            Y = Zf[tgt]
            P_last = Zf[last_in][:, None, :]
            for yon, P in (("repeat_last", P_last), ("sifir_train_ort", np.zeros_like(P_last))):
                err = Y - P
                nv.append({"kat": kn, "dilim": "val" if kn == "K3" else f"kat_{kn}_val", "ufuk": H,
                           "seq_len": SEQ_LEN, "val_satir[o0,o1)": f"[{v0},{v1})", "pencere": n,
                           "drop_last_True_bs32_atilan": n % BATCH, "yontem": yon,
                           "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                           "mse_OT": float((err[..., -1] ** 2).mean()), "mae_OT": float(np.abs(err[..., -1]).mean())})
    nvdf = pd.DataFrame(nv)
    nvdf.to_csv(data_dir / f"naive_val_{ad}.csv", index=False, float_format="%.6f")
    log(f"[{ad}][naive val]\n" + nvdf[nvdf.yontem == "repeat_last"].to_string(index=False))
    M["naive_val"] = nv
    M["naive_val_ort_repeat_last"] = {
        kn: {"mse": float(g.mse.mean()), "mae": float(g.mae.mean())}
        for kn, g in nvdf[nvdf.yontem == "repeat_last"].groupby("kat")}

    split_path = data_dir / "splits" / f"{ad}_sl96.json"
    split_path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    M["split_dosyasi"] = str(split_path)
    M["split"] = split
    M["drop_last_tablosu"] = drop_tab
    log(f"[{ad}][yazildi]", split_path)
    return M


import os  # FAZ III

# ====================================================================================================== FAZ III
# H16 on-kayitli veri setleri. Kaynaklar (hepsi programatik; resmi git blob SHA-1 / boyut HF tree API ve GitHub
# contents API'den, 2026-10-15):
#   * ETTm1/ETTm2/Weather: HF thuml/Time-Series-Library @ HF_REV (ETTh ile ayni revizyon) = BIRINCIL.
#     Karsilastirma: ETDataset @ ETD_REV (ETTm) ve iTransformer resmi veri arsivi (README Google Drive linki).
#   * Solar-Energy: TSLib HF deposunda YOK. BIRINCIL = iTransformer resmi veri arsivindeki solar_AL.txt
#     (iTransformer'in kullandigi surum). Karsilastirma + resmi hash zinciri: LSTNet deposu
#     laiguokun/multivariate-time-series-data @ LSTNET_REV, solar-energy/solar_AL.txt.gz (gzip acilip bayt kiyasi).
ITR_ZIP_URL = ("https://drive.usercontent.google.com/download?id=1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot"
               "&export=download&confirm=t")   # rakip/iTransformer/scripts/multivariate_forecasting/README.md
ITR_ZIP_DOSYA = "iTransformer_datasets.zip"
LSTNET_REPO = "laiguokun/multivariate-time-series-data"
LSTNET_REV = "7f402f185cc2435b5e66aed13a3b560ed142e023"  # master ucu, 2017-04-09
_REPO_ITR = Path(__file__).resolve().parents[1] / "rakip" / "iTransformer"
FAZ3 = {
    "ettm1": {"dosya": "ETTm1.csv", "kaynak": "hf", "hf_yol": "ETT-small/ETTm1.csv", "etd": True,
              "git_blob_sha1": "62c9f979e2dc5b05d9cc5b38891e60e6baf77417", "boyut": 10360719,
              "bolme": "ett_minute", "data_anahtari": "ETTm1", "sinif": "Dataset_ETT_minute",
              "sh": "scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh", "zip_son_ek": "ETTm1.csv"},
    "ettm2": {"dosya": "ETTm2.csv", "kaynak": "hf", "hf_yol": "ETT-small/ETTm2.csv", "etd": True,
              "git_blob_sha1": "f9d46027c987d442d2d8f81308d1d8c4d1d35db8", "boyut": 9677236,
              "bolme": "ett_minute", "data_anahtari": "ETTm2", "sinif": "Dataset_ETT_minute",
              "sh": "scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh", "zip_son_ek": "ETTm2.csv"},
    "weather": {"dosya": "weather.csv", "kaynak": "hf", "hf_yol": "weather/weather.csv", "etd": False,
                "git_blob_sha1": "490cacf73d5129165cbdabe74f99c39763ae8ebf", "boyut": 7235425,
                "bolme": "custom", "data_anahtari": "custom", "sinif": "Dataset_Custom",
                "sh": "scripts/multivariate_forecasting/Weather/iTransformer.sh", "zip_son_ek": "weather.csv"},
    "solar": {"dosya": "solar_AL.txt", "kaynak": "itr_zip", "etd": False,
              "lstnet_yol": "solar-energy/solar_AL.txt.gz",
              "lstnet_git_blob_sha1": "08a1f235546b3708309ff4fb544a54bb82e4f5e7", "lstnet_boyut": 8748993,
              "bolme": "solar", "data_anahtari": "Solar", "sinif": "Dataset_Solar",
              "sh": "scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh", "zip_son_ek": "solar_AL.txt"},
}


def faz3_sinirlar(bolme, N):
    """data_provider/data_loader.py ile birebir (seq_len=96). Donus: border1s, border2s, aciklama."""
    L = SEQ_LEN
    if bolme == "ett_minute":   # Dataset_ETT_minute: 12/4/4 ay x 30 gun x 24 saat x 4 (15 dk)
        T, V, E = 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
        b1 = [0, T - L, T + V - L]
        b2 = [T, T + V, T + V + E]
        acik = "Dataset_ETT_minute: border2s=[12*30*24*4, +4*30*24*4, +8*30*24*4]; kuyruk kullanilmaz"
    elif bolme == "custom":     # Dataset_Custom: num_train=int(.7N), num_test=int(.2N), num_vali=N-train-test
        T, E = int(N * 0.7), int(N * 0.2)
        V = N - T - E
        b1 = [0, T - L, N - E - L]
        b2 = [T, T + V, N]
        acik = "Dataset_Custom: num_train=int(N*0.7), num_test=int(N*0.2), num_vali=N-num_train-num_test"
    elif bolme == "solar":      # Dataset_Solar: num_valid=int(.1N) AYRI hesaplanir; border2s=[T, T+Vv, N]
        T, E, V = int(N * 0.7), int(N * 0.2), int(N * 0.1)
        b1 = [0, T - L, N - E - L]
        b2 = [T, T + V, N]
        acik = ("Dataset_Solar: num_train=int(N*0.7), num_test=int(N*0.2), num_valid=int(N*0.1); "
                "border2s=[num_train, num_train+num_valid, N]")
    else:
        raise ValueError(bolme)
    return b1, b2, acik


def oku_resmi_bicim(ad, yol):
    """Resmi __read_data__ ile ayni okuma; donus (tarih Series|None, X float64 (N,C), sutun adlari)."""
    if ad == "solar":   # Dataset_Solar: satir satir float(), tarih sutunu yok
        rows = []
        with open(yol, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip("\n").split(",")
                rows.append(np.stack([float(i) for i in line]))
        X = np.stack(rows, 0)
        return None, X, [str(j) for j in range(X.shape[1])]
    df = pd.read_csv(yol)
    if FAZ3[ad]["bolme"] == "custom":   # Dataset_Custom sutun duzeni: date, digerleri, target(OT)
        cols = list(df.columns)
        cols.remove(TARGET)
        cols.remove("date")
        df = df[["date"] + cols + [TARGET]]
    feat = list(df.columns[1:])
    return pd.to_datetime(df["date"]), df[feat].to_numpy(dtype=np.float64), feat


def itr_zip_uyeleri(ws, force):
    """iTransformer resmi veri arsivi -> {son_ek: (uye_adi, bayt)} (yalniz FAZ III dosyalari, bellekte okunur)."""
    import zipfile
    zp = ws / "data" / "kaynak_itransformer_zip" / ITR_ZIP_DOSYA
    b, yeni = download(ITR_ZIP_URL, zp, force)
    h = hashes(b)
    log("[itr_zip][V1]", ITR_ZIP_URL, "yeni indirildi" if yeni else "(mevcut dosya)", h["sha256"], h["bytes"])
    out, liste = {}, []
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            liste.append([info.filename, info.file_size])
            for ad, r in FAZ3.items():
                if info.filename.replace("\\", "/").split("/")[-1] == r["zip_son_ek"]:
                    if r["zip_son_ek"] in out:
                        raise SystemExit(f"arsivde birden cok {r['zip_son_ek']}")
                    out[r["zip_son_ek"]] = (info.filename, z.read(info))
    meta = {"url": ITR_ZIP_URL, "yerel_yol": str(zp), "hash": h, "uye_sayisi": len(liste), "uyeler": liste,
            "resmi_hash": "yayimlanmamis (README yalniz link veriyor); olculen sha256 kayitli"}
    return out, meta


def resmi_dataset(ad, ws, flag, H):
    """rakip/iTransformer resmi Dataset sinifi (rakip_kos.py/run.py ile ayni argumanlar: features=M, target=OT,
    timeenc=1 (embed=timeF), freq=h (run.py varsayilani; betikler --freq vermiyor))."""
    if str(_REPO_ITR) not in sys.path:
        sys.path.insert(0, str(_REPO_ITR))
    from data_provider import data_loader as dl
    r = FAZ3[ad]
    Cls = getattr(dl, r["sinif"])
    return Cls(root_path=str(ws / "data" / ad) + os.sep, data_path=r["dosya"], flag=flag,
               size=[SEQ_LEN, LABEL_LEN, H], features="M", target=TARGET, timeenc=1, freq="h")


def hedef_yaz(yol: Path, data_y, n, H):
    """rakip_kos.hedef_dogrula_yaz ile ayni icerik/bicim: np.save(float32 (n,H,C)); bellek dostu (memmap, parca).
    Dosya varsa yeniden yazmaz, birebir esitligi dogrular."""
    from numpy.lib.stride_tricks import sliding_window_view
    C = data_y.shape[1]
    sw = sliding_window_view(data_y[SEQ_LEN:], H, axis=0)        # (len-96-H+1, C, H)
    assert sw.shape[0] == n, (sw.shape, n)
    parca = max(1, int(2e8 // (H * C * 4)))                     # ~200 MB parcalar
    if yol.exists():
        ref = np.load(yol, mmap_mode="r")
        if ref.shape != (n, H, C) or ref.dtype != np.float32:
            raise SystemExit(f"{yol}: mevcut dosya sekli/tipi farkli {ref.shape} {ref.dtype}")
        for i in range(0, n, parca):
            if not np.array_equal(ref[i:i + parca], sw[i:i + parca].transpose(0, 2, 1).astype(np.float32)):
                raise SystemExit(f"VERI HATTI DEGISTI: {yol}")
        del ref
        durum = "mevcut, birebir dogrulandi"
    else:
        tmp = yol.with_name(yol.name + ".part.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, H, C))
        for i in range(0, n, parca):
            mm[i:i + parca] = sw[i:i + parca].transpose(0, 2, 1).astype(np.float32)
        mm.flush()
        del mm
        os.replace(tmp, yol)
        durum = "yazildi"
    hs = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 24), b""):
            hs.update(blok)
    return {"dosya": str(yol), "sekil": [n, H, C], "bayt": yol.stat().st_size, "sha256": hs.hexdigest(),
            "durum": durum}


def hazirla_faz3(ad: str, ws: Path, force: bool, zip_uyeleri: dict, hedef: bool) -> dict:
    data_dir = ws / "data"
    r = FAZ3[ad]
    fn = r["dosya"]
    yol = data_dir / ad / fn
    M = {"veri": ad}

    # ---------------- V1 / V2
    if r["kaynak"] == "hf":
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/{r['hf_yol']}"
        b, yeni = download(url, yol, force)
        h = hashes(b)
        log(f"[{ad}][V1]", url, "yeni indirildi" if yeni else "(mevcut dosya)")
        log(f"[{ad}][V2]", h)
        if h["bytes"] != r["boyut"] or h["git_blob_sha1"] != r["git_blob_sha1"]:
            raise SystemExit(f"{ad}: HASH/BOYUT UYUSMAZ: beklenen {r['boyut']} / {r['git_blob_sha1']}")
        M["kaynak"] = {"birincil": "HF thuml/Time-Series-Library", "url": url, "hf_revizyon": HF_REV,
                       "resmi_git_blob_oid": r["git_blob_sha1"], "resmi_boyut": r["boyut"], "olculen": h,
                       "resmi_eslesme": True, "yerel_yol": str(yol)}
        if r["etd"]:
            etd_url = f"https://raw.githubusercontent.com/{ETD_REPO}/{ETD_REV}/ETT-small/{fn}"
            be, yeni_e = download(etd_url, data_dir / ad / "kaynak_etdataset" / fn, force)
            he = hashes(be)
            log(f"[{ad}][V1] ETDataset:", etd_url, "yeni indirildi" if yeni_e else "(mevcut dosya)", he["sha256"])
            M["kaynak"]["etdataset"] = {"url": etd_url, "revizyon": ETD_REV, "hash": he,
                                        "bayt_bayt_ayni": bool(be == b)}
    else:  # solar: iTransformer resmi arsivi birincil
        uye, b = zip_uyeleri[r["zip_son_ek"]]
        yol.parent.mkdir(parents=True, exist_ok=True)
        if not yol.exists() or force:
            tmp = yol.with_suffix(yol.suffix + ".part")
            tmp.write_bytes(b)
            tmp.replace(yol)
            log(f"[{ad}][V1] arsiv uyesi yazildi:", uye, "->", yol)
        else:
            if yol.read_bytes() != b:
                raise SystemExit(f"{ad}: mevcut {yol} arsiv uyesiyle ayni degil")
            log(f"[{ad}][V1] (mevcut dosya, arsiv uyesiyle bayt bayt ayni)")
        h = hashes(b)
        log(f"[{ad}][V2]", h)
        import gzip
        lurl = f"https://raw.githubusercontent.com/{LSTNET_REPO}/{LSTNET_REV}/{r['lstnet_yol']}"
        bg, yeni_g = download(lurl, data_dir / ad / "kaynak_lstnet" / Path(r["lstnet_yol"]).name, force)
        hg = hashes(bg)
        log(f"[{ad}][V1] LSTNet:", lurl, "yeni indirildi" if yeni_g else "(mevcut dosya)", hg)
        if hg["bytes"] != r["lstnet_boyut"] or hg["git_blob_sha1"] != r["lstnet_git_blob_sha1"]:
            raise SystemExit(f"{ad}: LSTNet gz HASH/BOYUT UYUSMAZ")
        bu = gzip.decompress(bg)
        hu = hashes(bu)
        M["kaynak"] = {"birincil": "iTransformer resmi veri arsivi (Google Drive, README linki)",
                       "url": ITR_ZIP_URL, "arsiv_uyesi": uye, "olculen": h, "yerel_yol": str(yol),
                       "lstnet": {"url": lurl, "revizyon": LSTNET_REV, "gz_resmi_git_blob_oid": r["lstnet_git_blob_sha1"],
                                  "gz_resmi_boyut": r["lstnet_boyut"], "gz_olculen": hg, "gz_resmi_eslesme": True,
                                  "acilmis_hash": hu, "itr_arsiv_ile_bayt_bayt_ayni": bool(bu == b)}}
    # iTransformer arsiviyle kiyas (hf kaynaklilar)
    if r["zip_son_ek"] in zip_uyeleri and r["kaynak"] == "hf":
        uye, bz = zip_uyeleri[r["zip_son_ek"]]
        M["kaynak"]["itransformer_arsivi"] = {"uye": uye, "hash": hashes(bz), "bayt_bayt_ayni": bool(bz == b)}
        if bz != b:   # bayt farki varsa deger farkini olc
            import io as _io
            dz = pd.read_csv(_io.BytesIO(bz))
            dh = pd.read_csv(yol)
            M["kaynak"]["itransformer_arsivi"].update({
                "sekil_ayni": list(dz.shape) == list(dh.shape), "sutunlar_ayni": list(dz.columns) == list(dh.columns),
                "max_mutlak_deger_farki": (float(np.abs(dz.iloc[:, 1:].to_numpy(float) - dh.iloc[:, 1:].to_numpy(float)).max())
                                           if dz.shape == dh.shape else None)})
    log(f"[{ad}][kiyas]", json.dumps({k: (v.get("bayt_bayt_ayni") if isinstance(v, dict) else None)
                                       for k, v in M["kaynak"].items() if isinstance(v, dict)}, default=str))

    # ---------------- okuma (resmi bicim) ve temel denetimler
    dates, X, feat = oku_resmi_bicim(ad, yol)
    N, C = X.shape
    keys = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * C))).ravel()
    _, row_id = np.unique(keys, return_inverse=True)
    row_id = row_id.ravel()
    dosya = {"satir": N, "degisken_sayisi": C, "sutunlar_resmi_sira": feat,
             "eksik_deger_toplam": int(np.isnan(X).sum()), "sonlu_olmayan_deger": int((~np.isfinite(X)).sum()),
             "tekrar_eden_deger_vektoru": int(N - len(np.unique(row_id))),
             "ardisik_ozdes_deger_satiri": int((np.diff(row_id) == 0).sum()),
             "sabit_sutun_sayisi": int((X.std(0) == 0).sum())}
    if dates is not None:
        dd = dates.diff().dropna().value_counts()
        dosya.update({"tarih_ilk": str(dates.iloc[0]), "tarih_son": str(dates.iloc[-1]),
                      "tarih_monoton_artan": bool(dates.is_monotonic_increasing),
                      "tekrar_eden_tarih": int(dates.duplicated().sum()),
                      "ardisik_tarih_farki_dagilimi": {str(k): int(v) for k, v in dd.items()}})
    else:
        dosya.update({"tarih": "dosyada tarih sutunu yok (LSTNet: 2006, Alabama, 10 dk; 365*144=52560)",
                      "satir_365x144_ile_tutarli": bool(N == 365 * 144)})
    if ad == "weather":
        dosya["deger_-9999_sayisi_sutun"] = {c: int((X[:, j] == -9999).sum()) for j, c in enumerate(feat)
                                             if (X[:, j] == -9999).any()}
    if ad == "solar":
        dosya["sifir_deger_orani"] = float((X == 0).mean())
        dosya["tum_degiskenler_sifir_satir"] = int((X == 0).all(1).sum())
        dosya["min_max"] = [float(X.min()), float(X.max())]
    M["dosya"] = dosya
    log(f"[{ad}][dosya]", json.dumps({k: v for k, v in dosya.items() if k != "sutunlar_resmi_sira"},
                                     ensure_ascii=False, default=str)[:1500])

    # ---------------- V3 split
    b1, b2, acik = faz3_sinirlar(r["bolme"], N)
    names = ["train", "val", "test"]
    own = {"train": (0, b2[0]), "val": (b2[0], b2[1]), "test": (b2[1], b2[2])}
    assert b1[1] + SEQ_LEN == b2[0] and b1[2] + SEQ_LEN == b2[1], "dilimler ardisik degil"
    d = (lambda i: str(dates.iloc[i])) if dates is not None else (lambda i: f"satir {i}")
    split = {"veri": fn, "protokol": f"iTransformer {r['sinif']} (rakip/iTransformer/data_provider/data_loader.py)",
             "seed": None, "not": "deterministik sabit sinirlar; rastgelelik yok", "oran_kodu": acik, "N": N,
             "num_train": b2[0], "num_vali": b2[1] - b2[0], "num_test": b2[2] - b2[1],
             "oran_gercek": [round(b2[0] / N, 6), round((b2[1] - b2[0]) / N, 6), round((b2[2] - b2[1]) / N, 6)],
             "kullanilmayan_kuyruk[o0,o1)": [b2[2], N], "seq_len": SEQ_LEN, "label_len": LABEL_LEN,
             "border1s": b1, "border2s": b2, "dilim": {}, "pencere": {}}
    for k, nm in enumerate(names):
        o0, o1 = own[nm]
        split["dilim"][nm] = {"dataset_satir_araligi[b1,b2)": [b1[k], b2[k]], "dataset_satir_sayisi": b2[k] - b1[k],
                              "kendi_satirlari[o0,o1)": [o0, o1], "kendi_satir_sayisi": o1 - o0,
                              "girdi_ilk_tarih": d(b1[k]), "kendi_ilk_tarih": d(o0), "son_tarih": d(o1 - 1)}
    for H in HORIZONS:
        split["pencere"][str(H)] = {}
        for k, nm in enumerate(names):
            n = (b2[k] - b1[k]) - SEQ_LEN - H + 1
            e = {"pencere": n, "ilk_hedef_satiri": b1[k] + SEQ_LEN, "son_hedef_satiri": b2[k] - 1,
                 "batch32_tam_batch": n // BATCH, "drop_last_True_atilan": n % BATCH}
            if nm == "test":
                e["itransformer_test_bs1_kayip"] = 0
            split["pencere"][str(H)][nm] = e
    log(f"[{ad}][V3] border1s", b1, "border2s", b2, "| pencere H96:",
        [split["pencere"]["96"][nm]["pencere"] for nm in names])

    # ---------------- scaler (yalniz train) + resmi Dataset ile capraz dogrulama
    sc = StandardScaler().fit(X[b1[0]:b2[0]])
    Z = sc.transform(X)
    split["scaler_train_only"] = {"mean": dict(zip(feat, map(float, sc.mean_))),
                                  "scale": dict(zip(feat, map(float, sc.scale_)))}
    capraz = {}
    resmi_y = {}
    for k, nm in enumerate(names):
        for H in HORIZONS:
            ds = resmi_dataset(ad, ws, nm, H)
            n_bek = split["pencere"][str(H)][nm]["pencere"]
            if len(ds) != n_bek:
                raise SystemExit(f"{ad} {nm} H{H}: resmi len {len(ds)} != {n_bek}")
            if H == HORIZONS[0]:
                fark = float(np.abs(np.asarray(ds.data_y, dtype=np.float64) - Z[b1[k]:b2[k]]).max())
                capraz[nm] = {"resmi_data_y_sekil": list(ds.data_y.shape), "bizim_Z_ile_max_mutlak_fark": fark,
                              "resmi_scaler_mean_ayni": bool(np.array_equal(ds.scaler.mean_, sc.mean_)),
                              "resmi_scaler_scale_ayni": bool(np.array_equal(ds.scaler.scale_, sc.scale_))}
                if fark > 1e-12:
                    raise SystemExit(f"{ad} {nm}: resmi data_y ile fark {fark}")
                if nm != "train":
                    resmi_y[nm] = np.asarray(ds.data_y)
                # __getitem__ ornegi: son pencerenin hedefi
                i = len(ds) - 1
                _, sy, _, _ = ds[i]
                assert np.array_equal(np.asarray(sy)[-H:], np.asarray(ds.data_y)[i + SEQ_LEN:i + SEQ_LEN + H])
            capraz.setdefault("len_dogrulandi", []).append([nm, H, len(ds)])
            del ds
    M["resmi_dataset_capraz_dogrulama"] = capraz
    log(f"[{ad}][capraz]", {k: v for k, v in capraz.items() if k != "len_dogrulandi"})

    # ---------------- V5 sizinti
    leak = {"hedef_sinirlari": {}}
    ok_all = True
    for H in HORIZONS:
        P = split["pencere"][str(H)]
        ok = (P["train"]["son_hedef_satiri"] < P["val"]["ilk_hedef_satiri"]
              and P["val"]["son_hedef_satiri"] < P["test"]["ilk_hedef_satiri"])
        if dates is not None:
            ok = ok and (dates.iloc[P["train"]["son_hedef_satiri"]] < dates.iloc[P["val"]["ilk_hedef_satiri"]]
                         and dates.iloc[P["val"]["son_hedef_satiri"]] < dates.iloc[P["test"]["ilk_hedef_satiri"]])
        ok_all &= bool(ok)
        leak["hedef_sinirlari"][str(H)] = {
            "train_son_hedef": d(P["train"]["son_hedef_satiri"]), "val_ilk_hedef": d(P["val"]["ilk_hedef_satiri"]),
            "val_son_hedef": d(P["val"]["son_hedef_satiri"]), "test_ilk_hedef": d(P["test"]["ilk_hedef_satiri"]),
            "kronolojik_ve_ayrik": bool(ok)}
    leak["tum_ufuklarda_hedef_ayrik"] = bool(ok_all)
    leak["girdi_geri_kaydirma_ortusmesi"] = {"val_girdisinde_train_satiri": b2[0] - b1[1],
                                             "test_girdisinde_val_satiri": b2[1] - b1[2], "not": "mesru"}
    tr_ids, va_ids, te_ids = (row_id[own[n][0]:own[n][1]] for n in names)
    tr_set, va_set = set(tr_ids.tolist()), set(va_ids.tolist())
    sifir_id = None
    if ad == "solar" and (X == 0).all(1).any():
        sifir_id = int(row_id[np.flatnonzero((X == 0).all(1))[0]])
    sc_ = {"test_satiri_train_icinde": int(np.isin(te_ids, tr_ids).sum()),
           "test_satiri_val_icinde": int(np.isin(te_ids, va_ids).sum()),
           "val_satiri_train_icinde": int(np.isin(va_ids, tr_ids).sum())}
    if sifir_id is not None:
        sc_["tum_sifir_satir_haric"] = {
            "test_satiri_train_icinde": int((np.isin(te_ids, tr_ids) & (te_ids != sifir_id)).sum()),
            "test_satiri_val_icinde": int((np.isin(te_ids, va_ids) & (te_ids != sifir_id)).sum()),
            "val_satiri_train_icinde": int((np.isin(va_ids, tr_ids) & (va_ids != sifir_id)).sum()),
            "tum_sifir_satir_sayisi_train_val_test": [int((tr_ids == sifir_id).sum()), int((va_ids == sifir_id).sum()),
                                                      int((te_ids == sifir_id).sum())]}
    if dates is not None:
        sc_["tarih_cakismasi_train_test"] = int(len(set(dates.iloc[own["train"][0]:own["train"][1]])
                                                    & set(dates.iloc[own["test"][0]:own["test"][1]])))
    leak["satir_deger_cakismasi"] = sc_
    win = {}
    rid = row_id.astype(np.int64)
    for H in HORIZONS:
        def blocks(k):
            n = (b2[k] - b1[k]) - SEQ_LEN - H + 1
            base = b1[k] + SEQ_LEN
            return {hashlib.sha1(rid[base + s: base + s + H].tobytes()).digest() for s in range(n)}
        btr = blocks(0)
        win[str(H)] = {"train_test": int(len(btr & blocks(2))), "train_val": int(len(btr & blocks(1)))}
    leak["hedef_blogu_cakismasi"] = win
    leak["hedef_blogu_karsilastirma_yontemi"] = ("satir degerleri birebir esitlige gore tamsayi kimligine cevrilir "
                                                 "(np.unique, float64 baytlari); blok = kimlik dizisinin SHA-1'i")
    leak["scaler_sizintisi"] = f"yok: fit yalniz [0, {b2[0]})"
    M["sizinti"] = leak
    log(f"[{ad}][V5]", json.dumps({k: leak[k] for k in ("tum_ufuklarda_hedef_ayrik", "satir_deger_cakismasi",
                                                         "hedef_blogu_cakismasi")}, ensure_ascii=False))

    # ---------------- betimsel dagilim (test'te yalniz istatistik)
    rows = []
    Ztr = Z[own["train"][0]:own["train"][1]]
    ozet = {}
    for nm in names:
        a, bnd = own[nm]
        R_, Zs = X[a:bnd], Z[a:bnd]
        ks_list, dmu, sor = [], [], []
        for j, c in enumerate(feat):
            ks = stats.ks_2samp(Ztr[:, j], Zs[:, j]).statistic if nm != "train" else 0.0
            sd_tr = Ztr[:, j].std()
            e = {"dilim": nm, "satir_bas": a, "satir_bit_haric": bnd, "satir_sayisi": bnd - a, "degisken": c,
                 "ham_ort": R_[:, j].mean(), "ham_std": R_[:, j].std(), "ham_min": R_[:, j].min(),
                 "ham_max": R_[:, j].max(), "z_ort": Zs[:, j].mean(), "z_std": Zs[:, j].std(),
                 "kayma_ort_farki_z": Zs[:, j].mean() - Ztr[:, j].mean(),
                 "kayma_std_orani": (Zs[:, j].std() / sd_tr) if sd_tr > 0 else np.nan, "ks_D": ks}
            if ad == "solar":
                e["sifir_orani"] = float((R_[:, j] == 0).mean())
            rows.append(e)
            ks_list.append(ks); dmu.append(abs(e["kayma_ort_farki_z"])); sor.append(e["kayma_std_orani"])
        ozet[nm] = {"satir": bnd - a, "ilk": d(a), "son": d(bnd - 1),
                    "ort_mutlak_ort_farki_z": float(np.nanmean(dmu)), "ort_std_orani": float(np.nanmean(sor)),
                    "ort_ks_D": float(np.mean(ks_list))}
        if ad == "solar":
            ozet[nm]["sifir_orani"] = float((R_ == 0).mean())
    pd.DataFrame(rows).to_csv(data_dir / f"kayma_istatistikleri_{ad}.csv", index=False, float_format="%.6g")
    M["dagilim_ozet"] = ozet
    M["scaler"] = {"mean": split["scaler_train_only"]["mean"], "scale": split["scaler_train_only"]["scale"]}
    log(f"[{ad}][dagilim]", json.dumps(ozet, ensure_ascii=False))

    # ---------------- hedef pencereleri (rakip_kos.py bicimi)
    hed_dir = data_dir / f"hedefler_sl96_{ad}"
    M["hedefler"] = {"dizin": str(hed_dir), "bicim": "np.save float32 (N,H,C); rakip_kos.hedef_dogrula_yaz ile ayni",
                     "kaynak": "resmi Dataset sinifinin data_y'si; true[i] = data_y[i+96 : i+96+H]", "dosyalar": []}
    if hedef:
        import shutil
        gerek = sum(split["pencere"][str(H)][nm]["pencere"] * H * C * 4 for H in HORIZONS for nm in ("val", "test"))
        bos = shutil.disk_usage(str(data_dir)).free
        M["hedefler"]["toplam_bayt"] = int(gerek)
        log(f"[{ad}][hedef] gereken {gerek / 2**30:.2f} GiB, bos {bos / 2**30:.1f} GiB")
        hed_dir.mkdir(parents=True, exist_ok=True)
        for nm in ("val", "test"):
            for H in HORIZONS:
                n = split["pencere"][str(H)][nm]["pencere"]
                f = hed_dir / f"{nm}_h{H}_true.npy"
                if not f.exists() and shutil.disk_usage(str(data_dir)).free < n * H * C * 4 + 8 * 2**30:
                    raise SystemExit(f"DISK: {f} icin yer yok (8 GiB pay korunur)")
                info = hedef_yaz(f, resmi_y[nm], n, H)
                M["hedefler"]["dosyalar"].append(info)
                log(f"[{ad}][hedef]", info["dosya"], info["sekil"], info["durum"], info["sha256"][:16])
    del resmi_y

    # ---------------- rakip_kos.py VERILER satiri onerisi (sarmalayici bu ajan tarafindan DEGISTIRILMEDI)
    M["rakip_kos_veriler_onerisi"] = {
        "sh": r["sh"], "dizin": f"data/{ad}", "csv": fn, "sha256": h["sha256"], "hedef": f"data/hedefler_sl96_{ad}",
        "val_b1": b1[1], "test_b1": b1[2], "bolme": r["bolme"], "data_dict_anahtari": r["data_anahtari"],
        "not": ("Solar: resmi test()/train() 'Solar' icin x_mark=None verir; rakip_kos.ileri() x_mark'i her zaman "
                "gecer -> Solar val olcumunde resmi davranisla AYNI DEGIL, uyarlanmali") if ad == "solar" else ""}

    split_path = data_dir / "splits" / f"{ad}_sl96.json"
    split_path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    M["split_dosyasi"] = str(split_path)
    M["split"] = split
    log(f"[{ad}][yazildi]", split_path)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--veri", nargs="+", default=["etth1", "etth2"], choices=list(RESMI) + list(FAZ3))
    ap.add_argument("--hedef-yok", action="store_true", help="FAZ III: hedef pencerelerini yazma")
    args = ap.parse_args()
    ws = Path(args.ws)
    (ws / "data" / "splits").mkdir(parents=True, exist_ok=True)
    ett = [v for v in args.veri if v in RESMI]
    f3 = [v for v in args.veri if v in FAZ3]
    if ett:  # --- FAZ II ETT yolu: degismedi
        out = {"olusturma_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "komut": "PYTHONIOENCODING=utf-8 HF_HOME=data/hf_cache env/.venv/Scripts/python.exe -I -X utf8 scripts/veri_hazirla_ett.py",
               "ortam": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
               "veri_setleri": {}}
        for ad in ett:
            out["veri_setleri"][ad] = hazirla(ad, ws, args.force)
        p = ws / "data" / "data_manifest_ett.json"
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        log("[yazildi]", p)
    if f3:   # --- FAZ III
        t0 = time.time()
        p = ws / "data" / "data_manifest_faz3.json"
        out = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"veri_setleri": {}}
        zu, zmeta = itr_zip_uyeleri(ws, args.force)
        out.update({"olusturma_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "komut": ("PYTHONIOENCODING=utf-8 HF_HOME=data/hf_cache env/.venv/Scripts/python.exe -I -X utf8 "
                              "scripts/veri_hazirla_ett.py --veri " + " ".join(f3)
                              + (" --hedef-yok" if args.hedef_yok else "")),
                    "ortam": {"python": sys.version.split()[0], "platform": platform.platform(),
                              "numpy": np.__version__, "pandas": pd.__version__,
                              "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
                    "itransformer_arsivi": zmeta})
        for ad in f3:
            out["veri_setleri"][ad] = hazirla_faz3(ad, ws, args.force, zu, not args.hedef_yok)
            p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        out["sure_sn"] = round(time.time() - t0, 1)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        log("[yazildi]", p, f"{out['sure_sn']} s")


if __name__ == "__main__":
    main()
