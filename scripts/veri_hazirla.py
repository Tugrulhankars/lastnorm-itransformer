#!/usr/bin/env python
"""K2 veri hazirligi -- Exchange-Rate (konu 5.3), TSLib Dataset_Custom protokolu.

Tekrar kosulabilir (idempotent): dosya zaten varsa ve hash'i tutuyorsa yeniden indirmez.
Kullanim (WS kokunden, Git Bash):
    PYTHONIOENCODING=utf-8 HF_HOME=data/hf_cache env/.venv/Scripts/python.exe -I -X utf8 scripts/veri_hazirla.py [--force]

Ciktilar (WS/data altinda):
    exchange_rate/exchange_rate.csv            -- TSLib'in kullandigi dosya (HF thuml, sabit revizyon)
    exchange_rate/kaynak_lai2018/exchange_rate.txt.gz -- LSTNet orijinali (deger karsilastirmasi icin)
    splits/exchange_rate_tslib_sl96.json       -- sabit kronolojik split + pencere sayilari
    kayma_istatistikleri.csv                   -- dilim x degisken dagilim/kayma tablosu
    kayma_mmd.csv                              -- MMD tanim varyantlari (dilim bazinda)
    naive_val.csv                              -- repeat-last / sifir baseline, YALNIZ validation
    data_manifest.json                         -- tum sayilar

TEST DILIMI UZERINDE MODEL/NAIVE METRIGI HESAPLANMAZ; testte yalniz betimsel istatistik.
"""
import argparse
import gzip
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
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- sabitler
HF_REPO = "thuml/Time-Series-Library"
HF_REV = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"  # HF dataset repo sha (2025-11-12)
HF_URL = (f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/"
          "exchange_rate/exchange_rate.csv")
HF_GIT_BLOB_OID = "593763acecde07a4fcb6582741e615cca91a07b7"  # HF tree API 'oid' (git blob sha1)
HF_SIZE = 637800
LAI_REV = "7f402f185cc2435b5e66aed13a3b560ed142e023"  # laiguokun/multivariate-time-series-data master
LAI_URL = (f"https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
           f"{LAI_REV}/exchange_rate/exchange_rate.txt.gz")

SEQ_LEN = 96
LABEL_LEN = 48          # iTransformer/TSLib run.py varsayilani (iTransformer kullanmaz)
HORIZONS = [96, 192, 336, 720]
BATCH = 32              # iTransformer/TSLib run.py varsayilani
TARGET = "OT"
# LSTNet README sirasi; CSV'de sutun adi yok ('0'..'6','OT') -> eslesme VARSAYIM
ULKE_VARSAYIM = ["Australia", "British", "Canada", "Switzerland", "China", "Japan",
                 "New Zealand", "Singapore"]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- indirme / hash
def hashes(b: bytes) -> dict:
    return {
        "bytes": len(b),
        "sha256": hashlib.sha256(b).hexdigest(),
        "md5": hashlib.md5(b).hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest(),
    }


def download(url: str, dest: Path, force: bool) -> tuple[bytes, bool]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest.read_bytes(), False
    req = urllib.request.Request(url, headers={"User-Agent": "veri_hazirla.py/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(b)
    tmp.replace(dest)
    return b, True


# ---------------------------------------------------------------- MMD
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


def median_dist(P):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P[:, None]
    return float(np.median(pdist(P)))


# ---------------------------------------------------------------- ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--force", action="store_true", help="dosyalari yeniden indir")
    args = ap.parse_args()
    ws = Path(args.ws)
    data_dir = ws / "data"
    csv_path = data_dir / "exchange_rate" / "exchange_rate.csv"
    lai_path = data_dir / "exchange_rate" / "kaynak_lai2018" / "exchange_rate.txt.gz"
    (data_dir / "splits").mkdir(parents=True, exist_ok=True)
    M = {"olusturma_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "komut": "PYTHONIOENCODING=utf-8 HF_HOME=data/hf_cache env/.venv/Scripts/python.exe -I -X utf8 scripts/veri_hazirla.py",
         "ortam": {"python": sys.version.split()[0], "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}}

    # ---- V1/V2 indirme + butunluk
    b, yeni = download(HF_URL, csv_path, args.force)
    h = hashes(b)
    log("[V1] TSLib CSV:", HF_URL, "yeni indirildi" if yeni else "(mevcut dosya)")
    log("[V2] ", h)
    if h["bytes"] != HF_SIZE or h["git_blob_sha1"] != HF_GIT_BLOB_OID:
        raise SystemExit(f"HASH/BOYUT UYUSMAZ: beklenen {HF_SIZE} / {HF_GIT_BLOB_OID}")
    M["kaynak_tslib"] = {"url": HF_URL, "hf_repo": HF_REPO, "hf_revizyon": HF_REV,
                         "resmi_git_blob_oid": HF_GIT_BLOB_OID, "resmi_boyut": HF_SIZE,
                         "olculen": h, "resmi_eslesme": True, "yerel_yol": str(csv_path)}

    bl, yeni_l = download(LAI_URL, lai_path, args.force)
    hl = hashes(bl)
    lai = np.loadtxt(gzip.decompress(bl).decode("utf-8").splitlines(), delimiter=",")
    log("[V1] Lai orijinal:", LAI_URL, lai.shape, hl["sha256"])

    # ---- okuma ve temel denetimler
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    feat = [c for c in cols if c not in ("date", TARGET)] + [TARGET]  # TSLib yeniden siralamasi
    df = df[["date"] + feat]
    N = len(df)
    dates = pd.to_datetime(df["date"])
    X = df[feat].to_numpy(dtype=np.float64)
    dd = dates.diff().dropna().value_counts()
    vals_str = df[feat].astype(str).agg("|".join, axis=1)
    consec_same = int((np.abs(np.diff(X, axis=0)).sum(1) == 0).sum())
    M["dosya"] = {
        "satir": N, "sutun": len(cols), "sutunlar": cols, "tslib_sutun_sirasi": ["date"] + feat,
        "degisken_sayisi": len(feat), "hedef": TARGET,
        "tarih_ilk": str(dates.iloc[0].date()), "tarih_son": str(dates.iloc[-1].date()),
        "tarih_monoton_artan": bool(dates.is_monotonic_increasing),
        "ardisik_tarih_farki_dagilimi": {str(k): int(v) for k, v in dd.items()},
        "hafta_sonu_tarih_sayisi": int((dates.dt.weekday >= 5).sum()),
        "eksik_deger_toplam": int(df.isna().sum().sum()),
        "eksik_deger_sutun": {c: int(v) for c, v in df.isna().sum().items()},
        "tekrar_eden_tarih": int(dates.duplicated().sum()),
        "tekrar_eden_tam_satir_tarih_dahil": int(df.duplicated().sum()),
        "tekrar_eden_deger_vektoru_tarih_haric": int(vals_str.duplicated().sum()),
        "ardisik_ozdes_deger_satiri": consec_same,
        "sonlu_olmayan_deger": int((~np.isfinite(X)).sum()),
        "ulke_eslesmesi_varsayim": dict(zip(feat, ULKE_VARSAYIM)),
    }
    log("[dosya]", json.dumps({k: v for k, v in M["dosya"].items()
                               if k not in ("eksik_deger_sutun", "ulke_eslesmesi_varsayim")},
                              ensure_ascii=False))

    # ---- orijinal (Lai) ile deger karsilastirmasi
    same_shape = lai.shape == X.shape
    cmp = {"lai_url": LAI_URL, "lai_revizyon": LAI_REV, "lai_hash": hl, "lai_sekil": list(lai.shape),
           "sekil_ayni": bool(same_shape), "lai_tarih_sutunu": False}
    if same_shape:
        diff = np.abs(lai - X)
        cmp.update({"dogrudan_max_mutlak_fark": float(diff.max()),
                    "dogrudan_fark_gt_1e-6_hucre_sutun_bazinda": dict(zip(feat, map(int, (diff > 1e-6).sum(0)))),
                    "toplam_hucre": int(diff.size)})
        # her CSV sutunu icin Lai'de birebir eslesen sutunu bul (sutun permutasyonu)
        perm = {}
        for j, c in enumerate(feat):
            m = [k for k in range(lai.shape[1]) if np.array_equal(lai[:, k], X[:, j])]
            perm[c] = m[0] if m else None
        cmp["csv_sutunu_to_lai_sutun_indeksi"] = perm
        cmp["permutasyon_sonrasi_birebir_ayni"] = all(v is not None for v in perm.values())
        M["dosya"]["ulke_eslesmesi_varsayim"] = {
            c: (ULKE_VARSAYIM[k] if k is not None else None) for c, k in perm.items()}
        M["dosya"]["ulke_eslesmesi_not"] = (
            "LSTNet README ulke sirasi Lai sutun sirasina uygulanir (varsayim); TSLib CSV Lai'nin "
            "son iki sutununu yer degistirmis -> OT = Lai sutun 6")
    M["orijinal_karsilastirma"] = cmp
    log("[Lai karsilastirma]", cmp)

    # ---- V3 split (TSLib Dataset_Custom ile birebir)
    num_train = int(N * 0.7)
    num_test = int(N * 0.2)
    num_vali = N - num_train - num_test
    b1 = [0, num_train - SEQ_LEN, N - num_test - SEQ_LEN]
    b2 = [num_train, num_train + num_vali, N]
    own = {"train": (0, num_train), "val": (num_train, num_train + num_vali),
           "test": (N - num_test, N)}
    names = ["train", "val", "test"]
    d = lambda i: str(dates.iloc[i].date())
    split = {
        "veri": "exchange_rate.csv", "protokol": "TSLib Dataset_Custom (data_provider/data_loader.py)",
        "seed": None, "not": "deterministik kronolojik bolme; rastgelelik yok, seed gerekmez",
        "oran": "train 0.7 / test 0.2 / val kalan (int() ile)", "N": N,
        "num_train": num_train, "num_vali": num_vali, "num_test": num_test,
        "seq_len": SEQ_LEN, "label_len": LABEL_LEN,
        "border1s": b1, "border2s": b2,
        "dilim": {},
        "pencere": {},
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
                # shuffle=False -> atilan pencereler EN SON pencerelerdir (deterministik)
                last_kept_t = b1[k] + (n - rem - 1) + SEQ_LEN + H - 1
                e["drop_last_True_hic_degerlendirilmeyen_son_satir"] = last_t - last_kept_t
                e["drop_last_True_son_degerlendirilen_tarih"] = d(last_kept_t)
                e["drop_last_True_atilan_pencere_ilk_hedef_tarihleri"] = [
                    d(b1[k] + (n - rem) + SEQ_LEN), d(b1[k] + (n - 1) + SEQ_LEN)] if rem else []
            split["pencere"][str(H)][nm] = e
            drop_tab.append({"ufuk": H, "dilim": nm, "pencere": n, "atilan_bs32": rem})
    log("[V3] num_train/vali/test =", num_train, num_vali, num_test, "| border1s", b1, "border2s", b2)
    for r in drop_tab:
        log("   ", r)

    # ---- scaler (yalniz train)
    sc = StandardScaler().fit(X[b1[0]:b2[0]])
    Z = sc.transform(X)
    sc_all = StandardScaler().fit(X)
    tr = X[:num_train]
    scal = {
        "fit_araligi": [b1[0], b2[0]], "fit_satir": b2[0] - b1[0],
        "mean_train_ile_ayni": bool(np.allclose(sc.mean_, tr.mean(0))),
        "scale_train_std_ddof0_ile_ayni": bool(np.allclose(sc.scale_, tr.std(0, ddof=0))),
        "train_z_ort_max_mutlak": float(np.abs(Z[:num_train].mean(0)).max()),
        "train_z_std_min_max": [float(Z[:num_train].std(0).min()), float(Z[:num_train].std(0).max())],
        "mean": dict(zip(feat, map(float, sc.mean_))), "scale": dict(zip(feat, map(float, sc.scale_))),
        "tum_veriye_fit_edilseydi_ort_farki_z": dict(zip(
            feat, map(float, (sc_all.mean_ - sc.mean_) / sc.scale_))),
        "tum_veriye_fit_edilseydi_scale_orani": dict(zip(feat, map(float, sc_all.scale_ / sc.scale_))),
        "tslib_kodu": "train_data = df_data[border1s[0]:border2s[0]]; self.scaler.fit(train_data.values)",
    }
    split["scaler_train_only"] = {"mean": scal["mean"], "scale": scal["scale"]}
    M["scaler"] = scal
    log("[scaler]", {k: scal[k] for k in ("mean_train_ile_ayni", "scale_train_std_ddof0_ile_ayni",
                                         "train_z_ort_max_mutlak", "train_z_std_min_max")})

    # ---- V5 sizinti denetimi
    leak = {"hedef_sinirlari": {}, "girdi_geri_kaydirma_ortusmesi": {}}
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
            "hedef_satiri_ortusmesi_val_test": max(0, P["val"]["son_hedef_satiri"] - P["test"]["ilk_hedef_satiri"] + 1),
        }
    leak["tum_ufuklarda_hedef_ayrik"] = bool(ok_all)
    leak["girdi_geri_kaydirma_ortusmesi"] = {
        "val_girdisinde_train_satiri": num_train - b1[1], "test_girdisinde_val_satiri": b2[1] - b1[2],
        "not": "meşru: girdi penceresi gecmise bakar; decoder label_len=48 kismi da girdi bolgesinde"}
    # deger-vektoru cakismasi (satir duzeyi) ve pencere duzeyi hedef blogu cakismasi
    keys = [r.tobytes() for r in X]
    tr_set = set(keys[0:num_train])
    va_set = set(keys[own["val"][0]:own["val"][1]])
    te_keys = keys[own["test"][0]:own["test"][1]]
    leak["satir_deger_cakismasi"] = {
        "test_satiri_train_icinde": int(sum(k in tr_set for k in te_keys)),
        "test_satiri_val_icinde": int(sum(k in va_set for k in te_keys)),
        "val_satiri_train_icinde": int(sum(k in tr_set for k in keys[own["val"][0]:own["val"][1]])),
        "tarih_cakismasi_train_test": int(len(set(dates.iloc[:num_train]) & set(dates.iloc[own["test"][0]:]))),
    }
    win_overlap = {}
    for H in HORIZONS:
        def blocks(k):
            n = (b2[k] - b1[k]) - SEQ_LEN - H + 1
            return {X[b1[k] + s + SEQ_LEN: b1[k] + s + SEQ_LEN + H].tobytes() for s in range(n)}
        btr, bte = blocks(0), blocks(2)
        win_overlap[str(H)] = int(len(btr & bte))
    leak["hedef_blogu_cakismasi_train_test"] = win_overlap
    leak["scaler_sizintisi"] = "yok: fit yalniz [0, num_train) (yukaridaki dogrulama)"
    M["sizinti"] = leak
    log("[V5]", json.dumps(leak, ensure_ascii=False)[:1500])

    # ---- dagilim kaymasi
    te0, te1 = own["test"]
    q_idx = np.array_split(np.arange(te0, te1), 4)
    slices = {"train": (0, num_train), "val": own["val"], "test": own["test"]}
    for i, q in enumerate(q_idx, 1):
        slices[f"test_Q{i}"] = (int(q[0]), int(q[-1]) + 1)
    split["test_ceyrekleri_kendi_satirlari"] = {
        k: {"satir[o0,o1)": list(v), "ilk_tarih": d(v[0]), "son_tarih": d(v[1] - 1)}
        for k, v in slices.items() if k.startswith("test_Q")}
    split["test_ceyrek_atama_onerisi"] = (
        "hedef-satiri bazinda: test pencere i, adim h (0-index) -> hedef satiri = "
        f"{b1[2]}+i+{SEQ_LEN}+h; ceyrek = bu satirin dustugu test_Qk araligi. "
        "Ilk-hedef-satiri bazli atama H=720'de Q3/Q4'u kapsamaz.")

    Ztr = Z[:num_train]
    sig_z = median_dist(np.vstack([Ztr, Z[te0:te1]]))
    sig_raw = median_dist(np.vstack([X[:num_train], X[te0:te1]]))
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
                "ham_ort": R[:, j].mean(), "ham_std": R[:, j].std(), "ham_min": R[:, j].min(),
                "ham_max": R[:, j].max(),
                "z_ort": Zs[:, j].mean(), "z_std": Zs[:, j].std(), "z_min": Zs[:, j].min(),
                "z_max": Zs[:, j].max(),
                "kayma_ort_farki_z": Zs[:, j].mean() - Ztr[:, j].mean(),
                "kayma_std_orani": Zs[:, j].std() / Ztr[:, j].std(),
                "z_train_araligi_disi_oran": float(((Zs[:, j] < Ztr[:, j].min()) |
                                                    (Zs[:, j] > Ztr[:, j].max())).mean()),
                "ks_D": ks.statistic if ks else 0.0,
                "mmd2_1d_rbf_medyan": m1,
            })
        if nm == "train":
            continue
        bz, uz = mmd2(Ztr, Zs, sig_z)
        b1z, u1z = mmd2(Ztr, Zs, 1.0)
        br, ur = mmd2(X[:num_train], R, sig_raw)
        sig_own = median_dist(np.vstack([Ztr, Zs]))
        bo, uo = mmd2(Ztr, Zs, sig_own)
        sub = [r for r in rows if r["dilim"] == nm]
        mr = {"dilim": nm, "satir_sayisi": bnd - a, "ilk_tarih": d(a), "son_tarih": d(bnd - 1),
              "ort_mutlak_ort_farki_z": float(np.mean([abs(r["kayma_ort_farki_z"]) for r in sub])),
              "ort_std_orani": float(np.mean([r["kayma_std_orani"] for r in sub])),
              "ort_ks_D": float(np.mean([r["ks_D"] for r in sub])),
              "ort_mmd2_1d": float(np.mean([r["mmd2_1d_rbf_medyan"] for r in sub])),
              "sigma_z_medyan_train+test": sig_z,
              "mmd2_b_z_sigmaMed": bz, "mmd_b_z_sigmaMed": float(np.sqrt(max(bz, 0))),
              "mmd2_u_z_sigmaMed": uz,
              "mmd2_b_z_sigma1": b1z, "mmd_b_z_sigma1": float(np.sqrt(max(b1z, 0))),
              "sigma_z_medyan_train+dilim": sig_own,
              "mmd2_b_z_sigmaMedDilim": bo, "mmd_b_z_sigmaMedDilim": float(np.sqrt(max(bo, 0))),
              "sigma_ham_medyan_train+test": sig_raw,
              "mmd2_b_ham_sigmaMed": br, "mmd_b_ham_sigmaMed": float(np.sqrt(max(br, 0)))}
        mmd_rows.append(mr)
        rows.append({"dilim": nm, "satir_bas": a, "satir_bit_haric": bnd, "satir_sayisi": bnd - a,
                     "ilk_tarih": d(a), "son_tarih": d(bnd - 1), "degisken": "TUM(8)",
                     "kayma_ort_farki_z": mr["ort_mutlak_ort_farki_z"],
                     "kayma_std_orani": mr["ort_std_orani"], "ks_D": mr["ort_ks_D"],
                     "mmd2_1d_rbf_medyan": mr["ort_mmd2_1d"],
                     "mmd2_8d_z_rbf_medyan": bz, "mmd_8d_z_rbf_medyan": mr["mmd_b_z_sigmaMed"]})
        log(f"[kayma] {nm:8s} |dmu|={mr['ort_mutlak_ort_farki_z']:.3f} std_or={mr['ort_std_orani']:.3f} "
            f"KS={mr['ort_ks_D']:.3f} MMD2(z,med)={bz:.4f} MMD(z,med)={mr['mmd_b_z_sigmaMed']:.4f} "
            f"MMD2(z,s1)={b1z:.4f} MMD(z,s1)={mr['mmd_b_z_sigma1']:.4f} MMD2(ham)={br:.4f}")
    kdf = pd.DataFrame(rows)
    kdf.to_csv(data_dir / "kayma_istatistikleri.csv", index=False, float_format="%.6g")
    pd.DataFrame(mmd_rows).to_csv(data_dir / "kayma_mmd.csv", index=False, float_format="%.6g")
    M["kayma"] = {"mmd_tanimlari": {
        "cekirdek": "RBF exp(-||x-y||^2/(2 sigma^2)); X=train kendi satirlari, Y=dilim satirlari (pencere degil, satir)",
        "sigmaMed": "train+test birlesik ornekte ciftler arasi Oklid uzakligi medyani (z-uzayi), tum dilimlerde sabit",
        "sigmaMedDilim": "train+ilgili dilim birlesik medyani", "sigma1": "sigma=1 (z-uzayi)",
        "b/u": "biased V-istatistigi / unbiased U-istatistigi; mmd_* = sqrt(mmd2_b)"},
        "ozet": mmd_rows}

    # ---- naive baseline: YALNIZ validation (z-uzayi, features=M, 8 degisken)
    nv = []
    for H in HORIZONS:
        n = (b2[1] - b1[1]) - SEQ_LEN - H + 1
        s = np.arange(n)
        last_in = b1[1] + s + SEQ_LEN - 1                      # son girdi satiri
        tgt = last_in[:, None] + 1 + np.arange(H)[None, :]      # (n,H)
        assert tgt.max() == b2[1] - 1 and tgt.min() == own["val"][0]
        Y = Z[tgt]                                              # (n,H,8)
        P_last = Z[last_in][:, None, :]
        for yon, P in (("repeat_last", P_last), ("sifir_train_ort", np.zeros_like(P_last))):
            err = Y - P
            nv.append({"ufuk": H, "seq_len": SEQ_LEN, "dilim": "val", "pencere": n,
                       "drop_last_True_bs32_atilan": n % BATCH, "yontem": yon,
                       "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                       "mse_OT": float((err[..., -1] ** 2).mean()), "mae_OT": float(np.abs(err[..., -1]).mean())})
    nvdf = pd.DataFrame(nv)
    nvdf.to_csv(data_dir / "naive_val.csv", index=False, float_format="%.6f")
    rl = nvdf[nvdf.yontem == "repeat_last"]
    log("[naive val]\n" + nvdf.to_string(index=False))
    M["naive_val"] = nv
    M["naive_val_ort_repeat_last"] = {"mse": float(rl.mse.mean()), "mae": float(rl.mae.mean())}

    # ---- yaz
    split_path = data_dir / "splits" / "exchange_rate_tslib_sl96.json"
    split_path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    M["split_dosyasi"] = str(split_path)
    M["split"] = split
    M["drop_last_tablosu"] = drop_tab
    (data_dir / "data_manifest.json").write_text(
        json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("[yazildi]", split_path, data_dir / "data_manifest.json")


if __name__ == "__main__":
    main()
