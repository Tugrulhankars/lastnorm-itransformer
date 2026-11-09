#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
onkosul.py -- H05 (SAN-mu) ON-KOSUL KAPISI (CPU; yalniz train/val; TEST YUKLEYICISI KURULMAZ).

Kart: hipotezler/H05.md, "On-kosul kapisi" satiri. Esikler karttan SABIT KODLANDI (asagida).

Veri: H01 tam kosum checkpoint'leri (runs/20261014-040042_san_s2023, ...040547_san_s2024, ...041102_san_s2025),
ckpt/*/checkpoint.pth, ufuk projektor sekliyle secilir (teshis_checkpoint.py yontemi). SAN modulu ortak fazda
guncellenmedigi icin buradaki mu-hat, SAN-mu'nun ayni seed'de kullanacagi mu-hat ile ayni kod yolundan gelir
(ayrica: ana checkpoint'teki san.* agirliklari station/checkpoint.pth ile birebir mi -> bilgi olarak basilir).

Her ufuk H, seed s: dilim k = 1..H/6, degisken d, pencere i uzerinden z uzayinda
  E_san  = MSE(mu_hat_k, mu_k)          (mu_hat: SAN ortalama-MLP cikisi, san.normalize(x)[1][..., :C])
  E_naif = MSE(mu_son_dilim, mu_k)      (son 6 girdi degerinin ortalamasi)
  E_son  = MSE(x_son, mu_k)             (raporlama)
  E_lb   = MSE(mu_lb, mu_k)             (raporlama; 96 adimlik pencere ortalamasi)
  mu_k = hedefin k. 6'lik diliminin ortalamasi.
  R_H = ort_s(E_san) / ort_s(E_naif)
KAPI (karar split'i: VAL, tum pencereler, adim 1; TRAIN yalniz bilgi olarak raporlanir):
  O1 GECER <=> #{H : R_H < 1.00} >= 3  VE  ort_H(R_H) < 1.00
  O2 GECER <=> val'de (pencere x degisken) hucrelerinin sigma_son_dilim < 1e-3 z orani <= %10
               (sigma_son_dilim = torch.std(son 6 girdi, unbiased), SAN normalize formulu; her ufkun val
               pencere kumesi icin ayri hesaplanir, kapida EN KOTU (maks.) ufuk kullanilir)
  KAPI = O1 ve O2. Son satir: KAPI=GECTI | KAPI=BASARISIZ

Cikti: varyantlar/san-mu/onkosul.log (stdout'un kopyasi) + sonuclar/h05-onkosul.csv
Kullanim (WS kokunden): env/.venv/Scripts/python.exe varyantlar/san-mu/onkosul.py
"""
import csv
import glob
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kart: CPU olcumu (GPU yok)

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
sys.path.insert(0, str(WS / "varyantlar" / "san"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# --- karttan sabit esikler (H05.md) ---
R_ESIK = 1.00          # R_H < 1.00
O1_MIN_UFUK = 3        # #{H : R_H < 1} >= 3
O1_ORT_ESIK = 1.00     # ort_H(R_H) < 1.00
O2_SIGMA_ESIK = 1e-3   # sigma_son_dilim < 1e-3 z
O2_ORAN_ESIK = 0.10    # oran <= %10
P = 6                  # SAN period_len (H01 / SAN resmi Exchange ayari)

KOSUMLAR = {2023: "20261014-040042_san_s2023", 2024: "20261014-040547_san_s2024", 2025: "20261014-041102_san_s2025"}
UFUKLAR = [96, 192, 336, 720]
SPLITLER = ["val", "train"]  # test YOK

LOG = HERE / "onkosul.log"
CSV = WS / "sonuclar" / "h05-onkosul.csv"
_log = open(LOG, "w", encoding="utf-8")


def yaz(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log.write(s + "\n")
    _log.flush()


def args_kur(parser, sh, H):
    argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN) + os.sep)
    argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
    argv = rakip_kos.argv_degistir(argv, "--use_norm", "0")
    return parser.parse_args(argv)


def ckpt_bul(run_dir, H):
    for p in sorted(glob.glob(str(run_dir / "ckpt" / "*" / "checkpoint.pth"))):
        sd = torch.load(p, map_location="cpu")
        if sd["backbone.projector.weight"].shape[0] == H:
            return Path(p), sd
    raise SystemExit(f"{run_dir}: h{H} checkpoint'i bulunamadi")


def veri(a, split):
    from data_provider.data_factory import data_dict
    assert split in ("train", "val"), split  # test bayragiyla veri KURULMAZ
    ds = data_dict["custom"](root_path=a.root_path, data_path=a.data_path, flag=split,
                             size=[a.seq_len, a.label_len, a.pred_len], features=a.features,
                             target=a.target, timeenc=1, freq=a.freq)
    X = np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32)          # [N, 96, C]
    Y = np.stack([ds[i][1] for i in range(len(ds))])[:, -a.pred_len:, :].astype(np.float32)  # [N, H, C]
    return torch.from_numpy(X), torch.from_numpy(Y)


def main():
    t0 = time.time()
    from model_san import SANiTransformer
    torch.set_grad_enabled(False)
    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    yaz(f"[H05 onkosul] {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | cihaz CPU "
        f"(cuda gorunur={torch.cuda.is_available()}) | kosumlar {KOSUMLAR}")
    yaz(f"[H05 onkosul] esikler: R_H<{R_ESIK}, O1: #{{R_H<1}}>={O1_MIN_UFUK} ve ort R<{O1_ORT_ESIK}; "
        f"O2: sigma_son_dilim<{O2_SIGMA_ESIK} orani<={O2_ORAN_ESIK:.0%} (val); P={P}")

    satirlar = []
    E = {}       # (split, H, s) -> dict
    o2 = {}      # H -> oran (val)
    for H in UFUKLAR:
        a = args_kur(parser, sh, H)
        assert a.seq_len % P == 0 and H % P == 0
        C = a.enc_in
        veriler = {sp: veri(a, sp) for sp in SPLITLER}
        # --- O2 (seed'den bagimsiz: yalniz girdi)
        Xv, _ = veriler["val"]
        sig = torch.std(Xv.reshape(Xv.shape[0], -1, P, C)[:, -1], dim=1)  # [N, C] unbiased
        o2[H] = float((sig < O2_SIGMA_ESIK).float().mean())
        yaz(f"h{H}: val pencere={Xv.shape[0]} | sigma_son_dilim ort={float(sig.mean()):.4f} min={float(sig.min()):.2e} "
            f"| <{O2_SIGMA_ESIK} orani={o2[H]:.4f}")
        for s, rid in KOSUMLAR.items():
            run_dir = WS / "runs" / rid
            p, sd = ckpt_bul(run_dir, H)
            m = SANiTransformer(a).float()
            m.load_state_dict(sd)
            m.eval()
            # bilgi: ana checkpoint'teki SAN agirliklari istasyon checkpoint'i ile birebir mi
            st_p = p.parent / "station" / "checkpoint.pth"
            st_esit = None
            if st_p.exists():
                st = torch.load(st_p, map_location="cpu")
                ms = m.san.state_dict()
                st_esit = st.keys() == ms.keys() and all(torch.equal(st[k], ms[k]) for k in st)
            for sp in SPLITLER:
                X, Y = veriler[sp]
                _, stats = m.san.normalize(X)                         # [N, H/P, 2C]
                mu_hat = stats[..., :C].double()
                sig_hat = stats[..., C:]
                mu_k = Y.reshape(Y.shape[0], -1, P, C).mean(2).double()  # [N, H/P, C]
                Xd = X.double()
                mu_son_dilim = Xd[:, -P:, :].mean(1, keepdim=True)
                x_son = Xd[:, -1:, :]
                mu_lb = Xd.mean(1, keepdim=True)
                e = {
                    "E_san": float(((mu_hat - mu_k) ** 2).mean()),
                    "E_naif": float(((mu_son_dilim - mu_k) ** 2).mean()),
                    "E_son": float(((x_son - mu_k) ** 2).mean()),
                    "E_lb": float(((mu_lb - mu_k) ** 2).mean()),
                    "sigma_hat_sifir_orani": float((sig_hat == 0).float().mean()),
                    "n_pencere": int(X.shape[0]),
                }
                e["R"] = e["E_san"] / e["E_naif"]
                E[(sp, H, s)] = e
                satirlar.append({"split": sp, "ufuk": H, "seed": s, "kosum": rid, **e,
                                 "o2_oran_val": o2[H] if sp == "val" else "", "san_ckpt_istasyonla_ayni": st_esit})
                yaz(f"  h{H} s{s} {sp:5s}: E_san={e['E_san']:.6f} E_naif={e['E_naif']:.6f} E_son={e['E_son']:.6f} "
                    f"E_lb={e['E_lb']:.6f} R={e['R']:.4f} | sigma_hat==0 orani={e['sigma_hat_sifir_orani']:.3f} "
                    f"| n={e['n_pencere']} | san_ckpt==istasyon_ckpt: {st_esit}")

    # --- ozet ve kapi
    R = {sp: {} for sp in SPLITLER}
    yaz("")
    yaz("OZET (seed ortalamalari; R_H = ort_s(E_san) / ort_s(E_naif)):")
    yaz("split | H | E_san | E_naif | E_son | E_lb | R_H")
    for sp in SPLITLER:
        for H in UFUKLAR:
            ort = {k: float(np.mean([E[(sp, H, s)][k] for s in KOSUMLAR])) for k in ("E_san", "E_naif", "E_son", "E_lb")}
            R[sp][H] = ort["E_san"] / ort["E_naif"]
            satirlar.append({"split": sp, "ufuk": H, "seed": "ort", "kosum": "3-seed", **ort, "R": R[sp][H],
                             "sigma_hat_sifir_orani": float(np.mean([E[(sp, H, s)]["sigma_hat_sifir_orani"] for s in KOSUMLAR])),
                             "n_pencere": E[(sp, H, 2023)]["n_pencere"],
                             "o2_oran_val": o2[H] if sp == "val" else "", "san_ckpt_istasyonla_ayni": ""})
            yaz(f"{sp} | {H} | {ort['E_san']:.6f} | {ort['E_naif']:.6f} | {ort['E_son']:.6f} | {ort['E_lb']:.6f} | {R[sp][H]:.4f}")

    rv = R["val"]
    n_alti = sum(1 for H in UFUKLAR if rv[H] < R_ESIK)
    ort_R = float(np.mean([rv[H] for H in UFUKLAR]))
    o1 = n_alti >= O1_MIN_UFUK and ort_R < O1_ORT_ESIK
    o2_maks = max(o2.values())
    o2_ok = o2_maks <= O2_ORAN_ESIK
    kapi = o1 and o2_ok
    yaz("")
    yaz(f"O1 (val): R_H = " + ", ".join(f"h{H} {rv[H]:.4f}" for H in UFUKLAR)
        + f" | #{{R_H<1}} = {n_alti}/4 (gerekli >= {O1_MIN_UFUK}) | ort R = {ort_R:.4f} (gerekli < {O1_ORT_ESIK})"
        + f" -> {'GECER' if o1 else 'BASARISIZ'}")
    yaz(f"O1 (train, yalniz bilgi): R_H = " + ", ".join(f"h{H} {R['train'][H]:.4f}" for H in UFUKLAR)
        + f" | ort R = {float(np.mean(list(R['train'].values()))):.4f}")
    yaz(f"O2 (val): sigma_son_dilim<{O2_SIGMA_ESIK} orani = " + ", ".join(f"h{H} {o2[H]:.4f}" for H in UFUKLAR)
        + f" | maks {o2_maks:.4f} (gerekli <= {O2_ORAN_ESIK}) -> {'GECER' if o2_ok else 'BASARISIZ'}")
    satirlar.append({"split": "KAPI", "ufuk": "", "seed": "", "kosum": "",
                     "E_san": "", "E_naif": "", "E_son": "", "E_lb": "", "R": ort_R,
                     "sigma_hat_sifir_orani": "", "n_pencere": "", "o2_oran_val": o2_maks,
                     "san_ckpt_istasyonla_ayni": f"O1={'GECER' if o1 else 'BASARISIZ'}(#R<1={n_alti});"
                                                 f"O2={'GECER' if o2_ok else 'BASARISIZ'};"
                                                 f"KAPI={'GECTI' if kapi else 'BASARISIZ'}"})
    CSV.parent.mkdir(parents=True, exist_ok=True)
    alanlar = ["split", "ufuk", "seed", "kosum", "E_san", "E_naif", "E_son", "E_lb", "R",
               "sigma_hat_sifir_orani", "n_pencere", "o2_oran_val", "san_ckpt_istasyonla_ayni"]
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar)
        w.writeheader()
        for r in satirlar:
            w.writerow({k: (f"{r[k]:.8g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in alanlar})
    yaz(f"[H05 onkosul] sure {time.time() - t0:.1f}s | csv: {CSV}")
    yaz(f"KAPI={'GECTI' if kapi else 'BASARISIZ'}")
    _log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
