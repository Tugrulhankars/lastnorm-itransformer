#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kapi.py -- H11 (san-softplus) ON-KOSUL KAPISI (CPU, CUDA_VISIBLE_DEVICES=-1; yalniz train/val; TEST YUKLEYICISI
KURULMAZ). Kart: hipotezler/H11.md "On-kosul kapisi" satiri. Esikler karttan SABIT KODLANDI.

Ne: yalniz SAN istasyon on-egitimi, 5 epoch, resmi erken durdurma (EarlyStopping patience=args.patience, en iyi
istasyon checkpoint'i yuklenir) -- exp_san.ExpSAN.train()'in ilk P=5 epoch'unun birebir kopyasi (ayni Exp nesnesi,
ayni _get_data / _select_optimizer / station_loss / vali_san / adjust_learning_rate; test _get_data cagrisi YOK, test
yukleyicisi on-egitimde hic iterate edilmedigi icin RNG akisi degismez). Kat K3 ve K1 (rakip_kos.kat_veri_sinifi),
seed 2023, 4 ufuk, iki kol: relu (H01 SANiTransformer) ve softplus (H11 SANSPiTransformer).

Olcumler (kat val pencereleri: tum pencereler, adim 1; tum dilimler x degiskenler; z uzayi = kat train scaler'i):
  o_H = #{sigma_hat < 1e-3} / hucre sayisi
  q_H = medyan(sigma_hat / sigma_gercek_dilim)   (sigma_gercek_dilim = torch.std(hedef 6'lik dilim, unbiased),
        station_loss ile ayni tanim; sigma_gercek_dilim == 0 olan hucreler orandan cikarilir, sayisi raporlanir)
  R_H = E_san / E_naif  (H05 tanimi: E_san = MSE(mu_hat_k, mu_k), E_naif = MSE(son 6 girdi ort., mu_k); bilgi)
KAPI:
  G1: softplus kolunda o_H <= 0.05, her iki katta 4/4 ufuk
  G2: softplus kolunda q_H in [0.5, 2.0], her katta >= 3/4 ufuk
  G3: relu kolunda K3'te o_H >= 0.80, >= 3/4 ufuk (H01 teshisinin yeniden uretimi; basarisizsa INCELE, kosum yok)
  KAPI = G1 ve G2 ve G3. Son satir: KAPI=GECTI | KAPI=BASARISIZ
Cikti: varyantlar/san-softplus/kapi.log (karttaki ad) + ayni icerik onkosul.log (orkestrator istegi) +
sonuclar/h11-kapi.csv
Kullanim (WS kokunden): env/.venv/Scripts/python.exe varyantlar/san-softplus/kapi.py
"""
import csv
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # bos deger Windows'ta CUDA'yi gizlemiyor (gunluk 05:02)

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
sys.path.insert(0, str(WS / "varyantlar" / "san"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# --- karttan sabit esikler (H11.md) ---
SEED = 2023
KATLAR = [3, 1]
UFUKLAR = [96, 192, 336, 720]
KOLLAR = ["relu", "softplus"]
OLU_ESIK = 1e-3        # sigma_hat < 1e-3 -> olu hucre
G1_OLU_UST = 0.05      # softplus o_H <= 0.05, 2 kat x 4/4 ufuk
G2_Q_ALT, G2_Q_UST, G2_MIN_UFUK = 0.5, 2.0, 3
G3_OLU_ALT, G3_MIN_UFUK = 0.80, 3
P = 6                  # SAN period_len

LOG = HERE / "kapi.log"
LOG2 = HERE / "onkosul.log"
CSV = WS / "sonuclar" / "h11-kapi.csv"
_log = open(LOG, "w", encoding="utf-8")


def yaz(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log.write(s + "\n")
    _log.flush()


def main():
    t0 = time.time()
    assert not torch.cuda.is_available(), "CPU kapisi: CUDA gorunur olmamali"
    import exp_san
    from data_provider.data_factory import data_dict
    from utils.tools import EarlyStopping
    from model_san import SANiTransformer
    from model_san_sp import SANSPiTransformer

    Taban = data_dict["custom"]
    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    gecici = Path(tempfile.mkdtemp(prefix="h11kapi_"))
    yaz(f"[H11 kapi] {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | cihaz CPU "
        f"(cuda gorunur={torch.cuda.is_available()}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}) "
        f"| seed {SEED} | katlar {KATLAR} | kollar {KOLLAR}")
    yaz(f"[H11 kapi] esikler: G1 softplus o_H<={G1_OLU_UST} (2 kat x 4/4); G2 softplus q_H in [{G2_Q_ALT},{G2_Q_UST}] "
        f"(her kat >={G2_MIN_UFUK}/4); G3 relu K3 o_H>={G3_OLU_ALT} (>={G3_MIN_UFUK}/4); olu: sigma_hat<{OLU_ESIK}")

    sonuc = {}
    satirlar = []
    for kat in KATLAR:
        data_dict["custom"] = rakip_kos.kat_veri_sinifi(Taban, "custom", kat)
        for H in UFUKLAR:
            argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN) + os.sep)
            argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
            argv = rakip_kos.argv_degistir(argv, "--use_norm", "0")
            argv = rakip_kos.argv_degistir(argv, "--checkpoints", str(gecici) + os.sep)
            args = parser.parse_args(argv)
            args.use_gpu = False
            args.num_workers = 0
            C = args.enc_in
            for kol in KOLLAR:
                Model = SANiTransformer if kol == "relu" else SANSPiTransformer

                class ExpKapi(exp_san.ExpSAN):
                    def _build_model(self):
                        return Model(self.args, period_len=exp_san.PERIOD_LEN,
                                     station_type=exp_san.STATION_TYPE).float()

                # rakip_kos.main() ile ayni tohumlama: Exp'ten hemen once
                random.seed(SEED)
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                exp = ExpKapi(args)
                fa = type(exp.model.san.model_std.final_activation).__name__
                train_data, train_loader = exp._get_data(flag='train')
                vali_data, vali_loader = exp._get_data(flag='val')
                # flag='test' KURULMAZ (kart)
                path_station = gecici / f"k{kat}_h{H}_{kol}" / "station"
                path_station.mkdir(parents=True, exist_ok=True)
                es = EarlyStopping(patience=args.patience, verbose=False)
                model_optim = exp._select_optimizer()
                exp.criterion = exp._select_criterion()
                criterion = exp.criterion
                Pn = exp.station_pretrain_epoch
                gecmis = []
                for epoch in range(Pn):  # ExpSAN.train, epoch + 1 <= P dali (birebir)
                    train_loss = []
                    exp.model.train()
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        model_optim.zero_grad()
                        batch_x = batch_x.float().to(exp.device)
                        batch_y = batch_y.float().to(exp.device)
                        _, statistics_pred = exp.model.san.normalize(batch_x)
                        y = batch_y[:, -args.pred_len:, :]
                        loss = exp.station_loss(y, statistics_pred)
                        train_loss.append(loss.item())
                        loss.backward()
                        exp.station_optim.step()
                        model_optim.zero_grad()
                        exp.station_optim.zero_grad()
                    train_loss = np.average(train_loss)
                    vali_loss = exp.vali_san(vali_loader, criterion, epoch)
                    gecmis.append((epoch + 1, float(train_loss), float(vali_loss)))
                    es(vali_loss, exp.model.san, str(path_station))
                    exp_san.adjust_learning_rate(exp.station_optim, epoch + 1, args, exp_san.STATION_LR)
                exp.model.san.load_state_dict(torch.load(path_station / "checkpoint.pth", map_location="cpu"))
                en_iyi = min(gecmis, key=lambda t: t[2])

                # --- olcum: kat val, tum pencereler
                exp.model.eval()
                with torch.no_grad():
                    X = torch.from_numpy(np.stack([vali_data[i][0] for i in range(len(vali_data))])).float()
                    Y = torch.from_numpy(np.stack([vali_data[i][1] for i in range(len(vali_data))])[:, -H:, :]).float()
                    _, st = exp.model.san.normalize(X)
                    mu_hat, sig_hat = st[..., :C].double(), st[..., C:].double()
                    Yd = Y.reshape(Y.shape[0], -1, P, C)
                    mu_k = Yd.mean(2).double()
                    sig_k = torch.std(Yd, dim=2).double()
                    o = float((sig_hat < OLU_ESIK).double().mean())
                    poz = sig_k > 0
                    oran = (sig_hat[poz] / sig_k[poz]).numpy()
                    q = float(np.median(oran))
                    mu_son = X.double()[:, -P:, :].mean(1, keepdim=True)
                    e_san = float(((mu_hat - mu_k) ** 2).mean())
                    e_naif = float(((mu_son - mu_k) ** 2).mean())
                    R = e_san / e_naif
                    sig_ort, sig_g_ort = float(sig_hat.mean()), float(sig_k.mean())
                d = {"kat": kat, "ufuk": H, "kol": kol, "final_activation": fa, "seed": SEED,
                     "n_pencere": int(X.shape[0]), "n_hucre": int(sig_hat.numel()),
                     "n_sigma_gercek_sifir": int((~poz).sum()), "olu_orani": o, "q": q, "R": R,
                     "E_san": e_san, "E_naif": e_naif, "sigma_hat_ort": sig_ort, "sigma_gercek_ort": sig_g_ort,
                     "onegitim_en_iyi_ep": en_iyi[0], "onegitim_val_min": en_iyi[2],
                     "onegitim_gecmis": ";".join(f"{e}:{tl:.6f}/{vl:.6f}" for e, tl, vl in gecmis)}
                sonuc[(kat, H, kol)] = d
                satirlar.append(d)
                yaz(f"  K{kat} h{H} {kol:8s} ({fa}): o_H={o:.4f} q_H={q:.4f} R_H={R:.4f} | sigma_hat ort {sig_ort:.4f} "
                    f"gercek ort {sig_g_ort:.4f} | n_pencere {d['n_pencere']} hucre {d['n_hucre']} "
                    f"(sigma_gercek=0: {d['n_sigma_gercek_sifir']}) | on-egitim en iyi ep {en_iyi[0]} "
                    f"val {en_iyi[2]:.6f} | {time.time() - t0:.0f}s")
    data_dict["custom"] = Taban
    shutil.rmtree(gecici, ignore_errors=True)

    # --- kapi
    yaz("")
    yaz("OZET (kat | kol | o_H h96/h192/h336/h720 | q_H | R_H):")
    for kat in KATLAR:
        for kol in KOLLAR:
            yaz(f"K{kat} | {kol} | " + "/".join(f"{sonuc[(kat, H, kol)]['olu_orani']:.4f}" for H in UFUKLAR) + " | "
                + "/".join(f"{sonuc[(kat, H, kol)]['q']:.4f}" for H in UFUKLAR) + " | "
                + "/".join(f"{sonuc[(kat, H, kol)]['R']:.4f}" for H in UFUKLAR))
    g1_say = {k: sum(sonuc[(k, H, 'softplus')]['olu_orani'] <= G1_OLU_UST for H in UFUKLAR) for k in KATLAR}
    g1 = all(g1_say[k] == len(UFUKLAR) for k in KATLAR)
    g2_say = {k: sum(G2_Q_ALT <= sonuc[(k, H, 'softplus')]['q'] <= G2_Q_UST for H in UFUKLAR) for k in KATLAR}
    g2 = all(g2_say[k] >= G2_MIN_UFUK for k in KATLAR)
    g3_say = sum(sonuc[(3, H, 'relu')]['olu_orani'] >= G3_OLU_ALT for H in UFUKLAR)
    g3 = g3_say >= G3_MIN_UFUK
    kapi = g1 and g2 and g3
    yaz("")
    yaz(f"G1 (softplus o_H<={G1_OLU_UST}, 4/4 her katta): " + ", ".join(f"K{k} {g1_say[k]}/4" for k in KATLAR)
        + f" -> {'GECER' if g1 else 'BASARISIZ'}")
    yaz(f"G2 (softplus q_H in [{G2_Q_ALT},{G2_Q_UST}], >={G2_MIN_UFUK}/4 her katta): "
        + ", ".join(f"K{k} {g2_say[k]}/4" for k in KATLAR) + f" -> {'GECER' if g2 else 'BASARISIZ'}")
    yaz(f"G3 (relu K3 o_H>={G3_OLU_ALT}, >={G3_MIN_UFUK}/4): {g3_say}/4 -> "
        + ("GECER (H01 teshisi yeniden uretildi)" if g3 else "BASARISIZ -> INCELE (olcum kodu supheli; kosum yok)"))
    alanlar = ["kat", "ufuk", "kol", "final_activation", "seed", "n_pencere", "n_hucre", "n_sigma_gercek_sifir",
               "olu_orani", "q", "R", "E_san", "E_naif", "sigma_hat_ort", "sigma_gercek_ort", "onegitim_en_iyi_ep",
               "onegitim_val_min", "onegitim_gecmis"]
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar + ["kapi"])
        w.writeheader()
        for r in satirlar:
            w.writerow({k: (f"{r[k]:.8g}" if isinstance(r[k], float) else r[k]) for k in alanlar} | {"kapi": ""})
        w.writerow({k: "" for k in alanlar} | {
            "kol": "KAPI", "kapi": f"G1={'GECER' if g1 else 'BASARISIZ'};G2={'GECER' if g2 else 'BASARISIZ'};"
                                   f"G3={'GECER' if g3 else 'BASARISIZ'};KAPI={'GECTI' if kapi else 'BASARISIZ'}"})
    yaz(f"[H11 kapi] sure {time.time() - t0:.1f}s | csv: {CSV}")
    yaz(f"KAPI={'GECTI' if kapi else 'BASARISIZ'}")
    _log.close()
    shutil.copyfile(LOG, LOG2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
