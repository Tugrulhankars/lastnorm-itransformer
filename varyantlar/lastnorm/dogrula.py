#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula.py -- H04 duman oncesi CPU saglamasi (kart uygulama notu (i)-(iv)). Biri bile KALIRSA rc=1 -> kart kosulmaz.

Her ufuk (resmi .sh argumanlari) icin, ayni seed ile:
 (i)  LastNormiTransformer(use_norm=0).backbone ilk agirliklari == resmi Model(use_norm=1) (state_dict, maks. fark 0)
      ve parametre sayisi ebeveynle esit (224 224 / 236 608 / 255 184 / 304 720);
 (ii) sigma_lb, resmi formulle (model/iTransformer.py forecast(), yerinde islemler dahil) ayni batch'te maks. fark 0;
 (iii) projektor agirlik+bias sifirlanmis kopyada cikis == x_enc[:, -1, :] tekrari (maks. fark < 1e-6; f=0 -> Repeat);
 (iv) BILGI: x_son girdi penceresinin son satiri; batch_x[:, -1] ile batch_y[:, label_len-1] esitligi basilir
      (resmi label_len=48 ortusmesi; hedef penceresinden alinmadigini gosterir).
Veri: yalniz train (bs 32, esit aralikli pencereler). Test verisi kullanilmaz. Cikti stdout + dogrula.log.
"""
import copy
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SEED = 2023
PARAM_BEKLENEN = {96: 224224, 192: 236608, 336: 255184, 720: 304720}
_log = open(HERE / "dogrula.log", "w", encoding="utf-8")


def yaz(s):
    print(s, flush=True)
    _log.write(s + "\n")
    _log.flush()


def tohum(s):
    random.seed(s); torch.manual_seed(s); np.random.seed(s)


def resmi_stdev(x_enc):
    # model/iTransformer.py forecast() use_norm blogu, birebir
    x_enc = x_enc.clone()
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return stdev


def main():
    from model import iTransformer
    from model_lastnorm import LastNormiTransformer
    from data_provider.data_factory import data_dict

    torch.set_grad_enabled(False)
    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    ok = True
    yaz(f"[H04 dogrula] torch {torch.__version__} | cihaz CPU (cuda gorunur={torch.cuda.is_available()}) | seed {SEED}")
    for H in rakip_kos.UFUKLAR:
        argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN) + os.sep)
        argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
        a1 = parser.parse_args(argv)
        a0 = parser.parse_args(rakip_kos.argv_degistir(argv, "--use_norm", "0"))
        assert int(a1.use_norm) == 1 and int(a0.use_norm) == 0

        tohum(SEED); A = iTransformer.Model(a1).float()
        tohum(SEED); B = LastNormiTransformer(a0).float()
        sa, sb = A.state_dict(), B.backbone.state_dict()
        ayni_anahtar = sa.keys() == sb.keys()
        maks_w = max(float((sa[k] - sb[k]).abs().max()) for k in sa) if ayni_anahtar else float("inf")
        nA = sum(p.numel() for p in A.parameters()); nB = sum(p.numel() for p in B.parameters())
        g1 = ayni_anahtar and maks_w == 0.0 and nA == nB == PARAM_BEKLENEN[H]

        ds = data_dict[a1.data](root_path=a1.root_path, data_path=a1.data_path, flag="train",
                                size=[a1.seq_len, a1.label_len, a1.pred_len], features=a1.features,
                                target=a1.target, timeenc=1, freq=a1.freq)
        idx = np.linspace(0, len(ds) - 1, 32).astype(int)
        bx, by, bxm, bym = [torch.tensor(np.stack([ds[i][k] for i in idx])).float() for k in range(4)]
        dec = torch.cat([by[:, :a1.label_len, :], torch.zeros_like(by[:, -a1.pred_len:, :])], dim=1)

        _, st_b, last = B.istatistik(bx)
        maks_sig = float((st_b - resmi_stdev(bx)).abs().max())
        g2 = maks_sig == 0.0

        C = copy.deepcopy(B)
        C.backbone.projector.weight.zero_(); C.backbone.projector.bias.zero_()
        C.eval()
        yC = C(bx, bxm, dec, bym)
        rep = bx[:, -1:, :].repeat(1, a1.pred_len, 1)
        maks_rep = float((yC - rep).abs().max())
        g3 = maks_rep < 1e-6 and tuple(yC.shape) == (32, a1.pred_len, a1.enc_in)

        bilgi_iv = float((bx[:, -1, :] - by[:, a1.label_len - 1, :]).abs().max())
        bilgi_hedef = float((bx[:, -1, :] - by[:, -a1.pred_len, :]).abs().max())
        last_girdi = float((last[:, 0, :] - bx[:, -1, :]).abs().max())

        gecti = g1 and g2 and g3 and last_girdi == 0.0
        ok &= gecti
        yaz(f"h{H}: (i) agirlik maks.fark={maks_w:.1e} param resmi={nA} varyant={nB} beklenen={PARAM_BEKLENEN[H]} -> "
            f"{'GECTI' if g1 else 'KALDI'} | (ii) sigma maks.fark={maks_sig:.1e} -> {'GECTI' if g2 else 'KALDI'} | "
            f"(iii) f=0 cikis vs Repeat maks.fark={maks_rep:.2e} -> {'GECTI' if g3 else 'KALDI'} | "
            f"(iv) BILGI: x_son==batch_x[:,-1] fark={last_girdi:.1e}; batch_x[:,-1] vs batch_y[:,label_len-1] "
            f"maks.fark={bilgi_iv:.1e} (ortusme); vs ilk hedef adimi batch_y[:,-pred_len] maks.fark={bilgi_hedef:.3f}")
    yaz("SONUC: " + ("GECTI (i-iii saglandi; kart kosulabilir)" if ok else "KALDI -> kart kosulmaz"))
    _log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
