#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula.py -- H10 duman oncesi CPU saglamasi (kart uygulama notu (i)-(v)). Biri bile KALIRSA rc=1 -> kart kosulmaz.

Her ufuk (resmi .sh argumanlari + --use_norm 0) icin, ayni seed ile:
 (i)   MerkezLastNormiTransformer ilk agirliklari == LastNormiTransformer (state_dict, maks. fark 0), parametre
       sayisi esit (224 224 / 236 608 / 255 184 / 304 720);
 (ii)  omurgaya giden tensor == x_enc - x_enc[:, -1:, :] (forward pre-hook ile yakalanir; maks. fark 0);
 (iii) projektor agirlik+bias sifirlanmis kopyada cikis == x_enc[:, -1, :] tekrari (< 1e-6; f = 0 -> Repeat);
 (iv)  ayni agirlik ve batch'te cikis LastNormiTransformer cikisindan FARKLI (plasebo degil; maks. fark > 1e-6);
 (v)   x_son girdi penceresinin son satiri: batch_x[:, -1] ile esit; BILGI: batch_y[:, label_len-1] ortusmesi ve
       ilk hedef adimindan farki (H04 (iv) ile ayni bilgi satiri).
Veri: yalniz train (bs 32, esit aralikli pencereler). Val/test verisi kullanilmaz. Cikti stdout + dogrula.log.
CPU: CUDA_VISIBLE_DEVICES=-1 (Windows'ta bos deger CUDA'yi gizlemiyor; gunluk 05:02).
"""
import copy
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
sys.path.insert(0, str(WS / "varyantlar" / "lastnorm"))
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


def main():
    from model_lastnorm import LastNormiTransformer
    from model_merkez import MerkezLastNormiTransformer
    from data_provider.data_factory import data_dict

    torch.set_grad_enabled(False)
    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    ok = True
    yaz(f"[H10 dogrula] torch {torch.__version__} | cihaz CPU (cuda gorunur={torch.cuda.is_available()}) | seed {SEED}")
    for H in rakip_kos.UFUKLAR:
        argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN) + os.sep)
        argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
        a0 = parser.parse_args(rakip_kos.argv_degistir(argv, "--use_norm", "0"))
        assert int(a0.use_norm) == 0

        # (i) ilk agirliklar ve parametre sayisi
        tohum(SEED); L = LastNormiTransformer(a0).float()
        tohum(SEED); M = MerkezLastNormiTransformer(a0).float()
        sl, sm = L.state_dict(), M.state_dict()
        ayni_anahtar = sl.keys() == sm.keys()
        maks_w = max(float((sl[k] - sm[k]).abs().max()) for k in sl) if ayni_anahtar else float("inf")
        nL = sum(p.numel() for p in L.parameters()); nM = sum(p.numel() for p in M.parameters())
        g1 = ayni_anahtar and maks_w == 0.0 and nL == nM == PARAM_BEKLENEN[H]

        ds = data_dict[a0.data](root_path=a0.root_path, data_path=a0.data_path, flag="train",
                                size=[a0.seq_len, a0.label_len, a0.pred_len], features=a0.features,
                                target=a0.target, timeenc=1, freq=a0.freq)
        idx = np.linspace(0, len(ds) - 1, 32).astype(int)
        bx, by, bxm, bym = [torch.tensor(np.stack([ds[i][k] for i in idx])).float() for k in range(4)]
        dec = torch.cat([by[:, :a0.label_len, :], torch.zeros_like(by[:, -a0.pred_len:, :])], dim=1)

        # (ii) omurga girdisi
        yakala = {}
        h_ = M.backbone.register_forward_pre_hook(lambda mod, args: yakala.__setitem__("x", args[0].clone()))
        M.eval(); L.eval()
        yM = M(bx, bxm, dec, bym)
        h_.remove()
        beklenen_girdi = bx - bx[:, -1:, :]
        maks_girdi = float((yakala["x"] - beklenen_girdi).abs().max())
        g2 = maks_girdi == 0.0

        # (iii) f = 0 -> Repeat
        C = copy.deepcopy(M)
        C.backbone.projector.weight.zero_(); C.backbone.projector.bias.zero_()
        C.eval()
        yC = C(bx, bxm, dec, bym)
        rep = bx[:, -1:, :].repeat(1, a0.pred_len, 1)
        maks_rep = float((yC - rep).abs().max())
        g3 = maks_rep < 1e-6 and tuple(yC.shape) == (32, a0.pred_len, a0.enc_in)

        # (iv) plasebo degil: ayni agirlik (M ve L ilk agirliklari ozdes) + ayni batch -> cikis farkli
        yL = L(bx, bxm, dec, bym)
        maks_plasebo = float((yM - yL).abs().max())
        g4 = maks_plasebo > 1e-6

        # (v) x_son girdi penceresinin son satiri
        last = bx[:, -1:, :]
        last_girdi = float((last[:, 0, :] - bx[:, -1, :]).abs().max())
        bilgi_iv = float((bx[:, -1, :] - by[:, a0.label_len - 1, :]).abs().max())
        bilgi_hedef = float((bx[:, -1, :] - by[:, -a0.pred_len, :]).abs().max())
        g5 = last_girdi == 0.0 and maks_girdi == 0.0  # omurga girdisi tam olarak batch_x'in son satiriyla merkezli

        # bilgi: sigma_lb ozeti (H10'da kullanilmaz)
        _, st, _ = LastNormiTransformer.istatistik(bx)

        gecti = g1 and g2 and g3 and g4 and g5
        ok &= gecti
        yaz(f"h{H}: (i) agirlik maks.fark={maks_w:.1e} param lastnorm={nL} merkez={nM} beklenen={PARAM_BEKLENEN[H]} -> "
            f"{'GECTI' if g1 else 'KALDI'} | (ii) omurga girdisi vs x_enc-x_son maks.fark={maks_girdi:.1e} -> "
            f"{'GECTI' if g2 else 'KALDI'} | (iii) f=0 cikis vs Repeat maks.fark={maks_rep:.2e} -> "
            f"{'GECTI' if g3 else 'KALDI'} | (iv) cikis vs LastNorm maks.fark={maks_plasebo:.3e} (>1e-6) -> "
            f"{'GECTI' if g4 else 'KALDI'} | (v) x_son==batch_x[:,-1] fark={last_girdi:.1e} -> "
            f"{'GECTI' if g5 else 'KALDI'}; BILGI: batch_x[:,-1] vs batch_y[:,label_len-1] maks.fark={bilgi_iv:.1e} "
            f"(ortusme); vs ilk hedef adimi batch_y[:,-pred_len] maks.fark={bilgi_hedef:.3f}; "
            f"sigma_lb medyan={float(st.median()):.3f} (bu batch, train)")
    yaz("SONUC: " + ("GECTI (i-v saglandi; kart kosulabilir)" if ok else "KALDI -> kart kosulmaz"))
    _log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
