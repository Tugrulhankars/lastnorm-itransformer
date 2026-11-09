#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula.py -- H07 duman oncesi CPU saglamasi (kart uygulama notu (i)-(iii)). Biri bile KALIRSA rc=1 -> kart kosulmaz.

Her ufuk (resmi .sh argumanlari, use_norm=1 AYNEN) icin, ayni seed ile:
 (i)   XmarkYokiTransformer.backbone ilk agirliklari == resmi Model (state_dict, maks. fark 0) ve parametre sayisi
       esit (224 224 / 236 608 / 255 184 / 304 720);
 (ii)  ayni batch'te cikis sekli [B, pred_len, 8]; kodlayici token sayisi 8 (enc_embedding cikisina hook);
       (bilgi: resmi modelde 12)
 (iii) ayni agirliklarla x_mark verilen (resmi) ve verilmeyen (varyant) cikislar FARKLI (yama devrede).
Veri: yalniz train (bs 32). Test verisi kullanilmaz. Cikti stdout + dogrula.log.
"""
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


def main():
    from model import iTransformer
    from model_xmark import XmarkYokiTransformer
    from data_provider.data_factory import data_dict

    torch.set_grad_enabled(False)
    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    ok = True
    yaz(f"[H07 dogrula] torch {torch.__version__} | cihaz CPU (cuda gorunur={torch.cuda.is_available()}) | seed {SEED}")
    for H in rakip_kos.UFUKLAR:
        argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN) + os.sep)
        argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
        a = parser.parse_args(argv)
        assert int(a.use_norm) == 1

        tohum(SEED); A = iTransformer.Model(a).float()
        tohum(SEED); B = XmarkYokiTransformer(a).float()
        sa, sb = A.state_dict(), B.backbone.state_dict()
        ayni_anahtar = sa.keys() == sb.keys()
        maks_w = max(float((sa[k] - sb[k]).abs().max()) for k in sa) if ayni_anahtar else float("inf")
        nA = sum(p.numel() for p in A.parameters()); nB = sum(p.numel() for p in B.parameters())
        g1 = ayni_anahtar and maks_w == 0.0 and nA == nB == PARAM_BEKLENEN[H]

        ds = data_dict[a.data](root_path=a.root_path, data_path=a.data_path, flag="train",
                               size=[a.seq_len, a.label_len, a.pred_len], features=a.features,
                               target=a.target, timeenc=1, freq=a.freq)
        idx = np.linspace(0, len(ds) - 1, 32).astype(int)
        bx, by, bxm, bym = [torch.tensor(np.stack([ds[i][k] for i in idx])).float() for k in range(4)]
        dec = torch.cat([by[:, :a.label_len, :], torch.zeros_like(by[:, -a.pred_len:, :])], dim=1)

        tok = {}
        hA = A.enc_embedding.register_forward_hook(lambda m, i, o: tok.__setitem__("resmi", o.shape[1]))
        hB = B.backbone.enc_embedding.register_forward_hook(lambda m, i, o: tok.__setitem__("varyant", o.shape[1]))
        A.eval(); B.eval()
        yA = A(bx.clone(), bxm, dec, bym)
        yB = B(bx.clone(), bxm, dec, bym)
        hA.remove(); hB.remove()
        g2 = tuple(yB.shape) == (32, a.pred_len, 8) and tok.get("varyant") == 8
        fark = float((yA - yB).abs().max())
        g3 = fark > 0.0
        gecti = g1 and g2 and g3
        ok &= gecti
        yaz(f"h{H}: (i) agirlik maks.fark={maks_w:.1e} param resmi={nA} varyant={nB} beklenen={PARAM_BEKLENEN[H]} -> "
            f"{'GECTI' if g1 else 'KALDI'} | (ii) cikis {tuple(yB.shape)} token varyant={tok.get('varyant')} "
            f"(resmi={tok.get('resmi')}; x_mark kanal={bxm.shape[-1]}) -> {'GECTI' if g2 else 'KALDI'} | "
            f"(iii) x_mark'li vs x_mark'siz cikis maks.fark={fark:.3e} -> {'GECTI (farkli)' if g3 else 'KALDI (ayni)'}")
    yaz("SONUC: " + ("GECTI (i-iii saglandi; kart kosulabilir)" if ok else "KALDI -> kart kosulmaz"))
    _log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
