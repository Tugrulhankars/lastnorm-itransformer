#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula_esdeger.py -- H02 tek-degisken saglamasi (kart: duman oncesi, CPU, maks. fark < 1e-6).

Her ufuk (resmi .sh argumanlari) icin, ayni seed ile:
  (A) resmi iTransformer Model(args, use_norm=1)
  (B) RevINiTransformer(args, use_norm=0)  [gamma=1, beta=0 init]
kurulur; (1) omurga agirliklari birebir ayni mi (ayni RNG tuketimi), (2) gercek train verisinden bir
batch (bs 32) ile eval modunda ve (3) train modunda (dropout; her ileri geciste ayni seed) cikislarin
maks. mutlak farki < 1e-6 mi, (4) parametre farki tam 2*enc_in (=16) mi. Gecmezse rc=1 -> kart kosulmaz.
Test verisi kullanilmaz.
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos  # noqa: E402

sys.path.insert(0, str(rakip_kos.REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ESIK = 1e-6
SEED = 2023


def tohum(s):
    random.seed(s); torch.manual_seed(s); np.random.seed(s)


def main():
    from model import iTransformer
    from model_revin import RevINiTransformer
    from data_provider.data_factory import data_dict

    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    ok = True
    for H in rakip_kos.UFUKLAR:
        argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN) + "/")
        argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
        a1 = parser.parse_args(argv)
        a0 = parser.parse_args(rakip_kos.argv_degistir(argv, "--use_norm", "0"))
        assert int(a1.use_norm) == 1 and int(a0.use_norm) == 0

        tohum(SEED); A = iTransformer.Model(a1).float()
        tohum(SEED); B = RevINiTransformer(a0).float()
        sa, sb = A.state_dict(), B.backbone.state_dict()
        agirlik_ayni = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        p_fark = sum(p.numel() for p in B.parameters()) - sum(p.numel() for p in A.parameters())

        ds = data_dict[a1.data](root_path=a1.root_path, data_path=a1.data_path, flag="train",
                                size=[a1.seq_len, a1.label_len, a1.pred_len], features=a1.features,
                                target=a1.target, timeenc=1, freq=a1.freq)
        idx = np.linspace(0, len(ds) - 1, 32).astype(int)
        bx, by, bxm, bym = [torch.tensor(np.stack([ds[i][k] for i in idx])).float() for k in range(4)]
        dec = torch.cat([by[:, :a1.label_len, :], torch.zeros_like(by[:, -a1.pred_len:, :])], dim=1)

        A.eval(); B.eval()
        with torch.no_grad():
            fe = float((A(bx, bxm, dec, bym) - B(bx, bxm, dec, bym)).abs().max())
        A.train(); B.train()
        with torch.no_grad():
            tohum(7); ya = A(bx, bxm, dec, bym)
            tohum(7); yb = B(bx, bxm, dec, bym)
            ft = float((ya - yb).abs().max())
        gecti = agirlik_ayni and p_fark == 2 * a1.enc_in and fe < ESIK and ft < ESIK
        ok &= gecti
        print(f"h{H}: omurga_agirlik_ayni={agirlik_ayni} param_fark={p_fark} maks_fark_eval={fe:.3e} "
              f"maks_fark_train={ft:.3e} -> {'GECTI' if gecti else 'KALDI'}")
    print("SONUC:", "GECTI (max fark < 1e-6, tek-degisken korunuyor)" if ok else "KALDI -> kart kosulmaz")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
