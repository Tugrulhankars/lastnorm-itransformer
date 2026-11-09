#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula.py -- H09 duman oncesi CPU saglamasi (kart uygulama notu (i)-(v)). Biri bile KALIRSA rc=1 -> kart kosulmaz.

 (i)   dort ufukta N_H = 5120 / 5024 / 4880 / 4496; grup sinirlari np.array_split ile birebir; gruplar indekste monoton;
       grup atamasina giren veri kumesi flag='train' (set_type 0); exp_dro.py'de IndeksliVeri yalniz train_data ile
       kuruluyor (kaynak metin denetimi);
 (ii)  IndeksliVeri tensorleri orijinalle maks. fark 0 (orneklem); ayni seed'le ilk 3 batch'in sirasi orijinal
       DataLoader (data_provider argumanlari) ile ayni ve RNG durumu 3 batch sonra ayni;
 (iii) q uniform (eta = 0) ve gruplara esit sayida ornek dusen yapay batch'te DRO kaybi == nn.MSELoss (fark < 1e-7;
       float64); BILGI: float32 farki ve eta = 0.01 ile tek adim farki;
 (iv)  q guncellemesi: pozitif, toplam 1 (fark < 1e-6), bos grubun agirligi goreli olarak azaliyor (100 adim dahil);
 (v)   ayni seed'de ExpLastNormDRO (lastnorm_kos.exp_sinifi_kur + exp_dro, CPU) modelinin ilk agirliklari
       LastNormiTransformer ile ozdes (maks. fark 0), parametre sayisi esit; ExpLastNormDRO yalniz `train`'i override ediyor;
 (vi)  BILGI (uygulama notu (e) icin): shuffle'siz bir DataLoader'in TAM turu ile yalniz iter() olusturulmasi ayni torch
       RNG durumunu birakir; hic dokunulmamasi FARKLI birakir -> exp_dro'daki `iter(test_loader)` resmi RNG akisini korur.
       (Vekil olarak train veri kumesi shuffle'siz kullanilir; val/test verisi okunmaz.)
Veri: yalniz train. Cikti stdout + dogrula.log. CPU: CUDA_VISIBLE_DEVICES=-1.
"""
import importlib.util
import inspect
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
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

SEED = 2023
PARAM_BEKLENEN = {96: 224224, 192: 236608, 336: 255184, 720: 304720}
N_BEKLENEN = {96: 5120, 192: 5024, 336: 4880, 720: 4496}
_log = open(HERE / "dogrula.log", "w", encoding="utf-8")


def yaz(s):
    print(s, flush=True)
    _log.write(s + "\n")
    _log.flush()


def tohum(s):
    random.seed(s); torch.manual_seed(s); np.random.seed(s)


def main():
    import exp_dro
    from exp_dro import IndeksliVeri, grup_ata, dro_kayip, DRO_K
    from model_lastnorm import LastNormiTransformer
    from data_provider.data_factory import data_dict

    spec = importlib.util.spec_from_file_location("lastnorm_kos", str(WS / "varyantlar" / "lastnorm" / "kos.py"))
    LK = importlib.util.module_from_spec(spec); sys.modules["lastnorm_kos"] = LK; spec.loader.exec_module(LK)
    ExpLastNorm = LK.exp_sinifi_kur()
    ExpDRO = exp_dro.dro_sinifi_kur(ExpLastNorm)

    parser, _ = rakip_kos.resmi_parser()
    sh = rakip_kos.resmi_sh_argumanlari()
    ok = True
    yaz(f"[H09 dogrula] torch {torch.__version__} | cihaz CPU (cuda gorunur={torch.cuda.is_available()}) | seed {SEED}"
        f" | K={exp_dro.DRO_K} eta={exp_dro.DRO_ETA}")

    # (i-b) kaynak metin: IndeksliVeri yalniz train_data ile
    kaynak = inspect.getsource(exp_dro.dro_sinifi_kur)
    kaynak_ok = kaynak.count("IndeksliVeri(") == 1 and "IndeksliVeri(train_data)" in kaynak
    override = sorted(k for k, v in ExpDRO.__dict__.items() if callable(v) and not k.startswith("__"))
    for H in rakip_kos.UFUKLAR:
        argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN) + os.sep)
        argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
        a0 = parser.parse_args(rakip_kos.argv_degistir(argv, "--use_norm", "0"))
        a0.use_gpu = False
        a0.num_workers = 0
        ds = data_dict[a0.data](root_path=a0.root_path, data_path=a0.data_path, flag="train",
                                size=[a0.seq_len, a0.label_len, a0.pred_len], features=a0.features,
                                target=a0.target, timeenc=1, freq=a0.freq)
        # (i)
        N = len(ds)
        g = grup_ata(np.arange(N), N)
        parcalar = np.array_split(np.arange(N), DRO_K)
        sinir_ok = all(np.array_equal(np.where(g == k)[0], parcalar[k]) for k in range(DRO_K))
        monoton = bool(np.all(np.diff(g) >= 0)) and set(g.tolist()) == set(range(DRO_K))
        g1 = N == N_BEKLENEN[H] and sinir_ok and monoton and ds.set_type == 0 and kaynak_ok
        boylar = [len(p) for p in parcalar]

        # (ii) sarmalayici ozdesligi + RNG/siralama
        iv = IndeksliVeri(ds)
        ornek = np.linspace(0, N - 1, 64).astype(int)
        maks_t = 0.0
        for i in ornek:
            o, w = ds[i], iv[i]
            maks_t = max(maks_t, max(float(np.abs(np.asarray(o[k]) - np.asarray(w[k])).max()) for k in range(4)))
            assert w[4] == i
        tohum(SEED)
        L0 = DataLoader(ds, batch_size=a0.batch_size, shuffle=True, num_workers=0, drop_last=True)
        b0 = []
        for j, b in enumerate(L0):
            b0.append(b[0].clone())
            if j == 2:
                break
        rng0 = torch.get_rng_state().clone()
        tohum(SEED)
        L1 = DataLoader(iv, batch_size=a0.batch_size, shuffle=True, num_workers=0, drop_last=True)
        b1, idx1 = [], []
        for j, b in enumerate(L1):
            b1.append(b[0].clone()); idx1.append(b[4].clone())
            if j == 2:
                break
        rng1 = torch.get_rng_state().clone()
        sira_ok = all(torch.equal(x, y) for x, y in zip(b0, b1))
        idx_ok = all(torch.equal(b1[j], torch.tensor(np.stack([ds[int(i)][0] for i in idx1[j]]))) for j in range(3))
        g2 = maks_t == 0.0 and sira_ok and idx_ok and torch.equal(rng0, rng1)

        # (iii) yapay batch: 32 ornek, her gruba 8
        gen = torch.Generator().manual_seed(SEED + H)
        o64 = torch.randn(32, a0.pred_len, 8, generator=gen, dtype=torch.float64)
        y64 = torch.randn(32, a0.pred_len, 8, generator=gen, dtype=torch.float64)
        grp = torch.arange(32) % DRO_K
        q0 = torch.full((DRO_K,), 1.0 / DRO_K, dtype=torch.float64)
        orn64 = ((o64 - y64) ** 2).mean(dim=(1, 2))
        l_dro, q_e0, _ = dro_kayip(orn64, grp, q0, eta=0.0)
        fark64 = abs(float(l_dro) - float(nn.MSELoss()(o64, y64)))
        o32, y32 = o64.float(), y64.float()
        l32, _, _ = dro_kayip(((o32 - y32) ** 2).mean(dim=(1, 2)), grp, q0.float(), eta=0.0)
        fark32 = abs(float(l32) - float(nn.MSELoss()(o32, y32)))
        l_eta, _, _ = dro_kayip(orn64, grp, q0, eta=exp_dro.DRO_ETA)
        fark_eta = abs(float(l_eta) - float(nn.MSELoss()(o64, y64)))
        g3 = fark64 < 1e-7 and torch.allclose(q_e0, q0)

        # (iv) q guncellemesi; grup 3 bos
        grp_b = torch.tensor([0, 1, 2] * 10 + [0, 1])
        orn_b = torch.rand(32, generator=gen, dtype=torch.float64) + 0.1
        q1 = q0.clone()
        q_poz = True; top_ok = True
        oran0 = float(q1[3] / q1[:3].mean())
        for _ in range(100):
            _, q1, ell = dro_kayip(orn_b, grp_b, q1)
            q_poz &= bool((q1 > 0).all()); top_ok &= abs(float(q1.sum()) - 1.0) < 1e-6
        oran1 = float(q1[3] / q1[:3].mean())
        _, q_tek, _ = dro_kayip(orn_b, grp_b, q0)
        g4 = q_poz and top_ok and float(q_tek[3]) < 0.25 and oran1 < oran0 and float(ell[3]) == 0.0

        # (v) ilk agirliklar: ExpLastNormDRO (gercek sinif zinciri, CPU) vs LastNormiTransformer
        tohum(SEED); L = LastNormiTransformer(a0).float()
        tohum(SEED); E = ExpDRO(a0)
        sl, se = L.state_dict(), E.model.state_dict()
        ayni = sl.keys() == se.keys()
        maks_w = max(float((sl[k] - se[k]).abs().max()) for k in sl) if ayni else float("inf")
        nL = sum(p.numel() for p in L.parameters()); nE = sum(p.numel() for p in E.model.parameters())
        g5 = ayni and maks_w == 0.0 and nL == nE == PARAM_BEKLENEN[H] and type(E.model) is LastNormiTransformer \
            and override == ["train"]

        # (vi) BILGI: RNG -- shuffle'siz loader tam tur == yalniz iter(); dokunmamak farkli
        Ls = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        tohum(SEED)
        for _b in Ls:
            pass
        r_tam = torch.get_rng_state().clone()
        tohum(SEED); _ = iter(Ls); r_iter = torch.get_rng_state().clone()
        tohum(SEED); r_hic = torch.get_rng_state().clone()
        bilgi6 = f"tam==iter {torch.equal(r_tam, r_iter)}, tam==hic {torch.equal(r_tam, r_hic)}"

        gecti = g1 and g2 and g3 and g4 and g5
        ok &= gecti
        yaz(f"h{H}: (i) N_H={N} (beklenen {N_BEKLENEN[H]}) grup boylari={boylar} array_split esit={sinir_ok} "
            f"monoton={monoton} set_type={ds.set_type} kaynak(IndeksliVeri yalniz train_data)={kaynak_ok} -> "
            f"{'GECTI' if g1 else 'KALDI'} | (ii) sarmalayici maks.fark={maks_t:.1e}, ilk 3 batch sirasi ayni={sira_ok}, "
            f"idx->veri ayni={idx_ok}, RNG ayni={torch.equal(rng0, rng1)} -> {'GECTI' if g2 else 'KALDI'} | "
            f"(iii) DRO(q uniform, eta 0) vs MSELoss fark64={fark64:.1e} (BILGI fark32={fark32:.1e}, eta 0.01 tek adim "
            f"fark={fark_eta:.2e}) -> {'GECTI' if g3 else 'KALDI'} | (iv) q>0={q_poz}, toplam 1={top_ok}, tek adim "
            f"q_bos={float(q_tek[3]):.6f}<0.25, 100 adim q_bos/ort(q_dolu) {oran0:.3f}->{oran1:.3f} -> "
            f"{'GECTI' if g4 else 'KALDI'} | (v) agirlik maks.fark={maks_w:.1e} param lastnorm={nL} dro={nE} "
            f"override={override} -> {'GECTI' if g5 else 'KALDI'} | (vi) BILGI RNG: {bilgi6}")
    yaz("SONUC: " + ("GECTI (i-v saglandi; kart kosulabilir)" if ok else "KALDI -> kart kosulmaz"))
    _log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
