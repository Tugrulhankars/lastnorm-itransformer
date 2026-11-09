#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""h16_seed_kanit.py -- FAZ III/H16: ufuk-basi surecin (rakip_kos --ufuk-basi --ufuklar H) resmi .sh ile ayni tohum
davranisinda oldugunun kod kaniti (CPU, GPU yok, rakip koduna dokunmaz).

(A) run.py statik (AST): __main__ blogu fix_seed atamasi + random.seed / torch.manual_seed / np.random.seed ile BASLAR;
    bu cagrilarla 'exp = Exp(args)' arasindaki tum cagrilar listelenir ve hicbirinin RNG tuketmedigi gosterilir.
    .sh her pred_len icin ayri 'python -u run.py' -> her ufuk tazece tohumlanmis surecte Exp(args).
(B) rakip_kos.py statik (AST): ufuk dongusunde ayni uc cagri, Exp_Long_Term_Forecast(args)'tan hemen once.
(C) CPU deneyi (ETTm1 .sh argumanlariyla resmi iTransformer.Model + train DataLoader karistirmasi):
    C1 taze surec gibi: tohum -> Model(h96) ilk agirlik ozeti + ilk 64 karistirma indeksi;
    C2 kirli durum: once h720 modeli kurulur ve bir karistirma cekilir (RNG tuketilir), sonra yeniden tohum -> Model(h96).
    C1 == C2 (bit-bit) => tek ufuklu surec, cok ufuklu surecteki o ufuk ve resmi tek-ufuk run.py sureci ayni RNG durumuyla
    Exp'e girer. (Egitimin kendisi GPU'da; cuDNN deterministik olmayan cekirdekler bu kanitin kapsami disinda -- resmi
    kodda da ayni.)
Kullanim (WS kokunden): env/.venv/Scripts/python.exe -I scripts/h16_seed_kanit.py
"""
import ast
import hashlib
import random
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
import rakip_kos as rk  # noqa: E402

RNG_SUPHELI = ("random", "rand", "shuffle", "seed", "normal", "uniform", "init", "Model", "Exp", "DataLoader",
               "manual_seed", "permutation", "choice", "dropout")


def cagri_adi(node):
    f = node.func
    parca = []
    while isinstance(f, ast.Attribute):
        parca.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parca.append(f.id)
    return ".".join(reversed(parca))


def kisim_a():
    src = rk.RUN_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    ana = next(n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test))
    govde = ana.body
    ilk4 = [ast.unparse(s) for s in govde[:4]]
    assert ilk4[0].startswith("fix_seed = "), ilk4
    assert ilk4[1:] == ["random.seed(fix_seed)", "torch.manual_seed(fix_seed)", "np.random.seed(fix_seed)"], ilk4
    # tohum uclusunden 'exp = Exp(args)' satirina kadar (kaynak sirasi, satir no ile) tum cagrilar
    exp_satir = min(n.lineno for n in ast.walk(ana) if isinstance(n, ast.Assign) and ast.unparse(n) == "exp = Exp(args)")
    tohum_son = govde[3].lineno
    cagrilar = [cagri_adi(n) for n in ast.walk(ana) if isinstance(n, ast.Call) and tohum_son < n.lineno < exp_satir]
    # 'parser.add_argument' disindakiler
    ozel = sorted(set(c for c in cagrilar if c != "parser.add_argument"))
    supheli = [c for c in ozel if any(k.lower() in c.lower() for k in RNG_SUPHELI) and c not in ("Exp",)]
    assert not supheli, supheli
    return ilk4, len(cagrilar), ozel


def kisim_b():
    src = Path(rk.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    dongu = next(n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.target) == "H")
    satirlar = [ast.unparse(s) for s in dongu.body]
    i = satirlar.index("random.seed(a.seed)")
    assert satirlar[i:i + 3] == ["random.seed(a.seed)", "torch.manual_seed(a.seed)", "np.random.seed(a.seed)"]
    # tohumdan sonra Exp'e kadar (kaynak sirasi, satir no ile)
    tohum_son = dongu.body[i + 2].lineno
    exp_satir = min(n.lineno for n in ast.walk(dongu) if isinstance(n, ast.Assign)
                    and ast.unparse(n).startswith("exp = Exp_Long_Term_Forecast(args)"))
    arada = [cagri_adi(n) for n in ast.walk(dongu) if isinstance(n, ast.Call) and tohum_son < n.lineno < exp_satir]
    supheli = [c for c in arada if any(k.lower() in c.lower() for k in RNG_SUPHELI)]
    assert not supheli, supheli
    return sorted(set(arada))


def kisim_c(seed=2023):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(rk.REPO))
    from model import iTransformer
    rk.AKTIF = rk.VERILER["ettm1"]
    parser, _ = rk.resmi_parser()
    sh = rk.resmi_sh_argumanlari()

    def args_h(H):
        argv = rk.argv_degistir(sh[H], "--root_path", str(rk.AKTIF["dizin"]) + "/")
        return parser.parse_args(argv)

    def ozet(H):
        a = args_h(H)
        m = iTransformer.Model(a).float()
        h = hashlib.sha256()
        for k, v in m.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        # resmi data_provider train ayari: shuffle=True -> RandomSampler, torch global uretecinden ceker
        dl = DataLoader(list(range(34369)), batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True)
        ilk = next(iter(dl))[:64].tolist()
        return h.hexdigest(), ilk, float(np.random.rand()), random.random()

    def tohum():
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    tohum(); c1 = ozet(96)
    tohum(); _ = ozet(720); _ = torch.randn(1000); _ = np.random.rand(1000)  # kirli durum
    tohum(); c2 = ozet(96)
    assert c1 == c2, "yeniden tohumlama durumu sifirlamadi"
    tohum(); c3 = ozet(192)
    assert c3[0] != c1[0]
    return c1[0][:16], c1[1][:8]


def main():
    ilk4, n, ozel = kisim_a()
    print("[seed_kanit] (A) run.py __main__ ilk 4 deyim:", ilk4)
    print(f"[seed_kanit] (A) tohum -> Exp(args) arasi {n} cagri; parser.add_argument disindakiler: {ozel}"
          " -> RNG tuketen yok")
    arada = kisim_b()
    print(f"[seed_kanit] (B) rakip_kos ufuk dongusu: tohum uclusu -> Exp_Long_Term_Forecast(args) arasi cagrilar: {arada}"
          " -> RNG tuketen yok")
    h, ilk = kisim_c()
    print(f"[seed_kanit] (C) ETTm1 h96 s2023: taze == kirli+yeniden tohum (agirlik sha256 {h}..., karistirma ilk 8 {ilk},"
          " np/random sonraki cekilis ayni); h192 farkli model ozeti")
    print("[seed_kanit] GECTI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
