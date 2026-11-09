# H01 teshis 2: SAN istasyon on-egitiminde std basligi (ReLU) olumu -- seed x ufuk taramasi. Yalniz train/val.
import sys, random
from pathlib import Path
WS = Path(sys.argv[1])
sys.path.insert(0, str(WS/"scripts")); import rakip_kos
sys.path.insert(0, str(rakip_kos.REPO)); sys.path.insert(0, str(WS/"varyantlar"/"san"))
import torch, numpy as np
from torch.utils.data import DataLoader
from model_san import SANiTransformer
from data_provider.data_factory import data_dict
dev = torch.device("cuda:0")
parser,_ = rakip_kos.resmi_parser(); sh = rakip_kos.resmi_sh_argumanlari()
def olu(m, X):
    with torch.no_grad():
        _, st = m.san.normalize(X); s = st[:, :, 8:]
        return float((s == 0).float().mean()), float(s.mean())
for H in (96, 192, 336, 720):
    argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN)+"/")
    argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
    a = parser.parse_args(rakip_kos.argv_degistir(argv, "--use_norm", "0"))
    mk = lambda f: data_dict["custom"](root_path=a.root_path, data_path=a.data_path, flag=f,
          size=[a.seq_len,a.label_len,a.pred_len], features="M", target="OT", timeenc=1, freq="h")
    tr, va = mk("train"), mk("val")
    Xv = torch.tensor(np.stack([va[i][0] for i in range(len(va))])).float().to(dev)
    for s in (2023, 2024, 2025):
        random.seed(s); torch.manual_seed(s); np.random.seed(s)
        m = SANiTransformer(a).float().to(dev)
        opt = torch.optim.Adam(m.san.parameters(), lr=1e-3)
        iz = [olu(m, Xv)]
        dl = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
        for ep in range(5):
            for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 ** ep
            for bx, by, _, _ in dl:
                bx = bx.float().to(dev); y = by.float().to(dev)[:, -H:, :]
                _, st = m.san.normalize(bx)
                yy = y.reshape(y.shape[0], -1, 6, 8)
                loss = torch.nn.functional.mse_loss(st, torch.cat([yy.mean(2), yy.std(2)], -1))
                opt.zero_grad(); loss.backward(); opt.step()
            iz.append(olu(m, Xv))
        print(f"h{H} s{s}: val std-cikti==0 orani init→ep1..5: " + " ".join(f"{d:.2f}" for d, _ in iz)
              + f" | son tahmin std ort {iz[-1][1]:.4f}")
