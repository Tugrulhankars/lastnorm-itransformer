# H01 teshis: SAN std basliginin olu ReLU'su mu? (yalniz VAL ve TRAIN verisi; test yok)
import sys, glob, os
from pathlib import Path
WS = Path(sys.argv[1]); RUN = Path(sys.argv[2])
sys.path.insert(0, str(WS/"scripts")); import rakip_kos
sys.path.insert(0, str(rakip_kos.REPO)); sys.path.insert(0, str(WS/"varyantlar"/"san"))
import torch, numpy as np
from model_san import SANiTransformer
from data_provider.data_factory import data_dict
parser,_ = rakip_kos.resmi_parser(); sh = rakip_kos.resmi_sh_argumanlari()
for H in (96, 336):
    argv = rakip_kos.argv_degistir(sh[H], "--root_path", str(rakip_kos.VERI_DIZIN)+"/")
    argv = rakip_kos.argv_degistir(argv, "--data_path", "exchange_rate.csv")
    a = parser.parse_args(rakip_kos.argv_degistir(argv, "--use_norm", "0"))
    m = SANiTransformer(a).float()
    ck = [p for p in glob.glob(str(RUN/"ckpt"/"*"/"checkpoint.pth")) if f"_pl{a.pred_len}_" in p or True]
    # setting adinda alan kaymasi var; pred_len 'll' alaninda: dogru dosyayi parametre sekliyle sec
    sd = None
    for p in glob.glob(str(RUN/"ckpt"/"*"/"checkpoint.pth")):
        s = torch.load(p, map_location="cpu")
        if s["backbone.projector.weight"].shape[0] == H: sd = s; break
    m.load_state_dict(sd); m.eval()
    for split in ("train", "val"):
        ds = data_dict["custom"](root_path=a.root_path, data_path=a.data_path, flag=split,
              size=[a.seq_len,a.label_len,a.pred_len], features="M", target="OT", timeenc=1, freq="h")
        X = torch.tensor(np.stack([ds[i][0] for i in range(0, len(ds), 3)])).float()
        Y = torch.tensor(np.stack([ds[i][1] for i in range(0, len(ds), 3)])).float()[:, -H:, :]
        with torch.no_grad():
            xn, st = m.san.normalize(X)
            std_pred = st[:, :, 8:]
            ys = Y.reshape(Y.shape[0], -1, 6, 8)
            print(f"h{H} {split}: tahmin std==0 orani {float((std_pred==0).float().mean()):.3f} | tahmin std ort {float(std_pred.mean()):.4f} | gercek dilim std ort {float(ys.std(2).mean()):.4f} | girdi dilim std ort {float(X.reshape(X.shape[0],-1,6,8).std(2).mean()):.4f}")
    # omurga gradyani (train batch)
    m.train(); X32, Y32 = X[:32], Y[:32]
    xm = torch.zeros(32, a.seq_len, 4)
    out = m(X32, xm, None, None); loss = ((out - Y32)**2).mean(); loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in m.backbone.parameters() if p.grad is not None)
    print(f"h{H}: omurga |grad| toplami = {g:.3e}")
