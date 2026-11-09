"""K5 naif referans (Repeat = son gözlem) TEST MSE — ayar/seçim içermez; test kilidi açıldıktan sonra
referans satırı olarak hesaplandı (karar.md'de beyan). TSLib Dataset_Custom, scaler yalnız train, L=96,
test batch=1 (tüm pencereler). Çalıştırma (WS kökü): env/.venv/Scripts/python.exe scripts/k5_naif_test.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[1]
sp = json.loads((WS / "data/splits/exchange_rate_tslib_sl96.json").read_text(encoding="utf-8"))
df = pd.read_csv(WS / "data/exchange_rate/exchange_rate.csv")
cols = [c for c in df.columns if c != "date"]
m = np.array([sp["scaler_train_only"]["mean"][c] for c in cols]); s = np.array([sp["scaler_train_only"]["scale"][c] for c in cols])
X = (df[cols].values - m) / s
b1, b2 = sp["border1s"][2], sp["border2s"][2]
D = X[b1:b2]; L = 96; out = []
for H in (96, 192, 336, 720):
    n = len(D) - L - H + 1
    last = np.stack([D[i + L - 1] for i in range(n)])[:, None, :]
    Y = np.stack([D[i + L:i + L + H] for i in range(n)])
    tru = np.load(WS / f"data/hedefler_sl96/test_h{H}_true.npy") if (WS / f"data/hedefler_sl96/test_h{H}_true.npy").exists() else None
    if tru is not None:
        assert tru.shape == Y.shape and np.allclose(tru, Y, atol=1e-5), (H, tru.shape, Y.shape)
    mse = float(((Y - last) ** 2).mean()); mae = float(np.abs(Y - last).mean())
    out.append((H, n, mse, mae)); print(f"h{H} n={n} repeat test_mse={mse:.4f} mae={mae:.4f} hedef_dogrulama={'OK' if tru is not None else 'yok'}")
print(f"ortalama test_mse={np.mean([o[2] for o in out]):.4f} mae={np.mean([o[3] for o in out]):.4f}")
pd.DataFrame(out, columns=["ufuk", "pencere", "test_mse", "test_mae"]).to_csv(WS / "sonuclar/naif-test.csv", index=False)
