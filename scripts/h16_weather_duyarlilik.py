# H16 Weather -9999 duyarliligi (ON-KAYITLI DEGIL; yalniz saglamlik bilgisi). Yalniz train. Kullanim (WS kokunden):
# env/.venv/Scripts/python.exe -I scripts/h16_weather_duyarlilik.py
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("data/weather/weather.csv"); c=[x for x in df.columns if x not in ("date","OT")]+["OT"]
X=df[c].iloc[:36887].to_numpy(float); bad=(X==-9999).any(1)
Z=StandardScaler().fit(X).transform(X); out=[]; nd=[]
for H in (96,192,336,720):
    a=b=0.0; n=0; d=0
    for s in range(0,36887-96-H+1,7):
        if bad[s:s+96+H].any(): d+=1; continue
        x=Z[s:s+96]; y=Z[s+96:s+96+H]; a+=((y-x[-1])**2).mean(); b+=((y-x.mean(0))**2).mean(); n+=1
    out.append(a/b); nd.append(d)
print("scaler tum train, -9999 pencereleri haric:", [round(v,4) for v in out], round(np.mean(out),4), "atilan", nd)
Z=StandardScaler().fit(X[~bad]).transform(X); out=[]
for H in (96,192,336,720):
    a=b=0.0
    for s in range(0,36887-96-H+1,7):
        if bad[s:s+96+H].any(): continue
        x=Z[s:s+96]; y=Z[s+96:s+96+H]; a+=((y-x[-1])**2).mean(); b+=((y-x.mean(0))**2).mean()
    out.append(a/b)
print("scaler -9999 haric, pencereler haric:", [round(v,4) for v in out], round(np.mean(out),4))
