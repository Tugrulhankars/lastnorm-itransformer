# lastnorm-birlesim — H12 (konveks birleşim, Bates–Granger)

- **Kart:** `hipotezler/H12.md`. **Ebeveyn:** `lastnorm` kat koşumları (`lastnorm-k1/k2` × 3 seed; K3 = resmî
  `052508/053601/054213`, şef K3 yeniden kullanım kararı).
- **Ebeveynden tek fark:** val tahmini son-işlem `C = R + clip(α̂,0,1)·(P − R)`, (H, 10 adım-dilimi) başına; α̂ yalnız
  katın train'i [0, b_k) içindeki iç-holdout [b_k − V, b_k)'tan, iç model = K_{k−1} kat koşumu (K1 için `lastnorm-k0`).
  Eğitim/model aynı; GPU 0 (CPU son-işlem).
- **Dosyalar:** `birlesim.py` (kestirim + uygulama + sağlamalar + türev kayıt; H13 bunu import eder), `dogrula.py`
  (`--varyant birlesim|olcek`) → `dogrula.log`: (i) w=1 → C == P bit-bit, (ii) w=0 → C == Repeat ve MSE ==
  `sonuclar/naif-kat.csv`, (iii) iç-holdout < b_k assert, (iv) P' MSE == K_{k−1} metrics (|d| = 0), (v) test yolu/anahtarı
  okunmaz; **sızıntı:** b_k ve sonrası NaN → ağırlıklar bit-bit aynı, b_k−1 bozulunca değişiyor; kat val tahminleri
  rastgele → ağırlıklar aynı. **GEÇTİ** (K2, K3; s2023).
- **İç model:** `scripts/rakip_kos.py --kat 0` (train [0,3031), val [3031,3791); `--kat`'sız/1/3 yolları bit-bit
  korunarak eklendi), kuyruk `k4t4-lastnorm-k0`.
- **Foreground duman:** `20261014-165651_lastnorm-birlesim-duman_s2023` (K2 s2023 uçtan uca, rc=0, 3.6 s).
- **Kuyruk:** `kuyruk/k4t4-lastnorm-birlesim.jsonl` (9 satır, CPU, aşama 3, sıra K3→K2→K1; **K1 satırları
  `k4t4-lastnorm-k0` bittikten sonra**), beklenti `kuyruk/k4t4-lastnorm-birlesim.beklenti.json`. Kart uygulama notundaki
  `k4t4-k0` adı yerine orkestratörün adlandırması kullanıldı (`k4t4-lastnorm-k0`).
- Çıktı: `runs/<ts>_lastnorm-birlesim-k{k}_s{s}/` (metrics.json + preds/val_h*.npz birleşik tahmin).
- Not (İNCELE (a)): `oracle_ihlal_sayisi` dilim düzeyi oracle sınırına göre (matematiksel olarak aşılamaz);
  kartın tek-α/hücre sınırı `oracle_ihlal_sayisi_tek_alfa_hucre` olarak ayrıca yazılır — 10 dilim ağırlığı onu meşru
  biçimde aşabilir (duman K3 s2023 olcek'te 1 hücre).

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/lastnorm-birlesim/dogrula.py --varyant birlesim
$PY varyantlar/lastnorm-birlesim/birlesim.py --kat <k> --seed <s> --out {RUN_DIR}/metrics.json   # kuyruk
```
