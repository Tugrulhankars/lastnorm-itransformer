# lastnorm-esit — H14 (sabit eşit birleşim, w ≡ 0.5)

- **Kart:** `hipotezler/H14.md`. **Ebeveyn:** `lastnorm` kat koşumları: K1 `163510/163633/163802`, K2 `163051/163203/163345`,
  K3 = resmî `052508/053601/054213`. Seed {2023, 2024, 2025}.
- **Ebeveynden tek fark:** val tahminine sabit son-işlem `C = 0.5·P + 0.5·R` (R = Repeat, kat-k scaler uzayında).
  Bütün kat/ufuk/adım/değişkende aynı ağırlık; hiçbir şey kestirilmez. Eğitim/model aynı; GPU 0.
- **Dosyalar:** `esit.py` (H12 `birlesim.py`'den R/Y yeniden kurma, `birlestir`, test kilidi süzgeci ve `pencere_kaydet`
  import edilir; `agirlik_kestir` çağrılmaz; K0/iç model açılmaz). `dogrula.py` → `dogrula.log`: 9 (kat, seed) için
  (i) w=1 → C == P bit-bit, MSE == ebeveyn; (ii) w=0 → C == Repeat bit-bit, MSE == `naif-kat.csv`; (iii) kimlik;
  (iv) test yolu/anahtarı okunmaz; (v) dosya açılışları izlenir, yalnız hedef koşum okunur; (vi) kestirim
  fonksiyonları yasaklanarak koşulur. **GEÇTİ.**
- **Ablasyon 1 (`baseline-esit`):** `esit.py --ebeveyn baseline [--kaynak <K3 run_id>]` (kart `H14-abl1`); kuyruğa
  eklenmedi (bu görevin kapsamı dışı).
- **Kuyruk:** `kuyruk/k4t5-lastnorm-esit.jsonl` (9 satır, CPU, aşama 3, K3 → K2 → K1); beklenti
  `kuyruk/k4t5-lastnorm-esit.beklenti.json`.
- Çıktı: `runs/<ts>_lastnorm-esit-k{k}_s{s}/` (metrics.json + `preds/val_h*.npz` birleşik tahmin).

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
CUDA_VISIBLE_DEVICES=-1 $PY varyantlar/lastnorm-esit/dogrula.py
CUDA_VISIBLE_DEVICES=-1 $PY scripts/satir_kos.py k4t5-lastnorm-esit <1..9>
```
