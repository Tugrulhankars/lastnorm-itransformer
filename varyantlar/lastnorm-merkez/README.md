# lastnorm-merkez — H10 (yalnız merkezleme, σ ≡ 1; H04 Ablasyon 1)

- **Kart:** `hipotezler/H10.md` (A7; Z1). Tam karttır (TUT/ÖLDÜR, durma sayacına girer) ve aynı zamanda K5 ablasyon
  tablosundaki "lastnorm − σ" satırıdır.
- **Ebeveyn:** `lastnorm` (H04). Koşumlar: `20261014-052508_lastnorm_s2023`, `20261014-053601_lastnorm_s2024`,
  `20261014-054213_lastnorm_s2025`.
- **Ebeveynden tek fark:** `forward`'da σ_lb ölçeklemesi kaldırıldı: `x = x_enc − x_son`, `ŷ = f(·) + x_son`
  (x_son = `x_enc[:, -1:, :]`, detach). Omurga, parametreler, eğitim, veri, seed ve h336 epoch'u (1) aynı.
- **Dosyalar:**
  - `model_merkez.py`: `MerkezLastNormiTransformer(LastNormiTransformer)`. Ebeveyn sınıfı
    `varyantlar/lastnorm/model_lastnorm.py`'den import edilir (kopya yok). `__init__` aynen kalır, yalnız `forward`
    override edilir; satırlar kartın uygulama notundakilerle birebir aynıdır.
  - `kos.py`: `lastnorm/kos.py` kalıbını izler (her ufukta `--use_norm 0`; `__init__` → `Exp_Basic.__init__`;
    `_build_model` → `MerkezLastNormiTransformer`). `metrics.json`'a `varyant`, `kart: H10`, `ebeveyn: lastnorm`,
    `ablasyon_of: H04-1` eklenir. `--kapi-test-metrics` reddedilir, `test_` anahtarı denetlenir. `sys.dont_write_bytecode`
    açık olduğundan `varyantlar/lastnorm/` altına yazılmaz.
  - `dogrula.py` → `dogrula.log` (CPU, `CUDA_VISIBLE_DEVICES=-1`, ~5 s), sonuç **GEÇTİ**:
    - (i) ilk ağırlıklar LastNorm ile özdeş (fark 0); parametre 224 224 / 236 608 / 255 184 / 304 720.
    - (ii) omurga girdisi = `x_enc − x_son` (fark 0).
    - (iii) projektör sıfırken çıkış = Repeat (fark 0).
    - (iv) çıkış LastNorm'dan farklı (maks. fark 1.93–2.25).
    - (v) x_son = `batch_x[:, -1]`.
- **Foreground (s2023, damgalı):**
  - Duman `20261014-071111_lastnorm-merkez-duman_s2023`: val_mse_h96 0.130346 (eşik ≤ 0.1923; `lastnorm-duman`'a göre
    +2.13%).
  - Ön-uçuş `20261014-071201_lastnorm-merkez-onucus_s2023`: val_mse_avg 0.420683 (eşik [0.38, 0.90]); `lastnorm-onucus`'a
    göre −5.54% (h96 +3.25 / h192 +3.79 / h336 −4.51 / h720 −8.85%). egitim_sure_sn 147.0 (eşik ≤ 328.6). Val tahminleri
    lastnorm-onucus ile bit-bit aynı değil (maks. fark 0.38–0.71), yani plasebo değil.
- **Kuyruk:** `kuyruk/k4t3-h10.jsonl` (5 satır: a1 duman, a2 ön-uçuş, a3 × {2023, 2024, 2025}), beklenti
  `kuyruk/k4t3-h10.beklenti.json`. Kuyruğu şef başlatır.

## Koşum (WS kökünden)
```bash
PY=env/.venv/Scripts/python.exe
$PY varyantlar/lastnorm-merkez/dogrula.py                                                                          # CPU sağlama, rc=0
$PY varyantlar/lastnorm-merkez/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --ufuklar 96 --epoch-ust 1 --num-workers 0  # duman
$PY varyantlar/lastnorm-merkez/kos.py --seed 2023 --out <RUN_DIR>/metrics.json --epoch-ust 2 --num-workers 0              # ön-uçuş
$PY varyantlar/lastnorm-merkez/kos.py --seed <s> --out {RUN_DIR}/metrics.json --num-workers 0                             # tam (kuyruk)
```
