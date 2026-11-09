# lastnorm-olcek — H13 (Granger–Ramanathan, toplam = 1, α ≥ 0)

- **Kart:** `hipotezler/H13.md`. **Ebeveyn:** `lastnorm` kat koşumları (H12 ile aynı).
- **Ebeveynden tek fark:** `C = R + max(α̂,0)·(P − R)`; α̂ kestiricisi, iç-holdout, dilimler, sağlamalar H12 ile birebir
  (`olcek.py`, `varyantlar/lastnorm-birlesim/birlesim.py`'yi import eder; tek fark dönüşüm). Ağırlıklar `alfa_h{H}`.
- **Doğrulama:** `../lastnorm-birlesim/dogrula.py --varyant olcek` → `dogrula.log` **GEÇTİ** (H12 (i)–(v) + α ≡ 1 →
  C == P + sızıntı testleri).
- **Foreground duman:** `20261014-165655_lastnorm-olcek-duman_s2023` (K3 s2023 uçtan uca, rc=0, 3.4 s).
- **Kuyruk:** `kuyruk/k4t4-lastnorm-olcek.jsonl` (9 satır, CPU, aşama 3, K3→K2→K1; K1 satırları `k4t4-lastnorm-k0`
  bittikten sonra), beklenti `kuyruk/k4t4-lastnorm-olcek.beklenti.json`.

## Koşum (WS kökünden)
```bash
env/.venv/Scripts/python.exe varyantlar/lastnorm-olcek/olcek.py --kat <k> --seed <s> --out {RUN_DIR}/metrics.json
```
