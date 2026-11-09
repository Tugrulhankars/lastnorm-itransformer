# san-softplus — H11 (= H01b), SAN std başlığı ReLU → Softplus

- **Kart:** `hipotezler/H11.md`. **Ebeveyn (karar):** `lastnorm` kat koşumları; **kod ebeveyni:** `varyantlar/san/` (H01).
- **Ebeveynden tek fark (kod):** `SANSPiTransformer` = `SANiTransformer` + `san.model_std.final_activation = nn.Softplus()`
  (`model_san_sp.py`, tek satır; parametre/RNG aynı).
- **Ön-koşul kapısı (CPU, `CUDA_VISIBLE_DEVICES=-1`):** `kapi.py` → `kapi.log` (karttaki ad) = `onkosul.log` (aynı içerik)
  + `sonuclar/h11-kapi.csv`. Yalnız istasyon ön-eğitimi (5 epoch, resmî erken durdurma; `ExpSAN.train` ilk 5 epoch'unun
  birebir kopyası, test yükleyicisi kurulmaz), K3 ve K1, s2023, 4 ufuk, ReLU/Softplus kolları. 128 s.
- **Sonuç (2026-10-14 16:45): KAPI=BAŞARISIZ.** Softplus kolunda ölü oranı o_H (σ̂ < 1e-3): K3 0.83/0.85/0.83/0.88,
  K1 0.56/0.55/0.55/0.68 → **G1 0/4, 0/4 BAŞARISIZ**; q_H = medyan(σ̂/σ_gerçek): K3 0.000 (×4), K1 0.015/0.015/0.014/0.004
  → **G2 BAŞARISIZ**; ReLU K3 o_H 1.00/1.00/1.00/0.99 → **G3 GEÇER** (H01 teşhisi yeniden üretildi, ölçüm kodu sağlam).
  Kart kuralı: kart koşulmaz ("ÖN-KOŞUL BAŞARISIZ"). `kos.py`/`dogrula.py`/kuyruk `k4t4-h11` YAZILMADI; GPU 0.
- Not (ölçüm tanımı): q_H'de σ_gerçek-dilim = 0 olan hücreler (yalnız K1'de, %0.6–0.9) orandan çıkarıldı, sayısı log'da.

## Koşum (WS kökünden)
```bash
CUDA_VISIBLE_DEVICES=-1 env/.venv/Scripts/python.exe varyantlar/san-softplus/kapi.py
```
