# san-mu — H05 (SAN-μ) ön-koşul kapısı

- **Kart:** `hipotezler/H05.md`. Bu klasörde yalnız kapı ölçümü var. Kapı **BAŞARISIZ** çıktığı için model sarmalayıcı
  (`model_san_mu.py`) ve `kos.py` **yazılmadı**, kuyruk yok. Kartın Sonuç bölümünü şef yazar.
- `onkosul.py` (CPU, ~12 s; yalnız train/val, test yükleyicisi kurulmaz): H01 tam koşum checkpoint'leri
  (`runs/20261014-040042_san_s2023`, `…040547_san_s2024`, `…041102_san_s2025`). Ufuk projektör şekliyle seçilir.
  Eşikler karttan sabit kodlandı. Çıktılar `onkosul.log` ve `sonuclar/h05-onkosul.csv`, son satır `KAPI=…`.
- Karar split'i val (tüm pencereler, adım 1). Train değerleri yalnız bilgi olarak raporlanır. Ö2, her ufkun val
  pencere kümesinde ayrı hesaplanır, kapıda en kötü ufuk kullanılır.
- Not: log'daki `cuda gorunur=True` yalnız bilgi satırı (`CUDA_VISIBLE_DEVICES=""` Windows'ta CUDA'yı gizlemedi).
  Ölçümde bütün tensörler CPU'daydı (`map_location="cpu"`, `.cuda()` çağrısı yok).

```bash
env/.venv/Scripts/python.exe varyantlar/san-mu/onkosul.py
```
