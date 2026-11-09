# Ortam Kurulumu — 5.3 Exchange-Rate / iTransformer

**Hesaplama hedefi:** A — bu makine, GPU (kullanıcı kararı 2026-10-14). Ölçüm: `env/ortam-yerel.md` (rc=0).
- NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MB, sürücü 566.36 · Windows 11 Pro · Python 3.11.4
- Sistem Python'u KULLANILMAZ (torch 2.2.2+cpu, NumPy 2.4 ↔ pyarrow ABI çakışması).

## Kurulum (Git Bash, WS kökünden; kopyala-çalıştır)
```bash
python -m venv env/.venv                     # uv/conda yok; tek yorumlayıcı 3.11
env/.venv/Scripts/python.exe -m pip install --upgrade pip
env/.venv/Scripts/python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
env/.venv/Scripts/python.exe -m pip install "numpy<2" pandas scikit-learn scipy einops matplotlib tqdm
env/.venv/Scripts/python.exe -c "import torch,numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
env/.venv/Scripts/python.exe -m pip freeze > env/requirements.lock
```
- `numpy<2`: TSLib `utils/tools.py` eski NumPy API'si (`np.Inf`) kullanır; NumPy 2'de kaldırıldı.
- Rakibe özgü ek bağımlılıklar K3'te `rakip-kurucu` tarafından eklenir ve lock yenilenir.

## Doğrulama çıktısı
```
TORCH 2.4.1+cu121 CUDA True NVIDIA GeForce RTX 3060 Laptop GPU numpy 1.26.4
```
(2026-10-14; `env/requirements.lock` 31 paket)

## Ortam değişkenleri
- `PYTHONIOENCODING=utf-8`
- Kuyruk koşumları `kosum.py --gpu-zorunlu` ile; tek GPU → koşumlar seri.

## Yazım araçları (K1'de kullanıcı isteğiyle kuruldu)
- tectonic 0.17.0 → proje kökünde `tectonic.exe` (resmî PowerShell yükleyici)
- PyMuPDF 1.28.2 → sistem Python'u (yalnız yazım/PDF denetimi için)

## K3 eklemeleri (rakip-kurucu, 2026-10-14)
```bash
env/.venv/Scripts/python.exe -m pip install --no-deps "reformer-pytorch==1.4.4" "local-attention==1.4.4" "product-key-memory==0.1.10" "axial-positional-embedding==0.2.1"
env/.venv/Scripts/python.exe -m pip check && env/.venv/Scripts/python.exe -m pip freeze > env/requirements.lock   # 35 paket
```
- `--no-deps` zorunlu: düz kurulum torch'u 2.14.1'e yükseltiyor (bkz. `rakip/KURULUM.md` §2; torch 2.4.1+cu121 geri kuruldu).
- pandas 3.0.6 korundu; iTransformer veri hattının pandas 1.5.3 ile özdeş çıktı verdiği doğrulandı (`rakip/KURULUM.md` §2).
- Kuyruk/kosum betikleri bu venv'in python'uyla çağrılmalı (`kosum.py --gpu-zorunlu` torch'u kendi yorumlayıcısında arar).
