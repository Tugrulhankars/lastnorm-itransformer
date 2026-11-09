#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rakip_kos.py -- K3 metrik koprusu: resmi iTransformer (thuml/iTransformer @ c2426e6) Exchange betigini
rakip koduna DOKUNMADAN calistirir, tek kosumda 4 ufku (96/192/336/720) egitir ve standart sozluk yazar.

Rakip kodu (WS/rakip/iTransformer) degistirilmez. Sarmalayicinin yaptiklari:
  * Hiperparametreler resmi dosyalardan OKUNUR (elle kopyalanmaz):
      - run.py'deki parser.add_argument(...) cagrilari AST ile cikarilir -> varsayilanlar birebir;
      - scripts/multivariate_forecasting/Exchange/iTransformer.sh ayristirilir -> ufuk basina argumanlar
        (ufuk 336 icin --train_epochs 1 DAHIL). Yalniz --root_path/--data_path yerel veri dosyasina,
        --checkpoints {RUN_DIR}/ckpt'ye cevrilir.
  * Seed baglama: run.py, Exp olusturulmadan once random/np/torch'u fix_seed=2023 ile tohumlar ve her ufuk
    ayri bir surec oldugu icin her ufukta yeniden tohumlanir. Sarmalayici ayni uc cagriyi (random.seed,
    torch.manual_seed, np.random.seed) her ufukta Exp(args)'tan hemen once --seed degeriyle yapar.
    --seed 2023, resmi surecle ayni tohum durumunu verir.
  * Egitim/test: resmi Exp_Long_Term_Forecast.train() ve .test() oldugu gibi cagrilir (erken durdurma,
    lr cizelgesi, val kaybi, test batch=1 -- hepsi resmi kod).
  * Val metrigi (bizim): en iyi checkpoint ile val kumesinin TUM pencereleri (shuffle yok, drop_last yok,
    batch 1), test ile ayni tanim (olcekli uzay, MSE = tum eleman ortalamasi). Resmi vali() kaybi
    (bs32, drop_last=True, shuffle) yalniz progress/log icin.
  * Test metrigi: resmi test()'in kendi yazdigi metrics.npy'den okunur -> {RUN_DIR}/test_metrics.json.
    Test sayilari stdout'a BASILMAZ (resmi log satirlarindaki 'Test Loss' ve 'mse:' maskelenir; ham log
    {RUN_DIR}/test_kilitli/ altinda). --kapi-test-metrics verilirse test_* anahtarlari metrics.json'a da
    eklenir (yalniz K3 replikasyon kapisi icin: replikasyon_kapisi.py metrigi .done['metrics'] =
    metrics.json'dan okur).
  * Pencere bazli ciktilar: {RUN_DIR}/preds/val_h{H}.npz ve {RUN_DIR}/preds/test_kilitli/test_h{H}.npz
    (pred float32 (N,H,8), mse/mae pencere x degisken (N,8), mse pencere x adim (N,H), hedef_ilk_satir (N,)).
    Gercek degerler seed'den bagimsizdir; WS/data/hedefler_sl96/<split>_h<H>_true.npy olarak bir kez
    yazilir, sonraki kosumlarda esitligi dogrulanir (veri hatti degisirse kosum durur).
  * progress.json: {"adim","ufuk","epoch","loss","val_loss","val_mse_h<H>"} -> kosum.py --kural.
  * Cihaz: CUDA yoksa SESSIZ CPU'ya dusus yok; cikis kodu 97.

Kullanim (WS kokunden; kuyrukta {RUN_DIR} kosum.py tarafindan doldurulur):
  env/.venv/Scripts/python.exe scripts/rakip_kos.py --seed 2023 --out {RUN_DIR}/metrics.json
  [--ufuklar 96,192,336,720] [--epoch-ust N]  (duman/on-ucus icin kisaltma; tam kosumda VERILMEZ)
  [--num-workers N]  (varsayilan: resmi deger 10) [--kapi-test-metrics]
  [--veri exchange|etth1|etth2]  (varsayilan exchange = bugunku davranis; ETT: resmi iTransformer_ETTh{1,2}.sh +
                                  Dataset_ETT_hour, sha256 sabit, degisken sayisi 7 -> npz son boyutu 7)
  [--kat 1|2|3]  (FAZ II P1 cok-dilimli dogrulama; makale icin sonradan eklendi -- deney-gelistirici, 2026-10-14)
  [--kat 0]      (tur 4, H12/H13 ortak ic modeli `lastnorm-k0`: ayni formul, val [T-3V, T-2V) = Exchange [3031,3791),
                  train [0,3031); yalniz K1 train'i icinde; degerlendirme kati DEGIL -- deney-yurutucu, 2026-10-14.
                  ETT'de val_bas=0 -> SystemExit (K0 tanimsiz, H12 karti). 1/2/3 ve --kat'siz yol degismedi)
  [--veri ettm1|ettm2|weather] [--ufuk-basi --ufuklar H]  (FAZ III / H16 -- deney-yurutucu, 2026-10-15)
     * VERILER girdileri data/data_manifest_faz3.json 'rakip_kos_veriler_onerisi' ile birebir (sha256 kontrolu).
     * --ufuk-basi: tek ufuk tek surec. Resmi .sh her pred_len icin AYRI 'python -u run.py' cagirir; run.py
       __main__'in ilk satirlari fix_seed ile random/np/torch'u tohumlar, Exp(args)'a kadar RNG tuketen satir yok
       (scripts/h16_seed_kanit.py: AST + CPU deneyi). Sarmalayici ayni uc cagriyi Exp(args)'tan hemen once yapar
       -> tek ufuklu surec resmi ufuk sureciyle ayni tohum durumunda baslar. metrics.json: 'ufuk', 'ufuk_basi',
       kisaltilmis yalniz --epoch-ust'e bagli. Analiz semasi: notlar/h16-kuyruk.md.
     * ince kayit (VERILER 'ince_kayit': True; yalniz bu uc veri): preds/*.npz'de 'pred' YOK (yalniz
       mse_pencere_degisken, mae_pencere_degisken, mse_pencere_adim, hedef_ilk_satir), ara diziler parca parca;
       metrikler ayni (scripts/h16_ince_dogrula.py bit-bit assert). exchange/etth1/etth2 yolu degismedi
       (Exchange duman bit-bit regresyonu, notlar/h16-kuyruk.md).

FAZ II P1 -- `--kat k` (verilmezse kod yolu BIREBIR eskisi; bit-bit sinandi, gunluk 2026-10-14):
  V = resmi val uzunlugu, T = resmi train sonu (Exchange: Dataset_Custom 0.7/0.1/0.2 -> T=5311, V=760;
  ETTh: Dataset_ETT_hour sabit sinirlar -> T=12*30*24=8640, V=4*30*24=2880).
  val_bas = T - (3-k)*V, val_son = val_bas + V; train satirlari [0, val_bas), val pencereleri [val_bas - L, val_son)
  (resmi border1 mantigi, L = seq_len = 96); scaler YALNIZ [0, val_bas) ile; k=3 == resmi train/val.
  Uygulama: data_provider.data_factory.data_dict[args.data] SUREC ICINDE KatVeri alt sinifiyla degistirilir
  (rakip dosyasi degismez). flag='test' bos veri kumesi (0 pencere, __getitem__ hata) -> resmi train() dongusundeki
  test vali() satiri hicbir test ornegi okumaz, test kaybi nan; ama DataLoader iterator'i yine acildigi icin torch
  global RNG cekilisi (dataloader.py _base_seed) resmi dongu ile ayni kalir (H09 exp_dro.py emsali).
  exp.test(), test pencereleri, test hedefleri ve test_metrics.json --kat'ta HIC uretilmez/okunmaz.
  Val hedefleri data/hedefler_sl96[_<veri>]/val_k{k}_h{H}_true.npy.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import os
import random
import re
import shlex
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
REPO = WS / "rakip" / "iTransformer"
RESMI_SH = REPO / "scripts" / "multivariate_forecasting" / "Exchange" / "iTransformer.sh"
RUN_PY = REPO / "run.py"
VERI_DIZIN = WS / "data" / "exchange_rate"
VERI_SHA256 = "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842"
HEDEF_DIZIN = WS / "data" / "hedefler_sl96"
RAKIP_COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"
UFUKLAR = [96, 192, 336, 720]

# FAZ II (makale icin sonradan eklendi -- deney-gelistirici): veri seti tablosu. 'exchange' satiri yukaridaki
# sabitlerin AYNISI (varsayilan; --veri verilmezse davranis degismez). val_b1/test_b1 = resmi border1 (pencere
# kaydinda hedef_ilk_satir = seq_len + border1). ETT: Dataset_ETT_hour sabit sinirlari (resmi data_dict['ETTh1'|
# 'ETTh2'] = Dataset_ETT_hour, --kat'siz yolda yamasiz kullanilir); hiperparametreler resmi
# iTransformer_ETTh{1,2}.sh'tan resmi_sh_argumanlari() ile ayristirilir (elle kopya yok; betikteki
# 'export CUDA_VISIBLE_DEVICES=1' satiri run.py blogu olmadigi icin atlanir, cihaz cuda:0).
# sha256 (P2 rakip-kurucu, 2026-10-14): data/data_manifest_ett.json (veri-hazirlayici; HF thuml@2b66e59e ==
# ETDataset@1d16c8f4 bayt bayt) ile ayni; kosum hash uyusmazsa durur.
_MF = REPO / "scripts" / "multivariate_forecasting"
VERILER = {
    "exchange": {"sh": RESMI_SH, "dizin": VERI_DIZIN, "csv": "exchange_rate.csv", "sha256": VERI_SHA256,
                 "hedef": HEDEF_DIZIN, "val_b1": 5215, "test_b1": 5975, "bolme": "custom"},
    "etth1": {"sh": _MF / "ETT" / "iTransformer_ETTh1.sh", "dizin": WS / "data" / "etth1", "csv": "ETTh1.csv",
              "sha256": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
              "hedef": WS / "data" / "hedefler_sl96_etth1",
              "val_b1": 12 * 30 * 24 - 96, "test_b1": 12 * 30 * 24 + 4 * 30 * 24 - 96, "bolme": "ett_hour"},
    "etth2": {"sh": _MF / "ETT" / "iTransformer_ETTh2.sh", "dizin": WS / "data" / "etth2", "csv": "ETTh2.csv",
              "sha256": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
              "hedef": WS / "data" / "hedefler_sl96_etth2",
              "val_b1": 12 * 30 * 24 - 96, "test_b1": 12 * 30 * 24 + 4 * 30 * 24 - 96, "bolme": "ett_hour"},
    # FAZ III / H16 (deney-yurutucu, 2026-10-15): girdiler data/data_manifest_faz3.json 'rakip_kos_veriler_onerisi'
    # ile birebir (sha256, sinirlar). 'ince_kayit': True -> tam pred dizisi KAYDEDILMEZ, pencere ciktilari parca
    # parca hesaplanir (asagida "ince kayit"); yalniz bu uc girdide acik, exchange/etth* yolu degismez.
    # Solar KESILDI (orkestrasyon_durumu.md FAZ III; RAM/disk) -> eklenmedi.
    "ettm1": {"sh": _MF / "ETT" / "iTransformer_ETTm1.sh", "dizin": WS / "data" / "ettm1", "csv": "ETTm1.csv",
              "sha256": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
              "hedef": WS / "data" / "hedefler_sl96_ettm1",
              "val_b1": 34464, "test_b1": 45984, "bolme": "ett_minute", "ince_kayit": True},
    "ettm2": {"sh": _MF / "ETT" / "iTransformer_ETTm2.sh", "dizin": WS / "data" / "ettm2", "csv": "ETTm2.csv",
              "sha256": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
              "hedef": WS / "data" / "hedefler_sl96_ettm2",
              "val_b1": 34464, "test_b1": 45984, "bolme": "ett_minute", "ince_kayit": True},
    "weather": {"sh": _MF / "Weather" / "iTransformer.sh", "dizin": WS / "data" / "weather", "csv": "weather.csv",
                "sha256": "34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63",
                "hedef": WS / "data" / "hedefler_sl96_weather",
                "val_b1": 36791, "test_b1": 42061, "bolme": "custom", "ince_kayit": True},
}
AKTIF = VERILER["exchange"]  # main() --veri ile degistirir; resmi_sh_argumanlari() bunu okur


def kat_sinirlari(bolme, n_satir, kat):
    """(T, V, val_bas, val_son): resmi train sonu T, resmi val uzunlugu V, katin val satir araligi [val_bas, val_son)."""
    if bolme == "custom":  # Dataset_Custom: num_train = int(0.7 n), num_test = int(0.2 n), V = kalan
        T = int(n_satir * 0.7)
        V = n_satir - T - int(n_satir * 0.2)
    elif bolme == "ett_hour":  # Dataset_ETT_hour: train 12 ay, val 4 ay (30 gun x 24 saat)
        T, V = 12 * 30 * 24, 4 * 30 * 24
    else:
        raise ValueError(bolme)
    assert kat in (0, 1, 2, 3), kat  # 0: H12/H13 ic modeli K0 (tur 4; yalniz train ici, degerlendirme kati degil)
    val_bas = T - (3 - kat) * V
    val_son = val_bas + V
    if val_bas <= 0:
        raise SystemExit(f"kat {kat}: train bos (val_bas={val_bas})")
    return T, V, val_bas, val_son


def kat_veri_sinifi(Taban, bolme, kat):
    """Resmi Dataset_Custom / Dataset_ETT_hour'un __read_data__'sini kat sinirlariyla tekrarlar (sutun duzeni,
    StandardScaler, time_features birebir). Test satirlari (>= val_son) diziye HIC girmez."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from utils.timefeatures import time_features

    class KatVeri(Taban):
        KAT = kat
        BOLME = bolme

        def __read_data__(self):
            self.scaler = StandardScaler()
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            if bolme == "custom":  # Dataset_Custom sutun duzeni: date, diger, hedef
                cols = list(df_raw.columns)
                cols.remove(self.target)
                cols.remove('date')
                df_raw = df_raw[['date'] + cols + [self.target]]
            T, V, val_bas, val_son = kat_sinirlari(bolme, len(df_raw), kat)
            self.kat_bilgi = {"n_satir_csv": len(df_raw), "T_resmi": T, "V": V, "val_bas": val_bas,
                              "val_son": val_son}
            df_raw = df_raw.iloc[:val_son]  # test (ve sonraki) satirlar burada atilir
            if self.set_type == 2:  # test: --kat'ta kilitli -> bos
                self.data_x = self.data_y = np.zeros((0, len(df_raw.columns) - 1))
                self.data_stamp = np.zeros((0, 1))
                self.kat_bilgi.update({"satir": [val_son, val_son]})
                return
            border1s = [0, val_bas - self.seq_len]
            border2s = [val_bas, val_son]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], axis=1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp
            self.kat_bilgi.update({"satir": [border1, border2]})

        def __getitem__(self, index):
            if self.set_type == 2:
                raise RuntimeError("--kat: test dilimi kilitli (okunamaz)")
            return super().__getitem__(index)

        def __len__(self):
            if self.set_type == 2:
                return 0
            return super().__len__()

    KatVeri.__name__ = f"KatVeri_{Taban.__name__}_k{kat}"
    return KatVeri


# ------------------------------------------------------------------ resmi ayarlari oku
def resmi_parser():
    """run.py icindeki parser.add_argument(...) cagrilarini AST ile cikarir; dosya degistirilmez."""
    src = RUN_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    p = argparse.ArgumentParser(description="iTransformer (run.py'den AST ile)")
    n = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "parser"):
            eval(compile(ast.Expression(body=node), str(RUN_PY), "eval"),
                 {"parser": p, "int": int, "str": str, "float": float, "bool": bool})
            n += 1
    if n < 40:
        raise SystemExit(f"run.py'den yalniz {n} arguman okunabildi; beklenmeyen dosya")
    # run.py'deki sabit tohum (bilgi icin)
    m = re.search(r"fix_seed\s*=\s*(\d+)", src)
    return p, (int(m.group(1)) if m else None)


def resmi_sh_argumanlari():
    """iTransformer.sh -> {pred_len: [argv...]} (her 'python -u run.py' blogu bir ufuk)."""
    txt = AKTIF["sh"].read_text(encoding="utf-8").replace("\\\r\n", " ").replace("\\\n", " ")
    txt = txt.replace("$model_name", "iTransformer")
    bloklar = {}
    for satir in txt.splitlines():
        satir = satir.strip()
        if "run.py" not in satir:
            continue
        tok = shlex.split(satir)
        argv = tok[tok.index("run.py") + 1:]
        pl = int(argv[argv.index("--pred_len") + 1])
        bloklar[pl] = argv
    if sorted(bloklar) != UFUKLAR:
        raise SystemExit(f"resmi betikte beklenen ufuklar yok: {sorted(bloklar)}")
    return bloklar


def argv_degistir(argv, anahtar, deger):
    argv = list(argv)
    if anahtar in argv:
        argv[argv.index(anahtar) + 1] = deger
    else:
        argv += [anahtar, deger]
    return argv


def setting_adi(args, ii=0):
    # run.py ile birebir ayni bicim
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
        args.distil, args.des, args.class_strategy, ii)


# ------------------------------------------------------------------ yardimcilar
def atomik_json(yol: Path, veri: dict):
    yol.parent.mkdir(parents=True, exist_ok=True)
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def sha256(yol: Path):
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


EPOCH_RE = re.compile(r"Epoch: (\d+), Steps: (\d+) \| Train Loss: (\S+) Vali Loss: (\S+)")
EPOCH_SURE_RE = re.compile(r"Epoch: (\d+) cost time: (\S+)")
MASKE = [(re.compile(r"Test Loss: \S+"), "Test Loss: <kilitli>"),
         (re.compile(r"^mse:.*$"), "<test metrigi kilitli: test_kilitli/ altinda>")]


class LogKoprusu(io.TextIOBase):
    """Resmi kodun stdout'unu yakalar: ham hali kilitli dosyaya, maskelenmis hali gercek stdout'a;
    epoch satirlarindan progress.json'u gunceller (rakip koduna dokunmadan, log okuyarak)."""

    def __init__(self, gercek, ham_dosya, ilerleme_cb):
        self.gercek, self.ham, self.cb, self.tampon = gercek, ham_dosya, ilerleme_cb, ""

    def write(self, s):
        self.ham.write(s)
        self.ham.flush()
        self.tampon += s
        while "\n" in self.tampon:
            satir, self.tampon = self.tampon.split("\n", 1)
            self._satir(satir)
        return len(s)

    def _satir(self, satir):
        m = EPOCH_RE.search(satir)
        if m:
            self.cb(int(m.group(1)), float(m.group(3)), float(m.group(4)))
        m2 = EPOCH_SURE_RE.search(satir)
        if m2:
            self.cb(None, None, None, epoch_sure=float(m2.group(2)))
        for rx, yerine in MASKE:
            satir = rx.sub(yerine, satir)
        self.gercek.write(satir + "\n")
        self.gercek.flush()

    def flush(self):
        if self.tampon:
            self._satir(self.tampon)
            self.tampon = ""
        self.ham.flush()
        self.gercek.flush()


def ileri(model, args, device, batch_x, batch_y, batch_x_mark, batch_y_mark, torch):
    """Resmi test() dongusundeki ileri gecisin birebir kopyasi (yalniz hesap; agirlik/veri degismez)."""
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    if args.output_attention:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    else:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if args.features == 'MS' else 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:]
    return outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()


def hedef_dogrula_yaz(split, H, true, np, dizin=None):
    dizin = HEDEF_DIZIN if dizin is None else dizin
    dizin.mkdir(parents=True, exist_ok=True)
    f = dizin / f"{split}_h{H}_true.npy"
    if f.exists():
        ref = np.load(f)
        if ref.shape != true.shape or float(np.abs(ref - true).max()) > 1e-6:
            raise SystemExit(f"VERI HATTI DEGISTI: {f} ile bu kosumun gercek degerleri farkli")
    else:
        np.save(f, true.astype(np.float32))
    return str(f)


def pencere_kaydet(yol: Path, pred, true, ilk_satir, np):
    e = pred.astype(np.float64) - true.astype(np.float64)
    yol.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        yol, pred=pred.astype(np.float32),
        mse_pencere_degisken=(e ** 2).mean(axis=1).astype(np.float32),
        mae_pencere_degisken=np.abs(e).mean(axis=1).astype(np.float32),
        mse_pencere_adim=(e ** 2).mean(axis=2).astype(np.float32),
        hedef_ilk_satir=(ilk_satir + np.arange(pred.shape[0])).astype(np.int32))


# ------------------------------------------------------------------ ince kayit (FAZ III / H16; yalniz ince_kayit=True)
# Neden: ETTm/Weather'da pencere sayisi 4x, Weather 21 degisken -> tam pred npz'si (disk) ve pencere_kaydet'in float64
# ara dizileri (RAM; Weather h720 test 9820x720x21: her float64 kopya 1.19 GB, tepe ~6 GB yalniz burada) buyuk.
# Ince yol: ayni ifadeler PARCA pencerelik dilimlerde; pencere basina sonuc dilimlemeden bagimsiz (bit-bit, assert:
# scripts/h16_ince_dogrula.py). Metrikler (val_mse/val_mae/alt_grup, resmi test metrics.npy) DEGISMEZ.
PARCA = 512


def hedef_dogrula_yaz_ince(split, H, true, np, dizin):
    """hedef_dogrula_yaz ile ayni karar (sekil esit ve max|ref-true| <= 1e-6), referans mmap + parca parca."""
    dizin.mkdir(parents=True, exist_ok=True)
    f = dizin / f"{split}_h{H}_true.npy"
    if f.exists():
        ref = np.load(f, mmap_mode="r")
        try:
            if ref.shape != true.shape:
                raise SystemExit(f"VERI HATTI DEGISTI: {f} ile bu kosumun gercek degerleri farkli (sekil)")
            m = 0.0
            for s in range(0, true.shape[0], PARCA):
                m = max(m, float(np.abs(ref[s:s + PARCA] - true[s:s + PARCA]).max()))
            if m > 1e-6:
                raise SystemExit(f"VERI HATTI DEGISTI: {f} ile bu kosumun gercek degerleri farkli")
        finally:
            del ref
    else:
        np.save(f, np.asarray(true).astype(np.float32))
    return str(f)


def pencere_kaydet_ince(yol: Path, pred, true, ilk_satir, np):
    """pencere_kaydet'in pred'siz, parca parca esdegeri (ayni anahtarlar eksi 'pred')."""
    N, H, C = pred.shape
    mpd = np.empty((N, C), np.float32)
    apd = np.empty((N, C), np.float32)
    mpa = np.empty((N, H), np.float32)
    for s in range(0, N, PARCA):
        e = pred[s:s + PARCA].astype(np.float64) - true[s:s + PARCA].astype(np.float64)
        mpd[s:s + PARCA] = (e ** 2).mean(axis=1).astype(np.float32)
        apd[s:s + PARCA] = np.abs(e).mean(axis=1).astype(np.float32)
        mpa[s:s + PARCA] = (e ** 2).mean(axis=2).astype(np.float32)
        del e
    yol.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        yol, mse_pencere_degisken=mpd, mae_pencere_degisken=apd, mse_pencere_adim=mpa,
        hedef_ilk_satir=(ilk_satir + np.arange(N)).astype(np.int32))


def ram_tepe_mb():
    """Surecin tepe bellek kullanimi (MB): (PeakWorkingSetSize, PeakPagefileUsage = tepe ozel commit). Windows
    psapi (ctypes; psutil yok). Baska platformda (None, None)."""
    if os.name != "nt":
        return None, None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ps = ctypes.windll.psapi
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None, None
    return round(pmc.PeakWorkingSetSize / 2 ** 20, 1), round(pmc.PeakPagefileUsage / 2 ** 20, 1)


# ------------------------------------------------------------------ ana akis
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="{RUN_DIR}/metrics.json")
    ap.add_argument("--ufuklar", default="96,192,336,720")
    ap.add_argument("--epoch-ust", type=int, default=None,
                    help="YALNIZ duman/on-ucus: train_epochs=min(resmi, N). Tam kosumda verilmez.")
    ap.add_argument("--num-workers", type=int, default=None, help="varsayilan: resmi run.py degeri (10)")
    ap.add_argument("--kapi-test-metrics", action="store_true",
                    help="test_* anahtarlarini metrics.json'a da yaz (yalniz K3 replikasyon kapisi)")
    ap.add_argument("--veri", choices=sorted(VERILER), default="exchange",
                    help="FAZ II: veri seti (varsayilan exchange = bugunku davranis)")
    ap.add_argument("--kat", type=int, choices=[0, 1, 2, 3], default=None,
                    help="FAZ II P1: cok-dilimli dogrulama kati; verilirse test HIC degerlendirilmez. "
                         "0 = H12/H13 ic modeli K0 (K1'den onceki V satir val; degerlendirme kati degil)")
    ap.add_argument("--hizlandirma-kapat", action="store_true",
                    help="YALNIZ dogrulama: ince_kayit verilerinde de (a) PDF ve (b) egitim-ici vali(test) resmi haliyle")
    ap.add_argument("--ufuk-basi", action="store_true",
                    help="FAZ III/H16: tek ufuk = tam kosum (resmi .sh her pred_len'i ayri run.py sureciyle kosar); "
                         "--ufuklar tek deger olmali; kisaltilmis yalniz --epoch-ust'e bagli; metrics.json'a ufuk/veri")
    a = ap.parse_args()
    global AKTIF
    AKTIF = VERILER[a.veri]
    kat = a.kat
    ince = bool(AKTIF.get("ince_kayit", False))  # yalniz FAZ III veri girdileri (ettm1/ettm2/weather)
    if kat is not None and a.kapi_test_metrics:
        print("[rakip_kos] HATA: --kat ile --kapi-test-metrics birlikte kullanilamaz (kat kosumu test okumaz)",
              flush=True)
        return 2
    if kat is not None and a.veri not in ("exchange", "etth1", "etth2"):
        print(f"[rakip_kos] HATA: --kat {a.veri} icin tanimli degil (FAZ III'te kat protokolu yok)", flush=True)
        return 2
    if a.ufuk_basi and len(a.ufuklar.split(",")) != 1:
        print("[rakip_kos] HATA: --ufuk-basi tek ufuk ister (ornek --ufuklar 96)", flush=True)
        return 2

    out = Path(a.out).resolve()
    rd = out.parent
    rd.mkdir(parents=True, exist_ok=True)
    if kat is None:
        kilit = rd / "test_kilitli"
        kilit.mkdir(exist_ok=True)
        ham_dizin = kilit
    else:  # --kat: test yok; ham log (Test Loss: nan) ayri klasorde
        kilit = None
        ham_dizin = rd / "ham_log"
        ham_dizin.mkdir(exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    ufuklar = [int(x) for x in a.ufuklar.split(",")]
    assert all(h in UFUKLAR for h in ufuklar), ufuklar

    # veri butunlugu (K2 hash'i)
    veri = AKTIF["dizin"] / AKTIF["csv"]
    veri_hash = sha256(veri)
    if AKTIF["sha256"] is None:
        print(f"[rakip_kos] UYARI: {a.veri} icin sabit hash yok; okunan sha256={veri_hash} (metrics.json'a yazilir)",
              flush=True)
    elif veri_hash != AKTIF["sha256"]:
        raise SystemExit(f"VERI HASH UYUSMUYOR: {veri}")

    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        print("[rakip_kos] HATA: CUDA yok -- GPU zorunlu kosum, CPU'ya dusus yok.", flush=True)
        return 97
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from data_provider.data_factory import data_dict
    from torch.utils.data import DataLoader
    hiz = ince and not a.hizlandirma_kapat
    if hiz:
        # FAZ III / H16 sef karari (2026-10-15): yalniz ince_kayit verileri; metrik/RNG notr, SUREC ICI (rakip dosyasi ayni).
        # (a) resmi test()'in her 20. pencerede cizdigi PDF (utils.tools.visual; plt.figure kapatilmiyor) -> no-op.
        #     Yalniz matplotlib yan urunu; tahmin/metrik/RNG'ye dokunmaz.
        # (b) resmi train() her epoch'ta vali(test_data, test_loader) cagirir; sonucu yalniz log satiri (Test Loss,
        #     zaten maskeli), erken durdurma vali(val) ile. Test DataLoader'i (shuffle yok, SequentialSampler)
        #     iter() edildiginde torch global uretecinden TEK cekilis (_BaseDataLoaderIter._base_seed) yapar, dongu
        #     baska cekilis yapmaz, eval'de dropout yok -> iter() acilip birakilir: RNG akisi ve model durumu ayni,
        #     log'da Test Loss nan. Kanit: scripts/h16_regresyon_karsilastir.py (acik vs kapali, ETTm1 h96 2 epoch).
        import experiments.exp_long_term_forecasting as _elf
        _elf.visual = lambda *x, **k: None
        _vali_resmi = Exp_Long_Term_Forecast.vali

        def _vali_h16(self, vali_data, vali_loader, criterion):
            if getattr(vali_data, "set_type", None) == 2:  # yalniz train() icindeki test vali'si
                _it = iter(vali_loader)  # RNG cekilisi korunur
                del _it
                return float("nan")
            return _vali_resmi(self, vali_data, vali_loader, criterion)

        Exp_Long_Term_Forecast.vali = _vali_h16
    kat_ek = {}
    if kat is not None:
        # data_dict SUREC ICINDE yamalanir (rakip dosyasi degismez); data_provider() ve asagidaki val olcumu ayni
        # sozlugu okur. Yalniz bu veri setinin anahtari degisir.
        for anahtar in {"exchange": ["custom"], "etth1": ["ETTh1"], "etth2": ["ETTh2"]}[a.veri]:
            data_dict[anahtar] = kat_veri_sinifi(data_dict[anahtar], AKTIF["bolme"], kat)
        kat_ek = {"veri": a.veri, "veri_sha256": veri_hash, "kat": kat}

    parser, resmi_seed = resmi_parser()
    sh = resmi_sh_argumanlari()
    cihaz = f"cuda:0 / {torch.cuda.get_device_name(0)}"
    print(f"[rakip_kos] rakip commit {RAKIP_COMMIT} | run.py fix_seed={resmi_seed} -> sarmalayici seed={a.seed}"
          f" | cihaz {cihaz} | torch {torch.__version__} | ufuklar {ufuklar}"
          f" | epoch_ust={a.epoch_ust} | num_workers={a.num_workers or 'resmi'}", flush=True)
    if kat is not None:
        print(f"[rakip_kos] --veri {a.veri} --kat {kat}: test HIC KULLANILMAZ (0 pencere, nan kayip, test() yok)",
              flush=True)

    ilerleme = {"adim": 0, "ufuk": None, "epoch": 0, "loss": None, "val_loss": None}
    pj = rd / "progress.json"

    def ilerleme_cb(ep, tr, vl, epoch_sure=None):
        if ep is None:
            ilerleme["son_epoch_sure_sn"] = epoch_sure
        else:
            ilerleme.update({"adim": ilerleme["adim"] + 1, "epoch": ep, "loss": tr, "val_loss": vl})
            ilerleme.setdefault("gecmis", []).append([ilerleme["ufuk"], ep, tr, vl])
        ilerleme["zaman"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomik_json(pj, ilerleme)

    metrik, test_m, ek = {}, {}, {}
    alt_grup = {}
    t_top = time.time()
    gercek_stdout = sys.stdout

    for H in ufuklar:
        argv = sh[H]
        argv = argv_degistir(argv, "--root_path", str(AKTIF["dizin"]) + os.sep)
        argv = argv_degistir(argv, "--data_path", AKTIF["csv"])
        argv = argv_degistir(argv, "--checkpoints", str(rd / "ckpt") + os.sep)
        args = parser.parse_args(argv)
        # run.py: args.use_gpu = cuda_available and use_gpu  -> burada dogrulanmis; sessiz dusus yok
        args.use_gpu = True
        args.gpu = 0
        epoch_resmi = args.train_epochs
        if a.epoch_ust is not None:
            args.train_epochs = min(args.train_epochs, a.epoch_ust)
        if a.num_workers is not None:
            args.num_workers = a.num_workers
        ek[f"epoch_ayar_h{H}"] = args.train_epochs
        ek[f"epoch_resmi_h{H}"] = epoch_resmi

        # --- seed baglama: run.py'deki uc cagri, Exp'ten hemen once (her ufuk ayri surecteki gibi)
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)

        ilerleme["ufuk"] = H
        torch.cuda.reset_peak_memory_stats()
        setting = setting_adi(args)
        ham = open(ham_dizin / f"rakip_stdout_h{H}.log", "w", encoding="utf-8")
        sys.stdout = LogKoprusu(gercek_stdout, ham, ilerleme_cb)
        t0 = time.time()
        try:
            print("Args in experiment:")
            print(args)
            exp = Exp_Long_Term_Forecast(args)
            ek[f"param_sayisi_h{H}"] = int(sum(p.numel() for p in exp.model.parameters()))
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)  # resmi: erken durdurma + en iyi checkpoint yuklenir
            t_egitim = time.time() - t0

            # --- val (bizim tanim = test tanimi): tum pencereler, batch 1, shuffle/drop_last yok
            Data = data_dict[args.data]
            vset = Data(root_path=args.root_path, data_path=args.data_path, flag='val',
                        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                        target=args.target, timeenc=0 if args.embed != 'timeF' else 1, freq=args.freq)
            vload = DataLoader(vset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            exp.model.eval()
            P, T = [], []
            with torch.no_grad():
                for bx, by, bxm, bym in vload:
                    p_, t_ = ileri(exp.model, args, exp.device, bx, by, bxm, bym, torch)
                    P.append(p_); T.append(t_)
            vp = np.concatenate(P, 0); vt = np.concatenate(T, 0)
            assert vp.shape[0] == len(vset), (vp.shape, len(vset))
            v_mse = float(np.mean((vp - vt) ** 2)); v_mae = float(np.mean(np.abs(vp - vt)))
            if ince:
                # ince kayit: val ciktilari test()'ten ONCE (ayni ifadeler), sonra buyuk diziler bosaltilir -> resmi
                # test()'in kendi preds/trues listeleriyle ust uste binmez (RAM tepesi)
                ag_val = {f"val_mse_h{H}_d{j}": float(np.mean((vp[:, :, j] - vt[:, :, j]) ** 2))
                          for j in range(vp.shape[2])}
                hedef_dogrula_yaz_ince("val", H, vt, np, AKTIF["hedef"])
                pencere_kaydet_ince(rd / "preds" / f"val_h{H}.npz", vp, vt, vset.seq_len + AKTIF["val_b1"], np)
                del P, T, vp, vt

            # --- test: resmi test() (batch 1); ciktilari kilitli klasore (--kat: test YOK)
            if kat is None:
                cwd0 = os.getcwd()
                os.chdir(kilit)
                try:
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)
                finally:
                    os.chdir(cwd0)
            torch.cuda.empty_cache()
        finally:
            sys.stdout.flush()
            sys.stdout = gercek_stdout
            ham.close()
        t_h = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 2 ** 20

        if kat is None and ince:
            rdir = kilit / "results" / setting
            mae_t, mse_t = [float(x) for x in np.load(rdir / "metrics.npy")[:2]]
            tp = np.load(rdir / "pred.npy", mmap_mode="r"); tt = np.load(rdir / "true.npy", mmap_mode="r")
            # tutarlilik: eski yolla ayni ifade (float32 np.mean) -> ayni karar
            if abs(float(np.mean((tp - tt) ** 2)) - mse_t) > 1e-6:
                raise SystemExit("test metrigi pred/true ile tutarsiz")
            hedef_dogrula_yaz_ince("test", H, tt, np, AKTIF["hedef"])
            pencere_kaydet_ince(rd / "preds" / "test_kilitli" / f"test_h{H}.npz", tp, tt,
                                args.seq_len + AKTIF["test_b1"], np)
            del tp, tt
            import gc
            gc.collect()  # mmap kollari kapanmadan Windows'ta unlink basarisiz olur
            for f in ("pred.npy", "true.npy"):
                (rdir / f).unlink()  # ince yol: sessiz gecis yok (disk); basarisizsa kosum hata verir
            test_m[f"test_mse_h{H}"] = mse_t
            test_m[f"test_mae_h{H}"] = mae_t
            ek[f"ram_tepe_ws_mb_h{H}"], ek[f"ram_tepe_commit_mb_h{H}"] = ram_tepe_mb()
        elif kat is None:
            rdir = kilit / "results" / setting
            mae_t, mse_t = [float(x) for x in np.load(rdir / "metrics.npy")[:2]]
            tp = np.load(rdir / "pred.npy"); tt = np.load(rdir / "true.npy")
            # tutarlilik: resmi metrik == kaydedilen tahminlerden yeniden hesap
            if abs(float(np.mean((tp - tt) ** 2)) - mse_t) > 1e-6:
                raise SystemExit("test metrigi pred/true ile tutarsiz")
            hedef_dogrula_yaz("val", H, vt, np, AKTIF["hedef"])
            hedef_dogrula_yaz("test", H, tt, np, AKTIF["hedef"])
            pencere_kaydet(rd / "preds" / f"val_h{H}.npz", vp, vt, vset.seq_len + AKTIF["val_b1"], np)
            pencere_kaydet(rd / "preds" / "test_kilitli" / f"test_h{H}.npz", tp, tt, args.seq_len + AKTIF["test_b1"],
                           np)
            # resmi pred/true kopyalari npz'ye alindi; tekrari silinir (metrics.npy kalir)
            for f in ("pred.npy", "true.npy"):
                try:
                    (rdir / f).unlink()
                except OSError:
                    pass
            test_m[f"test_mse_h{H}"] = mse_t
            test_m[f"test_mae_h{H}"] = mae_t
        else:
            kb = vset.kat_bilgi
            if kat == 3:  # k3 == resmi val: mevcut resmi val hedefleriyle esit olmali (veri hatti saglamasi)
                hedef_dogrula_yaz("val", H, vt, np, AKTIF["hedef"])
            hedef_dogrula_yaz(f"val_k{kat}", H, vt, np, AKTIF["hedef"])
            pencere_kaydet(rd / "preds" / f"val_h{H}.npz", vp, vt, kb["satir"][0] + vset.seq_len, np)
            kat_ek.update({"kat_val_satir": [kb["val_bas"], kb["val_son"]], "kat_train_satir": [0, kb["val_bas"]],
                           "kat_val_girdi_satir": kb["satir"], "kat_V": kb["V"], "kat_T_resmi": kb["T_resmi"],
                           f"kat_pencere_val_h{H}": int(len(vset)),
                           f"kat_pencere_train_h{H}": int(kb["val_bas"] - vset.seq_len - H + 1),
                           "kat_scaler_ort": [float(x) for x in vset.scaler.mean_],
                           "kat_scaler_std": [float(x) for x in vset.scaler.scale_]})

        gecmis = [g for g in ilerleme.get("gecmis", []) if g[0] == H]
        metrik[f"val_mse_h{H}"] = v_mse
        metrik[f"val_mae_h{H}"] = v_mae
        ek[f"epoch_kosulan_h{H}"] = len(gecmis)
        ek[f"val_resmi_kayip_min_h{H}"] = float(min(g[3] for g in gecmis)) if gecmis else float("nan")
        ek[f"sure_sn_h{H}"] = round(t_h, 2)
        ek[f"egitim_sure_sn_h{H}"] = round(t_egitim, 2)
        ek[f"vram_tepe_mb_h{H}"] = round(vram, 1)
        if ince:
            alt_grup.update(ag_val)  # ayni ifadeyle test()'ten once hesaplandi
        else:
            for j in range(vp.shape[2]):
                alt_grup[f"val_mse_h{H}_d{j}"] = float(np.mean((vp[:, :, j] - vt[:, :, j]) ** 2))
        ilerleme[f"val_mse_h{H}"] = v_mse
        atomik_json(pj, ilerleme)
        print(f"[rakip_kos] h{H}: val_mse={v_mse:.6f} val_mae={v_mae:.6f} epoch={len(gecmis)}/{args.train_epochs}"
              f" sure={t_h:.1f}s vram_tepe={vram:.0f}MB "
              + ("(test hesaplandi, kilitli)" if kat is None else f"(--kat {kat}: test yok)"), flush=True)

    hs = [h for h in UFUKLAR if h in ufuklar]
    sonuc = {
        "val_mse_avg": float(np.mean([metrik[f"val_mse_h{h}"] for h in hs])),
        "val_mae_avg": float(np.mean([metrik[f"val_mae_h{h}"] for h in hs])),
        **{f"val_mse_h{h}": metrik[f"val_mse_h{h}"] for h in hs},
        **{f"val_mae_h{h}": metrik[f"val_mae_h{h}"] for h in hs},
        "egitim_sure_sn": round(time.time() - t_top, 1),
        "cihaz": cihaz,
        "seed": a.seed,
        "ufuk_sayisi": len(hs),
        "kisaltilmis": int(a.epoch_ust is not None or len(hs) < 4),
        "param_sayisi": ek.get(f"param_sayisi_h{hs[-1]}"),
        **ek,
        **kat_ek,
        "alt_grup": alt_grup,
    }
    if a.ufuk_basi or ince:  # FAZ III ek anahtarlari (exchange/etth* --ufuk-basi'siz yolda YOK -> eski sema aynen)
        if a.ufuk_basi:
            # resmi .sh: her pred_len ayri run.py sureci = tam protokol; tek ufuk kisaltma DEGIL
            sonuc["kisaltilmis"] = int(a.epoch_ust is not None)
            sonuc["ufuk"] = hs[0]
            sonuc["ufuk_basi"] = 1
        sonuc["veri"] = a.veri
        sonuc["veri_sha256"] = veri_hash
        sonuc["pred_kaydedildi"] = int(not ince)
        sonuc["hizlandirma_pdf_yok_vali_test_yok"] = int(hiz)
        sonuc["ram_tepe_ws_mb"], sonuc["ram_tepe_commit_mb"] = ram_tepe_mb()
    if kat is not None:  # test yok: test_metrics.json yazilmaz
        atomik_json(out, sonuc)
        print(f"[rakip_kos] bitti (--veri {a.veri} --kat {kat}): val_mse_avg={sonuc['val_mse_avg']:.6f} "
              f"sure={sonuc['egitim_sure_sn']}s -> {out}", flush=True)
        return 0
    test_sonuc = {
        "test_mse_avg": float(np.mean([test_m[f"test_mse_h{h}"] for h in hs])),
        "test_mae_avg": float(np.mean([test_m[f"test_mae_h{h}"] for h in hs])),
        **{f"test_mse_h{h}": test_m[f"test_mse_h{h}"] for h in hs},
        **{f"test_mae_h{h}": test_m[f"test_mae_h{h}"] for h in hs},
        "seed": a.seed, "ufuk_sayisi": len(hs), "kisaltilmis": sonuc["kisaltilmis"],
        "not": "KILITLI: K4 ajanlarina verilmez; yalniz K3 replikasyon kapisi ve K5.",
    }
    if a.ufuk_basi:
        test_sonuc.update({"ufuk": hs[0], "ufuk_basi": 1, "veri": a.veri})
    atomik_json(rd / "test_metrics.json", test_sonuc)
    if a.kapi_test_metrics:
        # replikasyon_kapisi.py metrigi .done['metrics'] (= metrics.json) icinden okur -> K3 istisnasi
        sonuc.update({k: v for k, v in test_sonuc.items() if k.startswith("test_")})
        sonuc["test_metrikleri_metrics_jsonda"] = 1
    atomik_json(out, sonuc)
    print(f"[rakip_kos] bitti: val_mse_avg={sonuc['val_mse_avg']:.6f} sure={sonuc['egitim_sure_sn']}s -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
