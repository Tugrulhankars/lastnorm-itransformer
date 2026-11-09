# H16: yalnız-train kalıcılık oranı R ve önceden yazılmış işaret tahminleri

*Veri-hazırlayıcı, FAZ III. Bu dosya H16 sınamasındaki model koşumlarından ÖNCE yazıldı; ETTm1, ETTm2, Weather ve
Solar için henüz hiçbir iTransformer ya da LastNorm koşumu yapılmadı.*

**Zaman damgaları**
- `hipotezler/H16.md` son değişiklik: 2026-10-15 05:49:07.
- R hesabı (`sonuclar/h16-R.csv` yazımı): 2026-10-15 06:04:37.
- Bu dosya: 2026-10-15 ~06:10.

**Komut** (WS kökünden): `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I -X utf8 scripts/h16_R.py`.
Çıktıları `sonuclar/h16-R.csv`, `sonuclar/h16-R.json` ve `sonuclar/h16-R.log`. Süre 51 s, yalnız CPU.

## Ölçüt (H16.md'den birebir; `scripts/h16_R.py` docstring'i)

- Kullanılan satırlar yalnız resmî split'in train dilimi [0, num_train). CSV okunduktan hemen sonra val ve test
  satırları atılıyor; hiçbir hesaba girmiyorlar.
- Ölçekleme: StandardScaler yalnız train'e fit ediliyor (resmî iTransformer scaler'ıyla aynı).
- Pencereler: L = 96, adım 7, s = 0, 7, 14, …, s + 96 + H ≤ num_train koşuluyla.
- R(H) = ort. MSE(son değeri H adım tekrarla) / ort. MSE(96'lık pencere ortalamasını H adım tekrarla). MSE, H adım
  × tüm değişkenler (features=M) üzerinden alınıyor; ortalama, pencereler üzerinden.
- R̄ = 4 ufkun (96/192/336/720) R ortalaması.
- Tahmin kuralı:
  - R̄ < 1 ⇒ LastNorm test_mse_avg < iTransformer → **iyileşir**
  - R̄ > 1 ⇒ LastNorm ≥ iTransformer → **kötüleşir**
  - |R̄ − 1| < 0.05 ⇒ **belirsiz**

## Kalibrasyon (sınama değil; H16.md beyanı): assert |R̄ − beklenen| ≤ 0.002

| Veri | num_train | C | R(96) | R(192) | R(336) | R(720) | **R̄** | beklenen | fark | gözlenen LastNorm Δ% (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exchange | 5311 | 8 | 0.5806 | 0.7517 | 0.8554 | 0.9281 | **0.7790** | 0.779 | 0.0000 ✓ | −4.96% |
| ETTh1 | 8640 | 7 | 1.5720 | 1.5648 | 1.5433 | 1.4795 | **1.5399** | 1.540 | 0.0001 ✓ | +3.82% |
| ETTh2 | 8640 | 7 | 1.1119 | 1.1915 | 1.2077 | 1.1968 | **1.1770** | 1.177 | 0.0000 ✓ | +1.88% |

Kalibrasyon değerleri, R tanımının bu betikte birebir uygulandığını gösteriyor.

## Sınama setleri: tahminler (değiştirilemez)

| Veri | num_train | C | R(96) | R(192) | R(336) | R(720) | **R̄** | \|R̄−1\| | **Tahmin** |
|---|---|---|---|---|---|---|---|---|---|
| ETTm1 | 34560 | 7 | 1.5236 | 1.5608 | 1.5483 | 1.5113 | **1.5360** | 0.536 | **kötüleşir** (LastNorm ≥ iTransformer) |
| ETTm2 | 34560 | 7 | 1.0431 | 1.0795 | 1.1248 | 1.1395 | **1.0967** | 0.097 | **kötüleşir** (LastNorm ≥ iTransformer) |
| Weather | 36887 | 21 | 1.3865 | 1.2702 | 1.3143 | 1.3559 | **1.3317** | 0.332 | **kötüleşir** (LastNorm ≥ iTransformer) |
| Solar | 36792 | 137 | 1.7291 | 1.4368 | 1.5444 | 1.6991 | **1.6024** | 0.602 | **kötüleşir** (LastNorm ≥ iTransformer) |

- Belirsiz set yok. En yakın set ETTm2 (|R̄−1| = 0.097, bandın yaklaşık iki katı).
- ETTm2'de R(96) = 1.043 tek başına bandın içinde kalırdı; ancak kural R̄ üzerinden işler.
- 16 sınama hücresinin (4 set × 4 ufuk) hepsinde R(H) > 1.
- Başarı tanımı (H16.md): belirsiz olmayan 4 setin dördünde de LastNorm test_mse_avg ≥ iTransformer test_mse_avg
  (3 seed ortalaması). Kalibrasyon dahil Spearman(R̄, Δ%) > 0 olmalı.
- Öldürme koşulu: ≥1 sette LastNorm < iTransformer çıkarsa sonuç "R rejimi öngörmez" diye raporlanır.

## Şefin bilmesi gerekenler (yorum; tahmini değiştirmez)

1. **Sınama tek yönlü kaldı.** Dört yeni setin hepsi R̄ > 1 tarafında. Bu yüzden ileriye dönük sınama yalnız
   "LastNorm burada kazanmaz" yönünü sınayabiliyor. R̄ < 1 → "iyileşir" yönünün tek dayanağı hâlâ kalibrasyon seti
   Exchange. "R rejimi öngörür" iddiası yazılırsa bu asimetri beyan edilmeli. Set listesi R'den önce sabitlendi ve
   değiştirilmemeli (H16.md: "sonuca bakarak kesilmez").
2. **Spearman (ikincil) için 7 nokta olacak** (3 kalibrasyon + 4 sınama). Yeni setlerde R̄'nin sırası şöyle:
   Solar 1.602 > ETTm1 1.536 > Weather 1.332 > ETTm2 1.097. Kalibrasyonla birlikte ön-kayıtlı öngörü, Δ%'nin de bu
   sırayla artması yönünde.
3. **Weather duyarlılığı (ön-kayıtlı DEĞİL, yalnız sağlamlık bilgisi).** Weather train'inde −9999 gözcü değerleri
   var: OT 50, max.PAR 30, wv 1 hücre; hepsi train içinde, `data/data_manifest_faz3.md`. Bu değerlere dokunan
   pencereler çıkarıldığında R̄ = 1.3313 oluyor (scaler da bu satırlar hariç fit edilirse 1.3457). Tahmin bu
   değerlere duyarlı değil. Komut: `env/.venv/Scripts/python.exe -I scripts/h16_weather_duyarlilik.py`. Kural R̄'yi resmî ham veriyle hesaplar ve
   tablodaki 1.3317 geçerlidir.
4. **R, LastNorm'un iTransformer'a göre etkisi hakkında yalnız dolaylı bir göstergedir.** R, train'de "son değer mi
   pencere ortalaması mı daha iyi naif tahmin" sorusunu ölçer. Test'in train'e göre kayması (Weather test std oranı
   0.54, ETTm2 test |Δμ_z| 1.29) R'ye girmez. Bu, H16'nın tasarım kararıdır; burada yalnız not ediliyor.
