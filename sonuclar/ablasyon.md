# Ablasyon (K5): kazanan `lastnorm` (H04)

*deney-tahlilci (mod 3) · 2026-10-14 · yeni koşum YOK. Kaynak: mevcut n = 3 tam koşumlar (aşama 3, rc = 0).*
*Betik: `scripts/k5_test_ac.py --asama val` (önce, yalnız val sütunları) → `--asama test` (test sütunları eklendi). Ham tablo: `sonuclar/ablasyon.csv`.*

Zincir: baseline (NST: μ_lb merkezleme + σ_lb ölçek) → **lastnorm** (merkez = son değer x_son, σ_lb korunur).
Tek-değişken kuralı sayesinde ablasyon satırları ebeveyn/kardeş koşumlardan geliyor:
- "lastnorm − çapa" = baseline (merkez μ_lb'ye geri döner)
- "lastnorm − σ" = lastnorm-merkez (H10)
- "+ ep336eq" = lastnorm-ep336eq (H08) / baseline-ep336eq (H03), protokol eki
- "− normalizasyon (çapa ve σ)" = normsuz (H06)
- referans: RevIN-afin (H02)

Seed {2023, 2024, 2025}, eşleştirilmiş. Δ referansa göre göreli. p eşleştirilmiş t. "ufuk" = Δ_H < 0 olan ufuk sayısı.

| Satır | Varyant (koşum id) | Ref. | val_mse_avg | val Δ% (p, ufuk) | test_mse_avg | test Δ% (p, ufuk) | test h96 / h192 / h336 / h720 |
|---|---|---|---|---|---|---|---|
| 0 | baseline NST (020531/021024/021448) | — | 0.4448 ± 0.0119 | — | **0.3646 ± 0.0008** | — | 0.0866 / 0.1794 / 0.3351 / 0.8574 |
| 1 | **lastnorm = KAZANAN** (052508/053601/054213) | baseline | 0.4278 ± 0.0059 | −3.83% (0.228, 4/4) | **0.3465 ± 0.0019** | **−4.96% (0.0064, 4/4)** | 0.0830 / 0.1728 / 0.3152 / 0.8151 |
| 2 | lastnorm − σ (H10; 143314/143820/144341) | lastnorm | 0.4321 ± 0.0107 | +1.01% (0.687, 1/4) | 0.3818 ± 0.0096 | **+10.18% (0.023, 0/4)** | 0.0850 / 0.1775 / 0.3311 / 0.9336 |
| 3 | lastnorm + ep336eq (H08; 140921/141641/142407) | lastnorm | 0.4191 ± 0.0051 | −2.04% (0.003; yalnız h336 değişir) | 0.3473 ± 0.0027 | +0.22% (0.271; h336 +0.95%) | 0.0830 / 0.1728 / 0.3182 / 0.8151 |
| 4 | baseline + ep336eq (H03; 031233/032853/033500) | baseline | 0.4376 ± 0.0126 | −1.62% (0.003; yalnız h336) | 0.3646 ± 0.0003 | −0.00% (0.993; h336 −0.01%) | 0.0866 / 0.1794 / 0.3351 / 0.8574 |
| 5 | normalizasyon yok (H06; 055518/060205/060859) | baseline | 2.1892 ± 0.0401 | +392% (0.0002, 0/4) | 0.9644 ± 0.0162 | +164% (0.0003, 0/4) | 0.3685 / 0.5991 / 1.4461 / 1.4439 |
| 6 | RevIN-afin (H02; 034304/034735/035213) | baseline | 0.4441 ± 0.0123 | −0.17% (0.134, 2/4) | 0.3653 ± 0.0006 | +0.17% (0.027, 0/4) | 0.0867 / 0.1799 / 0.3354 / 0.8591 |

Epoch-eşit protokolde çapa etkisi (lastnorm-ep336eq vs baseline-ep336eq): val −4.24% (p = 0.20, 4/4), **test −4.75% (p = 0.0070, 4/4)**.

## Okuma
1. **Her iki bileşen gerekli, ablasyon tutarlı (test).**
   - Çapayı geri almak (→ baseline): +5.2% (lastnorm'a göre; tersi −4.96%).
   - σ'yı geri almak (→ lastnorm-merkez): +10.2%, 0/4 ufuk, en büyük zarar h720'de (+14.5%).
   - İkisini birden geri almak (normsuz): +178% (lastnorm'a göre).
   - Kazanç ne yalnız merkezlemeden ne yalnız ölçekten geliyor. "Son değere merkezle + pencere σ'sıyla ölçekle" birleşimi taşıyor.
2. **Val σ katkısını küçümsedi.** Val'de −σ yalnız +1.0% idi (h336'da iyileşme bile vardı). Test'te +10.2% ve 4/4 ufukta zarar var. tahlil-tur3 §5.2'deki "normalizasyonun değeri neredeyse tümüyle merkezlemeden" cümlesi **test'te geçerli değil**: σ test'te taşıyıcı.
3. **ep336eq eki test'te nötr.**
   - val h336: −8.9% (lastnorm), −7.2% (baseline).
   - test h336: +0.95% (p = 0.27) ve −0.01% (p = 0.99).
   - Resmî betiğin h336 için 1-epoch ayarı test'te kayıp yaratmıyor. Val'deki eksik-eğitim etkisi test'e taşınmıyor.
   - Kazananın etkisi iki protokolde de aynı (−4.96% / −4.75%), yani protokol seçimine duyarsız.
4. **RevIN-afin** test'te de nötr: +0.17%, kesir düzeyinde ama tutarlı kötü.

## Sınırlılık
- n = 3 seed. Seed varyansı yalnız başlatma gürültüsünü ölçer; tek test dönemi (1517 satır) örnekleme belirsizliğini kapsamaz.
- Ablasyon satırları ayrı kartlar olarak ön-kayıtlıydı. Hiçbiri K5'te yeni koşulmadı.
