# H16 — ileriye dönük rejim sınaması: TEST tek açılış 2026-10-15T14:10:52

| veri | R̄ | tahmin | iTr test (ort±sd) | LN test | Δ% | eşli t p | LN<iTr seed | ufuk LN<iTr | val Δ% | lit iTr ±tol | replikasyon | tahmin doğru? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ettm1 | 1.536 | kötüleşir | 0.4074±0.0002 | 0.4080±0.0017 | +0.15 | 0.605 | 2/3 | 1/4 | +0.81 | 0.407±0.009 | YEŞİL | EVET |
|  | h96 |  | 0.3434 | 0.3445 | +0.32 |  |  |  |  |  |  |  |
|  | h192 |  | 0.3802 | 0.3819 | +0.44 |  |  |  |  |  |  |  |
|  | h336 |  | 0.4183 | 0.4189 | +0.16 |  |  |  |  |  |  |  |
|  | h720 |  | 0.4877 | 0.4867 | -0.20 |  |  |  |  |  |  |  |
| ettm2 | 1.097 | kötüleşir | 0.2915±0.0007 | 0.2888±0.0002 | -0.90 | 0.0146 | 3/3 | 4/4 | +0.13 | 0.288±0.008 | YEŞİL | HAYIR |
|  | h96 |  | 0.1846 | 0.1821 | -1.31 |  |  |  |  |  |  |  |
|  | h192 |  | 0.2536 | 0.2499 | -1.48 |  |  |  |  |  |  |  |
|  | h336 |  | 0.3146 | 0.3124 | -0.70 |  |  |  |  |  |  |  |
|  | h720 |  | 0.4131 | 0.4110 | -0.52 |  |  |  |  |  |  |  |

**Önceden tanımlı başarı (belirsiz-olmayan tüm yeni setlerde işaret doğru):** TUTMADI (1/2)
**İkincil:** Spearman(R̄, Δ%) 5 set (3 kalibrasyon + 2 sınama) ρ=0.900 (p=0.0374); değerler: exchange R̄=0.779 Δ=-4.96%, etth1 R̄=1.540 Δ=+3.82%, etth2 R̄=1.177 Δ=+1.88%, ettm1 R̄=1.536 Δ=+0.15%, ettm2 R̄=1.097 Δ=-0.90%
**Beyan:** sınama tek yönlü (yeni setlerin hepsi R̄>1); kalibrasyon setleri sınama değildir; Solar/Weather sonuç görülmeden kesildi.
