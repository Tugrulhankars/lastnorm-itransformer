"""README grafiklerini uretir: her grafik icin acik (*-light.svg) ve koyu (*-dark.svg) tema.

Calistirma (depo kokunden):  python docs/figures/make_figures.py
Gereken: matplotlib >= 3.7. Tek veri kaynagi ayni klasordeki veri.json'dur.

Tasarim kurali (on-dikkat / pre-attentive): her grafikte tek bir vurgu rengi vardir ve
o renk yalniz LastNorm'a aittir. iTransformer gri, Repeat (naif taban) siyah/beyaz elmastir.
Okurun gozunun once gitmesi gereken oge turuncudur; geri kalan her sey geri plandadir.
Lejant yerine dogrudan etiket kullanilir.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

KOK = Path(__file__).resolve().parent
V = json.loads((KOK / "veri.json").read_text(encoding="utf-8"))

TEMALAR = {
    "light": dict(metin="#1f2328", ikincil="#59636e", gri="#afb8c1", grid="#eaeef2",
                  vurgu="#d95f02", vurgu_acik="#f4b183", repeat="#1f2328", zemin="none"),
    "dark": dict(metin="#e6edf3", ikincil="#9198a1", gri="#5c636b", grid="#262c33",
                 vurgu="#f0883e", vurgu_acik="#7a4520", repeat="#e6edf3", zemin="none"),
}


def stil(t):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "svg.fonttype": "path", "svg.hashsalt": "lastnorm",
        "text.color": t["metin"], "axes.labelcolor": t["ikincil"],
        "xtick.color": t["ikincil"], "ytick.color": t["ikincil"],
        "axes.edgecolor": t["gri"], "axes.facecolor": "none", "figure.facecolor": "none",
        "savefig.facecolor": "none", "savefig.transparent": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": t["metin"], "axes.titlepad": 10,
    })


def kaydet(fig, ad, tema):
    yol = KOK / f"{ad}-{tema}.svg"
    fig.savefig(yol, format="svg", bbox_inches="tight", pad_inches=0.08, metadata={"Date": None})
    plt.close(fig)
    print("yazildi:", yol.relative_to(KOK.parent.parent))


def fmt(x, n=2):
    return f"{x:+.{n}f}".replace("-", "−")


# ---------------------------------------------------------------- 1. hero
def hero(tema):
    t = TEMALAR[tema]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.42))

    # (a) temsili seri
    gecmis = [28.2, 28.6, 29.0, 29.5, 30.1, 30.8, 31.4, 32.4]
    gelecek = [32.4, 32.5, 32.3, 32.6, 32.7]
    mu = sum(gecmis) / len(gecmis)
    a.axvspan(8.5, 12.6, color=t["grid"], lw=0)
    a.text(10.55, 27.75, "tahmin ufku", ha="center", va="bottom", color=t["ikincil"], fontsize=9.5)
    a.plot(range(1, 9), gecmis, "-o", color=t["metin"], ms=4.5, lw=1.6, zorder=3)
    a.plot(range(8, 13), gelecek, "--o", color=t["ikincil"], ms=4, lw=1.2, mfc="none", zorder=3)
    a.plot([1, 12.4], [mu, mu], "--", color=t["gri"], lw=2)
    a.plot([8, 12.4], [gecmis[-1]] * 2, "-", color=t["vurgu"], lw=2.6, zorder=4)
    a.annotate("", xy=(9.4, gecmis[-1] - 0.08), xytext=(9.4, mu + 0.08),
               arrowprops=dict(arrowstyle="<->", color=t["metin"], lw=1.4))
    a.text(9.6, (mu + gecmis[-1]) / 2, "ağın kendisinin\nüretmesi gereken\nfark", va="center", fontsize=9, color=t["metin"])
    a.text(0.6, mu + 0.12, "iTransformer merkezi: ortalama μ", color=t["ikincil"], fontsize=9.5, va="bottom")
    a.text(8.25, gecmis[-1] + 0.25, "LastNorm merkezi: son değer", color=t["vurgu"], fontsize=9.5,
           ha="right", va="bottom", fontweight="bold")
    a.set_xlim(0.4, 12.6), a.set_ylim(27.6, 33.4)
    a.set_xticks([]), a.set_yticks([])
    a.set_xlabel("zaman →"), a.set_ylabel("değer")
    a.set_title("Sorun: ortalama, son değerden uzakta kalıyor")

    # (b) bes veri seti
    s = V["bes_veri_seti"]["satirlar"]
    adlar = [r["veri"] for r in s][::-1]
    dp = [r["dpct"] for r in s][::-1]
    renk = [t["vurgu"] if r == "Exchange" else t["gri"] for r in adlar]
    y = range(len(adlar))
    b.barh(y, dp, color=renk, height=0.62, zorder=2)
    b.axvline(0, color=t["ikincil"], lw=1)
    for yi, d, ad in zip(y, dp, adlar):
        vurgulu = ad == "Exchange"
        b.text(d + (0.15 if d >= 0 else -0.15), yi, fmt(d) + "%", va="center", ha="left" if d >= 0 else "right",
               color=t["vurgu"] if vurgulu else t["ikincil"], fontweight="bold" if vurgulu else "normal", fontsize=9.5)
    rep = V["bes_veri_seti"]["repeat_exchange_dpct"]["deger"]
    yex = adlar.index("Exchange")
    b.plot([rep], [yex + 0.47], marker="D", ms=6.5, color=t["repeat"], ls="none", zorder=4)
    b.text(rep + 0.25, yex + 0.47, f"Repeat (naif): {fmt(rep)}%", va="center", fontsize=9, color=t["metin"])
    b.set_yticks(list(y), adlar)
    b.set_xlim(-8.6, 5.6), b.set_ylim(-0.6, len(adlar) - 0.15)
    b.set_xlabel("LastNorm'un MSE değişimi [%]   ← daha düşük hata")
    b.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_title("Sonuç: kazanç döviz kurunda, ETT'de karışık")
    kaydet(fig, "hero", tema)


# ---------------------------------------------------------------- 2. exchange ufuk
def exchange_ufuk(tema):
    t = TEMALAR[tema]
    e = V["exchange_ufuk"]
    H = e["ufuklar"]
    itr, ln, rp = e["itransformer"]["deger"], e["lastnorm"]["deger"], e["repeat"]["deger"]
    ln_d = [100 * (l / i - 1) for l, i in zip(ln, itr)]
    rp_d = [100 * (r / i - 1) for r, i in zip(rp, itr)]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    y = list(range(len(H)))[::-1]
    for yi, l, r in zip(y, ln_d, rp_d):
        ax.plot([l, 0], [yi, yi], color=t["vurgu_acik"], lw=2.2, zorder=1, solid_capstyle="round")
        ax.plot([r], [yi], marker="D", ms=6.5, color=t["repeat"], ls="none", zorder=3)
        ax.plot([l], [yi], marker="o", ms=9, color=t["vurgu"], ls="none", zorder=4)
        ax.text(l, yi + 0.3, fmt(l) + "%", ha="center", va="bottom", color=t["vurgu"], fontweight="bold", fontsize=9.5)
        ax.text(r, yi - 0.3, fmt(r) + "%", ha="center", va="top", color=t["ikincil"], fontsize=8.5)
    ax.axvline(0, color=t["ikincil"], lw=1)
    ax.text(0.15, y[0] + 0.62, "iTransformer = 0", color=t["ikincil"], fontsize=9, va="center")
    ax.set_yticks(y, [f"H = {h}" for h in H])
    ax.set_xlim(-10.5, 2.4), ax.set_ylim(-0.75, len(H) - 0.1)
    ax.set_xlabel("iTransformer'a göre test MSE değişimi [%]   ← daha düşük hata")
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ms=8, color=t["vurgu"], ls="none", label="LastNorm"),
                       Line2D([], [], marker="D", ms=6, color=t["repeat"], ls="none", label="Repeat (naif)")],
              loc="upper left", frameon=False, labelcolor=t["metin"], fontsize=9.5)
    ax.set_title("Exchange: LastNorm her ufukta önde, Repeat daha da önde")
    kaydet(fig, "exchange-ufuk", tema)


# ---------------------------------------------------------------- 3. on seed
def on_seed(tema):
    t = TEMALAR[tema]
    s = V["on_seed"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    n = len(s["seedler"])
    y = list(range(n))[::-1]
    for yi, sd, pre, i, l in zip(y, s["seedler"], s["on_kayitli"], s["itransformer"], s["lastnorm"]):
        ax.annotate("", xy=(l, yi), xytext=(i, yi),
                    arrowprops=dict(arrowstyle="-|>", color=t["vurgu_acik"], lw=1.8, shrinkA=4, shrinkB=5, mutation_scale=11))
        ax.plot([i], [yi], "o", ms=7, color=t["gri"], zorder=3)
        ax.plot([l], [yi], "o", ms=8, color=t["vurgu"], zorder=4, mfc=t["vurgu"] if pre else "none", mew=1.8)
    rep = s["repeat"]["deger"]
    ax.axvline(rep, color=t["repeat"], lw=1.2, ls=(0, (4, 3)))
    ax.text(rep - 0.0004, n - 0.35, f"Repeat\n{rep:.4f}", ha="right", va="top", fontsize=9, color=t["metin"])
    ax.text(s["itransformer"][0], n - 0.35, "iTransformer", ha="center", va="top", fontsize=9, color=t["ikincil"])
    ax.text(s["lastnorm"][0], n - 0.35, "LastNorm", ha="center", va="top", fontsize=9, color=t["vurgu"], fontweight="bold")
    ax.set_yticks(y, [f"{sd}{'' if pre else ' *'}" for sd, pre in zip(s["seedler"], s["on_kayitli"])])
    ax.set_ylim(-0.6, n + 0.3)
    ax.set_xlim(0.3385, 0.3672)
    ax.set_xlabel("Exchange test MSE (4 ufuk ortalaması)   ← daha düşük hata")
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("On seed'in onunda LastNorm iTransformer'dan iyi, hiçbirinde Repeat'ten iyi değil")
    fig.text(0.01, -0.02, "* Test açıldıktan sonra eklenen seed (betimsel). Dolu turuncu: ön-kayıtlı seed; boş turuncu: ek seed.",
             fontsize=8.5, color=t["ikincil"])
    kaydet(fig, "on-seed", tema)


# ---------------------------------------------------------------- 4. ilk adim
def ilk_adim(tema):
    t = TEMALAR[tema]
    e = V["ilk_adim"]
    H = e["ufuklar"]
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    y = list(range(len(H)))[::-1]
    for yi, i, l in zip(y, e["itransformer"]["deger"], e["lastnorm"]["deger"]):
        ax.plot([l, i], [yi, yi], color=t["grid"], lw=6, solid_capstyle="round", zorder=1)
        ax.plot([i], [yi], "o", ms=8, color=t["gri"], zorder=3)
        ax.plot([l], [yi], "o", ms=9, color=t["vurgu"], zorder=4)
        ax.text(i + 0.07, yi, f"{i:.2f}×", va="center", color=t["ikincil"], fontsize=9.5)
        ax.text(l + 0.05, yi + 0.2, f"{l:.2f}×", va="bottom", ha="left", color=t["vurgu"], fontsize=9.5, fontweight="bold")
    ax.axvline(1, color=t["repeat"], lw=1.2, ls=(0, (4, 3)))
    ax.text(0.97, len(H) - 0.35, "Repeat düzeyi (1×)", ha="right", va="bottom", fontsize=9, color=t["metin"])
    ax.text(e["itransformer"]["deger"][0], len(H) - 0.35, "iTransformer", ha="center", va="bottom", fontsize=9, color=t["ikincil"])
    ax.set_yticks(y, [f"H = {h}" for h in H])
    ax.set_xlim(0.45, 3.8), ax.set_ylim(-0.6, len(H) + 0.05)
    ax.set_xlabel("ilk tahmin adımının hatası / Repeat'in ilk adım hatası")
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Kazancın kaynağı: ilk adımdaki seviye basamağı neredeyse kayboluyor")
    kaydet(fig, "ilk-adim", tema)


# ---------------------------------------------------------------- 5. ablasyon
def ablasyon(tema):
    t = TEMALAR[tema]
    a = V["ablasyon"]
    satir = a["satirlar"][::-1]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    UST = 0.405
    for yi, r in enumerate(satir):
        vurgu = r["ad"].startswith("LastNorm (")
        w = min(r["mse"], UST)
        ax.barh(yi, w - 0.30, left=0.30, height=0.6, color=t["vurgu"] if vurgu else t["gri"], zorder=2)
        etiket = f"{r['mse']:.4f}"
        if r["mse"] > UST:
            etiket = f"{r['mse']:.4f}  (eksen dışında →)"
        ax.text(w + 0.0008 if r["mse"] <= UST else UST - 0.001, yi, etiket, va="center",
                ha="left" if r["mse"] <= UST else "right",
                color=(t["vurgu"] if vurgu else (t["zemin"] if False else t["ikincil"])) if r["mse"] <= UST else t["metin"],
                fontweight="bold" if vurgu else "normal", fontsize=9.5, zorder=5)
    rep = a["repeat"]["mse"]
    ax.axvline(rep, color=t["repeat"], lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(rep - 0.0006, len(satir) - 0.35, f"Repeat {rep:.4f}", ha="right", va="bottom", fontsize=9, color=t["metin"])
    ax.set_yticks(range(len(satir)), [r["ad"] for r in satir])
    ax.set_xlim(0.30, UST), ax.set_ylim(-0.55, len(satir) - 0.05)
    ax.set_xlabel("Exchange test MSE (x ekseni 0.30'dan başlar)   ← daha düşük hata")
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Ablasyon: kazanç için son-değer merkezi ve σ birlikte gerekli")
    kaydet(fig, "ablasyon", tema)


# ---------------------------------------------------------------- 6. kalicilik orani
def kalicilik(tema):
    t = TEMALAR[tema]
    k = V["kalicilik"]["satirlar"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.axvline(1, color=t["ikincil"], lw=1, ls=(0, (2, 2)))
    ax.axhline(0, color=t["ikincil"], lw=1)
    ax.text(0.985, 4.55, "ön-kayıtlı eşik R = 1", ha="right", va="top", fontsize=9, color=t["ikincil"])
    for r in k:
        yanlis = r["veri"] == "ETTm2"
        dolu = r["rol"] == "kalibrasyon"
        c = t["vurgu"] if yanlis else t["gri"]
        ax.plot([r["R"]], [r["dpct"]], "o", ms=10, color=c, mfc=c if dolu else "none", mew=2, zorder=3)
        dx, ha = (0.025, "left")
        if r["veri"] == "ETTm1":
            dx, ha = (-0.025, "right")
        ax.text(r["R"] + dx, r["dpct"], f"{r['veri']}  {fmt(r['dpct'])}%", va="center", ha=ha, fontsize=9.5,
                color=t["vurgu"] if yanlis else t["metin"], fontweight="bold" if yanlis else "normal")
    ax.annotate("kural \"kazanç yok\" dedi,\nama LastNorm daha iyiydi", xy=(1.0967, -0.9), xytext=(1.22, -3.6),
                fontsize=9, color=t["vurgu"], arrowprops=dict(arrowstyle="->", color=t["vurgu"], lw=1.2))
    ax.set_xlim(0.7, 1.68), ax.set_ylim(-5.9, 4.9)
    ax.set_xlabel(r"kalıcılık oranı $\bar{R}$ (yalnız eğitim verisinden)")
    ax.set_ylabel("LastNorm MSE değişimi [%]")
    ax.grid(color=t["grid"], lw=0.8, zorder=0)
    ax.set_title("Sıralama tutuyor (Spearman ρ = 0.90), eşik kuralı ETTm2'de yanlış")
    fig.text(0.01, -0.03, "Dolu: kalibrasyon setleri (Exchange, ETTh1, ETTh2). Boş: testi kural yazıldıktan sonra açılan setler (ETTm1, ETTm2).",
             fontsize=8.5, color=t["ikincil"])
    kaydet(fig, "kalicilik", tema)


if __name__ == "__main__":
    for tema in TEMALAR:
        stil(TEMALAR[tema])
        hero(tema)
        exchange_ufuk(tema)
        on_seed(tema)
        ilk_adim(tema)
        ablasyon(tema)
        kalicilik(tema)
