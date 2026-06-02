"""Deney sonuclarindan rapor tablolari uretir (Markdown + LaTeX).

Girdi : results/olcumler.csv, results/istatistik_testleri.json
Cikti : rapor/tablolar/*.md, *.tex

Uretilen tablolar (ister IX.A):
  1. tablo_skab_fold.md       - SKAB kat-bazli F1 (fold x model, orijinal senaryo)
  2. tablo_skab_seed.md       - SKAB seed/kat varyansi AYRI (DL modelleri)
  3. tablo_dayaniklilik.md    - senaryo bazli (orijinal/gurultu/unseen) F1 + AUC
  4. tablo_batadal.md         - BATADAL zaman-sirali test (seed ort+std)
  5. tablo_istatistik.md      - Wilcoxon ve McNemar testleri
  6. tablo_unseen.md          - VI.A sozluk-disi Detection Rate / Mapping Accuracy (otomata)
  7. tablo_runtime.md         - model bazli egitim/cikarim suresi (EK Tablo5)

ozet.csv kat ve seed varyansini tek std'de karistirir; bu script ikisini AYIRIR
(SKAB icin kat-bazli varyasyon ile seed-bazli varyasyon ayri raporlanir).

Kullanim:
    python -m scripts.make_tables
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.utils.config import PROJE_KOK, konfig_yukle

MODEL_ETIKET = {"automata": "Otomata", "lstm": "LSTM", "gru": "GRU", "cnn1d": "1D-CNN"}
MODEL_SIRA = ["automata", "lstm", "gru", "cnn1d"]
SENARYO_ETIKET = {"orijinal": "Orijinal", "gurultu": "Gurultu", "unseen": "Unseen"}


def _yaz(cikti: str, ad: str, icerik: str) -> None:
    yol = os.path.join(cikti, ad)
    with open(yol, "w", encoding="utf-8") as f:
        f.write(icerik)
    print(f"  yazildi: {ad}")


def _md_tablo(basliklar: list[str], satirlar: list[list[str]]) -> str:
    ust = "| " + " | ".join(basliklar) + " |"
    ayrac = "| " + " | ".join("---" for _ in basliklar) + " |"
    govde = "\n".join("| " + " | ".join(s) + " |" for s in satirlar)
    return f"{ust}\n{ayrac}\n{govde}\n"


# ---- 1) SKAB kat-bazli F1 (orijinal senaryo) ----
def tablo_skab_fold(df: pd.DataFrame) -> str:
    alt = df[(df.veri_seti == "SKAB") & (df.senaryo == "orijinal")]
    if alt.empty:
        return "SKAB orijinal senaryo verisi yok.\n"
    modeller = [m for m in MODEL_SIRA if m in alt.model.unique()]
    # DL: kat icinde seed ortalamasi; otomata: tek deger (seed=-1)
    kat_model = alt.groupby(["fold", "model"]).f1.mean().unstack("model")
    foldlar = sorted(kat_model.index)
    basliklar = ["Kat"] + [MODEL_ETIKET[m] for m in modeller]
    satirlar = []
    for f in foldlar:
        satir = [f"{int(f)+1}"]
        for m in modeller:
            v = kat_model.loc[f, m] if m in kat_model.columns else np.nan
            satir.append(f"{v:.3f}" if pd.notna(v) else "-")
        satirlar.append(satir)
    # Kat-bazli ortalama +- std (kat varyasyonu)
    ort = ["**Ortalama**"]
    std = ["**Std (kat)**"]
    for m in modeller:
        kol = kat_model[m].dropna() if m in kat_model.columns else pd.Series(dtype=float)
        ort.append(f"**{kol.mean():.3f}**" if not kol.empty else "-")
        std.append(f"**{kol.std(ddof=0):.3f}**" if not kol.empty else "-")
    satirlar.extend([ort, std])
    aciklama = ("\n*SKAB, orijinal senaryo, F1. Derin ogrenme degerleri kat icinde 5 seed "
                "ortalamasidir; otomata deterministiktir. Son iki satir KAT-bazli "
                "ortalama ve standart sapmadir (seed varyansi ayri tabloda).*\n")
    return "## SKAB - Kat Bazli F1 (orijinal)\n\n" + _md_tablo(basliklar, satirlar) + aciklama


# ---- 2) SKAB seed vs kat varyansi (DL) ----
def tablo_skab_seed(df: pd.DataFrame) -> str:
    alt = df[(df.veri_seti == "SKAB") & (df.senaryo == "orijinal") & (df.model != "automata")]
    if alt.empty:
        return ""
    modeller = [m for m in MODEL_SIRA if m in alt.model.unique()]
    basliklar = ["Model", "F1 ort", "Kat std", "Seed std (kat-ici ort)"]
    satirlar = []
    for m in modeller:
        dm = alt[alt.model == m]
        kat_ort = dm.groupby("fold").f1.mean()             # kat ortalamalari
        seed_std_kat_ici = dm.groupby("fold").f1.std(ddof=0)  # her katta seed std
        satirlar.append([
            MODEL_ETIKET[m],
            f"{dm.f1.mean():.3f}",
            f"{kat_ort.std(ddof=0):.3f}",                  # katlar arasi (seed-ortalamali) std
            f"{seed_std_kat_ici.mean():.3f}",              # kat icinde seed kaynakli std
        ])
    aciklama = ("\n*Iki varyans kaynagi ayrildi: 'Kat std' katlar arasi (veri bolmesi) "
                "degiskenligi; 'Seed std' ayni kat icinde rastgele tohum kaynakli degiskenlik.*\n")
    return "## SKAB - Varyans Ayrimi (derin ogrenme)\n\n" + _md_tablo(basliklar, satirlar) + aciklama


# ---- 3) senaryo dayanikliligi (F1 + AUC) ----
def tablo_dayaniklilik(df: pd.DataFrame, veri_seti: str) -> str:
    alt = df[df.veri_seti == veri_seti]
    if alt.empty:
        return ""
    modeller = [m for m in MODEL_SIRA if m in alt.model.unique()]
    senaryolar = ["orijinal", "gurultu", "unseen"]
    var_auc = "roc_auc" in alt.columns
    basliklar = ["Model"] + [SENARYO_ETIKET[s] for s in senaryolar] + (["ROC-AUC (orij.)"] if var_auc else [])
    satirlar = []
    for m in modeller:
        satir = [MODEL_ETIKET[m]]
        for s in senaryolar:
            d = alt[(alt.model == m) & (alt.senaryo == s)]
            satir.append(f"{d.f1.mean():.3f}" if not d.empty else "-")
        if var_auc:
            d = alt[(alt.model == m) & (alt.senaryo == "orijinal")]
            au = d.roc_auc.mean() if not d.empty else np.nan
            satir.append(f"{au:.3f}" if pd.notna(au) else "-")
        satirlar.append(satir)
    return (f"## {veri_seti} - Senaryo Dayanikliligi (F1)\n\n"
            + _md_tablo(basliklar, satirlar)
            + "\n*F1, mevcut tum kat/seed uzerinden ortalamadir.*\n")


# ---- 4) BATADAL zaman-sirali test (seed ort+std) ----
def tablo_batadal(df: pd.DataFrame) -> str:
    alt = df[(df.veri_seti == "BATADAL") & (df.senaryo == "orijinal")]
    if alt.empty:
        return ""
    modeller = [m for m in MODEL_SIRA if m in alt.model.unique()]
    olcut = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]
    if "roc_auc" in alt.columns:
        olcut.append(("roc_auc", "ROC-AUC"))
    basliklar = ["Model"] + [b for _, b in olcut]
    satirlar = []
    for m in modeller:
        dm = alt[alt.model == m]
        satir = [MODEL_ETIKET[m]]
        for sutun, _ in olcut:
            if sutun not in dm.columns or dm[sutun].dropna().empty:
                satir.append("-")
            elif m == "automata":
                satir.append(f"{dm[sutun].mean():.3f}")            # tek kosu
            else:
                satir.append(f"{dm[sutun].mean():.3f} ± {dm[sutun].std(ddof=0):.3f}")
        satirlar.append(satir)
    return ("## BATADAL - Zaman Sirali Test (seed ort ± std)\n\n"
            + _md_tablo(basliklar, satirlar)
            + "\n*Tek zaman-sirali bolme; derin ogrenme degerleri 5 seed uzerinden "
            "ortalama ± standart sapmadir (otomata deterministik).*\n")


# ---- 5) istatistik testleri ----
def tablo_istatistik(ist: dict) -> str:
    parcalar = []
    wil = ist.get("wilcoxon", [])
    if wil:
        satirlar = [[f"{s['model_a']} vs {s['model_b']}", s.get("veri_seti", "-"),
                     f"{s['istatistik']:.3f}" if s['istatistik'] == s['istatistik'] else "-",
                     f"{s['p_deger']:.4f}" if s['p_deger'] == s['p_deger'] else "-"]
                    for s in wil]
        parcalar.append("## Wilcoxon Isaretli Sira Testi (SKAB, kat-bazli F1)\n\n"
                        + _md_tablo(["Karsilastirma", "Veri", "Istatistik", "p"], satirlar))
    mc = ist.get("mcnemar", [])
    if mc:
        satirlar = [[s.get("veri_seti", "-"), f"{s['model_a']} vs {s['model_b']}",
                     f"{s['istatistik']:.3f}" if s['istatistik'] == s['istatistik'] else "-",
                     f"{s['p_deger']:.2e}" if s['p_deger'] == s['p_deger'] else "-",
                     str(s.get("yalniz_a_dogru", "-")), str(s.get("yalniz_b_dogru", "-"))]
                    for s in mc]
        parcalar.append("## McNemar Testi\n\n"
                        + _md_tablo(["Veri", "Karsilastirma", "Istatistik", "p",
                                     "Yalniz A dogru", "Yalniz B dogru"], satirlar))
    return "\n".join(parcalar)


# ---- 6) unseen (VI.A) Detection Rate / Mapping Accuracy ----
def tablo_unseen(df) -> str:
    """Otomata sozluk-disi (VI.A) yonetim metrikleri: Detection Rate / Mapping Accuracy."""
    if df is None or df.empty:
        return "Unseen (sozluk-disi) analiz verisi yok.\n"
    basliklar = ["Veri", "Kat", "Toplam konum", "Sozluk-disi",
                 "Detection Rate", "Map.Acc (tam)", "Map.Acc (mesafe<=1)"]
    satirlar = []
    for _, r in df.iterrows():
        satirlar.append([
            str(r["veri_seti"]), f"{int(r['fold'])+1}",
            f"{int(r['toplam_konum'])}", f"{int(r['sozluk_disi_konum'])}",
            f"{float(r['detection_rate']):.3f}",
            f"{float(r['mapping_accuracy_tam']):.3f}",
            f"{float(r['mapping_accuracy_yumusak']):.3f}",
        ])
    # Veri seti bazli ortalama satiri
    for veri in df["veri_seti"].unique():
        alt = df[df.veri_seti == veri]
        satirlar.append([
            f"**{veri} ort.**", "-",
            f"{alt['toplam_konum'].mean():.0f}", f"{alt['sozluk_disi_konum'].mean():.0f}",
            f"**{alt['detection_rate'].mean():.3f}**",
            f"**{alt['mapping_accuracy_tam'].mean():.3f}**",
            f"**{alt['mapping_accuracy_yumusak'].mean():.3f}**",
        ])
    aciklama = ("\n*VI.A sozluk-disi (out-of-vocabulary) yonetimi (yalniz otomata). "
                "Detection Rate = sozluk-disi pattern tasiyan test konumu orani; "
                "Mapping Accuracy = Levenshtein ile eslenen en yakin bilinen pattern'in, "
                "genlik kaydirmasindan ONCEKI gercek pattern'e dogrulugu (tam esitlik / "
                "mesafe<=1). Derin ogrenme modellerinde kavramsal karsiligi yoktur.*\n")
    return "## Sozluk-disi Yonetimi - Detection Rate / Mapping Accuracy (VI.A)\n\n" + \
           _md_tablo(basliklar, satirlar) + aciklama


# ---- 7) calisma sureleri (EK Tablo5) ----
def tablo_runtime(df) -> str:
    """Model+veri bazli ortalama egitim/cikarim suresi (EK Tablo5)."""
    if df is None or df.empty:
        return "Calisma suresi verisi yok.\n"
    basliklar = ["Veri", "Model", "Egitim suresi (sn)", "Cikarim suresi (sn)"]
    satirlar = []
    for _, r in df.iterrows():
        satirlar.append([
            str(r["veri_seti"]),
            MODEL_ETIKET.get(str(r["model"]), str(r["model"])),
            f"{float(r['egitim_suresi_sn']):.3f}",
            f"{float(r['cikarim_suresi_sn']):.4f}",
        ])
    aciklama = ("\n*Tek bir tam kosuda olculen sureler: egitim suresi modelin egitimi "
                "(tum senaryolarda ayni), cikarim suresi orijinal test seti uzerinde. "
                "Otomata deterministik; derin ogrenme degerleri seed ortalamasidir.*\n")
    return ("## Calisma Sureleri - Egitim / Cikarim (EK Tablo5)\n\n"
            + _md_tablo(basliklar, satirlar) + aciklama)


# ---- LaTeX (ana SKAB kat tablosu) ----
def latex_skab_fold(df: pd.DataFrame) -> str:
    alt = df[(df.veri_seti == "SKAB") & (df.senaryo == "orijinal")]
    if alt.empty:
        return ""
    modeller = [m for m in MODEL_SIRA if m in alt.model.unique()]
    kat_model = alt.groupby(["fold", "model"]).f1.mean().unstack("model")
    satirlar = []
    for f in sorted(kat_model.index):
        hucre = " & ".join(f"{kat_model.loc[f, m]:.3f}" if m in kat_model.columns
                           and pd.notna(kat_model.loc[f, m]) else "-" for m in modeller)
        satirlar.append(f"{int(f)+1} & {hucre} \\\\")
    govde = "\n".join(satirlar)
    kolonlar = "l" + "c" * len(modeller)
    baslik = " & ".join(MODEL_ETIKET[m] for m in modeller)
    return (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{SKAB kat bazli F1 (orijinal senaryo)}\n"
        f"\\begin{{tabular}}{{{kolonlar}}}\n\\hline\n"
        f"Kat & {baslik} \\\\\n\\hline\n{govde}\n\\hline\n"
        "\\end{tabular}\n\\end{table}\n"
    )


def main() -> None:
    cfg = konfig_yukle()
    sonuc = os.path.join(PROJE_KOK, cfg.genel.cikti_dizini)
    cikti = os.path.join(PROJE_KOK, "rapor", "tablolar")
    os.makedirs(cikti, exist_ok=True)

    df = pd.read_csv(os.path.join(sonuc, "olcumler.csv"))
    ist_yol = os.path.join(sonuc, "istatistik_testleri.json")
    ist = json.load(open(ist_yol, encoding="utf-8")) if os.path.exists(ist_yol) else {}
    unseen_yol = os.path.join(sonuc, "unseen_analizi.csv")
    unseen_df = pd.read_csv(unseen_yol) if os.path.exists(unseen_yol) else None
    sure_yol = os.path.join(sonuc, "calisma_sureleri.csv")
    sure_df = pd.read_csv(sure_yol) if os.path.exists(sure_yol) else None

    _yaz(cikti, "tablo_skab_fold.md", tablo_skab_fold(df))
    _yaz(cikti, "tablo_skab_seed.md", tablo_skab_seed(df))
    _yaz(cikti, "tablo_dayaniklilik.md",
         tablo_dayaniklilik(df, "SKAB") + "\n" + tablo_dayaniklilik(df, "BATADAL"))
    _yaz(cikti, "tablo_batadal.md", tablo_batadal(df))
    _yaz(cikti, "tablo_istatistik.md", tablo_istatistik(ist))
    _yaz(cikti, "tablo_unseen.md", tablo_unseen(unseen_df))
    _yaz(cikti, "tablo_runtime.md", tablo_runtime(sure_df))
    _yaz(cikti, "tablo_skab_fold.tex", latex_skab_fold(df))

    print(f"\nTum tablolar '{cikti}' dizinine yazildi.")


if __name__ == "__main__":
    main()
