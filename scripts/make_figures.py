"""Deney sonuclarindan gorsel uretir (rapor/sunum icin).

Girdi : results/ozet.csv, results/olcumler.csv, results/parametre_taramasi_skab.csv
Cikti : results/figurler/*.png

Uretilen figurler:
  1. fig_f1_karsilastirma.png    - model F1 karsilastirmasi (orijinal senaryo)
  2. fig_senaryo_dayaniklilik.png - senaryolar arasi F1 (orijinal/gurultu/unseen)
  3. fig_parametre_duyarlilik.png - otomata window x alphabet F1 isimaritasi
  4. fig_gecis_matrisi.png        - otomata durum-gecis olasilik matrisi
  5. fig_durum_diyagrami.png      - otomata durum gecis grafigi (en guclu gecisler)
  6. fig_karmasiklik_matrisi.png  - otomata vs derin ogrenme karmasiklik matrisi

3-6 numarali figurler SKAB ilk fold'unda modelleri yeniden egitir (deterministik).

Kullanim:
    python -m scripts.make_figures
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # ekransiz ortam
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.experiments.scenarios import temiz_girdi
from src.models.automata.automata_model import OtomataAnomaliModeli
from src.models.deep_learning.dl_model import DerinOgrenmeModeli
from src.preprocessing.data_loader import veri_yukle
from src.preprocessing.preprocess import OnIslemci
from src.preprocessing.splitting import grup_train_val_bol, skab_foldlar
from src.utils.config import PROJE_KOK, konfig_yukle
from src.utils.seeding import seed_ayarla

sns.set_theme(style="whitegrid")
RENKLER = {"automata": "#d1495b", "lstm": "#30638e", "gru": "#003d5b", "cnn1d": "#00798c"}
MODEL_ETIKET = {"automata": "Otomata", "lstm": "LSTM", "gru": "GRU", "cnn1d": "1D-CNN"}


def _kaydet(fig, cikti, ad):
    yol = os.path.join(cikti, ad)
    fig.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  yazildi: {ad}")


# ---- 1) model F1 karsilastirmasi ----
def fig_f1_karsilastirma(ozet, cikti):
    alt = ozet[ozet.senaryo == "orijinal"].copy()
    veri_setleri = ["SKAB", "BATADAL"]
    modeller = ["automata", "lstm", "gru", "cnn1d"]
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, vs in zip(eksenler, veri_setleri):
        d = alt[alt.veri_seti == vs].set_index("model")
        x = np.arange(len(modeller))
        f1 = [d.loc[m, "f1_mean"] if m in d.index else 0 for m in modeller]
        std = [(d.loc[m, "f1_std"] if not pd.isna(d.loc[m, "f1_std"]) else 0)
               if m in d.index else 0 for m in modeller]
        ax.bar(x, f1, yerr=std, capsize=4, color=[RENKLER[m] for m in modeller])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_ETIKET[m] for m in modeller])
        ax.set_title(f"{vs} - F1 (orijinal senaryo)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 skoru")
        for i, v in enumerate(f1):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    _kaydet(fig, cikti, "fig_f1_karsilastirma.png")


# ---- 2) senaryo dayanikliligi ----
def fig_senaryo_dayaniklilik(ozet, cikti):
    senaryolar = ["orijinal", "gurultu", "unseen"]
    modeller = ["automata", "lstm", "gru", "cnn1d"]
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, vs in zip(eksenler, ["SKAB", "BATADAL"]):
        d = ozet[ozet.veri_seti == vs]
        x = np.arange(len(senaryolar))
        genislik = 0.2
        for i, m in enumerate(modeller):
            dm = d[d.model == m].set_index("senaryo")
            f1 = [dm.loc[s, "f1_mean"] if s in dm.index else 0 for s in senaryolar]
            ax.bar(x + i * genislik, f1, genislik, label=MODEL_ETIKET[m], color=RENKLER[m])
        ax.set_xticks(x + 1.5 * genislik)
        ax.set_xticklabels(["Orijinal", "Gurultu", "Unseen"])
        ax.set_title(f"{vs} - senaryo dayanikliligi")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 skoru")
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _kaydet(fig, cikti, "fig_senaryo_dayaniklilik.png")


# ---- 3) parametre duyarliligi (3 eksen: F1 + durum sayisi + gecis yogunlugu) ----
def fig_parametre_duyarlilik(tarama, cikti):
    paneller = [
        ("f1_ortalama", "Ortalama F1", "viridis", ".3f"),
        ("state_sayisi", "Durum sayisi", "rocket_r", ".0f"),
        ("gecis_yogunlugu", "Gecis yogunlugu", "mako_r", ".3f"),
    ]
    mevcut = [p for p in paneller if p[0] in tarama.columns]
    fig, eksenler = plt.subplots(1, len(mevcut), figsize=(5.2 * len(mevcut), 4.6))
    if len(mevcut) == 1:
        eksenler = [eksenler]
    for ax, (sutun, baslik, cmap, bicim) in zip(eksenler, mevcut):
        pivot = tarama.pivot(index="window_size", columns="alphabet_size", values=sutun)
        sns.heatmap(pivot, annot=True, fmt=bicim, cmap=cmap, ax=ax,
                    cbar_kws={"label": baslik})
        ax.set_title(f"Otomata - {baslik} (SKAB)")
        ax.set_xlabel("Alfabe boyutu")
        ax.set_ylabel("Pencere boyutu")
    fig.tight_layout()
    _kaydet(fig, cikti, "fig_parametre_duyarlilik.png")


# ---- ROC ve PR egrileri (SKAB, en iyi DL modeli) ----
def fig_roc_pr(model, model_ad, g_test, cikti):
    """Verilen DL modelinin SKAB test setindeki ROC ve PR egrilerini cizer."""
    skorlar, konumlar = model.skor(g_test)
    y = g_test.y[konumlar]
    if len(np.unique(y)) < 2:
        print("  atlandi: fig_roc_pr (tek sinif)")
        return
    fpr, tpr, _ = roc_curve(y, skorlar)
    roc_auc = roc_auc_score(y, skorlar)
    kesinlik, duyarlilik, _ = precision_recall_curve(y, skorlar)
    pr_auc = average_precision_score(y, skorlar)
    taban = float(np.mean(y))   # PR icin rastgele taban (pozitif orani)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    etiket = MODEL_ETIKET.get(model_ad, model_ad)
    ax1.plot(fpr, tpr, color=RENKLER.get(model_ad, "#30638e"), lw=2,
             label=f"{etiket} (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Rastgele")
    ax1.set_xlabel("Yanlis pozitif orani"); ax1.set_ylabel("Dogru pozitif orani")
    ax1.set_title("ROC egrisi (SKAB)"); ax1.legend(loc="lower right", fontsize=9)
    ax2.plot(duyarlilik, kesinlik, color=RENKLER.get(model_ad, "#00798c"), lw=2,
             label=f"{etiket} (PR-AUC = {pr_auc:.3f})")
    ax2.axhline(taban, ls="--", color="gray", lw=1, label=f"Taban (poz={taban:.2f})")
    ax2.set_xlabel("Duyarlilik (recall)"); ax2.set_ylabel("Kesinlik (precision)")
    ax2.set_title("Precision-Recall egrisi (SKAB)"); ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(0, 1.02)
    fig.tight_layout()
    _kaydet(fig, cikti, "fig_roc_pr.png")


# ---- SKAB ilk fold girdileri (3-6 icin) ----
def _skab_fold0(cfg):
    sk = cfg.veri_setleri.skab
    ham = veri_yukle(cfg, "skab")
    tr_idx, test_idx = next(iter(skab_foldlar(ham, sk.fold_sayisi, sk.fold_tohumu)))
    ytr, val = grup_train_val_bol(tr_idx, ham.gruplar, sk.dogrulama_orani, sk.fold_tohumu)
    on = OnIslemci(cfg).fit(ham.X[ytr])
    g_tr = temiz_girdi(on, ham.X[ytr], ham.y[ytr], ham.gruplar[ytr])
    g_val = temiz_girdi(on, ham.X[val], ham.y[val], ham.gruplar[val])
    g_test = temiz_girdi(on, ham.X[test_idx], ham.y[test_idx], ham.gruplar[test_idx])
    return g_tr, g_val, g_test


# ---- 4) gecis olasilik matrisi ----
def fig_gecis_matrisi(model, cikti):
    M, durumlar = model.oto.gecis_matrisi()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(M, cmap="magma", ax=ax, cbar_kws={"label": "P(kaynak -> hedef)"})
    ax.set_title(f"Otomata gecis olasilik matrisi ({model.oto.K} durum)")
    ax.set_xlabel("Hedef durum")
    ax.set_ylabel("Kaynak durum")
    if model.oto.K <= 20:
        ax.set_xticklabels(durumlar, rotation=90, fontsize=7)
        ax.set_yticklabels(durumlar, rotation=0, fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    _kaydet(fig, cikti, "fig_gecis_matrisi.png")


# ---- 5) durum gecis diyagrami ----
def fig_durum_diyagrami(model, cikti, ust_kenar=25):
    oto = model.oto
    kenarlar = sorted(oto.gecis_sayim.items(), key=lambda kv: kv[1], reverse=True)[:ust_kenar]
    G = nx.DiGraph()
    for (k, h), sayi in kenarlar:
        G.add_edge(k, h, weight=sayi, olasilik=oto.gecis_olasiligi(k, h))
    if G.number_of_nodes() == 0:
        return
    boyut = [300 + 40 * oto.kaynak_toplam.get(n, 0) for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(8, 7))
    yer = nx.spring_layout(G, seed=42, k=0.9)
    agirliklar = [G[u][v]["olasilik"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, yer, node_size=boyut, node_color="#00798c", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, yer, font_size=8, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, yer, width=[1 + 4 * w for w in agirliklar],
                           edge_color="#555", alpha=0.6, arrowsize=12,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    ax.set_title(f"Otomata durum gecis diyagrami (en guclu {len(kenarlar)} gecis)")
    ax.axis("off")
    fig.tight_layout()
    _kaydet(fig, cikti, "fig_durum_diyagrami.png")


# ---- 6) karmasiklik matrisi (otomata vs derin ogrenme) ----
def _tahmin(model, girdi):
    skorlar, konumlar = model.skor(girdi)
    tahmin = (skorlar >= model.esik).astype(int)
    return girdi.y[konumlar], tahmin


def fig_karmasiklik_matrisi(auto, dl, dl_ad, g_test, cikti):
    fig, eksenler = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (model, ad) in zip(eksenler, [(auto, "Otomata"), (dl, MODEL_ETIKET[dl_ad])]):
        y, tahmin = _tahmin(model, g_test)
        cm = confusion_matrix(y, tahmin, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Normal", "Anomali"], yticklabels=["Normal", "Anomali"])
        ax.set_title(f"{ad} (SKAB fold-1)")
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gercek")
    fig.tight_layout()
    _kaydet(fig, cikti, "fig_karmasiklik_matrisi.png")


def main() -> None:
    cfg = konfig_yukle()
    sonuc = os.path.join(PROJE_KOK, cfg.genel.cikti_dizini)
    cikti = os.path.join(sonuc, "figurler")
    os.makedirs(cikti, exist_ok=True)

    ozet = pd.read_csv(os.path.join(sonuc, "ozet.csv"))
    tarama = pd.read_csv(os.path.join(sonuc, "parametre_taramasi_skab.csv"))

    print("CSV tabanli figurler...")
    fig_f1_karsilastirma(ozet, cikti)
    fig_senaryo_dayaniklilik(ozet, cikti)
    fig_parametre_duyarlilik(tarama, cikti)

    print("Model tabanli figurler (SKAB fold-1 yeniden egitim)...")
    seed_ayarla(cfg.genel.rastgele_seedler[0])
    g_tr, g_val, g_test = _skab_fold0(cfg)
    auto = OtomataAnomaliModeli(cfg).egit(g_tr, g_val)
    fig_gecis_matrisi(auto, cikti)
    fig_durum_diyagrami(auto, cikti)
    seed_ayarla(cfg.genel.rastgele_seedler[0])
    gru = DerinOgrenmeModeli(cfg, "gru").egit(g_tr, g_val)
    fig_karmasiklik_matrisi(auto, gru, "gru", g_test, cikti)
    fig_roc_pr(gru, "gru", g_test, cikti)

    print(f"\nTum figurler '{cikti}' dizinine yazildi.")


if __name__ == "__main__":
    main()
