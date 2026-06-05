"""Sunum icin tek komutluk demo.

Canli sunumda tek komutla projenin ozetini gosterir:
  1) Otomatanin bir kararini ACIKLAR (durum, yol, unseen, JSON, karar),
  2) sonuc tablosunu (F1, orijinal senaryo) yazar,
  3) uretilen figurleri listeler.

Derin ogrenme egitmez; yalnizca otomatayi (hizli, deterministik) egitir ve
hazir ``results/ozet.csv`` ciktisini okur. Birkac saniyede biter.

Kullanim:
    python -m scripts.sunum_demo
"""
from __future__ import annotations

import os

import pandas as pd

from scripts.demo_explain import _skab_ilk_fold
from src.explainability.explainer import OtomataAciklayici
from src.models.automata.automata_model import OtomataAnomaliModeli
from src.utils.config import PROJE_KOK, konfig_yukle
from src.utils.seeding import seed_ayarla


def _baslik(metin: str) -> None:
    cizgi = "=" * 66
    print(f"\n{cizgi}\n  {metin}\n{cizgi}")


def main() -> None:
    cfg = konfig_yukle()
    seed_ayarla(cfg.genel.rastgele_seedler[0])
    sonuc_dizini = os.path.join(PROJE_KOK, cfg.genel.cikti_dizini)

    _baslik("PROJE: Kara Kutudan Aciklanabilirlige - Olasiliksal Otomatalar")
    print("  Beyaz kutu otomata  vs  kara kutu derin ogrenme (LSTM/GRU/1D-CNN)")
    print("  Veri setleri: SKAB + BATADAL | 3 senaryo | 5 tohum | istatistik testleri")

    # 1) Canli aciklama (otomata, SKAB ilk fold)
    _baslik("1) OTOMATA KARAR ACIKLAMASI (SKAB) - canli uretiliyor")
    g_tr, g_val, g_test, g_unseen = _skab_ilk_fold(cfg)
    model = OtomataAnomaliModeli(cfg).egit(g_tr, g_val)
    ornekler = OtomataAciklayici(model).secili_ornekler(g_test, k_anomali=1, ek_veri=g_unseen)
    for ack in ornekler[:3]:
        print(f"\n  [{ack.get('kategori', '?')}] konum={ack['konum']}  "
              f"durum={ack['durum_sax']}  karar={ack['karar_metni']}  "
              f"(gercek etiket={ack['teshis']['gercek_etiket']})")
        print(f"    Yol      : {' -> '.join(ack['yol'])}")
        print(f"    Aciklama : {ack['aciklama_metni']}")
        print(f"    JSON(X.F): {ack['spec_formati']}")

    # 2) Sonuc tablosu
    _baslik("2) SONUC TABLOSU - F1 (orijinal senaryo)")
    ozet_yolu = os.path.join(sonuc_dizini, "ozet.csv")
    if os.path.exists(ozet_yolu):
        ozet = pd.read_csv(ozet_yolu)
        alt = (ozet[ozet.senaryo == "orijinal"][["veri_seti", "model", "f1_mean"]]
               .sort_values(["veri_seti", "model"]))
        alt["f1_mean"] = alt["f1_mean"].round(3)
        print(alt.to_string(index=False))
    else:
        print("  (ozet.csv yok; once 'python -m scripts.run_experiments' calistirin)")

    # 3) Figurler
    _baslik("3) URETILEN FIGURLER (results/figurler/)")
    fig_dizini = os.path.join(sonuc_dizini, "figurler")
    if os.path.isdir(fig_dizini):
        for ad in sorted(os.listdir(fig_dizini)):
            print(f"  - {ad}")
    else:
        print("  (figurler yok; 'python -m scripts.make_figures' calistirin)")

    _baslik("BITTI")
    print("  Ayrica gosterin: GitHub README, rapor/rapor.md tablolari, pytest -q (92 test).")


if __name__ == "__main__":
    main()
