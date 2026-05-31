"""Otomata aciklanabilirlik demosu (black-box -> white-box).

SKAB'in ilk fold'u uzerinde otomata egitir, ardindan test setindeki en
yuksek skorlu anomali noktalarinin kararlarini ayrintili olarak aciklar.
Her aciklama hem makine-okur (JSON) hem insan-okur (Turkce metin) icerir.

Cikti:
    results/aciklamalar/ornek_NN.json   (her bir aciklama)
    results/aciklamalar/ozet.json       (tum aciklamalar tek dosyada)
    ayrica ekrana ilk ornegin Turkce anlatimi yazilir.

Kullanim:
    python -m scripts.demo_explain [--k 5] [--veri skab|batadal]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.experiments.scenarios import temiz_girdi
from src.explainability.explainer import OtomataAciklayici
from src.models.automata.automata_model import OtomataAnomaliModeli
from src.preprocessing.data_loader import veri_yukle
from src.preprocessing.preprocess import OnIslemci
from src.preprocessing.splitting import (
    grup_train_val_bol,
    skab_foldlar,
    zaman_sirali_bol,
)
from src.utils.config import PROJE_KOK, konfig_yukle
from src.utils.seeding import seed_ayarla


def _skab_ilk_fold(cfg):
    """SKAB ilk fold'u icin (egitim, dogrulama, test) girdilerini uretir."""
    sk = cfg.veri_setleri.skab
    ham = veri_yukle(cfg, "skab")
    tr_idx, test_idx = next(iter(skab_foldlar(ham, sk.fold_sayisi, sk.fold_tohumu)))
    ytr, val = grup_train_val_bol(tr_idx, ham.gruplar, sk.dogrulama_orani, sk.fold_tohumu)
    on = OnIslemci(cfg).fit(ham.X[ytr])
    g_tr = temiz_girdi(on, ham.X[ytr], ham.y[ytr], ham.gruplar[ytr])
    g_val = temiz_girdi(on, ham.X[val], ham.y[val], ham.gruplar[val])
    g_test = temiz_girdi(on, ham.X[test_idx], ham.y[test_idx], ham.gruplar[test_idx])
    return g_tr, g_val, g_test


def _batadal_bolme(cfg):
    """BATADAL zaman sirali bolme icin (egitim, dogrulama, test) girdilerini uretir."""
    bc = cfg.veri_setleri.batadal
    ham = veri_yukle(cfg, "batadal")
    n = ham.X.shape[0]
    tr, val, test = zaman_sirali_bol(n, bc.egitim_orani, bc.dogrulama_orani)
    seg = np.zeros(n, dtype=int)
    on = OnIslemci(cfg).fit(ham.X[tr])
    g = lambda idx: temiz_girdi(on, ham.X[idx], ham.y[idx], seg[idx])
    return g(tr), g(val), g(test)


def main() -> None:
    ap = argparse.ArgumentParser(description="Otomata aciklanabilirlik demosu")
    ap.add_argument("--k", type=int, default=5, help="Aciklanacak en anomalik nokta sayisi")
    ap.add_argument("--veri", choices=["skab", "batadal"], default="skab")
    args = ap.parse_args()

    cfg = konfig_yukle()
    seed_ayarla(cfg.genel.rastgele_seedler[0])

    if args.veri == "skab":
        g_tr, g_val, g_test = _skab_ilk_fold(cfg)
    else:
        g_tr, g_val, g_test = _batadal_bolme(cfg)

    model = OtomataAnomaliModeli(cfg).egit(g_tr, g_val)
    aciklayici = OtomataAciklayici(model)
    aciklamalar = aciklayici.en_anomalileri_acikla(g_test, k=args.k)

    cikti = os.path.join(PROJE_KOK, cfg.genel.cikti_dizini, "aciklamalar")
    os.makedirs(cikti, exist_ok=True)
    for i, ack in enumerate(aciklamalar, start=1):
        with open(os.path.join(cikti, f"ornek_{i:02d}.json"), "w", encoding="utf-8") as f:
            json.dump(ack, f, ensure_ascii=False, indent=2)
    with open(os.path.join(cikti, "ozet.json"), "w", encoding="utf-8") as f:
        json.dump(aciklamalar, f, ensure_ascii=False, indent=2)

    print(f"{len(aciklamalar)} aciklama '{cikti}' dizinine yazildi "
          f"(otomata durum sayisi={model.oto.K}, esik={model.esik:.3f}).\n")
    if aciklamalar:
        ilk = aciklamalar[0]
        print("--- En anomalik nokta (ornek) ---")
        print(f"Konum       : {ilk['konum']}")
        print(f"Durum (SAX) : {ilk['durum_sax']}")
        print(f"Yol         : {' -> '.join(ilk['yol'])}")
        print(f"Skor / Esik : {ilk['anomali_skoru']:.3f} / {ilk['esik']:.3f}")
        print(f"Karar       : {ilk['karar_metni']} (gercek etiket={ilk['gercek_etiket']})")
        print(f"Aciklama    : {ilk['aciklama_metni']}")


if __name__ == "__main__":
    main()
