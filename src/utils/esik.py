"""Karar esigi secimi.

Anomali skorlari surekli degerlerdir; ikili karar icin bir esik gerekir.
Esik, dogrulama (veya egitim) seti uzerinde F1-skorunu maksimize edecek
sekilde secilir. Boylece dengesiz veri setlerinde de makul kararlar uretilir.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import f1_score


def f1_maksimize_esik(
    skorlar: np.ndarray, y: np.ndarray, aday_sayisi: int = 200,
    tek_sinif_persentil: float = 0.95,
) -> float:
    """Verilen skor/etiket icin F1'i maksimize eden esigi dondurur.

    Dogrulama setinde tek sinif (yalniz normal) varsa F1 tanimsizdir; bu durumda
    medyan (noktalarin ~%50'sini anomali isaretler) cok agresiftir. Bunun yerine
    yuksek bir skor persentili kullanilir: anomali kaniti yokken yalnizca en uc
    noktalar isaretlenir (daha guvenli, dusuk yanlis-alarm).
    """
    skorlar = np.asarray(skorlar, dtype=float)
    y = np.asarray(y, dtype=int)
    if skorlar.size == 0:
        return 0.0
    if len(np.unique(y)) < 2:
        warnings.warn(
            f"Esik secimi: dogrulama tek sinifli (n={y.size}, pozitif={int(y.sum())}); "
            f"%{tek_sinif_persentil * 100:.0f} skor persentili esik olarak kullanildi.",
            RuntimeWarning, stacklevel=2,
        )
        return float(np.quantile(skorlar, tek_sinif_persentil))
    adaylar = np.unique(skorlar)
    if len(adaylar) > aday_sayisi:
        adaylar = np.quantile(skorlar, np.linspace(0.0, 1.0, aday_sayisi))
    en_iyi_esik = float(adaylar[0])
    en_iyi_f1 = -1.0
    for esik in adaylar:
        tahmin = (skorlar >= esik).astype(int)
        f = f1_score(y, tahmin, zero_division=0)
        if f > en_iyi_f1:
            en_iyi_f1 = f
            en_iyi_esik = float(esik)
    return en_iyi_esik
