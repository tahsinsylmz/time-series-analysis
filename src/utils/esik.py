"""Karar esigi secimi.

Anomali skorlari surekli degerlerdir; ikili karar icin bir esik gerekir.
Esik, dogrulama (veya egitim) seti uzerinde F1-skorunu maksimize edecek
sekilde secilir. Boylece dengesiz veri setlerinde de makul kararlar uretilir.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def f1_maksimize_esik(skorlar: np.ndarray, y: np.ndarray, aday_sayisi: int = 200) -> float:
    """Verilen skor/etiket icin F1'i maksimize eden esigi dondurur."""
    skorlar = np.asarray(skorlar, dtype=float)
    y = np.asarray(y, dtype=int)
    if skorlar.size == 0:
        return 0.0
    if len(np.unique(y)) < 2:
        # Tek sinif: orta deger esik olarak yeterli
        return float(np.median(skorlar))
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
