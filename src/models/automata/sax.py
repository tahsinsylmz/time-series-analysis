"""SAX - Symbolic Aggregate approXimation.

PAA ile indirgenmis degerleri, Gauss dagiliminin esit olasilikli bolgelerine
karsilik gelen kesim noktalarini kullanarak sembollere (harflere) cevirir.
``alphabet_size`` kadar farkli harf uretilir.

Not (z-normalizasyon tercihi): Klasik SAX her alt-diziyi (pencereyi) ayri ayri
z-normalize eder; boylece yalnizca SEKIL korunur, mutlak seviye silinir. Bu
projede z-normalizasyon bunun yerine egitim istatistikleriyle GLOBAL yapilir
(bkz. ``OtomataAnomaliModeli._normalize``). Boylece pencerelerin mutlak seviyesi
korunur; bu, anomalilerin cogunlukla seviye/genlik sapmasi oldugu bu veri
setlerinde kasitli bir tercihtir (ampirik olarak pencere-bazli z-norm ile F1
ayni, fakat global yaklasim daha az durum/daha sade bir otomata uretir).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def sax_kesim_noktalari(alphabet_size: int) -> np.ndarray:
    """Alfabe boyutu icin Gauss kuantil kesim noktalari (alphabet_size - 1 adet)."""
    if alphabet_size < 2:
        raise ValueError("alphabet_size en az 2 olmali")
    olasiliklar = np.arange(1, alphabet_size) / alphabet_size
    return norm.ppf(olasiliklar)


def sax_sembolize(degerler: np.ndarray, kesimler: np.ndarray) -> str:
    """PAA degerlerini harf dizisine (ornek: 'aab') cevirir.

    'a' en dusuk bolge, sonraki harfler artan degerlere karsilik gelir.
    """
    indeksler = np.searchsorted(kesimler, np.asarray(degerler, dtype=float))
    return "".join(chr(ord("a") + int(i)) for i in indeksler)
