"""PAA - Piecewise Aggregate Approximation.

Bir zaman serisi penceresini, esit uzunluktaki segmentlerin ortalamasi ile
daha dusuk boyutlu bir temsile indirger. SAX sembollestirmesinden onceki
ilk adimdir.
"""
from __future__ import annotations

import numpy as np


def paa_segment(pencere: np.ndarray, segment_sayisi: int) -> np.ndarray:
    """Pencereyi ``segment_sayisi`` segmentin ortalamasina indirger.

    Pencere uzunlugunun segment sayisina tam bolunmesi beklenir (otomata modeli
    pencereleri ``segment_sayisi * paa_bolme_faktoru`` uzunlugunda olusturur).
    """
    pencere = np.asarray(pencere, dtype=float)
    n = pencere.size
    if segment_sayisi <= 0:
        raise ValueError("segment_sayisi pozitif olmali")
    if n == segment_sayisi:
        return pencere.copy()
    if n % segment_sayisi != 0:
        raise ValueError(f"Pencere uzunlugu ({n}) segment sayisina ({segment_sayisi}) bolunemiyor")
    k = n // segment_sayisi
    return pencere.reshape(segment_sayisi, k).mean(axis=1)
