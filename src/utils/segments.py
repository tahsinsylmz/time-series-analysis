"""Bitisik (contiguous) segment yardimcilari.

SKAB verisi dosya bazli birlestirildiginde, her .csv ayri bir zaman serisidir.
Sliding window ve otomata gecisleri dosya sinirlarini asmamalidir. Bu modul,
ardisik ayni etikete sahip satir bloklarini bulur.
"""
from __future__ import annotations

from typing import Sequence


def bitisik_bloklar(etiketler: Sequence) -> list[tuple[int, int]]:
    """Ardisik ayni degerli bloklarin (bas, son) indekslerini dondurur.

    Ornek: ['a','a','b','b','b'] -> [(0, 2), (2, 5)]
    """
    bloklar: list[tuple[int, int]] = []
    n = len(etiketler)
    if n == 0:
        return bloklar
    bas = 0
    for i in range(1, n):
        if etiketler[i] != etiketler[i - 1]:
            bloklar.append((bas, i))
            bas = i
    bloklar.append((bas, n))
    return bloklar
