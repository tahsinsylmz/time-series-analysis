"""Levenshtein (edit distance) algoritmasi ve en yakin pattern eslemesi.

Test asamasinda egitim sozlugunde bulunmayan (unseen) bir pattern ile
karsilasildiginda, sisteme devam edebilmek icin sozlukteki en yakin pattern
Levenshtein mesafesi ile belirlenir. Bu modul birim testlerle dogrulanir.
"""
from __future__ import annotations

from typing import Iterable


def levenshtein_mesafe(a: str, b: str) -> int:
    """Iki dizi arasindaki Levenshtein (ekle/sil/degistir) mesafesini hesaplar."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    onceki = list(range(n + 1))
    for i in range(1, m + 1):
        simdiki = [i] + [0] * n
        for j in range(1, n + 1):
            bedel = 0 if a[i - 1] == b[j - 1] else 1
            simdiki[j] = min(
                onceki[j] + 1,        # silme
                simdiki[j - 1] + 1,   # ekleme
                onceki[j - 1] + bedel,  # degistirme / eslesme
            )
        onceki = simdiki
    return onceki[n]


def en_yakin_pattern(hedef: str, sozluk: Iterable[str]) -> tuple[str | None, int]:
    """Sozlukteki en kucuk Levenshtein mesafeli pattern'i ve mesafesini dondurur.

    Esit mesafe durumunda alfabetik olarak ilk gelen secilir (deterministik).
    """
    en_iyi: str | None = None
    en_mesafe = -1
    for aday in sozluk:
        d = levenshtein_mesafe(hedef, aday)
        if en_iyi is None or d < en_mesafe or (d == en_mesafe and aday < en_iyi):
            en_iyi = aday
            en_mesafe = d
    return en_iyi, en_mesafe
