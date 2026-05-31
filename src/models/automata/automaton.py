"""Olasiliksal otomata (probabilistic automaton).

Her benzersiz SAX pattern'i bir durum (state) olarak tanimlanir. Durumlar arasi
gecisler frekans tabanli olarak ogrenilir ve Laplace yumusatma ile olasiliga
cevrilir:

    P(Si -> Sj) = (gecis_sayisi + alpha) / (toplam_cikis + alpha * |durumlar|)

Bir pattern dizisinin olasiligi (path probability), ardisik gecis
olasiliklarinin carpimidir. Sozlukte olmayan (unseen) pattern'lar Levenshtein
ile en yakin bilinen pattern'a eslenir.
"""
from __future__ import annotations

import numpy as np

from src.models.automata.levenshtein import en_yakin_pattern


class OlasiliksalOtomata:
    """Frekans tabanli ogrenen olasiliksal durum-gecis otomatasi."""

    def __init__(self, window_size: int, alphabet_size: int, laplace_alpha: float = 1.0) -> None:
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.alpha = laplace_alpha
        self.sozluk: set[str] = set()                     # gorulen tum pattern'lar
        self.gecis_sayim: dict[tuple[str, str], int] = {}  # (Si, Sj) -> sayi
        self.kaynak_toplam: dict[str, int] = {}            # Si -> toplam cikis sayisi
        self.durumlar: list[str] = []
        self.K = 0
        self._coz_cache: dict[str, tuple[str, bool, str, int]] = {}

    # ---- egitim ----
    def sozluge_ekle(self, pattern: str) -> None:
        self.sozluk.add(pattern)

    def gecis_ekle(self, kaynak: str, hedef: str) -> None:
        anahtar = (kaynak, hedef)
        self.gecis_sayim[anahtar] = self.gecis_sayim.get(anahtar, 0) + 1
        self.kaynak_toplam[kaynak] = self.kaynak_toplam.get(kaynak, 0) + 1

    def sonlandir(self) -> None:
        """Egitim bittikten sonra durum listesini ve sabitleri hesaplar."""
        self.durumlar = sorted(self.sozluk)
        self.K = len(self.durumlar)
        self._coz_cache.clear()

    # ---- sorgulama ----
    def gecis_olasiligi(self, kaynak: str, hedef: str) -> float:
        """Laplace yumusatmali P(kaynak -> hedef)."""
        sayi = self.gecis_sayim.get((kaynak, hedef), 0)
        toplam = self.kaynak_toplam.get(kaynak, 0)
        payda = toplam + self.alpha * max(self.K, 1)
        return (sayi + self.alpha) / payda

    def pattern_coz(self, pattern: str) -> tuple[str, bool, str, int]:
        """Pattern'i cozer.

        Donen: (etkin_pattern, unseen_mi, en_yakin_pattern, mesafe)
        - Sozlukteyse: (pattern, False, pattern, 0)
        - Unseen ise : (en_yakin, True, en_yakin, mesafe>0)
        """
        if pattern in self._coz_cache:
            return self._coz_cache[pattern]
        if pattern in self.sozluk:
            sonuc = (pattern, False, pattern, 0)
        else:
            en_yakin, mesafe = en_yakin_pattern(pattern, self.durumlar)
            sonuc = (en_yakin if en_yakin is not None else pattern, True,
                     en_yakin if en_yakin is not None else pattern, int(mesafe))
        self._coz_cache[pattern] = sonuc
        return sonuc

    def path_olasiligi(self, kelimeler: list[str]) -> float:
        """Bir pattern dizisinin yol olasiligi (ardisik gecislerin carpimi)."""
        if len(kelimeler) < 2:
            return 1.0
        olasilik = 1.0
        for t in range(1, len(kelimeler)):
            kaynak = self.pattern_coz(kelimeler[t - 1])[0]
            hedef = self.pattern_coz(kelimeler[t])[0]
            olasilik *= self.gecis_olasiligi(kaynak, hedef)
        return olasilik

    # ---- analiz / gorsellestirme ----
    def gecis_yogunlugu(self) -> float:
        """Gozlemlenen farkli gecis sayisinin olasi gecislere orani."""
        if self.K == 0:
            return 0.0
        return len(self.gecis_sayim) / (self.K * self.K)

    def gecis_matrisi(self) -> tuple[np.ndarray, list[str]]:
        """Durumlar x durumlar olasilik matrisi (heatmap icin)."""
        M = np.zeros((self.K, self.K), dtype=float)
        indeks = {s: i for i, s in enumerate(self.durumlar)}
        for i, kaynak in enumerate(self.durumlar):
            for j, hedef in enumerate(self.durumlar):
                M[i, j] = self.gecis_olasiligi(kaynak, hedef)
        return M, self.durumlar
