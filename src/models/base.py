"""Model katmani icin ortak arayuz (SOLID - bagimliligin tersine cevrilmesi).

Hem derin ogrenme hem de otomata modelleri ``AnomaliModeli`` arayuzunu uygular.
Deney yoneticisi somut sinifa degil, bu soyut arayuze bagimlidir; boylece yeni
bir model eklemek mevcut kodu degistirmeyi gerektirmez (acik/kapali ilkesi).

``ModelGirdisi`` her iki model turunun de ihtiyac duydugu gosterimleri tasir:
- ``X_olcekli`` : normalize edilmis cok degiskenli veri (derin ogrenme girdisi)
- ``pc1``       : PCA ile tek boyuta indirgenmis seri (otomata girdisi)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ModelGirdisi:
    """Bir veri bolumunun model-bagimsiz gosterimi."""

    X_olcekli: np.ndarray   # (N, F) normalize cok degiskenli
    pc1: np.ndarray         # (N,)  tek boyut (ham PC1)
    y: np.ndarray           # (N,)  ikili etiket (0/1)
    segmentler: np.ndarray  # (N,)  segment/dosya kimligi (pencere sinirlari icin)


class AnomaliModeli(ABC):
    """Tum anomali tespit modelleri icin ortak arayuz."""

    ad: str = "model"
    esik: float = 0.0

    @abstractmethod
    def egit(self, egitim: ModelGirdisi, dogrulama: "ModelGirdisi | None" = None) -> "AnomaliModeli":
        """Modeli egitim verisiyle egitir; karar esigini dogrulama setinde secer."""

    @abstractmethod
    def skor(self, veri: ModelGirdisi) -> tuple[np.ndarray, np.ndarray]:
        """Her degerlendirilebilir nokta icin anomali skoru (yuksek = anomali) ve
        bu noktalarin ``veri`` dizilerindeki konumlarini dondurur."""

    def tahmin_et(self, veri: ModelGirdisi) -> tuple[np.ndarray, np.ndarray]:
        """Karar esigine gore 0/1 tahmin ve konumlari dondurur."""
        skorlar, konumlar = self.skor(veri)
        return (skorlar >= self.esik).astype(int), konumlar
