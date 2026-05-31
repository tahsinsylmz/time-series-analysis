"""Derin ogrenme icin kayan pencere (sliding window) veri hazirligi.

Cok degiskenli seri, sabit uzunlukta zaman pencerelerine bolunur. Her pencere
bir orneklemdir; etiketi pencerenin SON adimina aittir (etiket_stratejisi).
Pencereler segment (SKAB dosya) sinirlarini asmaz.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.segments import bitisik_bloklar


def pencereler_olustur(
    X: np.ndarray,
    y: np.ndarray,
    segmentler: np.ndarray,
    dizi_uzunlugu: int,
    adim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment sinirlarini koruyarak (pencereler, etiketler, konumlar) uretir.

    - pencereler : (M, dizi_uzunlugu, ozellik_sayisi)
    - etiketler  : (M,) pencerenin son adiminin etiketi
    - konumlar   : (M,) son adimin ``X`` icindeki mutlak indeksi
    """
    pencereler: list[np.ndarray] = []
    etiketler: list[int] = []
    konumlar: list[int] = []
    for bas, son in bitisik_bloklar(segmentler):
        for t in range(bas, son - dizi_uzunlugu + 1, adim):
            bitis = t + dizi_uzunlugu
            pencereler.append(X[t:bitis])
            etiketler.append(int(y[bitis - 1]))
            konumlar.append(bitis - 1)
    if not pencereler:
        F = X.shape[1]
        bos = np.empty((0, dizi_uzunlugu, F), dtype=np.float32)
        return bos, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (
        np.asarray(pencereler, dtype=np.float32),
        np.asarray(etiketler, dtype=np.int64),
        np.asarray(konumlar, dtype=np.int64),
    )


class PencereVeriKumesi(Dataset):
    """Torch DataLoader icin pencere/etiket sarmalayicisi."""

    def __init__(self, pencereler: np.ndarray, etiketler: np.ndarray) -> None:
        self.X = torch.from_numpy(pencereler)
        self.y = torch.from_numpy(etiketler.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]
