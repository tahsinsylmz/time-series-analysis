"""Gurultu senaryosu icin Gaussian gurultu enjeksiyonu.

Senaryo testi (dayaniklilik) test aninda uygulanir: model temiz veriyle
egitilir, ardindan normalize edilmis test verisine N(0, std) gurultu eklenir.
"""
from __future__ import annotations

import numpy as np


def gaussian_gurultu_ekle(X: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Normalize edilmis veriye sifir ortalamali Gaussian gurultu ekler."""
    return X + rng.normal(loc=0.0, scale=std, size=X.shape)
