"""Veri bolme stratejileri.

- SKAB  : dosya bazli StratifiedGroupKFold (ayni .csv hem egitimde hem testte
          olamaz; satir bazli rastgele bolme YAPILMAZ).
- BATADAL: zaman sirali %60 egitim / %20 dogrulama / %20 test.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def skab_foldlar(ham, n_splits: int, seed: int):
    """SKAB icin (egitim_idx, test_idx) fold'larini uretir."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for egitim_idx, test_idx in sgkf.split(ham.X, ham.y, groups=ham.gruplar):
        yield egitim_idx, test_idx


def grup_train_val_bol(egitim_idx: np.ndarray, gruplar: np.ndarray, val_orani: float, seed: int):
    """Egitim indekslerini grup (dosya) butunlugunu koruyarak train/val ayirir."""
    rng = np.random.default_rng(seed)
    egitim_gruplar = np.array(sorted(set(gruplar[egitim_idx])))
    rng.shuffle(egitim_gruplar)
    n_val = max(1, int(round(len(egitim_gruplar) * val_orani)))
    val_kume = set(egitim_gruplar[:n_val].tolist())
    val_idx = np.array([i for i in egitim_idx if gruplar[i] in val_kume], dtype=int)
    yeni_egitim_idx = np.array([i for i in egitim_idx if gruplar[i] not in val_kume], dtype=int)
    return yeni_egitim_idx, val_idx


def zaman_sirali_bol(n: int, egitim_orani: float, dogrulama_orani: float):
    """BATADAL icin zaman sirali (egitim_idx, val_idx, test_idx)."""
    i1 = int(n * egitim_orani)
    i2 = int(n * (egitim_orani + dogrulama_orani))
    egitim_idx = np.arange(0, i1)
    val_idx = np.arange(i1, i2)
    test_idx = np.arange(i2, n)
    return egitim_idx, val_idx, test_idx
