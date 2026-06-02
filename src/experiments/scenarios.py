"""Test-zamani senaryolari (dayaniklilik analizi).

Model temiz veriyle bir kez egitilir, ardindan uc kosulda degerlendirilir:
  - orijinal : test verisi oldugu gibi,
  - gurultu  : normalize veriye Gauss gurultusu eklenir (sensor bozulmasi),
  - unseen   : genlik (kazanc) kaydirmasiyla uretilen dagilim disi (covariate
               shift) test girdisi; egitimde gorulmemis SAX pattern'leri tetikler.

Ucunun de uctan uca girdisi ayri uretilir. Genlik kaydirmasi yalnizca otomata
icin sozluk-disi (out-of-vocabulary) pattern'ler dogurur; bu sozluk-disi
yonetimin sayisal degerlendirmesi (Detection Rate / Mapping Accuracy) ayrica
yapilir (bkz. automata_model.unseen_analizi, VI.A).
"""
from __future__ import annotations

import numpy as np

from src.models.base import ModelGirdisi
from src.preprocessing.preprocess import OnIslemci


def temiz_girdi(on: OnIslemci, X: np.ndarray, y: np.ndarray, seg: np.ndarray) -> ModelGirdisi:
    """Olceklenmis cok degiskenli + PC1 gosterimli temiz girdi."""
    return ModelGirdisi(X_olcekli=on.olcekle(X), pc1=on.pc1(X), y=y, segmentler=seg)


def gurultulu_girdi(
    on: OnIslemci, X: np.ndarray, y: np.ndarray, seg: np.ndarray, std: float, rng: np.random.Generator
) -> ModelGirdisi:
    """Normalize veriye Gauss gurultusu eklenmis girdi (PC1 tutarli sekilde yeniden hesaplanir)."""
    olcekli = on.olcekle(X)
    gurultulu = olcekli + rng.normal(0.0, std, size=olcekli.shape)
    pc1 = on.pc1_olcekliden(gurultulu)
    return ModelGirdisi(X_olcekli=gurultulu, pc1=pc1, y=y, segmentler=seg)


def kaydirilmis_girdi(
    on: OnIslemci, X: np.ndarray, y: np.ndarray, seg: np.ndarray, faktor: float
) -> ModelGirdisi:
    """Normalize sinyali ``faktor`` ile olceklenmis girdi (sensor kazanc kaymasi).

    Genlik kaymasi, egitimde gorulmemis (daha uc) SAX pattern'leri ureterek
    ``unseen`` (novelty) senaryosunu olusturur.
    """
    olcekli = on.olcekle(X) * float(faktor)
    pc1 = on.pc1_olcekliden(olcekli)
    return ModelGirdisi(X_olcekli=olcekli, pc1=pc1, y=y, segmentler=seg)
