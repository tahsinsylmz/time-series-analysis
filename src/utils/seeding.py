"""Tekrar uretilebilirlik icin merkezi seed ayari.

Her deney sabit seedler ([42, 123, 2026, 7, 999]) ile calistirilir; bu modul
random, numpy ve torch icin ayni seed'i ayarlayarak sonuclarin
tekrarlanabilir olmasini saglar.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_ayarla(seed: int) -> None:
    """random, numpy ve (varsa) torch icin seed ayarlar."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
