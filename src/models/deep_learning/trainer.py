"""Derin ogrenme egitim dongusu (early stopping + sinif agirligi).

Egitim, dogrulama kaybi iyilesmeyi birakinca erken durur ve en iyi (en dusuk
dogrulama kayipli) agirliklar geri yuklenir. Dengesiz veride pozitif sinifa
``pos_weight`` ile agirlik verilir.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.deep_learning.datasets import PencereVeriKumesi


def _pos_weight(etiketler: np.ndarray) -> float:
    """Pozitif sinif agirligi = negatif/pozitif (dengesizligi telafi eder)."""
    pozitif = int(etiketler.sum())
    negatif = int(len(etiketler) - pozitif)
    if pozitif == 0:
        return 1.0
    return negatif / pozitif


def egit_dongusu(
    ag: nn.Module,
    egitim_pencere: np.ndarray,
    egitim_etiket: np.ndarray,
    val_pencere: np.ndarray,
    val_etiket: np.ndarray,
    cfg,
) -> nn.Module:
    """Agi egitir ve en iyi dogrulama agirliklariyla geri dondurur."""
    dc = cfg.derin_ogrenme
    cihaz = torch.device("cpu")
    ag = ag.to(cihaz)

    yukleyici = DataLoader(
        PencereVeriKumesi(egitim_pencere, egitim_etiket),
        batch_size=dc.batch_boyutu,
        shuffle=True,
    )
    pw = torch.tensor([_pos_weight(egitim_etiket)] if dc.sinif_agirligi else [1.0], dtype=torch.float32)
    kayip_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(ag.parameters(), lr=dc.ogrenme_orani)

    val_X = torch.from_numpy(val_pencere).to(cihaz) if len(val_pencere) else None
    val_y = torch.from_numpy(val_etiket.astype(np.float32)).to(cihaz) if len(val_etiket) else None

    en_iyi_kayip = float("inf")
    en_iyi_durum = copy.deepcopy(ag.state_dict())
    sabir = 0

    for _ in range(dc.epoch):
        ag.train()
        for xb, yb in yukleyici:
            optimizer.zero_grad()
            logit = ag(xb.to(cihaz))
            kayip = kayip_fn(logit, yb.to(cihaz))
            kayip.backward()
            optimizer.step()

        # Dogrulama kaybi (yoksa egitim kaybiyla izlenir)
        ag.eval()
        with torch.no_grad():
            if val_X is not None and len(val_X) > 0:
                val_kayip = float(kayip_fn(ag(val_X), val_y))
            else:
                val_kayip = float(kayip.detach())
        if val_kayip < en_iyi_kayip - 1e-5:
            en_iyi_kayip = val_kayip
            en_iyi_durum = copy.deepcopy(ag.state_dict())
            sabir = 0
        else:
            sabir += 1
            if sabir >= dc.early_stopping_sabri:
                break

    ag.load_state_dict(en_iyi_durum)
    return ag


def olasilik_uret(ag: nn.Module, pencereler: np.ndarray) -> np.ndarray:
    """Pencereler icin sigmoid anomali olasiligi dizisi uretir."""
    if len(pencereler) == 0:
        return np.empty(0, dtype=float)
    ag.eval()
    with torch.no_grad():
        logit = ag(torch.from_numpy(pencereler))
        return torch.sigmoid(logit).cpu().numpy().astype(float)
