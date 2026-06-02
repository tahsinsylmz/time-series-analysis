"""Degerlendirme metrikleri (anomali = pozitif sinif).

Dengesiz veri setlerinde dogruluk (accuracy) yaniltici olabildiginden asil
olcut F1, precision ve recall'dur. Skor verildiginde ROC-AUC ve PR-AUC da
hesaplanir (her iki sinif da mevcutsa).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def ikili_metrikler(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    skorlar: np.ndarray | None = None,
    pozitif_sinif: int = 1,
) -> dict:
    """accuracy/precision/recall/f1 (+ skor verilirse roc_auc/pr_auc) dondurur.

    ``pozitif_sinif`` config: degerlendirme.pozitif_sinif (anomali sinifi).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pozitif_sinif, zero_division=0
    )
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "n": int(y_true.size),
        "pozitif": int((y_true == pozitif_sinif).sum()),
    }
    if skorlar is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, skorlar))
        m["pr_auc"] = float(average_precision_score(y_true, skorlar))
    return m
