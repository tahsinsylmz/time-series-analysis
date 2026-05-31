"""Istatistiksel anlamlilik testleri (model karsilastirmasi).

- Wilcoxon isaretli sira testi: eslestirilmis F1 skorlarini (ornegin SKAB
  fold'lari) karsilastirir; iki modelin performans farkinin rastlantisal olup
  olmadigini sinar (parametrik olmayan, normallik varsayimi gerektirmez).
- McNemar testi: ayni test ornekleri uzerinde iki modelin dogru/yanlis karar
  desenini karsilastirir (eslestirilmis ikili kararlar).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def wilcoxon_imzali(a: np.ndarray, b: np.ndarray) -> dict:
    """Eslestirilmis ``a`` ve ``b`` (orn. fold bazli F1) icin Wilcoxon testi."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sonuc = {"istatistik": float("nan"), "p_deger": float("nan"), "n": int(a.size),
             "a_ortalama": float(np.mean(a)) if a.size else float("nan"),
             "b_ortalama": float(np.mean(b)) if b.size else float("nan")}
    if a.size < 2 or np.allclose(a, b):
        return sonuc
    try:
        stat, p = wilcoxon(a, b)
        sonuc["istatistik"] = float(stat)
        sonuc["p_deger"] = float(p)
    except ValueError:
        pass
    return sonuc


def mcnemar_testi(y_true: np.ndarray, tahmin_a: np.ndarray, tahmin_b: np.ndarray) -> dict:
    """Iki modelin ayni noktalardaki kararlari icin McNemar testi.

    ``tahmin_a`` ve ``tahmin_b`` ayni ``y_true`` noktalarina hizali olmalidir.
    """
    y_true = np.asarray(y_true, dtype=int)
    a_dogru = np.asarray(tahmin_a, dtype=int) == y_true
    b_dogru = np.asarray(tahmin_b, dtype=int) == y_true
    n11 = int(np.sum(a_dogru & b_dogru))
    n10 = int(np.sum(a_dogru & ~b_dogru))   # yalniz A dogru
    n01 = int(np.sum(~a_dogru & b_dogru))   # yalniz B dogru
    n00 = int(np.sum(~a_dogru & ~b_dogru))
    tablo = [[n11, n10], [n01, n00]]
    sonuc = {"istatistik": float("nan"), "p_deger": float("nan"),
             "yalniz_a_dogru": n10, "yalniz_b_dogru": n01, "n": int(y_true.size)}
    if n10 + n01 > 0:
        # Az sayida uyumsuz cift varsa exact binom, yoksa duzeltmeli ki-kare
        test = mcnemar(tablo, exact=(n10 + n01) < 25, correction=True)
        sonuc["istatistik"] = float(test.statistic)
        sonuc["p_deger"] = float(test.pvalue)
    return sonuc


def hizala(konum_a: np.ndarray, tahmin_a: np.ndarray,
           konum_b: np.ndarray, tahmin_b: np.ndarray,
           y_konum: np.ndarray, y_deger: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iki modelin ortak konumlarindaki tahminlerini ve gercek etiketi eslestirir."""
    a_harita = {int(k): int(t) for k, t in zip(konum_a, tahmin_a)}
    b_harita = {int(k): int(t) for k, t in zip(konum_b, tahmin_b)}
    y_harita = {int(k): int(v) for k, v in zip(y_konum, y_deger)}
    ortak = sorted(set(a_harita) & set(b_harita) & set(y_harita))
    pa = np.array([a_harita[k] for k in ortak], dtype=int)
    pb = np.array([b_harita[k] for k in ortak], dtype=int)
    yt = np.array([y_harita[k] for k in ortak], dtype=int)
    return yt, pa, pb
