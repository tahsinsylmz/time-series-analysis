"""Istatistiksel test ve hizalama yardimcilari icin birim testler."""
import math

import numpy as np

from src.experiments.stats_tests import hizala, mcnemar_testi, wilcoxon_imzali


def test_wilcoxon_az_ornekte_nan():
    s = wilcoxon_imzali(np.array([0.5]), np.array([0.4]))
    assert math.isnan(s["istatistik"]) and math.isnan(s["p_deger"])
    assert s["n"] == 1


def test_wilcoxon_tum_esit_nan():
    a = np.array([0.1, 0.2, 0.3])
    s = wilcoxon_imzali(a, a.copy())
    assert math.isnan(s["p_deger"])


def test_wilcoxon_gercek_fark_p_uretir():
    a = np.array([0.9, 0.8, 0.85, 0.95, 0.7])
    b = np.array([0.1, 0.2, 0.15, 0.05, 0.3])
    s = wilcoxon_imzali(a, b)
    assert not math.isnan(s["p_deger"])
    assert s["a_ortalama"] > s["b_ortalama"]


def test_mcnemar_uyumsuz_cift_yoksa_nan():
    y = np.array([1, 0, 1, 0])
    a = y.copy()
    b = y.copy()                          # iki model de ayni (hep dogru) -> n10=n01=0
    s = mcnemar_testi(y, a, b)
    assert math.isnan(s["istatistik"]) and math.isnan(s["p_deger"])
    assert s["yalniz_a_dogru"] == 0 and s["yalniz_b_dogru"] == 0


def test_mcnemar_uyumsuz_ciftle_deger_uretir():
    y = np.ones(20, dtype=int)
    a = np.ones(20, dtype=int)            # hepsi dogru
    b = np.zeros(20, dtype=int)           # hepsi yanlis
    s = mcnemar_testi(y, a, b)
    assert s["yalniz_a_dogru"] == 20 and s["yalniz_b_dogru"] == 0
    assert not math.isnan(s["p_deger"])


def test_hizala_ortak_konumlar():
    konum_a = np.array([1, 2, 3])
    tahmin_a = np.array([1, 0, 1])
    konum_b = np.array([2, 3, 4])
    tahmin_b = np.array([0, 1, 1])
    y_konum = np.array([1, 2, 3, 4])
    y_deger = np.array([0, 1, 1, 0])
    yt, pa, pb = hizala(konum_a, tahmin_a, konum_b, tahmin_b, y_konum, y_deger)
    # ortak konumlar: 2 ve 3
    assert yt.tolist() == [1, 1]
    assert pa.tolist() == [0, 1]          # tahmin_a[2], tahmin_a[3]
    assert pb.tolist() == [0, 1]          # tahmin_b[2], tahmin_b[3]


def test_mcnemar_cikti_anahtarlari_temel():
    # Saf mcnemar_testi temel anahtarlari uretir; dejenere bayragi (yorum_disi)
    # runner katmaninda (_mcnemar_hesapla) eklenir. Bu test temel sozlesmeyi korur.
    y = np.ones(20, dtype=int)
    a = np.ones(20, dtype=int)
    b = np.zeros(20, dtype=int)
    s = mcnemar_testi(y, a, b)
    assert {"istatistik", "p_deger", "yalniz_a_dogru", "yalniz_b_dogru", "n"}.issubset(s)
