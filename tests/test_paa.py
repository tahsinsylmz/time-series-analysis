"""PAA (Piecewise Aggregate Approximation) birim testleri."""
import numpy as np
import pytest

from src.models.automata.paa import paa_segment


def test_segment_sayisi_uzunluga_esit_kimlik():
    pencere = np.array([1.0, 2.0, 3.0, 4.0])
    sonuc = paa_segment(pencere, 4)
    np.testing.assert_array_equal(sonuc, pencere)


def test_ortalama_ile_indirger():
    pencere = np.array([1.0, 2.0, 3.0, 4.0])
    sonuc = paa_segment(pencere, 2)
    np.testing.assert_allclose(sonuc, [1.5, 3.5])


def test_sabit_seri_sabit_kalir():
    pencere = np.full(6, 5.0)
    sonuc = paa_segment(pencere, 3)
    np.testing.assert_allclose(sonuc, [5.0, 5.0, 5.0])


def test_bolunemeyen_uzunluk_hata():
    with pytest.raises(ValueError):
        paa_segment(np.arange(5.0), 2)


def test_pozitif_olmayan_segment_hata():
    with pytest.raises(ValueError):
        paa_segment(np.arange(4.0), 0)
