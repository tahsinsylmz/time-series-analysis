"""SAX (Symbolic Aggregate approXimation) birim testleri."""
import numpy as np
import pytest

from src.models.automata.sax import sax_kesim_noktalari, sax_sembolize


def test_kesim_noktasi_sayisi():
    # alphabet_size icin (alphabet_size - 1) kesim noktasi uretilir.
    for a in (2, 3, 4, 5):
        assert len(sax_kesim_noktalari(a)) == a - 1


def test_ikili_alfabe_kesimi_sifir():
    # 2 harf icin tek kesim noktasi Gauss medyaninda (0) olmali.
    np.testing.assert_allclose(sax_kesim_noktalari(2), [0.0], atol=1e-9)


def test_kesim_noktalari_simetrik_ve_artan():
    kesimler = sax_kesim_noktalari(4)
    assert np.all(np.diff(kesimler) > 0)            # artan
    np.testing.assert_allclose(kesimler, -kesimler[::-1], atol=1e-9)  # simetrik


def test_gecersiz_alfabe_hata():
    with pytest.raises(ValueError):
        sax_kesim_noktalari(1)


def test_sembolize_dogru_harfler():
    kesimler = sax_kesim_noktalari(3)  # ~[-0.43, 0.43]
    # dusuk -> 'a', orta -> 'b', yuksek -> 'c'
    assert sax_sembolize(np.array([-1.0, 0.0, 1.0]), kesimler) == "abc"


def test_sembolize_monotonluk():
    # Artan degerler -> azalmayan harf dizisi.
    kesimler = sax_kesim_noktalari(4)
    kelime = sax_sembolize(np.array([-2.0, -0.1, 0.1, 2.0]), kesimler)
    assert list(kelime) == sorted(kelime)


def test_sembolize_uzunlugu_girdi_ile_ayni():
    kesimler = sax_kesim_noktalari(3)
    assert len(sax_sembolize(np.array([0.1, 0.2, 0.3, 0.4]), kesimler)) == 4
