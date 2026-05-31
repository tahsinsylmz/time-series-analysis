"""Olasiliksal otomata birim testleri (Laplace yumusatma, cozumleme, yol)."""
import numpy as np
import pytest

from src.models.automata.automaton import OlasiliksalOtomata


def kucuk_otomata() -> OlasiliksalOtomata:
    """aa --2--> ab --1--> bb seklinde kucuk bir otomata kurar."""
    oto = OlasiliksalOtomata(window_size=2, alphabet_size=2, laplace_alpha=1.0)
    for p in ("aa", "ab", "bb"):
        oto.sozluge_ekle(p)
    oto.gecis_ekle("aa", "ab")
    oto.gecis_ekle("aa", "ab")
    oto.gecis_ekle("ab", "bb")
    oto.sonlandir()
    return oto


def test_durumlar_sirali_ve_K():
    oto = kucuk_otomata()
    assert oto.durumlar == ["aa", "ab", "bb"]
    assert oto.K == 3


def test_laplace_olasiligi_formul():
    oto = kucuk_otomata()
    # P(aa->ab) = (2 + 1) / (2 + 1*3) = 3/5
    assert oto.gecis_olasiligi("aa", "ab") == pytest.approx(0.6)
    # Gorulmemis gecis: P(aa->aa) = (0 + 1) / (2 + 3) = 1/5
    assert oto.gecis_olasiligi("aa", "aa") == pytest.approx(0.2)


def test_olasiliklar_toplami_bir():
    oto = kucuk_otomata()
    for kaynak in oto.durumlar:
        toplam = sum(oto.gecis_olasiligi(kaynak, hedef) for hedef in oto.durumlar)
        assert toplam == pytest.approx(1.0)


def test_cikissiz_durum_uniform():
    oto = kucuk_otomata()
    # "bb" durumunun cikis gecisi yok -> tum hedeflere esit (1/K).
    for hedef in oto.durumlar:
        assert oto.gecis_olasiligi("bb", hedef) == pytest.approx(1.0 / 3.0)


def test_pattern_coz_gorulen():
    oto = kucuk_otomata()
    assert oto.pattern_coz("ab") == ("ab", False, "ab", 0)


def test_pattern_coz_gorulmemis():
    oto = kucuk_otomata()
    etkin, unseen, en_yakin, mesafe = oto.pattern_coz("ba")
    assert unseen is True
    assert mesafe == 1
    assert etkin == en_yakin
    assert en_yakin in oto.durumlar


def test_path_olasiligi_carpim():
    oto = kucuk_otomata()
    # P(aa->ab) * P(ab->bb) = 0.6 * 0.5 = 0.3
    assert oto.path_olasiligi(["aa", "ab", "bb"]) == pytest.approx(0.3)


def test_path_tek_eleman_bir():
    oto = kucuk_otomata()
    assert oto.path_olasiligi(["aa"]) == 1.0


def test_gecis_yogunlugu():
    oto = kucuk_otomata()
    # 2 farkli gecis / (3*3) olasi gecis
    assert oto.gecis_yogunlugu() == pytest.approx(2.0 / 9.0)


def test_gecis_matrisi_boyut_ve_satir_toplami():
    oto = kucuk_otomata()
    M, durumlar = oto.gecis_matrisi()
    assert M.shape == (3, 3)
    assert durumlar == oto.durumlar
    np.testing.assert_allclose(M.sum(axis=1), np.ones(3))
