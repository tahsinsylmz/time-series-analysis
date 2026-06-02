"""Veri yukleme doldurma davranisi birim testleri.

BATADAL (gruplu olmayan, zaman sirali) dalda doldurmanin yalniz nedensel
(ileri-yon, ffill) oldugunu; geri-doldurma (bfill) yapilmadigini ve doldurma
sonrasi model-girdisi sutunlarinda sessizce NaN birakilmadigini (guvenlik
asserti) dogrular.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_loader import _eksikleri_doldur


def test_grupsuz_ffill_strateji_yalniz_ileri_yon():
    # 'ffill' stratejisi: NaN onceki gecerli deger ile (ileri-yon) doldurulur;
    # degerler gelecekten gelmez (bfill yok). Bastaki deger gecerli -> NaN kalmaz.
    df = pd.DataFrame({"s1": [1.0, np.nan, np.nan, 4.0, np.nan]})
    sonuc = _eksikleri_doldur(df, ["s1"], strateji="ffill")
    np.testing.assert_array_equal(sonuc["s1"].to_numpy(), [1.0, 1.0, 1.0, 4.0, 4.0])


def test_grupsuz_interpolate_strateji_ileri_yon():
    # Varsayilan 'interpolate' strateji: ic NaN ileri-yon dogrusal interpolasyonla
    # dolar; bastaki deger gecerliyse NaN kalmaz, geriye dogru tasima olmaz.
    df = pd.DataFrame({"s1": [1.0, np.nan, 3.0]})
    sonuc = _eksikleri_doldur(df, ["s1"])
    np.testing.assert_allclose(sonuc["s1"].to_numpy(), [1.0, 2.0, 3.0])


def test_grupsuz_geri_doldurma_yapmaz_bastaki_nan_assert():
    # bfill kullanilsaydi bastaki NaN sonraki (gelecek) degerle dolardi; nedensel
    # ffill-only doldurmada bos kalir ve guvenlik asserti tetiklenir (sessizce
    # NaN birakilmaz).
    df = pd.DataFrame({"s1": [np.nan, np.nan, 2.0]})
    with pytest.raises(AssertionError):
        _eksikleri_doldur(df, ["s1"])


def test_grupsuz_bastaki_nan_yoksa_tam_dolar():
    # Bastaki deger gecerli ise ffill tum bosluklari kapatir; NaN kalmaz.
    df = pd.DataFrame({"s1": [0.0, np.nan, np.nan, 4.0], "s2": [1.0, 2.0, np.nan, 5.0]})
    sonuc = _eksikleri_doldur(df, ["s1", "s2"])
    assert not sonuc[["s1", "s2"]].isna().any().any()


def test_grupsuz_meta_sutun_etkilenmez():
    # Doldurma yalniz verilen sutunlar uzerinde calisir; meta sutun degismez.
    df = pd.DataFrame({"s1": [1.0, np.nan, 3.0], "meta": ["a", "b", "c"]})
    sonuc = _eksikleri_doldur(df, ["s1"])
    assert list(sonuc["meta"]) == ["a", "b", "c"]


def test_grupsuz_tamamen_dolu_sutun_degismez():
    # Bos olmayan girdide degerler aynen kalir.
    df = pd.DataFrame({"s1": [1.0, 2.0, 3.0]})
    sonuc = _eksikleri_doldur(df, ["s1"])
    np.testing.assert_array_equal(sonuc["s1"].to_numpy(), [1.0, 2.0, 3.0])


def test_gruplu_doldurma_grup_icinde_kalir_ve_nan_kalmaz():
    # Gruplu (SKAB) dalda interpolasyon + iki-yon doldurma grup icinde tum
    # bosluklari kapatir; gruplar arasi tasima olmaz.
    df = pd.DataFrame({
        "s1": [np.nan, 2.0, np.nan, 10.0, np.nan, 30.0],
        "grup": ["a", "a", "a", "b", "b", "b"],
    })
    sonuc = _eksikleri_doldur(df, ["s1"], grup_sutunu="grup")
    assert not sonuc["s1"].isna().any()
    # a grubunun bastaki NaN'i a-ici (2.0) ile dolar, b grubundan (10.0) gelmez
    assert sonuc["s1"].iloc[0] == 2.0
