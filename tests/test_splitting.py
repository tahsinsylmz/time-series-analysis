"""Veri bolme ve bitisik segment yardimcilari birim testleri.

Kapsam:
- grup_train_val_bol: sizinti-kritik; train/val grup (dosya) kesisimi BOS,
  birlesim tum egitim gruplarini kapsar.
- bitisik_bloklar: bos -> [], tek deger -> [(0, n)], cok blok, bitisik-olmayan id.
- zaman_sirali_bol: oranlar, zaman sirasi ve tam kapsama (kayip satir yok).
"""
import numpy as np
import pytest

from src.preprocessing.splitting import grup_train_val_bol, zaman_sirali_bol
from src.utils.segments import bitisik_bloklar


# ---------------------------------------------------------------------------
# grup_train_val_bol (sizinti regresyonu)
# ---------------------------------------------------------------------------

def test_grup_bol_kesisim_bos_ve_birlesim_tam():
    # 6 dosya (grup), her dosyadan 2 satir; egitim indeksleri tum satirlar.
    gruplar = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    egitim_idx = np.arange(len(gruplar))
    yeni_egitim_idx, val_idx = grup_train_val_bol(
        egitim_idx, gruplar, val_orani=0.34, seed=0
    )
    egitim_gruplar = set(gruplar[yeni_egitim_idx].tolist())
    val_gruplar = set(gruplar[val_idx].tolist())
    # ayni dosya hem train hem val'de olamaz (grup butunlugu / sizinti yok)
    assert egitim_gruplar.isdisjoint(val_gruplar)
    # birlesim, baslangictaki tum egitim gruplarini kapsar
    assert egitim_gruplar | val_gruplar == set(gruplar[egitim_idx].tolist())


def test_grup_bol_indeks_butunlugu_ve_kapsama():
    gruplar = np.array([10, 10, 10, 20, 20, 30, 40, 40])
    egitim_idx = np.arange(len(gruplar))
    yeni_egitim_idx, val_idx = grup_train_val_bol(
        egitim_idx, gruplar, val_orani=0.5, seed=7
    )
    # satir indeksleri ortusmez ve tum egitim indekslerini tam kapsar
    assert set(yeni_egitim_idx.tolist()).isdisjoint(set(val_idx.tolist()))
    birlesim = np.sort(np.concatenate([yeni_egitim_idx, val_idx]))
    np.testing.assert_array_equal(birlesim, np.sort(egitim_idx))
    # bir dosyanin tum satirlari ayni tarafa gider (grup butunlugu)
    for g in np.unique(gruplar):
        satirlar = set(np.where(gruplar == g)[0].tolist())
        train_tarafi = satirlar <= set(yeni_egitim_idx.tolist())
        val_tarafi = satirlar <= set(val_idx.tolist())
        assert train_tarafi or val_tarafi


def test_grup_bol_val_en_az_bir_grup():
    # kucuk val_orani -> max(1, ...) ile en az bir grup val'e dusmeli
    gruplar = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    egitim_idx = np.arange(len(gruplar))
    yeni_egitim_idx, val_idx = grup_train_val_bol(
        egitim_idx, gruplar, val_orani=0.01, seed=3
    )
    assert len(val_idx) > 0
    assert len(set(gruplar[val_idx].tolist())) >= 1
    # egitim tarafi da bos kalmamali
    assert len(yeni_egitim_idx) > 0


def test_grup_bol_egitim_idx_alt_kumesi():
    # egitim_idx tum satirlarin alt kumesi oldugunda yalniz o gruplar bolunur
    gruplar = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    egitim_idx = np.array([0, 1, 2, 3, 4, 5])  # yalniz 0,1,2 gruplari
    yeni_egitim_idx, val_idx = grup_train_val_bol(
        egitim_idx, gruplar, val_orani=0.34, seed=1
    )
    secilen = set(gruplar[np.concatenate([yeni_egitim_idx, val_idx])].tolist())
    # egitim_idx disindaki gruplar (3, 4) sonuca sizmaz
    assert secilen == {0, 1, 2}
    assert set(yeni_egitim_idx.tolist()) | set(val_idx.tolist()) == set(egitim_idx.tolist())


def test_grup_bol_determinist_seed():
    gruplar = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    egitim_idx = np.arange(len(gruplar))
    a1, b1 = grup_train_val_bol(egitim_idx, gruplar, val_orani=0.34, seed=42)
    a2, b2 = grup_train_val_bol(egitim_idx, gruplar, val_orani=0.34, seed=42)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)


# ---------------------------------------------------------------------------
# bitisik_bloklar
# ---------------------------------------------------------------------------

def test_bitisik_bos_girdi():
    assert bitisik_bloklar([]) == []


def test_bitisik_tek_blok_tum_aralik():
    assert bitisik_bloklar([5, 5, 5, 5]) == [(0, 4)]
    # tek elemanli girdi
    assert bitisik_bloklar([7]) == [(0, 1)]


def test_bitisik_cok_blok():
    # docstring ornegi: ['a','a','b','b','b'] -> [(0, 2), (2, 5)]
    assert bitisik_bloklar(["a", "a", "b", "b", "b"]) == [(0, 2), (2, 5)]
    assert bitisik_bloklar([0, 0, 1, 0]) == [(0, 2), (2, 3), (3, 4)]


def test_bitisik_hepsi_farkli():
    assert bitisik_bloklar([1, 2, 3]) == [(0, 1), (1, 2), (2, 3)]


def test_bitisik_olmayan_id_tekrar_yeni_blok():
    # ayni deger bitisik degilse ayri bloklar olur (id tekrari birlesmez)
    etiketler = [1, 1, 2, 1, 1]
    bloklar = bitisik_bloklar(etiketler)
    assert bloklar == [(0, 2), (2, 3), (3, 5)]


def test_bitisik_bloklar_tam_kapsama():
    # bloklarin birlesimi tum aralik; bitisik (son == sonraki bas); kapsama tam
    etiketler = ["x", "x", "y", "z", "z", "x"]
    bloklar = bitisik_bloklar(etiketler)
    assert bloklar[0][0] == 0
    assert bloklar[-1][1] == len(etiketler)
    for (b1, s1), (b2, s2) in zip(bloklar, bloklar[1:]):
        assert s1 == b2  # bitisik, bosluk yok
    # her satir tam bir blokta
    kapsanan = sum(s - b for b, s in bloklar)
    assert kapsanan == len(etiketler)


# ---------------------------------------------------------------------------
# zaman_sirali_bol
# ---------------------------------------------------------------------------

def test_zaman_bol_oranlar():
    egitim_idx, val_idx, test_idx = zaman_sirali_bol(100, 0.6, 0.2)
    assert len(egitim_idx) == 60
    assert len(val_idx) == 20
    assert len(test_idx) == 20


def test_zaman_bol_sira_korunur():
    egitim_idx, val_idx, test_idx = zaman_sirali_bol(100, 0.6, 0.2)
    # her dilim icinde artan sirali ve dilimler arasi zaman sirasi korunur
    np.testing.assert_array_equal(egitim_idx, np.arange(0, 60))
    np.testing.assert_array_equal(val_idx, np.arange(60, 80))
    np.testing.assert_array_equal(test_idx, np.arange(80, 100))
    assert egitim_idx[-1] < val_idx[0] < test_idx[0]


def test_zaman_bol_tam_kapsama_kayip_satir_yok():
    n = 137  # tam bolunmeyen uzunluk
    egitim_idx, val_idx, test_idx = zaman_sirali_bol(n, 0.6, 0.2)
    birlesim = np.concatenate([egitim_idx, val_idx, test_idx])
    # ortusme yok, bosluk yok: 0..n-1 tam kapsanir
    np.testing.assert_array_equal(birlesim, np.arange(n))
    assert len(birlesim) == n
    # dilimler ortusmez
    assert len(set(birlesim.tolist())) == n


def test_zaman_bol_test_kalan_tum_satirlar():
    # test dilimi (egitim+dogrulama)'dan sonraki tum kalan satirlari alir
    n = 50
    egitim_idx, val_idx, test_idx = zaman_sirali_bol(n, 0.7, 0.1)
    assert test_idx[-1] == n - 1
    assert test_idx[0] == val_idx[-1] + 1


def test_zaman_bol_test_orani_tutarlilik_kontrolu():
    # test_orani verilince oranlarin toplami 1.0 degilse hata; dogru toplamda gecer.
    egitim_idx, val_idx, test_idx = zaman_sirali_bol(100, 0.6, 0.2, test_orani=0.2)
    assert len(test_idx) == 20
    with pytest.raises(ValueError):
        zaman_sirali_bol(100, 0.6, 0.2, test_orani=0.3)
