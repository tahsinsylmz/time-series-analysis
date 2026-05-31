"""Levenshtein mesafesi ve en yakin pattern eslemesi birim testleri.

Proje gereksinimi: Levenshtein algoritmasi birim testlerle dogrulanmalidir.
"""
import pytest

from src.models.automata.levenshtein import en_yakin_pattern, levenshtein_mesafe


class TestLevenshteinMesafe:
    def test_ayni_dizi_sifir(self):
        assert levenshtein_mesafe("abc", "abc") == 0

    def test_bos_diziler(self):
        assert levenshtein_mesafe("", "") == 0
        assert levenshtein_mesafe("", "abc") == 3
        assert levenshtein_mesafe("abc", "") == 3

    def test_tek_degistirme(self):
        assert levenshtein_mesafe("aba", "aca") == 1

    def test_tek_ekleme(self):
        assert levenshtein_mesafe("ac", "abc") == 1

    def test_tek_silme(self):
        assert levenshtein_mesafe("abc", "ac") == 1

    def test_klasik_ornek_kitten_sitting(self):
        # Bilinen referans: kitten -> sitting = 3
        assert levenshtein_mesafe("kitten", "sitting") == 3

    def test_simetri(self):
        assert levenshtein_mesafe("aab", "bba") == levenshtein_mesafe("bba", "aab")

    def test_ucgen_esitsizligi(self):
        a, b, c = "aab", "abb", "bbb"
        assert levenshtein_mesafe(a, c) <= levenshtein_mesafe(a, b) + levenshtein_mesafe(b, c)


class TestEnYakinPattern:
    def test_tam_eslesme_sifir_mesafe(self):
        en_yakin, mesafe = en_yakin_pattern("abc", ["xyz", "abc", "aaa"])
        assert en_yakin == "abc"
        assert mesafe == 0

    def test_en_yakini_secer(self):
        en_yakin, mesafe = en_yakin_pattern("aab", ["aaa", "bbb", "ccc"])
        assert en_yakin == "aaa"
        assert mesafe == 1

    def test_esit_mesafede_alfabetik_ilk(self):
        # "aXa" hem "aaa" hem "aba"ya 1 mesafede; deterministik olarak "aaa".
        en_yakin, mesafe = en_yakin_pattern("aza", ["aba", "aaa"])
        assert mesafe == 1
        assert en_yakin == "aaa"

    def test_bos_sozluk(self):
        en_yakin, mesafe = en_yakin_pattern("abc", [])
        assert en_yakin is None
        assert mesafe == -1
