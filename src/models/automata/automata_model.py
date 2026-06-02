"""Otomata tabanli anomali tespit modeli.

PC1 (tek boyut) serisi uzerinde:
1. Egitim istatistikleriyle z-normalize (sizinti onleme)
2. Sliding window -> PAA -> SAX ile pattern (state) dizisi
3. NORMAL egitim verisinden frekans tabanli gecis olasiliklari
4. Test noktasi icin yol olasiligi dusukse anomali (dusuk olasilik = anomali)

Sozluk-disi (unseen / out-of-vocabulary) pattern'lar Levenshtein ile en yakin
bilinen pattern'a eslenir ve ek anomali cezasi alir. ``unseen`` deney senaryosu
ise genlik kaydirmasiyla (covariate shift) uretilen, dagilim disi noktalardan
olusur; sozluk-disi yonetiminin sayisal degerlendirmesi ``unseen_analizi`` ile
Detection Rate ve Mapping Accuracy olarak ayrica raporlanir (VI.A).
"""
from __future__ import annotations

import numpy as np

from src.models.automata.automaton import OlasiliksalOtomata
from src.models.automata.levenshtein import levenshtein_mesafe
from src.models.automata.paa import paa_segment
from src.models.automata.sax import sax_kesim_noktalari, sax_sembolize
from src.models.base import AnomaliModeli, ModelGirdisi
from src.utils.esik import f1_maksimize_esik
from src.utils.segments import bitisik_bloklar


class OtomataAnomaliModeli(AnomaliModeli):
    """``AnomaliModeli`` arayuzunu uygulayan olasiliksal otomata modeli."""

    ad = "automata"

    def __init__(self, cfg, window_size: int | None = None, alphabet_size: int | None = None) -> None:
        oc = cfg.otomata
        self.cfg = cfg
        self.w = int(window_size if window_size is not None else oc.window_size)
        self.a = int(alphabet_size if alphabet_size is not None else oc.alphabet_size)
        self.paa_faktoru = int(oc.paa_bolme_faktoru)
        self.alpha = float(oc.laplace_alpha)
        self.path_uzunlugu = int(oc.path_uzunlugu)
        self.unseen_ceza_agirligi = float(oc.unseen_ceza_agirligi)
        self.esik_aday_sayisi = int(cfg.degerlendirme.esik_aday_sayisi)
        self.tek_sinif_persentil = float(cfg.degerlendirme.tek_sinif_esik_persentili)
        self.smoothing = str(oc.smoothing).lower()
        if self.smoothing != "laplace":
            raise ValueError(
                f"Desteklenmeyen smoothing stratejisi: {oc.smoothing!r} "
                "(su an yalnizca 'laplace' uygulanmaktadir)."
            )
        self.esik_stratejisi = str(oc.esik_stratejisi).lower()
        if self.esik_stratejisi != "f1_maks":
            raise ValueError(
                f"Desteklenmeyen esik stratejisi: {oc.esik_stratejisi!r} "
                "(su an yalnizca 'f1_maks' uygulanmaktadir)."
            )
        self.ham_pencere = self.w * self.paa_faktoru   # ham nokta sayisi
        self.kesimler = sax_kesim_noktalari(self.a)
        self.oto: OlasiliksalOtomata | None = None
        self.pc1_mu = 0.0
        self.pc1_sd = 1.0
        self.esik = 0.0
        self.ref_skor_sd = 1.0   # guven skoru icin skor olcegi (egitimde hesaplanir)

    # ---- yardimci: bir segmentin pattern dizisi ----
    def _segment_kelimeleri(self, seri_norm: np.ndarray) -> list[str]:
        """Normalize segment icin sliding window -> PAA -> SAX pattern dizisi.

        kelime t -> ham bitis konumu = t + ham_pencere - 1
        """
        n = seri_norm.size
        L = self.ham_pencere
        kelimeler: list[str] = []
        for t in range(0, n - L + 1):
            pencere = seri_norm[t:t + L]
            paa = paa_segment(pencere, self.w)
            kelimeler.append(sax_sembolize(paa, self.kesimler))
        return kelimeler

    def _normalize(self, pc1: np.ndarray) -> np.ndarray:
        # GLOBAL z-normalizasyon (egitim mu/sd ile). Klasik SAX'in pencere-bazli
        # z-norm'unun aksine pencerelerin mutlak seviyesini korur; seviye/genlik
        # sapmasi olan anomaliler icin kasitli tercih (bkz. sax.py docstring).
        return (pc1 - self.pc1_mu) / self.pc1_sd

    # ---- egitim ----
    def egit(self, egitim: ModelGirdisi, dogrulama: ModelGirdisi | None = None) -> "OtomataAnomaliModeli":
        self.pc1_mu = float(np.mean(egitim.pc1))
        self.pc1_sd = float(np.std(egitim.pc1) + 1e-8)
        self.oto = OlasiliksalOtomata(self.w, self.a, self.alpha)
        L = self.ham_pencere

        for bas, son in bitisik_bloklar(egitim.segmentler):
            seri = self._normalize(egitim.pc1[bas:son])
            etiket = egitim.y[bas:son]
            kelimeler = self._segment_kelimeleri(seri)
            # Sozluk: tum egitim pattern'lari (unseen tespiti icin)
            for kelime in kelimeler:
                self.oto.sozluge_ekle(kelime)
            # Gecisler: yalnizca normal-normal ardisik pencereler (normal davranis modeli)
            for t in range(1, len(kelimeler)):
                onceki_bitis = (t - 1) + L - 1
                simdiki_bitis = t + L - 1
                if etiket[onceki_bitis] == 0 and etiket[simdiki_bitis] == 0:
                    self.oto.gecis_ekle(kelimeler[t - 1], kelimeler[t])
        self.oto.sonlandir()

        # Karar esigi: dogrulama (yoksa egitim) uzerinde F1 maksimize
        referans = dogrulama if dogrulama is not None else egitim
        skorlar, konumlar = self.skor(referans)
        if self.esik_stratejisi == "f1_maks":
            self.esik = f1_maksimize_esik(skorlar, referans.y[konumlar], self.esik_aday_sayisi,
                                          self.tek_sinif_persentil)
        self.ref_skor_sd = float(np.std(skorlar) + 1e-8)
        return self

    # ---- yol skoru (skorlama ve aciklama icin tek kaynak) ----
    def _yol_bilgisi(self, kelimeler: list[str], t: int) -> dict:
        """t indeksli pencere icin yol skoru ve aciklama bilesenlerini hesaplar.

        Hem ``skor`` hem de aciklayici bu tek fonksiyondan beslenir; boylece
        raporlanan olasiliklar ile karar uretiminde kullanilanlar daima ayni olur.
        """
        log_olasilik = 0.0
        unseen_ceza = 0.0
        gecisler: list[dict] = []
        for k in range(t - self.path_uzunlugu + 1, t + 1):
            kaynak_coz = self.oto.pattern_coz(kelimeler[k - 1])
            hedef_coz = self.oto.pattern_coz(kelimeler[k])
            kaynak, hedef = kaynak_coz[0], hedef_coz[0]
            p = self.oto.gecis_olasiligi(kaynak, hedef)
            log_olasilik += float(np.log(p + 1e-12))
            if hedef_coz[1]:  # unseen hedef pattern
                # hedef_coz[3] Levenshtein mesafesidir; sozluk bos oldugunda (K=0)
                # en_yakin_pattern -1 dondurur. Negatif cezayi (ve dolayisiyla negatif
                # skoru) engellemek icin mesafe 0'a kelepcelenir. Gercek veride K>0
                # oldugundan bu kelepce sonuc-notrdur.
                unseen_ceza += max(int(hedef_coz[3]), 0)
            gecisler.append({
                "kaynak_pattern": kelimeler[k - 1],
                "hedef_pattern": kelimeler[k],
                "etkin_kaynak": kaynak,
                "etkin_hedef": hedef,
                "gecis_olasiligi": float(p),
                "hedef_unseen": bool(hedef_coz[1]),
                "en_yakin_pattern": hedef_coz[2],
                "levenshtein_mesafe": int(hedef_coz[3]),
            })
        skor = -log_olasilik + self.unseen_ceza_agirligi * unseen_ceza   # yuksek skor = anomali
        return {
            "skor": float(skor),
            "log_olasilik": float(log_olasilik),
            "path_olasiligi": float(np.exp(log_olasilik)),
            "unseen_ceza": float(unseen_ceza),
            "gecisler": gecisler,
        }

    # ---- skorlama ----
    def skor(self, veri: ModelGirdisi) -> tuple[np.ndarray, np.ndarray]:
        if self.oto is None:
            raise RuntimeError("Model once egitilmeli (egit).")
        L = self.ham_pencere
        skorlar: list[float] = []
        konumlar: list[int] = []
        for bas, son in bitisik_bloklar(veri.segmentler):
            seri = self._normalize(veri.pc1[bas:son])
            kelimeler = self._segment_kelimeleri(seri)
            for t in range(self.path_uzunlugu, len(kelimeler)):
                skorlar.append(self._yol_bilgisi(kelimeler, t)["skor"])
                konumlar.append(bas + t + L - 1)
        return np.asarray(skorlar, dtype=float), np.asarray(konumlar, dtype=int)

    def unseen_konumlar(self, veri: ModelGirdisi) -> np.ndarray:
        """Mevcut pattern'i egitim sozlugunde bulunmayan test konumlari.

        ``unseen`` senaryosu, modelleri yalnizca bu novel-pattern noktalarinda
        degerlendirmek icin kullanilir.
        """
        if self.oto is None:
            raise RuntimeError("Model once egitilmeli (egit).")
        L = self.ham_pencere
        konumlar: list[int] = []
        for bas, son in bitisik_bloklar(veri.segmentler):
            seri = self._normalize(veri.pc1[bas:son])
            kelimeler = self._segment_kelimeleri(seri)
            for t in range(self.path_uzunlugu, len(kelimeler)):
                if kelimeler[t] not in self.oto.sozluk:
                    konumlar.append(bas + t + L - 1)
        return np.asarray(konumlar, dtype=int)

    def unseen_analizi(self, unseen: ModelGirdisi, orijinal: ModelGirdisi) -> dict:
        """VI.A sozluk-disi (out-of-vocabulary) yonetim metrikleri.

        ``unseen`` genlik kaydirilmis (covariate shift) girdi, ``orijinal`` ise
        ayni test noktalarinin kaydirilmamis halidir. Iki metrik dondurulur:

        - Detection Rate: degerlendirilebilir test konumlarindan kac tanesinin
          SAX pattern'i egitim sozlugunde bulunmadigi (sozluk-disi tespit orani).
        - Mapping Accuracy: her sozluk-disi pattern Levenshtein ile en yakin
          bilinen pattern'a eslenir; bu eslesmenin AYNI konumun kaydirma-oncesi
          (orijinal) pattern'ine olan dogrulugu. ``tam`` = eslenen pattern
          orijinal pattern'e birebir esit oran; ``yumusak`` = orijinal pattern'e
          Levenshtein mesafesi <= ``esik`` olan oran.

        Iki girdi ayni segment yapisina ve uzunluga sahip olmalidir (kaydirma
        yalnizca genligi degistirir, konumlari degil).
        """
        if self.oto is None:
            raise RuntimeError("Model once egitilmeli (egit).")
        esik = int(self.cfg.degerlendirme.unseen_eslesme_mesafe_esigi)
        toplam = 0
        sozluk_disi = 0
        tam_eslesme = 0
        yumusak_eslesme = 0
        u_bloklar = list(bitisik_bloklar(unseen.segmentler))
        o_bloklar = list(bitisik_bloklar(orijinal.segmentler))
        for (u_bas, u_son), (o_bas, o_son) in zip(u_bloklar, o_bloklar):
            u_kelimeler = self._segment_kelimeleri(self._normalize(unseen.pc1[u_bas:u_son]))
            o_kelimeler = self._segment_kelimeleri(self._normalize(orijinal.pc1[o_bas:o_son]))
            ust = min(len(u_kelimeler), len(o_kelimeler))
            for t in range(self.path_uzunlugu, ust):
                toplam += 1
                if u_kelimeler[t] in self.oto.sozluk:
                    continue
                sozluk_disi += 1
                en_yakin = self.oto.pattern_coz(u_kelimeler[t])[2]
                orijinal_pattern = o_kelimeler[t]
                if en_yakin == orijinal_pattern:
                    tam_eslesme += 1
                if levenshtein_mesafe(en_yakin, orijinal_pattern) <= esik:
                    yumusak_eslesme += 1
        detection_rate = (sozluk_disi / toplam) if toplam else 0.0
        tam_oran = (tam_eslesme / sozluk_disi) if sozluk_disi else 0.0
        yumusak_oran = (yumusak_eslesme / sozluk_disi) if sozluk_disi else 0.0
        return {
            "toplam_konum": int(toplam),
            "sozluk_disi_konum": int(sozluk_disi),
            "detection_rate": float(detection_rate),
            "mapping_accuracy_tam": float(tam_oran),
            "mapping_accuracy_yumusak": float(yumusak_oran),
        }
