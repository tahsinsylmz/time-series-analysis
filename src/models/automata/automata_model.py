"""Otomata tabanli anomali tespit modeli.

PC1 (tek boyut) serisi uzerinde:
1. Egitim istatistikleriyle z-normalize (sizinti onleme)
2. Sliding window -> PAA -> SAX ile pattern (state) dizisi
3. NORMAL egitim verisinden frekans tabanli gecis olasiliklari
4. Test noktasi icin yol olasiligi dusukse anomali (dusuk olasilik = anomali)

Unseen pattern'lar Levenshtein ile en yakin bilinen pattern'a eslenir ve ek
anomali cezasi alir.
"""
from __future__ import annotations

import numpy as np

from src.models.automata.automaton import OlasiliksalOtomata
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
        self.esik = f1_maksimize_esik(skorlar, referans.y[konumlar])
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
                unseen_ceza += hedef_coz[3]
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
        skor = -log_olasilik + unseen_ceza   # yuksek skor = anomali
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
