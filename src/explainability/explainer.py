"""Otomata kararlarinin aciklanabilirligi (black-box -> white-box).

Derin ogrenme modelleri bir kararin NEDEN verildigini gizler. Olasiliksal
otomata ise her karari izlenebilir bilesenlere ayirir:

  - mevcut DURUM (SAX pattern) ve ona goturen ham pencere/PAA degerleri,
  - son ``path_uzunlugu`` GECIS ve her birinin olasiligi,
  - yol (path) olasiligi ve -log olasilik katkisi,
  - egitimde GORULMEMIS (unseen) pattern'lar, en yakin esleri ve Levenshtein
    mesafesi (eklenen ceza),
  - nihai anomali skoru, karar esigi, karar ve bir guven skoru.

Cikti hem makine-okur (JSON) hem insan-okur (Turkce metin) olarak uretilir.
"""
from __future__ import annotations

import numpy as np

from src.models.automata.automata_model import OtomataAnomaliModeli
from src.models.automata.paa import paa_segment
from src.models.automata.sax import sax_sembolize
from src.models.base import ModelGirdisi
from src.utils.segments import bitisik_bloklar


class OtomataAciklayici:
    """Egitilmis bir otomata modelinin kararlarini aciklayan sinif."""

    def __init__(self, model: OtomataAnomaliModeli) -> None:
        if model.oto is None:
            raise RuntimeError("Model once egitilmeli (egit).")
        self.model = model

    # ---- konum -> segment/pencere cozumu ----
    def _konum_coz(self, veri: ModelGirdisi, hedef_konum: int):
        L = self.model.ham_pencere
        for bas, son in bitisik_bloklar(veri.segmentler):
            if bas <= hedef_konum < son:
                yerel_bitis = hedef_konum - bas
                t = yerel_bitis - (L - 1)
                if t < self.model.path_uzunlugu:
                    raise ValueError("Bu konum aciklanamaz (yeterli gecmis pencere yok).")
                seri = self.model._normalize(veri.pc1[bas:son])
                kelimeler = self.model._segment_kelimeleri(seri)
                if t >= len(kelimeler):
                    raise ValueError("Konum segment sinirlari disinda.")
                return bas, t, kelimeler, seri, yerel_bitis
        raise ValueError("Konum hicbir segmentte bulunamadi.")

    def _guven(self, skor: float) -> tuple[float, float]:
        """(anomali_olasiligi, karar_guveni) dondurur.

        Skor, esige gore standardize edilip sigmoidden gecirilir. Karar guveni
        kararin esikten ne kadar kesin ayrildigini [0,1] araliginda olcer.
        """
        z = (skor - self.model.esik) / self.model.ref_skor_sd
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p), float(abs(2.0 * p - 1.0))

    def _metin(self, durum, bilgi, skor, karar, guven) -> str:
        unseen_gecisler = [g for g in bilgi["gecisler"] if g["hedef_unseen"]]
        parcalar = [
            f"Pencere sonu durumu '{durum}'.",
            f"Son {len(bilgi['gecisler'])} gecisin yol olasiligi "
            f"{bilgi['path_olasiligi']:.3e} (dusuk olasilik = beklenmedik gidisat).",
        ]
        if unseen_gecisler:
            g = unseen_gecisler[0]
            parcalar.append(
                f"{len(unseen_gecisler)} gecis egitimde gorulmemis pattern iceriyor; "
                f"ornegin '{g['hedef_pattern']}' en yakin bilinen '{g['en_yakin_pattern']}' "
                f"pattern'ine Levenshtein={g['levenshtein_mesafe']} uzakliginda, "
                f"toplam +{bilgi['unseen_ceza']:.1f} ceza eklendi."
            )
        karar_metni = "ANOMALI" if karar else "NORMAL"
        iliski = "astigi" if karar else "altinda kaldigi"
        parcalar.append(
            f"Toplam anomali skoru {skor:.2f}, esik ({self.model.esik:.2f}) degerini "
            f"{iliski} icin karar: {karar_metni} (guven {guven:.2f})."
        )
        return " ".join(parcalar)

    # ---- tekil aciklama ----
    def acikla(self, veri: ModelGirdisi, hedef_konum: int) -> dict:
        """Verilen konumdaki karari ayrintili (JSON-uyumlu) bir sozluge cevirir."""
        bas, t, kelimeler, seri, yerel_bitis = self._konum_coz(veri, hedef_konum)
        L = self.model.ham_pencere
        ham_pencere = seri[yerel_bitis - L + 1: yerel_bitis + 1]
        paa = paa_segment(ham_pencere, self.model.w)
        durum = sax_sembolize(paa, self.model.kesimler)   # == kelimeler[t]

        bilgi = self.model._yol_bilgisi(kelimeler, t)
        skor = bilgi["skor"]
        karar = int(skor >= self.model.esik)
        anomali_olasiligi, guven = self._guven(skor)

        # Ister X.F: zorunlu standart cikti formati (son gecis uzerinden)
        son_gecis = bilgi["gecisler"][-1]
        spec_formati = {
            "time_step": int(hedef_konum),
            "state": son_gecis["kaynak_pattern"],          # onceki durum
            "pattern": son_gecis["hedef_pattern"],         # gelen oruntu
            "status": "unseen" if son_gecis["hedef_unseen"] else "seen",
            "mapped_to": (son_gecis["en_yakin_pattern"] if son_gecis["hedef_unseen"]
                          else son_gecis["hedef_pattern"]),
            "probability": round(bilgi["path_olasiligi"], 6),
            "decision": "anomaly" if karar else "normal",
        }
        # Ister X.F alanlari ust seviyede de erisilebilir olsun
        gercek = int(veri.y[hedef_konum])

        return {
            "konum": int(hedef_konum),
            "ham_pencere_normalize": [round(float(x), 4) for x in ham_pencere],
            "paa": [round(float(x), 4) for x in paa],
            "durum_sax": durum,
            "status": spec_formati["status"],
            "mapped_to": spec_formati["mapped_to"],
            "yol": [bilgi["gecisler"][0]["kaynak_pattern"]]
                   + [g["hedef_pattern"] for g in bilgi["gecisler"]],
            "gecisler": bilgi["gecisler"],
            "path_olasiligi": bilgi["path_olasiligi"],
            "log_olasilik": bilgi["log_olasilik"],
            "unseen_ceza": bilgi["unseen_ceza"],
            "anomali_skoru": skor,
            "esik": float(self.model.esik),
            "karar": karar,
            "karar_metni": "ANOMALI" if karar else "NORMAL",
            "anomali_olasiligi": round(anomali_olasiligi, 4),
            "guven_skoru": round(guven, 4),
            "spec_formati": spec_formati,
            # Yer-gercegi aciklamanin PARCASI degildir; ayri teshis alaninda tutulur
            # (aciklama yalnizca modelin ic hesaplamalarina dayanir).
            "teshis": {"gercek_etiket": gercek, "karar_dogru": bool(karar == gercek)},
            "aciklama_metni": self._metin(durum, bilgi, skor, karar, guven),
        }

    # ---- toplu: en anomalik noktalar ----
    def en_anomalileri_acikla(self, veri: ModelGirdisi, k: int = 3) -> list[dict]:
        """En yuksek skorlu k noktayi aciklar (demo/gorsel icin)."""
        skorlar, konumlar = self.model.skor(veri)
        if skorlar.size == 0:
            return []
        sira = np.argsort(skorlar)[::-1][:k]
        return [self.acikla(veri, int(konumlar[i])) for i in sira]

    # ---- toplu: temsili (cesitli) ornek seti ----
    def secili_ornekler(self, veri: ModelGirdisi, k_anomali: int = 3) -> list[dict]:
        """Demo icin CESITLI bir ornek seti aciklar.

        Yalnizca en yuksek skorlu (anomali tahmini) noktalar degil; aciklamanin
        her iki yonu de gorulebilsin diye en az bir DOGRU-POZITIF (gercek=1 ve
        model anomali dedi) ve bir GUVENLI NORMAL (gercek=0 ve model normal dedi)
        ornek de eklenir. Her aciklamaya ``kategori`` etiketi konur.
        """
        skorlar, konumlar = self.model.skor(veri)
        if skorlar.size == 0:
            return []
        y = veri.y[konumlar]
        karar = (skorlar >= self.model.esik).astype(int)

        secili: list[tuple[int, str]] = []   # (dizi_indeksi, kategori)
        kullanilan: set[int] = set()

        def ekle(idx: int, kategori: str) -> None:
            if idx >= 0 and idx not in kullanilan:
                secili.append((idx, kategori))
                kullanilan.add(idx)

        # 1) En yuksek skorlu k_anomali nokta
        for i in np.argsort(skorlar)[::-1][:k_anomali]:
            ekle(int(i), "en_anomalik")
        # 2) En guclu dogru-pozitif (gercek=1 ve anomali kararli, en yuksek skor)
        tp = np.where((y == 1) & (karar == 1))[0]
        if tp.size:
            ekle(int(tp[np.argmax(skorlar[tp])]), "dogru_pozitif")
        # 3) Dusuk skorlu (normal kategorisi): once guvenli normal (gercek=0 & karar=0),
        #    yoksa (esik cok dusuk olabilir) genel olarak EN DUSUK skorlu nokta secilir.
        tn = np.where((y == 0) & (karar == 0))[0]
        if tn.size:
            ekle(int(tn[np.argmin(skorlar[tn])]), "normal")
        else:
            ekle(int(np.argmin(skorlar)), "en_dusuk_skor")

        sonuc = []
        for idx, kategori in secili:
            ack = self.acikla(veri, int(konumlar[idx]))
            ack["kategori"] = kategori
            sonuc.append(ack)
        return sonuc
