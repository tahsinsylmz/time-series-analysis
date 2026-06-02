"""Aciklanabilirlik (OtomataAciklayici) birim testleri.

Proje gereksinimi (X.B / X.F): otomata kararlari izlenebilir bilesenlere ayrilir
ve hem makine-okur (JSON-uyumlu sozluk) hem standart bir spec formati uretilir.
Bu testler explainer.py cekirdegini korur: spec formatinin anahtarlari ve status
alani, guven skorunun [0,1] araliginda olmasi, kararin esik karsilastirmasiyla
tutarliligi, sozluk-disi (unseen) konumlarin Levenshtein eslemesi ve guven
fonksiyonunun sinir davranisi sinanir.

Kurulum test_unseen.py ile ayni cizgidedir: tekrar eden testere-disi bir sinyal
kucuk ama anlamli bir SAX sozlugu uretir; egitim tek-sinifli oldugundan esik
secimi bir RuntimeWarning yayar, bu uyari kurulum yardimcisinda bastirilir.
"""
import warnings

import numpy as np
import pytest

from src.explainability.explainer import OtomataAciklayici
from src.models.automata.automata_model import OtomataAnomaliModeli
from src.models.base import ModelGirdisi
from src.utils.config import konfig_yukle

# Standart spec ciktisi (Ister X.F) icin zorunlu anahtar kumesi
SPEC_ANAHTARLARI = {
    "time_step", "state", "pattern", "status", "mapped_to", "probability", "decision",
}


def _girdi(pc1: np.ndarray) -> ModelGirdisi:
    """Otomata icin tek segmentli, normal etiketli model girdisi kurar."""
    n = pc1.size
    return ModelGirdisi(
        X_olcekli=pc1.reshape(-1, 1),
        pc1=pc1.astype(float),
        y=np.zeros(n, dtype=int),
        segmentler=np.zeros(n, dtype=int),
    )


def _egitilmis_model() -> OtomataAnomaliModeli:
    """Kucuk bir SAX sozlugu ureten testere-disi sinyalle egitilmis model.

    Egitim tek-sinifli (hepsi normal) oldugundan esik secimi tek-sinif uyarisi
    verir; bu beklenen uyari burada bastirilir (testin konusu degil).
    """
    cfg = konfig_yukle()
    egitim_pc1 = np.tile([0.0, 1.0, 2.0, 3.0], 30)
    model = OtomataAnomaliModeli(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.egit(_girdi(egitim_pc1))
    return model


def _kaydirilmis_girdi() -> ModelGirdisi:
    """Egitim dagiliminin disina itilmis sinyal -> sozluk-disi (unseen) pattern'lar."""
    return _girdi(np.tile([0.0, 1.0, 2.0, 3.0], 30) * 6.0 + 25.0)


def _ilk_unseen_konum(model: OtomataAnomaliModeli, girdi: ModelGirdisi) -> int:
    """Karar veren (son) gecisi sozluk-disi olan bir konum dondurur."""
    konumlar = model.unseen_konumlar(girdi)
    assert konumlar.size > 0, "test kurulumu sozluk-disi konum uretmedi"
    return int(konumlar[0])


def test_spec_formati_anahtarlari_ve_status():
    # Ister X.F: standart cikti tam olarak beklenen anahtarlari tasimali; status
    # yalnizca {seen, unseen} alabilmeli.
    model = _egitilmis_model()
    girdi = _kaydirilmis_girdi()
    aciklayici = OtomataAciklayici(model)
    for kategori_acik in aciklayici.secili_ornekler(girdi, k_anomali=2):
        spec = kategori_acik["spec_formati"]
        assert set(spec.keys()) == SPEC_ANAHTARLARI
        assert spec["status"] in {"seen", "unseen"}
        assert spec["decision"] in {"anomaly", "normal"}


def test_guven_skoru_araligi():
    # Uretilen tum aciklamalarda guven skoru [0,1] araliginda olmali.
    model = _egitilmis_model()
    girdi = _kaydirilmis_girdi()
    aciklayici = OtomataAciklayici(model)
    ornekler = aciklayici.secili_ornekler(girdi, k_anomali=3)
    assert ornekler, "aciklama seti bos olmamali"
    for acik in ornekler:
        assert 0.0 <= acik["guven_skoru"] <= 1.0


def test_karar_esik_ve_metin_tutarli():
    # karar, anomali_skoru ile esik karsilastirmasinin aynisi olmali; karar_metni
    # ve spec decision alani karar ile birebir tutarli olmali.
    model = _egitilmis_model()
    girdi = _kaydirilmis_girdi()
    aciklayici = OtomataAciklayici(model)
    for acik in aciklayici.secili_ornekler(girdi, k_anomali=3):
        beklenen = int(acik["anomali_skoru"] >= model.esik)
        assert acik["karar"] == beklenen
        assert (acik["karar_metni"] == "ANOMALI") == (acik["karar"] == 1)
        assert (acik["spec_formati"]["decision"] == "anomaly") == (acik["karar"] == 1)
        assert acik["esik"] == pytest.approx(float(model.esik))


def test_unseen_konumda_levenshtein_eslemesi():
    # Sozluk-disi (unseen) bir konumda: status=='unseen', en yakin esleme bilinen
    # bir pattern'a (sozlukte) yapilmali ve mesafe gercek bir duzenleme mesafesi
    # (>=1) olmali.
    model = _egitilmis_model()
    girdi = _kaydirilmis_girdi()
    aciklayici = OtomataAciklayici(model)
    acik = aciklayici.acikla(girdi, _ilk_unseen_konum(model, girdi))

    assert acik["status"] == "unseen"
    assert acik["spec_formati"]["status"] == "unseen"
    assert acik["mapped_to"] in model.oto.sozluk
    # mapped_to gercekten sozluk-disi durumdan FARKLI bilinen bir pattern
    assert acik["mapped_to"] != acik["durum_sax"]
    # Karar veren (son) gecisin Levenshtein mesafesi gercek bir duzenleme mesafesi
    son_gecis = acik["gecisler"][-1]
    assert son_gecis["hedef_unseen"] is True
    assert son_gecis["levenshtein_mesafe"] >= 1


def test_guven_sinir_skor_esige_esit():
    # _guven sinir davranisi: skor tam esige esitse anomali olasiligi 0.5 ve karar
    # guveni 0 olmali (karar esikten ayrismaz).
    model = _egitilmis_model()
    aciklayici = OtomataAciklayici(model)
    anomali_olasiligi, guven = aciklayici._guven(float(model.esik))
    assert anomali_olasiligi == pytest.approx(0.5)
    assert guven == pytest.approx(0.0)


def test_unseen_mesafe_ozeti_tutarli():
    # unseen_mesafe_ozeti: dagilimin frekanslari toplami toplam unseen sayisina
    # esit olmali; ozet alanlari ic tutarli olmali.
    model = _egitilmis_model()
    girdi = _kaydirilmis_girdi()
    aciklayici = OtomataAciklayici(model)
    ozet = aciklayici.unseen_mesafe_ozeti(girdi)

    assert ozet["unseen_sayisi"] > 0
    assert sum(ozet["mesafe_dagilimi"].values()) == ozet["unseen_sayisi"]
    assert ozet["maks_mesafe"] == max(ozet["mesafe_dagilimi"].keys())
    assert min(ozet["mesafe_dagilimi"].keys()) >= 1
