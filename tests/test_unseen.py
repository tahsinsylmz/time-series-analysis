"""Unseen (sozluk-disi) pattern tespiti ve Levenshtein eslemesi birim testleri.

Proje gereksinimi (VI / VI.A): egitim SAX sozlugu cikarilir; testte sozlukte
bulunmayan pattern'lar unseen kabul edilir ve Levenshtein ile en yakin bilinen
pattern'a eslenir. Bu mekanizmanin birim testlerle dogrulanmasi zorunludur.

Burada mekanizma MODEL seviyesinde (uctan uca) sinanir: ``unseen_konumlar``
gercekten sozluk-disi konumlari isaretliyor mu ve skorlama Levenshtein
eslemesini uyguluyor mu.
"""
import numpy as np

from src.models.automata.automata_model import OtomataAnomaliModeli
from src.models.base import ModelGirdisi
from src.utils.config import konfig_yukle


def _girdi(pc1: np.ndarray) -> ModelGirdisi:
    """Otomata icin tek segmentli, normal etiketli model girdisi kurar."""
    n = pc1.size
    return ModelGirdisi(
        X_olcekli=pc1.reshape(-1, 1),
        pc1=pc1.astype(float),
        y=np.zeros(n, dtype=int),
        segmentler=np.zeros(n, dtype=int),
    )


def _sozluk_disi_konumlar(model: OtomataAnomaliModeli, girdi: ModelGirdisi) -> list[int]:
    """unseen_konumlar'dan BAGIMSIZ olarak, sozluk-disi konumlari yeniden hesaplar."""
    L = model.ham_pencere
    seri = model._normalize(girdi.pc1)
    kelimeler = model._segment_kelimeleri(seri)
    return [t + L - 1 for t in range(model.path_uzunlugu, len(kelimeler))
            if kelimeler[t] not in model.oto.sozluk]


def _egitilmis_model() -> OtomataAnomaliModeli:
    cfg = konfig_yukle()
    # Tekrar eden testere disi sinyal -> kucuk ama anlamli bir SAX sozlugu
    egitim_pc1 = np.tile([0.0, 1.0, 2.0, 3.0], 30)
    model = OtomataAnomaliModeli(cfg)
    model.egit(_girdi(egitim_pc1))
    return model


def test_egitim_sozlugu_bos_degil():
    model = _egitilmis_model()
    assert len(model.oto.sozluk) >= 1


def test_unseen_konumlar_kontrati():
    # unseen_konumlar tam olarak sozluk-disi kelime tasiyan konumlari dondurmeli
    model = _egitilmis_model()
    kaydirilmis = np.tile([0.0, 1.0, 2.0, 3.0], 30) * 6.0 + 25.0  # dagilim disina it
    girdi = _girdi(kaydirilmis)
    beklenen = _sozluk_disi_konumlar(model, girdi)
    elde = model.unseen_konumlar(girdi).tolist()
    assert elde == beklenen


def test_kaydirma_unseen_tetikler():
    # Egitim dagiliminin disindaki sinyal en az bir novel pattern uretmeli
    model = _egitilmis_model()
    kaydirilmis = np.tile([0.0, 1.0, 2.0, 3.0], 30) * 6.0 + 25.0
    konumlar = model.unseen_konumlar(_girdi(kaydirilmis))
    assert konumlar.size > 0


def test_ayni_dagilim_az_veya_sifir_unseen():
    # Egitimle ayni sureci izleyen girdi sozluk-disi olmamali (sozlukten geliyor)
    model = _egitilmis_model()
    ayni = np.tile([0.0, 1.0, 2.0, 3.0], 20)
    beklenen = _sozluk_disi_konumlar(model, _girdi(ayni))
    assert model.unseen_konumlar(_girdi(ayni)).tolist() == beklenen
    assert len(beklenen) == 0


def test_unseen_pattern_levenshtein_ile_eslenir():
    # Novel bir hedef pattern, skorlama sirasinda sozlukteki en yakina eslenmeli
    model = _egitilmis_model()
    kaydirilmis = np.tile([0.0, 1.0, 2.0, 3.0], 30) * 6.0 + 25.0
    seri = model._normalize(kaydirilmis)
    kelimeler = model._segment_kelimeleri(seri)
    # En az bir sozluk-disi kelime icin pattern_coz Levenshtein eslemesi yapmali
    novel = [k for k in kelimeler if k not in model.oto.sozluk]
    assert novel, "test kurulumu novel pattern uretmedi"
    etkin, unseen_mi, en_yakin, mesafe = model.oto.pattern_coz(novel[0])
    assert unseen_mi is True
    assert en_yakin in model.oto.sozluk     # en yakin bilinen pattern sozlukte
    assert mesafe >= 1                        # gercek bir duzenleme mesafesi
