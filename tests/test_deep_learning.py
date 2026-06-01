"""Derin ogrenme bilesenleri icin birim testler.

Kapsam: kayan pencere uretimi (sekil, etiket, segment siniri), pozitif sinif
agirligi, olasilik ciktisi araligi ve egitim dongusunde kaybin dusmesi.
"""
import numpy as np
import torch

from src.models.deep_learning.datasets import pencereler_olustur
from src.models.deep_learning.networks import ag_olustur
from src.models.deep_learning.trainer import _pos_weight, egit_dongusu, olasilik_uret
from src.utils.config import konfig_yukle
from src.utils.seeding import seed_ayarla


def test_pencere_sekil_ve_etiket():
    # 10 zaman adimi, 2 ozellik, tek segment
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=int)
    seg = np.zeros(10, dtype=int)
    L, adim = 3, 1
    pen, et, konum = pencereler_olustur(X, y, seg, L, adim)
    # M = 10 - 3 + 1 = 8 pencere
    assert pen.shape == (8, L, 2)
    # etiket = pencerenin SON adiminin etiketi; konum = son adim indeksi
    assert konum.tolist() == [2, 3, 4, 5, 6, 7, 8, 9]
    assert et.tolist() == y[konum].tolist()


def test_pencere_adim_ile_kayar():
    X = np.zeros((12, 1), dtype=np.float32)
    y = np.zeros(12, dtype=int)
    seg = np.zeros(12, dtype=int)
    pen, _, konum = pencereler_olustur(X, y, seg, dizi_uzunlugu=4, adim=3)
    # t = 0,3,6,8(son) -> baslangic 0,3,6; bitis-1 = 3,6,9
    assert konum.tolist() == [3, 6, 9]
    assert pen.shape[0] == 3


def test_pencere_segment_sinirini_asmaz():
    # iki segment (orn. iki SKAB dosyasi); pencere sinir asmamali
    X = np.zeros((10, 1), dtype=np.float32)
    y = np.zeros(10, dtype=int)
    seg = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    pen, _, konum = pencereler_olustur(X, y, seg, dizi_uzunlugu=3, adim=1)
    # Her segmentte (5 - 3 + 1) = 3 pencere -> toplam 6
    assert pen.shape[0] == 6
    # Hicbir pencere [4,5] sinirini asmaz: konumlar ilk segmentte <=4, ikincide >=7
    assert set(konum.tolist()) == {2, 3, 4, 7, 8, 9}


def test_pencere_bos_girdi():
    X = np.zeros((2, 3), dtype=np.float32)
    y = np.zeros(2, dtype=int)
    seg = np.zeros(2, dtype=int)
    pen, et, konum = pencereler_olustur(X, y, seg, dizi_uzunlugu=5, adim=1)
    assert pen.shape == (0, 5, 3)
    assert et.shape == (0,) and konum.shape == (0,)


def test_pos_weight():
    # dengeli -> 1.0
    assert _pos_weight(np.array([0, 1, 0, 1])) == 1.0
    # dengesiz: 9 negatif / 1 pozitif -> 9.0
    assert _pos_weight(np.array([0] * 9 + [1])) == 9.0
    # hic pozitif yok -> 1.0 (sifira bolme korumasi)
    assert _pos_weight(np.zeros(5, dtype=int)) == 1.0


def test_olasilik_araligi_ve_uzunluk():
    seed_ayarla(0)
    ag = ag_olustur("lstm", ozellik_sayisi=2, gizli_boyut=4, katman_sayisi=1, dropout=0.0)
    pen = np.random.randn(7, 5, 2).astype(np.float32)
    olas = olasilik_uret(ag, pen)
    assert olas.shape == (7,)
    assert np.all((olas >= 0.0) & (olas <= 1.0))
    # bos girdi -> bos cikti
    assert olasilik_uret(ag, np.empty((0, 5, 2), dtype=np.float32)).shape == (0,)


def test_egit_dongusu_kaybi_dusurur():
    seed_ayarla(0)
    cfg = konfig_yukle()
    cfg.derin_ogrenme.epoch = 30           # test icin kisa egitim
    cfg.derin_ogrenme.early_stopping_sabri = 30

    # Ayrilabilir oyuncak veri: pozitif pencereler yuksek ortalamali
    n, L, F = 64, 5, 2
    X = np.random.randn(n, L, F).astype(np.float32)
    et = (np.arange(n) % 2).astype(np.int64)
    X[et == 1] += 2.0                       # sinifi ayristir

    kayip_fn = torch.nn.BCEWithLogitsLoss()

    def egitim_kaybi(ag):
        ag.eval()
        with torch.no_grad():
            logit = ag(torch.from_numpy(X))
            return float(kayip_fn(logit, torch.from_numpy(et.astype(np.float32))))

    seed_ayarla(0)
    ag = ag_olustur("lstm", ozellik_sayisi=F, gizli_boyut=8, katman_sayisi=1, dropout=0.0)
    once = egitim_kaybi(ag)
    ag = egit_dongusu(ag, X, et, X, et, cfg)
    sonra = egitim_kaybi(ag)
    assert sonra < once
