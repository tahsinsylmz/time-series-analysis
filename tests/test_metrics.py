"""Metrik ve karar esigi birim testleri."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.experiments.metrics import ikili_metrikler
from src.utils.esik import f1_maksimize_esik


def test_ikili_metrikler_elle_dogrulanmis():
    # TP=2, FP=1, FN=1, TN=1 olacak sekilde kuruldu
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    m = ikili_metrikler(y_true, y_pred)
    assert m["precision"] == pytest.approx(2 / 3)   # TP/(TP+FP)
    assert m["recall"] == pytest.approx(2 / 3)       # TP/(TP+FN)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(0.6)        # (TP+TN)/5
    assert m["n"] == 5 and m["pozitif"] == 3


def test_roc_auc_yonu():
    # skorlar pozitifte yuksek -> mukemmel ayrim
    y = np.array([0, 0, 1, 1])
    skor = np.array([0.1, 0.2, 0.8, 0.9])
    m = ikili_metrikler(y, (skor >= 0.5).astype(int), skor)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    # skorlar ters cevrilirse roc_auc = 0
    m2 = ikili_metrikler(y, (skor >= 0.5).astype(int), skor[::-1])
    assert m2["roc_auc"] == pytest.approx(0.0)


def test_tek_sinifta_auc_uretilmez():
    y = np.zeros(4, dtype=int)
    m = ikili_metrikler(y, np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]))
    assert "roc_auc" not in m and "pr_auc" not in m


def test_esik_ayrilabilir_veride_mukemmel_f1():
    skor = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    esik = f1_maksimize_esik(skor, y)
    tahmin = (skor >= esik).astype(int)
    # esik anomalileri normalden tam ayirmali
    assert tahmin.tolist() == y.tolist()


def test_esik_bos_girdi():
    assert f1_maksimize_esik(np.array([]), np.array([])) == 0.0


def test_esik_tek_sinif_persentil_fallback():
    # Tek sinifli (anomalisiz) val -> medyan degil, yuksek persentil (guvenli) esik
    skor = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.zeros(4, dtype=int)
    with pytest.warns(RuntimeWarning):
        esik = f1_maksimize_esik(skor, y, tek_sinif_persentil=0.95)
    assert esik == pytest.approx(np.quantile(skor, 0.95))
    # medyandan kesinlikle daha yuksek (daha az nokta isaretler)
    assert esik > np.median(skor)


def test_esik_aday_sayisi_ile_ornekleme_ayristirir():
    # aday_sayisi'ndan COK benzersiz skor -> nicelik (quantile) ile ornekleme yapilir;
    # yine de ayrilabilir veride esik siniflari ANLAMLI ayirir (tautoloji degil).
    rng = np.random.default_rng(0)
    normal = rng.uniform(0.0, 0.4, size=500)
    anomali = rng.uniform(0.6, 1.0, size=500)
    skor = np.concatenate([normal, anomali])
    y = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    # benzersiz skor sayisi aday_sayisi'ndan cok daha fazla -> ornekleme devrede
    assert np.unique(skor).size > 50

    esik = f1_maksimize_esik(skor, y, aday_sayisi=50)
    tahmin = (skor >= esik).astype(int)
    assert f1_score(y, tahmin) > 0.9
    # esik gercekten ayristirir: normallerin cogu altinda, anomalilerin cogu ustunde
    assert (skor[y == 0] < esik).mean() > 0.9
    assert (skor[y == 1] >= esik).mean() > 0.9
