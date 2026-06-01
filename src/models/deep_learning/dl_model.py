"""Derin ogrenme modeli (``AnomaliModeli`` arayuzunu uygular).

Tek bir sinif uc mimariyi (lstm/gru/cnn1d) ``mimari`` parametresiyle sarar.
Boylece deney yoneticisi otomata ile derin ogrenme modellerini ayni arayuz
uzerinden kullanir (Liskov yerine gecme ilkesi).

Girdi olarak cok degiskenli ``X_olcekli`` kullanilir (otomata yalnizca PC1
kullanir). Skor = pencere son adiminin sigmoid anomali olasiligi.
"""
from __future__ import annotations

import numpy as np

from src.models.base import AnomaliModeli, ModelGirdisi
from src.models.deep_learning.datasets import pencereler_olustur
from src.models.deep_learning.networks import ag_olustur
from src.models.deep_learning.trainer import egit_dongusu, olasilik_uret
from src.utils.esik import f1_maksimize_esik


class DerinOgrenmeModeli(AnomaliModeli):
    """LSTM / GRU / 1B-CNN tabanli black-box anomali tespit modeli."""

    def __init__(self, cfg, mimari: str) -> None:
        self.cfg = cfg
        self.ad = mimari.lower()
        dc = cfg.derin_ogrenme
        self.dizi = int(dc.dizi_uzunlugu)
        self.adim = int(dc.adim)
        self.gizli = int(dc.gizli_boyut)
        self.katman = int(dc.katman_sayisi)
        self.dropout = float(dc.dropout)
        self.cnn_kernel = int(dc.cnn_kernel)
        self.esik_aday_sayisi = int(cfg.degerlendirme.esik_aday_sayisi)
        self.tek_sinif_persentil = float(cfg.degerlendirme.tek_sinif_esik_persentili)
        self.ag = None
        self.esik = 0.5

    def _pencerele(self, veri: ModelGirdisi):
        return pencereler_olustur(veri.X_olcekli, veri.y, veri.segmentler, self.dizi, self.adim)

    def egit(self, egitim: ModelGirdisi, dogrulama: ModelGirdisi | None = None) -> "DerinOgrenmeModeli":
        e_pen, e_et, _ = self._pencerele(egitim)
        if dogrulama is not None:
            v_pen, v_et, _ = self._pencerele(dogrulama)
        else:
            v_pen, v_et = e_pen, e_et
        ozellik_sayisi = e_pen.shape[2]
        self.ag = ag_olustur(self.ad, ozellik_sayisi, self.gizli, self.katman, self.dropout,
                             self.cnn_kernel)
        self.ag = egit_dongusu(self.ag, e_pen, e_et, v_pen, v_et, self.cfg)

        # Karar esigi: dogrulama (yoksa egitim) olasiliklarinda F1 maksimize
        ref_pen, ref_et = (v_pen, v_et) if dogrulama is not None else (e_pen, e_et)
        ref_olasilik = olasilik_uret(self.ag, ref_pen)
        self.esik = f1_maksimize_esik(ref_olasilik, ref_et, self.esik_aday_sayisi,
                                      self.tek_sinif_persentil)
        return self

    def skor(self, veri: ModelGirdisi) -> tuple[np.ndarray, np.ndarray]:
        if self.ag is None:
            raise RuntimeError("Model once egitilmeli (egit).")
        pencereler, _, konumlar = self._pencerele(veri)
        skorlar = olasilik_uret(self.ag, pencereler)
        return skorlar, konumlar
