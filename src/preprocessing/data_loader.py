"""Veri yukleme katmani.

SKAB ve BATADAL veri setlerini izole bir sekilde yukleyip ortak bir
``HamVeri`` yapisina donusturur. Model girdisi olan sutunlar ile yalnizca
takip/analiz amacli meta sutunlar (datetime, source_file vb.) ayrilir.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import PROJE_KOK


@dataclass
class HamVeri:
    """Bir veri setinin standartlastirilmis gosterimi."""

    ad: str
    df: pd.DataFrame
    ozellik_adlari: list[str]
    hedef_sutun: str
    grup_sutunu: str | None = None
    zaman_sutunu: str | None = None

    @property
    def X(self) -> np.ndarray:
        return self.df[self.ozellik_adlari].to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray:
        return self.df[self.hedef_sutun].to_numpy(dtype=int)

    @property
    def gruplar(self) -> np.ndarray | None:
        if self.grup_sutunu is None:
            return None
        return self.df[self.grup_sutunu].to_numpy()


def _eksikleri_doldur(df: pd.DataFrame, sutunlar: list[str]) -> pd.DataFrame:
    """Sensor sutunlarindaki eksik degerleri dogrusal interpolasyon ile doldurur."""
    df = df.copy()
    df[sutunlar] = (
        df[sutunlar].interpolate(method="linear", limit_direction="both").bfill().ffill()
    )
    return df


def skab_yukle(cfg) -> HamVeri:
    """SKAB veri setini (valve1 + valve2) birlestirerek yukler."""
    konf = cfg.veri_setleri.skab
    kok = os.path.join(PROJE_KOK, konf.kok_dizin)
    parcalar: list[pd.DataFrame] = []
    for klasor in konf.klasorler:
        desen = os.path.join(kok, klasor, "*.csv")
        for yol in sorted(glob.glob(desen), key=lambda p: int(os.path.splitext(os.path.basename(p))[0])):
            alt = pd.read_csv(yol, sep=konf.ayrac)
            # Ek takip sutunlari (model girdisine girmez):
            alt["source_group"] = klasor
            alt["source_file"] = f"{klasor}/{os.path.basename(yol)}"
            parcalar.append(alt)
    if not parcalar:
        raise FileNotFoundError(f"SKAB verisi bulunamadi: {kok}")
    df = pd.concat(parcalar, ignore_index=True)
    df[konf.hedef_sutun] = df[konf.hedef_sutun].astype(int)
    ozellikler = [s for s in df.columns if s not in konf.haric_sutunlar]
    df = _eksikleri_doldur(df, ozellikler)
    return HamVeri(konf.ad, df, ozellikler, konf.hedef_sutun, konf.grup_sutunu, konf.zaman_sutunu)


def batadal_yukle(cfg) -> HamVeri:
    """BATADAL Training Dataset 2 dosyasini yukler ve ikili hedef uretir."""
    konf = cfg.veri_setleri.batadal
    yol = os.path.join(PROJE_KOK, konf.dosya)
    df = pd.read_csv(yol, skipinitialspace=True)
    df.columns = [s.strip() for s in df.columns]  # basliklarda bosluk temizligi
    # ATT_FLAG == 1 -> anomali (1), diger degerler (-999) -> normal (0)
    df[konf.hedef_sutun] = (df[konf.hedef_sutun] == konf.pozitif_deger).astype(int)
    ozellikler = [s for s in df.columns if s not in konf.haric_sutunlar]
    df = _eksikleri_doldur(df, ozellikler)
    return HamVeri(konf.ad, df, ozellikler, konf.hedef_sutun, None, konf.zaman_sutunu)


def veri_yukle(cfg, ad: str) -> HamVeri:
    """Veri seti adina gore uygun yukleyiciyi cagirir."""
    ad = ad.lower()
    if ad == "skab":
        return skab_yukle(cfg)
    if ad == "batadal":
        return batadal_yukle(cfg)
    raise ValueError(f"Bilinmeyen veri seti: {ad}")
