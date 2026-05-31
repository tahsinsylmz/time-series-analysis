"""Merkezi konfigurasyon yukleyici.

config/config.yaml dosyasini okuyup hem ham sozluk hem de nokta ile
erisilebilen (cfg.otomata.window_size gibi) bir nesne olarak sunar.
Boylece tum moduller parametreleri tek bir kaynaktan alir.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import yaml

# Proje kok dizini (bu dosya: <kok>/src/utils/config.py)
PROJE_KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _namespace_yap(deger: Any) -> Any:
    """Ic ice sozlukleri nokta erisimli SimpleNamespace'e cevirir."""
    if isinstance(deger, dict):
        return SimpleNamespace(**{anahtar: _namespace_yap(alt) for anahtar, alt in deger.items()})
    if isinstance(deger, list):
        return [_namespace_yap(eleman) for eleman in deger]
    return deger


class Konfig:
    """Konfigurasyon sarmalayicisi: hem .ham (dict) hem nokta erisimi sunar."""

    def __init__(self, ham: dict) -> None:
        object.__setattr__(self, "ham", ham)
        object.__setattr__(self, "_ns", _namespace_yap(ham))

    def __getattr__(self, ad: str) -> Any:
        return getattr(object.__getattribute__(self, "_ns"), ad)

    def yol(self, *parcalar: str) -> str:
        """Proje koküne gore mutlak yol uretir."""
        return os.path.join(PROJE_KOK, *parcalar)


def konfig_yukle(yol: str | None = None) -> Konfig:
    """config.yaml dosyasini yukler ve Konfig nesnesi dondurur."""
    if yol is None:
        yol = os.path.join(PROJE_KOK, "config", "config.yaml")
    with open(yol, "r", encoding="utf-8") as dosya:
        ham = yaml.safe_load(dosya)
    return Konfig(ham)
