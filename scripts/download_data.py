"""Veri seti varligini dogrular ve ozetler.

Bu proje iki veri seti kullanir:
  - SKAB   : https://github.com/waico/SKAB  (valve1/, valve2/ altinda CSV'ler)
  - BATADAL: https://www.batadal.net/data.html  (BATADAL_dataset04.csv)

Veriler ``data/`` altinda hazir gelir. Bu betik dosyalarin yerinde oldugunu
dogrular, satir/sutun ve anomali oranlarini ekrana yazar. Eksik dosya varsa
nereden indirilecegini bildirir (hata firlatmadan).

Kullanim:
    python -m scripts.download_data
"""
from __future__ import annotations

import os

from src.preprocessing.data_loader import veri_yukle
from src.utils.config import PROJE_KOK, konfig_yukle

KAYNAKLAR = {
    "skab": "https://github.com/waico/SKAB",
    "batadal": "https://www.batadal.net/data.html",
}


def _ozet_yaz(ad: str, cfg) -> bool:
    try:
        ham = veri_yukle(cfg, ad)
    except FileNotFoundError as hata:
        print(f"[EKSIK] {ad.upper()} bulunamadi: {hata}")
        print(f"        Indirme adresi: {KAYNAKLAR[ad]}")
        return False
    y = ham.y
    pozitif = int(y.sum())
    oran = pozitif / len(y) if len(y) else 0.0
    print(f"[TAMAM] {ham.ad}")
    print(f"        satir={len(ham.df)}  ozellik={len(ham.ozellik_adlari)}  "
          f"anomali={pozitif} ({oran:.1%})")
    if ham.grup_sutunu:
        print(f"        grup sayisi ({ham.grup_sutunu})={ham.df[ham.grup_sutunu].nunique()}")
    return True


def main() -> None:
    cfg = konfig_yukle()
    print(f"Proje kok: {PROJE_KOK}")
    print(f"Veri dizini: {os.path.join(PROJE_KOK, 'data')}\n")
    tum_tamam = True
    for ad in ("skab", "batadal"):
        tum_tamam &= _ozet_yaz(ad, cfg)
        print()
    if tum_tamam:
        print("Tum veri setleri hazir.")
    else:
        print("Bazi veri setleri eksik; yukaridaki adreslerden indirip 'data/' altina koyun.")


if __name__ == "__main__":
    main()
