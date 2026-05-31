"""Tum deneyleri calistiran giris noktasi.

Kullanim:
    python -m scripts.run_experiments            # tam calisma (5 fold x 5 seed + tarama)
    python -m scripts.run_experiments --hizli    # hizli duman testi (1 fold x 1 seed)
    python -m scripts.run_experiments --tarama-yok  # parametre taramasini atla

Sonuclar ``results/`` dizinine yazilir:
    olcumler.csv, ozet.csv, istatistik_testleri.json, parametre_taramasi_skab.csv
"""
from __future__ import annotations

import argparse
import time

from src.experiments.runner import DeneyYoneticisi
from src.utils.config import konfig_yukle


def main() -> None:
    ayristirici = argparse.ArgumentParser(description="Zaman serisi anomali deneyleri")
    ayristirici.add_argument("--hizli", action="store_true",
                             help="Hizli duman testi (1 fold, 1 seed)")
    ayristirici.add_argument("--tarama-yok", action="store_true",
                             help="Parametre duyarlilik taramasini atla")
    args = ayristirici.parse_args()

    cfg = konfig_yukle()
    yonetici = DeneyYoneticisi(cfg)

    t0 = time.time()
    yonetici.calistir(hizli=args.hizli, tarama=not args.tarama_yok)
    print(f"\nToplam sure: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
