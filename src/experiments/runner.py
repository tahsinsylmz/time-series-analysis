"""Deney yoneticisi: tum modelleri, senaryolari ve istatistik testlerini calistirir.

Tasarim (SOLID): yonetici somut modellere degil ``AnomaliModeli`` arayuzune
bagimlidir. Yeni bir model eklemek icin yalnizca ``_model_olustur`` genisletilir.

Akis:
  SKAB    -> StratifiedGroupKFold (fold_sayisi), her fold icin tum modeller
  BATADAL -> zaman sirali tek bolme, her seed icin derin ogrenme modelleri
  Her egitilmis model 3 senaryoda (orijinal/gurultu/unseen) degerlendirilir.
  Otomata deterministik oldugundan fold basina bir kez egitilir.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from src.experiments.metrics import ikili_metrikler
from src.experiments.scenarios import gurultulu_girdi, kaydirilmis_girdi, temiz_girdi
from src.experiments.stats_tests import hizala, mcnemar_testi, wilcoxon_imzali
from src.models.automata.automata_model import OtomataAnomaliModeli
from src.models.deep_learning.dl_model import DerinOgrenmeModeli
from src.preprocessing.data_loader import veri_yukle
from src.preprocessing.preprocess import OnIslemci
from src.preprocessing.splitting import (
    grup_train_val_bol,
    skab_foldlar,
    zaman_sirali_bol,
)
from src.utils.config import PROJE_KOK
from src.utils.seeding import seed_ayarla


class DeneyYoneticisi:
    """Iki veri seti uzerinde tum karsilastirmali deneyleri yurutur."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.seeds = list(cfg.genel.rastgele_seedler)
        self.senaryolar = list(cfg.senaryolar.liste)
        self.gurultu_std = float(cfg.senaryolar.gurultu.std)
        self.gurultu_tohum_skab = int(cfg.senaryolar.gurultu.tohum_skab)
        self.gurultu_tohum_batadal = int(cfg.senaryolar.gurultu.tohum_batadal)
        self.unseen_faktor = float(cfg.senaryolar.unseen.olcek_faktoru)
        self.dl_modeller = list(cfg.derin_ogrenme.modeller)
        self.cikti = os.path.join(PROJE_KOK, cfg.genel.cikti_dizini)
        os.makedirs(self.cikti, exist_ok=True)

    # ---- model fabrikasi (acik/kapali ilkesi) ----
    def _model_olustur(self, ad: str):
        if ad == "automata":
            return OtomataAnomaliModeli(self.cfg)
        return DerinOgrenmeModeli(self.cfg, ad)

    # ---- tek bir degerlendirme (opsiyonel konum maskesi) ----
    def _degerlendir(self, model, girdi, izin: np.ndarray | None = None):
        skorlar, konumlar = model.skor(girdi)
        tahmin = (skorlar >= model.esik).astype(int)
        if izin is not None:
            maske = np.isin(konumlar, izin) if len(izin) else np.zeros(len(konumlar), bool)
            skorlar, konumlar, tahmin = skorlar[maske], konumlar[maske], tahmin[maske]
        y = girdi.y[konumlar]
        return ikili_metrikler(y, tahmin, skorlar), konumlar, tahmin, y

    def _senaryolar(self, model, ad, veri_seti, fold, seed, g_test, g_gurultu, g_unseen):
        """Bir egitilmis model icin 3 senaryonun kayitlarini ve orijinal tahminleri uretir.

        unseen senaryosu, genlik kaydirilmis test setinin TAMAMI uzerinde degerlendirilir
        (covariate shift); olcekleme egitimde gorulmemis pattern'leri tetikler.
        """
        kayitlar = []
        eslesme = None
        for senaryo in self.senaryolar:
            if senaryo == "orijinal":
                m, konum, tahmin, y = self._degerlendir(model, g_test)
                eslesme = (konum, tahmin, y)
            elif senaryo == "gurultu":
                m, _, _, _ = self._degerlendir(model, g_gurultu)
            elif senaryo == "unseen":
                m, _, _, _ = self._degerlendir(model, g_unseen)
            else:
                continue
            kayitlar.append({
                "veri_seti": veri_seti, "model": ad, "senaryo": senaryo,
                "fold": fold, "seed": seed, **m,
            })
        return kayitlar, eslesme

    # ---- SKAB: dosya bazli capraz dogrulama ----
    def skab_calistir(self, fold_limit: int | None = None, seed_limit: int | None = None):
        cfg = self.cfg
        sk = cfg.veri_setleri.skab
        ham = veri_yukle(cfg, "skab")
        seeds = self.seeds[:seed_limit] if seed_limit else self.seeds
        kayitlar: list[dict] = []
        mcnemar_birikim = {"y": [], "auto": [], "dl": []}
        ref_dl = self.dl_modeller[-1]   # McNemar referansi (genelde cnn1d)

        foldlar = list(skab_foldlar(ham, sk.fold_sayisi, sk.fold_tohumu))
        if fold_limit:
            foldlar = foldlar[:fold_limit]

        for fold_i, (tr_idx, test_idx) in enumerate(foldlar):
            t0 = time.time()
            ytr, val = grup_train_val_bol(tr_idx, ham.gruplar, sk.dogrulama_orani, sk.fold_tohumu)
            on = OnIslemci(cfg).fit(ham.X[ytr])
            g_tr = temiz_girdi(on, ham.X[ytr], ham.y[ytr], ham.gruplar[ytr])
            g_val = temiz_girdi(on, ham.X[val], ham.y[val], ham.gruplar[val])
            g_test = temiz_girdi(on, ham.X[test_idx], ham.y[test_idx], ham.gruplar[test_idx])
            rng = np.random.default_rng(self.gurultu_tohum_skab + fold_i)
            g_gurultu = gurultulu_girdi(on, ham.X[test_idx], ham.y[test_idx], ham.gruplar[test_idx],
                                        self.gurultu_std, rng)
            g_unseen = kaydirilmis_girdi(on, ham.X[test_idx], ham.y[test_idx], ham.gruplar[test_idx],
                                         self.unseen_faktor)

            # Otomata (deterministik): fold basina bir kez
            seed_ayarla(seeds[0])
            auto = self._model_olustur("automata").egit(g_tr, g_val)
            novelty = len(auto.unseen_konumlar(g_unseen))   # raporlama icin novelty sayisi
            a_kayit, a_eslesme = self._senaryolar(auto, "automata", "SKAB", fold_i, -1,
                                                  g_test, g_gurultu, g_unseen)
            kayitlar.extend(a_kayit)

            # Derin ogrenme: her seed icin
            dl_ref_eslesme = None
            for seed in seeds:
                for mimari in self.dl_modeller:
                    seed_ayarla(seed)
                    model = self._model_olustur(mimari).egit(g_tr, g_val)
                    d_kayit, d_eslesme = self._senaryolar(model, mimari, "SKAB", fold_i, seed,
                                                          g_test, g_gurultu, g_unseen)
                    kayitlar.extend(d_kayit)
                    if mimari == ref_dl and seed == seeds[0]:
                        dl_ref_eslesme = d_eslesme

            # McNemar verisi (otomata vs referans DL, ayni fold test noktalari)
            if a_eslesme is not None and dl_ref_eslesme is not None:
                yt, pa, pb = hizala(a_eslesme[0], a_eslesme[1],
                                    dl_ref_eslesme[0], dl_ref_eslesme[1],
                                    a_eslesme[0], a_eslesme[2])
                mcnemar_birikim["y"].append(yt)
                mcnemar_birikim["auto"].append(pa)
                mcnemar_birikim["dl"].append(pb)
            print(f"  [SKAB] fold {fold_i+1}/{len(foldlar)} bitti "
                  f"({time.time()-t0:.1f}s, otomata durum={auto.oto.K}, novelty={novelty})")

        mcnemar = self._mcnemar_hesapla(mcnemar_birikim, "SKAB", "automata", ref_dl)
        return kayitlar, mcnemar

    # ---- BATADAL: zaman sirali tek bolme ----
    def batadal_calistir(self, seed_limit: int | None = None):
        cfg = self.cfg
        bc = cfg.veri_setleri.batadal
        ham = veri_yukle(cfg, "batadal")
        seeds = self.seeds[:seed_limit] if seed_limit else self.seeds
        n = ham.X.shape[0]
        tr, val, test = zaman_sirali_bol(n, bc.egitim_orani, bc.dogrulama_orani)
        seg = np.zeros(n, dtype=int)
        on = OnIslemci(cfg).fit(ham.X[tr])

        def g(idx):
            return temiz_girdi(on, ham.X[idx], ham.y[idx], seg[idx])

        g_tr, g_val, g_test = g(tr), g(val), g(test)
        rng = np.random.default_rng(self.gurultu_tohum_batadal)
        g_gurultu = gurultulu_girdi(on, ham.X[test], ham.y[test], seg[test], self.gurultu_std, rng)
        g_unseen = kaydirilmis_girdi(on, ham.X[test], ham.y[test], seg[test], self.unseen_faktor)

        kayitlar: list[dict] = []
        mcnemar_birikim = {"y": [], "auto": [], "dl": []}
        ref_dl = self.dl_modeller[-1]

        seed_ayarla(seeds[0])
        auto = self._model_olustur("automata").egit(g_tr, g_val)
        novelty = len(auto.unseen_konumlar(g_unseen))
        a_kayit, a_eslesme = self._senaryolar(auto, "automata", "BATADAL", 0, -1,
                                              g_test, g_gurultu, g_unseen)
        kayitlar.extend(a_kayit)
        print(f"  [BATADAL] otomata bitti (durum={auto.oto.K}, novelty={novelty})")

        dl_ref_eslesme = None
        for seed in seeds:
            for mimari in self.dl_modeller:
                t0 = time.time()
                seed_ayarla(seed)
                model = self._model_olustur(mimari).egit(g_tr, g_val)
                d_kayit, d_eslesme = self._senaryolar(model, mimari, "BATADAL", 0, seed,
                                                      g_test, g_gurultu, g_unseen)
                kayitlar.extend(d_kayit)
                if mimari == ref_dl and seed == seeds[0]:
                    dl_ref_eslesme = d_eslesme
                print(f"  [BATADAL] {mimari} seed={seed} bitti ({time.time()-t0:.1f}s)")

        if a_eslesme is not None and dl_ref_eslesme is not None:
            yt, pa, pb = hizala(a_eslesme[0], a_eslesme[1],
                                dl_ref_eslesme[0], dl_ref_eslesme[1],
                                a_eslesme[0], a_eslesme[2])
            mcnemar_birikim["y"].append(yt)
            mcnemar_birikim["auto"].append(pa)
            mcnemar_birikim["dl"].append(pb)
        mcnemar = self._mcnemar_hesapla(mcnemar_birikim, "BATADAL", "automata", ref_dl)
        return kayitlar, mcnemar

    def _mcnemar_hesapla(self, birikim, veri_seti, a_ad, b_ad):
        if not birikim["y"]:
            return None
        yt = np.concatenate(birikim["y"])
        pa = np.concatenate(birikim["auto"])
        pb = np.concatenate(birikim["dl"])
        sonuc = mcnemar_testi(yt, pa, pb)
        sonuc.update({"veri_seti": veri_seti, "model_a": a_ad, "model_b": b_ad})
        return sonuc

    # ---- parametre duyarlilik analizi (otomata, SKAB) ----
    def parametre_taramasi(self, fold_limit: int | None = None):
        cfg = self.cfg
        sk = cfg.veri_setleri.skab
        ham = veri_yukle(cfg, "skab")
        ws = list(cfg.otomata.parametre_taramasi.window_size)
        as_ = list(cfg.otomata.parametre_taramasi.alphabet_size)
        foldlar = list(skab_foldlar(ham, sk.fold_sayisi, sk.fold_tohumu))
        if fold_limit:
            foldlar = foldlar[:fold_limit]

        kayitlar = []
        for w in ws:
            for a in as_:
                f1ler = []
                durumlar = []      # her fold icin otomata durum (state) sayisi
                yogunluklar = []   # her fold icin gecis yogunlugu (gozlenen/olasi gecis)
                for tr_idx, test_idx in foldlar:
                    ytr, val = grup_train_val_bol(tr_idx, ham.gruplar, sk.dogrulama_orani, sk.fold_tohumu)
                    on = OnIslemci(cfg).fit(ham.X[ytr])
                    g_tr = temiz_girdi(on, ham.X[ytr], ham.y[ytr], ham.gruplar[ytr])
                    g_val = temiz_girdi(on, ham.X[val], ham.y[val], ham.gruplar[val])
                    g_test = temiz_girdi(on, ham.X[test_idx], ham.y[test_idx], ham.gruplar[test_idx])
                    model = OtomataAnomaliModeli(cfg, window_size=w, alphabet_size=a).egit(g_tr, g_val)
                    m, _, _, _ = self._degerlendir(model, g_test)
                    f1ler.append(m["f1"])
                    durumlar.append(model.oto.K)
                    yogunluklar.append(model.oto.gecis_yogunlugu())
                kayitlar.append({"window_size": w, "alphabet_size": a,
                                 "f1_ortalama": float(np.mean(f1ler)), "f1_std": float(np.std(f1ler)),
                                 "state_sayisi": float(np.mean(durumlar)),
                                 "gecis_yogunlugu": float(np.mean(yogunluklar))})
                print(f"  [tarama] w={w} a={a} F1={np.mean(f1ler):.3f} "
                      f"durum={np.mean(durumlar):.1f} yogunluk={np.mean(yogunluklar):.3f}")
        return kayitlar

    # ---- ozet ve istatistik ----
    def _ozetle(self, df: pd.DataFrame) -> pd.DataFrame:
        olcut = ["accuracy", "precision", "recall", "f1"]
        ozet = (df.groupby(["veri_seti", "model", "senaryo"])[olcut]
                .agg(["mean", "std"]).reset_index())
        ozet.columns = ["_".join(c).rstrip("_") for c in ozet.columns]
        return ozet

    def _wilcoxon(self, df: pd.DataFrame, veri_seti: str) -> list[dict]:
        """SKAB orijinal senaryosunda fold bazli F1 ile otomata vs her DL."""
        alt = df[(df.veri_seti == veri_seti) & (df.senaryo == "orijinal")]
        if alt.empty or alt.fold.nunique() < 2:
            return []
        auto = (alt[alt.model == "automata"].groupby("fold").f1.mean().sort_index())
        sonuclar = []
        for mimari in self.dl_modeller:
            dl = (alt[alt.model == mimari].groupby("fold").f1.mean().sort_index())
            ortak = auto.index.intersection(dl.index)
            s = wilcoxon_imzali(auto.loc[ortak].to_numpy(), dl.loc[ortak].to_numpy())
            s.update({"veri_seti": veri_seti, "model_a": "automata", "model_b": mimari})
            sonuclar.append(s)
        return sonuclar

    # ---- ana akis ----
    def calistir(self, hizli: bool = False, tarama: bool = True) -> dict:
        fold_limit = 1 if hizli else None
        seed_limit = 1 if hizli else None
        print("== SKAB deneyleri ==")
        skab_kayit, skab_mcnemar = self.skab_calistir(fold_limit, seed_limit)
        print("== BATADAL deneyleri ==")
        bat_kayit, bat_mcnemar = self.batadal_calistir(seed_limit)

        df = pd.DataFrame(skab_kayit + bat_kayit)
        df.to_csv(os.path.join(self.cikti, "olcumler.csv"), index=False)
        ozet = self._ozetle(df)
        ozet.to_csv(os.path.join(self.cikti, "ozet.csv"), index=False)

        istatistik = {
            "wilcoxon": self._wilcoxon(df, "SKAB"),
            "mcnemar": [s for s in (skab_mcnemar, bat_mcnemar) if s is not None],
        }
        pd.Series(istatistik).to_json(os.path.join(self.cikti, "istatistik_testleri.json"),
                                      force_ascii=False, indent=2)

        if tarama:
            print("== Parametre taramasi (otomata, SKAB) ==")
            tarama_kayit = self.parametre_taramasi(fold_limit)
            pd.DataFrame(tarama_kayit).to_csv(
                os.path.join(self.cikti, "parametre_taramasi_skab.csv"), index=False)

        print(f"\nSonuclar '{self.cikti}' dizinine yazildi.")
        return {"ozet": ozet, "istatistik": istatistik}
