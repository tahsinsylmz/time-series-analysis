"""On isleme: normalizasyon ve PCA (boyut indirgeme).

Veri sizintisini onlemek icin tum istatistikler (scaler ortalama/sapma ve PCA
bilesenleri) YALNIZCA egitim verisi uzerinde fit edilir; ayni donusum
dogrulama ve test verisine uygulanir.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class OnIslemci:
    """Egitimde fit edilen, butun bolumlere uygulanan on isleme bloku.

    - ``olcekle``  : cok degiskenli normalize veri (derin ogrenme girdisi)
    - ``pc1``      : PCA ile tek boyuta indirgenmis seri (otomata girdisi)
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def fit(self, X_egitim: np.ndarray) -> "OnIslemci":
        self.scaler = StandardScaler().fit(X_egitim)
        if self.cfg.on_isleme.pca.aktif:
            olcekli = self.scaler.transform(X_egitim)
            n = self.cfg.on_isleme.pca.bilesen_sayisi
            self.pca = PCA(n_components=n, random_state=0).fit(olcekli)
        return self

    def olcekle(self, X: np.ndarray) -> np.ndarray:
        """Cok degiskenli normalize gosterim."""
        if self.scaler is None:
            raise RuntimeError("OnIslemci once fit edilmeli.")
        return self.scaler.transform(X)

    def pc1(self, X: np.ndarray) -> np.ndarray:
        """Otomata icin tek boyutlu (PC1) gosterim."""
        if self.pca is None:
            raise RuntimeError("PCA aktif degil veya fit edilmedi.")
        return self.pca.transform(self.scaler.transform(X))[:, 0]

    def pc1_olcekliden(self, olcekli: np.ndarray) -> np.ndarray:
        """Zaten olceklenmis veriden PC1 uretir (gurultu senaryosu icin).

        Gurultu, normalize edilmis veri uzerine eklendiginden tekrar olceklemeye
        gerek yoktur; dogrudan PCA uygulanir.
        """
        if self.pca is None:
            raise RuntimeError("PCA aktif degil veya fit edilmedi.")
        return self.pca.transform(olcekli)[:, 0]
