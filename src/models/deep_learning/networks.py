"""Derin ogrenme ag mimarileri (black-box karsilastirma modelleri).

Uc mimari de ayni girdi/cikti sozlesmesini paylasir:
- Girdi : (batch, dizi_uzunlugu, ozellik_sayisi)
- Cikti : (batch,) tek logit (anomali olasiligi icin sigmoid oncesi)

Boylece egitim ve skorlama kodu mimariye bagimsiz kalir (acik/kapali ilkesi).
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMAgi(nn.Module):
    """Son zaman adiminin gizli durumu uzerinden ikili siniflandirma."""

    def __init__(self, ozellik_sayisi: int, gizli_boyut: int, katman_sayisi: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=ozellik_sayisi,
            hidden_size=gizli_boyut,
            num_layers=katman_sayisi,
            batch_first=True,
            dropout=dropout if katman_sayisi > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.cikis = nn.Linear(gizli_boyut, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        son = self.dropout(h_n[-1])
        return self.cikis(son).squeeze(-1)


class GRUAgi(nn.Module):
    """LSTM ile ayni yapida fakat GRU hucresi kullanan varyant."""

    def __init__(self, ozellik_sayisi: int, gizli_boyut: int, katman_sayisi: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=ozellik_sayisi,
            hidden_size=gizli_boyut,
            num_layers=katman_sayisi,
            batch_first=True,
            dropout=dropout if katman_sayisi > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.cikis = nn.Linear(gizli_boyut, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        son = self.dropout(h_n[-1])
        return self.cikis(son).squeeze(-1)


class CNN1DAgi(nn.Module):
    """1B evrisimli ag: yerel zamansal oruntuleri yakalar."""

    def __init__(self, ozellik_sayisi: int, gizli_boyut: int, dropout: float, cekirdek: int) -> None:
        super().__init__()
        dolgu = cekirdek // 2   # dizi uzunlugunu koruyacak simetrik dolgu
        self.evrisim = nn.Sequential(
            nn.Conv1d(ozellik_sayisi, gizli_boyut, kernel_size=cekirdek, padding=dolgu),
            nn.ReLU(),
            nn.Conv1d(gizli_boyut, gizli_boyut, kernel_size=cekirdek, padding=dolgu),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.cikis = nn.Linear(gizli_boyut, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, dizi, ozellik) -> (batch, ozellik, dizi)
        x = x.transpose(1, 2)
        ozet = self.evrisim(x).squeeze(-1)
        return self.cikis(self.dropout(ozet)).squeeze(-1)


def ag_olustur(mimari: str, ozellik_sayisi: int, gizli_boyut: int, katman_sayisi: int,
               dropout: float, cnn_cekirdek: int = 3) -> nn.Module:
    """Mimari adina gore uygun ag nesnesini uretir (fabrika)."""
    mimari = mimari.lower()
    if mimari == "lstm":
        return LSTMAgi(ozellik_sayisi, gizli_boyut, katman_sayisi, dropout)
    if mimari == "gru":
        return GRUAgi(ozellik_sayisi, gizli_boyut, katman_sayisi, dropout)
    if mimari == "cnn1d":
        return CNN1DAgi(ozellik_sayisi, gizli_boyut, dropout, cnn_cekirdek)
    raise ValueError(f"Bilinmeyen mimari: {mimari}")
