# -*- coding: utf-8 -*-
"""
model_san_sp.py -- H11 (= H01b): SANSPiTransformer = H01'in SANiTransformer'i, std-MLP son aktivasyonu
nn.ReLU() -> nn.Softplus() (beta=1, threshold=20; PyTorch varsayilani). Kart hipotezler/H11.md "Degisen tek sey".

Softplus parametresiz oldugu icin parametre sayisi, ilk agirliklar ve RNG akisi H01 ile ayni seed'de bit-bit aynidir
(dogrula.py ile sinanir). varyantlar/san/ dosyalarina dokunulmaz.
"""
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "san"))
from model_san import SANiTransformer  # noqa: E402


class SANSPiTransformer(SANiTransformer):
    def __init__(self, configs, period_len=6, station_type="adaptive"):
        super().__init__(configs, period_len=period_len, station_type=station_type)
        self.san.model_std.final_activation = nn.Softplus()  # H11: tek satir
