# -*- coding: utf-8 -*-
"""
model_xmark.py -- H07: resmi iTransformer'a kodlayici takvim kovaryatlari (x_mark_enc) VERILMEZ.

Resmi model (use_norm=1, NST AYNEN) ONCE kurulur (argv yamasi yok; RNG tuketimi ebeveynle ayni; yeni parametre yok).
forward: backbone(x_enc, None, x_dec, x_mark_dec) -> DataEmbedding_inverted `x_mark is None` dali: yalniz 8 degisken
token'i gomulur (12 -> 8 token). Resmi `[:, :, :N]` filtresi N = 8 ile aynen calisir.
"""
import torch.nn as nn


class XmarkYokiTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        from model import iTransformer  # resmi rakip modeli (WS/rakip/iTransformer, sys.path'te)
        if int(configs.use_norm) != 1:
            raise ValueError("XmarkYokiTransformer: use_norm=1 korunmali (kart H07; argv yamasi yok)")
        if configs.output_attention:
            raise ValueError("XmarkYokiTransformer: output_attention desteklenmiyor (Exchange betigi kullanmiyor)")
        self.backbone = iTransformer.Model(configs)
        self.pred_len = configs.pred_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.backbone(x_enc, None, x_dec, x_mark_dec)
