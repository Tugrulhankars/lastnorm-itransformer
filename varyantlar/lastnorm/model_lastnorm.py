# -*- coding: utf-8 -*-
"""
model_lastnorm.py -- H04: resmi iTransformer NST normalizasyonunda merkez mu_lb -> son girdi degeri x_son.

Resmi (model/iTransformer.py, use_norm=1):
  means = x_enc.mean(1, keepdim=True).detach(); x_enc = x_enc - means
  stdev = sqrt(var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5); x_enc /= stdev
  ... dec = dec * stdev + means
H04 (kart uygulama notu, birebir):
  means = x_enc.mean(1, keepdim=True).detach()
  stdev = sqrt(var(x_enc - means, dim=1, keepdim=True, unbiased=False) + 1e-5)   (resmi ile ayni; detach yok)
  last  = x_enc[:, -1:, :].detach()                                                 (GIRDI penceresinin son satiri)
  x = (x_enc - last) / stdev ; dec = backbone(x, ...) ; dec = dec * stdev + last
Omurga resmi Model(configs) (use_norm=0; kos.py argv yamasi), ONCE kurulur, yeni parametre yok -> ayni seed'de
ilk agirliklar ve RNG tuketimi ebeveynle ayni.
"""
import torch
import torch.nn as nn


class LastNormiTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        from model import iTransformer  # resmi rakip modeli (WS/rakip/iTransformer, sys.path'te)
        if int(configs.use_norm) != 0:
            raise ValueError("LastNormiTransformer: omurga use_norm=0 ile kurulmali (kos.py --use_norm 0 yamasi)")
        if configs.output_attention:
            raise ValueError("LastNormiTransformer: output_attention desteklenmiyor (Exchange betigi kullanmiyor)")
        self.backbone = iTransformer.Model(configs)
        self.pred_len = configs.pred_len

    @staticmethod
    def istatistik(x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
        last = x_enc[:, -1:, :].detach()
        return means, stdev, last

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        _, stdev, last = self.istatistik(x_enc)
        x = (x_enc - last) / stdev
        dec = self.backbone(x, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, N]
        dec = dec * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1) \
            + last[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec
