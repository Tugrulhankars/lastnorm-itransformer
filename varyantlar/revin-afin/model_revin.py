# -*- coding: utf-8 -*-
"""
model_revin.py -- H02: resmi iTransformer'in afinsiz NST normalizasyonu yerine RevIN (afin).

RevIN: Kim, T. et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against
Distribution Shift", ICLR 2022 (https://openreview.net/forum?id=cGDAkQo1C0p; github.com/ts-kim/RevIN).
  normalize   : x_n = gamma * (x - mu) / sigma + beta
  denormalize : y   = ((y_n - beta) / (gamma + eps^2)) * sigma + mu      (eps = 1e-5, RevIN kodu)

mu/sigma hesabi resmi iTransformer kodu (model/iTransformer.py, use_norm=1) ile KARAKTER KARAKTER ayni:
  means = x.mean(1, keepdim=True).detach(); x = x - means
  stdev = sqrt(var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  (resmi kodda stdev detach edilmez)
gamma (ones) / beta (zeros) RNG cekmez; omurga (resmi Model, use_norm=0) ONCE kurulur -> ayni seed'de
ilk agirliklar ebeveynle birebir ayni. Init'te ileri gecis ebeveynle matematiksel olarak ozdes
(dogrula_esdeger.py, CPU, maks. fark < 1e-6).
"""
import torch
import torch.nn as nn


class RevINiTransformer(nn.Module):
    EPS = 1e-5

    def __init__(self, configs):
        super().__init__()
        from model import iTransformer  # resmi rakip modeli (WS/rakip/iTransformer, sys.path'te)
        if int(configs.use_norm) != 0:
            raise ValueError("RevINiTransformer: omurga use_norm=0 ile kurulmali (kos.py --use_norm 0 yamasi)")
        if configs.output_attention:
            raise ValueError("RevINiTransformer: output_attention desteklenmiyor (Exchange betigi kullanmiyor)")
        self.backbone = iTransformer.Model(configs)  # RNG tuketimi ebeveynle ayni (use_norm parametre yaratmaz)
        self.pred_len = configs.pred_len
        n = configs.enc_in
        self.gamma = nn.Parameter(torch.ones(n))   # RevIN affine_weight
        self.beta = nn.Parameter(torch.zeros(n))   # RevIN affine_bias

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # --- normalize (resmi NST mu/sigma + RevIN afin)
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        x = x * self.gamma + self.beta
        # --- resmi omurga (use_norm=0)
        dec_out = self.backbone(x, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, N]
        # --- de-normalize (RevIN)
        dec_out = dec_out - self.beta
        dec_out = dec_out / (self.gamma + self.EPS * self.EPS)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
