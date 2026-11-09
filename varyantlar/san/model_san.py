# -*- coding: utf-8 -*-
"""
model_san.py -- H01: SANiTransformer = SAN.normalize -> resmi iTransformer (use_norm=0) -> SAN.de_normalize.

forward(x_enc, x_mark_enc, x_dec, x_mark_dec) imzasi resmi modelle ayni; boylece rakip_kos.ileri() (bizim val)
ve resmi test() degismeden calisir. SAN exp_main.py'deki ileri gecisle ayni sira: batch_x normalize edilir,
omurgaya normalize girdi verilir, cikis [-pred_len:] alinir ve SAN istatistik tahminiyle de-normalize edilir.
(iTransformer x_dec'i kullanmaz; SAN'in dec_label farki etkisizdir.)

Kurulum sirasi: OMURGA ONCE (ayni seed'de ilk omurga agirliklari ebeveynle birebir ayni -> eslestirme
korelasyonu), sonra SAN modulu. (SAN exp_main'de statistics_pred once kuruluyor; bu yalniz RNG sirasi
farkidir, hesap ayni.)
"""
import torch.nn as nn

from san_modul import Statistics_prediction


class SANiTransformer(nn.Module):
    def __init__(self, configs, period_len=6, station_type="adaptive"):
        super().__init__()
        from model import iTransformer  # resmi rakip modeli
        if int(configs.use_norm) != 0:
            raise ValueError("SANiTransformer: omurga use_norm=0 ile kurulmali (kos.py --use_norm 0 yamasi)")
        if configs.output_attention:
            raise ValueError("SANiTransformer: output_attention desteklenmiyor")
        if configs.features != "M":
            raise ValueError("SANiTransformer: yalniz features=M")
        self.backbone = iTransformer.Model(configs)
        self.san = Statistics_prediction(configs.seq_len, configs.pred_len, period_len, configs.enc_in,
                                         station_type)
        self.pred_len = configs.pred_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_n, stats = self.san.normalize(x_enc)
        y_n = self.backbone(x_n, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        return self.san.de_normalize(y_n, stats)
