# -*- coding: utf-8 -*-
"""
san_modul.py -- SAN istatistik tahmin modulu (Statistics_prediction + MLP), PORT.

Kaynak: Liu, Z. et al., "Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice
Perspective", NeurIPS 2023. Kod: https://github.com/ustc-time-series/SAN @ 7e1ca66251a91a89290846b310145c5f5db3ffc3
(2024-03-23), dosya models/Statistics_prediction.py. Lisans: Apache License 2.0 (repo LICENSE, "Copyright 2022
DLinear Authors"). Hesap birebir; yalniz (a) `copy.deepcopy(configs)` yerine MLP'nin kullandigi alanlar
(seq_len//period_len, pred_len//period_len, enc_in, period_len) acikca geciriliyor, (b) MS ozelligi (bu deneyde
features=M) ve kullanilmayan import'lar cikarildi. Exchange ayari (run_trm.sh): period_len=6, station_type=adaptive.
"""
import torch
import torch.nn as nn
from types import SimpleNamespace


class Statistics_prediction(nn.Module):
    def __init__(self, seq_len, pred_len, period_len, enc_in, station_type="adaptive"):
        super(Statistics_prediction, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in  # features == 'M'
        self.station_type = station_type
        if seq_len % period_len or pred_len % period_len:
            raise ValueError(f"SAN: seq_len {seq_len} / pred_len {pred_len} period_len {period_len}'a bolunmuyor")

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        args = SimpleNamespace(seq_len=self.seq_len // self.period_len, pred_len=self.pred_len_new,
                               enc_in=self.channels, period_len=self.period_len)
        self.model = MLP(args, mode='mean').float()
        self.model_std = MLP(args, mode='std').float()

    def normalize(self, input):
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(input, dim=-2, keepdim=True)
            std = torch.std(input, dim=-2, keepdim=True)
            norm_input = (input - mean) / (std + self.epsilon)
            input = input.reshape(bs, len, dim)
            mean_all = torch.mean(input, dim=1, keepdim=True)
            outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), input)

            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)

            return norm_input.reshape(bs, len, dim), outputs[:, -self.pred_len_new:, :]

        else:
            return input, None

    def de_normalize(self, input, station_pred):
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = input * (std + self.epsilon) + mean
            return output.reshape(bs, len, dim)
        else:
            return input


class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_len = configs.period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
