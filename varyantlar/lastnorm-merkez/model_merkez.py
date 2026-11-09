# -*- coding: utf-8 -*-
"""
model_merkez.py -- H10: lastnorm yalniz-merkezleme (sigma_lb olceklemesi kaldirilir; NLinear tarzi).

Ebeveyn LastNormiTransformer (varyantlar/lastnorm/model_lastnorm.py; import edilir, kopyalanmaz):
  x = (x_enc - last) / stdev ; dec = backbone(x, ...) ; dec = dec * stdev + last
H10 (kart uygulama notu, birebir):
  last = x_enc[:, -1:, :].detach() ; x = x_enc - last
  dec = backbone(x, x_mark_enc, x_dec, x_mark_dec)
  dec = dec + last[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)          (sigma_lb hesaplanmaz)
__init__ aynen (omurga use_norm=0, once kurulur, yeni parametre yok) -> ilk agirliklar ve RNG tuketimi ebeveynle ayni.
"""
from model_lastnorm import LastNormiTransformer


class MerkezLastNormiTransformer(LastNormiTransformer):
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        last = x_enc[:, -1:, :].detach()
        x = x_enc - last
        dec = self.backbone(x, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, N]
        dec = dec + last[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec
