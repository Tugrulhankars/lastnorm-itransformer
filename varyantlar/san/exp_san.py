# -*- coding: utf-8 -*-
"""
exp_san.py -- H01: resmi Exp_Long_Term_Forecast alt sinifi; SAN iki asamali egitim semasi.

Egitim mantigi SAN@7e1ca66 exp/exp_main.py (Exp_Main.train / vali / station_loss) ile ayni:
  * epoch dongusu range(train_epochs + 5); ilk 5 epoch YALNIZ istatistik modulu: kayip = station_loss
    (hedef ufkun period_len dilimlerinin mean/std'si [torch.std, unbiased] ile SAN tahmini arasinda MSE),
    station_optim (Adam, lr station_lr=1e-3) step; ayri EarlyStopping (val istasyon kaybi) ve checkpoint.
  * epoch == 5'te en iyi istasyon checkpoint'i yuklenir; sonra omurga epoch'lari: kayip
    MSE(de_normalize(omurga(normalize(x))), y); YALNIZ model_optim (omurga parametreleri, Adam lr 1e-4) step
    (SAN kodu: `if epoch + 1 <= station_pretrain_epoch: station_optim.step() else: model_optim.step()`;
    istatistik modulunun gradyanlari sifirlanir, adim atilmaz). Erken durdurma resmi patience (3) ile omurga
    val kaybina gore; lr cizelgesi adjust_learning_rate(model_optim, epoch+1-5, lr) ve
    adjust_learning_rate(station_optim, epoch+1-5, station_lr) (SAN kodu; ikincisi etkisiz).
  * Omurga epoch sayisi ebeveynle ayni (resmi .sh; h336 = 1).
Farklar (hesaba etkisiz): (1) on-egitim epoch'larinda test kumesinde kayip hesaplanmaz/basilmaz (SAN yalniz
basiyordu; test yukleyici karistirilmadigi icin RNG tuketmez); (2) on-egitim satirlari resmi
"Epoch: N, Steps:" bicimiyle BASILMAZ ("[SAN-onegitim] ep N ..."), boylece rakip_kos EPOCH_RE yalniz omurga
epoch'larini progress.json'a yazar (kosum.py --kural val_loss<0.01 cakismasi, kart H01 uygulama notu);
(3) istasyon checkpoint yolu {RUN_DIR}/ckpt/<setting>/station/ ('./station/' yerine);
(4) omurga checkpoint'i sarmalayicinin tamamini (omurga + SAN) kaydeder; SAN omurga fazinda guncellenmedigi icin
SAN'in yalniz-omurga kaydiyla esdeger. Omurga epoch satirlari resmi bicimde basilir (Test Loss rakip_kos
tarafindan maskelenir).
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from experiments.exp_basic import Exp_Basic
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast as _ResmiExp
from utils.tools import EarlyStopping

from model_san import SANiTransformer

PERIOD_LEN = 6        # SAN run_trm.sh Exchange satiri
STATION_LR = 0.001    # SAN run_trm.sh Exchange satiri
STATION_TYPE = "adaptive"
STATION_PRETRAIN_EPOCH = 5  # SAN exp_main.py: 5 if station_type == 'adaptive'

KAYIT = {}  # pred_len -> ek metrikler


def adjust_learning_rate(optimizer, epoch, args, learning_rate):
    """SAN utils/tools.py adjust_learning_rate (type1) -- acik lr argumani."""
    if args.lradj != "type1":
        raise ValueError("Exchange betigi lradj=type1 bekleniyor")
    lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class ExpSAN(_ResmiExp):
    def __init__(self, args):
        # resmi __init__ = super(Exp_Long_Term_Forecast, self).__init__(args); modul adi yamalandigi icin
        # dogrudan Exp_Basic (esdeger, ozyineleme yok)
        Exp_Basic.__init__(self, args)
        self.station_pretrain_epoch = STATION_PRETRAIN_EPOCH

    def _build_model(self):
        return SANiTransformer(self.args, period_len=PERIOD_LEN, station_type=STATION_TYPE).float()

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.backbone.parameters(), lr=self.args.learning_rate)
        self.station_optim = optim.Adam(self.model.san.parameters(), lr=STATION_LR)
        return model_optim

    def station_loss(self, y, statistics_pred):
        bs, len_, dim = y.shape
        y = y.reshape(bs, -1, PERIOD_LEN, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        return self.criterion(statistics_pred, station_ture)

    def _dec_inp(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        return torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

    def vali_san(self, loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if epoch + 1 <= self.station_pretrain_epoch:
                    _, statistics_pred = self.model.san.normalize(batch_x)
                    y = batch_y[:, -self.args.pred_len:, :]
                    loss = self.station_loss(y, statistics_pred)
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, self._dec_inp(batch_y), batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    y = batch_y[:, -self.args.pred_len:, :]
                    loss = criterion(outputs.detach().cpu(), y.detach().cpu())
                total_loss.append(loss.cpu().item())
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        path_station = os.path.join(path, 'station')
        os.makedirs(path_station, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_station_model = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()
        criterion = self.criterion
        P = self.station_pretrain_epoch
        if self.args.use_amp:
            raise ValueError("use_amp desteklenmiyor (Exchange betigi kullanmiyor)")

        on_egitim = []
        for epoch in range(self.args.train_epochs + P):
            iter_count = 0
            train_loss = []
            if epoch == P:
                self.model.san.load_state_dict(torch.load(os.path.join(path_station, 'checkpoint.pth')))
                print('[SAN] on-egitim bitti: en iyi istasyon checkpoint yuklendi (omurga fazinda SAN adim atmaz)')

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if epoch + 1 <= P:
                    _, statistics_pred = self.model.san.normalize(batch_x)
                    y = batch_y[:, -self.args.pred_len:, :]
                    loss = self.station_loss(y, statistics_pred)
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, self._dec_inp(batch_y), batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    y = batch_y[:, -self.args.pred_len:, :]
                    loss = criterion(outputs, y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs + P - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # SAN iki asamali sema
                if epoch + 1 <= P:
                    self.station_optim.step()
                else:
                    model_optim.step()
                model_optim.zero_grad()
                self.station_optim.zero_grad()

            train_loss = np.average(train_loss)
            if epoch + 1 <= P:
                vali_loss = self.vali_san(vali_loader, criterion, epoch)
                print("[SAN-onegitim] ep {0}/{1} adim {2} sure {3:.1f}s | istasyon_train={4:.7f} istasyon_val={5:.7f}"
                      .format(epoch + 1, P, train_steps, time.time() - epoch_time, train_loss, vali_loss))
                on_egitim.append((epoch + 1, float(train_loss), float(vali_loss)))
                early_stopping_station_model(vali_loss, self.model.san, path_station)
                adjust_learning_rate(self.station_optim, epoch + 1, self.args, STATION_LR)
            else:
                print("Epoch: {} cost time: {}".format(epoch + 1 - P, time.time() - epoch_time))
                vali_loss = self.vali_san(vali_loader, criterion, epoch)
                test_loss = self.vali_san(test_loader, criterion, epoch)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1 - P, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(model_optim, epoch + 1 - P, self.args, self.args.learning_rate)
                adjust_learning_rate(self.station_optim, epoch + 1 - P, self.args, STATION_LR)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        H = int(self.args.pred_len)
        en_iyi = min(on_egitim, key=lambda t: t[2]) if on_egitim else (None, None, float("nan"))
        KAYIT[H] = {
            f"san_onegitim_epoch_h{H}": len(on_egitim),
            f"san_onegitim_en_iyi_ep_h{H}": en_iyi[0],
            f"san_onegitim_val_min_h{H}": en_iyi[2],
            f"san_onegitim_train_son_h{H}": on_egitim[-1][1] if on_egitim else float("nan"),
            f"san_param_sayisi_h{H}": int(sum(p.numel() for p in self.model.san.parameters())),
            f"omurga_param_sayisi_h{H}": int(sum(p.numel() for p in self.model.backbone.parameters())),
        }
        return self.model
