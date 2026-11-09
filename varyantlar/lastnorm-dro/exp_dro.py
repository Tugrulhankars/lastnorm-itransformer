# -*- coding: utf-8 -*-
"""
exp_dro.py -- H09: lastnorm + zaman-grubu GroupDRO egitim kaybi (Sagawa et al. 2020, cevrimici Alg. 1).

ExpLastNormDRO(ExpLastNorm): ExpLastNorm = varyantlar/lastnorm/kos.py `exp_sinifi_kur()` ciktisi (birebir; kos.py
yukler). YALNIZ `train` override edilir: resmi Exp_Long_Term_Forecast.train (experiments/exp_long_term_forecasting.py
satir 82-184) birebir kopyalanmistir; farklar (kart uygulama notu (a)-(e)) `# [H09 x]` ile isaretli:
  (a) train_data IndeksliVeri sarmalayicisina alinir (__getitem__(i) -> orijinal 4'lu + i); DataLoader data_provider
      ile AYNI argumanlarla yeniden kurulur (batch_size, shuffle=True, num_workers, drop_last=True) -> sampler RNG
      tuketimi degismez.
  (b) grup = (idx * K) // N_H  (== np.array_split(np.arange(N_H), K) sinirlari; dogrula.py sinar).
  (c) kayip: orn = ((outputs - batch_y)**2).mean(dim=(1,2)); l_g = orn[grup==g].mean() (bossa 0);
      q = q * exp(eta * l.detach()); q = q / q.sum(); loss = (q * l).sum(). q her ufuk basinda uniform, epoch'lar
      boyunca surer. train_loss gunlugune ERM MSE (orn.mean()) yazilir; DRO kaybi ve q ayri satirda (stdout).
  (d) vali(vali_data, vali_loader, criterion) resmi nn.MSELoss ile AYNEN; erken durdurma ve lr cizelgesi aynen.
  (e) test_loss = self.vali(test_data, test_loader, ...) HESAPLANMAZ (nan basilir). NOT (uygulama): resmi satir
      test_loader uzerinde bir iterator acar ve torch DataLoader her iterator acilisinda global RNG'den bir int64
      ceker (torch/utils/data/dataloader.py:601, shuffle'siz loader dahil). Bu cekilis atlanirsa epoch >= 2 train
      karistirma sirasi ebeveynden ayrisir (kartin "egitimi degistirmez / train siralamasi ayni RNG" sartina aykiri).
      Bu yuzden yalniz `iter(test_loader)` olusturulur ve atilir: num_workers=0'da hicbir test ornegi okunmaz,
      hicbir test kaybi hesaplanmaz, RNG akisi resmiyle ayni kalir (dogrula.py (vi) sinar).
"""
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DRO_ETA = 0.01
DRO_K = 4
DRO_KAYIT = {}  # pred_len -> {"q_son", "grup_n", "grup_mse_son", "q_iz", "dro_kayip_iz"}


class IndeksliVeri(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        return (*self.ds[i], i)

    def __len__(self):
        return len(self.ds)


def grup_ata(idx, n, k=DRO_K):
    return (idx * k) // n


def dro_kayip(orn, grup, q, eta=DRO_ETA, k=DRO_K):
    """orn [B] (ornek basina MSE, grad'li), grup [B] long, q [K] -> (loss, q_yeni, l [K])."""
    sifir = orn.new_zeros(())
    ell = torch.stack([orn[grup == g].mean() if bool((grup == g).any()) else sifir for g in range(k)])
    q = q * torch.exp(eta * ell.detach())
    q = q / q.sum()
    loss = (q * ell).sum()
    return loss, q, ell


def dro_sinifi_kur(ExpLastNorm):
    from utils.tools import EarlyStopping, adjust_learning_rate

    class ExpLastNormDRO(ExpLastNorm):
        def train(self, setting):
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

            # [H09 a] indeks sarmalayici + data_provider ile ayni argumanlarla DataLoader
            N_H = len(train_data)
            train_loader = DataLoader(IndeksliVeri(train_data), batch_size=self.args.batch_size, shuffle=True,
                                      num_workers=self.args.num_workers, drop_last=True)
            grup_n = np.bincount(grup_ata(np.arange(N_H), N_H), minlength=DRO_K).tolist()
            q = torch.full((DRO_K,), 1.0 / DRO_K, device=self.device)  # [H09 c] her ufuk basinda uniform
            kayit = {"q_iz": [], "dro_kayip_iz": [], "grup_n": grup_n}
            print(f"[H09] GroupDRO: K={DRO_K} eta={DRO_ETA} N_H={N_H} grup_n={grup_n}")

            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)

            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion()

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []
                dro_loss_ep = []
                g_top = torch.zeros(DRO_K, dtype=torch.float64)
                g_say = torch.zeros(DRO_K, dtype=torch.float64)

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, idx) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                    grup = grup_ata(idx.long(), N_H).to(self.device)  # [H09 b]

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            orn = ((outputs - batch_y) ** 2).mean(dim=(1, 2))  # [H09 c]
                            loss, q, ell = dro_kayip(orn, grup, q)
                            train_loss.append(orn.mean().item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        orn = ((outputs - batch_y) ** 2).mean(dim=(1, 2))  # [H09 c]
                        loss, q, ell = dro_kayip(orn, grup, q)
                        train_loss.append(orn.mean().item())  # ERM MSE (karsilastirilabilirlik)
                    dro_loss_ep.append(loss.item())
                    od = orn.detach().double().cpu(); gd = grup.cpu()
                    for g in range(DRO_K):
                        m_ = gd == g
                        g_top[g] += od[m_].sum(); g_say[g] += m_.sum()

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)  # [H09 d] resmi MSE, aynen
                _ = iter(test_loader)  # [H09 e] yalniz RNG cekilisi (resmi satirla ayni); test verisi okunmaz
                test_loss = float("nan")  # [H09 e] test kaybi hesaplanmaz

                q_l = [float(x) for x in q.detach().cpu()]
                g_mse = [float(g_top[g] / g_say[g]) if g_say[g] > 0 else float("nan") for g in range(DRO_K)]
                kayit["q_iz"].append(q_l)
                kayit["dro_kayip_iz"].append(float(np.average(dro_loss_ep)))
                kayit["q_son"], kayit["grup_mse_son"] = q_l, g_mse
                print("[H09] DRO epoch {0} | dro_kayip {1:.7f} | q {2} | grup_train_mse {3}".format(
                    epoch + 1, kayit["dro_kayip_iz"][-1], [round(x, 4) for x in q_l], [round(x, 5) for x in g_mse]))
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

                # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

            DRO_KAYIT[int(self.args.pred_len)] = kayit
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            return self.model

    return ExpLastNormDRO
