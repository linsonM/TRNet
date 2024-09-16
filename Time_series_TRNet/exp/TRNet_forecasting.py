# @Time : 2023/3/17 20:19
#  :LSM
# @FileName: TRNet_forecasting.py
# @Software: PyCharm
import numpy

import model
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from my_utils.tools import EarlyStopping, adjust_learning_rate
from my_utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class TRNet_Forecast(Exp_Basic):
    def __init__(self, args):
        super(TRNet_Forecast, self).__init__(args)
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.label_len = args.label_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate
                                 )

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = torch.nn.HuberLoss(reduction='mean')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - A_Corss_Attention_Layer
                dec_inp = batch_y
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, batch_y = self.model(vali_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, batch_y = self.model(vali_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, batch_y = self.model(vali_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, batch_y = self.model(vali_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)

                loss = criterion(outputs, batch_y)

                total_loss.append(loss)
        total_loss = torch.tensor(total_loss, device='cpu')
        total_loss_mean = np.average(total_loss)
        self.model.train()
        return total_loss_mean, total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        criterion = self._select_criterion()

        Time = torch.tensor([]).reshape(-1)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for j, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # x_dec = batch_y[:, -self.label_len - self.pred_len:-self.pred_len, :].float().to(self.device)

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                iter_count += 1
                model_optim.zero_grad()
                dec_inp = batch_y
                outputs, batch_y = self.model(train_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (j + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(j + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - j)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            # total_loss.append(train_loss)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            T = torch.tensor(time.time() - epoch_time).reshape(-1)
            Time = torch.cat([Time, T], dim=-1)
            train_loss_mean = np.average(train_loss)
            vali_loss_mean, total_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss_mean, test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_mean, vali_loss_mean, test_loss_mean))
            early_stopping(vali_loss_mean, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if epoch == 0:
                total_loss_all = vali_loss_mean
            else:
                total_loss_all = np.append(total_loss_all, vali_loss_mean)
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.savetxt(folder_path + 'time.csv', Time, delimiter=',')
        np.savetxt(folder_path + 'total_loss1.csv', total_loss_all, delimiter=',')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - A_Corss_Attention_Layer
                dec_inp = batch_y
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, batch_y = self.model(test_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, batch_y = self.model(test_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, batch_y = self.model(test_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs, batch_y = self.model(test_data, batch_x, batch_x_mark, dec_inp, batch_y_mark)

                preds.append(outputs)
                trues.append(batch_y)

        preds = torch.tensor([item.cpu().detach().numpy() for item in preds]).squeeze(-1)
        trues = torch.tensor([item.cpu().detach().numpy() for item in trues]).squeeze(-1)

        trues = np.array(trues)
        preds = np.array(preds)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.savetxt(folder_path + 'metrics.csv', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.savetxt(folder_path + 'pred.csv', preds, delimiter=',')
        np.savetxt(folder_path + 'true.csv', trues, delimiter=',')
        print('mae:', mae, 'mse:', mse, 'rmse:', rmse, 'mape:', mape, 'mspe:', mspe, 'r2:', r2)
        return
