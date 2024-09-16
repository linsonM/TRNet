# @Time : 2023/4/13 11:18
#  :LSM
# @FileName: GRUcs.py
# @Software: PyCharm
# Define LSTM Neural Networks
from torch import nn
import numpy as np
import logging
import sys
import torch
import torch.optim


class RNN_unit(nn.Module):
    def __init__(self, input_size, select, hidden_size=20, output_size=1, num_layers=1):
        super(RNN_unit, self).__init__()
        if select == 0:
            self.lstm = nn.GRU(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn

        self.linear1 = nn.Linear(hidden_size, output_size)
    def forward(self, _x):
        _x = _x.float()
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # residual is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x
class RNN_layer(nn.Module):
    def __init__(self, c_in, c_out, pre_len, dropout, select):
        super(RNN_layer, self).__init__()
        self.lstm_model = RNN_unit(c_in,  select, 20, c_out, 1).cuda()
        self.pred_len = pre_len
        self.drop = torch.nn.Dropout(dropout)
    def forward(self, x):
        x_lstm = torch.cat([x[:, :-self.pred_len, :], torch.zeros_like(x)[:, -self.pred_len:, :]], dim=1).cuda()
        x_lstm = x_lstm.transpose(0, 1)
        pred = self.lstm_model(x_lstm)
        pred = pred.transpose(0, 1)
        x = torch.cat([x[:, :self.pred_len, :], pred], dim=1)
        x = self.drop(x)
        lstm_pred = self.drop(pred)
        return x, lstm_pred
