# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:49
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : CityTransfer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.W = nn.parameter.Parameter(torch.Tensor(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('sigmoid'))
        self.y1 = nn.parameter.Parameter(torch.Tensor(out_dim))
        self.y2 = nn.parameter.Parameter(torch.Tensor(in_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.activation(torch.matmul(x, self.W.t()) + self.y1)
        # Equation (17 & 18)
        decoded_x = self.activation(torch.matmul(encoded_x, self.W) + self.y2)
        return encoded_x, decoded_x

