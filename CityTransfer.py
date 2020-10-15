# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:49
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : CityTransfer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse


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


class CityTransfer(nn.Module):
    def __init__(self, args, feature_dim, n_source_grid, n_target_grid):
        super(CityTransfer, self).__init__()
        self.args = args

        # auto encoder
        self.auto_encoder = nn.ModuleList()
        self.auto_encoder.append(AutoEncoder(feature_dim, self.args.auto_encoder_dim))  # source
        self.auto_encoder.append(AutoEncoder(feature_dim, self.args.auto_encoder_dim))  # target

        # matrix factorization
        self.u = nn.Parameter(torch.Tensor(len(self.args.enterprise), self.args.auto_encoder_dim))
        self.b = nn.Parameter(torch.Tensor(len(self.args.enterprise), 1))
        self.e_source = nn.Parameter(torch.Tensor(n_source_grid, 1))
        self.e_target = nn.Parameter(torch.Tensor(n_target_grid, 1))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.b)
        nn.init.xavier_uniform_(self.e_source)
        nn.init.xavier_uniform_(self.e_target)

    def encode(self, origin_feature, ae_type):
        if ae_type == 's':
            encoded_x, decoded_x = self.auto_encoder[0](origin_feature)  # source
        else:
            encoded_x, decoded_x = self.auto_encoder[0](origin_feature)  # target
        return encoded_x, decoded_x

    def cal_auto_encoder_loss(self, source_feature, target_feature):
        _, decoded_source = self.encode(source_feature, 's')
        _, decoded_target = self.encode(target_feature, 't')
        # Equation (19)
        loss_ae = F.mse_loss(source_feature, decoded_source, reduction='sum') + \
                  F.mse_loss(target_feature, decoded_target, reduction='sum')
        return loss_ae

    def cal_inter_city_loss(self, score, source_feature, target_feature):
        encoded_source, _ = self.encode(source_feature, 's')
        encoded_target, _ = self.encode(target_feature, 't')
        # Equation (14)
        loss = torch.sum(score * torch.sum(torch.pow(encoded_source - encoded_target, 2), dim=1))
        return loss

    def cal_prediction_score(self, enterprise_index, grid_index, grid_feature, grid_type):
        enterprise_feature = self.u[enterprise_index]
        enterprise_bias = self.b[enterprise_index]
        if grid_type == 's':
            grid_bias = self.e_source[grid_index].reshape(-1, len(grid_index))
        else:
            grid_bias = self.e_target[grid_index].reshape(-1, len(grid_index))
        score = enterprise_bias + grid_bias + torch.matmul(enterprise_feature, grid_feature.T)
        return score


    def forward(self, mode, *inputs):
        if mode == 'cal_auto_encoder_loss':
            return self.cal_auto_encoder_loss(*inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="City Transfer Args.")
    args = parser.parse_args()
    args.auto_encoder_dim = 9
    args.enterprise = ['a', 'b', 'c']
    c = CityTransfer(args, 5, 1000, 1000)
    aa = torch.Tensor([[0.15, 0.71, 0.5, 0.4, 0.3], [0.15, 0.71, 0.5, 0.4, 0.3], [0.15, 0.71, 0.5, 0.4, 0.3]])
    bb = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1]])
    cc = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1, 0.61, 0.4, 0.9, 0.1], [0.2, 0.7, 0.4, 0.4, 0.1, 0.9, 0.4, 0.2, 0.1]])
    res = c.cal_prediction_score([1, 2, 0], [30, 40], cc, 's')
