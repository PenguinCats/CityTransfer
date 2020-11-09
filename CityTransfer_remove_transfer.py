# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:49
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : _discarded_CityTransfer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utility.log_helper import logging


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.mid_dim = math.ceil(math.sqrt(in_dim*out_dim))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, in_dim),
        )

    def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.encoder(x)
        # Equation (17 & 18)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x


class CityTransfer(nn.Module):
    def __init__(self, args, feature_dim, n_grid):
        super(CityTransfer, self).__init__()
        self.args = args

        # auto encoder
        self.auto_encoder = nn.ModuleList()
        self.auto_encoder.append(AutoEncoder(feature_dim, self.args.auto_encoder_dim))  # source

        # matrix factorization
        self.u = nn.Parameter(torch.Tensor(len(self.args.enterprise), self.args.auto_encoder_dim))
        self.b = nn.Parameter(torch.Tensor(len(self.args.enterprise), 1))
        self.e = nn.Parameter(torch.Tensor(n_grid, 1))

        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.b)
        nn.init.xavier_uniform_(self.e)

    def encode(self, origin_feature):
        encoded_x, decoded_x = self.auto_encoder[0](origin_feature)
        return encoded_x, decoded_x

    def cal_auto_encoder_loss(self, grid_feature):
        _, decoded_grid = self.encode(grid_feature)
        # Equation (19)
        loss_ae = F.mse_loss(grid_feature, decoded_grid, reduction='sum')
        # loss_ae = F.mse_loss(grid_feature, decoded_grid, reduction='mean')
        return loss_ae

    def cal_prediction_score(self, enterprise_index, grid_index, grid_feature):
        encoded_feature, _ = self.encode(grid_feature)
        enterprise_feature = self.u[enterprise_index]
        enterprise_bias = self.b[enterprise_index]
        grid_bias = self.e[grid_index].reshape(-1, len(grid_index))

        # Equation (10 & 11)
        if grid_feature.ndim > 2:
            score = torch.matmul(enterprise_feature.unsqueeze(1), encoded_feature.permute(0, 2, 1)).squeeze(1) + \
                enterprise_bias + grid_bias
        else:
            score = torch.matmul(enterprise_feature.unsqueeze(0), encoded_feature.t()).squeeze(0) + \
                enterprise_bias + grid_bias
        return score

    def cal_prediction_loss(self, enterprise_index, grid_index, grid_feature, real_score):
        score = self.cal_prediction_score(enterprise_index, grid_index, grid_feature)
        # origin_index = torch.argsort(real_score, descending=True)
        # predict_index = torch.argsort(score, descending=True)
        # Equation (12)
        loss = F.mse_loss(score, real_score, reduction='sum')
        # loss = F.mse_loss(score, real_score, reduction='mean')
        return loss

    def prediction(self, target_enterprise_index, grid_index, grid_feature):
        encoded_feature, _ = self.encode(grid_feature)
        enterprise_feature = self.u[target_enterprise_index]
        enterprise_bias = self.b[target_enterprise_index]
        grid_bias = self.e[grid_index].reshape(-1, len(grid_index))

        score = (torch.matmul(enterprise_feature, encoded_feature.T) + enterprise_bias + grid_bias).squeeze()
        return score

    def forward(self, mode, *inputs):
        if mode == 'cal_auto_encoder_loss':
            return self.cal_auto_encoder_loss(*inputs)
        elif mode == 'cal_prediction_loss':
            return self.cal_prediction_loss(*inputs)
        elif mode == 'prediction':
            return self.prediction(*inputs)
        else:
            logging.error('wrong parameters!')
            exit(1)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="City Transfer Args.")
#     args = parser.parse_args()
#     args.auto_encoder_dim = 9
#     args.enterprise = ['a', 'b', 'c']
#     c = CityTransfer(args, 5, 1000, 1000)
#     aa = torch.Tensor([[0.15, 0.71, 0.5, 0.4, 0.3], [0.15, 0.71, 0.5, 0.4, 0.3], [0.15, 0.71, 0.5, 0.4, 0.3]])
#     bb = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1]])
#     ab = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1]])
#     cc = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1, 0.61, 0.4, 0.9, 0.1],
#                       [0.2, 0.7, 0.4, 0.4, 0.1, 0.9, 0.4, 0.2, 0.1]])
#     dd = torch.Tensor([[0.8, 0.7], [0.3, 0.1], [0.6, 0.9]])
#     # res = c.cal_prediction_loss([1, 2, 0], [30, 40], ab, 's', dd)
#     res2 = c.cal_prediction_score([1, 2, 0], [30, 40], ab, 't')
#     print(res2)
