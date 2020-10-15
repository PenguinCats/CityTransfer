# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 15:23
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : utility_tool.py

import os
import numpy as np
import torch


def ensure_dir_exist(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def cal_pearson_correlation_coefficient(v1, v2):
    return np.mean(np.multiply((v1 - np.mean(v1)), (v2 - np.mean(v2)))) / (np.std(v1) * np.std(v2) + 1e-10)


def l2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)
