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


def _norm(data, mmax, mmin):
    if mmax == mmin:
        return 0
    else:
        return (data - mmin) / (mmax - mmin)


def _trans_to_zero_to_five(data, mmax):
    return 1.0 / (1 + np.exp(-(data-(mmax / 2.0)) * 10 / (1.0 * mmax)))
