# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:48
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : Main.py

import random
import numpy as np
import pandas as pd
import torch
from utility.args_parser import parse_args
from utility.log_helper import logging, logging_config
from utility.data_loader import DataLoader
from CityTransfer import CityTransfer

CUDA_AVAILABLE = False
DEVICE = None
N_GPU = 0


def system_init(system_args):
    # set seed
    random.seed(system_args.seed)
    np.random.seed(system_args.seed)
    torch.manual_seed(system_args.seed)

    # init log
    logging_config(system_args.save_dir, no_console=False)

    # CUDA
    global CUDA_AVAILABLE, DEVICE, N_GPU
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    N_GPU = torch.cuda.device_count()
    if N_GPU > 0:
        torch.cuda.manual_seed_all(system_args.seed)
    CUDA_AVAILABLE = False
    DEVICE = torch.device('cpu')
    N_GPU = 0

    # other settings
    # 显示所有列
    pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    # get args and init
    args = parse_args()
    system_init(args)
    logging.info(args)
    logging.info("--------------parse args and init done")

    # load data
    data = DataLoader(args)
    logging.info("--------------load data done")

    # miao
    model = CityTransfer(args, data.feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_4)