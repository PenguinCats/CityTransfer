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

cuda_available = False
device = None
n_gpu = 0


def system_init(system_args):
    # set seed
    random.seed(system_args.seed)
    np.random.seed(system_args.seed)
    torch.manual_seed(system_args.seed)

    # init log
    logging_config(system_args.save_dir, no_console=False)

    # CUDA
    global cuda_available, device, n_gpu
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(system_args.seed)
    cuda_available = False
    device = torch.device('cpu')
    n_gpu = 0

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
