# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:48
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : Main.py

import random
import numpy as np
import pandas as pd
import torch
from time import time
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
    logging.info("--------------parse args and init done.")

    # load data
    data = DataLoader(args)
    source_batch = [data.source_grid_ids[i: i+args.batch_size]
                    for i in range(0, len(data.source_grid_ids), args.batch_size)]
    target_batch = [data.target_grid_ids[i: i+args.batch_size]
                    for i in range(0, len(data.target_grid_ids), args.batch_size)]
    logging.info("--------------load data done.")

    # construct model and optimizer
    model = CityTransfer(args, data.feature_dim, data.n_source_grid, data.n_target_grid)
    model.to(DEVICE)
    logging.info(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_4)
    logging.info("--------------construct model and optimizer done.")

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    mse_list = []
    ndcg_list = []
    logging.info("--------------initialize metrics done.")

    logging.info("[!]-----------start training.")
    for epoch in range(args.n_epoch):
        model.train()

        # update AutoEncoder
        ae_total_loss = 0
        for batch_iter, batch_index in enumerate(source_batch):
            time_iter = time()
            grid_feature = data.source_feature[:, batch_index]
            optimizer.zero_grad()
            batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', grid_feature, 's')
            batch_loss.backward()
            optimizer.step()
            ae_total_loss += batch_loss.item()
            if (batch_iter % args.O3_print_every) == 0:
                logging.info('AE Source Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s '
                             '| Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.
                             format(epoch, batch_iter, len(source_batch), time() - time_iter,
                                    batch_loss.item(), ae_total_loss / (batch_iter+1)))

        ae_total_loss = 0
        for batch_iter, batch_index in enumerate(target_batch):
            time_iter = time()
            grid_feature = data.target_feature[:, batch_index]
            optimizer.zero_grad()
            batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', grid_feature, 't')
            batch_loss.backward()
            optimizer.step()
            ae_total_loss += batch_loss.item()
            if (batch_iter % args.O3_print_every) == 0:
                logging.info('AE Target Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s '
                             '| Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.
                             format(epoch, batch_iter, len(target_batch), time() - time_iter,
                                    batch_loss.item(), ae_total_loss / (batch_iter+1)))

    logging.info("[!]-----------training done.")
