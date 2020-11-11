# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:48
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : Main_remove_transfer.py

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from time import time
from Remove_Transfer.utility.args_parser_remove_transfer import parse_args
from Remove_Transfer.utility.log_helper import logging, logging_config
from Remove_Transfer.utility.data_remove_transfer import DataLoader
from Remove_Transfer.utility.metrics_remove_transfer import ndcf_at_k, ndcf_at_k_test
from Remove_Transfer.utility.visualization_remove_transfer import VisualizationTool
from Remove_Transfer.CityTransfer_remove_transfer import CityTransfer

DEBUG = True
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
    target_grid_batch = [data.target_train_grids[i: i + args.batch_size]
                         for i in range(0, len(data.target_train_grids), args.batch_size)]
    portion_grid_batch = [data.portion_grids[i: i + args.batch_size]
                          for i in range(0, len(data.portion_grids), args.batch_size)]
    while len(target_grid_batch) < len(portion_grid_batch):
        random_i = random.randint(0, len(data.target_train_grids) - args.batch_size)
        target_grid_batch.append(data.target_train_grids[random_i: random_i + args.batch_size])
    logging.info("--------------load data done.")

    # construct model and optimizer
    model = CityTransfer(args, data.feature_dim, data.n_grid)
    model.to(DEVICE)
    logging.info(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda_4)
    logging.info("--------------construct model and optimizer done.")

    # initialize metrics
    best_epoch = -1

    mse_list = []
    ndcg_list = []
    logging.info("--------------initialize metrics done.")

    # move to GPU
    if CUDA_AVAILABLE:
        model.to(DEVICE)
        data.rating_matrix = data.rating_matrix.to(DEVICE)
        data.feature = data.feature.to(DEVICE)

    # training
    logging.info("[!]-----------start training.")
    for epoch in range(args.n_epoch):
        model.train()
        iter_total_loss = 0
        for batch_iter in range(len(portion_grid_batch)):
            time_iter = time()

            optimizer.zero_grad()
            batch_total_loss = 0

            batch_target_grid_index = target_grid_batch[batch_iter]
            batch_portion_grid_index = portion_grid_batch[batch_iter]

            # Auto Encoder
            target_grid_feature = data.feature[data.target_enterprise_index, batch_target_grid_index]
            portion_grid_feature = data.feature[data.portion_enterprise_index][:, batch_portion_grid_index]

            ae_target_batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', target_grid_feature)
            ae_portion_batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', portion_grid_feature)

            batch_total_loss += ae_target_batch_loss + ae_portion_batch_loss

            # Prediction Model
            feature_target, score_target = \
                data.get_feature_and_rel_score_for_prediction_model(batch_target_grid_index, 't')
            feature_portion, score_portion = \
                data.get_feature_and_rel_score_for_prediction_model(batch_portion_grid_index, 'p')

            prediction_portion_batch_loss = model('cal_prediction_loss', data.portion_enterprise_index,
                                                  batch_portion_grid_index, feature_portion, score_portion)
            prediction_target_batch_loss = model('cal_prediction_loss', data.target_enterprise_index,
                                                 batch_target_grid_index, feature_target, score_target)

            batch_total_loss += prediction_portion_batch_loss + args.lambda_1 * prediction_target_batch_loss

            # calculate total loss and backward
            iter_total_loss += batch_total_loss.item()
            batch_total_loss.backward()
            optimizer.step()

            if DEBUG and (batch_iter % args.print_every) == 0:
                logging.info('Training: Epoch {:04d} / {:04d} | Iter {:04d} / {:04d} | Time {:.1f}s '
                             '| Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.
                             format(epoch, args.n_epoch, batch_iter, len(portion_grid_batch) - 1, time() - time_iter,
                                    batch_total_loss.item(), iter_total_loss / (batch_iter + 1)))

        # evaluate prediction model
        if (epoch % args.evaluate_every) == 0:
            model.eval()
            with torch.no_grad():
                feature, real_score = data.get_feature_and_rel_score_for_evaluate(data.target_test_grids)
                predict_score = model('prediction', data.target_enterprise_index,
                                      data.target_test_grids, feature)

                mse_epoch = torch.nn.functional.mse_loss(real_score, predict_score)
                ndcg_epoch = ndcf_at_k(real_score, predict_score, args.K)

                ndcg_list.append(ndcg_epoch)
                mse_list.append(mse_epoch)

                if DEBUG:
                    logging.info('Evaluate: Epoch {:04d} | NDCG {:.4f} | MSE {:.4f}'.
                                 format(epoch, ndcg_epoch, mse_epoch))

    logging.info("[!]-----------training done.")

    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("ndcg")
    plt.plot(range(len(ndcg_list)), ndcg_list)
    plt.subplot(1, 2, 2)
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.plot(range(len(mse_list)), mse_list)
    plt.savefig('ndcg.png')
    plt.show()

    # testing
    logging.info("[!]-----------start testing.")
    model.eval()
    with torch.no_grad():
        feature, real_score = data.get_feature_and_rel_score_for_evaluate(data.target_test_grids)
        predict_score = model('prediction', data.target_enterprise_index,
                              data.target_test_grids, feature)

        final_mse = torch.nn.functional.mse_loss(real_score, predict_score)
        final_ndcg, exist_grids, pred_rank, pred_back_rank = ndcf_at_k_test(real_score, predict_score, args.K)

        logging.info('Test Result: NDCG {:.4f} | MSE {:.4f}'.format(final_ndcg, final_mse))

        exist_grids = data.target_test_grids[exist_grids]
        pred_rank = data.target_test_grids[pred_rank]
        pred_back_rank = data.target_test_grids[pred_back_rank]

        real_grids_draw_info, pred_grids_draw_info, pred_back_grids_draw_info \
            = data.get_grid_coordinate(exist_grids, pred_rank, pred_back_rank)
        other_shops_draw_info = data.get_other_shops_coordinate()

        visualization_tool = VisualizationTool(args)
        visualization_tool.draw_map(real_grids_draw_info, pred_grids_draw_info,
                                    pred_back_grids_draw_info, other_shops_draw_info)
