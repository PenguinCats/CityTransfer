# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:48
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : _discarded_Main.py

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from time import time
from utility.args_parser import parse_args
from utility.log_helper import logging, logging_config
from utility.data_loader import DataLoader
from utility.metrics import ndcf_at_k, ndcf_at_k_test
from utility.visualization import VisualizationTool
from CityTransfer import CityTransfer

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
    source_batch = [data.source_grid_ids[i: i + args.batch_size]
                    for i in range(0, len(data.source_grid_ids), args.batch_size)]
    target_batch = [data.target_grid_ids[i: i + args.batch_size]
                    for i in range(0, len(data.target_grid_ids), args.batch_size)]
    while len(source_batch) < len(target_batch):
        random_i = random.randint(0, len(data.source_grid_ids))
        source_batch.append(data.source_grid_ids[random_i: random_i + args.batch_size])
    while len(target_batch) < len(source_batch):
        random_i = random.randint(0, len(data.target_grid_ids))
        target_batch.append(data.target_grid_ids[random_i: random_i + args.batch_size])
    logging.info("--------------load data done.")

    # construct model and optimizer
    model = CityTransfer(args, data.feature_dim, data.n_source_grid, data.n_target_grid)
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
        data.source_rating_matrix = data.source_rating_matrix.to(DEVICE)
        data.target_rating_matrix = data.target_rating_matrix.to(DEVICE)
        data.source_feature = data.source_feature.to(DEVICE)
        data.target_feature = data.target_feature.to(DEVICE)
        data.PCCS_score = data.PCCS_score.to(DEVICE)

    # # training
    # logging.info("[!]-----------start training.")
    # for epoch in range(args.n_epoch):
    #     model.train()
    #     iter_total_loss = 0
    #     for batch_iter in range(len(source_batch)):
    #         time_iter = time()
    #
    #         optimizer.zero_grad()
    #         batch_total_loss = 0
    #
    #         batch_source_index = source_batch[batch_iter]
    #         batch_target_index = target_batch[batch_iter]
    #
    #         # Auto Encoder
    #         source_grid_feature = data.source_feature[:, batch_source_index]
    #         target_grid_feature = data.target_feature[:, batch_target_index]
    #
    #         ae_source_batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', source_grid_feature, 's')
    #         ae_target_batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', target_grid_feature, 't')
    #
    #         batch_total_loss += ae_source_batch_loss + ae_target_batch_loss
    #
    #         # Inter-City Knowledge Association
    #         score_1, source_feature_1, target_feature_1 = \
    #             data.get_score_and_feature_for_inter_city(batch_source_index, 's')
    #         score_2, source_feature_2, target_feature_2 = \
    #             data.get_score_and_feature_for_inter_city(batch_target_index, 't')
    #
    #         inter_city_source_batch_loss_1 = \
    #             args.lambda_2 * model('cal_inter_city_loss', score_1, source_feature_1, target_feature_1)
    #         inter_city_source_batch_loss_2 = \
    #             args.lambda_2 * model('cal_inter_city_loss', score_2, source_feature_2, target_feature_2)
    #
    #         batch_total_loss += inter_city_source_batch_loss_1 + inter_city_source_batch_loss_2
    #
    #         # Prediction Model
    #         feature_source, score_source = data.get_feature_and_rel_score_for_prediction_model(batch_source_index, 's')
    #         feature_target, score_target = data.get_feature_and_rel_score_for_prediction_model(batch_target_index, 't')
    #
    #         prediction_source_batch_loss = model('cal_prediction_loss', data.all_enterprise_index,
    #                                              batch_source_index, feature_source, 's', score_source)
    #         prediction_target_batch_loss = model('cal_prediction_loss', data.portion_enterprise_index,
    #                                              batch_target_index, feature_target, 't', score_target)
    #
    #         batch_total_loss += prediction_source_batch_loss + args.lambda_1 * prediction_target_batch_loss
    #
    #         # calculate total loss and backward
    #         iter_total_loss += batch_total_loss.item()
    #         batch_total_loss.backward()
    #         optimizer.step()
    #
    #         if DEBUG and (batch_iter % args.print_every) == 0:
    #             logging.info('Training: Epoch {:04d} / {:04d} | Iter {:04d} / {:04d} | Time {:.1f}s '
    #                          '| Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.
    #                          format(epoch, args.n_epoch, batch_iter, len(source_batch) - 1, time() - time_iter,
    #                                 batch_total_loss.item(), iter_total_loss / (batch_iter + 1)))
    #
    #     # evaluate prediction model
    #     if (epoch % args.evaluate_every) == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             feature, real_score = data.get_feature_and_rel_score_for_evaluate(data.target_grid_ids)
    #             predict_score = model('prediction', data.target_enterprise_index,
    #                                   data.target_grid_ids, feature)
    #
    #             mse_epoch = torch.nn.functional.mse_loss(real_score, predict_score)
    #             ndcg_epoch = ndcf_at_k(real_score, predict_score, args.K)
    #
    #             ndcg_list.append(ndcg_epoch)
    #             mse_list.append(mse_epoch)
    #
    #             if DEBUG:
    #                 logging.info('Evaluate: Epoch {:04d} | NDCG {:.4f} | MSE {:.4f}'.
    #                              format(epoch, ndcg_epoch, mse_epoch))
    #
    # logging.info("[!]-----------training done.")
    #
    # plt.subplot(1, 2, 1)
    # plt.xlabel("epoch")
    # plt.ylabel("ndcg")
    # plt.plot(range(len(ndcg_list)), ndcg_list)
    # plt.subplot(1, 2, 2)
    # plt.xlabel("epoch")
    # plt.ylabel("mse")
    # plt.plot(range(len(mse_list)), mse_list)
    # plt.show()

    # testing
    logging.info("[!]-----------start testing.")
    model.eval()
    with torch.no_grad():
        feature, real_score = data.get_feature_and_rel_score_for_evaluate(data.target_grid_ids)
        predict_score = model('prediction', data.target_enterprise_index,
                              data.target_grid_ids, feature)

        final_mse = torch.nn.functional.mse_loss(real_score, predict_score)
        final_ndcg, real_rank, pred_rank, pred_back_rank = ndcf_at_k_test(real_score, predict_score, args.K)

        logging.info('Test Result: NDCG {:.4f} | MSE {:.4f}'.format(final_ndcg, final_mse))

        real_rank = data.target_grid_ids[real_rank]
        pred_rank = data.target_grid_ids[pred_rank]
        pred_back_rank = data.target_grid_ids[pred_back_rank]

        real_grids_draw_info, pred_grids_draw_info, pred_back_grids_draw_info \
            = data.get_grid_coordinate(real_rank, pred_rank, pred_back_rank)
        other_shops_draw_info = data.get_target_other_shops_coordinate()

        visualization_tool = VisualizationTool(args)
        visualization_tool.draw_map(real_grids_draw_info, pred_grids_draw_info,
                                    pred_back_grids_draw_info, other_shops_draw_info)
