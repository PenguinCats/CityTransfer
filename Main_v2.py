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
from utility.data_loader_V2 import DataLoader
from utility.metrics import ndcg_at_k
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
    source_batch = [data.source_grid_ids[i: i+args.batch_size]
                    for i in range(0, len(data.source_grid_ids), args.batch_size)]
    target_batch = [data.target_grid_ids[i: i+args.batch_size]
                    for i in range(0, len(data.target_grid_ids), args.batch_size)]
    while len(source_batch) < len(target_batch):
        random_i = random.randint(0, len(data.source_grid_ids))
        source_batch.append(data.source_grid_ids[random_i: random_i+args.batch_size])
    while len(target_batch) < len(source_batch):
        random_i = random.randint(0, len(data.target_grid_ids))
        target_batch.append(data.target_grid_ids[random_i: random_i+args.batch_size])
    logging.info("--------------load data done.")

    # construct model and optimizer
    model = CityTransfer(args, data.feature_dim, data.n_source_grid, data.n_target_grid)
    model.to(DEVICE)
    logging.info(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_4)
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

    # training
    logging.info("[!]-----------start training.")
    for epoch in range(args.n_epoch):
        model.train()

        for batch_iter in range(len(source_batch)):
            optimizer.zero_grad()
            total_loss = 0

            batch_source_index = source_batch[batch_iter]
            batch_target_index = target_batch[batch_iter]

            source_grid_feature = data.source_feature[:, batch_source_index]
            target_grid_feature = data.target_feature[:, batch_target_index]

            ae_source_batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', source_grid_feature, 's')
            ae_target_batch_loss = args.lambda_3 * model('cal_auto_encoder_loss', target_grid_feature, 't')

            total_loss += ae_source_batch_loss.item() + ae_target_batch_loss.item()


        # update Inter-City Knowledge Association
        inter_city_total_loss = 0
        for batch_iter, batch_index in enumerate(delta_batch):
            time_iter = time()
            score, source_feature, target_feature = data.get_score_and_feature_for_inter_city(batch_index)
            optimizer.zero_grad()
            batch_loss = args.lambda_2 * model('cal_inter_city_loss', score, source_feature, target_feature)
            batch_loss.backward()
            optimizer.step()
            inter_city_total_loss += batch_loss.item()
            if DEBUG:
                if (batch_iter % args.O2_print_every) == 0:
                    logging.info('Inter-City Knowledge Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s '
                                 '| Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.
                                 format(epoch, batch_iter, len(delta_batch)-1, time() - time_iter,
                                        batch_loss.item(), inter_city_total_loss / (batch_iter+1)))

        # update Prediction Model
        prediction_model_loss = 0
        for batch_iter, batch_index in enumerate(source_batch):
            time_iter = time()
            feature, score = data.get_feature_and_rel_score_for_prediction_model(batch_index, 's')
            optimizer.zero_grad()
            batch_loss = model('cal_prediction_loss', data.all_enterprise_index, data.source_grid_ids[batch_index],
                               feature, 's', score)
            batch_loss.backward()
            optimizer.step()
            prediction_model_loss += batch_loss.item()
            if DEBUG:
                if (batch_iter % args.O1_print_every) == 0:
                    logging.info('Prediction Model Source Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s '
                                 '| Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.
                                 format(epoch, batch_iter, len(source_batch) - 1, time() - time_iter,
                                        batch_loss.item(), prediction_model_loss / (batch_iter + 1)))

        prediction_model_loss = 0
        for batch_iter, batch_index in enumerate(target_batch):
            time_iter = time()
            feature, score = data.get_feature_and_rel_score_for_prediction_model(batch_index, 't')
            optimizer.zero_grad()
            batch_loss = model('cal_prediction_loss', data.portion_enterprise_index, data.target_grid_ids[batch_index],
                               feature, 't', score)
            batch_loss.backward()
            optimizer.step()
            prediction_model_loss += batch_loss.item()
            if DEBUG:
                if (batch_iter % args.O1_print_every) == 0:
                    logging.info('Prediction Model Target Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s '
                                 '| Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.
                                 format(epoch, batch_iter, len(target_batch) - 1, time() - time_iter,
                                        batch_loss.item(), prediction_model_loss / (batch_iter + 1)))

        # evaluate prediction model
        if (epoch % args.evaluate_every) == 0:
            with torch.no_grad():
                mse_epoch = []
                ndcg_epoch = []
                for batch_iter, batch_index in enumerate(target_batch):
                    feature, real_score_batch = data.get_feature_and_rel_score_for_evaluate(batch_index)
                    predict_score_batch = model('prediction', data.target_enterprise_index,
                                                data.target_grid_ids[batch_index], feature)

                    mse = torch.nn.functional.mse_loss(real_score_batch, predict_score_batch)
                    mse_epoch.append(mse)

                    real_score_batch_index = torch.argsort(real_score_batch, descending=True)
                    predict_score_batch_index = torch.argsort(predict_score_batch, descending=True)
                    # print(real_score_batch_index)
                    # print(predict_score_batch_index)
                    ndcg = ndcg_at_k(real_score_batch_index, predict_score_batch_index, args.K)
                    ndcg_epoch.append(ndcg)

                ndcg_epoch = np.sum(ndcg_epoch) / len(ndcg_epoch)
                mse_epoch = np.sum(mse_epoch) / len(mse_epoch)
                ndcg_list.append(ndcg_epoch)
                mse_list.append(mse_epoch)

                if DEBUG:
                    logging.info('Evaluate: Epoch {:04d} | NDCG {:.4f} | MSE {:.4f}'.
                                 format(epoch, ndcg_epoch, mse_epoch))

logging.info("[!]-----------training done.")
