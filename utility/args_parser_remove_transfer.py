# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 11:03
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : args_parser.py

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="City Transfer Args.")

    parser.add_argument('--seed', type=int, default=981125,
                        help='Random seed.')

    parser.add_argument('--city_name', nargs='?', default='Nanjing',
                        help='Choose a city from {Nanjing}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input datasets path.')

    # parser.add_argument('--enterprise', nargs='?', default=['大众书局', '西西弗书店'],
    #                     help='Input enterprise to be selected.')
    # parser.add_argument('--target_enterprise', nargs='?', default='大众书局',
    #                     help='Input target enterprise to be transferred.')
    parser.add_argument('--enterprise', nargs='?', default=['luckin coffee瑞幸咖啡', 'CoCo都可', '星巴克'],
                        help='Input enterprise to be selected.')
    parser.add_argument('--target_enterprise', nargs='?', default='CoCo都可',
                        help='Input target enterprise to be transferred.')
    # parser.add_argument('--enterprise', nargs='?', default=['NIKE', 'New Balance', '李宁'],
    #                     help='Input enterprise to be selected.')
    # parser.add_argument('--target_enterprise', nargs='?', default='NIKE',
    #                     help='Input target enterprise to be transferred.')
    # parser.add_argument('--enterprise', nargs='?', default=['肯德基', '麦当劳', '汉堡王'],
    #                     help='Input enterprise to be selected.')
    # parser.add_argument('--target_enterprise', nargs='?', default='肯德基',
    #                     help='Input target enterprise to be transferred.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size.')
    # parser.add_argument('--rating_batch_size', type=int, default=512,
    #                     help='Transfer Rating Prediction Model batch size.')
    # parser.add_argument('--inter_batch_size', type=int, default=512,`
    #                     help='Inter-City Knowledge Association batch size.')
    # parser.add_argument('--intra_batch_size', type=int, default=512,
    #                     help='Intra-City Semantic Extraction.')
    # parser.add_argument('--test_batch_size', type=int, default=10000,
    #                     help='Test batch size (the shop number to test every batch).')

    # Area 5
    parser.add_argument('--area_coordinate', nargs=8, type=float,
                        default=[118.715099, 118.818182, 31.986015, 32.148505],
                        help='Source area coordinate. [longitude1, longitude2, latitude1， latitude2]')

    parser.add_argument('--grid_size_longitude_degree', type=float, default=0.005,
                        help='Location grid size (by longitude degree).')
    parser.add_argument('--grid_size_latitude_degree', type=float, default=0.005,
                        help='Location grid size (by latitude degree).')
    parser.add_argument('--circle_size', type=float, default=500,
                        help='circle size (by meters).')

    parser.add_argument('--auto_encoder_dim', type=int, default=9,
                        help='Dimension of Auto Encoder.')
    parser.add_argument('--mess_dropout', type=int, default=0.1,
                        help='Dropout probability w.r.t. message dropout. 0: no dropout.')

    parser.add_argument('--gamma', type=int, default=8,
                        help='gamma for generate delta set.')
    parser.add_argument('--lambda_1', type=float, default=1,
                        help='trade-off parameter for O1.')
    parser.add_argument('--lambda_2', type=float, default=0.5,
                        help='trade-off parameter for O2.')
    parser.add_argument('--lambda_3', type=float, default=0.5,
                        help='trade-off parameter for O3.')
    parser.add_argument('--lambda_4', type=float, default=0.025,
                        help='trade-off parameter for regular terms.')
    parser.add_argument('--eps', type=float, default=1e-9,
                        help='eps4.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=10000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--print_every', type=int, default=1,
                        help='Iter interval of printing loss.')
    parser.add_argument('--O1_print_every', type=int, default=1,
                        help='Iter interval of printing O1 loss.')
    parser.add_argument('--O2_print_every', type=int, default=1,
                        help='Iter interval of printing O2 loss.')
    parser.add_argument('--O3_print_every', type=int, default=1,
                        help='Iter interval of printing O3 loss.')
    parser.add_argument('--O4_print_every', type=int, default=1,
                        help='Iter interval of printing O4 loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating.')

    parser.add_argument('--K', type=int, default=30,
                        help='Calculate metric@K when evaluating.')
    parser.add_argument('--score_norm_max', type=int, default=400,
                        help='score norm max.')

    args = parser.parse_args()

    save_dir = 'trained_model/{}/area_coordinate{}/'.format(
        args.city_name, '-'.join([str(item) for item in args.area_coordinate]))
    args.save_dir = save_dir

    return args
