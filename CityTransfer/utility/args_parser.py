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

    # Area 4
    # parser.add_argument('--source_area_coordinate', nargs=8, type=float,
    # # parser.add_argument('--target_area_coordinate', nargs=8, type=float,
    #                     default=[118.730506, 118.757457, 31.975167, 32.072533],
    #                     help='Source area coordinate. [longitude1, longitude2, latitude1， latitude2]')
    # parser.add_argument('--target_area_coordinate', nargs=8, type=float,
    # # parser.add_argument('--source_area_coordinate', nargs=8, type=float,
    #                     default=[118.757457, 118.80123, 31.975167, 32.072533],
    #                     help='Target area coordinate. [longitude1, longitude2, latitude1， latitude2]')

    # Area 1
    parser.add_argument('--source_area_coordinate', nargs=8, type=float,
                        default=[118.739776, 118.814792, 32.055803, 32.100893],
                        help='Source area coordinate. [longitude1, longitude2, latitude1， latitude2]')
    parser.add_argument('--target_area_coordinate', nargs=8, type=float,
                        default=[118.729991, 118.808783, 32.011709, 32.055803],
                        help='Target area coordinate. [longitude1, longitude2, latitude1， latitude2]')

    # Area 2
    # parser.add_argument('--source_area_coordinate', nargs=8, type=float,
    #                     default=[118.768014, 118.827563, 32.004111, 32.066481],
    #                     help='Source area coordinate. [longitude1, longitude2, latitude1， latitude2]')
    # parser.add_argument('--target_area_coordinate', nargs=8, type=float,
    #                     default=[118.774311, 118.928619, 31.864258, 31.992135],
    #                     help='Target area coordinate. [longitude1, longitude2, latitude1， latitude2]')

    # Area 3
    # parser.add_argument('--source_area_coordinate', nargs=8, type=float,
    #                     default=[118.733768, 118.802089, 32.056531, 32.093186],
    #                     help='Source area coordinate. [longitude1, longitude2, latitude1， latitude2]')
    # parser.add_argument('--target_area_coordinate', nargs=8, type=float,
    #                     default=[118.766555, 118.824749, 32.013893, 32.052457],
    #                     help='Target area coordinate. [longitude1, longitude2, latitude1， latitude2]')

    # Area 0
    # parser.add_argument('--source_area_coordinate', nargs=8, type=float,
    #                     default=[118.735647, 118.788862, 32.042283, 32.094942],
    #                     help='Source area coordinate. [longitude1, longitude2, latitude1， latitude2]')
    # parser.add_argument('--target_area_coordinate', nargs=8, type=float,
    #                     default=[118.771352, 118.823537, 32.013759, 32.060615],
    #                     help='Target area coordinate. [longitude1, longitude2, latitude1， latitude2]')

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

    parser.add_argument('--K', type=int, default=10,
                        help='Calculate metric@K when evaluating.')
    parser.add_argument('--score_norm_max', type=int, default=400,
                        help='score norm max.')

    args = parser.parse_args()

    save_dir = 'trained_model/{}/source_area_coordinate{}_target_area_coordinate{}/'.format(
        args.city_name, '-'.join([str(item) for item in args.source_area_coordinate]),
        '-'.join([str(item) for item in args.source_area_coordinate]))
    args.save_dir = save_dir

    return args
