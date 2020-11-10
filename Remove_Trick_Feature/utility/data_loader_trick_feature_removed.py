# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:49
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : _discarded_data_loader.py
import os
import pandas as pd
import numpy as np
import collections
import random
import torch
from sklearn import preprocessing
from Remove_Trick_Feature.utility.log_helper import logging
from Remove_Trick_Feature.utility.utility_tool import _norm


class DataLoader(object):
    def __init__(self, args):
        self.args = args

        # define data path
        data_dir = os.path.join(args.data_dir, args.city_name)
        # dianping_data_path = os.path.join(data_dir, 'dianping.csv')
        dianping_data_path = os.path.join(data_dir, 'dianping_bookshop_edit.csv')

        # load dianping data
        source_area_data, target_area_data, self.big_category_dict, self.big_category_dict_reverse, \
            self.small_category_dict, self.small_category_dict_reverse = self.load_dianping_data(dianping_data_path)
        self.n_big_category = len(self.big_category_dict)
        self.n_small_category = len(self.small_category_dict)
        logging.info("[1 /10]       load dianping data done.")

        # check enterprise and get small category set
        valid_small_category_set, self.target_enterprise_index, self.all_enterprise_index, \
            self.portion_enterprise_index = self.check_enterprise(source_area_data, target_area_data)
        logging.info("[2 /10]       check enterprise and get small category set.")

        # split grid
        self.n_source_grid, self.n_target_grid, self.source_area_longitude_boundary, \
            self.source_area_latitude_boundary, self.target_area_longitude_boundary, self.target_area_latitude_boundary\
            = self.split_grid()
        logging.info("[3 /10]       split grid done.")

        # distribute data into grids
        source_data_dict, target_data_dict, source_grid_enterprise_data, target_grid_enterprise_data \
            = self.distribute_data(source_area_data, target_area_data)
        logging.info("[4 /10]       distribute data into grids done.")

        # generate rating matrix for Transfer Rating Prediction Model
        self.source_rating_matrix, self.target_rating_matrix = self.generate_rating_matrix(source_grid_enterprise_data,
                                                                                           target_grid_enterprise_data)
        logging.info("[5 /10]       generate rating matrix for Transfer Rating Prediction Model done.")

        # extract geographic features
        source_geographic_features, target_geographic_features = self.extract_geographic_features(source_data_dict,
                                                                                                  target_data_dict)
        logging.info("[6 /10]       extract geographic features done.")

        # extract commercial features
        source_commercial_features, target_commercial_features = \
            self.extract_commercial_features(source_data_dict, target_data_dict, valid_small_category_set)
        logging.info("[7 /10]       extract commercial features done.")

        # combine features
        self.source_feature, self.target_feature, self.feature_dim = \
            self.combine_features(source_geographic_features, target_geographic_features,
                                  source_commercial_features, target_commercial_features)
        logging.info("[8 /10]       combine features done.")

        # get PCCS and generate delta set
        self.PCCS_score, self.delta_source_grid, self.delta_target_grid = \
            self.generate_delta_set(self.source_feature, self.target_feature)
        logging.info("[9 /10]       get PCCS and generate delta set done.")

        # generate training and testing index
        self.source_grid_ids, self.target_grid_ids = self.generate_training_and_testing_index()
        logging.info("[10/10]       generate training and testing index done.")

        # change data to tensor
        self.source_feature = torch.Tensor(self.source_feature)  # not sure
        self.target_feature = torch.Tensor(self.target_feature)  # not sure

    def load_dianping_data(self, dianping_data_path):
        dianping_data = pd.read_csv(dianping_data_path, usecols=[0, 1, 2, 14, 15, 17, 18, 23, 27])
        dianping_data = dianping_data[dianping_data['status'] == 0].drop(columns='status')  # 筛出正常营业的店铺
        dianping_data['branchname'].fillna("-1", inplace=True)  # 将 branch name 为空值用0填充
        dianping_data.drop_duplicates(subset=['name', 'longitude', 'latitude'],
                                      keep='first', inplace=True)  # 利用 名称+经纬度 去重

        # remap category to id
        big_category_name = dianping_data['big_category'].unique()
        small_category_name = dianping_data['small_category'].unique()
        big_category_dict = dict()
        big_category_dict_reverse = dict()
        small_category_dict = dict()
        small_category_dict_reverse = dict()
        big_category_id, small_category_id = 0, 0
        for name in big_category_name:
            big_category_dict[name] = big_category_id
            big_category_dict_reverse[big_category_id] = name
            big_category_id += 1
        for name in small_category_name:
            small_category_dict[name] = small_category_id
            small_category_dict_reverse[small_category_id] = name
            small_category_id += 1
        dianping_data['big_category'] = dianping_data['big_category'].map(lambda x: big_category_dict[x])
        dianping_data['small_category'] = dianping_data['small_category'].map(lambda x: small_category_dict[x])

        #  split into source data and target data.
        #  (shop_id, name, big_category, small_category, longitude, latitude, review_count, branchname)
        source_area_data = []
        target_area_data = []
        for row in dianping_data.itertuples():
            if self.args.source_area_coordinate[0] <= row.longitude <= self.args.source_area_coordinate[1] \
                    and self.args.source_area_coordinate[2] <= row.latitude <= self.args.source_area_coordinate[3]:
                source_area_data.append(list(row)[1:])

            elif self.args.target_area_coordinate[0] <= row.longitude <= self.args.target_area_coordinate[1] \
                    and self.args.target_area_coordinate[2] <= row.latitude <= self.args.target_area_coordinate[3]:
                target_area_data.append(list(row)[1:])

        return source_area_data, target_area_data, big_category_dict, big_category_dict_reverse, \
               small_category_dict, small_category_dict_reverse

    def check_enterprise(self, source_area_data, target_area_data):
        #  columns = ['shop_id', 'name', 'big_category', 'small_category',
        #             'longitude', 'latitude', 'review_count', 'branchname']

        source_chains = collections.defaultdict(list)
        target_chains = collections.defaultdict(list)

        valid_small_category_set = set()
        for item in source_area_data:
            if item[1] in self.args.enterprise:
                source_chains[item[1]].append(item)
                valid_small_category_set.add(item[3])
        for item in target_area_data:
            if item[1] in self.args.enterprise:
                target_chains[item[1]].append(item)
                valid_small_category_set.add(item[3])

        for name in self.args.enterprise:
            if len(source_chains[name]) == 0 or len(target_chains[name]) == 0:
                logging.error('品牌 {} 并非在原地区和目的地区都有门店'.format(name))
                exit(1)

        target_enterprise_index = -1
        for idx, name in enumerate(self.args.enterprise):
            if name == self.args.target_enterprise:
                target_enterprise_index = idx
        if target_enterprise_index < 0:
            logging.error('目标企业{}必须在所选择的几家连锁企业中'.format(self.args.target_enterprise))
            exit(1)

        all_enterprise_index = [idx for idx, _ in enumerate(self.args.enterprise)]
        portion_enterprise_index = [idx for idx, _ in enumerate(self.args.enterprise)
                                    if idx != target_enterprise_index]

        return valid_small_category_set, target_enterprise_index, all_enterprise_index, portion_enterprise_index

    def split_grid(self):
        # source_area_longitude_boundary = np.append(np.arange(self.args.source_area_coordinate[0],
        #                                                      self.args.source_area_coordinate[1],
        #                                                      self.args.grid_size_longitude_degree),
        #                                            self.args.source_area_coordinate[1])
        source_area_longitude_boundary = np.arange(self.args.source_area_coordinate[0],
                                                   self.args.source_area_coordinate[1],
                                                   self.args.grid_size_longitude_degree)
        source_area_latitude_boundary = np.arange(self.args.source_area_coordinate[2],
                                                  self.args.source_area_coordinate[3],
                                                  self.args.grid_size_latitude_degree)
        target_area_longitude_boundary = np.arange(self.args.target_area_coordinate[0],
                                                   self.args.target_area_coordinate[1],
                                                   self.args.grid_size_longitude_degree)
        target_area_latitude_boundary = np.arange(self.args.target_area_coordinate[2],
                                                  self.args.target_area_coordinate[3],
                                                  self.args.grid_size_latitude_degree)

        n_source_grid = (len(source_area_longitude_boundary) - 1) * (len(source_area_latitude_boundary) - 1)
        n_target_grid = (len(target_area_longitude_boundary) - 1) * (len(target_area_latitude_boundary) - 1)
        logging.info('n_source_grid: {}, n_target_grid: {}'.format(n_source_grid, n_target_grid))

        return n_source_grid, n_target_grid, source_area_longitude_boundary, source_area_latitude_boundary, \
            target_area_longitude_boundary, target_area_latitude_boundary

    def distribute_data(self, source_area_data, target_area_data):
        #  columns = ['shop_id', 'name', 'big_category', 'small_category',
        #             'longitude', 'latitude', 'review_count', 'branchname']
        source_data_dict = collections.defaultdict(list)
        target_data_dict = collections.defaultdict(list)
        source_grid_enterprise_data = collections.defaultdict(list)
        target_grid_enterprise_data = collections.defaultdict(list)
        for item in source_area_data:
            lon_index = 0
            for index, _ in enumerate(self.source_area_longitude_boundary[:-1]):
                if self.source_area_longitude_boundary[index] <= item[4] <= self.source_area_longitude_boundary[index + 1]:
                    lon_index = index
                    break
            lat_index = 0
            for index, _ in enumerate(self.source_area_latitude_boundary[:-1]):
                if self.source_area_latitude_boundary[index] <= item[5] <= self.source_area_latitude_boundary[index + 1]:
                    lat_index = index
                    break
            grid_id = lon_index * (len(self.source_area_latitude_boundary) - 1) + lat_index
            source_data_dict[grid_id].append(item)
            if item[1] in self.args.enterprise:
                source_grid_enterprise_data[grid_id].append(item)

        for item in target_area_data:
            lon_index = 0
            for index, _ in enumerate(self.target_area_longitude_boundary[:-1]):
                if self.target_area_longitude_boundary[index] <= item[4] <= self.target_area_longitude_boundary[index + 1]:
                    lon_index = index
                    break
            lat_index = 0
            for index, _ in enumerate(self.target_area_latitude_boundary[:-1]):
                if self.target_area_latitude_boundary[index] <= item[5] <= self.target_area_latitude_boundary[index + 1]:
                    lat_index = index
                    break
            grid_id = lon_index * (len(self.target_area_latitude_boundary) - 1) + lat_index
            target_data_dict[grid_id].append(item)
            if item[1] in self.args.enterprise:
                target_grid_enterprise_data[grid_id].append(item)

        return source_data_dict, target_data_dict, source_grid_enterprise_data, target_grid_enterprise_data

    def extract_geographic_features(self, source_data_dict, target_data_dict):
        traffic_convenience_corresponding_ids = [self.small_category_dict[x]
                                                 for x in ['公交车', '地铁站', '停车场'] if x in self.small_category_dict]

        def get_feature(grid_info):
            #  columns = ['shop_id', 'name', 'big_category', 'small_category',
            #             'longitude', 'latitude', 'review_count', 'branchname']

            n_grid_POI = len(grid_info)

            human_flow = 0
            traffic_convenience = 0
            POI_count = np.zeros(self.n_big_category)

            for POI in grid_info:
                # Equation (3)
                if POI[3] in traffic_convenience_corresponding_ids:
                    traffic_convenience -= 1
                # Equation (4)
                POI_count[POI[2]] += 1
                # Equation (2)
                human_flow -= POI[6]

            # Equation (1)
            diversity = -1 * np.sum([(v / (1.0 * n_grid_POI)) * np.log(v / (1.0 * n_grid_POI))
                                     if v != 0 else 0 for v in POI_count])

            return np.concatenate(([diversity, human_flow, traffic_convenience], POI_count))

        source_geographic_features = []
        target_geographic_features = []
        for index in range(self.n_source_grid):
            source_geographic_features.append(get_feature(source_data_dict[index]))
        for index in range(self.n_target_grid):
            target_geographic_features.append(get_feature(target_data_dict[index]))

        source_geographic_features, target_geographic_features = \
            np.array(source_geographic_features), np.array(target_geographic_features)

        diversity_max = max(np.max(source_geographic_features[:, 0]), np.max(target_geographic_features[:, 0]))
        diversity_min = min(np.min(source_geographic_features[:, 0]), np.min(target_geographic_features[:, 0]))
        human_flow_max = max(np.max(source_geographic_features[:, 1]), np.max(target_geographic_features[:, 1]))
        human_flow_min = min(np.min(source_geographic_features[:, 1]), np.min(target_geographic_features[:, 1]))
        traffic_conv_max = max(np.max(source_geographic_features[:, 2]), np.max(target_geographic_features[:, 2]))
        traffic_conv_min = min(np.min(source_geographic_features[:, 2]), np.min(target_geographic_features[:, 2]))
        POI_cnt_max = max(np.max(source_geographic_features[:, 3:]), np.max(target_geographic_features[:, 3:]))
        POI_cnt_min = min(np.min(source_geographic_features[:, 3:]), np.min(target_geographic_features[:, 3:]))

        source_geographic_features[:, 0] = _norm(source_geographic_features[:, 0], diversity_max, diversity_min)
        source_geographic_features[:, 1] = _norm(source_geographic_features[:, 1], human_flow_max, human_flow_min)
        source_geographic_features[:, 2] = _norm(source_geographic_features[:, 2], traffic_conv_max, traffic_conv_min)
        source_geographic_features[:, 3:] = _norm(source_geographic_features[:, 3:], POI_cnt_max, POI_cnt_min)
        target_geographic_features[:, 0] = _norm(target_geographic_features[:, 0], diversity_max, diversity_min)
        target_geographic_features[:, 1] = _norm(target_geographic_features[:, 1], human_flow_max, human_flow_min)
        target_geographic_features[:, 2] = _norm(target_geographic_features[:, 2], traffic_conv_max, traffic_conv_min)
        target_geographic_features[:, 3:] = _norm(target_geographic_features[:, 3:], POI_cnt_max, POI_cnt_min)

        return source_geographic_features, target_geographic_features

    def extract_commercial_features(self, source_data_dict, target_data_dict, valid_small_category_set):
        #  由于房价数据不准确，目前没有使用
        #  columns = ['shop_id', 'name', 'big_category', 'small_category',
        #             'longitude', 'latitude', 'review_count', 'branchname']

        def get_feature(grid_info):
            # note that: When calculating competitiveness, we use small category (from valid_small_category_set).
            #            When calculating Complementarity, we use big category.
            #  enterprise size * commercial features size i.e. Density, Competitiveness, Complementarity

            grid_feature = np.zeros((len(self.args.enterprise), 3))
            Nc = 0
            big_category_POI_count = np.zeros(self.n_big_category)
            for POI in grid_info:
                big_category_POI_count[POI[2]] += 1
                if POI[3] in valid_small_category_set:
                    Nc += 1
                for idx, name in enumerate(self.args.enterprise):
                    if POI[1] == name:
                        # Equation (5)
                        grid_feature[idx][0] += 1

            # Equation (6)
            if Nc > 0:
                grid_feature[:, 1] = -1 * (Nc - grid_feature[:, 0]) / (1.0 * Nc)

            # remove trick feature
            grid_feature[:, 0] = 0

            return grid_feature[:, :2]
            # Equation (7 & 8)
            # Have PROBLEMS! What is t' and t? What is the meaning of the equations?
            # rho = np.sum(big_category_POI_count > 0)
            # rho = (rho * (rho-1)) / (self.n_big_category * (self.n_big_category - 1))
            # return grid_feature

        source_commercial_features = []
        target_commercial_features = []

        for index in range(self.n_source_grid):
            source_commercial_features.append(get_feature(source_data_dict[index]))
        for index in range(self.n_target_grid):
            target_commercial_features.append(get_feature(target_data_dict[index]))

        source_commercial_features = np.swapaxes(np.array(source_commercial_features), 0, 1)
        target_commercial_features = np.swapaxes(np.array(target_commercial_features), 0, 1)

        density_max = max(np.max(source_commercial_features[:, :, 0]), np.max(target_commercial_features[:, :, 0]))
        density_min = min(np.min(source_commercial_features[:, :, 0]), np.min(target_commercial_features[:, :, 0]))

        source_commercial_features[:, :, 0] = _norm(source_commercial_features[:, :, 0], density_max, density_min)
        target_commercial_features[:, :, 0] = _norm(target_commercial_features[:, :, 0], density_max, density_min)

        return source_commercial_features, target_commercial_features

    def combine_features(self, source_geographic_features, target_geographic_features,
                         source_commercial_features, target_commercial_features):
        source_geographic_features = np.expand_dims(source_geographic_features, 0).repeat(len(self.args.enterprise),
                                                                                          axis=0)
        target_geographic_features = np.expand_dims(target_geographic_features, 0).repeat(len(self.args.enterprise),
                                                                                          axis=0)
        source_feature = np.concatenate((source_geographic_features, source_commercial_features), axis=2)
        target_feature = np.concatenate((target_geographic_features, target_commercial_features), axis=2)

        feature_dim = source_feature.shape[2]

        # enterprise size * grid size * feature size
        return source_feature, target_feature, feature_dim

    def generate_rating_matrix(self, source_grid_enterprise_data, target_grid_enterprise_data):
        # columns = ['shop_id', 'name', 'big_category', 'small_category',
        #             'longitude', 'latitude', 'review_count', 'branchname']
        source_rating_matrix = np.zeros((len(self.args.enterprise), self.n_source_grid))
        target_rating_matrix = np.zeros((len(self.args.enterprise), self.n_target_grid))
        for grid_id in range(self.n_source_grid):
            for item in source_grid_enterprise_data[grid_id]:
                for idx, name in enumerate(self.args.enterprise):
                    if item[1] == name:
                        source_rating_matrix[idx][grid_id] += item[6]

        for grid_id in range(self.n_target_grid):
            for item in target_grid_enterprise_data[grid_id]:
                for idx, name in enumerate(self.args.enterprise):
                    if item[1] == name:
                        target_rating_matrix[idx][grid_id] += item[6]
        # score_max = max(np.max(source_rating_matrix), np.max(target_rating_matrix))
        # score_min = min(np.min(source_rating_matrix), np.min(target_rating_matrix))
        # source_rating_matrix = _norm(source_rating_matrix, score_max, score_min) * 5
        # target_rating_matrix = _norm(target_rating_matrix, score_max, score_min) * 5

        source_rating_matrix = _norm(source_rating_matrix, self.args.score_norm_max, 0) * 5
        target_rating_matrix = _norm(target_rating_matrix, self.args.score_norm_max, 0) * 5

        source_rating_matrix = torch.Tensor(source_rating_matrix)
        target_rating_matrix = torch.Tensor(target_rating_matrix)
        res0 = torch.sort(target_rating_matrix[0], descending=True)
        res1 = torch.sort(target_rating_matrix[1], descending=True)

        return source_rating_matrix, target_rating_matrix

    def generate_delta_set(self, source_feature, target_feature):
        # Equation (13)
        score = []
        for idx, _ in enumerate(self.args.enterprise):
            source_info = source_feature[idx]
            target_info = target_feature[idx]
            source_mean = np.mean(source_info, axis=1)[:, None]
            target_mean = np.mean(target_info, axis=1)[:, None]
            source_std = np.std(source_info, axis=1)[:, None]
            target_std = np.std(target_info, axis=1)[:, None]
            idx_score = (np.matmul((source_info - source_mean), (target_info - target_mean).T) / self.feature_dim) / \
                        (np.matmul(source_std, target_std.T) + self.args.eps)
            score.append(idx_score)
        # score = np.array(score)
        score = torch.Tensor(score)

        delta_source_grid = [[[] for _ in range(self.n_source_grid)] for _ in self.args.enterprise]
        delta_target_grid = [[[] for _ in range(self.n_target_grid)] for _ in self.args.enterprise]

        for idx, _ in enumerate(self.args.enterprise):
            for source_grid_id in range(self.n_source_grid):
                sorted_index = np.argsort(-score[idx][source_grid_id])
                for k in range(min(self.args.gamma, self.n_target_grid)):
                    delta_source_grid[idx][source_grid_id].append(sorted_index[k])

        for idx, _ in enumerate(self.args.enterprise):
            for target_grid_id in range(self.n_target_grid):
                sorted_index = np.argsort(-score[idx][:, target_grid_id])
                for k in range(min(self.args.gamma, self.n_source_grid)):
                    delta_target_grid[idx][target_grid_id].append(sorted_index[k])
        # for idx in self.portion_enterprise_index:
        #     for target_grid_id in range(self.n_target_grid):
        #         sorted_index = np.argsort(-score[idx][:, target_grid_id])
        #         for k in range(min(self.args.gamma, self.n_source_grid)):
        #             delta_target_grid[idx][target_grid_id].append(sorted_index[k])

        delta_source_grid = np.array(delta_source_grid)
        delta_target_grid = np.array(delta_target_grid)

        return score, delta_source_grid, delta_target_grid

    def generate_training_and_testing_index(self):
        source_grid_ids = np.arange(self.n_source_grid)
        target_grid_ids = np.arange(self.n_target_grid)
        random.shuffle(source_grid_ids)
        random.shuffle(target_grid_ids)
        return source_grid_ids, target_grid_ids

    def get_score_and_feature_for_inter_city(self, batch_index, batch_type):
        score, source_feature, target_feature = [], [], []

        if batch_type == 's':
            for idx in self.all_enterprise_index:
                source_index = []
                target_index = []
                for index in batch_index:
                    source_index.extend([index for _ in range(self.delta_source_grid[idx][index].shape[0])])
                    target_index.extend(self.delta_source_grid[idx][index])
                score.append(self.PCCS_score[idx][source_index, target_index])
                source_feature.append(self.source_feature[idx][source_index])
                target_feature.append(self.target_feature[idx][target_index])
        else:
            for idx in self.portion_enterprise_index:
                source_index = []
                target_index = []
                for index in batch_index:
                    source_index.extend(self.delta_target_grid[idx][index])
                    target_index.extend([index for _ in range(self.delta_target_grid[idx][index].shape[0])])
                score.append(self.PCCS_score[idx][source_index, target_index])
                source_feature.append(self.source_feature[idx][source_index])
                target_feature.append(self.target_feature[idx][target_index])

        score = torch.stack(score, dim=0)
        source_feature = torch.stack(source_feature, dim=0)
        target_feature = torch.stack(target_feature, dim=0)

        return score, source_feature, target_feature

    def get_feature_and_rel_score_for_prediction_model(self, grid_index, grid_type):
        if grid_type == 's':
            feature = self.source_feature[:, grid_index]
            score = self.source_rating_matrix[:, grid_index]
        elif grid_type == 't':
            feature = self.target_feature[self.portion_enterprise_index][:, grid_index]
            score = self.target_rating_matrix[self.portion_enterprise_index][:, grid_index]
        else:
            logging.error('未定义类型')
            exit(1)
        return feature, score

    def get_feature_and_rel_score_for_evaluate(self, grid_index):
        feature = self.target_feature[self.target_enterprise_index, grid_index]
        score = self.target_rating_matrix[self.target_enterprise_index, grid_index]
        return feature, score

    def get_grid_coordinate_rectangle_by_grid_id(self, grid_id, grid_type):
        if grid_type == 's':
            row_id = grid_id // (len(self.source_area_latitude_boundary) - 1)
            col_id = grid_id % (len(self.source_area_latitude_boundary) - 1)
            lon_lef = self.source_area_longitude_boundary[row_id]
            lon_rig = self.source_area_longitude_boundary[row_id + 1]
            lat_down = self.source_area_latitude_boundary[col_id]
            lat_up = self.source_area_latitude_boundary[col_id + 1]
        else:
            row_id = grid_id // (len(self.target_area_latitude_boundary) - 1)
            col_id = grid_id % (len(self.target_area_latitude_boundary) - 1)
            lon_lef = self.target_area_longitude_boundary[row_id]
            lon_rig = self.target_area_longitude_boundary[row_id+1]
            lat_down = self.target_area_latitude_boundary[col_id]
            lat_up = self.target_area_latitude_boundary[col_id+1]
        return [[lat_up, lon_lef], [lat_down, lon_rig]]

    def get_grid_coordinate_circle_by_grid_id(self, grid_id, grid_type):
        if grid_type == 's':
            row_id = grid_id // (len(self.source_area_latitude_boundary) - 1)
            col_id = grid_id % (len(self.source_area_latitude_boundary) - 1)
            lon = (self.source_area_longitude_boundary[row_id] + self.source_area_longitude_boundary[row_id + 1]) / 2
            lat = (self.source_area_latitude_boundary[col_id] + self.source_area_latitude_boundary[col_id + 1]) / 2
        else:
            row_id = grid_id // (len(self.target_area_latitude_boundary) - 1)
            col_id = grid_id % (len(self.target_area_latitude_boundary) - 1)
            lon = (self.target_area_longitude_boundary[row_id] + self.target_area_longitude_boundary[row_id+1]) / 2
            lat = (self.target_area_latitude_boundary[col_id] + self.target_area_latitude_boundary[col_id+1]) / 2
        return [lat, lon]

    def get_grid_coordinate(self, real_grids, pred_grids, pred_back_grids):
        real_grids_draw_info = [self.get_grid_coordinate_rectangle_by_grid_id(grid, 't') for grid in real_grids]
        pred_grids_draw_info = [self.get_grid_coordinate_circle_by_grid_id(grid, 't') for grid in pred_grids]
        pred_back_grids_draw_info = [self.get_grid_coordinate_circle_by_grid_id(grid, 't') for grid in pred_back_grids]

        return real_grids_draw_info, pred_grids_draw_info, pred_back_grids_draw_info

    def get_grid_coordinate_rhombus_by_grid_id(self, grid_id, grid_type):
        if grid_type == 's':
            row_id = grid_id // (len(self.source_area_latitude_boundary) - 1)
            col_id = grid_id % (len(self.source_area_latitude_boundary) - 1)
            lon_lef = self.source_area_longitude_boundary[row_id]
            lon_rig = self.source_area_longitude_boundary[row_id + 1]
            lat_down = self.source_area_latitude_boundary[col_id]
            lat_up = self.source_area_latitude_boundary[col_id + 1]
        else:
            row_id = grid_id // (len(self.target_area_latitude_boundary) - 1)
            col_id = grid_id % (len(self.target_area_latitude_boundary) - 1)
            lon_lef = self.target_area_longitude_boundary[row_id]
            lon_rig = self.target_area_longitude_boundary[row_id+1]
            lat_down = self.target_area_latitude_boundary[col_id]
            lat_up = self.target_area_latitude_boundary[col_id+1]
        return [[lat_up, (lon_lef+lon_rig)/2], [(lat_up+lat_down)/2, lon_lef],
                [lat_down, (lon_lef+lon_rig)/2], [(lat_up+lat_down)/2, lon_rig]]

    def get_target_other_shops_coordinate(self):
        other_shops_draw_info = []

        _, enterprises_grids = torch.sort(self.target_rating_matrix[self.portion_enterprise_index], descending=True)
        for index, enterprise_grids in enumerate(enterprises_grids):
            valid_len = len(torch.nonzero(self.target_rating_matrix[self.portion_enterprise_index][index]))
            valid_grid = enterprise_grids[:valid_len]
            for grid in valid_grid:
                other_shops_draw_info.append(self.get_grid_coordinate_rhombus_by_grid_id(grid, 't'))

        return other_shops_draw_info
