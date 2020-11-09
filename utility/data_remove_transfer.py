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
from utility.log_helper import logging
from utility.utility_tool import _norm


class DataLoader(object):
    def __init__(self, args):
        self.args = args

        # define data path
        data_dir = os.path.join(args.data_dir, args.city_name)
        # dianping_data_path = os.path.join(data_dir, 'dianping.csv')
        dianping_data_path = os.path.join(data_dir, 'dianping_bookshop_edit.csv')

        # load dianping data
        area_data, self.big_category_dict, self.big_category_dict_reverse, \
            self.small_category_dict, self.small_category_dict_reverse = self.load_dianping_data(dianping_data_path)
        self.n_big_category = len(self.big_category_dict)
        self.n_small_category = len(self.small_category_dict)
        logging.info("[1 /9]       load dianping data done.")

        # check enterprise and get small category set
        valid_small_category_set, self.target_enterprise_index, self.all_enterprise_index, \
            self.portion_enterprise_index = self.check_enterprise(area_data)
        logging.info("[2 /9]       check enterprise and get small category set.")

        # split grid
        self.n_grid, self.area_longitude_boundary, self.area_latitude_boundary = self.split_grid()
        logging.info("[3 /9]       split grid done.")

        # distribute data into grids
        data_dict, grid_enterprise_data = self.distribute_data(area_data)
        logging.info("[4 /9]       distribute data into grids done.")

        # generate rating matrix for Transfer Rating Prediction Model
        self.rating_matrix = self.generate_rating_matrix(grid_enterprise_data)
        logging.info("[5 /9]       generate rating matrix for Transfer Rating Prediction Model done.")

        # extract geographic features
        geographic_features = self.extract_geographic_features(data_dict)
        logging.info("[6 /9]       extract geographic features done.")

        # extract commercial features
        commercial_features = self.extract_commercial_features(data_dict, valid_small_category_set)
        logging.info("[7 /9]       extract commercial features done.")

        # combine features
        self.feature, self.feature_dim = self.combine_features(geographic_features, commercial_features)
        logging.info("[8 /9]       combine features done.")

        # generate training and testing index
        self.target_train_grids, self.target_test_grids, self.all_grids = self.generate_training_and_testing_index()
        logging.info("[9 /9]       generate training and testing index done.")

        # change data to tensor
        self.feature = torch.Tensor(self.feature)  # not sure

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

        #  filter data in area
        #  (shop_id, name, big_category, small_category, longitude, latitude, review_count, branchname)
        area_data = []
        for row in dianping_data.itertuples():
            if self.args.area_coordinate[0] <= row.longitude <= self.args.area_coordinate[1] \
                    and self.args.area_coordinate[2] <= row.latitude <= self.args.area_coordinate[3]:
                area_data.append(list(row)[1:])

        return area_data, big_category_dict, big_category_dict_reverse, \
            small_category_dict, small_category_dict_reverse

    def check_enterprise(self, area_data):
        #  columns = ['shop_id', 'name', 'big_category', 'small_category',
        #             'longitude', 'latitude', 'review_count', 'branchname']

        chains = collections.defaultdict(list)

        valid_small_category_set = set()
        for item in area_data:
            if item[1] in self.args.enterprise:
                chains[item[1]].append(item)
                valid_small_category_set.add(item[3])

        for name in self.args.enterprise:
            if len(chains[name]) <= 1:
                logging.error('品牌 {} 在该区域门店少于 2 家'.format(name))
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
        area_longitude_boundary = np.arange(self.args.area_coordinate[0],
                                            self.args.area_coordinate[1],
                                            self.args.grid_size_longitude_degree)
        area_latitude_boundary = np.arange(self.args.area_coordinate[2],
                                           self.args.area_coordinate[3],
                                           self.args.grid_size_latitude_degree)

        n_grid = (len(area_longitude_boundary) - 1) * (len(area_latitude_boundary) - 1)
        logging.info('source_grid: {}'.format(n_grid))

        return n_grid, area_longitude_boundary, area_latitude_boundary

    def distribute_data(self, area_data):
        #  columns = ['shop_id', 'name', 'big_category', 'small_category',
        #             'longitude', 'latitude', 'review_count', 'branchname']
        data_dict = collections.defaultdict(list)
        grid_enterprise_data = collections.defaultdict(list)
        for item in area_data:
            lon_index = 0
            for index, _ in enumerate(self.area_longitude_boundary[:-1]):
                if self.area_longitude_boundary[index] <= item[4] <= self.area_longitude_boundary[index + 1]:
                    lon_index = index
                    break
            lat_index = 0
            for index, _ in enumerate(self.area_latitude_boundary[:-1]):
                if self.area_latitude_boundary[index] <= item[5] <= self.area_latitude_boundary[index + 1]:
                    lat_index = index
                    break
            grid_id = lon_index * (len(self.area_latitude_boundary) - 1) + lat_index
            data_dict[grid_id].append(item)
            if item[1] in self.args.enterprise:
                grid_enterprise_data[grid_id].append(item)

        return data_dict, grid_enterprise_data

    def extract_geographic_features(self, data_dict):
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

        geographic_features = []
        for index in range(self.n_grid):
            geographic_features.append(get_feature(data_dict[index]))

        geographic_features = np.array(geographic_features)

        diversity_max = np.max(geographic_features[:, 0])
        diversity_min = np.min(geographic_features[:, 0])
        human_flow_max = np.max(geographic_features[:, 1])
        human_flow_min = np.min(geographic_features[:, 1])
        traffic_conv_max = np.max(geographic_features[:, 2])
        traffic_conv_min = np.min(geographic_features[:, 2])
        POI_cnt_max = np.max(geographic_features[:, 3:])
        POI_cnt_min = np.min(geographic_features[:, 3:])

        geographic_features[:, 0] = _norm(geographic_features[:, 0], diversity_max, diversity_min)
        geographic_features[:, 1] = _norm(geographic_features[:, 1], human_flow_max, human_flow_min)
        geographic_features[:, 2] = _norm(geographic_features[:, 2], traffic_conv_max, traffic_conv_min)
        geographic_features[:, 3:] = _norm(geographic_features[:, 3:], POI_cnt_max, POI_cnt_min)

        return geographic_features

    def extract_commercial_features(self, data_dict, valid_small_category_set):
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

        commercial_features = []

        for index in range(self.n_grid):
            commercial_features.append(get_feature(data_dict[index]))

        commercial_features = np.swapaxes(np.array(commercial_features), 0, 1)

        density_max = np.max(commercial_features[:, :, 0])
        density_min = np.min(commercial_features[:, :, 0])

        commercial_features[:, :, 0] = _norm(commercial_features[:, :, 0], density_max, density_min)

        return commercial_features

    def combine_features(self, geographic_features, commercial_features):
        geographic_features = np.expand_dims(geographic_features, 0).repeat(len(self.args.enterprise), axis=0)
        feature = np.concatenate((geographic_features, commercial_features), axis=2)

        feature_dim = feature.shape[2]

        # enterprise size * grid size * feature size
        return feature, feature_dim

    def generate_rating_matrix(self, grid_enterprise_data):
        # columns = ['shop_id', 'name', 'big_category', 'small_category',
        #             'longitude', 'latitude', 'review_count', 'branchname']
        rating_matrix = np.zeros((len(self.args.enterprise), self.n_grid))
        for grid_id in range(self.n_grid):
            for item in grid_enterprise_data[grid_id]:
                for idx, name in enumerate(self.args.enterprise):
                    if item[1] == name:
                        rating_matrix[idx][grid_id] += item[6]

        # score_max = max(np.max(source_rating_matrix), np.max(target_rating_matrix))
        # score_min = min(np.min(source_rating_matrix), np.min(target_rating_matrix))
        # source_rating_matrix = _norm(source_rating_matrix, score_max, score_min) * 5
        # target_rating_matrix = _norm(target_rating_matrix, score_max, score_min) * 5

        rating_matrix = _norm(rating_matrix, self.args.score_norm_max, 0) * 5

        rating_matrix = torch.Tensor(rating_matrix)

        return rating_matrix

    def generate_training_and_testing_index(self):
        sorted_target_score, target_grids_id = torch.sort(self.rating_matrix[self.target_enterprise_index],
                                                          descending=True)
        target_grids_id = target_grids_id.numpy().tolist()
        target_chain_id = target_grids_id[:len(torch.nonzero(sorted_target_score))]
        random.shuffle(target_chain_id)
        target_train_grids = target_chain_id[:len(target_chain_id)//2]
        target_test_grids = target_chain_id[len(target_chain_id)//2:]

        target_other_id = target_grids_id[len(torch.nonzero(sorted_target_score)):]
        random.shuffle(target_other_id)
        target_other_train_grids = target_other_id[:len(target_other_id) // 2]
        target_other_test_grids = target_other_id[len(target_other_id) // 2:]

        target_train = target_train_grids + target_other_train_grids
        target_test = target_test_grids + target_other_test_grids
        random.shuffle(target_train)
        random.shuffle(target_test)

        all_grids = [grid_id for grid_id in np.arange(self.n_grid)]
        random.shuffle(all_grids)

        return np.array(target_train), np.array(target_test), np.array(all_grids)

    def get_feature_and_rel_score_for_prediction_model(self, grid_index, enterprise_type):
        if enterprise_type == 't':
            feature = self.feature[self.target_enterprise_index, grid_index]
            score = self.rating_matrix[self.target_enterprise_index, grid_index]
        else:
            feature = self.feature[self.portion_enterprise_index][:, grid_index]
            score = self.rating_matrix[self.portion_enterprise_index][:, grid_index]
        return feature, score

    def get_feature_and_rel_score_for_evaluate(self, grid_index):
        feature = self.feature[self.target_enterprise_index, grid_index]
        score = self.rating_matrix[self.target_enterprise_index, grid_index]
        return feature, score

    def get_grid_coordinate_rectangle_by_grid_id(self, grid_id):
        row_id = grid_id // (len(self.area_latitude_boundary) - 1)
        col_id = grid_id % (len(self.area_latitude_boundary) - 1)
        lon_lef = self.area_longitude_boundary[row_id]
        lon_rig = self.area_longitude_boundary[row_id + 1]
        lat_down = self.area_latitude_boundary[col_id]
        lat_up = self.area_latitude_boundary[col_id + 1]
        return [[lat_up, lon_lef], [lat_down, lon_rig]]

    def get_grid_coordinate_circle_by_grid_id(self, grid_id):
        row_id = grid_id // (len(self.area_latitude_boundary) - 1)
        col_id = grid_id % (len(self.area_latitude_boundary) - 1)
        lon = (self.area_longitude_boundary[row_id] + self.area_longitude_boundary[row_id + 1]) / 2
        lat = (self.area_latitude_boundary[col_id] + self.area_latitude_boundary[col_id + 1]) / 2
        return [lat, lon]

    def get_grid_coordinate(self, real_grids, pred_grids, pred_back_grids):
        real_grids_draw_info = [self.get_grid_coordinate_rectangle_by_grid_id(grid) for grid in real_grids]
        pred_grids_draw_info = [self.get_grid_coordinate_circle_by_grid_id(grid) for grid in pred_grids]
        pred_back_grids_draw_info = [self.get_grid_coordinate_circle_by_grid_id(grid) for grid in pred_back_grids]

        return real_grids_draw_info, pred_grids_draw_info, pred_back_grids_draw_info

    def get_grid_coordinate_rhombus_by_grid_id(self, grid_id):
        row_id = grid_id // (len(self.area_latitude_boundary) - 1)
        col_id = grid_id % (len(self.area_latitude_boundary) - 1)
        lon_lef = self.area_longitude_boundary[row_id]
        lon_rig = self.area_longitude_boundary[row_id + 1]
        lat_down = self.area_latitude_boundary[col_id]
        lat_up = self.area_latitude_boundary[col_id + 1]
        return [[lat_up, (lon_lef+lon_rig)/2], [(lat_up+lat_down)/2, lon_lef],
                [lat_down, (lon_lef+lon_rig)/2], [(lat_up+lat_down)/2, lon_rig]]

    def get_other_shops_coordinate(self):
        other_shops_draw_info = []

        _, enterprises_grids = torch.sort(self.rating_matrix[self.portion_enterprise_index], descending=True)
        for index, enterprise_grids in enumerate(enterprises_grids):
            valid_len = len(torch.nonzero(self.rating_matrix[self.portion_enterprise_index][index]))
            valid_grid = enterprise_grids[:valid_len]
            for grid in valid_grid:
                other_shops_draw_info.append(self.get_grid_coordinate_rhombus_by_grid_id(grid))

        return other_shops_draw_info
