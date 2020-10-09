# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:49
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : data_loader.py
import os
import pandas as pd
import numpy as np
import collections
from utility.log_helper import logging


class DataLoader(object):
    def __init__(self, args):
        self.args = args

        # define data path
        data_dir = os.path.join(args.data_dir, args.city_name)
        dianping_data_path = os.path.join(data_dir, 'dianping.csv')

        # load dianping data
        source_area_data, target_area_data = self.load_dianping_data(dianping_data_path)
        logging.info("[1/5]       load dianping data done.")

        # split grid
        self.n_source_grid, self.n_target_grid, source_area_longitude_boundary, source_area_latitude_boundary, \
            target_area_longitude_boundary, target_area_latitude_boundary = self.split_grid()
        logging.info("[2/5]       split grid done.")

        # distribute data into grid
        self.distribute_data(source_area_data, target_area_data,
                             source_area_longitude_boundary, source_area_latitude_boundary,
                             target_area_longitude_boundary, target_area_latitude_boundary)

    def load_dianping_data(self, dianping_data_path):
        dianping_data = pd.read_csv(dianping_data_path, usecols=[0, 1, 2, 14, 15, 17, 18, 23, 27])
        dianping_data = dianping_data[dianping_data['status'] == 0].drop(columns='status')  # 筛出正常营业的店铺
        dianping_data['branchname'].fillna("-1", inplace=True)  # 将 branch name 为空值用0填充
        dianping_data.drop_duplicates(subset=['name', 'longitude', 'latitude'],
                                      keep='first', inplace=True)  # 利用 名称+经纬度 去重
        # print(dianping_data)

        source_area_data = []
        target_area_data = []
        for row in dianping_data.itertuples():
            if self.args.source_area_coordinate[0] <= row.longitude <= self.args.source_area_coordinate[1] \
                    and self.args.source_area_coordinate[2] <= row.latitude <= self.args.source_area_coordinate[3]:
                source_area_data.append(list(row)[1:])

            elif self.args.target_area_coordinate[0] <= row.longitude <= self.args.target_area_coordinate[1] \
                    and self.args.target_area_coordinate[2] <= row.latitude <= self.args.target_area_coordinate[3]:
                target_area_data.append(list(row)[1:])
        # (shop_id, name, big_category_id, small_category, longitude, latitude, review_count, branchname)
        return source_area_data, target_area_data

    def split_grid(self):
        source_area_longitude_boundary = np.append(np.arange(self.args.source_area_coordinate[0],
                                                   self.args.source_area_coordinate[1],
                                                   self.args.grid_size_longitude_degree),
                                                   self.args.source_area_coordinate[1])
        source_area_latitude_boundary = np.append(np.arange(self.args.source_area_coordinate[2],
                                                  self.args.source_area_coordinate[3],
                                                  self.args.grid_size_latitude_degree),
                                                  self.args.source_area_coordinate[3])
        target_area_longitude_boundary = np.arange(self.args.target_area_coordinate[0],
                                                   self.args.target_area_coordinate[1],
                                                   self.args.grid_size_longitude_degree)
        target_area_latitude_boundary = np.arange(self.args.target_area_coordinate[2],
                                                  self.args.target_area_coordinate[3],
                                                  self.args.grid_size_latitude_degree)
        n_source_grid = (len(source_area_longitude_boundary) - 1) * (len(source_area_latitude_boundary) - 1)
        n_target_grid = (len(target_area_longitude_boundary) - 1) * (len(target_area_latitude_boundary) - 1)
        return n_source_grid, n_target_grid, source_area_longitude_boundary, source_area_latitude_boundary, \
               target_area_longitude_boundary, target_area_latitude_boundary

    def distribute_data(self, source_area_data, target_area_data,
                        source_area_longitude_boundary, source_area_latitude_boundary,
                        target_area_longitude_boundary, target_area_latitude_boundary):
        source_data_dict = collections.defaultdict(list)
        target_data_dict = collections.defaultdict(list)

        for item in source_area_data:
            lon_index = 0
            for index, _ in enumerate(source_area_longitude_boundary[:-1]):
                if source_area_longitude_boundary[index] <= item[4] <= source_area_longitude_boundary[index + 1]:
                    lon_index = index
                    break
            lat_index = 0
            for index, _ in enumerate(source_area_latitude_boundary[:-1]):
                if source_area_latitude_boundary[index] <= item[5] <= source_area_latitude_boundary[index + 1]:
                    lat_index = index
                    break
            grid_id = lon_index * (len(source_area_latitude_boundary) - 1) + lat_index
            source_data_dict[grid_id].append(item)

        for item in target_area_data:
            lon_index = 0
            for index, _ in enumerate(target_area_longitude_boundary[:-1]):
                if target_area_longitude_boundary[index] <= item[4] <= target_area_longitude_boundary[index + 1]:
                    lon_index = index
                    break
            lat_index = 0
            for index, _ in enumerate(target_area_latitude_boundary[:-1]):
                if target_area_latitude_boundary[index] <= item[5] <= target_area_latitude_boundary[index + 1]:
                    lat_index = index
                    break
            grid_id = lon_index * (len(target_area_latitude_boundary) - 1) + lat_index
            target_data_dict[grid_id].append(item)

        # for index, val in target_data_dict.items():
        #     if len(val) > 10:
        #         print(index, val)
        #         print('_____________________________')

        return
