# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 15:23
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : utility_tool.py

import os


def ensure_dir_exist(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)