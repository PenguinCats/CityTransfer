# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 15:23
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : log_helper.py

import logging
import re
from utility.utility_tool import *


def create_log_id(dir_path):
    ensure_dir_exist(dir_path)
    dir_list = os.listdir(dir_path)
    log_list = [item for item in dir_list if (re.match("log(.*)\.log", item))]
    log_save_id = len(log_list)
    return log_save_id


def logging_config(folder=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):

    # clear handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []

    # define log dir path & log file path
    log_dir_path = os.path.join(folder, 'log/')
    ensure_dir_exist(log_dir_path)
    log_id = create_log_id(log_dir_path)
    log_path = os.path.join(log_dir_path, "log_{}.log".format(log_id))

    # make log handler
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(log_path)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        log_console = logging.StreamHandler()
        log_console.setLevel(console_level)
        log_console.setFormatter(formatter)
        logging.root.addHandler(log_console)
