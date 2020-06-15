#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/14 下午9:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : init_logger.py
# @IDE: PyCharm
"""
Log relative utils
"""
import os.path as ops
import time

import loguru

from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg


def get_logger(log_file_name_prefix):
    """

    :param log_file_name_prefix: log文件名前缀
    :return:
    """
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file_name = '{:s}_{:s}.log'.format(log_file_name_prefix, start_time)
    log_file_path = ops.join(CFG.LOG.SAVE_DIR, log_file_name)

    logger = loguru.logger
    log_level = 'INFO'
    if CFG.LOG.LEVEL == "DEBUG":
        log_level = 'DEBUG'
    elif CFG.LOG.LEVEL == "WARNING":
        log_level = 'WARNING'
    elif CFG.LOG.LEVEL == "ERROR":
        log_level = 'ERROR'

    logger.add(
        log_file_path,
        level=log_level,
        format="{time} {level} {message}",
        retention="10 days",
        rotation="1 week"
    )

    return logger
