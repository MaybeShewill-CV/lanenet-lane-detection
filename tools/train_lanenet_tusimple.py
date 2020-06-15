#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午9:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train_lanenet_tusimple.py
# @IDE: PyCharm
"""
Train lanenet script
"""
from trainner import tusimple_lanenet_single_gpu_trainner as single_gpu_trainner
from trainner import tusimple_lanenet_multi_gpu_trainner as multi_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger(log_file_name_prefix='lanenet_train')
CFG = parse_config_utils.lanenet_cfg


def train_model():
    """

    :return:
    """
    if CFG.TRAIN.MULTI_GPU.ENABLE:
        LOG.info('Using multi gpu trainner ...')
        worker = multi_gpu_trainner.LaneNetTusimpleMultiTrainer()
    else:
        LOG.info('Using single gpu trainner ...')
        worker = single_gpu_trainner.LaneNetTusimpleTrainer()

    worker.train()
    return


if __name__ == '__main__':
    """
    main function
    """
    train_model()

