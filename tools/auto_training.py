#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : auto_training.py
# @IDE: PyCharm
"""
Train lanenet script
"""
import yaml
import numpy as np
from trainner import tusimple_lanenet_single_gpu_trainner as single_gpu_trainner
from trainner import tusimple_lanenet_multi_gpu_trainner as multi_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger(log_file_name_prefix='lanenet_train')
CFG = parse_config_utils.lanenet_cfg

def update_paras(para_name, lr):
    file_data = ''
    with open('/workspace/lanenet-bisenetv2/config/tusimple_lanenet.yaml', 'r', encoding='utf-8') as f:
        for line in f:
            if para_name in line:
                line = para_name + str(lr) + '\n'
            file_data += line
    with open('/workspace/lanenet-bisenetv2/config/tusimple_lanenet.yaml', 'w', encoding='utf-8') as f:
        f.write(file_data)

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
    lr = np.array([0.001, 0.01, 0.1])
    for i, v in enumerate(lr):
        LOG.info('lr= {:3f}'.format(v))
        update_paras('    LR: ',v)
        train_model()

