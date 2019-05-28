#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-31 上午11:21
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : global_config.py
# @IDE: PyCharm Community Edition
"""
Set global configuration
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 80010
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.VAL_DISPLAY_STEP = 1000
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.0005
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.95
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 4
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 4
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 2
# Set the image height
__C.TRAIN.IMG_HEIGHT = 256
# Set the image width
__C.TRAIN.IMG_WIDTH = 512
# Set the embedding features dims
__C.TRAIN.EMBEDDING_FEATS_DIMS = 4
# Set the random crop pad size
__C.TRAIN.CROP_PAD_SIZE = 32
# Set cpu multi process thread nums
__C.TRAIN.CPU_MULTI_PROCESS_NUMS = 6
# Set the train moving average decay
__C.TRAIN.MOVING_AVERAGE_DECAY = 0.9999
# Set the GPU nums
__C.TRAIN.GPU_NUM = 2

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = True
# Set the test batch size
__C.TEST.BATCH_SIZE = 2

# Test options
__C.POSTPROCESS = edict()

# Set the post process connect components analysis min area threshold
__C.POSTPROCESS.MIN_AREA_THRESHOLD = 100
# Set the post process dbscan search radius threshold
__C.POSTPROCESS.DBSCAN_EPS = 0.35
# Set the post process dbscan min samples threshold
__C.POSTPROCESS.DBSCAN_MIN_SAMPLES = 1000
