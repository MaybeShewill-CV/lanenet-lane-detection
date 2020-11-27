#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pdb
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='The video path or the src video save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def num_to_BGR(num):
    numbers = {
        0: (255, 0, 0),#blue
        1: (0, 255, 255),  # yellow
        2: (0, 0, 255),  # red
        3: (0, 255, 0),#green
        4: (255, 255, 0),#cyan
    }
    return numbers.get(num, None)

def gen_color_img(binary, instance, n_sticks):
    p_sem_pred = []
    for sp in binary:
        p_sem_pred.append(ndi.morphology.binary_fill_holes(sp > 0.5))
    log.info(binary.shape)
    log.info(len(p_sem_pred))
    log.info(instance.shape)
    embeddings = instance[p_sem_pred, :]
    log.info(embeddings.shape)
    clustering = KMeans(n_sticks).fit(embeddings)
    labels = clustering.labels_
    log.info( 'labels_len: %d', len(labels))
    instance_mask = np.zeros_like(p_sem_pred, dtype=np.uint8)
    for i in range(n_sticks):
        lbl = np.zeros_like(labels, dtype=np.uint8)
        lbl[labels == i] = i + 1
        instance_mask[p_sem_pred] += lbl
    mask = instance_mask
   
    log.info('mask min: %f, max: %f', mask.min(), mask.max()) 
    ins_color_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    n_ins = len(np.unique(mask)) - 1
    log.info('n_ins: %d', n_ins)
    for i in range(n_ins):
        ins_color_img[mask == i + 1] = i*30
    return ins_color_img


def test_lanenet(video_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(video_path), '{:s} not exist'.format(video_path)
       
    obj_w = 1920
    obj_h = 320
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, obj_h, obj_w, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info('width: %d, height: %d', width, height)
        out = cv2.VideoWriter('out.mp4', fourcc, 15, (obj_w, obj_h), isColor=True)
        nn = 1000
        #while(cap.isOpened() and nn >= 0):
        while(cap.isOpened()):
            nn = nn - 1
            ret, image = cap.read()
            if ret:
                #image = cv2.imread('/workspace/lanenet-lane-detection/1410486660.jpg')
                # image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
                image = image[442:762, :]
                image_vis = image.copy()
                image = image / 127.5 - 1.0

                t_start = time.time()
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
                t_cost = time.time() - t_start
                LOG.info('Single image inference cost time: {:.5f}s'.format(t_cost))
                #log.info(instance_seg_image.shape)

                #mask_image = gen_color_img(binary_seg_image[0], instance_seg_image[0], 5)
                #cv2.imwrite('test.png', mask_image)
                #return
                postprocess_result = postprocessor.postprocess2(
                    binary_seg_result = binary_seg_image[0],
                    instance_seg_result = instance_seg_image[0],
                    source_image = image_vis
                )
                binary_image = postprocess_result['binary_image']
                instance_image = postprocess_result['instance_image']
                '''
                np.set_printoptions(suppress = True, precision = 2, threshold = np.inf)
                print('before: ')
                print(instance_seg_image[0])
                for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
                    instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
                embedding_image = np.array(instance_seg_image[0], np.uint8)
                print('after: ')
                print(embedding_image)
                '''
                out.write(np.uint8(instance_image))
                #out.write(np.uint8(mask_image))
                #cv2.imwrite('test.png', binary_image)
                #return
            else:
                break
        cap.release()
        out.release() 

    sess.close()
    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.video_path, args.weights_path)
