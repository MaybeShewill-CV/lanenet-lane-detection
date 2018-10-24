#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 下午3:04
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_lanenet_hnet.py
# @IDE: PyCharm Community Edition
"""
训练LaneNet的HNet部分
"""
import argparse
import os
import os.path as ops
import glob
import time

import tensorflow as tf
import glog as log
import numpy as np
import math
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from config import global_config
from lanenet_model import lanenet_hnet_model
from data_provider import lanenet_hnet_data_processor

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='The origin tusimple dataset dir')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')

    return parser.parse_args()


def train_net(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path
    :return:
    """
    assert ops.exists(dataset_dir), '{:s} not exist'.format(dataset_dir)

    json_file_list = glob.glob('{:s}/*.json'.format(dataset_dir))
    json_file_list = [tmp for tmp in json_file_list if 'test' not in tmp]
    if not json_file_list:
        log.error('Can not find any suitable label json file')
        return
    train_dataset = lanenet_hnet_data_processor.DataSet(json_file_list)

    input_image_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 64, 128, 3], name='input_image_tensor')
    input_label_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='input_label_tensor')

    phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

    net = lanenet_hnet_model.LaneNetHNet(phase=phase)

    # 计算hnet损失
    total_loss = net.compute_loss(input_tensor=input_image_tensor, gt_label_pts=input_label_tensor, name='hnet')

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                               5000, 0.96, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=
                                           learning_rate).minimize(loss=total_loss,
                                                                   var_list=tf.trainable_variables(),
                                                                   global_step=global_step)
    # Set tf saver
    saver = tf.train.Saver()
    model_save_dir = 'model/tusimple_lanenet_hnet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_hnet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary
    tboard_save_path = 'tboard/tusimple_lanenet_hnet/hnet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([train_cost_scalar, learning_rate_scalar])

    summary_writer = tf.summary.FileWriter(tboard_save_path)

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_hnet.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model check point{:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        train_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            gt_imgs, gt_pts_labels = train_dataset.next_batch(1)
            gt_imgs = [cv2.resize(tmp,
                                  dsize=(128, 64),
                                  dst=tmp,
                                  interpolation=cv2.INTER_LINEAR)
                       for tmp in gt_imgs]
            gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]

            lane_index = 0
            for lane_pts in gt_pts_labels[0]:
                lane_pts = np.concatenate((lane_pts, np.ones(shape=[len(lane_pts), 1])), axis=1)

                phase_train = 'train'

                _, c, train_summary = sess.run([optimizer, total_loss, train_merge_summary_op],
                                               feed_dict={input_image_tensor: gt_imgs,
                                                          input_label_tensor: lane_pts,
                                                          phase: phase_train})
                if math.isnan(c):
                    log.info('Loss error: loss is nan')
                    return

                cost_time = time.time() - t_start
                train_cost_time_mean.append(cost_time)
                summary_writer.add_summary(summary=train_summary, global_step=epoch)

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    log.info('[Epoch/Lane]: [{:d}/{:d}] total_loss= {:6f} cost_time= {:.6f}'.
                             format(epoch + 1, lane_index + 1, c, np.mean(train_cost_time_mean)))
                    train_cost_time_mean.clear()

                if epoch % 2000 == 0:
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                lane_index += 1
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train hnet
    train_net(dataset_dir=args.dataset_dir)
