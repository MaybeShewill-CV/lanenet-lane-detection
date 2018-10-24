#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-29 下午7:41
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_encoder_decoder.py
# @IDE: PyCharm Community Edition
"""
测试encoder decoder模型
"""
import argparse
import os.path as ops
import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from encoder_decoder_model import semantic_segmentation
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The test image path')
    parser.add_argument('--net', type=str, help='The base net model', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The weights path')

    return parser.parse_args()


def test_net(image_path, weights_path, net_flag):
    """

    :param image_path:
    :param weights_path:
    :param net_flag:
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = image - [103.939, 116.779, 123.68]
    image = np.expand_dims(image, axis=0)

    with tf.device('/gpu:1'):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf.constant('train', dtype=tf.string)

        net = semantic_segmentation.SemanticSeg(net_flag=net_flag, phase=phase_tensor)
        with tf.variable_scope('kitti_loss'):
            net_out = net.build_model(input_tensor=input_tensor, name='inference')

        out_logits = tf.nn.softmax(net_out['logits'])
        out = tf.argmax(out_logits, axis=-1)
        out = tf.squeeze(input=out, axis=0)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with tf.device('/gpu:1'):
        with sess.as_default():

            saver.restore(sess=sess, save_path=weights_path)

            predict_map = sess.run(out, feed_dict={input_tensor: image})
            predict_color = np.zeros([predict_map.shape[0], predict_map.shape[1], 3], dtype=np.uint8)
            idx = np.where(predict_map[:, :] == 1)
            print(idx)
            predict_color[idx] = [0, 255, 0]
            predict_color = cv2.resize(predict_color, dsize=(512, 256))
            cv2.imwrite('mask_prediction.png', predict_color)

            plt.figure('predict image: {:s}'.format(ops.split(image_path)[1]))
            plt.suptitle('predict image: {:s}'.format(ops.split(image_path)[1]))
            plt.imshow(predict_color[:, :, (2, 1, 0)])

            plt.figure('origin image: {:s}'.format(ops.split(image_path)[1]))
            plt.suptitle('origin image: {:s}'.format(ops.split(image_path)[1]))
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.show()

        sess.close()

    return


def test_net_batch(src_image_dir, weights_path, net_flag, output_dir):
    """

    :param src_image_dir:
    :param weights_path:
    :param net_flag:
    :param output_dir:
    :return:
    """
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(src_image_dir), recursive=True)

    with tf.device('/gpu:1'):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 512, 1024, 3], name='input_tensor')
        phase_tensor = tf.constant('train', dtype=tf.string)

        net = semantic_segmentation.SemanticSeg(net_flag=net_flag, phase=phase_tensor)
        with tf.variable_scope('kitti_loss'):
            net_out = net.build_model(input_tensor=input_tensor, name='inference')

        out_logits = tf.nn.softmax(net_out['logits'])
        out = tf.argmax(out_logits, axis=-1)
        out = tf.squeeze(input=out, axis=0)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placemeat=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with tf.device('/gpu:1'):
        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            for image_path in image_path_list:

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = image[0:1350, :, :]
                image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_CUBIC)
                image_vis = image
                image = image - [103.939, 116.779, 123.68]
                image = np.expand_dims(image, axis=0)

                predict_map = sess.run(out, feed_dict={input_tensor: image})
                predict_map_vis = predict_map
                idx_vis = np.where(predict_map[:, :] == 1)
                predict_map = np.array(predict_map, np.uint8)
                predict_map = cv2.resize(predict_map, (2448, 1350), interpolation=cv2.INTER_LANCZOS4)
                idx = np.where(predict_map[:, :] == 1)
                predict_color = np.zeros([predict_map.shape[0], predict_map.shape[1], 3], dtype=np.uint8)
                predict_color[idx] = [0, 255, 0]
                predict_color_vis = np.zeros([predict_map_vis.shape[0], predict_map_vis.shape[1], 3],
                                             dtype=np.uint8)
                predict_color_vis[idx_vis] = [0, 255, 0]
                image_vis[idx_vis] = [0, 255, 0]
                # predict_color = cv2.resize(predict_color, dsize=(512, 256), interpolation=cv2.INTER_NEAREST)

                mask_image = cv2.addWeighted(image_vis, 1.0, predict_color_vis, 1, 0)
                mask_image_name = ops.split(image_path)[1]
                mask_image_path = ops.join(output_dir, mask_image_name)
                cv2.imwrite(mask_image_path, image_vis)
                print('Draw mask image {:s} complete'.format(mask_image_path))

                image_name = ops.split(image_path)[1]
                coord_file_name = image_name.split('.')[0] + '.txt'
                coord_file_path = ops.join(output_dir, coord_file_name)
                with open(coord_file_path, 'w') as file:
                    for i in range(len(idx[0])):
                        pty = idx[0][i]
                        ptx = idx[1][i]
                        file.write('{:d} {:d}'.format(ptx, pty) + '\n')

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test encoder decoder
    # test_net(args.image_path, args.weights_path, args.net)
    test_net_batch('/media/baidu/Data/高精图像质检/20180716t090055_gm/20180716T091458',
                   '/home/baidu/Silly_Project/ICode/baidu/beec/semantic-road-estimation/model/tusimple_lane/tusimple_lane_vgg_2018-05-18-19-26-50.ckpt-200000',
                   'vgg',
                   '/media/baidu/Data/高精图像质检/20180716t090055_gm/lane_mask_ret')
