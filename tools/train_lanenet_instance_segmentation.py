#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:23
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_lanenet_instance_segmentation.py.py
# @IDE: PyCharm Community Edition
"""
训练LaneNet实例分割子模型
"""
import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf

try:
    from cv2 import cv2
except ImportError:
    pass

from config import global_config
from lanenet_model import lanenet_instance_segmentation
from data_provider import lanenet_data_processor

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')

    return parser.parse_args()


def train_net(dataset_dir, weights_path=None, net_flag='vgg'):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file)

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[None, CFG.TRAIN.IMG_HEIGHT,
                                         CFG.TRAIN.IMG_WIDTH, 3],
                                  name='input_tensor')
    binary_label_tensor = tf.placeholder(dtype=tf.int64,
                                         shape=[None, CFG.TRAIN.IMG_HEIGHT,
                                                CFG.TRAIN.IMG_WIDTH, 1],
                                         name='binary_input_label')
    instance_label_tensor = tf.placeholder(dtype=tf.float32,
                                           shape=[None, CFG.TRAIN.IMG_HEIGHT,
                                                  CFG.TRAIN.IMG_WIDTH],
                                           name='instance_input_label')
    phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

    net = lanenet_instance_segmentation.LaneNetInstanceSeg(net_flag=net_flag, phase=phase)

    # calculate the loss
    compute_ret = net.compute_loss(input_tensor=input_tensor, label=instance_label_tensor, name='lanenet_loss')
    total_loss = compute_ret['total_loss']
    loss_var = compute_ret['loss_var']
    loss_dist = compute_ret['loss_dist']
    loss_reg = compute_ret['loss_reg']
    pix_embedding = compute_ret['embedding']

    # calculate the accuracy
    out_logits = compute_ret['binary_seg_logits']
    out_logits = tf.nn.softmax(logits=out_logits)
    out = tf.argmax(out_logits, axis=-1)
    out = tf.expand_dims(out, axis=-1)
    accuracy = tf.add(binary_label_tensor, -1 * out)
    accuracy = tf.count_nonzero(accuracy, axis=[1, 2, 3])
    accuracy = tf.add(tf.constant(1, dtype=tf.float64),
                      -1 * tf.divide(accuracy,
                                     CFG.TRAIN.IMG_HEIGHT * CFG.TRAIN.IMG_WIDTH))
    accuracy = tf.reduce_mean(accuracy, axis=0)

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
    model_save_dir = 'model/tusimple_lanenet_instance'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary
    tboard_save_path = 'tboard/tusimple_lanenet_instance/{:s}'.format(net_flag)
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar, learning_rate_scalar])
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar])

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/encoder_decoder.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model check point{:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # 加载预训练参数
        if net_flag == 'vgg':
            pretrained_weights = np.load(
                '/home/baidu/Silly_Project/ICode/baidu/beec/semantic-road-estimation/data/vgg16.npy',
                encoding='latin1').item()

            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

        train_cost_time_mean = []
        val_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            gt_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)
            gt_imgs = [cv2.resize(tmp,
                                  dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                  dst=tmp,
                                  interpolation=cv2.INTER_LINEAR)
                       for tmp in gt_imgs]
            gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
            binary_gt_labels = [cv2.resize(tmp,
                                           dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                           dst=tmp,
                                           interpolation=cv2.INTER_NEAREST)
                                for tmp in binary_gt_labels]
            binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]
            instance_gt_labels = [cv2.resize(tmp,
                                             dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                             dst=tmp,
                                             interpolation=cv2.INTER_NEAREST)
                                  for tmp in instance_gt_labels]
            phase_train = 'train'

            _, c, train_accuracy, train_summary, l_var, l_dist, l_reg, embedding = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          loss_var,
                          loss_dist,
                          loss_reg, pix_embedding],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label_tensor: binary_gt_labels,
                                    instance_label_tensor: instance_gt_labels,
                                    phase: phase_train})

            if math.isnan(c) or math.isnan(l_dist):
                cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('nan_label.png', instance_gt_labels[0])
                cv2.imwrite('nan_embedding.png', embedding[0])
                return
            cv2.imwrite('image.png', gt_imgs[0] + VGG_MEAN)
            cv2.imwrite('label.png', instance_gt_labels[0])
            cv2.imwrite('embedding.png', embedding[0])

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val \
                = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
            gt_imgs_val = [cv2.resize(tmp,
                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      dst=tmp,
                                      interpolation=cv2.INTER_LINEAR)
                           for tmp in gt_imgs_val]
            gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
            binary_gt_labels_val = [cv2.resize(tmp,
                                               dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                               dst=tmp)
                                    for tmp in binary_gt_labels_val]
            binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]
            instance_gt_labels_val = [cv2.resize(tmp,
                                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                 dst=tmp,
                                                 interpolation=cv2.INTER_NEAREST)
                                      for tmp in instance_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            c_val, val_summary, val_accuracy, val_l_var, val_l_dist, val_l_reg = \
                sess.run([total_loss, val_merge_summary_op, accuracy, loss_var, loss_dist, loss_reg],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label_tensor: binary_gt_labels_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    phase: phase_val})

            cv2.imwrite('test_image.png', gt_imgs_val[0] + VGG_MEAN)
            cv2.imwrite('test_label.png', instance_gt_labels_val[0])

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} l_var= {:6f} l_dist={:6f} l_reg={:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, l_var, l_dist, l_reg, train_accuracy,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} l_var= {:6f} l_dist={:6f} l_reg={:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val, val_l_var, val_l_dist, val_l_reg, val_accuracy,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)
