#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午9:33
# @Author  : LuoYao
# @Site    : ICode
# @File    : train_lanenet_new.py
# @IDE: PyCharm
"""
Train lanenet script
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

from config import global_config
from lanenet_model import lanenet
from data_provider import lanenet_data_feed_pipline
from tools import evaluate_model_performance

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


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def train_net(dataset_dir, weights_path=None, net_flag='vgg'):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='train'
    )

    with tf.device('/gpu:1'):
        # set lanenet
        train_net = lanenet.LaneNet(net_flag=net_flag, phase='train', reuse=False)

        # set train sample and val sample
        train_images, train_binary_labels, train_instance_labels = train_dataset.inputs(
            CFG.TRAIN.BATCH_SIZE, 1
        )

        # calculate the loss
        compute_ret = train_net.compute_loss(
            input_tensor=train_images, binary_label=train_binary_labels,
            instance_label=train_instance_labels, name='lanenet_model'
        )
        total_loss = compute_ret['total_loss']
        binary_seg_loss = compute_ret['binary_seg_loss']
        disc_loss = compute_ret['discriminative_loss']
        pix_embedding = compute_ret['instance_seg_logits']

        # calculate the accuracy
        out_logits = compute_ret['binary_seg_logits']
        out_logits = tf.nn.softmax(logits=out_logits)
        out_logits_out = tf.argmax(out_logits, axis=-1)

        accuracy = evaluate_model_performance.calculate_model_precision(
            compute_ret['binary_seg_logits'], train_binary_labels)
        fp = evaluate_model_performance.calculate_model_fp(
            compute_ret['binary_seg_logits'], train_binary_labels)
        fn = evaluate_model_performance.calculate_model_fn(
            compute_ret['binary_seg_logits'], train_binary_labels)

        # calculate the binary segmentation result for summary
        binary_seg_ret_for_summary = evaluate_model_performance.get_image_summary(img=out_logits_out)
        embedding_ret_for_summary = evaluate_model_performance.get_image_summary(img=pix_embedding)

        # set optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
            learning_rate=CFG.TRAIN.LEARNING_RATE,
            global_step=global_step,
            decay_steps=CFG.TRAIN.EPOCHS,
            power=0.9
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(
                loss=total_loss,
                var_list=tf.trainable_variables(),
                global_step=global_step)
            # optimizer = tf.train.AdamOptimizer(
            #     learning_rate=learning_rate, epsilon=1e-8).minimize(
            #     loss=total_loss,
            #     var_list=tf.trainable_variables(),
            #     global_step=global_step)

    # Set tf saver
    saver = tf.train.Saver()
    model_save_dir = 'model/tusimple_lanenet_{:s}'.format(net_flag)
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary
    tboard_save_path = 'tboard/tusimple_lanenet_{:s}'.format(net_flag)
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)

    # Set tf scalar
    train_cost_scalar = tf.summary.scalar(name='train_cost',
                                          tensor=total_loss)

    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy',
                                              tensor=accuracy)

    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss',
                                                     tensor=binary_seg_loss)

    train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss',
                                                       tensor=disc_loss)

    learning_rate_scalar = tf.summary.scalar(name='learning_rate',
                                             tensor=learning_rate)

    train_fn_scalar = tf.summary.scalar(name='train_fn',
                                        tensor=fn)

    train_fp_scalar = tf.summary.scalar(name='train_fp',
                                        tensor=fp)

    train_binary_seg_ret_img = tf.summary.image(name='train_binary_seg_ret',
                                                tensor=binary_seg_ret_for_summary)

    train_embedding_feats_ret_img = tf.summary.image(name='train_embedding_feats_ret',
                                                     tensor=embedding_ret_for_summary)
    # Merge tf summary op
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar,
                                               train_cost_scalar,
                                               learning_rate_scalar,
                                               train_binary_seg_loss_scalar,
                                               train_instance_seg_loss_scalar,
                                               train_fn_scalar,
                                               train_fp_scalar,
                                               train_binary_seg_ret_img,
                                               train_embedding_feats_ret_img])

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
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
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # 加载预训练参数
        if net_flag == 'vgg' and weights_path is None:
            pretrained_weights = np.load(
                './data/vgg16.npy',
                encoding='latin1').item()

            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                if 'conv5' in weights_key:
                    weights_key = '{:s}_{:s}'.format(weights_key.split('_')[0], weights_key.split('_')[1])
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

        train_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            _, c, train_accuracy, train_fn, train_fp, lr, train_summary, binary_loss, \
            instance_loss, embedding, binary_seg_img, gt_imgs, binary_gt_labels, instance_gt_labels = \
                sess.run([optimizer,
                          total_loss,
                          accuracy,
                          fn,
                          fp,
                          learning_rate,
                          train_merge_summary_op,
                          binary_seg_loss,
                          disc_loss,
                          pix_embedding,
                          out_logits_out,
                          train_images,
                          train_binary_labels,
                          train_instance_labels])

            if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
                log.error('cost is: {:.5f}'.format(c))
                log.error('binary cost is: {:.5f}'.format(binary_loss))
                log.error('instance cost is: {:.5f}'.format(instance_loss))
                return

            if epoch % 100 == 0:
                cv2.imwrite('image.png', np.array((gt_imgs[0] + 1.0) * 127.5, dtype=np.uint8))
                cv2.imwrite('binary_label.png', np.array(binary_gt_labels[0] * 255, dtype=np.uint8))
                cv2.imwrite('instance_label.png', np.array(instance_gt_labels[0], dtype=np.uint8))
                cv2.imwrite('binary_seg_img.png', np.array(binary_seg_img[0] * 255, dtype=np.uint8))

                for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
                    embedding[0][:, :, i] = minmax_scale(embedding[0][:, :, i])
                embedding_image = np.array(embedding[0], np.uint8)
                cv2.imwrite('embedding.png', embedding_image)

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary,
                                       global_step=epoch)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                         ' lr= {:6f} mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, binary_loss, instance_loss, train_accuracy,
                                train_fp, train_fn, lr, np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)
