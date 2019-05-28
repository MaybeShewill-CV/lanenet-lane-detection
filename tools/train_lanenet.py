#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午9:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train_lanenet.py
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
from data_provider import lanenet_data_feed_pipline
from lanenet_model import lanenet
from tools import evaluate_model_utils

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Lanenet Dataset dir')
    parser.add_argument('-w', '--weights_path', type=str,
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-m', '--multi_gpus', type=args_str2bool, default=False,
                        nargs='?', const=True, help='Use multi gpus to train')
    parser.add_argument('--net_flag', type=str, default='vgg',
                        help='The net flag which determins the net\'s architecture')

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


def load_pretrained_weights(variables, pretrained_weights_path, sess):
    """

    :param variables:
    :param pretrained_weights_path:
    :param sess:
    :return:
    """
    assert ops.exists(pretrained_weights_path), '{:s} not exist'.format(pretrained_weights_path)

    pretrained_weights = np.load(
        './data/vgg16.npy', encoding='latin1').item()

    for vv in variables:
        weights_key = vv.name.split('/')[-3]
        if 'conv5' in weights_key:
            weights_key = '{:s}_{:s}'.format(weights_key.split('_')[0], weights_key.split('_')[1])
        try:
            weights = pretrained_weights[weights_key][0]
            _op = tf.assign(vv, weights)
            sess.run(_op)
        except Exception as _:
            continue

    return


def record_training_intermediate_result(gt_images, gt_binary_labels, gt_instance_labels,
                                        binary_seg_images, pix_embeddings, flag='train',
                                        save_dir='./tmp'):
    """
    record intermediate result during training process for monitoring
    :param gt_images:
    :param gt_binary_labels:
    :param gt_instance_labels:
    :param binary_seg_images:
    :param pix_embeddings:
    :param flag:
    :param save_dir:
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)

    for index, gt_image in enumerate(gt_images):
        gt_image_name = '{:s}_{:d}_gt_image.png'.format(flag, index + 1)
        gt_image_path = ops.join(save_dir, gt_image_name)
        gt_image = (gt_images[index] + 1.0) * 127.5
        cv2.imwrite(gt_image_path, np.array(gt_image, dtype=np.uint8))

        gt_binary_label_name = '{:s}_{:d}_gt_binary_label.png'.format(flag, index + 1)
        gt_binary_label_path = ops.join(save_dir, gt_binary_label_name)
        cv2.imwrite(gt_binary_label_path, np.array(gt_binary_labels[index][:, :, 0] * 255, dtype=np.uint8))

        gt_instance_label_name = '{:s}_{:d}_gt_instance_label.png'.format(flag, index + 1)
        gt_instance_label_path = ops.join(save_dir, gt_instance_label_name)
        cv2.imwrite(gt_instance_label_path, np.array(gt_instance_labels[index][:, :, 0], dtype=np.uint8))

        gt_binary_seg_name = '{:s}_{:d}_gt_binary_seg.png'.format(flag, index + 1)
        gt_binary_seg_path = ops.join(save_dir, gt_binary_seg_name)
        cv2.imwrite(gt_binary_seg_path, np.array(binary_seg_images[index] * 255, dtype=np.uint8))

        embedding_image_name = '{:s}_{:d}_pix_embedding.png'.format(flag, index + 1)
        embedding_image_path = ops.join(save_dir, embedding_image_name)
        embedding_image = pix_embeddings[index]
        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            embedding_image[:, :, i] = minmax_scale(embedding_image[:, :, i])
        embedding_image = np.array(embedding_image, np.uint8)
        cv2.imwrite(embedding_image_path, embedding_image)

    return


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def compute_net_gradients(gt_images, gt_binary_labels, gt_instance_labels,
                          net, optimizer=None):
    """
    Calculate gradients for single GPU
    :param gt_images:
    :param gt_binary_labels:
    :param gt_instance_labels:
    :param net:
    :param optimizer:
    :return:
    """

    compute_ret = net.compute_loss(
        input_tensor=gt_images, binary_label=gt_binary_labels,
        instance_label=gt_instance_labels, name='lanenet_model'
    )
    total_loss = compute_ret['total_loss']

    if optimizer is not None:
        grads = optimizer.compute_gradients(total_loss)
    else:
        grads = None

    return total_loss, grads


def train_lanenet(dataset_dir, weights_path=None, net_flag='vgg'):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='train'
    )
    val_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='val'
    )

    with tf.device('/gpu:1'):
        # set lanenet
        train_net = lanenet.LaneNet(net_flag=net_flag, phase='train', reuse=False)
        val_net = lanenet.LaneNet(net_flag=net_flag, phase='val', reuse=True)

        # set compute graph node for training
        train_images, train_binary_labels, train_instance_labels = train_dataset.inputs(
            CFG.TRAIN.BATCH_SIZE, 1
        )

        train_compute_ret = train_net.compute_loss(
            input_tensor=train_images, binary_label=train_binary_labels,
            instance_label=train_instance_labels, name='lanenet_model'
        )
        train_total_loss = train_compute_ret['total_loss']
        train_binary_seg_loss = train_compute_ret['binary_seg_loss']
        train_disc_loss = train_compute_ret['discriminative_loss']
        train_pix_embedding = train_compute_ret['instance_seg_logits']

        train_prediction_logits = train_compute_ret['binary_seg_logits']
        train_prediction_score = tf.nn.softmax(logits=train_prediction_logits)
        train_prediction = tf.argmax(train_prediction_score, axis=-1)

        train_accuracy = evaluate_model_utils.calculate_model_precision(
            train_compute_ret['binary_seg_logits'], train_binary_labels
        )
        train_fp = evaluate_model_utils.calculate_model_fp(
            train_compute_ret['binary_seg_logits'], train_binary_labels
        )
        train_fn = evaluate_model_utils.calculate_model_fn(
            train_compute_ret['binary_seg_logits'], train_binary_labels
        )
        train_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=train_prediction
        )
        train_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=train_pix_embedding
        )

        train_cost_scalar = tf.summary.scalar(
            name='train_cost', tensor=train_total_loss
        )
        train_accuracy_scalar = tf.summary.scalar(
            name='train_accuracy', tensor=train_accuracy
        )
        train_binary_seg_loss_scalar = tf.summary.scalar(
            name='train_binary_seg_loss', tensor=train_binary_seg_loss
        )
        train_instance_seg_loss_scalar = tf.summary.scalar(
            name='train_instance_seg_loss', tensor=train_disc_loss
        )
        train_fn_scalar = tf.summary.scalar(
            name='train_fn', tensor=train_fn
        )
        train_fp_scalar = tf.summary.scalar(
            name='train_fp', tensor=train_fp
        )
        train_binary_seg_ret_img = tf.summary.image(
            name='train_binary_seg_ret', tensor=train_binary_seg_ret_for_summary
        )
        train_embedding_feats_ret_img = tf.summary.image(
            name='train_embedding_feats_ret', tensor=train_embedding_ret_for_summary
        )
        train_merge_summary_op = tf.summary.merge(
            [train_accuracy_scalar, train_cost_scalar, train_binary_seg_loss_scalar,
             train_instance_seg_loss_scalar, train_fn_scalar, train_fp_scalar,
             train_binary_seg_ret_img, train_embedding_feats_ret_img]
        )

        # set compute graph node for validation
        val_images, val_binary_labels, val_instance_labels = val_dataset.inputs(
            CFG.TRAIN.VAL_BATCH_SIZE, 1
        )

        val_compute_ret = val_net.compute_loss(
            input_tensor=val_images, binary_label=val_binary_labels,
            instance_label=val_instance_labels, name='lanenet_model'
        )
        val_total_loss = val_compute_ret['total_loss']
        val_binary_seg_loss = val_compute_ret['binary_seg_loss']
        val_disc_loss = val_compute_ret['discriminative_loss']
        val_pix_embedding = val_compute_ret['instance_seg_logits']

        val_prediction_logits = val_compute_ret['binary_seg_logits']
        val_prediction_score = tf.nn.softmax(logits=val_prediction_logits)
        val_prediction = tf.argmax(val_prediction_score, axis=-1)

        val_accuracy = evaluate_model_utils.calculate_model_precision(
            val_compute_ret['binary_seg_logits'], val_binary_labels
        )
        val_fp = evaluate_model_utils.calculate_model_fp(
            val_compute_ret['binary_seg_logits'], val_binary_labels
        )
        val_fn = evaluate_model_utils.calculate_model_fn(
            val_compute_ret['binary_seg_logits'], val_binary_labels
        )
        val_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=val_prediction
        )
        val_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=val_pix_embedding
        )

        val_cost_scalar = tf.summary.scalar(
            name='val_cost', tensor=val_total_loss
        )
        val_accuracy_scalar = tf.summary.scalar(
            name='val_accuracy', tensor=val_accuracy
        )
        val_binary_seg_loss_scalar = tf.summary.scalar(
            name='val_binary_seg_loss', tensor=val_binary_seg_loss
        )
        val_instance_seg_loss_scalar = tf.summary.scalar(
            name='val_instance_seg_loss', tensor=val_disc_loss
        )
        val_fn_scalar = tf.summary.scalar(
            name='val_fn', tensor=val_fn
        )
        val_fp_scalar = tf.summary.scalar(
            name='val_fp', tensor=val_fp
        )
        val_binary_seg_ret_img = tf.summary.image(
            name='val_binary_seg_ret', tensor=val_binary_seg_ret_for_summary
        )
        val_embedding_feats_ret_img = tf.summary.image(
            name='val_embedding_feats_ret', tensor=val_embedding_ret_for_summary
        )
        val_merge_summary_op = tf.summary.merge(
            [val_accuracy_scalar, val_cost_scalar, val_binary_seg_loss_scalar,
             val_instance_seg_loss_scalar, val_fn_scalar, val_fp_scalar,
             val_binary_seg_ret_img, val_embedding_feats_ret_img]
        )

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
                learning_rate=learning_rate, momentum=CFG.TRAIN.MOMENTUM).minimize(
                loss=train_total_loss,
                var_list=tf.trainable_variables(),
                global_step=global_step
            )

    # Set tf model save path
    model_save_dir = 'model/tusimple_lanenet_{:s}'.format(net_flag)
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)
    saver = tf.train.Saver()

    # Set tf summary save path
    tboard_save_path = 'tboard/tusimple_lanenet_{:s}'.format(net_flag)
    os.makedirs(tboard_save_path, exist_ok=True)

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

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        if net_flag == 'vgg' and weights_path is None:
            load_pretrained_weights(tf.trainable_variables(), './data/vgg16.npy', sess)

        train_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            _, train_c, train_accuracy_figure, train_fn_figure, train_fp_figure, \
                lr, train_summary, train_binary_loss, \
                train_instance_loss, train_embeddings, train_binary_seg_imgs, train_gt_imgs, \
                train_binary_gt_labels, train_instance_gt_labels = \
                sess.run([optimizer, train_total_loss, train_accuracy, train_fn, train_fp,
                          learning_rate, train_merge_summary_op, train_binary_seg_loss,
                          train_disc_loss, train_pix_embedding, train_prediction,
                          train_images, train_binary_labels, train_instance_labels])

            if math.isnan(train_c) or math.isnan(train_binary_loss) or math.isnan(train_instance_loss):
                log.error('cost is: {:.5f}'.format(train_c))
                log.error('binary cost is: {:.5f}'.format(train_binary_loss))
                log.error('instance cost is: {:.5f}'.format(train_instance_loss))
                return

            if epoch % 100 == 0:
                record_training_intermediate_result(
                    gt_images=train_gt_imgs, gt_binary_labels=train_binary_gt_labels,
                    gt_instance_labels=train_instance_gt_labels, binary_seg_images=train_binary_seg_imgs,
                    pix_embeddings=train_embeddings
                )
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                         ' lr= {:6f} mean_cost_time= {:5f}s '.
                         format(epoch + 1, train_c, train_binary_loss, train_instance_loss, train_accuracy_figure,
                                train_fp_figure, train_fn_figure, lr, np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            # validation part
            val_c, val_accuracy_figure, val_fn_figure, val_fp_figure, \
                val_summary, val_binary_loss, val_instance_loss, \
                val_embeddings, val_binary_seg_imgs, val_gt_imgs, \
                val_binary_gt_labels, val_instance_gt_labels = \
                sess.run([val_total_loss, val_accuracy, val_fn, val_fp,
                          val_merge_summary_op, val_binary_seg_loss,
                          val_disc_loss, val_pix_embedding, val_prediction,
                          val_images, val_binary_labels, val_instance_labels])

            if math.isnan(val_c) or math.isnan(val_binary_loss) or math.isnan(val_instance_loss):
                log.error('cost is: {:.5f}'.format(val_c))
                log.error('binary cost is: {:.5f}'.format(val_binary_loss))
                log.error('instance cost is: {:.5f}'.format(val_instance_loss))
                return

            if epoch % 100 == 0:
                record_training_intermediate_result(
                    gt_images=val_gt_imgs, gt_binary_labels=val_binary_gt_labels,
                    gt_instance_labels=val_instance_gt_labels, binary_seg_images=val_binary_seg_imgs,
                    pix_embeddings=val_embeddings, flag='val'
                )

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=val_summary, global_step=epoch)

            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, val_c, val_binary_loss, val_instance_loss, val_accuracy_figure,
                                val_fp_figure, val_fn_figure, np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=global_step)

    return


def train_lanenet_multi_gpu(dataset_dir, weights_path=None, net_flag='vgg'):
    """
    train lanenet with multi gpu
    :param dataset_dir:
    :param weights_path:
    :param net_flag:
    :return:
    """
    # set lanenet dataset
    train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='train'
    )
    val_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='val'
    )

    # set lanenet
    train_net = lanenet.LaneNet(net_flag=net_flag, phase='train', reuse=False)
    val_net = lanenet.LaneNet(net_flag=net_flag, phase='val', reuse=True)

    # set compute graph node
    train_images, train_binary_labels, train_instance_labels = train_dataset.inputs(
        CFG.TRAIN.BATCH_SIZE, 1
    )
    val_images, val_binary_labels, val_instance_labels = val_dataset.inputs(
        CFG.TRAIN.VAL_BATCH_SIZE, 1
    )

    # set average container
    tower_grads = []
    train_tower_loss = []
    val_tower_loss = []
    batchnorm_updates = None
    train_summary_op_updates = None

    # set lr
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(
        learning_rate=CFG.TRAIN.LEARNING_RATE,
        global_step=global_step,
        decay_steps=CFG.TRAIN.EPOCHS,
        power=0.9
    )

    # set optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=CFG.TRAIN.MOMENTUM
    )

    # set distributed train op
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(CFG.TRAIN.GPU_NUM):
            with tf.device('/gpu:{:d}'.format(i)):
                with tf.name_scope('tower_{:d}'.format(i)) as _:
                    train_loss, grads = compute_net_gradients(
                        train_images, train_binary_labels, train_instance_labels, train_net, optimizer
                    )

                    # Only use the mean and var in the first gpu tower to update the parameter
                    if i == 0:
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)
                    tower_grads.append(grads)
                    train_tower_loss.append(train_loss)

                with tf.name_scope('validation_{:d}'.format(i)) as _:
                    val_loss, _ = compute_net_gradients(
                        val_images, val_binary_labels, val_instance_labels, val_net, optimizer)
                    val_tower_loss.append(val_loss)

    grads = average_gradients(tower_grads)
    avg_train_loss = tf.reduce_mean(train_tower_loss)
    avg_val_loss = tf.reduce_mean(val_tower_loss)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.TRAIN.MOVING_AVERAGE_DECAY, num_updates=global_step)
    variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all the op needed for training
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # Set tf summary save path
    tboard_save_path = 'tboard/tusimple_lanenet_multi_gpu_{:s}'.format(net_flag)
    os.makedirs(tboard_save_path, exist_ok=True)

    summary_writer = tf.summary.FileWriter(tboard_save_path)

    avg_train_loss_scalar = tf.summary.scalar(
        name='average_train_loss', tensor=avg_train_loss
    )
    avg_val_loss_scalar = tf.summary.scalar(
        name='average_val_loss', tensor=avg_val_loss
    )
    learning_rate_scalar = tf.summary.scalar(
        name='learning_rate_scalar', tensor=learning_rate
    )

    train_merge_summary_op = tf.summary.merge(
        [avg_train_loss_scalar, learning_rate_scalar] + train_summary_op_updates
    )
    val_merge_summary_op = tf.summary.merge([avg_val_loss_scalar])

    # set tensorflow saver
    saver = tf.train.Saver()
    model_save_dir = 'model/tusimple_lanenet_multi_gpu_{:s}'.format(net_flag)
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # set sess config
    sess_config = tf.ConfigProto(device_count={'GPU': CFG.TRAIN.GPU_NUM}, allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    sess = tf.Session(config=sess_config)

    summary_writer.add_graph(sess.graph)

    with sess.as_default():

        tf.train.write_graph(
            graph_or_graph_def=sess.graph, logdir='',
            name='{:s}/lanenet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        train_cost_time_mean = []
        val_cost_time_mean = []

        for epoch in range(train_epochs):

            # training part
            t_start = time.time()

            _, train_loss_value, train_summary, lr = \
                sess.run(
                    fetches=[train_op, avg_train_loss,
                             train_merge_summary_op, learning_rate]
                )

            if math.isnan(train_loss_value):
                log.error('Train loss is nan')
                return

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            t_start_val = time.time()

            val_loss_value, val_summary = \
                sess.run(fetches=[avg_val_loss, val_merge_summary_op])

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch_Train: {:d} total_loss= {:6f} '
                         'lr= {:6f} mean_cost_time= {:5f}s '.
                         format(epoch + 1,
                                train_loss_value,
                                lr,
                                np.mean(train_cost_time_mean))
                         )
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1,
                                val_loss_value,
                                np.mean(val_cost_time_mean))
                         )
                val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if CFG.TRAIN.GPU_NUM < 2:
        args.use_multi_gpu = False

    # train lanenet
    if not args.multi_gpus:
        train_lanenet(args.dataset_dir, args.weights_path, net_flag=args.net_flag)
    else:
        train_lanenet_multi_gpu(args.dataset_dir, args.weights_path, net_flag=args.net_flag)
