#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 下午1:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : tusimple_lanenet_single_gpu_trainner.py
# @IDE: PyCharm
"""
Tusimple lanenet trainner
"""
import os
import os.path as ops
import shutil
import time
import math

import numpy as np
import tensorflow as tf
import loguru
import tqdm

from data_provider import lanenet_data_feed_pipline
from lanenet_model import lanenet

LOG = loguru.logger


class LaneNetTusimpleTrainer(object):
    """
    init lanenet single gpu trainner
    """

    def __init__(self, cfg):
        """
        initialize lanenet trainner
        """
        self._cfg = cfg
        # define solver params and dataset
        self._train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(flags='train')
        self._steps_per_epoch = len(self._train_dataset)

        self._model_name = '{:s}_{:s}'.format(self._cfg.MODEL.FRONT_END, self._cfg.MODEL.MODEL_NAME)

        self._train_epoch_nums = self._cfg.TRAIN.EPOCH_NUMS
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._snapshot_epoch = self._cfg.TRAIN.SNAPSHOT_EPOCH
        self._model_save_dir = ops.join(self._cfg.TRAIN.MODEL_SAVE_DIR, self._model_name)
        self._tboard_save_dir = ops.join(self._cfg.TRAIN.TBOARD_SAVE_DIR, self._model_name)
        self._enable_miou = self._cfg.TRAIN.COMPUTE_MIOU.ENABLE
        if self._enable_miou:
            self._record_miou_epoch = self._cfg.TRAIN.COMPUTE_MIOU.EPOCH
        self._input_tensor_size = [int(tmp) for tmp in self._cfg.AUG.TRAIN_CROP_SIZE]

        self._init_learning_rate = self._cfg.SOLVER.LR
        self._moving_ave_decay = self._cfg.SOLVER.MOVING_AVE_DECAY
        self._momentum = self._cfg.SOLVER.MOMENTUM
        self._lr_polynimal_decay_power = self._cfg.SOLVER.LR_POLYNOMIAL_POWER
        self._optimizer_mode = self._cfg.SOLVER.OPTIMIZER.lower()

        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            self._initial_weight = self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.SNAPSHOT_PATH
        else:
            self._initial_weight = None
        if self._cfg.TRAIN.WARM_UP.ENABLE:
            self._warmup_epoches = self._cfg.TRAIN.WARM_UP.EPOCH_NUMS
            self._warmup_init_learning_rate = self._init_learning_rate / 1000.0
        else:
            self._warmup_epoches = 0

        # define tensorflow session
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = self._cfg.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = self._cfg.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self._sess = tf.Session(config=sess_config)

        # define graph input tensor
        with tf.variable_scope(name_or_scope='graph_input_node'):
            self._input_src_image, self._input_binary_label_image, self._input_instance_label_image = \
                self._train_dataset.next_batch(batch_size=self._batch_size)

        # define model loss
        self._model = lanenet.LaneNet(phase='train', cfg=self._cfg)
        loss_set = self._model.compute_loss(
            input_tensor=self._input_src_image,
            binary_label=self._input_binary_label_image,
            instance_label=self._input_instance_label_image,
            name='LaneNet',
            reuse=False
        )
        self._binary_prediciton, self._instance_prediction = self._model.inference(
            input_tensor=self._input_src_image,
            name='LaneNet',
            reuse=True
        )

        self._loss = loss_set['total_loss']
        self._binary_seg_loss = loss_set['binary_seg_loss']
        self._disc_loss = loss_set['discriminative_loss']
        self._pix_embedding = loss_set['instance_seg_logits']
        self._binary_prediciton = tf.identity(self._binary_prediciton, name='binary_segmentation_result')

        # define miou
        if self._enable_miou:
            with tf.variable_scope('miou'):
                pred = tf.reshape(self._binary_prediciton, [-1, ])
                gt = tf.reshape(self._input_binary_label_image, [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(gt, self._cfg.DATASET.NUM_CLASSES - 1)), 1)
                gt = tf.gather(gt, indices)
                pred = tf.gather(pred, indices)
                self._miou, self._miou_update_op = tf.metrics.mean_iou(
                    labels=gt,
                    predictions=pred,
                    num_classes=self._cfg.DATASET.NUM_CLASSES
                )

        # define learning rate
        with tf.variable_scope('learning_rate'):
            self._global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            warmup_steps = tf.constant(
                self._warmup_epoches * self._steps_per_epoch, dtype=tf.float32, name='warmup_steps'
            )
            train_steps = tf.constant(
                self._train_epoch_nums * self._steps_per_epoch, dtype=tf.float32, name='train_steps'
            )
            self._learn_rate = tf.cond(
                pred=self._global_step < warmup_steps,
                true_fn=lambda: self._compute_warmup_lr(warmup_steps=warmup_steps, name='warmup_lr'),
                false_fn=lambda: tf.train.polynomial_decay(
                    learning_rate=self._init_learning_rate,
                    global_step=self._global_step,
                    decay_steps=train_steps,
                    end_learning_rate=0.000001,
                    power=self._lr_polynimal_decay_power)
            )
            self._learn_rate = tf.identity(self._learn_rate, 'lr')
            global_step_update = tf.assign_add(self._global_step, 1.0)

        # define moving average op
        with tf.variable_scope(name_or_scope='moving_avg'):
            if self._cfg.TRAIN.FREEZE_BN.ENABLE:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            moving_ave_op = tf.train.ExponentialMovingAverage(
                self._moving_ave_decay).apply(train_var_list + tf.moving_average_variables())
            # define saver
            self._loader = tf.train.Saver(tf.moving_average_variables())

        # define training op
        with tf.variable_scope(name_or_scope='train_step'):
            if self._cfg.TRAIN.FREEZE_BN.ENABLE:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            if self._optimizer_mode == 'sgd':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self._learn_rate,
                    momentum=self._momentum
                )
            elif self._optimizer_mode == 'adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self._learn_rate,
                )
            else:
                raise ValueError('Not support optimizer: {:s}'.format(self._optimizer_mode))
            optimize_op = optimizer.minimize(self._loss, var_list=train_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([optimize_op, global_step_update]):
                    with tf.control_dependencies([moving_ave_op]):
                        self._train_op = tf.no_op()

        # define saver and loader
        with tf.variable_scope('loader_and_saver'):
            self._net_var = [vv for vv in tf.global_variables() if 'lr' not in vv.name]
            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # define summary
        with tf.variable_scope('summary'):
            summary_merge_list = [
                tf.summary.scalar('learn_rate', self._learn_rate),
                tf.summary.scalar('total_loss', self._loss),
                tf.summary.scalar('binary_seg_loss', self._binary_seg_loss),
                tf.summary.scalar('discriminative_loss', self._disc_loss),
            ]
            if self._enable_miou:
                with tf.control_dependencies([self._miou_update_op]):
                    summary_merge_list_with_miou = [
                        tf.summary.scalar('learn_rate', self._learn_rate),
                        tf.summary.scalar('total_loss', self._loss),
                        tf.summary.scalar('binary_seg_loss', self._binary_seg_loss),
                        tf.summary.scalar('discriminative_loss', self._disc_loss),
                        tf.summary.scalar('miou', self._miou)
                    ]
                    self._write_summary_op_with_miou = tf.summary.merge(summary_merge_list_with_miou)
            if ops.exists(self._tboard_save_dir):
                shutil.rmtree(self._tboard_save_dir)
            os.makedirs(self._tboard_save_dir, exist_ok=True)
            model_params_file_save_path = ops.join(self._tboard_save_dir, self._cfg.TRAIN.MODEL_PARAMS_CONFIG_FILE_NAME)
            with open(model_params_file_save_path, 'w', encoding='utf-8') as f_obj:
                self._cfg.dump_to_json_file(f_obj)
            self._write_summary_op = tf.summary.merge(summary_merge_list)
            self._summary_writer = tf.summary.FileWriter(self._tboard_save_dir, graph=self._sess.graph)

        LOG.info('Initialize tusimple lanenet trainner complete')

    def _compute_warmup_lr(self, warmup_steps, name):
        """

        :param warmup_steps:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            factor = tf.math.pow(self._init_learning_rate / self._warmup_init_learning_rate, 1.0 / warmup_steps)
            warmup_lr = self._warmup_init_learning_rate * tf.math.pow(factor, self._global_step)
        return warmup_lr

    def train(self):
        """

        :return:
        """
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            try:
                LOG.info('=> Restoring weights from: {:s} ... '.format(self._initial_weight))
                self._loader.restore(self._sess, self._initial_weight)
                global_step_value = self._sess.run(self._global_step)
                remain_epoch_nums = self._train_epoch_nums - math.floor(global_step_value / self._steps_per_epoch)
                epoch_start_pt = self._train_epoch_nums - remain_epoch_nums
            except OSError as e:
                LOG.error(e)
                LOG.info('=> {:s} does not exist !!!'.format(self._initial_weight))
                LOG.info('=> Now it starts to train LaneNet from scratch ...')
                epoch_start_pt = 1
            except Exception as e:
                LOG.error(e)
                LOG.info('=> Can not load pretrained model weights: {:s}'.format(self._initial_weight))
                LOG.info('=> Now it starts to train LaneNet from scratch ...')
                epoch_start_pt = 1
        else:
            LOG.info('=> Starts to train LaneNet from scratch ...')
            epoch_start_pt = 1

        for epoch in range(epoch_start_pt, self._train_epoch_nums):
            train_epoch_losses = []
            train_epoch_mious = []
            traindataset_pbar = tqdm.tqdm(range(1, self._steps_per_epoch))

            for _ in traindataset_pbar:

                if self._enable_miou and epoch % self._record_miou_epoch == 0:
                    _, _, summary, train_step_loss, train_step_binary_loss, \
                        train_step_instance_loss, global_step_val = \
                        self._sess.run(
                            fetches=[
                                self._train_op, self._miou_update_op,
                                self._write_summary_op_with_miou,
                                self._loss, self._binary_seg_loss, self._disc_loss,
                                self._global_step
                            ]
                        )
                    train_step_miou = self._sess.run(
                        fetches=self._miou
                    )
                    train_epoch_losses.append(train_step_loss)
                    train_epoch_mious.append(train_step_miou)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    traindataset_pbar.set_description(
                        'train loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}, miou: {:.5f}'.format(
                            train_step_loss, train_step_binary_loss, train_step_instance_loss, train_step_miou
                        )
                    )
                else:
                    _, summary, train_step_loss, train_step_binary_loss, \
                        train_step_instance_loss, global_step_val = self._sess.run(
                            fetches=[
                                self._train_op, self._write_summary_op,
                                self._loss, self._binary_seg_loss, self._disc_loss,
                                self._global_step
                            ]
                    )
                    train_epoch_losses.append(train_step_loss)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    traindataset_pbar.set_description(
                        'train loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}'.format(
                            train_step_loss, train_step_binary_loss, train_step_instance_loss
                        )
                    )

            train_epoch_losses = np.mean(train_epoch_losses)
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                train_epoch_mious = np.mean(train_epoch_mious)

            if epoch % self._snapshot_epoch == 0:
                if self._enable_miou:
                    snapshot_model_name = 'tusimple_train_miou={:.4f}.ckpt'.format(train_epoch_mious)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)
                else:
                    snapshot_model_name = 'tusimple_train_loss={:.4f}.ckpt'.format(train_epoch_losses)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} '
                    'Train miou: {:.5f} ...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                        train_epoch_mious,
                    )
                )
            else:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} ...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                    )
                )
        LOG.info('Complete training process good luck!!')

        return
