#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 下午2:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : tusimple_lanenet_multi_gpu_trainner.py
# @IDE: PyCharm
"""
LanetNet multi gpu trainner for cityscapes dataset
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
from local_utils.config_utils import parse_config_utils
from lanenet_model import lanenet

LOG = loguru.logger


class LaneNetTusimpleMultiTrainer(object):
    """
    init lanenet multi gpu trainner
    """
    def __init__(self, cfg):
        """
        initialize lanenet multi gpu trainner
        """
        self._cfg = cfg
        # define solver params and dataset
        self._train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(flags='train')
        self._val_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(flags='val')
        self._steps_per_epoch = len(self._train_dataset)
        self._val_steps_per_epoch = len(self._val_dataset)

        self._model_name = '{:s}_{:s}'.format(self._cfg.MODEL.FRONT_END, self._cfg.MODEL.MODEL_NAME)

        self._train_epoch_nums = self._cfg.TRAIN.EPOCH_NUMS
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._val_batch_size = self._cfg.TRAIN.VAL_BATCH_SIZE
        self._snapshot_epoch = self._cfg.TRAIN.SNAPSHOT_EPOCH
        self._model_save_dir = ops.join(self._cfg.TRAIN.MODEL_SAVE_DIR, self._model_name)
        self._tboard_save_dir = ops.join(self._cfg.TRAIN.TBOARD_SAVE_DIR, self._model_name)
        self._enable_miou = self._cfg.TRAIN.COMPUTE_MIOU.ENABLE
        if self._enable_miou:
            self._record_miou_epoch = self._cfg.TRAIN.COMPUTE_MIOU.EPOCH
        self._input_tensor_size = [int(tmp) for tmp in self._cfg.AUG.TRAIN_CROP_SIZE]
        self._gpu_devices = self._cfg.TRAIN.MULTI_GPU.GPU_DEVICES
        self._gpu_nums = len(self._gpu_devices)
        self._chief_gpu_index = self._cfg.TRAIN.MULTI_GPU.CHIEF_DEVICE_INDEX
        self._batch_size_per_gpu = int(self._batch_size / self._gpu_nums)

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
            self._input_src_image_list = []
            self._input_binary_label_image_list = []
            self._input_instance_label_image_list = []
            for i in range(self._gpu_nums):
                src_imgs, binary_label_imgs, instance_label_imgs = self._train_dataset.next_batch(
                    batch_size=self._batch_size_per_gpu
                )
                self._input_src_image_list.append(src_imgs)
                self._input_binary_label_image_list.append(binary_label_imgs)
                self._input_instance_label_image_list.append(instance_label_imgs)
            self._val_input_src_image, self._val_input_binary_label_image, self._val_input_instance_label_image = \
                self._val_dataset.next_batch(batch_size=self._val_batch_size)

        # define model
        self._model = lanenet.LaneNet(phase='train', cfg=self._cfg)
        self._val_model = lanenet.LaneNet(phase='test', cfg=self._cfg)

        # define average container
        tower_grads = []
        tower_total_loss = []
        tower_binary_seg_loss = []
        tower_instance_seg_loss = []
        batchnorm_updates = None

        # define learning rate
        with tf.variable_scope('learning_rate'):
            self._global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            self._val_global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='val_global_step')
            self._val_global_step_update = tf.assign_add(self._val_global_step, 1.0)
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
                    end_learning_rate=0.000000001,
                    power=self._lr_polynimal_decay_power)
            )
            self._learn_rate = tf.identity(self._learn_rate, 'lr')

        # define optimizer
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
            raise NotImplementedError('Not support optimizer: {:s} for now'.format(self._optimizer_mode))

        # define distributed train op
        with tf.variable_scope(tf.get_variable_scope()):
            is_network_initialized = False
            for i in range(self._gpu_nums):
                with tf.device('/gpu:{:d}'.format(i)):
                    with tf.name_scope('tower_{:d}'.format(i)) as _:
                        input_images = self._input_src_image_list[i]
                        input_binary_labels = self._input_binary_label_image_list[i]
                        input_instance_labels = self._input_instance_label_image_list[i]
                        tmp_loss, tmp_grads = self._compute_net_gradients(
                            input_images, input_binary_labels, input_instance_labels, optimizer,
                            is_net_first_initialized=is_network_initialized
                        )
                        is_network_initialized = True

                        # Only use the mean and var in the chief gpu tower to update the parameter
                        if i == self._chief_gpu_index:
                            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                        tower_grads.append(tmp_grads)
                        tower_total_loss.append(tmp_loss['total_loss'])
                        tower_binary_seg_loss.append(tmp_loss['binary_seg_loss'])
                        tower_instance_seg_loss.append(tmp_loss['discriminative_loss'])
        grads = self._average_gradients(tower_grads)
        self._loss = tf.reduce_mean(tower_total_loss, name='reduce_mean_tower_total_loss')
        self._binary_loss = tf.reduce_mean(tower_binary_seg_loss, name='reduce_mean_tower_binary_loss')
        self._instance_loss = tf.reduce_mean(tower_instance_seg_loss, name='reduce_mean_tower_instance_loss')
        ret = self._val_model.compute_loss(
            input_tensor=self._val_input_src_image,
            binary_label=self._val_input_binary_label_image,
            instance_label=self._val_input_instance_label_image,
            name='LaneNet',
            reuse=True
        )
        self._val_loss = ret['total_loss']
        self._val_binary_loss = ret['binary_seg_loss']
        self._val_instance_loss = ret['discriminative_loss']

        # define moving average op
        with tf.variable_scope(name_or_scope='moving_avg'):
            if self._cfg.TRAIN.FREEZE_BN.ENABLE:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            moving_ave_op = tf.train.ExponentialMovingAverage(self._moving_ave_decay).apply(
                train_var_list + tf.moving_average_variables()
            )
            # define saver
            self._loader = tf.train.Saver(tf.moving_average_variables())

        # group all the op needed for training
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=self._global_step)
        self._train_op = tf.group(apply_gradient_op, moving_ave_op, batchnorm_updates_op)

        # define prediction
        self._binary_prediciton, self._instance_prediciton = self._model.inference(
            input_tensor=self._input_src_image_list[self._chief_gpu_index],
            name='LaneNet',
            reuse=True
        )
        self._binary_prediciton = tf.identity(self._binary_prediciton, name='binary_segmentation_result')
        self._val_binary_prediction, self._val_instance_prediciton = self._val_model.inference(
            input_tensor=self._val_input_src_image,
            name='LaneNet',
            reuse=True
        )
        self._val_binary_prediction = tf.identity(self._val_binary_prediction, name='val_binary_segmentation_result')

        # define miou
        if self._enable_miou:
            with tf.variable_scope('miou'):
                pred = tf.reshape(self._binary_prediciton, [-1, ])
                gt = tf.reshape(self._input_binary_label_image_list[self._chief_gpu_index], [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(gt, self._cfg.DATASET.NUM_CLASSES - 1)), 1)
                gt = tf.gather(gt, indices)
                pred = tf.gather(pred, indices)
                self._miou, self._miou_update_op = tf.metrics.mean_iou(
                    labels=gt,
                    predictions=pred,
                    num_classes=self._cfg.DATASET.NUM_CLASSES
                )

                val_pred = tf.reshape(self._val_binary_prediction, [-1, ])
                val_gt = tf.reshape(self._val_input_binary_label_image, [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(val_gt, self._cfg.DATASET.NUM_CLASSES - 1)), 1)
                val_gt = tf.gather(val_gt, indices)
                val_pred = tf.gather(val_pred, indices)
                self._val_miou, self._val_miou_update_op = tf.metrics.mean_iou(
                    labels=val_gt,
                    predictions=val_pred,
                    num_classes=self._cfg.DATASET.NUM_CLASSES
                )

        # define saver and loader
        with tf.variable_scope('loader_and_saver'):
            self._net_var = [vv for vv in tf.global_variables() if 'lr' not in vv.name]
            self._saver = tf.train.Saver(max_to_keep=10)

        # define summary
        with tf.variable_scope('summary'):
            summary_merge_list = [
                tf.summary.scalar("learn_rate", self._learn_rate),
                tf.summary.scalar("total_loss", self._loss),
                tf.summary.scalar('binary_loss', self._binary_loss),
                tf.summary.scalar('instance_loss', self._instance_loss),
            ]
            val_summary_merge_list = [
                tf.summary.scalar('val_total_loss', self._val_loss),
                tf.summary.scalar('val_binary_loss', self._val_binary_loss),
                tf.summary.scalar('val_instance_loss', self._val_instance_loss),
            ]
            if self._enable_miou:
                with tf.control_dependencies([self._miou_update_op]):
                    summary_merge_list_with_miou = [
                        tf.summary.scalar("learn_rate", self._learn_rate),
                        tf.summary.scalar("total_loss", self._loss),
                        tf.summary.scalar('binary_loss', self._binary_loss),
                        tf.summary.scalar('instance_loss', self._instance_loss),
                        tf.summary.scalar('miou', self._miou)
                    ]
                    self._write_summary_op_with_miou = tf.summary.merge(summary_merge_list_with_miou)
                with tf.control_dependencies([self._val_miou_update_op, self._val_global_step_update]):
                    val_summary_merge_list_with_miou = [
                        tf.summary.scalar("total_loss", self._loss),
                        tf.summary.scalar('binary_loss', self._binary_loss),
                        tf.summary.scalar('instance_loss', self._instance_loss),
                        tf.summary.scalar('val_miou', self._val_miou),
                    ]
                    self._val_write_summary_op_with_miou = tf.summary.merge(val_summary_merge_list_with_miou)
            if ops.exists(self._tboard_save_dir):
                shutil.rmtree(self._tboard_save_dir)
            os.makedirs(self._tboard_save_dir, exist_ok=True)
            model_params_file_save_path = ops.join(self._tboard_save_dir, self._cfg.TRAIN.MODEL_PARAMS_CONFIG_FILE_NAME)
            with open(model_params_file_save_path, 'w', encoding='utf-8') as f_obj:
                self._cfg.dump_to_json_file(f_obj)
            self._write_summary_op = tf.summary.merge(summary_merge_list)
            self._val_write_summary_op = tf.summary.merge(val_summary_merge_list)
            self._summary_writer = tf.summary.FileWriter(self._tboard_save_dir, graph=self._sess.graph)

        LOG.info('Initialize tusimple lanenet multi gpu trainner complete')

    @staticmethod
    def _average_gradients(tower_grads):
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

    def _compute_net_gradients(self, images, binary_labels, instance_labels, optimizer=None,
                               is_net_first_initialized=False):
        """
        Calculate gradients for single GPU
        :param images: images for training
        :param binary_labels: binary labels corresponding to images
        :param instance_labels: instance labels corresponding to images
        :param optimizer: network optimizer
        :param is_net_first_initialized: if the network is initialized
        :return:
        """
        net_loss = self._model.compute_loss(
            input_tensor=images,
            binary_label=binary_labels,
            instance_label=instance_labels,
            name='LaneNet',
            reuse=is_net_first_initialized
        )

        if self._cfg.TRAIN.FREEZE_BN.ENABLE:
            train_var_list = [
                v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
            ]
        else:
            train_var_list = tf.trainable_variables()

        if optimizer is not None:
            grads = optimizer.compute_gradients(net_loss['total_loss'], var_list=train_var_list)
        else:
            grads = None

        return net_loss, grads

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

        best_model = []
        for epoch in range(epoch_start_pt, self._train_epoch_nums):
            # training part
            train_epoch_losses = []
            train_epoch_mious = []
            traindataset_pbar = tqdm.tqdm(range(1, self._steps_per_epoch))
            for _ in traindataset_pbar:
                if self._enable_miou and epoch % self._record_miou_epoch == 0:
                    _, _, summary, train_step_loss, train_step_binary_loss, \
                        train_step_instance_loss, global_step_val = self._sess.run(
                            fetches=[
                                self._train_op, self._miou_update_op, self._write_summary_op_with_miou,
                                self._loss, self._binary_loss, self._instance_loss,
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
                                self._loss, self._binary_loss, self._instance_loss,
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

            # validation part
            val_epoch_losses = []
            val_epoch_mious = []
            valdataset_pbar = tqdm.tqdm(range(1, self._val_steps_per_epoch))
            for _ in valdataset_pbar:
                try:
                    if self._enable_miou and epoch % self._record_miou_epoch == 0:
                        _, val_summary, val_step_loss, val_step_binary_loss, \
                            val_step_instance_loss, val_global_step_val = self._sess.run(
                                    fetches=[
                                        self._val_miou_update_op, self._val_write_summary_op_with_miou,
                                        self._val_loss, self._val_binary_loss, self._val_instance_loss,
                                        self._val_global_step
                                    ]
                            )
                        val_step_miou = self._sess.run(
                            fetches=self._val_miou
                        )
                        val_epoch_losses.append(val_step_loss)
                        val_epoch_mious.append(val_step_miou)
                        self._summary_writer.add_summary(val_summary, global_step=val_global_step_val)
                        valdataset_pbar.set_description(
                            'val loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}, val miou: {:.5f}'.format(
                                val_step_loss, val_step_binary_loss, val_step_instance_loss, val_step_miou)
                        )
                    else:
                        val_summary, val_step_loss, val_step_binary_loss, \
                            val_step_instance_loss, val_global_step_val = self._sess.run(
                                fetches=[
                                    self._val_write_summary_op,
                                    self._val_loss, self._val_binary_loss, self._val_instance_loss,
                                    self._val_global_step
                                ]
                            )
                        val_epoch_losses.append(val_step_loss)
                        self._summary_writer.add_summary(val_summary, global_step=val_global_step_val)
                        valdataset_pbar.set_description(
                            'val loss: {:.5f} b_loss: {:.5f}, i_loss: {:.5f}'.format(
                                val_step_loss, val_step_binary_loss, val_step_instance_loss
                            )
                        )
                except tf.errors.OutOfRangeError as _:
                    break
            val_epoch_losses = np.mean(val_epoch_losses)
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                val_epoch_mious = np.mean(val_epoch_mious)

            # model saving part
            if epoch % self._snapshot_epoch == 0:
                if self._enable_miou:
                    if len(best_model) < 10:
                        best_model.append(val_epoch_mious)
                        best_model = sorted(best_model)
                        snapshot_model_name = 'tusimple_val_miou={:.4f}.ckpt'.format(val_epoch_mious)
                        snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                        os.makedirs(self._model_save_dir, exist_ok=True)
                        self._saver.save(self._sess, snapshot_model_path, global_step=epoch)
                    else:
                        best_model = sorted(best_model)
                        if val_epoch_mious > best_model[0]:
                            best_model[0] = val_epoch_mious
                            best_model = sorted(best_model)
                            snapshot_model_name = 'tusimple_val_miou={:.4f}.ckpt'.format(val_epoch_mious)
                            snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                            os.makedirs(self._model_save_dir, exist_ok=True)
                            self._saver.save(self._sess, snapshot_model_path, global_step=epoch)
                        else:
                            pass
                else:
                    snapshot_model_name = 'tusimple_val_loss={:.4f}.ckpt'.format(val_epoch_losses)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} Train miou: {:.5f} '
                    'Val loss: {:.5f} Val miou: {:.5f}...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                        train_epoch_mious,
                        val_epoch_losses,
                        val_epoch_mious
                    )
                )
            else:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} Val loss: {:.5f}...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                        val_epoch_losses
                    )
                )
        if self._enable_miou:
            LOG.info('Best model\'s val mious are: {}'.format(best_model))
        LOG.info('Complete training process good luck!!')

        return


if __name__ == '__main__':
    """
    test code
    """
    worker = LaneNetTusimpleMultiTrainer(cfg=parse_config_utils.lanenet_cfg)
    print('Init complete')
