#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午3:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_back_end.py
# @IDE: PyCharm
"""
LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
"""
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_discriminative_loss
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class LaneNetBackEnd(cnn_basenet.CNNBaseModel):
    """
    LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
    """
    def __init__(self, phase):
        """
        init lanenet backend
        :param phase: train or test
        """
        super(LaneNetBackEnd, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    @staticmethod
    def _compute_class_weighted_cross_entropy_loss(onehot_labels, logits, classes_weights):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )

        return loss

    def compute_loss(self, binary_seg_logits, binary_label,
                     instance_seg_logits, instance_label,
                     name, reuse):
        """
        compute lanenet loss
        :param binary_seg_logits:
        :param binary_label:
        :param instance_seg_logits:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # calculate class weighted binary seg loss
            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_label_onehot = tf.one_hot(
                    tf.reshape(
                        tf.cast(binary_label, tf.int32),
                        shape=[binary_label.get_shape().as_list()[0],
                               binary_label.get_shape().as_list()[1],
                               binary_label.get_shape().as_list()[2]]),
                    depth=2,
                    axis=-1
                )

                classes_weights = [1.4506131276238088, 21.525424601474068]
                binary_segmenatation_loss = self._compute_class_weighted_cross_entropy_loss(
                    onehot_labels=binary_label_onehot,
                    logits=binary_seg_logits,
                    classes_weights=classes_weights
                )

            # calculate class weighted instance seg loss
            with tf.variable_scope(name_or_scope='instance_seg'):

                pix_bn = self.layerbn(
                    inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                pix_embedding = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )
                pix_image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
                disc_loss, l_var, l_dist, l_reg = \
                    lanenet_discriminative_loss.discriminative_loss(
                        pix_embedding, instance_label, CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                        pix_image_shape, 0.5, 3.5, 1.0, 1.0, 0.001
                    )

            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'gn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = 0.5 * binary_segmenatation_loss + 0.5 * disc_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': binary_seg_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': disc_loss
            }

        return ret

    def inference(self, binary_seg_logits, instance_seg_logits, name, reuse):
        """

        :param binary_seg_logits:
        :param instance_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)
                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)

            with tf.variable_scope(name_or_scope='instance_seg'):

                pix_bn = self.layerbn(
                    inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                instance_seg_prediction = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )

        return binary_seg_prediction, instance_seg_prediction
