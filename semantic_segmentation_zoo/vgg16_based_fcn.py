#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午6:42
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : vgg16_based_fcn.py
# @IDE: PyCharm
"""
Implement VGG16 based fcn net for semantic segmentation
"""
import collections

import tensorflow as tf

from config import global_config
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class VGG16FCN(cnn_basenet.CNNBaseModel):
    """
    VGG 16 based fcn net for semantic segmentation
    """
    def __init__(self, phase):
        """

        """
        super(VGG16FCN, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._net_intermediate_results = collections.OrderedDict()

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

    def _vgg16_conv_stage(self, input_tensor, k_size, out_dims, name,
                          stride=1, pad='SAME', need_layer_norm=True):
        """
        stack conv and activation in vgg16
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :param need_layer_norm:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(
                inputdata=input_tensor, out_channel=out_dims,
                kernel_size=k_size, stride=stride,
                use_bias=False, padding=pad, name='conv'
            )

            if need_layer_norm:
                bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

                relu = self.relu(inputdata=bn, name='relu')
            else:
                relu = self.relu(inputdata=conv, name='relu')

        return relu

    def _decode_block(self, input_tensor, previous_feats_tensor,
                      out_channels_nums, name, kernel_size=4,
                      stride=2, use_bias=False,
                      previous_kernel_size=4, need_activate=True):
        """

        :param input_tensor:
        :param previous_feats_tensor:
        :param out_channels_nums:
        :param kernel_size:
        :param previous_kernel_size:
        :param use_bias:
        :param stride:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):

            deconv_weights_stddev = tf.sqrt(
                tf.divide(tf.constant(2.0, tf.float32),
                          tf.multiply(tf.cast(previous_kernel_size * previous_kernel_size, tf.float32),
                                      tf.cast(tf.shape(input_tensor)[3], tf.float32)))
            )
            deconv_weights_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=deconv_weights_stddev)

            deconv = self.deconv2d(
                inputdata=input_tensor, out_channel=out_channels_nums, kernel_size=kernel_size,
                stride=stride, use_bias=use_bias, w_init=deconv_weights_init,
                name='deconv'
            )

            deconv = self.layerbn(inputdata=deconv, is_training=self._is_training, name='deconv_bn')

            deconv = self.relu(inputdata=deconv, name='deconv_relu')

            fuse_feats = tf.add(
                previous_feats_tensor, deconv, name='fuse_feats'
            )

            if need_activate:

                fuse_feats = self.layerbn(
                    inputdata=fuse_feats, is_training=self._is_training, name='fuse_gn'
                )

                fuse_feats = self.relu(inputdata=fuse_feats, name='fuse_relu')

        return fuse_feats

    def _vgg16_fcn_encode(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # encode stage 1
            conv_1_1 = self._vgg16_conv_stage(
                input_tensor=input_tensor, k_size=3,
                out_dims=64, name='conv1_1',
                need_layer_norm=True
            )
            conv_1_2 = self._vgg16_conv_stage(
                input_tensor=conv_1_1, k_size=3,
                out_dims=64, name='conv1_2',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_1_share'] = {
                'data': conv_1_2,
                'shape': conv_1_2.get_shape().as_list()
            }

            # encode stage 2
            pool1 = self.maxpooling(
                inputdata=conv_1_2, kernel_size=2,
                stride=2, name='pool1'
            )
            conv_2_1 = self._vgg16_conv_stage(
                input_tensor=pool1, k_size=3,
                out_dims=128, name='conv2_1',
                need_layer_norm=True
            )
            conv_2_2 = self._vgg16_conv_stage(
                input_tensor=conv_2_1, k_size=3,
                out_dims=128, name='conv2_2',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_2_share'] = {
                'data': conv_2_2,
                'shape': conv_2_2.get_shape().as_list()
            }

            # encode stage 3
            pool2 = self.maxpooling(
                inputdata=conv_2_2, kernel_size=2,
                stride=2, name='pool2'
            )
            conv_3_1 = self._vgg16_conv_stage(
                input_tensor=pool2, k_size=3,
                out_dims=256, name='conv3_1',
                need_layer_norm=True
            )
            conv_3_2 = self._vgg16_conv_stage(
                input_tensor=conv_3_1, k_size=3,
                out_dims=256, name='conv3_2',
                need_layer_norm=True
            )
            conv_3_3 = self._vgg16_conv_stage(
                input_tensor=conv_3_2, k_size=3,
                out_dims=256, name='conv3_3',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_3_share'] = {
                'data': conv_3_3,
                'shape': conv_3_3.get_shape().as_list()
            }

            # encode stage 4
            pool3 = self.maxpooling(
                inputdata=conv_3_3, kernel_size=2,
                stride=2, name='pool3'
            )
            conv_4_1 = self._vgg16_conv_stage(
                input_tensor=pool3, k_size=3,
                out_dims=512, name='conv4_1',
                need_layer_norm=True
            )
            conv_4_2 = self._vgg16_conv_stage(
                input_tensor=conv_4_1, k_size=3,
                out_dims=512, name='conv4_2',
                need_layer_norm=True
            )
            conv_4_3 = self._vgg16_conv_stage(
                input_tensor=conv_4_2, k_size=3,
                out_dims=512, name='conv4_3',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_4_share'] = {
                'data': conv_4_3,
                'shape': conv_4_3.get_shape().as_list()
            }

            # encode stage 5 for binary segmentation
            pool4 = self.maxpooling(
                inputdata=conv_4_3, kernel_size=2,
                stride=2, name='pool4'
            )
            conv_5_1_binary = self._vgg16_conv_stage(
                input_tensor=pool4, k_size=3,
                out_dims=512, name='conv5_1_binary',
                need_layer_norm=True
            )
            conv_5_2_binary = self._vgg16_conv_stage(
                input_tensor=conv_5_1_binary, k_size=3,
                out_dims=512, name='conv5_2_binary',
                need_layer_norm=True
            )
            conv_5_3_binary = self._vgg16_conv_stage(
                input_tensor=conv_5_2_binary, k_size=3,
                out_dims=512, name='conv5_3_binary',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_5_binary'] = {
                'data': conv_5_3_binary,
                'shape': conv_5_3_binary.get_shape().as_list()
            }

            # encode stage 5 for instance segmentation
            conv_5_1_instance = self._vgg16_conv_stage(
                input_tensor=pool4, k_size=3,
                out_dims=512, name='conv5_1_instance',
                need_layer_norm=True
            )
            conv_5_2_instance = self._vgg16_conv_stage(
                input_tensor=conv_5_1_instance, k_size=3,
                out_dims=512, name='conv5_2_instance',
                need_layer_norm=True
            )
            conv_5_3_instance = self._vgg16_conv_stage(
                input_tensor=conv_5_2_instance, k_size=3,
                out_dims=512, name='conv5_3_instance',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_5_instance'] = {
                'data': conv_5_3_instance,
                'shape': conv_5_3_instance.get_shape().as_list()
            }

        return

    def _vgg16_fcn_decode(self, name):
        """

        :return:
        """
        with tf.variable_scope(name):

            # decode part for binary segmentation
            with tf.variable_scope(name_or_scope='binary_seg_decode'):

                decode_stage_5_binary = self._net_intermediate_results['encode_stage_5_binary']['data']

                decode_stage_4_fuse = self._decode_block(
                    input_tensor=decode_stage_5_binary,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_4_share']['data'],
                    name='decode_stage_4_fuse', out_channels_nums=512, previous_kernel_size=3
                )
                decode_stage_3_fuse = self._decode_block(
                    input_tensor=decode_stage_4_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_3_share']['data'],
                    name='decode_stage_3_fuse', out_channels_nums=256
                )
                decode_stage_2_fuse = self._decode_block(
                    input_tensor=decode_stage_3_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_2_share']['data'],
                    name='decode_stage_2_fuse', out_channels_nums=128
                )
                decode_stage_1_fuse = self._decode_block(
                    input_tensor=decode_stage_2_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_1_share']['data'],
                    name='decode_stage_1_fuse', out_channels_nums=64
                )
                binary_final_logits_conv_weights_stddev = tf.sqrt(
                    tf.divide(tf.constant(2.0, tf.float32),
                              tf.multiply(4.0 * 4.0,
                                          tf.cast(tf.shape(decode_stage_1_fuse)[3], tf.float32)))
                )
                binary_final_logits_conv_weights_init = tf.truncated_normal_initializer(
                    mean=0.0, stddev=binary_final_logits_conv_weights_stddev)

                binary_final_logits = self.conv2d(
                    inputdata=decode_stage_1_fuse, out_channel=CFG.TRAIN.CLASSES_NUMS,
                    kernel_size=1, use_bias=False,
                    w_init=binary_final_logits_conv_weights_init,
                    name='binary_final_logits')

                self._net_intermediate_results['binary_segment_logits'] = {
                    'data': binary_final_logits,
                    'shape': binary_final_logits.get_shape().as_list()
                }

            with tf.variable_scope(name_or_scope='instance_seg_decode'):

                decode_stage_5_instance = self._net_intermediate_results['encode_stage_5_instance']['data']

                decode_stage_4_fuse = self._decode_block(
                    input_tensor=decode_stage_5_instance,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_4_share']['data'],
                    name='decode_stage_4_fuse', out_channels_nums=512, previous_kernel_size=3)

                decode_stage_3_fuse = self._decode_block(
                    input_tensor=decode_stage_4_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_3_share']['data'],
                    name='decode_stage_3_fuse', out_channels_nums=256)

                decode_stage_2_fuse = self._decode_block(
                    input_tensor=decode_stage_3_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_2_share']['data'],
                    name='decode_stage_2_fuse', out_channels_nums=128)

                decode_stage_1_fuse = self._decode_block(
                    input_tensor=decode_stage_2_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_1_share']['data'],
                    name='decode_stage_1_fuse', out_channels_nums=64, need_activate=False)

                self._net_intermediate_results['instance_segment_logits'] = {
                    'data': decode_stage_1_fuse,
                    'shape': decode_stage_1_fuse.get_shape().as_list()
                }

    def build_model(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # vgg16 fcn encode part
            self._vgg16_fcn_encode(input_tensor=input_tensor, name='vgg16_encode_module')
            # vgg16 fcn decode part
            self._vgg16_fcn_decode(name='vgg16_decode_module')

        return self._net_intermediate_results


if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    model = VGG16FCN(phase='train')
    ret = model.build_model(test_in_tensor, name='vgg16fcn')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
