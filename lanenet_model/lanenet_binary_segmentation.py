#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_binary_segmentation.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中的二分类图像分割模型
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet


class LaneNetBinarySeg(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """
    def __init__(self, phase, net_flag='vgg'):
        """

        """
        super(LaneNetBinarySeg, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=self._phase,
                                                       n=5)
        self._decoder = fcn_decoder.FCNDecoder()
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def build_model(self, input_tensor, name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret
            elif self._net_flag.lower() == 'dense':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dense_Block_5',
                                                                     'Dense_Block_4',
                                                                     'Dense_Block_3'])
                return decode_ret

    def compute_loss(self, input_tensor, label, name):
        """
        计算损失函数
        :param input_tensor:
        :param label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # 计算损失
            decode_logits = inference_ret['logits']
            # 加入bounded inverse class weights
            inverse_class_weights = tf.divide(1.0,
                                              tf.log(tf.add(tf.constant(1.02, tf.float32),
                                                            tf.nn.softmax(decode_logits))))
            decode_logits_weighted = tf.multiply(decode_logits, inverse_class_weights)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decode_logits_weighted, labels=tf.squeeze(label, squeeze_dims=[3]),
                name='entropy_loss')

            ret = dict()
            ret['entropy_loss'] = loss
            ret['inference_logits'] = inference_ret['logits']

            return ret
