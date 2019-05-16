#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午8:50
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet.py
# @IDE: PyCharm
"""
Implement LaneNet Model
"""
import tensorflow as tf

from config import global_config
from data_provider import lanenet_data_feed_pipline
from lanenet_model import lanenet_back_end
from lanenet_model import lanenet_front_end
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class LaneNet(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, net_flag='vgg', reuse=False):
        """

        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._reuse = reuse

        self._frontend = lanenet_front_end.LaneNetFrondEnd(
            phase=phase, net_flag=net_flag
        )
        self._backend = lanenet_back_end.LaneNetBackEnd(
            phase=phase
        )

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )

            # second apply backend process
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse
            )

            if not self._reuse:
                self._reuse = True

        return binary_seg_prediction, instance_seg_prediction

    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        """
        calculate lanenet loss for training
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )

            # second apply backend process
            calculated_losses = self._backend.compute_loss(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                binary_label=binary_label,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                instance_label=instance_label,
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse
            )

            if not self._reuse:
                self._reuse = True

        return calculated_losses


if __name__ == '__main__':
    """
    test code
    """
    tusimple_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir='/IMAGE_SCENE_SEGMENTATION/TUSIMPLE_DATASET/train_set/training',
        flags='train'
    )

    test_in_tensor, test_binary_tensor, test_instance_tensor = tusimple_dataset.inputs(4, 1)

    train_model = LaneNet(phase='train', net_flag='vgg')
    val_model = LaneNet(phase='val', net_flag='vgg', reuse=True)

    train_inference = train_model.inference(input_tensor=test_in_tensor, name='lanenet')
    train_loss = train_model.compute_loss(
        input_tensor=test_in_tensor,
        binary_label=test_binary_tensor,
        instance_label=test_instance_tensor,
        name='lanenet'
    )

    val_inference = val_model.inference(input_tensor=test_in_tensor, name='lanenet')
    val_loss = val_model.compute_loss(
        input_tensor=test_in_tensor,
        binary_label=test_binary_tensor,
        instance_label=test_instance_tensor,
        name='lanenet'
    )

    for vv in tf.global_variables():
        print(vv.name)

    print(train_inference)

    for layer_name, layer_info in train_loss.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info.get_shape().as_list()))
