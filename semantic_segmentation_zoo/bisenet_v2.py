#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/9 上午11:05
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : bisenet_v2.py
# @IDE: PyCharm
"""
BiseNet V2 Model
"""
import collections

import tensorflow as tf

from semantic_segmentation_zoo import cnn_basenet
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg


class _StemBlock(cnn_basenet.CNNBaseModel):
    """
    implementation of stem block module
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_StemBlock, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        with tf.variable_scope(name_or_scope=name_scope):
            input_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=output_channels,
                stride=2,
                name='conv_block_1',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            with tf.variable_scope(name_or_scope='downsample_branch_left'):
                branch_left_output = self._conv_block(
                    input_tensor=input_tensor,
                    k_size=1,
                    output_channels=int(output_channels / 2),
                    stride=1,
                    name='1x1_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=True
                )
                branch_left_output = self._conv_block(
                    input_tensor=branch_left_output,
                    k_size=3,
                    output_channels=output_channels,
                    stride=2,
                    name='3x3_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=True
                )
            with tf.variable_scope(name_or_scope='downsample_branch_right'):
                branch_right_output = self.maxpooling(
                    inputdata=input_tensor,
                    kernel_size=3,
                    stride=2,
                    padding=self._padding,
                    name='maxpooling_block'
                )
            result = tf.concat([branch_left_output, branch_right_output], axis=-1, name='concate_features')
            result = self._conv_block(
                input_tensor=result,
                k_size=3,
                output_channels=output_channels,
                stride=1,
                name='final_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
        return result


class _ContextEmbedding(cnn_basenet.CNNBaseModel):
    """
    implementation of context embedding module in bisenetv2
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_ContextEmbedding, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        output_channels = input_tensor.get_shape().as_list()[-1]
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        with tf.variable_scope(name_or_scope=name_scope):
            result = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True, name='global_avg_pooling')
            result = self.layerbn(result, self._is_training, 'bn')
            result = self._conv_block(
                input_tensor=result,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='conv_block_1',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = tf.add(result, input_tensor, name='fused_features')
            result = self.conv2d(
                inputdata=result,
                out_channel=output_channels,
                kernel_size=3,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='final_conv_block'
            )
        return result


class _GatherExpansion(cnn_basenet.CNNBaseModel):
    """
    implementation of gather and expansion module in bisenetv2
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_GatherExpansion, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'
        self._stride = 1
        self._expansion_factor = 6

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def _apply_ge_when_stride_equal_one(self, input_tensor, e, name):
        """

        :param input_tensor:
        :param e:
        :param name
        :return:
        """
        input_tensor_channels = input_tensor.get_shape().as_list()[-1]
        with tf.variable_scope(name_or_scope=name):
            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=input_tensor_channels,
                stride=1,
                name='3x3_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = self.depthwise_conv(
                input_tensor=result,
                kernel_size=3,
                depth_multiplier=e,
                padding=self._padding,
                stride=1,
                name='depthwise_conv_block'
            )
            result = self.layerbn(result, self._is_training, name='dw_bn')
            result = self._conv_block(
                input_tensor=result,
                k_size=1,
                output_channels=input_tensor_channels,
                stride=1,
                name='1x1_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=False
            )
            result = tf.add(input_tensor, result, name='fused_features')
            result = self.relu(result, name='ge_output')
        return result

    def _apply_ge_when_stride_equal_two(self, input_tensor, output_channels, e, name):
        """

        :param input_tensor:
        :param output_channels:
        :param e:
        :param name
        :return:
        """
        input_tensor_channels = input_tensor.get_shape().as_list()[-1]
        with tf.variable_scope(name_or_scope=name):
            input_proj = self.depthwise_conv(
                input_tensor=input_tensor,
                kernel_size=3,
                name='input_project_dw_conv_block',
                depth_multiplier=1,
                padding=self._padding,
                stride=self._stride
            )
            input_proj = self.layerbn(input_proj, self._is_training, name='input_project_bn')
            input_proj = self._conv_block(
                input_tensor=input_proj,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='input_project_1x1_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=False
            )

            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=input_tensor_channels,
                stride=1,
                name='3x3_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = self.depthwise_conv(
                input_tensor=result,
                kernel_size=3,
                depth_multiplier=e,
                padding=self._padding,
                stride=2,
                name='depthwise_conv_block_1'
            )
            result = self.layerbn(result, self._is_training, name='dw_bn_1')
            result = self.depthwise_conv(
                input_tensor=result,
                kernel_size=3,
                depth_multiplier=1,
                padding=self._padding,
                stride=1,
                name='depthwise_conv_block_2'
            )
            result = self.layerbn(result, self._is_training, name='dw_bn_2')
            result = self._conv_block(
                input_tensor=result,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='1x1_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=False
            )
            result = tf.add(input_proj, result, name='fused_features')
            result = self.relu(result, name='ge_output')
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        output_channels = input_tensor.get_shape().as_list()[-1]
        if 'output_channels' in kwargs:
            output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        if 'stride' in kwargs:
            self._stride = kwargs['stride']
        if 'e' in kwargs:
            self._expansion_factor = kwargs['e']

        with tf.variable_scope(name_or_scope=name_scope):
            if self._stride == 1:
                result = self._apply_ge_when_stride_equal_one(
                    input_tensor=input_tensor,
                    e=self._expansion_factor,
                    name='stride_equal_one_module'
                )
            elif self._stride == 2:
                result = self._apply_ge_when_stride_equal_two(
                    input_tensor=input_tensor,
                    output_channels=output_channels,
                    e=self._expansion_factor,
                    name='stride_equal_two_module'
                )
            else:
                raise NotImplementedError('No function matched with stride of {}'.format(self._stride))
        return result


class _GuidedAggregation(cnn_basenet.CNNBaseModel):
    """
    implementation of guided aggregation module in bisenetv2
    """

    def __init__(self, phase):
        """

        :param phase:
        """
        super(_GuidedAggregation, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        detail_input_tensor = kwargs['detail_input_tensor']
        semantic_input_tensor = kwargs['semantic_input_tensor']
        name_scope = kwargs['name']
        output_channels = detail_input_tensor.get_shape().as_list()[-1]
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        with tf.variable_scope(name_or_scope=name_scope):
            with tf.variable_scope(name_or_scope='detail_branch'):
                detail_branch_remain = self.depthwise_conv(
                    input_tensor=detail_input_tensor,
                    kernel_size=3,
                    name='3x3_dw_conv_block',
                    depth_multiplier=1,
                    padding=self._padding,
                    stride=1
                )
                detail_branch_remain = self.layerbn(detail_branch_remain, self._is_training, name='bn_1')
                detail_branch_remain = self.conv2d(
                    inputdata=detail_branch_remain,
                    out_channel=output_channels,
                    kernel_size=1,
                    padding=self._padding,
                    stride=1,
                    use_bias=False,
                    name='1x1_conv_block'
                )

                detail_branch_downsample = self._conv_block(
                    input_tensor=detail_input_tensor,
                    k_size=3,
                    output_channels=output_channels,
                    stride=2,
                    name='3x3_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=False
                )
                detail_branch_downsample = self.avgpooling(
                    inputdata=detail_branch_downsample,
                    kernel_size=3,
                    stride=2,
                    padding=self._padding,
                    name='avg_pooling_block'
                )

            with tf.variable_scope(name_or_scope='semantic_branch'):
                semantic_branch_remain = self.depthwise_conv(
                    input_tensor=semantic_input_tensor,
                    kernel_size=3,
                    name='3x3_dw_conv_block',
                    depth_multiplier=1,
                    padding=self._padding,
                    stride=1
                )
                semantic_branch_remain = self.layerbn(semantic_branch_remain, self._is_training, name='bn_1')
                semantic_branch_remain = self.conv2d(
                    inputdata=semantic_branch_remain,
                    out_channel=output_channels,
                    kernel_size=1,
                    padding=self._padding,
                    stride=1,
                    use_bias=False,
                    name='1x1_conv_block'
                )
                semantic_branch_remain = self.sigmoid(semantic_branch_remain, name='semantic_remain_sigmoid')

                semantic_branch_upsample = self._conv_block(
                    input_tensor=semantic_input_tensor,
                    k_size=3,
                    output_channels=output_channels,
                    stride=1,
                    name='3x3_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=False
                )
                semantic_branch_upsample = tf.image.resize_bilinear(
                    semantic_branch_upsample,
                    detail_input_tensor.shape[1:3],
                    name='semantic_upsample_features'
                )
                semantic_branch_upsample = self.sigmoid(semantic_branch_upsample, name='semantic_upsample_sigmoid')

            with tf.variable_scope(name_or_scope='aggregation_features'):
                guided_features_remain = tf.multiply(
                    detail_branch_remain,
                    semantic_branch_upsample,
                    name='guided_detail_features'
                )
                guided_features_downsample = tf.multiply(
                    detail_branch_downsample,
                    semantic_branch_remain,
                    name='guided_semantic_features'
                )
                guided_features_upsample = tf.image.resize_bilinear(
                    guided_features_downsample,
                    detail_input_tensor.shape[1:3],
                    name='guided_upsample_features'
                )
                guided_features = tf.add(guided_features_remain, guided_features_upsample, name='fused_features')
                guided_features = self._conv_block(
                    input_tensor=guided_features,
                    k_size=3,
                    output_channels=output_channels,
                    stride=1,
                    name='aggregation_feature_output',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=True
                )
        return guided_features


class _SegmentationHead(cnn_basenet.CNNBaseModel):
    """
    implementation of segmentation head in bisenet v2
    """
    def __init__(self, phase):
        """

        """
        super(_SegmentationHead, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        ratio = kwargs['upsample_ratio']
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * ratio) for tmp in input_tensor_size]
        feature_dims = kwargs['feature_dims']
        classes_nums = kwargs['classes_nums']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        with tf.variable_scope(name_or_scope=name_scope):
            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=feature_dims,
                stride=1,
                name='3x3_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = self.conv2d(
                inputdata=result,
                out_channel=classes_nums,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='1x1_conv_block'
            )
            result = tf.image.resize_bilinear(
                result,
                output_tensor_size,
                name='segmentation_head_logits'
            )
        return result


class BiseNetV2(cnn_basenet.CNNBaseModel):
    """
    implementation of bisenet v2
    """
    def __init__(self, phase):
        """

        """
        super(BiseNetV2, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

        # set model hyper params
        self._class_nums = CFG.DATASET.NUM_CLASSES
        self._weights_decay = CFG.SOLVER.WEIGHT_DECAY
        self._loss_type = CFG.SOLVER.LOSS_TYPE
        self._enable_ohem = CFG.SOLVER.OHEM.ENABLE
        if self._enable_ohem:
            self._ohem_score_thresh = CFG.SOLVER.OHEM.SCORE_THRESH
            self._ohem_min_sample_nums = CFG.SOLVER.OHEM.MIN_SAMPLE_NUMS
        self._ge_expand_ratio = CFG.MODEL.BISENETV2.GE_EXPAND_RATIO
        self._semantic_channel_ratio = CFG.MODEL.BISENETV2.SEMANTIC_CHANNEL_LAMBDA
        self._seg_head_ratio = CFG.MODEL.BISENETV2.SEGHEAD_CHANNEL_EXPAND_RATIO

        # set module used in bisenetv2
        self._se_block = _StemBlock(phase=phase)
        self._context_embedding_block = _ContextEmbedding(phase=phase)
        self._ge_block = _GatherExpansion(phase=phase)
        self._guided_aggregation_block = _GuidedAggregation(phase=phase)
        self._seg_head_block = _SegmentationHead(phase=phase)

        # set detail branch channels
        self._detail_branch_channels = self._build_detail_branch_hyper_params()
        # set semantic branch channels
        self._semantic_branch_channels = self._build_semantic_branch_hyper_params()

        # set op block params
        self._block_maps = {
            'conv_block': self._conv_block,
            'se': self._se_block,
            'ge': self._ge_block,
            'ce': self._context_embedding_block,
        }

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

    @classmethod
    def _build_detail_branch_hyper_params(cls):
        """

        :return:
        """
        params = [
            ('stage_1', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 1)]),
            ('stage_2', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 2)]),
            ('stage_3', [('conv_block', 3, 128, 2, 1), ('conv_block', 3, 128, 1, 2)]),
        ]
        return collections.OrderedDict(params)

    def _build_semantic_branch_hyper_params(self):
        """

        :return:
        """
        stage_1_channels = int(self._detail_branch_channels['stage_1'][0][2] * self._semantic_channel_ratio)
        stage_3_channels = int(self._detail_branch_channels['stage_3'][0][2] * self._semantic_channel_ratio)
        params = [
            ('stage_1', [('se', 3, stage_1_channels, 1, 4, 1)]),
            ('stage_3', [('ge', 3, stage_3_channels, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels, self._ge_expand_ratio, 1, 1)]),
            ('stage_4', [('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 1, 1)]),
            ('stage_5', [('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 3),
                         ('ce', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 1)])
        ]
        return collections.OrderedDict(params)

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def build_detail_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        result = input_tensor
        with tf.variable_scope(name_or_scope=name):
            for stage_name, stage_params in self._detail_branch_channels.items():
                with tf.variable_scope(stage_name):
                    for block_index, param in enumerate(stage_params):
                        block_op = self._block_maps[param[0]]
                        k_size = param[1]
                        output_channels = param[2]
                        stride = param[3]
                        repeat_times = param[4]
                        for repeat_index in range(repeat_times):
                            with tf.variable_scope(name_or_scope='conv_block_{:d}_repeat_{:d}'.format(
                                    block_index + 1, repeat_index + 1)):
                                if stage_name == 'stage_3' and block_index == 1 and repeat_index == 1:
                                    result = block_op(
                                        input_tensor=result,
                                        k_size=k_size,
                                        output_channels=output_channels,
                                        stride=stride,
                                        name='3x3_conv',
                                        padding='SAME',
                                        use_bias=False,
                                        need_activate=False
                                    )
                                else:
                                    result = block_op(
                                        input_tensor=result,
                                        k_size=k_size,
                                        output_channels=output_channels,
                                        stride=stride,
                                        name='3x3_conv',
                                        padding='SAME',
                                        use_bias=False,
                                        need_activate=True
                                    )
        return result

    def build_semantic_branch(self, input_tensor, name, prepare_data_for_booster=False):
        """

        :param input_tensor:
        :param name:
        :param prepare_data_for_booster:
        :return:
        """
        seg_head_inputs = collections.OrderedDict()
        result = input_tensor
        source_input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        with tf.variable_scope(name_or_scope=name):
            for stage_name, stage_params in self._semantic_branch_channels.items():
                seg_head_input = input_tensor
                with tf.variable_scope(stage_name):
                    for block_index, param in enumerate(stage_params):
                        block_op_name = param[0]
                        block_op = self._block_maps[block_op_name]
                        output_channels = param[2]
                        expand_ratio = param[3]
                        stride = param[4]
                        repeat_times = param[5]
                        for repeat_index in range(repeat_times):
                            with tf.variable_scope(name_or_scope='{:s}_block_{:d}_repeat_{:d}'.format(
                                    block_op_name, block_index + 1, repeat_index + 1)):
                                if block_op_name == 'ge':
                                    result = block_op(
                                        input_tensor=result,
                                        name='gather_expansion_block',
                                        stride=stride,
                                        e=expand_ratio,
                                        output_channels=output_channels
                                    )
                                    seg_head_input = result
                                elif block_op_name == 'ce':
                                    result = block_op(
                                        input_tensor=result,
                                        name='context_embedding_block'
                                    )
                                elif block_op_name == 'se':
                                    result = block_op(
                                        input_tensor=result,
                                        output_channels=output_channels,
                                        name='stem_block'
                                    )
                                    seg_head_input = result
                                else:
                                    raise NotImplementedError('Not support block type: {:s}'.format(block_op_name))
                    if prepare_data_for_booster:
                        result_tensor_size = result.get_shape().as_list()[1:3]
                        result_tensor_dims = result.get_shape().as_list()[-1]
                        upsample_ratio = int(source_input_tensor_size[0] / result_tensor_size[0])
                        feature_dims = result_tensor_dims * self._seg_head_ratio
                        seg_head_inputs[stage_name] = self._seg_head_block(
                            input_tensor=seg_head_input,
                            name='block_{:d}_seg_head_block'.format(block_index + 1),
                            upsample_ratio=upsample_ratio,
                            feature_dims=feature_dims,
                            classes_nums=self._class_nums
                        )
        return result, seg_head_inputs

    def build_aggregation_branch(self, detail_output, semantic_output, name):
        """

        :param detail_output:
        :param semantic_output:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self._guided_aggregation_block(
                detail_input_tensor=detail_output,
                semantic_input_tensor=semantic_output,
                name='guided_aggregation_block'
            )
        return result

    def build_instance_segmentation_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * 8) for tmp in input_tensor_size]

        with tf.variable_scope(name_or_scope=name):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=64,
                stride=1,
                name='conv_3x3',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=128,
                stride=1,
                name='conv_1x1',
                use_bias=False,
                need_activate=False
            )
            output_tensor = tf.image.resize_bilinear(
                output_tensor,
                output_tensor_size,
                name='instance_logits'
            )
        return output_tensor

    def build_binary_segmentation_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * 8) for tmp in input_tensor_size]

        with tf.variable_scope(name_or_scope=name):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=64,
                stride=1,
                name='conv_3x3',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=128,
                stride=1,
                name='conv_1x1',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=self._class_nums,
                stride=1,
                name='final_conv',
                use_bias=False,
                need_activate=False
            )
            output_tensor = tf.image.resize_bilinear(
                output_tensor,
                output_tensor_size,
                name='binary_logits'
            )
        return output_tensor

    def build_model(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # build detail branch
            detail_branch_output = self.build_detail_branch(
                input_tensor=input_tensor,
                name='detail_branch'
            )
            # build semantic branch
            semantic_branch_output, _ = self.build_semantic_branch(
                input_tensor=input_tensor,
                name='semantic_branch',
                prepare_data_for_booster=False
            )
            # build aggregation branch
            aggregation_branch_output = self.build_aggregation_branch(
                detail_output=detail_branch_output,
                semantic_output=semantic_branch_output,
                name='aggregation_branch'
            )
            # build binary and instance segmentation branch
            binary_seg_branch_output = self.build_binary_segmentation_branch(
                input_tensor=aggregation_branch_output,
                name='binary_segmentation_branch'
            )
            instance_seg_branch_output = self.build_instance_segmentation_branch(
                input_tensor=aggregation_branch_output,
                name='instance_segmentation_branch'
            )
            # gather frontend output result
            self._net_intermediate_results['binary_segment_logits'] = {
                'data': binary_seg_branch_output,
                'shape': binary_seg_branch_output.get_shape().as_list()
            }
            self._net_intermediate_results['instance_segment_logits'] = {
                'data': instance_seg_branch_output,
                'shape': instance_seg_branch_output.get_shape().as_list()
            }
        return self._net_intermediate_results


if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    model = BiseNetV2(phase='train')
    ret = model.build_model(test_in_tensor, name='bisenetv2')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))

