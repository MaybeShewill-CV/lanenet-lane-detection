#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 上午11:38
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_hnet_model.py
# @IDE: PyCharm Community Edition
"""
LaneNet中的HNet模型
"""
import tensorflow as tf

from encoder_decoder_model import cnn_basenet
from lanenet_model import lanenet_hnet_loss


class LaneNetHNet(cnn_basenet.CNNBaseModel):
    """
    实现lanenet中的hnet模型
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(LaneNetHNet, self).__init__()
        self._train_phase = tf.constant('train', tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

        return

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, inputdata, out_channel, name):
        """

        :param inputdata:
        :param out_channel:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=inputdata, out_channel=out_channel, kernel_size=3, use_bias=False, name='conv')
            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')
            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _build_model(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv_stage_1 = self._conv_stage(inputdata=input_tensor, out_channel=16, name='conv_stage_1')
            conv_stage_2 = self._conv_stage(inputdata=conv_stage_1, out_channel=16, name='conv_stage_2')
            maxpool_1 = self.maxpooling(inputdata=conv_stage_2, kernel_size=2, stride=2, name='maxpool_1')
            conv_stage_3 = self._conv_stage(inputdata=maxpool_1, out_channel=32, name='conv_stage_3')
            conv_stage_4 = self._conv_stage(inputdata=conv_stage_3, out_channel=32, name='conv_stage_4')
            maxpool_2 = self.maxpooling(inputdata=conv_stage_4, kernel_size=2, stride=2, name='maxpool_2')
            conv_stage_5 = self._conv_stage(inputdata=maxpool_2, out_channel=64, name='conv_stage_5')
            conv_stage_6 = self._conv_stage(inputdata=conv_stage_5, out_channel=64, name='conv_stage_6')
            maxpool_3 = self.maxpooling(inputdata=conv_stage_6, kernel_size=2, stride=2, name='maxpool_3')
            fc = self.fullyconnect(inputdata=maxpool_3, out_dim=1024, use_bias=False, name='fc')
            fc_relu = self.relu(inputdata=fc, name='fc_relu')
            output = self.fullyconnect(inputdata=fc_relu, out_dim=6, use_bias=False, name='fc_output')
            output = self.squeeze(inputdata=output, axis=0)

        return output

    def compute_loss(self, input_tensor, gt_label_pts, name):
        """
        计算hnet损失函数
        :param input_tensor: 原始图像[n, h, w, c]
        :param gt_label_pts: 原始图像对应的标签点集[x, y, 1]
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            transformation_coefficient = self._build_model(input_tensor, name='transfomation_coefficient')
            loss = lanenet_hnet_loss.hnet_loss(gt_pts=gt_label_pts,
                                               transformation_coeffcient=transformation_coefficient,
                                               name='hnet_loss')

            return loss, transformation_coefficient

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            return self._build_model(input_tensor, name='transfomation_coefficient')


if __name__ == '__main__':
    tensor_in = tf.placeholder(dtype=tf.float32, shape=[2, 64, 128, 3])
    gt_label_pts = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    net = LaneNetHNet(phase=tf.constant('train', tf.string))
    coffe = net.inference(tensor_in, name='hnet')
    # c_loss = net.compute_loss(tensor_in, gt_label_pts=gt_label_pts, name='hnet')

    saver = tf.train.Saver()

    from data_provider import lanenet_hnet_data_processor
    import numpy as np
    import cv2
    try:
        from cv2 import cv2
    except ImportError:
        pass
    train_dataset = lanenet_hnet_data_processor.DataSet(
        ['/media/baidu/Data/Semantic_Segmentation/TUSimple_Lane_Detection/training/label_data_0531.json'])

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())

        saver.restore(sess=sess,
                      save_path='../model/tusimple_lanenet_hnet/tusimple_lanenet_hnet_2018-08-08-19-32-01.ckpt-200000')

        image, label_pts = train_dataset.next_batch(1)
        label_pts = label_pts[0]
        image = [cv2.resize(tmp, (128, 64), interpolation=cv2.INTER_LINEAR) for tmp in image]
        c_val = sess.run(coffe, feed_dict={tensor_in: image, gt_label_pts: label_pts})
        R = np.zeros([3, 3], np.float32)
        R[0, 0] = c_val[0]
        R[0, 1] = c_val[1]
        R[0, 2] = c_val[2]
        R[1, 1] = c_val[3]
        R[1, 2] = c_val[4]
        R[2, 1] = c_val[5]
        R[2, 2] = 1
        print(np.mat(R).I)
        print(R)
        print(c_val)

        warp_image = cv2.warpPerspective(image[0], R, dsize=(image[0].shape[1], image[0].shape[0]))
        cv2.imwrite("src.jpg", image[0])
        cv2.imwrite("ret.jpg", warp_image)
