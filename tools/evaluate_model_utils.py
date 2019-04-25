#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-21 上午11:17
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : evaluate_model_performance.py
# @IDE: PyCharm
"""
计算模型的准确率,FP和FN
"""
import tensorflow as tf


def calculate_model_precision(input_tensor, label_tensor):
    """
    按照公式 acc = correct_nums / ground_truth_nums 计算准确率
    :param input_tensor: binary segmentation logits
    :param label_tensor: binary segmentation label
    :return:
    """

    logits = tf.nn.softmax(logits=input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)

    idx = tf.where(tf.equal(label_tensor, 1))
    pix_cls_ret = tf.gather_nd(final_output, idx)
    accuracy = tf.count_nonzero(pix_cls_ret)
    accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

    return accuracy


def calculate_model_fp(input_tensor, label_tensor):
    """
    计算模型FP
    :param input_tensor:
    :param label_tensor:
    :return:
    """
    logits = tf.nn.softmax(logits=input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)

    idx = tf.where(tf.equal(final_output, 1))
    pix_cls_ret = tf.gather_nd(label_tensor, idx)
    false_pred = tf.cast(tf.shape(pix_cls_ret)[0], tf.int64) - tf.count_nonzero(pix_cls_ret)

    return tf.divide(false_pred, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))


def calculate_model_fn(input_tensor, label_tensor):
    """
    计算模型FN
    :param input_tensor:
    :param label_tensor:
    :return:
    """
    logits = tf.nn.softmax(logits=input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)

    idx = tf.where(tf.equal(label_tensor, 1))
    pix_cls_ret = tf.gather_nd(final_output, idx)
    mis_pred = tf.cast(tf.shape(pix_cls_ret)[0], tf.int64) - tf.count_nonzero(pix_cls_ret)

    return tf.divide(mis_pred, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))


def get_image_summary(img):
    """
    Make an image summary for 4d tensor image with index idx
    :param img:
    """

    if len(img.get_shape().as_list()) == 3:
        img = tf.expand_dims(img, -1)

    image = img - tf.reduce_min(img)
    image /= tf.reduce_max(img) - tf.reduce_min(img)
    image *= 255

    return image


if __name__ == '__main__':
    """
    test
    """

    import numpy as np

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[2, 8, 8])
    label_tensor = tf.placeholder(dtype=tf.float32, shape=[2, 8, 8, 1])

    fp = calculate_model_fp(input_tensor, label_tensor)
    fn = calculate_model_fn(input_tensor, label_tensor)

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=sess_config) as sess:

        input_image = np.eye(8, dtype=np.float32)
        label_image = np.eye(8, dtype=np.float32)

        input_image[1, 2] = 1
        label_image[2, 1] = 1
        label_image = np.expand_dims(label_image, -1)

        fp_value = sess.run(fp, feed_dict={input_tensor: [input_image, input_image],
                                           label_tensor: [label_image, label_image]})

        print(fp_value)
