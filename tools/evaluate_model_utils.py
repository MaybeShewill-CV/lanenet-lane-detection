#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-21 上午11:17
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : evaluate_model_utils.py
# @IDE: PyCharm
"""
Calculate model's fp fn and precision
"""
import tensorflow as tf


def calculate_model_precision(input_tensor, label_tensor):
    """
    calculate accuracy acc = correct_nums / ground_truth_nums
    :param input_tensor: binary segmentation logits
    :param label_tensor: binary segmentation label
    :return:
    """

    logits = tf.nn.softmax(logits=input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)

    idx = tf.where(tf.equal(final_output, 1))
    pix_cls_ret = tf.gather_nd(label_tensor, idx)
    accuracy = tf.count_nonzero(pix_cls_ret)
    accuracy = tf.divide(
        accuracy,
        tf.cast(tf.shape(tf.gather_nd(label_tensor, tf.where(tf.equal(label_tensor, 1))))[0], tf.int64))

    return accuracy


def calculate_model_fp(input_tensor, label_tensor):
    """
    calculate fp figure
    :param input_tensor:
    :param label_tensor:
    :return:
    """
    logits = tf.nn.softmax(logits=input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)

    idx = tf.where(tf.equal(final_output, 1))
    pix_cls_ret = tf.gather_nd(final_output, idx)
    false_pred = tf.cast(tf.shape(pix_cls_ret)[0], tf.int64) - tf.count_nonzero(
        tf.gather_nd(label_tensor, idx)
    )

    return tf.divide(false_pred, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))


def calculate_model_fn(input_tensor, label_tensor):
    """
    calculate fn figure
    :param input_tensor:
    :param label_tensor:
    :return:
    """
    logits = tf.nn.softmax(logits=input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)

    idx = tf.where(tf.equal(label_tensor, 1))
    pix_cls_ret = tf.gather_nd(final_output, idx)
    label_cls_ret = tf.gather_nd(label_tensor, tf.where(tf.equal(label_tensor, 1)))
    mis_pred = tf.cast(tf.shape(label_cls_ret)[0], tf.int64) - tf.count_nonzero(pix_cls_ret)

    return tf.divide(mis_pred, tf.cast(tf.shape(label_cls_ret)[0], tf.int64))


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
