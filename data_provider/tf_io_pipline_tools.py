#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-23 下午3:53
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : tf_io_pipline_tools.py
# @IDE: PyCharm
"""
tensorflow io pip line tools
"""
import os
import os.path as ops

import cv2
import glog as log
import numpy as np
import tensorflow as tf

from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg

RESIZE_IMAGE_HEIGHT = CFG.AUG.TRAIN_CROP_SIZE[1] + CFG.AUG.CROP_PAD_SIZE
RESIZE_IMAGE_WIDTH = CFG.AUG.TRAIN_CROP_SIZE[0] + CFG.AUG.CROP_PAD_SIZE
CROP_IMAGE_HEIGHT = CFG.AUG.TRAIN_CROP_SIZE[1]
CROP_IMAGE_WIDTH = CFG.AUG.TRAIN_CROP_SIZE[0]


def int64_feature(value):
    """

    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """

    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_example_tfrecords(gt_images_paths, gt_binary_images_paths, gt_instance_images_paths, tfrecords_path):
    """
    write tfrecords
    :param gt_images_paths:
    :param gt_binary_images_paths:
    :param gt_instance_images_paths:
    :param tfrecords_path:
    :return:
    """
    _tfrecords_dir = ops.split(tfrecords_path)[0]
    os.makedirs(_tfrecords_dir, exist_ok=True)

    log.info('Writing {:s}....'.format(tfrecords_path))

    with tf.python_io.TFRecordWriter(tfrecords_path) as _writer:
        for _index, _gt_image_path in enumerate(gt_images_paths):

            # prepare gt image
            _gt_image = cv2.imread(_gt_image_path, cv2.IMREAD_UNCHANGED)
            if _gt_image.shape != (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 3):
                _gt_image = cv2.resize(
                    _gt_image,
                    dsize=(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
                    interpolation=cv2.INTER_LINEAR
                )
            _gt_image_raw = _gt_image.tostring()

            # prepare gt binary image
            _gt_binary_image = cv2.imread(gt_binary_images_paths[_index], cv2.IMREAD_UNCHANGED)
            if _gt_binary_image.shape != (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT):
                _gt_binary_image = cv2.resize(
                    _gt_binary_image,
                    dsize=(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
                    interpolation=cv2.INTER_NEAREST
                )
                _gt_binary_image = np.array(_gt_binary_image / 255.0, dtype=np.uint8)
            _gt_binary_image_raw = _gt_binary_image.tostring()

            # prepare gt instance image
            _gt_instance_image = cv2.imread(gt_instance_images_paths[_index], cv2.IMREAD_UNCHANGED)
            if _gt_instance_image.shape != (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT):
                _gt_instance_image = cv2.resize(
                    _gt_instance_image,
                    dsize=(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
                    interpolation=cv2.INTER_NEAREST
                )
            _gt_instance_image_raw = _gt_instance_image.tostring()

            _example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'gt_image_raw': bytes_feature(_gt_image_raw),
                        'gt_binary_image_raw': bytes_feature(_gt_binary_image_raw),
                        'gt_instance_image_raw': bytes_feature(_gt_instance_image_raw)
                    }))
            _writer.write(_example.SerializeToString())

    log.info('Writing {:s} complete'.format(tfrecords_path))

    return


def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`
    :param serialized_example:
    :return:
    """
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'gt_image_raw': tf.FixedLenFeature([], tf.string),
            'gt_binary_image_raw': tf.FixedLenFeature([], tf.string),
            'gt_instance_image_raw': tf.FixedLenFeature([], tf.string)
        })

    # decode gt image
    gt_image_shape = tf.stack([RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 3])
    gt_image = tf.decode_raw(features['gt_image_raw'], tf.uint8)
    gt_image = tf.reshape(gt_image, gt_image_shape)

    # decode gt binary image
    gt_binary_image_shape = tf.stack([RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 1])
    gt_binary_image = tf.decode_raw(features['gt_binary_image_raw'], tf.uint8)
    gt_binary_image = tf.reshape(gt_binary_image, gt_binary_image_shape)

    # decode gt instance image
    gt_instance_image_shape = tf.stack([RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 1])
    gt_instance_image = tf.decode_raw(features['gt_instance_image_raw'], tf.uint8)
    gt_instance_image = tf.reshape(gt_instance_image, gt_instance_image_shape)

    return gt_image, gt_binary_image, gt_instance_image


def central_crop(image, crop_height, crop_width):
    """
    Performs central crops of the given image
    :param image:
    :param crop_height:
    :param crop_width:
    :return:
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def augment_for_train(gt_image, gt_binary_image, gt_instance_image):
    """

    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    # convert image from uint8 to float32
    gt_image = tf.cast(gt_image, tf.float32)
    gt_binary_image = tf.cast(gt_binary_image, tf.float32)
    gt_instance_image = tf.cast(gt_instance_image, tf.float32)

    # apply random color augmentation
    gt_image, gt_binary_image, gt_instance_image = random_color_augmentation(
        gt_image, gt_binary_image, gt_instance_image
    )

    # apply random flip augmentation
    gt_image, gt_binary_image, gt_instance_image = random_horizon_flip_batch_images(
        gt_image, gt_binary_image, gt_instance_image
    )

    # apply random crop image
    return random_crop_batch_images(
        gt_image=gt_image,
        gt_binary_image=gt_binary_image,
        gt_instance_image=gt_instance_image,
        cropped_size=[CROP_IMAGE_WIDTH, CROP_IMAGE_HEIGHT]
    )


def augment_for_test(gt_image, gt_binary_image, gt_instance_image):
    """

    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    # apply central crop
    gt_image = central_crop(
        image=gt_image, crop_height=CROP_IMAGE_HEIGHT, crop_width=CROP_IMAGE_WIDTH
    )
    gt_binary_image = central_crop(
        image=gt_binary_image, crop_height=CROP_IMAGE_HEIGHT, crop_width=CROP_IMAGE_WIDTH
    )
    gt_instance_image = central_crop(
        image=gt_instance_image, crop_height=CROP_IMAGE_HEIGHT, crop_width=CROP_IMAGE_WIDTH
    )

    return gt_image, gt_binary_image, gt_instance_image


def normalize(gt_image, gt_binary_image, gt_instance_image):
    """
    Normalize the image data by substracting the imagenet mean value
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """

    if gt_image.get_shape().as_list()[-1] != 3 \
            or gt_binary_image.get_shape().as_list()[-1] != 1 \
            or gt_instance_image.get_shape().as_list()[-1] != 1:
        log.error(gt_image.get_shape())
        log.error(gt_binary_image.get_shape())
        log.error(gt_instance_image.get_shape())
        raise ValueError('Input must be of size [height, width, C>0]')

    gt_image = tf.cast(gt_image, dtype=tf.float32)
    gt_image = tf.subtract(tf.divide(gt_image, tf.constant(127.5, dtype=tf.float32)),
                           tf.constant(1.0, dtype=tf.float32))

    return gt_image, gt_binary_image, gt_instance_image


def random_crop_batch_images(gt_image, gt_binary_image, gt_instance_image, cropped_size):
    """
    Random crop image batch data for training
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :param cropped_size:
    :return:
    """
    concat_images = tf.concat([gt_image, gt_binary_image, gt_instance_image], axis=-1)

    concat_cropped_images = tf.image.random_crop(
        concat_images,
        [cropped_size[1], cropped_size[0], tf.shape(concat_images)[-1]],
        seed=tf.random.set_random_seed(1234)
    )

    cropped_gt_image = tf.slice(
        concat_cropped_images,
        begin=[0, 0, 0],
        size=[cropped_size[1], cropped_size[0], 3]
    )
    cropped_gt_binary_image = tf.slice(
        concat_cropped_images,
        begin=[0, 0, 3],
        size=[cropped_size[1], cropped_size[0], 1]
    )
    cropped_gt_instance_image = tf.slice(
        concat_cropped_images,
        begin=[0, 0, 4],
        size=[cropped_size[1], cropped_size[0], 1]
    )

    return cropped_gt_image, cropped_gt_binary_image, cropped_gt_instance_image


def random_horizon_flip_batch_images(gt_image, gt_binary_image, gt_instance_image):
    """
    Random horizon flip image batch data for training
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    concat_images = tf.concat([gt_image, gt_binary_image, gt_instance_image], axis=-1)

    [image_height, image_width, _] = gt_image.get_shape().as_list()

    concat_flipped_images = tf.image.random_flip_left_right(
        image=concat_images,
        seed=tf.random.set_random_seed(1)
    )

    flipped_gt_image = tf.slice(
        concat_flipped_images,
        begin=[0, 0, 0],
        size=[image_height, image_width, 3]
    )
    flipped_gt_binary_image = tf.slice(
        concat_flipped_images,
        begin=[0, 0, 3],
        size=[image_height, image_width, 1]
    )
    flipped_gt_instance_image = tf.slice(
        concat_flipped_images,
        begin=[0, 0, 4],
        size=[image_height, image_width, 1]
    )

    return flipped_gt_image, flipped_gt_binary_image, flipped_gt_instance_image


def random_color_augmentation(gt_image, gt_binary_image, gt_instance_image):
    """
    andom color augmentation
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    # first apply random saturation augmentation
    gt_image = tf.image.random_saturation(gt_image, 0.8, 1.2)
    # sencond apply random brightness augmentation
    gt_image = tf.image.random_brightness(gt_image, 0.05)
    # third apply random contrast augmentation
    gt_image = tf.image.random_contrast(gt_image, 0.7, 1.3)

    gt_image = tf.clip_by_value(gt_image, 0.0, 255.0)

    return gt_image, gt_binary_image, gt_instance_image
