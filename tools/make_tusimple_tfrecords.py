#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 下午8:29
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : make_cityscapes_tfrecords.py
# @IDE: PyCharm
"""
Generate cityscapes tfrecords tools
"""
from data_provider import cityscapes_tf_io
from local_utils.log_util import init_logger

LOG = init_logger.get_logger(log_file_name_prefix='generate_cityscapes_tfrecords')


def generate_tfrecords():
    """

    :return:
    """
    io = cityscapes_tf_io.CityScapesTfIO()
    io.writer.write_tfrecords()

    return


if __name__ == '__main__':
    """
    test
    """
    generate_tfrecords()
